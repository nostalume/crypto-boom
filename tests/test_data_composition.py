"""Composable dimensions and project-wide locations, without a universal reader."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from crypto_boom.bars import (
    SOURCE_COLUMNS,
    BarPeriod,
    aggregate_minute_bars,
    read_bar_window,
)
from crypto_boom.config import project_settings
from crypto_boom.market import Environment, InstrumentId, TimeWindow, VenueId
from crypto_boom.market_scan import scan_settings
from crypto_boom.storage.source import select_minute_partitions
from test_forward_prediction import BASE, bars


def test_project_root_is_stable_from_nested_directory(tmp_path, monkeypatch):
    config = tmp_path / "crypto-boom.toml"
    config.write_text(
        '[data]\nroot="shared"\nlegacy_corpora=["legacy/corpus"]\n[scan]\nworkers=2\n'
    )
    child = tmp_path / "deep" / "work"
    child.mkdir(parents=True)
    monkeypatch.chdir(child)
    settings = project_settings()
    assert settings.data_root == tmp_path / "shared"
    assert settings.corpus_roots == (tmp_path / "shared/legacy/corpus",)
    assert scan_settings().data_dir == settings.data_root
    assert project_settings(config) == settings
    config.write_text('[data]\nroot="shared"\n[scan]\ndata_dir="other"\n')
    with pytest.raises(ValueError, match="conflicting"):
        project_settings(config)


def test_project_default_is_project_relative_not_cwd(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="crypto-boom"\n')
    child = tmp_path / "sub"
    child.mkdir()
    monkeypatch.chdir(child)
    assert project_settings().data_root == tmp_path / "data"


@pytest.mark.parametrize("period", [1, 5, 60])
def test_window_and_scale_are_separate_operations(tmp_path, period):
    source = bars(120).with_columns(pl.col("close_price").alias("open_price"))
    path = tmp_path / "minute.parquet"
    source.write_parquet(path)
    start = datetime.fromtimestamp(BASE / 1e6, UTC)
    window = TimeWindow.from_datetimes(start, start + timedelta(hours=1))
    frame, _ = read_bar_window([path], window, columns=(*SOURCE_COLUMNS, "open_price"))
    assert len(frame) == 60
    result = aggregate_minute_bars(frame, BarPeriod(period))
    assert len(result) == 60 // period
    assert result["open_price"][0] == source["open_price"][0]
    assert result["close_price"][-1] == source["close_price"][59]
    assert result["quote_turnover"].sum() == source["quote_turnover"][:60].sum()
    assert result["high_price"].max() == source["high_price"][:60].max()
    with pytest.raises(ValueError, match="complete aligned"):
        aggregate_minute_bars(
            frame.filter(pl.col("open_time") != frame["open_time"][1]),
            BarPeriod(period),
        )
    with pytest.raises(ValueError, match="genuine"):
        aggregate_minute_bars(frame.drop("open_price"), BarPeriod(period))
    point, _ = read_bar_window([path], TimeWindow(BASE, BASE + 60_000_000))
    assert len(point) == 1


def test_time_and_period_rejections():
    with pytest.raises(ValueError, match="timezones"):
        TimeWindow.from_datetimes(datetime(2026, 1, 1), datetime(2026, 1, 2))
    with pytest.raises(ValueError):
        TimeWindow(10, 10)
    with pytest.raises(ValueError):
        BarPeriod(0)


def test_archive_selection_reports_missing_and_rejects_unsupported_source(tmp_path):
    window = TimeWindow.from_datetimes(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    )
    symbol = InstrumentId(VenueId("binance", "spot"), Environment.PRODUCTION, "AAAUSDT")
    selected = select_minute_partitions((tmp_path,), symbol, window)
    assert not selected.partitions and selected.missing_months == ("2026-01", "2026-02")
    with pytest.raises(ValueError, match="Binance Spot"):
        select_minute_partitions(
            (tmp_path,),
            InstrumentId(VenueId("other", "spot"), Environment.PRODUCTION, "AAAUSDT"),
            window,
        )


def test_research_runs_reuse_shared_derived_data(tmp_path, monkeypatch):
    from crypto_boom.research import path_dataset
    from crypto_boom.research.path_targets import PathTargetSpec

    source = tmp_path / "source.parquet"
    bars(2200).write_parquet(source)
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    monkeypatch.setattr(
        path_dataset,
        "load_sample_pool",
        lambda p: {
            "selection_id": "fixture",
            "availability_id": "fixture",
            "symbols_with_data": ["AAAUSDT"],
            "partitions": [
                {
                    "symbol": "AAAUSDT",
                    "month": "2026-01",
                    "state": "available",
                    "path": str(source),
                }
            ],
        },
    )
    shared = tmp_path / "shared"
    runs = [
        path_dataset.build_path_dataset(
            pool,
            output_root=shared / "runs" / name,
            data_root=shared,
            spec=PathTargetSpec((10,), ()),
        )
        for name in ("a", "b")
    ]
    assert runs[0]["dataset_id"] == runs[1]["dataset_id"]
    assert (
        runs[1]["batches"][0]["feature_reused"]
        and runs[1]["batches"][0]["target_reused"]
    )
    assert Path(runs[1]["batches"][0]["features"]).is_relative_to(
        shared / "derived/features"
    )


@pytest.mark.parametrize("conflict", [False, True])
def test_archive_selection_deduplicates_identity_not_conflicting_revisions(
    tmp_path, monkeypatch, conflict
):
    from types import SimpleNamespace

    from crypto_boom.storage import source

    roots = (tmp_path / "a", tmp_path / "b")
    for root in roots:
        directory = root / "binance/spot/research-source-klines/AAAUSDT/1m/2026-01/id"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{}")
        (directory / "klines.parquet").write_bytes(b"fixture")

    def load(path):
        return SimpleNamespace(
            path=path,
            manifest=SimpleNamespace(
                symbol="AAAUSDT",
                month="2026-01",
                manifest_id="other"
                if conflict and path.is_relative_to(roots[1])
                else "same",
            ),
        )

    monkeypatch.setattr(source, "load_published_research_partition", load)
    window = TimeWindow.from_datetimes(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    )
    instrument = InstrumentId(
        VenueId("binance", "spot"), Environment.PRODUCTION, "AAAUSDT"
    )
    if conflict:
        with pytest.raises(ValueError, match="ambiguous"):
            select_minute_partitions(roots, instrument, window)
    else:
        selected = select_minute_partitions(roots, instrument, window)
        assert len(selected.partitions) == 1 and selected.missing_months == ("2026-02",)


def test_acquire_cli_passes_project_storage_independently_of_run(tmp_path, monkeypatch):
    from crypto_boom import sample_pool
    from crypto_boom.research.study_cli import main

    config = tmp_path / "crypto-boom.toml"
    config.write_text('[data]\nroot="shared"\nlegacy_corpora=["old/corpus"]\n')
    seen = {}

    async def acquire(request, **kwargs):
        seen.update(kwargs)
        return {"state": "fixture"}

    monkeypatch.setattr(sample_pool, "acquire_sample_pool", acquire)
    assert (
        main(
            [
                "acquire",
                "--start-month",
                "2026-01",
                "--end-month",
                "2026-01",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert seen["data_root"] == tmp_path / "shared"
    assert seen["output"].is_relative_to(tmp_path / "shared/runs")
    assert seen["reuse_corpora"] == (tmp_path / "shared/old/corpus",)
