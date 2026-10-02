from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from crypto_boom.bars import MINUTE_US
from crypto_boom.feature_batch import (
    BATCH_FEATURES,
    build_feature_cache,
    read_feature_cache,
)
from crypto_boom.history.availability import HistoricalArchiveSymbolCatalog
from crypto_boom.research.path_model import (
    forecast_path,
    load_path_model,
    render_path_forecast,
    save_path_model,
)
from crypto_boom.research.path_screen import screen_path_features
from crypto_boom.research.path_targets import (
    PathTargetSpec,
    build_target_cache,
    path_targets,
    read_target_cache,
)
from crypto_boom.sample_pool import select_archive_sample
from test_forward_prediction import BASE, bars


def test_targets_arithmetic_order_and_unknown_future():
    source = bars(5).with_columns(
        pl.Series("close_price", [100.0, 110.0, 90.0, 120.0, 105.0]),
        pl.lit(125.0).alias("high_price"),
        pl.lit(85.0).alias("low_price"),
    )
    origins = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"] * 2,
            "decision_us": [BASE + MINUTE_US, BASE + 5 * MINUTE_US],
        }
    )
    rows = path_targets(source, origins, PathTargetSpec((4,), (0.15,)))
    first = rows.row(0, named=True)
    assert first["up_4"] == pytest.approx(0.2)
    assert first["down_4"] == pytest.approx(0.1)
    assert first["terminal_4"] == pytest.approx(0.05)
    assert first["retention_up_4"] == pytest.approx(0.25)
    assert first["efficiency_4"] == pytest.approx(0.05 / 0.75)
    assert first["drawdown_4"] == pytest.approx(1 - 90 / 110)
    assert first["above_fraction_4"] == 0.75
    assert first["up_first_minute_4_0p15"] == 3
    assert first["up_adverse_before_4_0p15"] == pytest.approx(0.1)
    assert first["down_hit_4_0p15"] == 0
    assert first["down_first_minute_4_0p15"] is None
    assert rows.row(1, named=True)["up_hit_4_0p15"] is None
    gapped = source.filter(pl.col("open_time").dt.epoch("us") != BASE + 2 * MINUTE_US)
    assert (
        path_targets(gapped, origins, PathTargetSpec((4,), ()))["up_4"].null_count()
        == 2
    )


def test_flat_retention_undefined_and_invalid_origin_rejected():
    source = bars(8).with_columns(
        pl.lit(10.0).alias("close_price"),
        pl.lit(11.0).alias("high_price"),
        pl.lit(9.0).alias("low_price"),
    )
    origin = pl.DataFrame({"symbol": ["AAAUSDT"], "decision_us": [BASE + MINUTE_US]})
    row = path_targets(source, origin, PathTargetSpec((2,), ())).row(0, named=True)
    assert row["retention_up_2"] is None
    assert row["retention_down_2"] is None
    assert row["efficiency_2"] == 0
    with pytest.raises(ValueError, match="origin"):
        path_targets(
            source,
            origin.with_columns(pl.col("decision_us") + 1),
            PathTargetSpec((2,), ()),
        )


def test_archive_selection_reproducible_and_not_trading_status_filtered():
    catalog = HistoricalArchiveSymbolCatalog(
        "https://example.test", "monthly/", 1, 1, ("AAAUSDT", "DEADUSDT", "BBBUSD")
    )
    selected = select_archive_sample(catalog, count=2, seed="frozen")
    assert selected["symbols"] == ["AAAUSDT", "DEADUSDT"]
    assert selected == select_archive_sample(catalog, count=2, seed="frozen")
    with pytest.raises(ValueError):
        select_archive_sample(
            catalog, count=2, seed="frozen", excluded_symbols=("DEADUSDT",)
        )


def test_feature_target_cache_reuse_causality_invalidation_and_tamper(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.parquet"
    bars(2200).write_parquet(source)
    cache, reused = build_feature_cache([source], output_root=tmp_path / "features")
    assert not reused
    features, receipt = read_feature_cache(cache)
    assert len(features) > 0 and len(BATCH_FEATURES) == 31
    from crypto_boom import features as feature_module

    with monkeypatch.context() as patch:
        patch.setattr(
            feature_module,
            "iter_feature_segments",
            lambda *a, **k: pytest.fail("cache hit recomputed features"),
        )
        assert build_feature_cache([source], output_root=tmp_path / "features") == (
            cache,
            True,
        )
    target, _ = build_target_cache(
        cache, output_root=tmp_path / "labels", spec=PathTargetSpec((10,), (0.1,))
    )
    assert build_target_cache(
        cache, output_root=tmp_path / "labels", spec=PathTargetSpec((10,), (0.1,))
    ) == (target, True)
    labels, _ = read_target_cache(target)
    assert labels["up_10"].null_count() > 0
    other, _ = build_target_cache(
        cache, output_root=tmp_path / "labels", spec=PathTargetSpec((20,), (0.1,))
    )
    assert other != target
    bars(2100).write_parquet(source)
    shorter, _ = build_feature_cache([source], output_root=tmp_path / "features")
    prefix, _ = read_feature_cache(shorter)
    assert prefix.equals(
        features.filter(pl.col("decision_us") <= BASE + 2100 * MINUTE_US)
    )
    with pytest.raises(ValueError, match="source changed"):
        build_target_cache(
            cache, output_root=tmp_path / "labels", spec=PathTargetSpec((30,), ())
        )
    with (cache / "features.parquet").open("ab") as out:
        out.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        read_feature_cache(cache)
    assert receipt["quality"]["missing_minutes_inside_observed_span"] == 0


@pytest.fixture(scope="module")
def screening():
    from crypto_boom.features import iter_feature_segments

    source = bars(4500)
    features = pl.concat(list(iter_feature_segments(source, rich=True))).select(
        "symbol", "decision_us", *BATCH_FEATURES
    )
    labels = path_targets(source, features, PathTargetSpec((10,), ()))
    rows = features.join(labels, on=["symbol", "decision_us"])
    kwargs = {
        "train_end_us": BASE + 3000 * MINUTE_US,
        "selection_end_us": BASE + 3700 * MINUTE_US,
        "horizon": 10,
        "max_iter": 2,
    }
    models, metadata = screen_path_features(rows, **kwargs)
    return models, metadata, rows, kwargs, source


def test_screen_purge_baselines_and_test_does_not_select(screening):
    models, metadata, rows, kwargs, _ = screening
    assert metadata["train_last_us"] + 10 * MINUTE_US < kwargs["train_end_us"]
    assert metadata["selection_last_us"] + 10 * MINUTE_US < kwargs["selection_end_us"]
    changed = rows.with_columns(
        [
            pl.when(pl.col("decision_us") >= kwargs["selection_end_us"])
            .then(pl.col(label) * 10)
            .otherwise(pl.col(label))
            .alias(label)
            for label in models
        ]
    )
    _, second = screen_path_features(changed, **kwargs)
    for first, after in zip(metadata["responses"], second["responses"], strict=True):
        assert first["attempts"] == after["attempts"]
        assert first["features"] == after["features"]
        assert len(first["attempts"]) == 4
        assert set(first["features"]).issubset(BATCH_FEATURES)
        assert set(first["evaluation"]["test"]["baselines"]) == {
            "fixed",
            "symbol",
            "volatility",
        }


def test_path_artifact_and_latest_refusal(screening, tmp_path):
    models, metadata, _, _, source = screening
    saved = save_path_model(tmp_path / "model", models, metadata)
    with pytest.raises(ValueError, match="trust"):
        load_path_model(tmp_path / "model")
    loaded, record = load_path_model(tmp_path / "model", trusted=True)
    clock = (BASE + len(source) * MINUTE_US) // 1000 + 1000
    result = forecast_path(loaded, record, source, clock)
    assert len(result["dimensions"]) == 4
    assert result["model_sha256"] == saved["model_sha256"]
    assert "success probability" in render_path_forecast(result)
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="stale"):
        forecast_path(loaded, record, source.head(-1), clock)
    invalid = source.with_columns(pl.lit(False).alias("quality_complete"))
    with pytest.raises(ValueError, match="history"):
        forecast_path(loaded, record, invalid, clock)
    assert np.isfinite([d["value"] for d in result["dimensions"]]).all()


def test_dataset_batch_reuse_and_identity(tmp_path, monkeypatch):
    from crypto_boom.research import path_dataset

    source = tmp_path / "source.parquet"
    bars(2200).write_parquet(source)
    pool = tmp_path / "pool.json"
    pool.write_text("{}", encoding="utf-8")
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
    first = path_dataset.build_path_dataset(
        pool, output_root=tmp_path / "batch", spec=PathTargetSpec((10,), ())
    )
    second = path_dataset.build_path_dataset(
        pool, output_root=tmp_path / "batch", spec=PathTargetSpec((10,), ())
    )
    assert first["dataset_id"] == second["dataset_id"]
    assert second["batches"][0]["feature_reused"]
    assert second["batches"][0]["target_reused"]
    from pathlib import Path

    manifest = Path(first["manifest"])
    rows, _ = path_dataset.load_path_dataset(manifest)
    assert len(rows) > 0 and "up_10" in rows.columns
    record = json.loads(manifest.read_text())
    record["batches"][0]["feature_id"] = "tampered"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        path_dataset.load_path_dataset(manifest)


@pytest.mark.asyncio
async def test_historical_probe_preserves_catalog_identity(tmp_path, monkeypatch):
    from dataclasses import asdict
    from datetime import date

    from crypto_boom import _artifacts
    from crypto_boom.history import availability

    catalog = HistoricalArchiveSymbolCatalog(
        "https://example.test", "monthly/", 1, 1, ("DEADUSDT",)
    )
    request = availability.MonthlyAvailabilityRequest(
        date(2025, 1, 1), date(2025, 1, 1)
    )

    async def probe(symbols, pool_id, received, **kwargs):
        assert symbols == ("DEADUSDT",)
        assert pool_id == _artifacts.content_id(asdict(catalog))
        assert received is request
        return "observed"

    monkeypatch.setattr(availability, "_probe_selected_symbols", probe)
    assert (
        await availability.probe_historical_archive_availability(
            catalog, request, symbols=("DEADUSDT",), output_root=tmp_path
        )
        == "observed"
    )
    with pytest.raises(availability.AvailabilityIntegrityError):
        await availability.probe_historical_archive_availability(
            catalog, request, symbols=("UNKNOWNUSDT",), output_root=tmp_path
        )


def test_paired_weekly_interval_and_insufficient_support():
    from crypto_boom.research.path_screen import _weekly_gain_interval

    times = np.repeat(np.arange(30), 2) * 1440 * MINUTE_US
    y = np.ones(len(times))
    result = _weekly_gain_interval(y, y * 0.5, y * 0, times, 0.9)
    assert result["covered_days"] == 30
    assert result["gain_interval_95"] == pytest.approx([0.5, 0.5])
    assert _weekly_gain_interval(y, y, y, times, 0.9)["status"] == "zero_baseline_loss"
    assert (
        _weekly_gain_interval(y[:20], y[:20], y[:20], times[:20], 0.9)["status"]
        == "insufficient_days"
    )


def test_rolling_end_purge_reuse_and_tamper(screening, tmp_path, monkeypatch):
    from pathlib import Path

    from crypto_boom.research import path_dataset, path_screen

    _, _, rows, kwargs, _ = screening
    monkeypatch.setattr(
        path_dataset,
        "load_path_dataset",
        lambda p: (rows, {"dataset_id": "test-dataset"}),
    )
    window = (
        kwargs["train_end_us"],
        kwargs["selection_end_us"],
        BASE + 4300 * MINUTE_US,
    )
    arguments = {"windows": (window,), "output": tmp_path, "horizon": 10, "max_iter": 2}
    first = path_screen.rolling_path_audit(Path("fixture"), **arguments)
    fold = first["folds"][0]
    assert not fold["reused"]
    assert fold["report"]["test_last_us"] + 10 * MINUTE_US < window[2]
    assert fold["report"]["rows"]["test"] < 600
    for response in fold["report"]["responses"]:
        score = response["evaluation"]["test"]
        assert set(score["by_symbol"]["AAAUSDT"]["gain_vs_baselines"]) == {
            "fixed",
            "symbol",
            "volatility",
        }
    monkeypatch.setattr(
        path_screen,
        "screen_path_features",
        lambda *a, **k: pytest.fail("cache hit refitted model"),
    )
    assert path_screen.rolling_path_audit(Path("fixture"), **arguments)["folds"][0][
        "reused"
    ]
    report = Path(fold["path"])
    saved = json.loads(report.read_text())
    saved["record"]["report"]["rows"]["test"] = 0
    report.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        path_screen.rolling_path_audit(Path("fixture"), **arguments)
    with pytest.raises(ValueError, match="overlap"):
        path_screen.rolling_path_audit(
            Path("fixture"),
            windows=(window, window),
            output=tmp_path,
            horizon=10,
            max_iter=2,
        )


def test_continuous_targets_default_without_amplitude_thresholds():
    assert PathTargetSpec().amplitudes == ()
    source = bars(400)
    origins = pl.DataFrame({"symbol": ["AAAUSDT"], "decision_us": [BASE + MINUTE_US]})
    continuous = path_targets(source, origins)
    queried = path_targets(source, origins, PathTargetSpec(amplitudes=(0.1,)))
    assert not any("hit_" in name for name in continuous.columns)
    assert queried.select(continuous.columns).equals(continuous)
    assert "up_hit_120_0p1" in queried.columns


def test_origin_observability_is_causal_and_keeps_inactive_and_gap_origins():
    from crypto_boom.feature_batch import origin_observability

    source = bars(1600).with_row_index()
    source = source.with_columns(
        pl.when(pl.col("index") >= 1500)
        .then(0)
        .otherwise(pl.col("trade_count"))
        .alias("trade_count"),
        pl.when(pl.col("index") >= 1500)
        .then(0.0)
        .otherwise(pl.col("quote_turnover"))
        .alias("quote_turnover"),
        pl.when(pl.col("index") >= 1500)
        .then(0.0)
        .otherwise(pl.col("taker_buy_quote_turnover"))
        .alias("taker_buy_quote_turnover"),
    ).drop("index")
    ledger = origin_observability(source)
    assert len(ledger) == len(source)
    assert ledger["origin_admissible"][1440]
    assert ledger["selection_reason"][1500] == "origin_without_trade"
    assert ledger["minutes_since_traded_bar"][-1] == 100
    assert ledger["observed_fraction_60"][-1] == 0
    assert origin_observability(source.head(1550)).equals(ledger.head(1550))
    broken = source.filter(
        pl.col("open_time").dt.epoch("us") != BASE + 1490 * MINUTE_US
    )
    last = origin_observability(broken).row(-1, named=True)
    assert not last["origin_admissible"]
    assert last["contiguous_valid_minutes"] == 109
    invalid = source.with_columns(pl.lit(False).alias("quality_complete"))
    assert origin_observability(invalid)["contiguous_valid_minutes"].sum() == 0


def test_local_audit_distinguishes_quality_partial_month_and_local_absence(
    tmp_path, monkeypatch
):
    from datetime import date
    from types import SimpleNamespace

    from crypto_boom import sample_pool
    from crypto_boom.history.availability import MonthlyAvailabilityRequest

    root = tmp_path / "binance/spot/research-source-klines/AAAUSDT/1m"
    january = root / "2025-01/revision"
    february = root / "2025-02/revision"
    for part in (january, february):
        part.mkdir(parents=True)
        (part / "manifest.json").write_text("{}")
    source = bars(10).filter(pl.col("open_time").dt.epoch("us") != BASE + 4 * MINUTE_US)
    source = source.with_columns(
        pl.lit(0.0).alias("quote_turnover"),
        pl.lit(0.0).alias("taker_buy_quote_turnover"),
        pl.lit(0).alias("trade_count"),
    )
    source.write_parquet(january / "klines.parquet")
    (february / "klines.parquet").write_bytes(b"broken")

    def load(path):
        if path == february:
            raise sample_pool.ResearchStorageError("checksum mismatch")
        return SimpleNamespace(
            manifest=SimpleNamespace(
                row_count=9, source_revision="revision", parquet_sha256="hash"
            )
        )

    monkeypatch.setattr(sample_pool, "load_published_research_partition", load)
    result = sample_pool.audit_local_corpus(
        tmp_path, MonthlyAvailabilityRequest(date(2025, 1, 1), date(2025, 3, 1))
    )
    assert result["verified_partitions"] == result["rejected_partitions"] == 1
    valid = result["partitions"][0]
    assert valid["internal_missing_minutes"] == 1
    assert valid["zero_quote_rows"] == valid["zero_trade_rows"] == 9
    assert valid["calendar_unobserved_minutes"] == 31 * 1440 - 9
    assert result["locally_absent_months"] == {"AAAUSDT": ["2025-03"]}
