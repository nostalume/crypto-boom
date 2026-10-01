from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from crypto_boom import latest_market
from crypto_boom.bars import MINUTE_US, admit_bars, load_bar_files
from crypto_boom.research.forward import (
    FEATURES,
    HORIZONS,
    QUANTILES,
    ForecastGrid,
    forecast,
    latest_features,
    load_model,
    ordered_predictions,
    save_model,
)
from crypto_boom.research.forward_training import backtest, fit_forward, training_rows
from crypto_boom.research.predict_cli import main

BASE = 1_735_689_600_000_000


def bars(n=2400):
    close = 10 + np.sin(np.arange(n) / 30) + np.arange(n) / 10000
    return pl.DataFrame(
        {
            "symbol": ["AAAUSDT"] * n,
            "open_time": BASE + np.arange(n) * MINUTE_US,
            "close_price": close,
            "high_price": close + 0.1,
            "low_price": close - 0.1,
            "quote_turnover": np.full(n, 10000.0),
            "taker_buy_quote_turnover": np.full(n, 5000.0),
            "trade_count": np.full(n, 100),
            "quality_complete": [True] * n,
            "quality_state": ["valid"] * n,
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us", "UTC")))


def test_path_label_includes_exact_horizon_and_censors_tail():
    source = bars()
    rows = training_rows(source)
    origin = rows.row(0, named=True)
    index = (origin["decision_us"] - BASE) // MINUTE_US - 1
    for h in HORIZONS:
        expected = (
            max(source["close_price"][index : index + h + 1])
            / source["close_price"][index]
            - 1
        )
        assert origin[f"up_{h}"] == pytest.approx(expected)
    assert rows.tail(1)["up_120"][0] is None
    assert (rows.drop_nulls()["up_360"] >= rows.drop_nulls()["up_120"]).all()


def test_no_future_features_and_gap_does_not_become_negative_label():
    source = bars(3600)
    prefix = training_rows(source.head(2000))
    whole = training_rows(source)
    assert prefix.select("decision_us", *FEATURES).equals(
        whole.filter(pl.col("decision_us") <= BASE + 2000 * MINUTE_US).select(
            "decision_us", *FEATURES
        )
    )
    broken = source.with_row_index().filter(pl.col("index") != 2000).drop("index")
    before_gap = training_rows(broken).filter(
        pl.col("decision_us") == BASE + 1995 * MINUTE_US
    )
    assert before_gap["up_120"][0] is None
    assert before_gap["up_360"][0] is None


def test_latest_refuses_stale_unfinished_bad_quality_and_liquidity():
    source = bars()
    clock = (BASE + len(source) * MINUTE_US) // 1000 + 10_000
    assert len(latest_features(source, clock)) == 1
    for bad in (
        source.head(-1),
        source.with_columns(pl.lit(False).alias("quality_complete")),
        source.with_columns(
            pl.lit(0).alias("quote_turnover"),
            pl.lit(0).alias("taker_buy_quote_turnover"),
        ),
    ):
        with pytest.raises(ValueError):
            latest_features(bad, clock)
    with pytest.raises(ValueError):
        latest_features(source, clock - 60_000)
    with pytest.raises(ValueError, match="duplicate"):
        admit_bars(pl.concat([source, source.tail(1)]))
    with pytest.raises(ValueError, match="invalid price"):
        admit_bars(source.with_columns(pl.lit(float("nan")).alias("close_price")))


@pytest.fixture(scope="module")
def fitted():
    rows = training_rows(bars(12000))
    split = BASE + 9000 * MINUTE_US
    models, metadata = fit_forward(rows, split)
    return models, metadata, rows


def test_fit_temporal_purge_and_exact_scoring_surface(fitted):
    models, metadata, rows = fitted
    assert metadata["train_last_us"] + 360 * MINUTE_US < metadata["split_us"]
    assert metadata["train_rows"] >= 1000
    assert len(metadata["metrics"]) == 4
    predictions = ordered_predictions(models, rows.head(3).select(FEATURES).to_numpy())
    assert predictions.shape == (3, len(HORIZONS), len(QUANTILES))
    assert (predictions >= 0).all()
    assert (np.diff(predictions, axis=1) >= 0).all()
    assert (np.diff(predictions, axis=2) >= 0).all()
    with pytest.raises(ValueError, match="duplicate"):
        fit_forward(pl.concat([rows, rows.head(1)]), metadata["split_us"])


def test_artifact_trust_hash_version_and_text_json(tmp_path, fitted):
    models, metadata, _ = fitted
    directory = tmp_path / "model"
    saved = save_model(directory, models, metadata)
    with pytest.raises(ValueError, match="explicitly trust"):
        load_model(directory)
    restored, info = load_model(directory, trusted=True)
    source = bars().with_columns(pl.col("open_time") + pl.duration(days=30))
    clock = int(source["open_time"].dt.epoch("ms")[-1]) + 60_000
    result = forecast(source, clock, restored, info)
    assert result["model_sha256"] == saved["model_sha256"]
    assert len(result["forecasts"]) == 4
    with pytest.raises(FileExistsError):
        save_model(directory, models, metadata)
    (directory / "model.joblib").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        load_model(directory, trusted=True)


def test_latest_cli_no_network_before_trust(tmp_path, monkeypatch, capsys):
    def forbidden(_):
        raise AssertionError("network should not run")

    monkeypatch.setattr(latest_market, "fetch_latest", forbidden)
    assert (
        main(
            [
                "latest",
                "--symbol",
                "AAAUSDT",
                "--model",
                str(tmp_path),
                "--format",
                "json",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["status"] == "refused"


def mock_rest(monkeypatch, mutation=None, final_clock_offset=0):
    clock = BASE // 1000 + 3000 * 60_000 + 5000
    calls = []

    def response(path, params):
        calls.append((path, params))
        if path.endswith("time"):
            return {"serverTime": clock + (final_clock_offset if len(calls) > 1 else 0)}
        rows = [
            [
                t,
                "10",
                "11",
                "9",
                "10",
                "1000",
                t + 59_999,
                "10000",
                10,
                "500",
                "5000",
                "0",
            ]
            for t in range(params["startTime"], params["endTime"], 60_000)
        ]
        if mutation:
            mutation(rows)
        return rows

    monkeypatch.setattr(latest_market, "_get_json", response)
    return calls, clock


def test_rest_two_pages_exact_closed_history(monkeypatch):
    calls, clock = mock_rest(monkeypatch)
    source, final_clock = latest_market.fetch_latest("AAAUSDT", minutes=1441)
    assert len(source) == 1441
    assert final_clock == clock
    assert len(calls) == 4
    assert calls[1][1]["limit"] == 1000 and calls[2][1]["limit"] == 441
    assert len(latest_features(source, final_clock)) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[0].__setitem__(0, rows[1][0]),
        lambda rows: rows[0].__setitem__(6, 0),
    ],
)
def test_rest_refuses_missing_duplicate_or_unfinished(monkeypatch, mutation):
    mock_rest(monkeypatch, mutation)
    with pytest.raises(ValueError):
        latest_market.fetch_latest("AAAUSDT", minutes=1441)


def test_rest_refuses_minute_rollover(monkeypatch):
    mock_rest(monkeypatch, final_clock_offset=60_000)
    with pytest.raises(ValueError, match="stale"):
        latest_market.fetch_latest("AAAUSDT", minutes=1441)


def test_cli_latest_human_and_json(tmp_path, fitted, monkeypatch, capsys):
    models, metadata, _ = fitted
    directory = tmp_path / "model"
    save_model(directory, models, metadata)
    source = bars().with_columns(pl.col("open_time") + pl.duration(days=30))
    clock = int(source["open_time"].dt.epoch("ms")[-1]) + 60_000
    monkeypatch.setattr(
        latest_market, "fetch_latest", lambda _, **kwargs: (source, clock)
    )
    args = ["latest", "--symbol", "AAAUSDT", "--model", str(directory), "--trust-model"]
    assert main(args) == 0
    assert "P90" in capsys.readouterr().out
    assert main([*args, "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "experimental_not_forward_validated"
    assert result["source"].startswith("Binance Spot")
    assert result["origin_age_seconds"] == 0
    changed = json.loads((directory / "metadata.json").read_text())
    changed["sklearn_version"] = "incompatible"
    (directory / "metadata.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="sklearn_version"):
        load_model(directory, trusted=True)


def test_cli_training_uses_raw_origins_and_saves_receipt(tmp_path, capsys):
    path = tmp_path / "bars.parquet"
    bars(12000).write_parquet(path)
    model = tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--bars",
                str(path),
                "--model",
                str(model),
                "--start",
                "2025-01-02T00:00:00Z",
                "--split",
                "2025-01-07T06:00:00Z",
                "--end",
                "2025-01-10T00:00:00Z",
                "--horizons",
                "60",
                "720",
                "--quantiles",
                "0.5",
                "0.8",
            ]
        )
        == 0
    )
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["inputs"][0]["name"] == path.name
    assert metadata["train_rows"] >= 1000
    assert metadata["validation_rows"] >= 200
    assert (model / "model.joblib").is_file()
    assert metadata["train_last_us"] + 720 * MINUTE_US < metadata["split_us"]
    restored, info = load_model(model, trusted=True)
    source = bars().with_columns(pl.col("open_time") + pl.duration(days=30))
    clock = int(source["open_time"].dt.epoch("ms")[-1]) + 60_000
    result = forecast(source, clock, restored, info)
    assert [(r["horizon_minutes"], r["quantile"]) for r in result["forecasts"]] == [
        (60, 0.5),
        (60, 0.8),
        (720, 0.5),
        (720, 0.8),
    ]


def test_training_bad_schema_is_structured_refusal(tmp_path, capsys):
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"candidate": [1]}).write_parquet(path)
    assert (
        main(
            [
                "train",
                "--bars",
                str(path),
                "--model",
                str(tmp_path / "model"),
                "--start",
                "2025-01-02T00:00:00Z",
                "--split",
                "2025-01-07T00:00:00Z",
                "--end",
                "2025-01-10T00:00:00Z",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["status"] == "refused"
    assert not (tmp_path / "model").exists()


def test_backtest_is_frozen_later_only_and_reports_slices(
    tmp_path, fitted, monkeypatch, capsys
):
    models, metadata, rows = fitted
    model = tmp_path / "model"
    metadata = save_model(model, models, metadata)
    original = (model / "model.joblib").read_bytes()

    def no_fit(*args, **kwargs):
        raise AssertionError("backtest must not fit")

    monkeypatch.setattr(type(models[0]), "fit", no_fit)
    with pytest.raises(ValueError, match="after all"):
        backtest(rows, models, metadata)
    later = rows.with_columns(pl.col("decision_us") + 30 * 1440 * MINUTE_US)
    report = backtest(later, models, metadata)
    assert report["censored_rows"] > 0
    assert report["metrics"][0]["by_month"]
    assert (
        report["metrics"][0]["by_symbol"]["AAAUSDT"]["rows"] == report["evaluated_rows"]
    )
    assert report["horizon_spaced_minutes"] > max(HORIZONS)
    assert report["metrics"][0]["horizon_spaced"]["rows"] < report["evaluated_rows"]
    assert all(
        m["baseline_quantile"] == old["baseline_quantile"]
        for m, old in zip(report["metrics"], metadata["metrics"], strict=True)
    )
    source = bars(4000).with_columns(pl.col("open_time") + pl.duration(days=30))
    path = tmp_path / "later.parquet"
    source.write_parquet(path)
    assert (
        main(
            [
                "backtest",
                "--bars",
                str(path),
                "--start",
                "2025-01-31T00:00:00Z",
                "--end",
                "2025-02-04T00:00:00Z",
                "--model",
                str(model),
                "--trust-model",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["model_sha256"] == metadata["model_sha256"]
    )
    assert (model / "model.joblib").read_bytes() == original


def test_common_bar_contract_is_model_free_and_windowed(tmp_path):
    source = bars().with_columns(pl.lit("ETHBTC").alias("symbol"))
    assert len(admit_bars(source)) == len(source)
    path = tmp_path / "canonical.parquet"
    source.write_parquet(path)
    frame, receipt = load_bar_files([path], start_us=BASE, end_us=BASE + 10 * MINUTE_US)
    assert len(frame) == 9  # exclusive observation cutoff, not merely bar open
    assert len(receipt[0]["sha256"]) == 64
    with pytest.raises(ValueError, match="duplicate"):
        load_bar_files([path, path], start_us=BASE, end_us=BASE + 10 * MINUTE_US)
    with pytest.raises(ValueError, match="USDT"):
        training_rows(source)


def test_output_grid_rejects_ambiguous_or_unbounded_coordinates():
    for horizons, quantiles in [
        ((360, 120), (0.5, 0.9)),
        ((1, 1), (0.5,)),
        ((True,), (0.5,)),
        ((4321,), (0.5,)),
        ((120,), (0.9, 0.5)),
        ((120,), (float("nan"),)),
        ((120,), (1.0,)),
    ]:
        with pytest.raises(ValueError):
            ForecastGrid(horizons, quantiles)


def test_framework_and_deployment_do_not_import_research_fitting(tmp_path):
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = f"""
import sys
sys.path.insert(0, {src!r})
import crypto_boom.bars, crypto_boom.features, crypto_boom.latest_market, crypto_boom.market_data
assert "crypto_boom.qualification" not in sys.modules
assert not any(k.startswith(('crypto_boom.research', 'sklearn')) for k in sys.modules)
import crypto_boom.research.forward, crypto_boom.research.predict_cli
assert 'crypto_boom.research.forward_training' not in sys.modules
assert 'tsfel' not in sys.modules and 'interpret' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_window_loading_preserves_file_order_duplicates_and_receipts(tmp_path, reverse):
    source = bars(30)
    pieces = [source.slice(10, 15), source.slice(0, 20)]
    if reverse:
        pieces.reverse()
    paths = [tmp_path / f"{i}.parquet" for i in range(2)]
    for piece, path in zip(pieces, paths, strict=True):
        piece.write_parquet(path, row_group_size=5)
    start, end = BASE + 5 * MINUTE_US, BASE + 23 * MINUTE_US
    result, receipts = load_bar_files(paths, start_us=start, end_us=end)
    expected = pl.concat(pieces).filter(
        (pl.col("open_time").dt.epoch("us") >= start)
        & (pl.col("open_time").dt.epoch("us") + MINUTE_US < end)
    )
    assert result.equals(expected)
    assert len(receipts) == 2 and [r["name"] for r in receipts] == [
        p.name for p in paths
    ]
    with pytest.raises(ValueError, match="duplicate"):
        admit_bars(result)


def test_window_loading_checks_schema_even_outside_requested_window(tmp_path):
    path = tmp_path / "wrong-time.parquet"
    bars(30).with_columns(pl.col("open_time").dt.replace_time_zone(None)).write_parquet(
        path
    )
    with pytest.raises(ValueError, match="UTC"):
        load_bar_files([path], start_us=0, end_us=MINUTE_US)
