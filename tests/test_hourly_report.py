"""Frozen hourly recipe, artifact boundary and live/replay report behavior."""

import json

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from crypto_boom import _artifacts, latest_market
from crypto_boom.bars import MINUTE_US
from crypto_boom.features import past_sequence
from crypto_boom.research.hourly import (
    export_hourly_study,
    fit_hourly_dataset,
    forecast_hourly,
    hourly_matrix,
    load_hourly_model,
    save_hourly_model,
)
from crypto_boom.research.hourly_cli import main
from crypto_boom.research.prediction_report import generate_prediction_report
from test_forward_prediction import BASE, bars


@pytest.fixture
def model_directory(tmp_path):
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=2, min_samples_leaf=2
    )
    rng = np.random.default_rng(0)
    with threadpool_limits(limits=2):
        model.fit(rng.normal(size=(30, 152)), rng.uniform(0.01, 0.1, 30))
    directory = tmp_path / "model"
    save_hourly_model(directory, model, provenance={"kind": "test"})
    return directory


def test_hourly_recipe_arithmetic_and_causal_prefix():
    source = bars(1600)
    keys = pl.DataFrame(
        {"symbol": ["AAAUSDT"], "decision_us": [BASE + 1500 * MINUTE_US]}
    )
    matrix = hourly_matrix(source, keys)
    np.testing.assert_array_equal(matrix, hourly_matrix(source.head(1500), keys))
    sequence = past_sequence(source, keys, history_minutes=1440, step_minutes=60)
    np.testing.assert_array_equal(
        matrix[0, :24], sequence["bucket_return"][0].to_numpy()
    )
    assert matrix.shape == (1, 152)
    assert matrix[0, 144] == pytest.approx(
        source["close_price"][1499] / source["close_price"][1439] - 1, abs=1e-6
    )
    assert matrix[0, 146] == pytest.approx(
        source["close_price"][1499] / source["close_price"][59] - 1, abs=1e-6
    )
    assert matrix[0, -1] == 1
    with pytest.raises(ValueError, match="UTC-hour"):
        hourly_matrix(source, keys.with_columns(pl.col("decision_us") + MINUTE_US))


def test_trust_hash_version_and_no_overwrite(model_directory):
    with pytest.raises(ValueError, match="trust"):
        load_hourly_model(model_directory)
    model, metadata = load_hourly_model(model_directory, trusted=True)
    with pytest.raises(FileExistsError):
        save_hourly_model(model_directory, model, provenance={})
    manifest = model_directory / "metadata.json"
    manifest.write_text(json.dumps({**metadata, "sklearn_version": "wrong"}))
    with pytest.raises(ValueError, match="version"):
        load_hourly_model(model_directory, trusted=True)
    manifest.write_text(json.dumps(metadata))
    with (model_directory / "model.joblib").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_hourly_model(model_directory, trusted=True)


def test_report_live_and_replay_same_hour(model_directory, tmp_path, monkeypatch):
    source = bars(1559)
    file = tmp_path / "bars.parquet"
    source.write_parquet(file)
    monkeypatch.setattr(
        latest_market,
        "fetch_latest",
        lambda symbol, minutes: (source, (BASE + 1559 * MINUTE_US) // 1000 + 100),
    )
    live = generate_prediction_report(
        model_directory, tmp_path / "live", symbol="AAAUSDT", trusted=True
    )
    replay = generate_prediction_report(
        model_directory, tmp_path / "replay", bar_paths=[file], trusted=True
    )
    assert live["decision_us"] == BASE + 1500 * MINUTE_US
    assert live["hour_lag_minutes"] == 59
    assert live["upside_p90"] == replay["upside_p90"]
    assert replay["mode"] == "historical_replay"
    assert replay["downside_forecast"] is None
    assert (tmp_path / "replay/report.md").is_file()
    assert (
        json.loads((tmp_path / "replay/prediction.json").read_text(encoding="utf-8"))[
            "upside_p90"
        ]
        == replay["upside_p90"]
    )
    with pytest.raises(FileExistsError):
        generate_prediction_report(
            model_directory, tmp_path / "replay", bar_paths=[file], trusted=True
        )
    with pytest.raises(ValueError, match="exactly one"):
        generate_prediction_report(model_directory, tmp_path / "none", trusted=True)
    model, metadata = load_hourly_model(model_directory, trusted=True)
    with pytest.raises(ValueError):
        forecast_hourly(model, metadata, source.tail(1000))


def test_report_cli_replay_and_failure_has_no_output(
    model_directory, tmp_path, monkeypatch
):
    source = bars(1501)
    file = tmp_path / "source.parquet"
    source.write_parquet(file)
    assert (
        main(
            [
                "report",
                "--model",
                str(model_directory),
                "--bars",
                str(file),
                "--output",
                str(tmp_path / "cli"),
                "--trust-model",
            ]
        )
        == 0
    )

    def fail(symbol, minutes):
        raise ValueError("network unavailable")

    monkeypatch.setattr(latest_market, "fetch_latest", fail)
    with pytest.raises(ValueError, match="network unavailable"):
        generate_prediction_report(
            model_directory, tmp_path / "failed", symbol="AAAUSDT", trusted=True
        )
    assert not (tmp_path / "failed").exists()


def test_export_only_selected_head(model_directory, tmp_path):
    import joblib

    model, _ = load_hourly_model(model_directory, trusted=True)
    study = tmp_path / "study"
    study.mkdir()
    joblib.dump(
        {"hour_scaled_context_up_360": model, "failed_head": None},
        study / "models.joblib",
    )
    (study / "report.json").write_text(
        json.dumps(
            {
                "selected_up_variant": "hour_scaled_context",
                "artifacts": {
                    "models.joblib": _artifacts.file_identity(study / "models.joblib")[
                        0
                    ]
                },
                "targets": {"up_360": {"hour_scaled_context": {}}},
            }
        )
    )
    result = export_hourly_study(study, tmp_path / "export", trusted=True)
    assert result["horizon_minutes"] == 360
    loaded, _ = load_hourly_model(tmp_path / "export", trusted=True)
    assert loaded.n_features_in_ == 152


def test_train_public_dataset_with_purge_and_provenance(tmp_path):
    from crypto_boom.feature_batch import build_feature_cache, read_feature_cache
    from crypto_boom.research.path_targets import (
        PathTargetSpec,
        build_target_cache,
        read_target_cache,
    )

    source = tmp_path / "bars.parquet"
    bars(40000).write_parquet(source)
    fp, _ = build_feature_cache(
        [source], output_root=tmp_path / "features", step_minutes=60, minimum_turnover=0
    )
    tp, _ = build_target_cache(
        fp, output_root=tmp_path / "targets", spec=PathTargetSpec((360,), ())
    )
    _, f = read_feature_cache(fp)
    _, t = read_target_cache(tp)
    batch = {
        "symbol": "AAAUSDT",
        "features": str(fp),
        "targets": str(tp),
        "feature_id": f["cache_id"],
        "target_id": t["cache_id"],
    }
    identity: dict[str, object] = {
        "pool_sha256": "test-local-pool",
        "batches": [{k: batch[k] for k in ("symbol", "feature_id", "target_id")}],
    }
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema": "path-dataset-v1",
                "identity": identity,
                "dataset_id": _artifacts.content_id(identity),
                "batches": [batch],
            }
        )
    )
    result = fit_hourly_dataset(
        dataset, train_end_us=BASE + 40000 * MINUTE_US, destination=tmp_path / "trained"
    )
    assert result["provenance"]["kind"] == "new_fit_not_evaluated"
    assert result["provenance"]["training_rows"] >= 100
    load_hourly_model(tmp_path / "trained", trusted=True)
