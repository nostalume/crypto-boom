"""Experimental hourly absolute-upside model; no transport or trading decisions."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US, admit_bars, load_bar_files
from crypto_boom.features import HOURLY_FEATURES, hourly_matrix


def _validate(model: HistGradientBoostingRegressor, metadata: dict) -> None:
    if (
        metadata.get("schema") != "hourly-upside-v1"
        or metadata.get("sklearn_version") != sklearn.__version__
        or metadata.get("features") != list(HOURLY_FEATURES)
        or metadata.get("horizon_minutes") != 360
        or metadata.get("quantile") != 0.9
        or not isinstance(metadata.get("provenance"), dict)
        or not isinstance(model, HistGradientBoostingRegressor)
        or model.n_features_in_ != 152
        or model.loss != "quantile"
        or model.quantile != 0.9
    ):
        raise ValueError("incompatible hourly model or feature recipe")


def save_hourly_model(
    directory: Path, model: HistGradientBoostingRegressor, *, provenance: dict
) -> dict:
    metadata = {
        "schema": "hourly-upside-v1",
        "sklearn_version": sklearn.__version__,
        "features": list(HOURLY_FEATURES),
        "horizon_minutes": 360,
        "quantile": 0.9,
        "provenance": provenance,
        "status": "experimental_not_trade_signal",
    }
    _validate(model, metadata)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        directory.parent, prefix="hourly-"
    ) as staging:
        joblib.dump(model, staging / "model.joblib", compress=3)
        metadata["model_sha256"] = _artifacts.file_identity(staging / "model.joblib")[0]
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        staging.rename(directory)
    return metadata


def _trusted_payload(path: Path, expected: str, trusted: bool) -> object:
    if not trusted:
        raise ValueError(
            "joblib can execute code; --trust-model is required for your own files"
        )
    if path.stat().st_size > 100_000_000:
        raise ValueError("model exceeds 100 MB budget")
    payload = path.read_bytes()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("model hash mismatch")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(io.BytesIO(payload))


def load_hourly_model(
    directory: Path, *, trusted: bool = False
) -> tuple[HistGradientBoostingRegressor, dict]:
    manifest = directory / "metadata.json"
    if manifest.stat().st_size > 2_000_000:
        raise ValueError("model metadata exceeds budget")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError("incompatible sklearn version")
    model = _trusted_payload(
        directory / "model.joblib", metadata["model_sha256"], trusted
    )
    if not isinstance(model, HistGradientBoostingRegressor):
        raise ValueError("invalid hourly estimator")
    _validate(model, metadata)
    return model, metadata


def export_hourly_study(
    study: Path, destination: Path, *, trusted: bool = False
) -> dict:
    """Import the selected scale study's one supported head, never failed heads."""
    if destination.exists():
        raise FileExistsError(destination)
    manifest = study / "report.json"
    if manifest.stat().st_size > 2_000_000:
        raise ValueError("study report exceeds budget")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if report.get("selected_up_variant") != "hour_scaled_context":
        raise ValueError("study did not select the hourly recipe")
    models = _trusted_payload(
        study / "models.joblib", report["artifacts"]["models.joblib"], trusted
    )
    if not isinstance(models, dict):
        raise ValueError("invalid study models")
    return save_hourly_model(
        destination,
        models["hour_scaled_context_up_360"],
        provenance={
            "kind": "selected_scale_study",
            "study_report_sha256": _artifacts.file_identity(manifest)[0],
            "source_model_sha256": report["artifacts"]["models.joblib"],
            "evaluation": report["targets"]["up_360"]["hour_scaled_context"],
            "training_symbols": "see study ledger; transfer outside studied universe unverified",
        },
    )


def fit_hourly_dataset(
    dataset_path: Path, *, train_end_us: int, destination: Path
) -> dict:
    """Fixed recipe on public path dataset; a new fit is NOT a validated model."""
    from crypto_boom.feature_batch import origin_observability, read_feature_cache
    from crypto_boom.research.path_dataset import load_path_dataset

    if destination.exists():
        raise FileExistsError(destination)
    rows, dataset = load_path_dataset(dataset_path)
    if "up_360" not in rows.columns:
        raise ValueError("dataset requires a 360-minute target")
    rows = rows.filter(
        (pl.col("decision_us") + 360 * MINUTE_US < train_end_us)
        & (pl.col("decision_us") % (360 * MINUTE_US) == 0)
        & pl.col("up_360").is_not_null()
    )
    if not 100 <= len(rows) <= 200_000:
        raise ValueError("require 100..200000 training origins")
    xs, ys, excluded = [], [], 0
    for batch in dataset["batches"]:
        selected = rows.filter(pl.col("symbol") == batch["symbol"])
        if selected.is_empty():
            continue
        _, receipt = read_feature_cache(Path(batch["features"]))
        paths = [Path(s["path"]) for s in receipt["spec"]["sources"]]
        for path, recorded in zip(paths, receipt["spec"]["sources"], strict=True):
            if _artifacts.file_identity(path)[0] != recorded["sha256"]:
                raise ValueError("source changed after dataset construction")
        source, _ = load_bar_files(paths, start_us=0, end_us=2**63 - 1)
        ready = (
            origin_observability(source)
            .filter(pl.col("feature_history_ready"))
            .select("symbol", "decision_us")
        )
        kept = selected.join(ready, on=["symbol", "decision_us"], how="semi")
        excluded += len(selected) - len(kept)
        if len(kept):
            xs.append(hourly_matrix(source, kept.select("symbol", "decision_us")))
            ys.append(kept["up_360"].to_numpy())
    if not ys or sum(len(y) for y in ys) < 100:
        raise ValueError("insufficient complete 24-hour history")
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.9,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        learning_rate=0.08,
        early_stopping=False,
        random_state=0,
    )
    with threadpool_limits(limits=2):
        model.fit(np.concatenate(xs), np.concatenate(ys))
    return save_hourly_model(
        destination,
        model,
        provenance={
            "kind": "new_fit_not_evaluated",
            "dataset_id": dataset["dataset_id"],
            "train_end_us": train_end_us,
            "training_rows": sum(len(y) for y in ys),
            "insufficient_history_rows": excluded,
        },
    )


def forecast_hourly(
    model: HistGradientBoostingRegressor, metadata: dict, source: pl.DataFrame
) -> dict:
    """Anchor to last completed UTC hour; disclose discarded newer minute bars."""
    _validate(model, metadata)
    source = admit_bars(source)
    last = int(source["open_time"].dt.epoch("us")[-1]) + MINUTE_US
    decision = last // (60 * MINUTE_US) * (60 * MINUTE_US)
    history = source.filter(
        pl.col("open_time").dt.epoch("us") + MINUTE_US <= decision
    ).tail(1441)
    keys = pl.DataFrame({"symbol": [source["symbol"][0]], "decision_us": [decision]})
    matrix = hourly_matrix(history, keys)
    with threadpool_limits(limits=2):
        value = float(model.predict(matrix)[0])
    if not np.isfinite(value) or value < 0:
        raise ValueError("invalid upside quantile")
    return {
        "symbol": source["symbol"][0],
        "decision_us": decision,
        "latest_closed_minute_us": last,
        "hour_lag_minutes": (last - decision) // MINUTE_US,
        "horizon_minutes": 360,
        "upside_p90": value,
        "model_sha256": metadata["model_sha256"],
        "observed_fraction_24h": float(matrix[0, -1]),
        "status": "experimental_not_trade_signal",
        "retention_forecast": None,
        "downside_forecast": None,
        "terminal_return_forecast": None,
        "interpretation": "未来六小时分钟收盘路径最大上涨幅度的P90估计;不是90%上涨概率或保证收益。",
    }
