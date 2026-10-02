"""Experimental multidimensional path inference; no acquisition or fitting."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from crypto_boom.bars import MINUTE_US, admit_bars
from crypto_boom.feature_batch import BATCH_FEATURES
from crypto_boom.features import iter_feature_segments


def _validate(models: dict, metadata: dict) -> None:
    if (
        metadata.get("schema") != "path-quantiles-v1"
        or metadata.get("sklearn_version") != sklearn.__version__
    ):
        raise ValueError("incompatible path model schema or sklearn version")
    h = metadata.get("horizon")
    if type(h) is not int or not 1 <= h <= 4320:
        raise ValueError("invalid path model horizon")
    expected = [
        (f"up_{h}", 0.9),
        (f"down_{h}", 0.9),
        (f"terminal_{h}", 0.5),
        (f"efficiency_{h}", 0.5),
    ]
    responses = metadata.get("responses", [])
    if [(r.get("target"), r.get("quantile")) for r in responses] != expected or set(
        models
    ) != {k for k, _ in expected}:
        raise ValueError("invalid path model coordinates")
    for r in responses:
        columns = r["features"]
        model = models[r["target"]]
        if (
            not columns
            or len(set(columns)) != len(columns)
            or not set(columns).issubset(BATCH_FEATURES)
        ):
            raise ValueError("invalid path model features")
        if (
            not isinstance(model, HistGradientBoostingRegressor)
            or model.n_features_in_ != len(columns)
            or model.loss != "quantile"
            or model.quantile != r["quantile"]
        ):
            raise ValueError("invalid path model estimator")
    if (
        not metadata.get("symbols")
        or not np.isfinite(metadata.get("minimum_turnover", 1_000_000))
        or metadata.get("minimum_turnover", 1_000_000) < 0
    ):
        raise ValueError("invalid model population policy")


def save_path_model(directory: Path, models: dict, metadata: dict) -> dict:
    _validate(models, metadata)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "model.joblib"
    joblib.dump(models, path, compress=3)
    record = {**metadata, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    return record


def load_path_model(directory: Path, *, trusted: bool = False) -> tuple[dict, dict]:
    if not trusted:
        raise ValueError(
            "joblib can execute code; explicitly trust only your own model"
        )
    path, manifest = directory / "model.joblib", directory / "metadata.json"
    if path.stat().st_size > 100_000_000 or manifest.stat().st_size > 2_000_000:
        raise ValueError("path model exceeds size budget")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.get("model_sha256"):
        raise ValueError("path model hash mismatch")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError("incompatible sklearn version")
    models = joblib.load(io.BytesIO(payload))
    _validate(models, metadata)
    return models, metadata


def forecast_path(
    models: dict, metadata: dict, source: pl.DataFrame, server_ms: int
) -> dict:
    """Predict from the exact latest closed minute, or fail instead of using stale data."""
    _validate(models, metadata)
    source = admit_bars(source)
    expected = server_ms // 60_000 * MINUTE_US
    if source["open_time"].dt.epoch("us")[-1] + MINUTE_US != expected:
        raise ValueError("stale or unfinished source bars")
    parts = list(iter_feature_segments(source, rich=True))
    if not parts:
        raise ValueError("insufficient continuous history")
    row = parts[-1].tail(1)
    if (
        row["decision_us"][0] != expected
        or row["turnover_1440"][0] < metadata.get("minimum_turnover", 1_000_000)
        or not row.select(pl.all_horizontal(pl.col(*BATCH_FEATURES).is_finite())).item()
    ):
        raise ValueError("latest minute is invalid or liquidity-ineligible")
    dimensions = []
    with threadpool_limits(limits=2):
        for r in metadata["responses"]:
            value = float(
                models[r["target"]].predict(row.select(r["features"]).to_numpy())[0]
            )
            if not np.isfinite(value):
                raise ValueError("nonfinite path prediction")
            dimensions.append(
                {"target": r["target"], "quantile": r["quantile"], "value": value}
            )
    symbol = row["symbol"][0]
    return {
        "symbol": symbol,
        "decision_us": expected,
        "horizon_minutes": metadata["horizon"],
        "dimensions": dimensions,
        "evidence_status": "experimental"
        if symbol in metadata["symbols"]
        else "outside_training_symbols",
        "assessment": "Interpret excursions, terminal direction and efficiency separately, not as a trading score. Marginal quantiles are not joint probabilities.",
        "model_sha256": metadata.get("model_sha256"),
    }


def render_path_forecast(result: dict) -> str:
    names = {
        "up": "Maximum upside",
        "down": "Maximum downside",
        "terminal": "Terminal return",
        "efficiency": "Signed path efficiency",
    }
    lines = [
        f"{result['symbol']} | Horizon: {result['horizon_minutes']} minutes | {result['evidence_status']}",
        f"Origin UTC: {datetime.fromtimestamp(result['decision_us'] / 1_000_000, tz=UTC).isoformat()}",
    ]
    for item in result["dimensions"]:
        name = item["target"].rsplit("_", 1)[0]
        value = (
            f"{item['value']:.3f}" if name == "efficiency" else f"{item['value']:+.2%}"
        )
        if name == "down":
            value = f"{item['value']:.2%}"
        lines.append(f"{names[name]} P{item['quantile'] * 100:.0f}: {value}")
    return "\n".join(
        [
            *lines,
            result["assessment"],
            "P90 is a quantile, not a success probability; fees, slippage and executability are excluded.",
        ]
    )
