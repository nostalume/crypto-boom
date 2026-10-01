"""Experimental origin-forward close-path quantiles, not M1 continuation scores.

Features see completed bars only. Labels see a complete future horizon within
the same quality-contiguous segment; a missing future is never a negative label.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from crypto_boom.bars import MINUTE_US, admit_bars
from crypto_boom.features import (
    FLOW_FEATURES,
    PRICE_FEATURES,
    iter_feature_segments,
)

FEATURES = PRICE_FEATURES + FLOW_FEATURES
HORIZONS = (120, 360)
QUANTILES = (0.5, 0.9)
SCHEMA = "forward-close-quantiles-v1"
MIN_TURNOVER = 1_000_000
HISTORY_MINUTES = 1441


@dataclass(frozen=True)
class ForecastGrid:
    """Ordered output coordinates, persisted with the artifact; not event gates."""

    horizons: tuple[int, ...] = HORIZONS
    quantiles: tuple[float, ...] = QUANTILES

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.horizons) <= 8
            or any(type(h) is not int or not 1 <= h <= 4320 for h in self.horizons)
            or tuple(sorted(set(self.horizons))) != self.horizons
        ):
            raise ValueError(
                "horizons must be 1-8 increasing unique integers in [1, 4320]"
            )
        if (
            not 1 <= len(self.quantiles) <= 5
            or any(
                type(q) not in (float, int) or not math.isfinite(q) or not 0 < q < 1
                for q in self.quantiles
            )
            or tuple(sorted(set(self.quantiles))) != self.quantiles
        ):
            raise ValueError("quantiles must be 1-5 increasing unique values in (0, 1)")

    @classmethod
    def from_metadata(cls, metadata: dict) -> ForecastGrid:
        if not isinstance(metadata.get("horizons_minutes"), list) or not isinstance(
            metadata.get("quantiles"), list
        ):
            raise ValueError("missing model output grid")
        return cls(tuple(metadata["horizons_minutes"]), tuple(metadata["quantiles"]))


DEFAULT_GRID = ForecastGrid()


def eligible_features(frame: pl.DataFrame) -> pl.DataFrame:
    if not all(str(symbol).endswith("USDT") for symbol in frame["symbol"].unique()):
        raise ValueError("forward model requires USDT quote turnover")
    return frame.filter(
        (pl.col("turnover_1440") >= MIN_TURNOVER)
        & pl.all_horizontal(pl.col(*FEATURES).is_finite())
    )


def latest_features(source: pl.DataFrame, server_ms: int) -> pl.DataFrame:
    bars = admit_bars(source)
    # Require the exact latest closed minute, not the last surviving valid row.
    expected_us = (server_ms // 60_000) * MINUTE_US
    parts = list(iter_feature_segments(bars))
    if not parts:
        raise ValueError("insufficient continuous history")
    row = eligible_features(parts[-1]).tail(1)
    if row.is_empty() or row["decision_us"][0] != expected_us:
        raise ValueError(
            "latest closed minute is missing, invalid or liquidity-ineligible"
        )
    if bars["open_time"].dt.epoch("us")[-1] + MINUTE_US != expected_us:
        raise ValueError("input includes unfinished/future bars or is stale")
    return row


def ordered_predictions(
    models: list, x: np.ndarray, grid: ForecastGrid = DEFAULT_GRID
) -> np.ndarray:
    """Nonnegative, nested horizon/quantile surface; also used in evaluation."""
    with threadpool_limits(limits=2):
        raw = np.column_stack([model.predict(x) for model in models])
    shaped = raw.reshape(len(x), len(grid.horizons), len(grid.quantiles))
    return np.maximum.accumulate(
        np.maximum.accumulate(np.maximum(shaped, 0), axis=2), axis=1
    )


def save_model(directory: Path, models: list, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.joblib"
    joblib.dump(models, model_path, compress=3)
    metadata = dict(
        metadata, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    # Last file is the completion marker; an interrupted directory cannot load.
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def load_model(directory: Path, *, trusted: bool = False) -> tuple[list, dict]:
    if not trusted:
        raise ValueError(
            "joblib can execute code; explicitly trust only your own model"
        )
    manifest = directory / "metadata.json"
    path = directory / "model.joblib"
    if manifest.stat().st_size > 2_000_000 or path.stat().st_size > 100_000_000:
        raise ValueError("model artifact exceeds size budget")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("invalid model metadata")
    for key, expected in {
        "schema": SCHEMA,
        "sklearn_version": sklearn.__version__,
        "features": list(FEATURES),
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(f"incompatible model {key}")
    grid = ForecastGrid.from_metadata(metadata)
    if (
        not isinstance(metadata.get("symbols"), list)
        or not metadata["symbols"]
        or any(
            not isinstance(s, str) or not s.endswith("USDT")
            for s in metadata["symbols"]
        )
        or any(
            type(metadata.get(key)) is not int
            for key in (
                "train_first_us",
                "train_last_us",
                "split_us",
                "validation_last_us",
            )
        )
        or not isinstance(metadata.get("metrics"), list)
        or metadata.get("status") != "experimental_not_forward_validated"
        or not isinstance(metadata.get("target"), str)
        or type(metadata.get("all_coordinates_beat_baseline")) is not bool
    ):
        raise ValueError("invalid model metadata fields")
    if (
        not metadata["train_first_us"]
        <= metadata["train_last_us"]
        < metadata["split_us"]
        <= metadata["validation_last_us"]
    ):
        raise ValueError("invalid model temporal interval")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.get("model_sha256"):
        raise ValueError("model hash mismatch")
    models = joblib.load(io.BytesIO(payload))
    if (
        not isinstance(models, list)
        or len(models) != len(grid.horizons) * len(grid.quantiles)
        or any(
            not isinstance(model, HistGradientBoostingRegressor)
            or model.n_features_in_ != len(FEATURES)
            for model in models
        )
    ):
        raise ValueError("invalid model estimators")
    if any(
        model.loss != "quantile" or model.quantile != q
        for model, q in zip(models, grid.quantiles * len(grid.horizons), strict=True)
    ):
        raise ValueError("model estimators do not match output grid")
    return models, metadata


def forecast(
    source: pl.DataFrame, server_ms: int, models: list, metadata: dict
) -> dict:
    row = latest_features(source, server_ms)
    grid = ForecastGrid.from_metadata(metadata)
    symbol = row["symbol"][0]
    if symbol not in metadata["symbols"]:
        raise ValueError("symbol outside fitted population; retrain explicitly")
    if (
        row["decision_us"][0]
        <= metadata["validation_last_us"] + max(grid.horizons) * MINUTE_US
    ):
        raise ValueError("latest origin must be after the model's validation interval")
    predictions = ordered_predictions(models, row.select(FEATURES).to_numpy(), grid)[0]
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite forecast")
    return {
        "status": metadata["status"],
        "symbol": symbol,
        "origin_utc": datetime.fromtimestamp(
            row["decision_us"][0] / 1e6, UTC
        ).isoformat(),
        "origin_close": row["close_price"][0],
        "server_time_ms": server_ms,
        "origin_age_seconds": (server_ms * 1000 - row["decision_us"][0]) / 1e6,
        "model_sha256": metadata["model_sha256"],
        "training_last_origin_utc": datetime.fromtimestamp(
            metadata["train_last_us"] / 1e6, UTC
        ).isoformat(),
        "validation_observation_end_utc": datetime.fromtimestamp(
            (metadata["validation_last_us"] + max(grid.horizons) * MINUTE_US) / 1e6, UTC
        ).isoformat(),
        "target": metadata["target"],
        "training_symbols": metadata["symbols"],
        "all_coordinates_beat_baseline": metadata["all_coordinates_beat_baseline"],
        "observed": {
            "return_60": row["return_60"][0],
            "turnover_24h_usdt": row["turnover_1440"][0],
        },
        "forecasts": [
            {
                "horizon_minutes": h,
                "quantile": q,
                "max_close_rise_fraction": float(predictions[i, j]),
            }
            for i, h in enumerate(grid.horizons)
            for j, q in enumerate(grid.quantiles)
        ],
        "warning": "实验性分位数估计,不是达到涨幅的概率、期末收益或收益保证;未预测下跌风险。",
    }
