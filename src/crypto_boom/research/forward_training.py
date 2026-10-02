"""Offline forward labels, fitting and frozen-model backtesting. No market I/O."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import sklearn
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits

from crypto_boom.bars import MINUTE_US, admit_bars
from crypto_boom.features import iter_feature_segments
from crypto_boom.research._estimators import path_regressor
from crypto_boom.research.forward import (
    DEFAULT_GRID,
    FEATURES,
    SCHEMA,
    ForecastGrid,
    eligible_features,
    ordered_predictions,
)


def training_rows(
    source: pl.DataFrame, grid: ForecastGrid = DEFAULT_GRID
) -> pl.DataFrame:
    """Every five minutes, origin-inclusive maximum future close return."""
    parts = []
    for frame in iter_feature_segments(admit_bars(source)):
        frame = frame.with_columns(
            *[
                (
                    pl.col("close_price").rolling_max(h + 1).shift(-h)
                    / pl.col("close_price")
                    - 1
                ).alias(f"up_{h}")
                for h in grid.horizons
            ]
        )
        parts.append(
            eligible_features(frame)
            .filter(pl.col("decision_us") % (5 * MINUTE_US) == 0)
            .select(
                "symbol", "decision_us", *FEATURES, *[f"up_{h}" for h in grid.horizons]
            )
        )
    if not parts:
        raise ValueError("no continuous 1441-minute feature history")
    return pl.concat(parts)


def fit_forward(
    rows: pl.DataFrame, split_us: int, grid: ForecastGrid = DEFAULT_GRID
) -> tuple[list, dict]:
    candidate_valid = rows.filter(pl.col("decision_us") >= split_us)
    rows = _complete_rows(rows, grid)
    train = rows.filter(
        pl.col("decision_us") + max(grid.horizons) * MINUTE_US < split_us
    )
    valid = rows.filter(pl.col("decision_us") >= split_us)
    if len(train) < 1000 or len(valid) < 200:
        raise ValueError(
            "need at least 1000 training and 200 validation origins after purge"
        )
    if set(valid["symbol"]) != set(train["symbol"]):
        raise ValueError("train and validation symbol coverage must match")
    x_train = train.select(FEATURES).to_numpy()
    models, baselines = [], []
    with threadpool_limits(limits=2):
        for h in grid.horizons:
            y = train[f"up_{h}"].to_numpy()
            for q in grid.quantiles:
                model = path_regressor(q, max_iter=80).fit(x_train, y)
                models.append(model)
                baselines.append(float(np.quantile(y, q)))
    evaluation = evaluate_rows(candidate_valid, models, baselines, grid)
    metrics = evaluation["metrics"]
    metadata = {
        "schema": SCHEMA,
        "sklearn_version": sklearn.__version__,
        "features": list(FEATURES),
        "horizons_minutes": list(grid.horizons),
        "quantiles": list(grid.quantiles),
        "status": "experimental_not_forward_validated",
        "target": "max(0, max future minute close / origin close - 1)",
        "symbols": sorted(set(train["symbol"])),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_first_us": int(train["decision_us"][0]),
        "train_last_us": int(train["decision_us"][-1]),
        "split_us": split_us,
        "validation_last_us": int(valid["decision_us"][-1]),
        "metrics": metrics,
        "validation_support": {
            key: value for key, value in evaluation.items() if key != "metrics"
        },
        "all_coordinates_beat_baseline": all(
            m["pinball"] < m["baseline_pinball"] for m in metrics
        ),
        "limitations": "Selected symbols; overlapping origins; exploratory temporal validation, not independent trials or calibrated crossing probabilities.",
        "created_utc": datetime.now(UTC).isoformat(),
    }
    return models, metadata


def _complete_rows(rows: pl.DataFrame, grid: ForecastGrid) -> pl.DataFrame:
    if rows.select(pl.struct("symbol", "decision_us").n_unique()).item() != len(rows):
        raise ValueError("duplicate training/backtest origin")
    complete = rows.drop_nulls([f"up_{h}" for h in grid.horizons]).sort(
        "decision_us", "symbol"
    )
    if complete.is_empty():
        raise ValueError("no complete future labels")
    if not np.isfinite(complete.select(FEATURES).to_numpy()).all():
        raise ValueError("nonfinite features")
    targets = complete.select([f"up_{h}" for h in grid.horizons]).to_numpy()
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("invalid forward labels")
    return complete


def _metrics(y: np.ndarray, prediction: np.ndarray, baseline: float, q: float) -> dict:
    loss = float(mean_pinball_loss(y, prediction, alpha=q))
    base_loss = float(mean_pinball_loss(y, np.full(len(y), baseline), alpha=q))
    return {
        "rows": len(y),
        "pinball": loss,
        "baseline_pinball": base_loss,
        "relative_improvement": 1 - loss / base_loss if base_loss > 0 else None,
        "observed_coverage": float(np.mean(y <= prediction)),
        "baseline_coverage": float(np.mean(y <= baseline)),
    }


def evaluate_rows(
    rows: pl.DataFrame, models: list, baselines: list[float], grid: ForecastGrid
) -> dict:
    """One evaluator for fit validation and frozen-model backtest; no fitting."""
    complete = _complete_rows(rows, grid)
    if (
        len(baselines) != len(grid.horizons) * len(grid.quantiles)
        or not np.isfinite(baselines).all()
    ):
        raise ValueError("incompatible baseline coordinates")
    predictions = ordered_predictions(
        models, complete.select(FEATURES).to_numpy(), grid
    )
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite predictions")
    symbols = complete["symbol"].to_numpy()
    months = (
        complete["decision_us"]
        .cast(pl.Datetime("us", "UTC"))
        .dt.strftime("%Y-%m")
        .to_numpy()
    )
    # More than the maximum label horizon, aligned to the five-minute origin grid.
    spacing = (max(grid.horizons) // 5 + 1) * 5
    sparse = complete["decision_us"].to_numpy() % (spacing * MINUTE_US) == 0
    metrics = []
    tail_support = []
    for i, h in enumerate(grid.horizons):
        y = complete[f"up_{h}"].to_numpy()
        for threshold in (0.10, 0.20):
            reached = complete.filter(pl.col(f"up_{h}") >= threshold)
            tail_support.append(
                {
                    "horizon_minutes": h,
                    "rise_fraction": threshold,
                    "origin_rows": len(reached),
                    "symbol_days": reached.select(
                        pl.struct(
                            "symbol",
                            (pl.col("decision_us") // (1440 * MINUTE_US)).alias("day"),
                        ).n_unique()
                    ).item(),
                }
            )
        for j, q in enumerate(grid.quantiles):
            pred = predictions[:, i, j]
            base = baselines[i * len(grid.quantiles) + j]
            metric = {
                "horizon_minutes": h,
                "quantile": q,
                "baseline_quantile": base,
                **_metrics(y, pred, base, q),
                "horizon_spaced": _metrics(y[sparse], pred[sparse], base, q)
                if sparse.any()
                else None,
            }
            for name, groups in (("by_symbol", symbols), ("by_month", months)):
                metric[name] = {
                    str(group): _metrics(y[mask], pred[mask], base, q)
                    for group in sorted(set(groups))
                    if (mask := groups == group).any()
                }
            metrics.append(metric)
    return {
        "candidate_rows": len(rows),
        "evaluated_rows": len(complete),
        "censored_rows": len(rows) - len(complete),
        "first_origin_us": int(complete["decision_us"][0]),
        "last_origin_us": int(complete["decision_us"][-1]),
        "symbols": sorted(set(symbols)),
        "horizon_spaced_minutes": spacing,
        "metrics": metrics,
        "observed_tail_support": tail_support,
        "limitations": "Selected symbols; overlapping origins and cross-symbol dependence. Spaced windows are a sensitivity check, not independent trials. No execution/PnL backtest.",
    }


def backtest(rows: pl.DataFrame, models: list, metadata: dict) -> dict:
    """Evaluate a frozen model strictly after its validation observation window."""
    grid = ForecastGrid.from_metadata(metadata)
    cutoff = metadata["validation_last_us"] + max(grid.horizons) * MINUTE_US
    if rows.is_empty() or rows.filter(pl.col("decision_us") <= cutoff).height:
        raise ValueError(
            "backtest origins must be after all training/validation observations"
        )
    if not set(rows["symbol"]).issubset(metadata["symbols"]):
        raise ValueError("backtest symbol outside fitted population")
    expected = [(h, q) for h in grid.horizons for q in grid.quantiles]
    coordinates = [(m["horizon_minutes"], m["quantile"]) for m in metadata["metrics"]]
    if coordinates != expected:
        raise ValueError("incompatible baseline metadata")
    report = evaluate_rows(
        rows, models, [m["baseline_quantile"] for m in metadata["metrics"]], grid
    )
    report.update(
        {
            "model_sha256": metadata["model_sha256"],
            "status": "historical_prediction_backtest_not_live_validation",
            "model_observations_end_us": cutoff,
            "missing_training_symbols": sorted(
                set(metadata["symbols"]) - set(report["symbols"])
            ),
            "all_coordinates_beat_baseline": all(
                m["pinball"] < m["baseline_pinball"] for m in report["metrics"]
            ),
        }
    )
    return report
