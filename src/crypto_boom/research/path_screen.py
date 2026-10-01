"""Bounded grouped feature selection on cached rows, followed by a later test."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US
from crypto_boom.feature_batch import BATCH_FEATURES, FEATURE_GROUPS


def _scores(y: np.ndarray, pred: np.ndarray, q: float) -> dict:
    return {
        "rows": len(y),
        "pinball": float(mean_pinball_loss(y, pred, alpha=q)),
        "coverage": float(np.mean(y <= pred)),
    }


def _comparison(y: np.ndarray, pred: np.ndarray, baselines: dict, q: float) -> dict:
    scored = _scores(y, pred, q)
    reference = {k: _scores(y, v, q) for k, v in baselines.items()}
    return {
        **scored,
        "baselines": reference,
        "gain_vs_baselines": {
            k: 1 - scored["pinball"] / v["pinball"] if v["pinball"] else None
            for k, v in reference.items()
        },
    }


def _weekly_gain_interval(
    y: np.ndarray, pred: np.ndarray, baseline: np.ndarray, times: np.ndarray, q: float
) -> dict:
    """Paired moving calendar-week bootstrap; all symbols share each time block."""
    days = times // (1440 * MINUTE_US)
    unique_days = np.unique(days)
    result = {
        "block_days": 7,
        "draws": 400,
        "covered_days": len(unique_days),
        "scope": "conditional_on_fitted_model; not_selection_adjusted; dependence_beyond_week_not_preserved",
    }
    if len(unique_days) < 28:
        return {**result, "status": "insufficient_days", "gain_interval_95": None}
    span = int(days.max() - days.min() + 1)
    if span > 3660:
        raise ValueError("bootstrap period exceeds ten years")
    index = days - days.min()
    losses = []
    for estimate in (pred, baseline):
        error = y - estimate
        losses.append(
            np.bincount(
                index, weights=np.maximum(q * error, (q - 1) * error), minlength=span
            )
        )
    starts = np.random.default_rng(0).integers(0, span - 6, size=(400, (span + 6) // 7))
    sampled = (starts[:, :, None] + np.arange(7)).reshape(400, -1)[:, :span]
    numerator, denominator = (loss[sampled].sum(axis=1) for loss in losses)
    if (denominator <= 0).any():
        return {**result, "status": "zero_baseline_loss", "gain_interval_95": None}
    return {
        **result,
        "status": "estimated",
        "gain_interval_95": np.quantile(
            1 - numerator / denominator, [0.025, 0.975]
        ).tolist(),
    }


def screen_path_features(
    rows: pl.DataFrame,
    *,
    train_end_us: int,
    selection_end_us: int,
    horizon: int = 360,
    max_iter: int = 60,
) -> tuple[dict, dict]:
    """Four cumulative feature sets, four responses; no test-period selection.

    Cuts are exclusive label boundaries: purge H minutes before both cuts.
    Models remain fitted only on training rows. Output is experimental, not alpha.
    """
    if type(horizon) is not int or not 1 <= horizon <= 4320 or not 1 <= max_iter <= 100:
        raise ValueError("invalid bounded screening configuration")
    if train_end_us >= selection_end_us or len(rows) > 2_000_000:
        raise ValueError("invalid temporal boundaries or excessive rows")
    coordinates = [
        (f"up_{horizon}", 0.9),
        (f"down_{horizon}", 0.9),
        (f"terminal_{horizon}", 0.5),
        (f"efficiency_{horizon}", 0.5),
    ]
    labels = [name for name, _ in coordinates]
    if rows.select(pl.struct("symbol", "decision_us").n_unique()).item() != len(rows):
        raise ValueError("duplicate origin keys")
    if not rows.select(
        pl.all_horizontal(pl.col(*BATCH_FEATURES).is_finite()).all()
    ).item():
        raise ValueError("nonfinite input features")
    complete = rows.drop_nulls(labels).sort("symbol", "decision_us")
    if not complete.select(pl.all_horizontal(pl.col(*labels).is_finite()).all()).item():
        raise ValueError("nonfinite targets")
    train = complete.filter(pl.col("decision_us") + horizon * MINUTE_US < train_end_us)
    selection = complete.filter(
        (pl.col("decision_us") >= train_end_us)
        & (pl.col("decision_us") + horizon * MINUTE_US < selection_end_us)
    )
    test = complete.filter(pl.col("decision_us") >= selection_end_us)
    if len(train) < 1000 or len(selection) < 100 or len(test) < 100:
        raise ValueError(
            "need at least 1000/100/100 complete train/selection/test origins"
        )
    groups = []
    features: list[str] = []
    for name, columns in FEATURE_GROUPS.items():
        features.extend(columns)
        groups.append((name, tuple(features)))
    # Training-only volatility bins; unseen symbols/bins fall back to fixed quantile.
    edges = np.unique(
        np.quantile(train["volatility_360"].to_numpy(), [0.25, 0.5, 0.75])
    )
    train_bins = np.searchsorted(edges, train["volatility_360"].to_numpy())
    chosen, responses = {}, []
    with threadpool_limits(limits=2):
        for label, q in coordinates:
            y_train = train[label].to_numpy()
            y_selection = selection[label].to_numpy()
            fixed = float(np.quantile(y_train, q))
            symbols = train["symbol"].to_numpy()
            symbol_q = {
                s: float(np.quantile(y_train[symbols == s], q))
                for s in sorted(set(symbols))
            }
            bin_q = {
                i: float(np.quantile(y_train[train_bins == i], q))
                for i in range(len(edges) + 1)
                if (train_bins == i).any()
            }
            attempts = []
            best_loss = float("inf")
            best_features = ()
            for group, columns in groups:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=max_iter,
                    max_leaf_nodes=15,
                    min_samples_leaf=50,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=0,
                ).fit(train.select(columns).to_numpy(), y_train)
                metrics = _scores(
                    y_selection, model.predict(selection.select(columns).to_numpy()), q
                )
                attempts.append(
                    {"through_group": group, "features": list(columns), **metrics}
                )
                if metrics["pinball"] < best_loss:
                    best_loss = metrics["pinball"]
                    chosen[label] = model
                    best_features = columns
            # Test rows are evaluated only after the selection winner is frozen.
            evaluations = {}
            for name, frame in (("selection", selection), ("test", test)):
                y = frame[label].to_numpy()
                pred = chosen[label].predict(frame.select(best_features).to_numpy())
                baselines = {
                    "fixed": np.full(len(frame), fixed),
                    "symbol": np.array(
                        [symbol_q.get(s, fixed) for s in frame["symbol"]]
                    ),
                    "volatility": np.array(
                        [
                            bin_q.get(int(i), fixed)
                            for i in np.searchsorted(
                                edges, frame["volatility_360"].to_numpy()
                            )
                        ]
                    ),
                }
                # Greedy nonoverlapping windows per symbol, not independent across symbols.
                mask = np.zeros(len(frame), dtype=bool)
                last: dict[str, int] = {}
                for i, (symbol, at) in enumerate(
                    frame.select("symbol", "decision_us").iter_rows()
                ):
                    if at > last.get(symbol, -(2**63)) + horizon * MINUTE_US:
                        mask[i] = True
                        last[symbol] = at
                symbols = frame["symbol"].to_numpy()
                evaluations[name] = {
                    **_comparison(y, pred, baselines, q),
                    "horizon_spaced": _comparison(
                        y[mask],
                        pred[mask],
                        {k: v[mask] for k, v in baselines.items()},
                        q,
                    ),
                    "by_symbol": {
                        s: _comparison(
                            y[symbols == s],
                            pred[symbols == s],
                            {k: v[symbols == s] for k, v in baselines.items()},
                            q,
                        )
                        for s in sorted(set(symbols))
                    },
                }
                if name == "test":
                    evaluations[name]["weekly_gain_vs_volatility"] = (
                        _weekly_gain_interval(
                            y,
                            pred,
                            baselines["volatility"],
                            frame["decision_us"].to_numpy(),
                            q,
                        )
                    )

            responses.append(
                {
                    "target": label,
                    "quantile": q,
                    "features": list(best_features),
                    "attempts": attempts,
                    "evaluation": evaluations,
                }
            )
    metadata = {
        "schema": "path-quantiles-v1",
        "sklearn_version": sklearn.__version__,
        "status": "experimental_selection_and_historical_test_not_live_validation",
        "horizon": horizon,
        "train_end_us": train_end_us,
        "selection_end_us": selection_end_us,
        "max_iter": max_iter,
        "rows": {
            "input": len(rows),
            "incomplete": len(rows) - len(complete),
            "train": len(train),
            "selection": len(selection),
            "test": len(test),
            "purged": len(complete) - len(train) - len(selection) - len(test),
        },
        "train_last_us": train["decision_us"].max(),
        "selection_last_us": selection["decision_us"].max(),
        "symbols": sorted(set(train["symbol"])),
        "responses": responses,
        "assessment": "Marginal quantiles are not joint path probabilities or expected trading returns.",
    }
    return chosen, metadata


def rolling_path_audit(
    dataset_path: Path,
    *,
    windows: tuple[tuple[int, int, int], ...],
    output: Path,
    horizon: int = 360,
    max_iter: int = 60,
) -> dict:
    """Cache up to eight expanding-window audits; test intervals may not overlap.

    Each triple is (training end, selection end/test start, test end), UTC us.
    Completed folds survive later failures. Cache hits verify data and report
    identities but never fit models. No new deployment artifact is selected.
    """
    from crypto_boom.research.path_dataset import load_path_dataset

    if not 1 <= len(windows) <= 8 or any(
        len(w) != 3
        or any(type(t) is not int or t % MINUTE_US for t in w)
        or not w[0] < w[1] < w[2]
        for w in windows
    ):
        raise ValueError("require 1-8 ordered minute-aligned window triples")
    if any(b[0] <= a[0] or b[1] < a[2] for a, b in pairwise(windows)):
        raise ValueError("training cuts must advance and test windows must not overlap")
    if (
        type(horizon) is not int
        or not 1 <= horizon <= 4320
        or type(max_iter) is not int
        or not 1 <= max_iter <= 100
    ):
        raise ValueError("invalid bounded screening configuration")
    rows, dataset = load_path_dataset(dataset_path)
    output.mkdir(parents=True, exist_ok=True)
    folds = []
    for train_end, selection_end, test_end in windows:
        identity: dict[str, object] = {
            "dataset_id": dataset["dataset_id"],
            "code": _artifacts.file_identity(Path(__file__))[0],
            "versions": [np.__version__, pl.__version__, sklearn.__version__],
            "window": [train_end, selection_end, test_end],
            "horizon": horizon,
            "max_iter": max_iter,
        }
        key = _artifacts.content_id(identity)
        path = output / (key.removeprefix("sha256:") + ".json")
        reused = path.exists()
        if reused:
            if path.stat().st_size > 4_000_000:
                raise ValueError("rolling report exceeds size budget")
            saved = json.loads(path.read_text(encoding="utf-8"))
            record = saved["record"]
            if (
                record["identity"] != identity
                or _artifacts.content_id(record) != saved["sha256"]
            ):
                raise ValueError("rolling report identity or hash mismatch")
        else:
            # Full label window must finish strictly before the test end.
            admitted = rows.filter(
                pl.col("decision_us") + horizon * MINUTE_US < test_end
            )
            _, report = screen_path_features(
                admitted,
                train_end_us=train_end,
                selection_end_us=selection_end,
                horizon=horizon,
                max_iter=max_iter,
            )
            report.update(
                test_end_us=test_end, test_last_us=admitted["decision_us"].max()
            )
            record: dict = {"identity": identity, "report": report}
            _artifacts.write_exclusive_bytes(
                path,
                _artifacts.canonical_json(
                    {"record": record, "sha256": _artifacts.content_id(record)}
                ),
            )
        folds.append({"path": str(path.resolve()), "reused": reused, **record})
    return {
        "schema": "rolling-path-audit-v1",
        "dataset_id": dataset["dataset_id"],
        "status": "retrospective_rolling_diagnostic_not_untouched_holdout",
        "folds": folds,
    }
