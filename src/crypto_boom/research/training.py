"""Research-only M1 design and weighted-logistic fitting arithmetic.

This is a migrated numerical kernel, not the frozen 2026-09-28 fit runner and
not an authorization to refit its model or write to its receipt directory.
The historical runner remains pinned for its November confirm read. New study
runners may use this module only with explicit admitted data, penalty, symbol
code space and a separate versioned output contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import BSpline

DAY_MINUTES = 1440
HARMONICS = 3
QUINTILE_DUMMIES = 4
SPLINE_DF = {
    "comove_share": 3,
    "hours_since_completion": 4,
    "log_range24": 3,
    "breadth": 3,
}
IRLS_ITERATIONS = 40
IRLS_TOLERANCE = 1e-9


class TrainingInputError(ValueError):
    """The admitted numerical training block is malformed."""


@dataclass(frozen=True, slots=True)
class SuccessorFit:
    """In-memory fit; not a published or validated model artifact."""

    beta: np.ndarray
    frailty: np.ndarray
    column_names: tuple[str, ...]
    imputation_medians: dict[str, float]
    imputed_rows: dict[str, int]
    oof_probability: np.ndarray
    layer_thresholds: np.ndarray
    layer_levels: np.ndarray
    effective_parameters: float
    train_months: tuple[str, ...]
    folds: tuple[tuple[str, ...], ...]
    penalty: float
    train_rows: int
    added_columns: tuple[str, ...]


def cluster_ids(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Mirror of `episode_co_movement.clusters_for`: sorted by start, new cluster iff start > end.

    An empty set of intervals has zero clusters, not an error: a block with no training month
    reaches here with empty arrays, and `end[order[0]]` would raise `IndexError`.
    """
    order = np.argsort(start, kind="stable")
    identifier = np.empty(start.size, dtype=np.int64)
    if start.size == 0:
        return identifier
    current = 0
    running_end = int(end[order[0]])
    identifier[order[0]] = 0
    for position in range(1, order.size):
        index = order[position]
        if start[index] > running_end:
            current += 1
            running_end = int(end[index])
        else:
            running_end = max(running_end, int(end[index]))
        identifier[index] = current
    return identifier


def bspline(values: np.ndarray, n_basis: int, reference: np.ndarray) -> np.ndarray:
    """B-spline design matrix with knots at `reference` quantiles; `n_basis` columns exactly.

    The degree is reduced for small bases: a cubic B-spline needs at least `degree + 1 = 4` basis
    functions, so asking for 3 with `degree = 3` silently yields 4 columns and an inflated parameter
    count. The trailing assertion makes that failure loud instead of silent.
    """
    degree = min(3, n_basis - 1)
    lo = float(np.min(reference))
    hi = float(np.max(reference))
    interior_count = n_basis - degree - 1
    interior = (
        np.quantile(reference, np.linspace(0.0, 1.0, interior_count + 2)[1:-1])
        if interior_count > 0
        else np.zeros(0)
    )
    knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    clipped = np.clip(values, lo, hi)
    block = np.asarray(
        BSpline.design_matrix(clipped, knots, degree).todense(), dtype=np.float64
    )
    assert block.shape[1] == n_basis, (
        f"bspline wanted {n_basis} columns, built {block.shape[1]}"
    )
    return block


def design(
    frame: pl.DataFrame, reference: pl.DataFrame | None
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """The 24 declared fixed-effect columns: 1 + 3 + 4 + 4 + 3 + 3 + 6."""
    source = frame if reference is None else reference
    columns: list[np.ndarray] = [np.ones((frame.height, 1), dtype=np.float64)]
    names: list[str] = ["intercept"]
    imputation: dict[str, float] = {}
    for feature, n_basis in SPLINE_DF.items():
        values = frame[feature].to_numpy().astype(np.float64)
        train_values = source[feature].to_numpy().astype(np.float64)
        median = float(np.nanmedian(train_values))
        if not np.isfinite(median):
            raise TrainingInputError(
                f"base feature {feature!r} has no finite training reference"
            )
        imputation[feature] = median
        values = np.where(np.isfinite(values), values, median)
        train_clean = np.where(np.isfinite(train_values), train_values, median)
        block = bspline(values, n_basis, train_clean)
        columns.append(block)
        names.extend(f"{feature}[{i}]" for i in range(n_basis))
    quintile = frame["size_quintile"].to_numpy().astype(np.int64)
    dummies = np.zeros((frame.height, QUINTILE_DUMMIES), dtype=np.float64)
    for level in range(2, 6):
        dummies[:, level - 2] = (quintile == level).astype(np.float64)
    columns.append(dummies)
    names.extend(f"quintile[{level}]" for level in range(2, 6))
    angle = 2.0 * np.pi * (frame["minute"].to_numpy() % DAY_MINUTES) / DAY_MINUTES
    harmonics = np.zeros((frame.height, 2 * HARMONICS), dtype=np.float64)
    for harmonic in range(1, HARMONICS + 1):
        harmonics[:, 2 * (harmonic - 1)] = np.sin(harmonic * angle)
        harmonics[:, 2 * (harmonic - 1) + 1] = np.cos(harmonic * angle)
    columns.append(harmonics)
    names.extend(
        f"tod[{'sin' if i % 2 == 0 else 'cos'}{i // 2 + 1}]"
        for i in range(2 * HARMONICS)
    )
    matrix = np.concatenate(columns, axis=1)
    return matrix, names, imputation


def successor_design(
    frame: pl.DataFrame, reference: pl.DataFrame, added: list[str]
) -> tuple[np.ndarray, list[str], dict[str, float], dict[str, int]]:
    """Append the declared successor columns to the frozen base M1 basis.

    Each non-finite added value is replaced by its training-reference median.
    The caller owns the fixed column order and the training-reference block;
    this function neither discovers features nor reads a study artifact.
    """
    basis, names, imputation = design(frame, reference)
    blocks = [basis]
    columns = list(names)
    imputed: dict[str, int] = {}
    if len(set(added)) != len(added) or any(name in columns for name in added):
        raise TrainingInputError("successor columns must be distinct from the basis")
    for name in added:
        values = frame[name].to_numpy().astype(np.float64)
        median = float(np.nanmedian(reference[name].to_numpy().astype(np.float64)))
        if not np.isfinite(median):
            raise TrainingInputError(
                f"added column {name!r} is non-finite on the whole reference block"
            )
        imputation[name] = median
        missing = int((~np.isfinite(values)).sum())
        if missing:
            values = np.where(np.isfinite(values), values, median)
            imputed[name] = missing
        blocks.append(values[:, None])
        columns.append(name)
    return np.concatenate(blocks, axis=1), columns, imputation, imputed


def cluster_weights(identifier: np.ndarray) -> np.ndarray:
    if (
        identifier.ndim != 1
        or identifier.size == 0
        or not np.issubdtype(identifier.dtype, np.integer)
        or np.any(identifier < 0)
    ):
        raise TrainingInputError(
            "cluster identifiers must be non-empty non-negative integers"
        )
    sizes = np.bincount(identifier, minlength=int(identifier.max()) + 1).astype(
        np.float64
    )
    return 1.0 / sizes[identifier]


BLAS_THREADS = 4
"""Thread ceiling for the design-matrix products inside `fit_weighted_logistic`.

Measured on the primary training block: the same arithmetic costs 0.08 s per IRLS
iteration at 4 threads and 11.20 s at the 20 threads OpenBLAS chooses by default
on this machine - a factor of 103. The products here are 47 wide, too narrow to
give 20 threads useful work, and OpenBLAS threads spin-wait rather than yield, so
the pool spends its time contending with every other process on the box. Four
threads is the fastest of the three counts measured and leaves the machine
usable for concurrent work.
"""


def _group_operator(symbol: np.ndarray, symbol_count: int):
    """A sparse `(symbol_count, rows)` summing operator.

    Summing a row vector by symbol is `group @ vector`; summing every column of a
    `(rows, width)` block is `group @ block`. Both are one pass over the nonzeros.
    Scipy canonicalises the index order, so within a symbol the accumulation is in
    the original row order - the same sum `np.bincount` returns, at one pass
    instead of one pass for each of the `width` columns.
    """
    from scipy.sparse import csr_matrix

    rows = symbol.size
    return csr_matrix(
        (np.ones(rows, dtype=np.float64), (symbol, np.arange(rows))),
        shape=(int(symbol_count), rows),
    )


def _chunked_cross(
    x: np.ndarray,
    scaled: np.ndarray,
    out: np.ndarray,
    rows_per_chunk: int = 131_072,
) -> np.ndarray:
    """`x.T @ scaled`, accumulated in row chunks into `out`.

    The unbounded form allocates one full `(rows, width)` temporary per call and
    hands a 47-wide product to a BLAS thread pool sized for much larger work. On a
    shared machine that is where this fit spent its time. Chunking bounds the
    working set to `rows_per_chunk x width` whatever the block size, and gives the
    cache a block that fits in it.
    """
    out[:] = 0.0
    for start in range(0, x.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, x.shape[0])
        out += x[start:stop].T @ scaled[start:stop]
    return out


def fit_weighted_logistic(
    x: np.ndarray,
    symbol: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    penalty: float,
    symbol_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """IRLS for a logistic model with fixed effects plus a penalised per-symbol intercept.

    Numerically this is the same iteration it has always been: the same working
    response, the same curvature, the same penalised normal equations, the same
    convergence test on the largest absolute step. What changed is only how the
    normal equations are assembled - the scaled design is built once per iteration
    instead of `width + 1` times, the per-symbol sums are one sparse product
    instead of `width` separate `np.bincount` passes, the cross-product is chunked,
    and every buffer is allocated once outside the loop.
    """
    if (
        x.ndim != 2
        or x.shape[0] == 0
        or symbol.ndim != 1
        or y.ndim != 1
        or weight.ndim != 1
        or x.shape[0] != symbol.size
        or x.shape[0] != y.size
        or x.shape[0] != weight.size
        or not np.issubdtype(symbol.dtype, np.integer)
        or symbol_count <= 0
        or np.any(symbol < 0)
        or np.any(symbol >= symbol_count)
        or not np.isfinite(penalty)
        or penalty <= 0
        or not np.isfinite(x).all()
        or not np.isin(y, (0.0, 1.0)).all()
        or not np.isfinite(weight).all()
        or np.any(weight <= 0)
    ):
        raise TrainingInputError("invalid weighted-logistic training block")

    from threadpoolctl import threadpool_limits

    _rows, width = x.shape
    symbol_count = int(symbol_count)
    beta = np.zeros(width, dtype=np.float64)
    frail = np.zeros(symbol_count, dtype=np.float64)
    total = width + symbol_count
    group = _group_operator(symbol, symbol_count)
    scaled = np.empty_like(x)
    top = np.empty((width, width), dtype=np.float64)
    cross = np.empty((width, symbol_count), dtype=np.float64)
    block = np.empty((total, total), dtype=np.float64)
    right = np.empty(total, dtype=np.float64)
    with threadpool_limits(limits=BLAS_THREADS):
        for _ in range(IRLS_ITERATIONS):
            eta = x @ beta + frail[symbol]
            probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
            variance = np.maximum(probability * (1.0 - probability), 1e-10)
            curvature = weight * variance
            working = eta + (y - probability) / variance
            np.multiply(x, curvature[:, None], out=scaled)
            _chunked_cross(x, scaled, top)
            np.copyto(cross, np.asarray(group @ scaled).T)
            block[:width, :width] = top
            block[:width, width:] = cross
            block[width:, :width] = cross.T
            block[width:, width:] = np.diag(
                np.asarray(group @ curvature).ravel() + penalty
            )
            right[:width] = x.T @ (curvature * working)
            right[width:] = group @ (curvature * working)
            try:
                step = np.linalg.solve(block + 1e-9 * np.eye(total), right)
            except np.linalg.LinAlgError:
                break
            new_beta = step[:width]
            new_frail = step[width:]
            shift = max(
                float(np.max(np.abs(new_beta - beta))),
                float(np.max(np.abs(new_frail - frail))),
            )
            beta, frail = new_beta, new_frail
            if shift < IRLS_TOLERANCE:
                break
        eta = x @ beta + frail[symbol]
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        variance = np.maximum(probability * (1.0 - probability), 1e-10)
        curvature = weight * variance
        np.multiply(x, curvature[:, None], out=scaled)
        _chunked_cross(x, scaled, top)
        np.copyto(cross, np.asarray(group @ scaled).T)
        unpenalised = np.empty((total, total), dtype=np.float64)
        unpenalised[:width, :width] = top
        unpenalised[:width, width:] = cross
        unpenalised[width:, :width] = cross.T
        unpenalised[width:, width:] = np.diag(np.asarray(group @ curvature).ravel())
        penalised = unpenalised.copy()
        penalised[width:, width:] += np.eye(symbol_count) * penalty
        effective = float(
            np.trace(np.linalg.solve(penalised + 1e-9 * np.eye(total), unpenalised))
        )
    return beta, frail, {"effective_parameters": effective}


def fit_successor(
    frame: pl.DataFrame,
    *,
    added_columns: list[str],
    symbol_count: int,
    folds: tuple[tuple[str, ...], ...],
    penalty: float,
) -> SuccessorFit:
    """Fit one successor cell on an explicitly supplied training block.

    Each month must belong to exactly one held-out fold. The fixed basis is
    built once from the full training block, matching the frozen construction;
    every row's calibration input comes from a fit that held out its fold.
    This function performs no input discovery, receipt publication or gate
    evaluation. Its caller must enforce study identity, the permitted month
    block, and the reserved-month guard before supplying rows.
    """
    if frame.is_empty() or not folds or not all(folds):
        raise TrainingInputError("training block and held-out folds must be non-empty")
    declared = [month for fold in folds for month in fold]
    if len(set(declared)) != len(declared):
        raise TrainingInputError("a month belongs to more than one held-out fold")
    months_present = set(frame["month"].to_list())
    if months_present != set(declared):
        raise TrainingInputError("training months differ from the declared folds")

    from sklearn.isotonic import IsotonicRegression

    ordered = frame.sort(["minute", "symbol_code"])
    x, names, medians, imputed = successor_design(ordered, ordered, added_columns)
    symbol_values = ordered["symbol_code"].to_numpy()
    if (
        not np.issubdtype(symbol_values.dtype, np.integer)
        or np.any(symbol_values < 0)
        or np.any(symbol_values >= symbol_count)
    ):
        raise TrainingInputError("training symbols exceed the admitted code space")
    symbol = symbol_values.astype(np.int64)
    y = ordered["completed"].to_numpy().astype(np.float64)
    minute = ordered["minute"].to_numpy()
    peak = ordered["peak_minute"].to_numpy()
    if (
        not np.isin(y, (0.0, 1.0)).all()
        or not np.issubdtype(minute.dtype, np.integer)
        or not np.issubdtype(peak.dtype, np.integer)
    ):
        raise TrainingInputError("training labels and interval times are malformed")
    cluster = cluster_ids(minute, peak)
    weight = cluster_weights(cluster)
    oof = np.full(ordered.height, np.nan, dtype=np.float64)
    month_series = ordered["month"]
    for fold in folds:
        holdout = month_series.is_in(fold).to_numpy()
        fit = ~holdout
        if not holdout.any() or not fit.any():
            raise TrainingInputError(f"fold {fold!r} does not split the training block")
        fit_cluster = cluster_ids(minute[fit], peak[fit])
        beta, frail, _info = fit_weighted_logistic(
            x[fit],
            symbol[fit],
            y[fit],
            cluster_weights(fit_cluster),
            penalty,
            symbol_count,
        )
        eta = x[holdout] @ beta + frail[symbol[holdout]]
        oof[holdout] = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
    if not np.isfinite(oof).all():
        raise TrainingInputError("some training rows have no out-of-fold prediction")

    beta, frail, info = fit_weighted_logistic(
        x, symbol, y, weight, penalty, symbol_count
    )
    layer = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    layer.fit(oof, y, sample_weight=weight)
    thresholds = np.asarray(layer.X_thresholds_, dtype=np.float64)
    levels = np.asarray(layer.y_thresholds_, dtype=np.float64)
    probe = np.linspace(0.0, 1.0, 1001)
    if thresholds.size != levels.size or not np.array_equal(
        np.interp(probe, thresholds, levels), layer.predict(probe)
    ):
        raise TrainingInputError(
            "isotonic knots cannot exactly replay the fitted layer"
        )
    return SuccessorFit(
        beta=beta,
        frailty=frail,
        column_names=tuple(names),
        imputation_medians=medians,
        imputed_rows=imputed,
        oof_probability=oof,
        layer_thresholds=thresholds,
        layer_levels=levels,
        effective_parameters=info["effective_parameters"],
        train_months=tuple(sorted(months_present)),
        folds=folds,
        penalty=penalty,
        train_rows=ordered.height,
        added_columns=tuple(added_columns),
    )
