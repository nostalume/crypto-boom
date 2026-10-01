"""Verified, read-only admission of the frozen successor training matrix.

This loader is research-specific. It does not fetch market data, imply that
historical rows were available at their event time, or read reserved confirm
months. Source-only market acquisition remains with history/storage; this
derived design matrix is admitted only for a reproducibility rehearsal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_boom._artifacts import file_identity

MATRIX_RELATIVE = "data/universe-expansion-successor-20260928-v1/design-matrix.parquet"
RECEIPT_RELATIVE = "data/universe-expansion-successor-20260928-v1/design-matrix.json"
LAYER_RELATIVE = "data/universe-expansion-successor-20260928-v1/calibration-layer.json"
INSTANTS_RELATIVE = "data/universe-expansion-evaluator-20260928-v1/instants.json"
TRAIN_MONTHS = ("2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02")
FOLDS = (("2025-09", "2025-10", "2025-11"), ("2025-12", "2026-01"), ("2026-02",))
CELLS = ("primary", "secondary")
PENALTY = 10.0


class TrainingDataError(ValueError):
    """The frozen training source is missing, altered, or outside its scope."""


@dataclass(frozen=True, slots=True)
class FrozenTrainingBlock:
    frame: pl.DataFrame
    symbols: tuple[str, ...]
    added_columns: tuple[str, ...]
    folds: tuple[tuple[str, ...], ...]
    penalty: float
    cell: str
    matrix_sha256: str
    receipt_sha256: str
    layer_sha256: str
    instants_sha256: str


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingDataError(f"cannot read frozen source {path}") from error
    if not isinstance(value, dict):
        raise TrainingDataError(f"frozen source {path} is not a JSON object")
    return value


def _expect_sha(path: Path, expected: str) -> None:
    try:
        actual, _size = file_identity(path)
    except OSError as error:
        raise TrainingDataError(f"frozen source is unavailable: {path}") from error
    if actual != f"sha256:{expected}":
        raise TrainingDataError(f"frozen source digest differs: {path}")


def load_frozen_successor_training_block(root: Path, cell: str) -> FrozenTrainingBlock:
    """Verify pinned source bytes, then read only one cell's training months.

    The package-local M1 bundle owns the source digests and symbol code order;
    the frozen layer owns the added-column order, folds and penalty. The receipt
    and Parquet metadata must agree with those pinned bytes before any rows are
    read. This does not admit a prospective/as-of feature path.
    """
    if cell not in CELLS:
        raise TrainingDataError(f"unknown frozen successor cell: {cell}")
    root = root.resolve()
    bundle = json.loads(
        resources.files("crypto_boom.research")
        .joinpath("m1-v1.json")
        .read_text(encoding="utf-8")
    )
    sources = bundle["sources"]
    matrix_path, receipt_path = root / MATRIX_RELATIVE, root / RECEIPT_RELATIVE
    layer_path, instants_path = root / LAYER_RELATIVE, root / INSTANTS_RELATIVE
    for path, name in (
        (matrix_path, MATRIX_RELATIVE),
        (layer_path, LAYER_RELATIVE),
        (instants_path, INSTANTS_RELATIVE),
    ):
        _expect_sha(path, sources[name])
    layer = _json(layer_path)
    expected_receipt_sha = layer["design_matrix_receipt"]["sha256"]
    _expect_sha(receipt_path, expected_receipt_sha)
    receipt = _json(receipt_path)
    instants = _json(instants_path)
    estimator = layer["estimator"]
    symbols = tuple(bundle["symbols"])
    if (
        tuple(instants["symbols"]) != symbols
        or len(symbols) != len(set(symbols))
        or tuple(estimator["train_months"]) != TRAIN_MONTHS
        or tuple(tuple(fold) for fold in estimator["folds"]) != FOLDS
        or float(estimator["penalty"]) != PENALTY
        or layer["confirm_block_read"] is not False
        or receipt["confirm_block_read"] is not False
        or receipt["fits_performed"] != 0
        or receipt["matrix"]["sha256"] != sources[MATRIX_RELATIVE]
        or layer["matrix"]["sha256"] != sources[MATRIX_RELATIVE]
    ):
        raise TrainingDataError(
            "frozen training receipt, layer or symbol order differs"
        )
    added = tuple(estimator["added_columns"])
    if added != tuple(bundle["cells"][cell]["added_columns"]):
        raise TrainingDataError("frozen added-column order differs")
    splines = tuple(entry["name"] for entry in bundle["cells"][cell]["splines"])
    try:
        parquet = pq.ParquetFile(matrix_path)
        rows = parquet.metadata.num_rows
        columns = len(parquet.schema_arrow.names)
    except (OSError, ValueError, pa.ArrowException) as error:
        raise TrainingDataError("frozen matrix metadata is unreadable") from error
    if (
        rows != receipt["matrix"]["rows"]
        or rows != layer["matrix"]["rows"]
        or columns != receipt["matrix"]["columns"]
        or columns != layer["matrix"]["columns"]
    ):
        raise TrainingDataError("frozen matrix shape differs from its receipts")

    frame = (
        pl.scan_parquet(matrix_path)
        .filter((pl.col("cell") == cell) & pl.col("month").is_in(TRAIN_MONTHS))
        .select(
            "month",
            "minute",
            "symbol_code",
            "completed",
            "peak_minute",
            "size_quintile",
            *splines,
            *added,
        )
        .collect()
    )
    recorded = next((entry for entry in layer["cells"] if entry["cell"] == cell), None)
    if (
        recorded is None
        or frame.height != recorded["train_rows"]
        or set(frame["month"].unique().to_list()) != set(TRAIN_MONTHS)
        or not frame["symbol_code"].is_between(0, len(symbols) - 1).all()
    ):
        raise TrainingDataError("frozen training rows differ from the layer record")
    return FrozenTrainingBlock(
        frame=frame,
        symbols=symbols,
        added_columns=added,
        folds=FOLDS,
        penalty=PENALTY,
        cell=cell,
        matrix_sha256=sources[MATRIX_RELATIVE],
        receipt_sha256=expected_receipt_sha,
        layer_sha256=sources[LAYER_RELATIVE],
        instants_sha256=sources[INSTANTS_RELATIVE],
    )
