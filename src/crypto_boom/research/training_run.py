"""Rehearse the frozen successor fit and publish a separate research receipt.

This is not a replacement for the pinned historical fit, a confirm-block read,
or a deployable model. It accepts only the verified training matrix and writes
under an explicit new output root; it never edits the frozen study directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from crypto_boom import _artifacts
from crypto_boom.research.training import (
    SuccessorFit,
    cluster_ids,
    fit_successor,
    successor_design,
)
from crypto_boom.research.training_data import (
    FOLDS,
    INSTANTS_RELATIVE,
    LAYER_RELATIVE,
    MATRIX_RELATIVE,
    PENALTY,
    TRAIN_MONTHS,
    FrozenTrainingBlock,
    load_frozen_successor_training_block,
)

_SCHEMA_VERSION = 1
_MAX_RECEIPT_BYTES = 1_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TrainingRunError(RuntimeError):
    """A research training run or its immutable receipt is invalid."""


def _source_sha256(path: Path) -> str:
    digest, _size = _artifacts.file_identity(path)
    return digest.removeprefix("sha256:")


def _cluster_log_loss(
    frame: pl.DataFrame, prediction: np.ndarray
) -> tuple[float, float]:
    y = frame["completed"].to_numpy().astype(np.float64)
    identifier = cluster_ids(
        frame["minute"].to_numpy(), frame["peak_minute"].to_numpy()
    )
    sizes = np.bincount(identifier)
    rates = np.bincount(identifier, weights=y) / sizes
    baseline = float(np.mean(rates))

    def loss(probability: np.ndarray) -> float:
        p = np.clip(probability, 1e-12, 1.0 - 1e-12)
        row_loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(np.mean(np.bincount(identifier, weights=row_loss) / sizes))

    return loss(np.full(y.size, baseline)), loss(prediction)


def _finite(values: np.ndarray, name: str) -> list[float]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise TrainingRunError(f"{name} contains non-finite or ragged values")
    return [float(value) for value in values]


def _finite_json_vector(values: Any) -> bool:
    return isinstance(values, list) and all(
        type(value) in (int, float) and math.isfinite(value) for value in values
    )


def _validate_receipt_payload(payload: dict[str, Any]) -> None:
    """Reject unsupported or internally inconsistent research claims."""
    try:
        bundle = json.loads(
            resources.files("crypto_boom.research")
            .joinpath("m1-v1.json")
            .read_text(encoding="utf-8")
        )
        cell = payload["cell"]
        source = payload["source"]
        selection = payload["selection"]
        feedback = payload["diagnostic_feedback"]
        beta = payload["coefficients"]["beta"]
        frailty = payload["coefficients"]["frailty"]
        thresholds = payload["calibration_layer"]["thresholds"]
        levels = payload["calibration_layer"]["levels"]
        symbols = payload["symbols"]
        names = payload["parameter_names"]
        added = payload["added_columns"]
        hashes = (
            source["matrix_sha256"],
            source["design_receipt_sha256"],
            source["layer_sha256"],
            source["instants_sha256"],
            source["training_code_sha256"],
            source["training_loader_sha256"],
            feedback["oof_float64_le_sha256"],
        )
        if not all(
            isinstance(value, str) and _SHA256.fullmatch(value) for value in hashes
        ):
            raise ValueError("invalid source or out-of-fold digest")
        pinned = bundle["sources"]
        if (
            source["matrix_sha256"] != pinned[MATRIX_RELATIVE]
            or source["layer_sha256"] != pinned[LAYER_RELATIVE]
            or source["instants_sha256"] != pinned[INSTANTS_RELATIVE]
            or symbols != bundle["symbols"]
            or added != bundle["cells"][cell]["added_columns"]
        ):
            raise ValueError(
                "source or fitted code space differs from the frozen bundle"
            )
        if (
            payload["schema_version"] != _SCHEMA_VERSION
            or payload["kind"] != "successor_m1_research_training_replay"
            or payload["evidence_class"] != "internal_diagnostic_not_confirmed"
            or payload["gates_evaluated"] != 0
            or payload["confirm_block_read"] is not False
            or payload["fits_performed"] != len(FOLDS) + 1
            or payload["folds"] != [list(fold) for fold in FOLDS]
            or payload["penalty"] != PENALTY
            or selection["months"] != list(TRAIN_MONTHS)
            or type(selection["rows"]) is not int
            or selection["rows"] <= 0
            or selection["kind"] not in ("bounded_rehearsal", "full_training_block")
        ):
            raise ValueError("unsupported research receipt claim")
        limit = selection["rows_per_month_limit"]
        if selection["kind"] == "bounded_rehearsal":
            if (
                type(limit) is not int
                or limit <= 0
                or selection["rows"] > len(TRAIN_MONTHS) * limit
            ):
                raise ValueError("invalid bounded sample declaration")
        elif limit is not None:
            raise ValueError("full fit cannot carry a sampling limit")
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
            or not _finite_json_vector(beta)
            or len(names) != len(beta)
            or not _finite_json_vector(frailty)
            or len(frailty) != len(symbols)
            or not _finite_json_vector(thresholds)
            or not _finite_json_vector(levels)
            or len(thresholds) < 2
            or len(thresholds) != len(levels)
            or any(left >= right for left, right in pairwise(thresholds))
            or any(left > right for left, right in pairwise(levels))
            or not all(0.0 <= value <= 1.0 for value in levels)
            or feedback["oof_rows"] != selection["rows"]
            or not all(
                type(feedback[key]) in (int, float) and math.isfinite(feedback[key])
                for key in (
                    "cluster_log_loss_baseline",
                    "cluster_log_loss_oof",
                    "effective_parameters",
                )
            )
        ):
            raise ValueError("training parameters or feedback are inconsistent")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise TrainingRunError(
            "training receipt has invalid research semantics"
        ) from error


def _payload(
    *,
    block: FrozenTrainingBlock,
    fit: SuccessorFit,
    chosen: pl.DataFrame,
    sample_per_month: int | None,
) -> dict[str, Any]:
    ordered = chosen.sort(["minute", "symbol_code"])
    baseline_loss, oof_loss = _cluster_log_loss(ordered, fit.oof_probability)
    if not np.isfinite(fit.effective_parameters):
        raise TrainingRunError("effective parameter count is non-finite")
    if len(block.symbols) != fit.frailty.size:
        raise TrainingRunError("frailty vector does not match the frozen symbol order")
    if fit.column_names[:1] != ("intercept",):
        raise TrainingRunError("the fitted design does not begin with an intercept")
    if len(set(fit.column_names)) != len(fit.column_names):
        raise TrainingRunError("the fitted design has duplicate columns")
    oof_bytes = np.asarray(fit.oof_probability, dtype="<f8").tobytes()
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "successor_m1_research_training_replay",
        "evidence_class": "internal_diagnostic_not_confirmed",
        "cell": block.cell,
        "source": {
            "matrix_sha256": block.matrix_sha256,
            "design_receipt_sha256": block.receipt_sha256,
            "layer_sha256": block.layer_sha256,
            "instants_sha256": block.instants_sha256,
            "training_code_sha256": _source_sha256(
                Path(__file__).with_name("training.py")
            ),
            "training_loader_sha256": _source_sha256(
                Path(__file__).with_name("training_data.py")
            ),
        },
        "selection": {
            "kind": "full_training_block"
            if sample_per_month is None
            else "bounded_rehearsal",
            "rows_per_month_limit": sample_per_month,
            "months": list(fit.train_months),
            "rows": fit.train_rows,
        },
        "fits_performed": len(fit.folds) + 1,
        "gates_evaluated": 0,
        "confirm_block_read": False,
        "symbols": list(block.symbols),
        "added_columns": list(fit.added_columns),
        "folds": [list(fold) for fold in fit.folds],
        "penalty": fit.penalty,
        "parameter_names": list(fit.column_names),
        "imputation_medians": fit.imputation_medians,
        "imputed_rows": fit.imputed_rows,
        "coefficients": {
            "beta": _finite(fit.beta, "beta"),
            "frailty": _finite(fit.frailty, "frailty"),
        },
        "calibration_layer": {
            "thresholds": _finite(fit.layer_thresholds, "thresholds"),
            "levels": _finite(fit.layer_levels, "levels"),
        },
        "diagnostic_feedback": {
            "oof_rows": int(fit.oof_probability.size),
            "oof_float64_le_sha256": hashlib.sha256(oof_bytes).hexdigest(),
            "cluster_log_loss_baseline": baseline_loss,
            "cluster_log_loss_oof": oof_loss,
            "effective_parameters": fit.effective_parameters,
            "note": "internal training diagnostic, not a gate or predictive-value claim",
        },
    }


def load_training_receipt(path: Path) -> dict[str, Any]:
    """Strictly verify a content-addressed research training receipt."""
    try:
        raw = (path / "manifest.json").read_bytes()
        if len(raw) > _MAX_RECEIPT_BYTES:
            raise TrainingRunError("training receipt exceeds its byte limit")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or raw != _artifacts.canonical_json(payload):
            raise TrainingRunError("training receipt is not canonical JSON")
        expected = _artifacts.content_id(payload).removeprefix("sha256:")
        if path.name != expected:
            raise TrainingRunError("training receipt path does not match its content")
        if sorted(child.name for child in path.iterdir()) != ["manifest.json"]:
            raise TrainingRunError("training receipt directory has unexpected content")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise TrainingRunError("training receipt is unreadable or invalid") from error
    _validate_receipt_payload(payload)
    return payload


def run_verified_training(
    root: Path,
    cell: str,
    output_root: Path,
    *,
    rows_per_month: int | None = None,
    allow_full: bool = False,
) -> Path:
    """Fit verified training rows and atomically publish a separate receipt.

    Full refitting is costly and requires ``allow_full=True``. A bounded run
    selects the earliest declared rows of each training month, records that
    selection, and is never eligible to replace the frozen model.
    """
    if rows_per_month is None:
        if not allow_full:
            raise TrainingRunError("full fit requires explicit allow_full=True")
    elif type(rows_per_month) is not int or rows_per_month <= 0 or allow_full:
        raise TrainingRunError(
            "choose one positive rehearsal limit or an explicit full fit"
        )
    root, output_root = root.resolve(), output_root.resolve()
    protected = (
        root / "data/universe-expansion-successor-20260928-v1",
        root / "data/universe-expansion-evaluator-20260928-v1",
    )
    if any(output_root.is_relative_to(path) for path in protected):
        raise TrainingRunError(
            "output root must not be inside a frozen study directory"
        )
    block = load_frozen_successor_training_block(root, cell)
    chosen = block.frame
    if rows_per_month is not None:
        chosen = (
            chosen.sort(["month", "minute", "symbol_code"])
            .group_by("month", maintain_order=True)
            .head(rows_per_month)
        )
    fit = fit_successor(
        chosen,
        added_columns=list(block.added_columns),
        symbol_count=len(block.symbols),
        folds=block.folds,
        penalty=block.penalty,
    )
    payload = _payload(
        block=block,
        fit=fit,
        chosen=chosen,
        sample_per_month=rows_per_month,
    )
    manifest = _artifacts.canonical_json(payload)
    if len(manifest) > _MAX_RECEIPT_BYTES:
        raise TrainingRunError("training receipt exceeds its byte limit")
    target = output_root / _artifacts.content_id(payload).removeprefix("sha256:")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with _artifacts.publication_staging_directory(
            output_root, prefix=".training-staging-"
        ) as staging:
            _artifacts.write_exclusive_bytes(staging / "manifest.json", manifest)

            def verify_existing(existing: Path) -> None:
                if load_training_receipt(existing) != payload:
                    raise TrainingRunError("existing training receipt conflicts")

            _artifacts.adopt_directory(staging, target, verify_existing=verify_existing)
    except TrainingRunError:
        raise
    except OSError as error:
        raise TrainingRunError("training receipt publication failed") from error
    load_training_receipt(target)
    return target


def compare_full_replay(
    root: Path, receipt_path: Path
) -> dict[str, float | int | bool | str]:
    """Compare full-fit scores/layer to pinned frozen artifacts on train rows.

    Coefficients can move along redundant design directions. The observable
    arithmetic claim is row-score and layer parity, not coefficient-bit parity.
    This is a retrospective training check, not an as-of or confirm verdict.
    """
    receipt = load_training_receipt(receipt_path)
    if receipt["selection"]["kind"] != "full_training_block":
        raise TrainingRunError("bounded rehearsals cannot claim full-fit parity")
    block = load_frozen_successor_training_block(root, receipt["cell"])
    if receipt["source"]["matrix_sha256"] != block.matrix_sha256:
        raise TrainingRunError("replay source does not match the admitted matrix")
    model_path = (
        root.resolve()
        / "data/universe-expansion-successor-20260928-v1/successor-model.json"
    )
    bundle = json.loads(
        resources.files("crypto_boom.research")
        .joinpath("m1-v1.json")
        .read_text(encoding="utf-8")
    )
    model_relative = (
        "data/universe-expansion-successor-20260928-v1/successor-model.json"
    )
    try:
        model_digest = _source_sha256(model_path)
    except OSError as error:
        raise TrainingRunError("frozen successor model is unavailable") from error
    if model_digest != bundle["sources"][model_relative]:
        raise TrainingRunError("frozen successor model digest differs")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    layer_path = root.resolve() / LAYER_RELATIVE
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    cell = receipt["cell"]
    frozen = next(entry for entry in model["cells"] if entry["cell"] == cell)
    frozen_layer = next(
        entry["layer"] for entry in layer["cells"] if entry["cell"] == cell
    )

    frame = block.frame.sort(["minute", "symbol_code"])
    design, names, _medians, _imputed = successor_design(
        frame, frame, list(block.added_columns)
    )
    if names != receipt["parameter_names"] or names != frozen["parameter_names"]:
        raise TrainingRunError("replay design columns differ from frozen parameters")
    symbol = frame["symbol_code"].to_numpy().astype(np.int64)
    new_beta = np.asarray(receipt["coefficients"]["beta"])
    old_beta = np.asarray(frozen["coefficients"]["beta"])
    new_frailty = np.asarray(receipt["coefficients"]["frailty"])
    old_frailty = np.asarray(frozen["coefficients"]["frailty"])
    with threadpool_limits(limits=4):
        new_eta = design @ new_beta + new_frailty[symbol]
        old_eta = design @ old_beta + old_frailty[symbol]
    new_raw = 1.0 / (1.0 + np.exp(-np.clip(new_eta, -35.0, 35.0)))
    old_raw = 1.0 / (1.0 + np.exp(-np.clip(old_eta, -35.0, 35.0)))
    new_x = np.asarray(receipt["calibration_layer"]["thresholds"])
    old_x = np.asarray(frozen_layer["thresholds"])
    new_y = np.asarray(receipt["calibration_layer"]["levels"])
    old_y = np.asarray(frozen_layer["levels"])
    if new_x.shape != old_x.shape or new_y.shape != old_y.shape:
        raise TrainingRunError("replay layer shape differs from frozen layer")
    raw_max = float(np.max(np.abs(new_raw - old_raw)))
    x_max = float(np.max(np.abs(new_x - old_x)))
    y_max = float(np.max(np.abs(new_y - old_y)))
    tolerance = 1e-9
    return {
        "cell": cell,
        "rows": frame.height,
        "max_abs_beta": float(np.max(np.abs(new_beta - old_beta))),
        "max_abs_frailty": float(np.max(np.abs(new_frailty - old_frailty))),
        "max_abs_raw_score": raw_max,
        "max_abs_layer_threshold": x_max,
        "max_abs_layer_level": y_max,
        "diagnostic_tolerance": tolerance,
        "score_and_layer_parity": bool(max(raw_max, x_max, y_max) <= tolerance),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", choices=("primary", "secondary"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--rows-per-month", type=int)
    choice.add_argument("--full", action="store_true")
    arguments = parser.parse_args(argv)
    target = run_verified_training(
        arguments.root,
        arguments.cell,
        arguments.output_root,
        rows_per_month=arguments.rows_per_month,
        allow_full=arguments.full,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
