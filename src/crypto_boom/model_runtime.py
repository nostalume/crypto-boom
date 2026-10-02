"""Cadence-neutral numeric model bundles and explicit local activation."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import warnings
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US
from crypto_boom.features import SequenceRecipe, sequence_matrix


def _recipe(record: dict) -> SequenceRecipe:
    return SequenceRecipe(
        record["history_minutes"],
        record["step_minutes"],
        tuple(record["return_steps"]),
        tuple(record["volatility_steps"]),
        record["activity_reference_steps"],
        record["buy_steps"],
        record["observed_steps"],
    )


def _validate(models: dict, identity: dict) -> SequenceRecipe:
    if (
        identity.get("schema") != "numeric-sequence-v1"
        or identity.get("sklearn_version") != sklearn.__version__
    ):
        raise ValueError("incompatible bundle schema or sklearn version")
    recipe = _recipe(identity["recipe"])
    cadence = identity["decision_step_minutes"]
    if type(cadence) is not int or not 1 <= cadence <= 1440:
        raise ValueError("invalid model decision cadence")
    outputs = identity["outputs"]
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= 8:
        raise ValueError("require 1..8 numeric outputs")
    names = [o["name"] for o in outputs]
    if (
        len(set(names)) != len(names)
        or set(models) != set(names)
        or identity["rank_by"] not in names
    ):
        raise ValueError("model/output identities differ")
    for output in outputs:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", output["name"])
            or not isinstance(output["label"], str)
            or not 1 <= len(output["label"]) <= 100
            or output["unit"] not in ("return_fraction", "fraction", "minutes")
            or type(output["horizon_minutes"]) is not int
            or not 1 <= output["horizon_minutes"] <= 4320
            or output["statistic"] not in ("mean", "quantile")
            or (
                output["statistic"] == "quantile"
                and not 0 < output.get("quantile", 0) < 1
            )
        ):
            raise ValueError("invalid output contract")
        model = models[output["name"]]
        if (
            not callable(getattr(model, "predict", None))
            or getattr(model, "n_features_in_", None) != recipe.feature_count
        ):
            raise ValueError("estimator/feature shape mismatch")
    return recipe


def publish_model(
    registry: Path,
    models: dict,
    *,
    recipe: SequenceRecipe,
    decision_step_minutes: int,
    outputs: list[dict],
    rank_by: str,
    provenance: dict,
) -> dict:
    """Research publishes immutable weights + contract; publication does not activate."""
    identity: dict[str, object] = {
        "schema": "numeric-sequence-v1",
        "sklearn_version": sklearn.__version__,
        "recipe": asdict(recipe),
        "decision_step_minutes": decision_step_minutes,
        "outputs": outputs,
        "rank_by": rank_by,
        "provenance": provenance,
        "evidence_status": "experimental_not_trade_signal",
    }
    _validate(models, identity)
    registry.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        registry, prefix="bundle-"
    ) as staging:
        joblib.dump(models, staging / "weights.joblib", compress=3)
        identity["weights_sha256"] = _artifacts.file_identity(
            staging / "weights.joblib"
        )[0]
        model_id = _artifacts.content_id(identity).removeprefix("sha256:")
        manifest = {"model_id": model_id, "identity": identity}
        (staging / "manifest.json").write_bytes(_artifacts.canonical_json(manifest))
        target = registry / model_id
        if target.exists():
            load_model(registry, model_id, trusted=True)
        else:
            staging.rename(target)
    logging.getLogger(__name__).info(
        "Model published: %s (activation unchanged)", model_id
    )
    return manifest


def load_model(
    registry: Path, model_id: str, *, trusted: bool = False
) -> tuple[dict, dict, SequenceRecipe]:
    if not trusted:
        raise ValueError("explicit local trust required: model files can execute code")
    if not re.fullmatch(r"[0-9a-f]{64}", model_id):
        raise ValueError("invalid model ID")
    directory = registry / model_id
    manifest, weights = directory / "manifest.json", directory / "weights.joblib"
    if manifest.stat().st_size > 2_000_000 or weights.stat().st_size > 100_000_000:
        raise ValueError("model bundle exceeds byte budget")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    identity = record["identity"]
    if (
        record["model_id"] != model_id
        or _artifacts.content_id(identity) != "sha256:" + model_id
    ):
        raise ValueError("model manifest identity mismatch")
    if identity.get("sklearn_version") != sklearn.__version__:
        raise ValueError("incompatible sklearn version")
    payload = weights.read_bytes()
    if "sha256:" + hashlib.sha256(payload).hexdigest() != identity["weights_sha256"]:
        raise ValueError("model weights hash mismatch")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        models = joblib.load(io.BytesIO(payload))
    if not isinstance(models, dict):
        raise ValueError("invalid model collection")
    return models, record, _validate(models, identity)


def activate_model(registry: Path, model_id: str, *, trusted: bool = False) -> None:
    load_model(registry, model_id, trusted=trusted)
    temporary = registry / f"active-{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(
            _artifacts.canonical_json({"model_id": model_id, "trusted": True})
        )
        os.replace(temporary, registry / "active.json")
        logging.getLogger(__name__).info("Model activated: %s", model_id)
    finally:
        temporary.unlink(missing_ok=True)


def load_active_model(registry: Path) -> tuple[dict, dict, SequenceRecipe]:
    path = registry / "active.json"
    if not path.exists():
        raise ValueError(
            "no active model; publish from research, then use crypto-boom model activate"
        )
    if path.stat().st_size > 4096:
        raise ValueError("invalid activation record")
    active = json.loads(path.read_text(encoding="utf-8"))
    return load_model(
        registry, active["model_id"], trusted=active.get("trusted") is True
    )


def predict_bars(
    models: dict,
    record: dict,
    source: pl.DataFrame,
    *,
    decision_us: int,
    include_inputs: bool = False,
) -> dict:
    """Predict one member at the run's common completed decision time."""
    identity = record["identity"]
    recipe = _validate(models, identity)
    if decision_us % (identity["decision_step_minutes"] * MINUTE_US):
        raise ValueError("decision does not match model cadence")
    keys = pl.DataFrame({"symbol": [source["symbol"][0]], "decision_us": [decision_us]})
    matrix = sequence_matrix(source, keys, recipe)
    inputs = matrix.copy() if include_inputs else None
    unchanged = True
    result = {}
    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            if inputs is not None:
                unchanged = unchanged and np.array_equal(matrix, inputs)
            prediction = np.asarray(estimator.predict(matrix))
            if prediction.shape != (1,) or not np.isfinite(prediction).all():
                raise ValueError("invalid numeric prediction")
            result[name] = float(prediction[0])
            if inputs is not None:
                unchanged = unchanged and np.array_equal(matrix, inputs)
    if inputs is not None:
        vector = inputs[0] if unchanged else None
        if vector is not None:
            vector.setflags(write=False)
        return {
            "values": result,
            "inputs": {
                "state": "available" if unchanged else "unavailable",
                "reason": None if unchanged else "estimator_mutated_input",
                "vector": vector,
            },
        }
    return result
