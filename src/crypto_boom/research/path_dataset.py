"""Join reusable causal batches to independently versioned future path labels."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from crypto_boom import _artifacts
from crypto_boom.feature_batch import build_feature_cache, read_feature_cache
from crypto_boom.research.path_targets import (
    DEFAULT_TARGETS,
    PathTargetSpec,
    build_target_cache,
    read_target_cache,
)
from crypto_boom.sample_pool import load_sample_pool


def build_path_dataset(
    pool_path: Path,
    *,
    output_root: Path,
    spec: PathTargetSpec = DEFAULT_TARGETS,
    step_minutes: int = 5,
    minimum_turnover: float = 1_000_000,
    data_root: Path | None = None,
) -> dict:
    """Build all available symbols; unavailable members remain in the pool ledger."""
    pool = load_sample_pool(pool_path)
    batches = []
    for symbol in pool["symbols_with_data"]:
        paths = [
            Path(p["path"])
            for p in pool["partitions"]
            if p["symbol"] == symbol and p["state"] == "available"
        ]
        feature_path, feature_reused = build_feature_cache(
            paths,
            output_root=(
                data_root / "derived/features"
                if data_root is not None
                else output_root / "features"
            ),
            step_minutes=step_minutes,
            minimum_turnover=minimum_turnover,
        )
        target_path, target_reused = build_target_cache(
            feature_path,
            output_root=(
                data_root / "derived/targets"
                if data_root is not None
                else output_root / "targets"
            ),
            spec=spec,
        )
        _, feature = read_feature_cache(feature_path)
        _, target = read_target_cache(target_path)
        batches.append(
            {
                "symbol": symbol,
                "features": str(feature_path.resolve()),
                "targets": str(target_path.resolve()),
                "feature_id": feature["cache_id"],
                "target_id": target["cache_id"],
                "quality": feature["quality"],
                "complete_by_horizon": target["complete_by_horizon"],
                "feature_reused": feature_reused,
                "target_reused": target_reused,
            }
        )
    identity: dict[str, object] = {
        "pool_sha256": _artifacts.file_identity(pool_path)[0],
        "batches": [
            {k: b[k] for k in ("symbol", "feature_id", "target_id")} for b in batches
        ],
    }
    dataset_id = _artifacts.content_id(identity)
    result: dict[str, object] = {
        "schema": "path-dataset-v1",
        "dataset_id": dataset_id,
        "identity": identity,
        "pool": str(pool_path.resolve()),
        "batches": batches,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / (dataset_id.removeprefix("sha256:") + ".json")
    if not destination.exists():
        _artifacts.write_exclusive_bytes(destination, _artifacts.canonical_json(result))
    return {**result, "manifest": str(destination.resolve())}


def load_path_dataset(path: Path) -> tuple[pl.DataFrame, dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != "path-dataset-v1"
        or _artifacts.content_id(record["identity"]) != record["dataset_id"]
    ):
        raise ValueError("invalid dataset identity")
    expected = [
        {k: b[k] for k in ("symbol", "feature_id", "target_id")}
        for b in record["batches"]
    ]
    if expected != record["identity"]["batches"]:
        raise ValueError("dataset batches differ from identity")
    frames = []
    count = 0
    for batch in record["batches"]:
        features, f = read_feature_cache(Path(batch["features"]))
        targets, t = read_target_cache(Path(batch["targets"]))
        if (f["cache_id"], t["cache_id"], t["spec"]["feature_cache_id"]) != (
            batch["feature_id"],
            batch["target_id"],
            f["cache_id"],
        ):
            raise ValueError("incompatible feature/target cache identities")
        keys = ["symbol", "decision_us"]
        if not features.select(keys).sort(keys).equals(targets.select(keys).sort(keys)):
            raise ValueError("feature/target origin mismatch")
        if len(features) and features["symbol"].unique().to_list() != [batch["symbol"]]:
            raise ValueError("dataset symbol mismatch")
        count += len(features)
        if count > 2_000_000:
            raise ValueError("dataset exceeds two million origins")
        frames.append(features.join(targets, on=keys, validate="1:1"))
    if not frames:
        raise ValueError("dataset has no available symbols")
    rows = pl.concat(frames).sort("symbol", "decision_us")
    if rows.select(pl.struct("symbol", "decision_us").n_unique()).item() != len(rows):
        raise ValueError("duplicate dataset origins")
    return rows, record
