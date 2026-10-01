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
from crypto_boom.sample_pool import load_sample_pool, sample_pool_identity


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
    batches: list[dict] = []
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
            source_paths=paths,
        )
        _, feature = read_feature_cache(feature_path)
        _, target = read_target_cache(target_path)
        batches.append(
            {
                "symbol": symbol,
                "source_paths": [str(p.resolve()) for p in paths],
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
        "schema": "path-dataset-identity-v2",
        "pool_id": sample_pool_identity(pool),
        "batches": [
            {k: b[k] for k in ("symbol", "feature_id", "target_id")} for b in batches
        ],
    }
    dataset_id = _artifacts.content_id(identity)
    result: dict[str, object] = {
        "schema": "path-dataset-v2",
        "dataset_id": dataset_id,
        "identity": identity,
        "pool": _artifacts.relative_reference(pool_path, base=output_root),
        "batches": [
            {
                **batch,
                "source_paths": [
                    _artifacts.relative_reference(Path(p), base=output_root)
                    for p in batch["source_paths"]
                ],
                "features": _artifacts.relative_reference(
                    Path(batch["features"]), base=output_root
                ),
                "targets": _artifacts.relative_reference(
                    Path(batch["targets"]), base=output_root
                ),
            }
            for batch in batches
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / (dataset_id.removeprefix("sha256:") + ".json")
    if destination.exists():
        stored = json.loads(destination.read_text(encoding="utf-8"))
        if (
            stored.get("schema") != "path-dataset-v2"
            or stored.get("identity") != identity
            or stored.get("dataset_id") != dataset_id
        ):
            raise ValueError("existing dataset identity conflict")
        if stored.get("pool") != result["pool"] or any(
            any(
                old.get(key) != new[key]
                for key in ("features", "targets", "source_paths")
            )
            for old, new in zip(stored["batches"], result["batches"], strict=True)
        ):
            raise ValueError(
                "existing dataset references differ; publish into a fresh run"
            )
    else:
        _artifacts.write_exclusive_bytes(destination, _artifacts.canonical_json(result))
    return {
        **_resolve_dataset_references(result, destination),
        "manifest": str(destination.resolve()),
    }


def load_path_dataset(path: Path) -> tuple[pl.DataFrame, dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") not in ("path-dataset-v1", "path-dataset-v2")
        or _artifacts.content_id(record["identity"]) != record["dataset_id"]
    ):
        raise ValueError("invalid dataset identity")
    expected = [
        {k: b[k] for k in ("symbol", "feature_id", "target_id")}
        for b in record["batches"]
    ]
    if expected != record["identity"]["batches"]:
        raise ValueError("dataset batches differ from identity")
    record = _resolve_dataset_references(record, path)
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


def _resolve_dataset_references(record: dict, path: Path) -> dict:
    """Return a runtime view; stored v2 bytes and identity are never rewritten."""
    if record["schema"] == "path-dataset-v1":
        return record
    return {
        **record,
        "pool": str(_artifacts.resolve_reference(record["pool"], base=path.parent)),
        "batches": [
            {
                **batch,
                "source_paths": [
                    str(_artifacts.resolve_reference(p, base=path.parent))
                    for p in batch["source_paths"]
                ],
                **{
                    name: str(
                        _artifacts.resolve_reference(batch[name], base=path.parent)
                    )
                    for name in ("features", "targets")
                },
            }
            for batch in record["batches"]
        ],
    }
