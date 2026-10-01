"""Portable references are locators; content and research policy own identity."""

import json
from pathlib import Path

import pytest

from crypto_boom import _artifacts, features
from crypto_boom.feature_batch import (
    build_feature_cache,
    feature_source_paths,
    read_feature_cache,
)
from crypto_boom.research.path_dataset import build_path_dataset, load_path_dataset
from crypto_boom.research.path_targets import (
    PathTargetSpec,
    build_target_cache,
    read_target_cache,
)
from crypto_boom.sample_pool import (
    export_sample_pool,
    load_sample_pool,
    sample_pool_identity,
)
from test_forward_prediction import bars


def test_feature_identity_survives_input_rename_order_and_rebinding(
    tmp_path, monkeypatch
):
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    source = bars(2200)
    source.head(1100).write_parquet(left)
    source.tail(1100).write_parquet(right)
    cache, _ = build_feature_cache([left, right], output_root=tmp_path / "features")
    original = (cache / "receipt.json").read_bytes()
    _, receipt = read_feature_cache(cache)
    assert all("path" not in item for item in receipt["spec"]["sources"])
    moved = tmp_path / "renamed.parquet"
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    left.rename(moved)
    with monkeypatch.context() as patch:
        patch.setattr(
            features,
            "iter_feature_segments",
            lambda *a, **k: pytest.fail("renaming recomputed features"),
        )
        assert build_feature_cache(
            [right, moved], output_root=tmp_path / "features"
        ) == (cache, True)
    assert (cache / "receipt.json").read_bytes() == original
    paths = feature_source_paths(cache, receipt, paths=[right, moved])
    target, _ = build_target_cache(
        cache,
        output_root=tmp_path / "targets",
        source_paths=paths,
        spec=PathTargetSpec((10,), ()),
    )
    assert len(read_target_cache(target)[0]) > 0
    moved.write_bytes(moved.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source changed"):
        build_target_cache(
            cache, output_root=tmp_path / "different-targets", source_paths=paths
        )
    with pytest.raises(ValueError, match="distinct"):
        build_feature_cache([right, right], output_root=tmp_path / "features")


def test_complete_tree_relocation_keeps_dataset_and_cache_identity(
    tmp_path, monkeypatch
):
    from crypto_boom.research import path_dataset

    root = tmp_path / "before"
    (root / "raw").mkdir(parents=True)
    bars(2200).write_parquet(root / "raw/source.parquet")
    pool = root / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "selection_id": "fixture",
                "availability_id": "fixture",
                "symbols_with_data": ["AAAUSDT"],
                "partitions": [
                    {
                        "symbol": "AAAUSDT",
                        "month": "2026-01",
                        "state": "available",
                        "path": "raw/source.parquet",
                    }
                ],
            }
        )
    )

    def load(p):
        value = json.loads(p.read_text())
        value["partitions"][0]["path"] = str(p.parent / value["partitions"][0]["path"])
        return value

    monkeypatch.setattr(path_dataset, "load_sample_pool", load)
    first = build_path_dataset(
        pool,
        output_root=root / "runs/one",
        data_root=root,
        spec=PathTargetSpec((10,), ()),
    )
    manifest = Path(first["manifest"])
    old_rows, _ = load_path_dataset(manifest)
    stored = manifest.read_bytes()
    relative = manifest.relative_to(root)
    moved = tmp_path / "after"
    assert root.resolve().is_relative_to(
        tmp_path.resolve()
    ) and moved.resolve().is_relative_to(tmp_path.resolve())
    root.rename(moved)
    new_rows, view = load_path_dataset(moved / relative)
    assert new_rows.equals(old_rows) and (moved / relative).read_bytes() == stored
    assert Path(view["batches"][0]["features"]).is_relative_to(moved)
    second = build_path_dataset(
        moved / "pool.json",
        output_root=moved / "runs/two",
        data_root=moved,
        spec=PathTargetSpec((10,), ()),
    )
    assert first["dataset_id"] == second["dataset_id"]
    assert (
        second["batches"][0]["feature_reused"] and second["batches"][0]["target_reused"]
    )
    cache = Path(second["batches"][0]["features"])
    target, _ = build_target_cache(
        cache, output_root=moved / "derived/targets", spec=PathTargetSpec((20,), ())
    )
    assert "up_20" in read_target_cache(target)[0].columns
    renamed = moved / "raw/renamed.parquet"
    assert renamed.resolve().is_relative_to(tmp_path.resolve())
    (moved / "raw/source.parquet").rename(renamed)
    pool_value = json.loads((moved / "pool.json").read_text())
    pool_value["partitions"][0]["path"] = "raw/renamed.parquet"
    (moved / "pool.json").write_text(json.dumps(pool_value))
    with pytest.raises(ValueError, match="fresh run"):
        build_path_dataset(
            moved / "pool.json",
            output_root=moved / "runs/two",
            data_root=moved,
            spec=PathTargetSpec((10,), ()),
        )
    rebound = build_path_dataset(
        moved / "pool.json",
        output_root=moved / "runs/three",
        data_root=moved,
        spec=PathTargetSpec((10,), ()),
    )
    assert rebound["dataset_id"] == first["dataset_id"]
    _, view = load_path_dataset(Path(rebound["manifest"]))
    _, cached = read_feature_cache(cache)
    assert feature_source_paths(
        cache, cached, paths=[Path(p) for p in view["batches"][0]["source_paths"]]
    ) == [renamed]
    legacy = json.loads((cache / "receipt.json").read_text())
    legacy["schema"] = "feature-cache-v1"
    legacy["spec"].pop("schema")
    legacy["spec"]["sources"] = [
        {
            "path": str(moved / "raw/renamed.parquet"),
            "sha256": _artifacts.file_identity(moved / "raw/renamed.parquet")[0],
        }
    ]
    legacy["cache_id"] = _artifacts.content_id(legacy["spec"])
    legacy.pop("sources")
    old_cache = moved / "legacy"
    old_cache.mkdir()
    (old_cache / "receipt.json").write_bytes(_artifacts.canonical_json(legacy))
    (old_cache / "features.parquet").write_bytes(
        (cache / "features.parquet").read_bytes()
    )
    _, read = read_feature_cache(old_cache)
    assert feature_source_paths(old_cache, read) == [moved / "raw/renamed.parquet"]


def test_pool_export_preserves_original_and_survives_tree_move(tmp_path, monkeypatch):
    from crypto_boom.storage.source import materialize_research_corpus
    from test_storage_source import _install_source, _report

    root = tmp_path / "before"
    root.mkdir()
    _install_source(monkeypatch, root)
    part = materialize_research_corpus(
        _report(), monthly_root=root / "raw", output_root=root / "canonical"
    ).publications[0]
    selection: dict[str, object] = {
        "symbols": ["ETHUSDT"],
        "start_month": "2025-01",
        "end_month": "2025-01",
    }
    old: dict[str, object] = {
        "schema": "sample-pool-v1",
        "selection": selection,
        "selection_id": _artifacts.content_id(selection),
        "availability_id": _report().report_id,
        "available_symbol_months": 1,
        "symbols_with_data": ["ETHUSDT"],
        "partitions": [
            {
                "symbol": "ETHUSDT",
                "month": "2025-01",
                "state": "available",
                "path": str(part.path / "klines.parquet"),
                "parquet_sha256": part.manifest.parquet_sha256,
                "checksum_sha256": part.manifest.source_revision,
            }
        ],
    }
    source = root / "old.json"
    original = _artifacts.canonical_json(old)
    source.write_bytes(original)
    target = root / "runs/pool.json"
    exported = export_sample_pool(source, target)
    assert source.read_bytes() == original
    assert exported["pool_id"] == sample_pool_identity(load_sample_pool(source))
    with pytest.raises(FileExistsError):
        export_sample_pool(source, target)
    moved = tmp_path / "after"
    assert root.resolve().is_relative_to(
        tmp_path.resolve()
    ) and moved.resolve().is_relative_to(tmp_path.resolve())
    root.rename(moved)
    assert load_sample_pool(moved / "runs/pool.json")["pool_id"] == exported["pool_id"]
    assert (moved / "old.json").read_bytes() == original
    target = moved / "runs/pool.json"
    changed = json.loads(target.read_text())
    changed["partitions"][0]["checksum_sha256"] = "tampered"
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        load_sample_pool(target)


@pytest.mark.parametrize(
    "reference", ["/absolute", "C:/absolute", "C:drive-relative", "bad\\separator", ""]
)
def test_portable_locator_rejects_ambiguous_absolute_paths(tmp_path, reference):
    with pytest.raises(ValueError, match="relative"):
        _artifacts.resolve_reference(reference, base=tmp_path)


@pytest.mark.parametrize("kind", ["features", "targets"])
def test_cache_hit_checks_expected_identity_not_only_self_consistent_receipt(
    tmp_path, kind
):
    source = tmp_path / "source.parquet"
    bars(2200).write_parquet(source)
    cache, _ = build_feature_cache([source], output_root=tmp_path / "features")
    target, _ = build_target_cache(
        cache, output_root=tmp_path / "targets", spec=PathTargetSpec((10,), ())
    )
    path = (cache if kind == "features" else target) / "receipt.json"
    record = json.loads(path.read_text())
    record["spec"]["other_identity"] = True
    record["cache_id"] = _artifacts.content_id(record["spec"])
    path.write_bytes(_artifacts.canonical_json(record))
    with pytest.raises(ValueError, match="another identity"):
        if kind == "features":
            build_feature_cache([source], output_root=tmp_path / "features")
        else:
            build_target_cache(
                cache, output_root=tmp_path / "targets", spec=PathTargetSpec((10,), ())
            )
