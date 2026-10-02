"""Typed immutable input evidence and descriptive research comparisons."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crypto_boom import _artifacts
from crypto_boom.research.input_drift import build_reference, compare


def snapshot(root, name, time, symbols, values):
    directory = root / name
    directory.mkdir()
    array = np.asarray(values, dtype=np.float32)
    table = pa.table(
        {
            "symbol": symbols,
            "decision_us": [time] * len(symbols),
            "cache_id": ["a" * 64] * len(symbols),
            "source_sha256": ["sha256:" + "b" * 64] * len(symbols),
            "features": pa.FixedSizeListArray.from_arrays(
                pa.array(array.reshape(-1)), array.shape[1]
            ),
        }
    )
    pq.write_table(table, directory / "inputs.parquet")
    digest, size = _artifacts.file_identity(directory / "inputs.parquet")
    recipe: dict = {"fixture": True}
    contract: dict = {"recipe": recipe, "outputs": []}
    model_id = _artifacts.content_id(contract).removeprefix("sha256:")
    evidence = {
        "schema": "inference-inputs-v1",
        "state": "complete",
        "file": "inputs.parquet",
        "model_id": model_id,
        "recipe_sha256": _artifacts.content_id(recipe),
        "dtype": "float32",
        "feature_count": array.shape[1],
        "rows": len(symbols),
        "successful_predictions": len(symbols),
        "bytes": size,
        "sha256": digest,
        "capture_code_sha256": "sha256:" + "c" * 64,
        "implementation": {
            k: "sha256:" + "d" * 64 for k in ["runtime", "features", "bars"]
        },
    }
    doc = {
        "schema": "market-scan-v3",
        "model_id": model_id,
        "model_contract": contract,
        "decision_us": time,
        "input_evidence": evidence,
        "rows": [
            {
                "symbol": s,
                "state": "success",
                "values": {},
                "cache_id": "a" * 64,
                "source_sha256": "sha256:" + "b" * 64,
                "input_evidence": {"state": "available"},
            }
            for s in symbols
        ],
    }
    path = directory / "report.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def pair(tmp_path):
    base = snapshot(
        tmp_path, "base", 60_000_000, ["AAAUSDT", "BBBUSDT"], [[0, 2], [1, 2]]
    )
    current = snapshot(
        tmp_path, "current", 120_000_000, ["AAAUSDT", "CCCUSDT"], [[2, 3], [3, 3]]
    )
    return base, current


def edit(path, fn):
    doc = json.loads(path.read_bytes())
    fn(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_constant_columns_and_population_changes(tmp_path, pair):
    ref = build_reference(pair[0], kind="replay", output=tmp_path / "refs")
    assert build_reference(pair[0], kind="replay", output=tmp_path / "refs") == ref
    output = compare(ref, pair[1], output=tmp_path / "diagnostics")
    report = json.loads((output / "report.json").read_bytes())
    assert report["population"]["added_members"] == ["CCCUSDT"]
    assert report["population"]["removed_members"] == ["BBBUSDT"]
    assert report["population"]["matched_symbols"] == ["AAAUSDT"]
    assert report["natural"][0]["cdf_distance"] == 1
    assert report["natural"][0]["standardized_mean_change"] == 4
    assert report["natural"][1]["standardized_mean_change"] is None
    assert report["natural"][1]["outside_reference_range"] == 1
    assert report["matched"][0]["standardized_mean_change"] is None
    assert compare(ref, pair[1], output=tmp_path / "diagnostics") == output
    (output / "report.md").write_text("corrupt")
    with pytest.raises(ValueError, match="publication differs"):
        compare(ref, pair[1], output=tmp_path / "diagnostics")


@pytest.mark.parametrize(
    "case",
    [
        "unavailable",
        "tamper",
        "binding",
        "width",
        "dtype",
        "path",
        "coverage",
        "nonfinite",
        "decoded_budget",
    ],
)
def test_invalid_input_evidence_refuses_reference(tmp_path, pair, case):
    path = pair[0]
    if case == "tamper":
        p = path.parent / "inputs.parquet"
        p.write_bytes(p.read_bytes() + b"bad")
    elif case == "nonfinite":
        table = pq.read_table(path.parent / "inputs.parquet")
        table = table.set_column(
            table.schema.get_field_index("features"),
            "features",
            pa.FixedSizeListArray.from_arrays(
                pa.array([float("nan"), 2.0, 1.0, 2.0], type=pa.float32()), 2
            ),
        )
        pq.write_table(table, path.parent / "inputs.parquet")
        digest, size = _artifacts.file_identity(path.parent / "inputs.parquet")
        edit(path, lambda d: d["input_evidence"].update(sha256=digest, bytes=size))
    elif case == "binding":
        edit(path, lambda d: d["rows"][0].update(source_sha256="sha256:" + "e" * 64))
    else:
        key, value = {
            "unavailable": ("state", "unavailable"),
            "width": ("feature_count", 3),
            "dtype": ("dtype", "float64"),
            "path": ("file", "../inputs.parquet"),
            "coverage": ("rows", 1),
            "decoded_budget": ("feature_count", 10_000_000),
        }[case]
        edit(path, lambda d: d["input_evidence"].update({key: value}))
    with pytest.raises(ValueError):
        build_reference(path, kind="observation", output=tmp_path / "refs")
    assert not (tmp_path / "refs").exists()


def test_time_compatibility_and_reference_integrity(tmp_path, pair):
    ref = build_reference(pair[0], kind="observation", output=tmp_path / "refs")
    with pytest.raises(ValueError, match="strictly later"):
        compare(ref, pair[0], output=tmp_path / "result")
    edit(
        pair[1],
        lambda d: d["input_evidence"]["implementation"].update(
            features="sha256:" + "e" * 64
        ),
    )
    with pytest.raises(ValueError, match="incompatible"):
        compare(ref, pair[1], output=tmp_path / "result")
    edit(ref / "reference.json", lambda d: d.update(kind="replay"))
    with pytest.raises(ValueError, match="reference identity"):
        compare(ref, pair[1], output=tmp_path / "result")
    with pytest.raises(ValueError, match="not training"):
        build_reference(pair[0], kind="training", output=tmp_path / "refs")
    assert not (tmp_path / "result").exists()


def test_partial_capture_preserves_universe_denominator(tmp_path, pair):
    def partial(doc):
        doc["rows"].append(
            {
                "symbol": "DD D",
                "state": "success",
                "values": {},
                "input_evidence": {"state": "unavailable"},
            }
        )
        doc["input_evidence"].update(state="partial", successful_predictions=3)

    edit(pair[0], partial)
    ref = build_reference(pair[0], kind="observation", output=tmp_path / "refs")
    output = compare(ref, pair[1], output=tmp_path / "results")
    report = json.loads((output / "report.json").read_bytes())
    assert report["reference_coverage"]["members"] == 3
    assert report["reference_coverage"]["captured"] == 2
    assert report["reference_coverage"]["input_state"] == "partial"


def test_equal_distributions_and_empty_matched_population(tmp_path, pair):
    ref = build_reference(pair[0], kind="replay", output=tmp_path / "refs")
    same = snapshot(
        tmp_path, "same", 180_000_000, ["XXXUSDT", "YYYUSDT"], [[0, 2], [1, 2]]
    )
    directory = compare(ref, same, output=tmp_path / "results")
    result = json.loads((directory / "report.json").read_bytes())
    assert result["matched"] == []
    assert result["population"]["matched_symbols"] == []
    assert all(row["cdf_distance"] == 0 for row in result["natural"])


def test_publication_interruption_leaves_no_reference(tmp_path, pair, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_artifacts, "write_exclusive_bytes", interrupt)
    with pytest.raises(KeyboardInterrupt):
        build_reference(pair[0], kind="replay", output=tmp_path / "refs")
    assert not list((tmp_path / "refs").iterdir())
