"""Offline input-reference diagnostics; no alarms, fitting or public scan policy."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_boom import _artifacts
from crypto_boom.config import project_settings
from crypto_boom.research.outcomes import read_scan_report

LOG = logging.getLogger(__name__)


def _bytes(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("input exceeds byte budget")
    return raw


def _hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _inputs(
    report: Path, *, admitted: tuple[dict, str] | None = None
) -> tuple[dict, str, bytes, list[str], np.ndarray]:
    doc, report_hash = admitted if admitted is not None else read_scan_report(report)
    evidence = doc.get("input_evidence")
    if not isinstance(evidence, dict) or evidence.get("state") not in {
        "complete",
        "partial",
    }:
        raise ValueError("input evidence unavailable; never reconstruct missing inputs")
    if (
        doc["schema"] != "market-scan-v3"
        or evidence.get("schema") != "inference-inputs-v1"
        or evidence.get("file") != "inputs.parquet"
        or evidence.get("model_id") != doc["model_id"]
        or evidence.get("recipe_sha256")
        != _artifacts.content_id(doc["model_contract"]["recipe"])
        or evidence.get("dtype") not in {"float32", "float64"}
        or type(evidence.get("rows")) is not int
        or not 1 <= evidence["rows"] <= 2000
        or type(evidence.get("feature_count")) is not int
        or not 1 <= evidence["feature_count"] <= 16_384
    ):
        raise ValueError("invalid input evidence contract")
    if not isinstance(evidence.get("implementation"), dict):
        raise ValueError("invalid implementation identity")
    for digest in [
        evidence.get("capture_code_sha256"),
        *(evidence.get("implementation") or {}).values(),
    ]:
        if not isinstance(digest, str) or not _artifacts.is_sha256(digest):
            raise ValueError("invalid implementation identity")
    if set(evidence.get("implementation", {})) != {"runtime", "features", "bars"}:
        raise ValueError("incomplete implementation identity")
    raw = _bytes(report.parent / "inputs.parquet", 20_000_000)
    if len(raw) != evidence.get("bytes") or _hash(raw) != evidence.get("sha256"):
        raise ValueError("input evidence hash/size mismatch")
    parquet = pq.ParquetFile(io.BytesIO(raw))
    metadata = parquet.metadata
    if (
        metadata.num_rows != evidence["rows"]
        or sum(
            metadata.row_group(i).total_byte_size
            for i in range(metadata.num_row_groups)
        )
        > 32_000_000
    ):
        raise ValueError("input evidence decoded budget mismatch")
    dtype = np.dtype(evidence["dtype"])
    if evidence["rows"] * evidence["feature_count"] * dtype.itemsize > 16_000_000:
        raise ValueError("numeric input payload exceeds 16 MB")
    schema = parquet.schema_arrow
    expected = pa.list_(
        pa.float32() if dtype == np.dtype("float32") else pa.float64(),
        evidence["feature_count"],
    )
    if (
        set(schema.names)
        != {"symbol", "decision_us", "cache_id", "source_sha256", "features"}
        or schema.field("features").type != expected
    ):
        raise ValueError("input evidence shape/type mismatch")
    if any(
        schema.field(k).type != dtype
        for k, dtype in {
            "symbol": pa.string(),
            "decision_us": pa.int64(),
            "cache_id": pa.string(),
            "source_sha256": pa.string(),
        }.items()
    ):
        raise ValueError("input key column type mismatch")
    table = parquet.read()
    if (
        any(c.null_count for c in table.columns)
        or table["features"].combine_chunks().values.null_count
    ):
        raise ValueError("null input evidence")
    symbols = table["symbol"].to_pylist()
    if len(set(symbols)) != len(symbols):
        raise ValueError("duplicate input symbol")
    successful = {s for s, r in doc["rows"].items() if r["state"] == "success"}
    available = {
        s
        for s in successful
        if (doc["rows"][s].get("input_evidence") or {}).get("state") == "available"
    }
    if (
        set(symbols) != available
        or evidence.get("successful_predictions") != len(successful)
        or (evidence["state"] == "complete") != (available == successful)
    ):
        raise ValueError("input coverage ledger mismatch")
    for i, symbol in enumerate(symbols):
        row = doc["rows"][symbol]
        if (
            not isinstance(row.get("cache_id"), str)
            or not _artifacts.is_sha256("sha256:" + row["cache_id"])
            or not isinstance(row.get("source_sha256"), str)
            or not _artifacts.is_sha256(row["source_sha256"])
        ):
            raise ValueError("invalid source identity")
        if table["decision_us"][i].as_py() != doc["decision_us"] or any(
            table[k][i].as_py() != row.get(k) for k in ("cache_id", "source_sha256")
        ):
            raise ValueError("input row/source binding mismatch")
    matrix = np.asarray(table["features"].combine_chunks().values).reshape(
        len(symbols), evidence["feature_count"]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite input evidence")
    return doc, report_hash, raw, symbols, matrix


def _publish(output: Path, identity: dict, files: dict[str, bytes]) -> Path:
    target = output / _artifacts.content_id(identity).removeprefix("sha256:")

    def verify(directory: Path) -> None:
        for name, payload in files.items():
            if _bytes(directory / name, len(payload)) != payload:
                raise ValueError("existing publication differs")

    output.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output, prefix="input-study-"
    ) as staging:
        for name, payload in files.items():
            _artifacts.write_exclusive_bytes(staging / name, payload)
        _artifacts.adopt_directory(staging, target, verify_existing=verify)
    return target


def build_reference(report: Path, *, kind: str, output: Path) -> Path:
    """Freeze one declared observation/replay, never a claimed training reference."""
    if kind not in {"observation", "replay"}:
        raise ValueError("reference kind must be observation or replay, not training")
    doc, source_hash, raw, _, _ = _inputs(report)
    compact = doc | {
        "rows": [{"symbol": s, **r} for s, r in sorted(doc["rows"].items())]
    }
    manifest = _artifacts.canonical_json(compact)
    identity: dict = {
        "schema": "input-reference-v1",
        "kind": kind,
        "source_report_sha256": source_hash,
        "scan_sha256": _hash(manifest),
        "inputs_sha256": _hash(raw),
    }
    path = _publish(
        output,
        identity,
        {
            "reference.json": _artifacts.canonical_json(identity),
            "scan.json": manifest,
            "inputs.parquet": raw,
        },
    )
    LOG.info("Frozen %s reference: %s", kind, path)
    return path


def _distribution(reference: np.ndarray, current: np.ndarray) -> list[dict]:
    if not len(reference) or not len(current):
        return []
    result = []
    for index in range(reference.shape[1]):
        x, y = (
            reference[:, index].astype(np.float64),
            current[:, index].astype(np.float64),
        )
        support = np.unique(np.concatenate([x, y]))
        distance = np.max(
            np.abs(
                np.searchsorted(np.sort(x), support, side="right") / len(x)
                - np.searchsorted(np.sort(y), support, side="right") / len(y)
            )
        )
        scale = float(x.std())
        result.append(
            {
                "feature_index": index,
                "reference_mean": float(x.mean()),
                "current_mean": float(y.mean()),
                "reference_median": float(np.median(x)),
                "current_median": float(np.median(y)),
                "reference_std": scale,
                "standardized_mean_change": float((y.mean() - x.mean()) / scale)
                if scale
                else None,
                "cdf_distance": float(distance),
                "outside_reference_range": float(
                    ((y < x.min()) | (y > x.max())).mean()
                ),
            }
        )
    if any(
        v is not None and not np.isfinite(v) for row in result for v in row.values()
    ):
        raise ValueError("nonfinite diagnostic; numeric range unsupported")
    return result


def compare(reference: Path, current: Path, *, output: Path) -> Path:
    """Describe a strictly later compatible snapshot; no significance/alert claim."""
    frozen = json.loads(_bytes(reference / "reference.json", 8192))
    if (
        frozen.get("schema") != "input-reference-v1"
        or frozen.get("kind") not in {"observation", "replay"}
        or _artifacts.content_id(frozen).removeprefix("sha256:") != reference.name
    ):
        raise ValueError("reference identity mismatch")
    base, base_hash, base_raw, base_symbols, x = _inputs(reference / "scan.json")
    if base_hash != frozen["scan_sha256"] or _hash(base_raw) != frozen["inputs_sha256"]:
        raise ValueError("reference content mismatch")
    latest, latest_hash, _, symbols, y = _inputs(current)
    if latest["decision_us"] <= base["decision_us"]:
        raise ValueError(
            "current origin must be strictly later; overlap is not new evidence"
        )
    keys = (
        "model_id",
        "recipe_sha256",
        "implementation",
        "capture_code_sha256",
        "dtype",
        "feature_count",
    )
    if any(base["input_evidence"][k] != latest["input_evidence"][k] for k in keys):
        raise ValueError(
            "incompatible model/feature/capture contract; do not call it drift"
        )
    common = sorted(set(base_symbols) & set(symbols))
    base_ix, current_ix = (
        {s: i for i, s in enumerate(base_symbols)},
        {s: i for i, s in enumerate(symbols)},
    )
    matched_x, matched_y = (
        x[[base_ix[s] for s in common]],
        y[[current_ix[s] for s in common]],
    )

    def coverage(doc):
        return {
            "members": len(doc["rows"]),
            "prediction_states": dict(
                sorted(Counter(r["state"] for r in doc["rows"].values()).items())
            ),
            "input_state": doc["input_evidence"]["state"],
            "captured": doc["input_evidence"]["rows"],
        }

    report = {
        "schema": "input-diagnostics-v1",
        "reference_id": reference.name,
        "reference_kind": frozen["kind"],
        "current_report_sha256": latest_hash,
        "reference_origin_us": base["decision_us"],
        "current_origin_us": latest["decision_us"],
        "implementation_sha256": _artifacts.file_identity(Path(__file__))[0],
        "reference_coverage": coverage(base),
        "current_coverage": coverage(latest),
        "population": {
            "added_members": sorted(set(latest["rows"]) - set(base["rows"])),
            "removed_members": sorted(set(base["rows"]) - set(latest["rows"])),
            "captured_only_reference": sorted(set(base_symbols) - set(symbols)),
            "captured_only_current": sorted(set(symbols) - set(base_symbols)),
            "matched_symbols": common,
        },
        "natural": _distribution(x, y),
        "matched": _distribution(matched_x, matched_y),
    }
    ranked = sorted(
        report["natural"], key=lambda d: (-d["cdf_distance"], d["feature_index"])
    )[:10]
    text = (
        "# Input distribution diagnostics\n\n"
        + f"Reference kind: {frozen['kind']}; matched symbols: {len(common)}.\n\nReference coverage: {report['reference_coverage']}\n\nCurrent coverage: {report['current_coverage']}\n\n## Largest descriptive CDF distances (natural captured populations)\n\n| Feature index | CDF distance | Outside reference range |\n|---|---:|---:|\n"
        + "".join(
            f"| {r['feature_index']} | {r['cdf_distance']:.4f} | {r['outside_reference_range']:.2%} |\n"
            for r in ranked
        )
        + "\nFeature indices refer to the pinned recipe/order, not independent hypotheses. Full natural and matched comparisons are in report.json. One snapshot is not a training distribution or representative temporal baseline. Distances have no p-values or alarm thresholds. Population changes and missing capture can change metrics; this is not proof of model degradation or a retraining/promotion instruction.\n"
    )
    path = _publish(
        output,
        report,
        {"report.json": _artifacts.canonical_json(report), "report.md": text.encode()},
    )
    LOG.info("Input diagnostics published: %s (%d matched symbols)", path, len(common))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--report", type=Path, required=True)
    ref.add_argument("--kind", choices=("observation", "replay"), required=True)
    ref.add_argument("--output", type=Path)
    diag = commands.add_parser("compare")
    diag.add_argument("--reference", type=Path, required=True)
    diag.add_argument("--current", type=Path, required=True)
    diag.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = project_settings(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.command == "reference":
        directory = build_reference(
            args.report,
            kind=args.kind,
            output=args.output or settings.data_root / "runs/input-references",
        )
    else:
        directory = compare(
            args.reference,
            args.current,
            output=args.output or settings.data_root / "runs/input-diagnostics",
        )
    print(json.dumps({"directory": str(directory)}))


if __name__ == "__main__":
    main()
