"""Explicit, bounded temporal input/outcome study. Research only, never a scheduler."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US
from crypto_boom.config import project_settings
from crypto_boom.research import input_drift as snapshots
from crypto_boom.research.outcomes import read_scan_report

LOG = logging.getLogger(__name__)
MAX_BYTES = 512_000_000
MAX_COMPARISONS = 200_000


def _outcome(path: Path, doc: dict, report_hash: str, as_of: int) -> dict:
    raw = snapshots._bytes(path, 16_000_000)
    value = json.loads(raw)
    if (
        value.get("schema") not in {"scan-outcomes-v1", "scan-outcomes-v2"}
        or _artifacts.content_id(value).removeprefix("sha256:") != path.parent.name
        or value.get("prediction_report_sha256") != report_hash
        or value.get("model_id") != doc["model_id"]
        or value.get("decision_us") != doc["decision_us"]
        or type(value.get("as_of_us")) is not int
        or not doc["decision_us"] <= value["as_of_us"] <= as_of
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) > 2000
    ):
        raise ValueError("outcome identity, time or source-report mismatch")
    target = value["target"]
    descriptors = [
        d for d in doc["model_contract"]["outputs"] if d["name"] == target.get("output")
    ]
    if (
        len(descriptors) != 1
        or target.get("declaration") != "maximum_minute_close_rise_including_origin"
        or descriptors[0].get("statistic") != "quantile"
        or descriptors[0].get("unit") != "return_fraction"
        or any(
            target.get(k) != descriptors[0].get(k)
            for k in ("quantile", "horizon_minutes")
        )
        or type(target.get("horizon_minutes")) is not int
        or not 1 <= target["horizon_minutes"] <= 1440
        or type(target.get("quantile")) not in (int, float)
        or not 0 < target["quantile"] < 1
    ):
        raise ValueError("incompatible outcome target")
    rows = value["rows"]
    if len({r["symbol"] for r in rows}) != len(rows) or {
        r["symbol"] for r in rows
    } != set(doc["rows"]):
        raise ValueError("outcome population mismatch")
    losses, below = [], []
    for row in rows:
        if row["state"] != "observed":
            continue
        original = doc["rows"][row["symbol"]]
        prediction, realized = row.get("prediction"), row.get("realized")
        if (
            original["state"] != "success"
            or prediction != original["values"].get(target["output"])
            or any(
                type(x) not in (int, float) or not math.isfinite(x)
                for x in (prediction, realized)
            )
            or realized < 0
            or value["as_of_us"]
            < doc["decision_us"] + target["horizon_minutes"] * MINUTE_US
        ):
            raise ValueError("invalid or immature realized outcome")
        residual = realized - prediction
        loss = max(target["quantile"] * residual, (target["quantile"] - 1) * residual)
        if (
            not math.isfinite(loss)
            or type(row.get("pinball_loss")) not in (int, float)
            or not math.isclose(row["pinball_loss"], loss, rel_tol=1e-10, abs_tol=1e-12)
            or row.get("below_quantile") is not (realized <= prediction)
        ):
            raise ValueError("outcome metric mismatch")
        losses.append(loss)
        below.append(realized <= prediction)
    return {
        "state": "verified" if losses else "no_observed_outcomes",
        "sha256": snapshots._hash(raw),
        "target": target,
        "observed": len(losses),
        "members": len(rows),
        "counts": dict(sorted(Counter(r["state"] for r in rows).items())),
        "mean_pinball_loss": float(np.mean(losses)) if losses else None,
        "quantile_coverage": float(np.mean(below)) if below else None,
    }


def study(manifest_path: Path, *, output: Path) -> Path:
    """One manifest fixes expected origins and paths; no implicit input discovery."""
    started = time.monotonic()
    raw = snapshots._bytes(manifest_path, 1_000_000)
    manifest = json.loads(raw)
    step = manifest.get("step_minutes")
    as_of = manifest.get("as_of_us")
    if (
        manifest.get("schema") != "temporal-input-study-v1"
        or manifest.get("kind") not in {"observation", "replay"}
        or type(step) is not int
        or not 1 <= step <= 1440
        or type(as_of) is not int
    ):
        raise ValueError("invalid study schema, kind, step or as-of")
    step *= MINUTE_US
    grids = {}
    for role, maximum in (("reference", 8), ("observation", 32)):
        bounds = manifest[role + "_window"]
        start, end = bounds["start_us"], bounds["end_us"]
        if (
            any(type(t) is not int or t <= 0 or t % step for t in (start, end))
            or not 0 < end - start <= maximum * step
        ):
            raise ValueError("invalid window or temporal slot budget")
        grids[role] = list(range(start, end, step))
    if (
        manifest["reference_window"]["end_us"]
        > manifest["observation_window"]["start_us"]
        or as_of < grids["observation"][-1]
    ):
        raise ValueError("overlapping windows or future observation slots")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) > 64:
        raise ValueError("manifest exceeds 64 entries")
    slots = {t: [] for grid in grids.values() for t in grid}
    for entry in entries:
        t = entry.get("decision_us")
        if type(t) is not int or t not in slots:
            raise ValueError("entry outside declared temporal grid")
        slots[t].append(entry)
    consumed, work = 0, 0

    def charge(path: Path) -> None:
        nonlocal consumed
        consumed += path.stat().st_size
        if consumed > MAX_BYTES:
            raise ValueError("study exceeds total input byte budget")
        if time.monotonic() - started > 900:
            raise TimeoutError("study exceeds 900 seconds")

    def locator(value: str) -> Path:
        return _artifacts.resolve_reference(value, base=manifest_path.parent)

    baseline = None
    study_model = None
    target = None
    reference_arrays = []
    reference_bytes = 0
    records: list[dict] = []
    duplicate_entries = 0
    for t, choices in sorted(slots.items()):
        role = "reference" if t in grids["reference"] else "observation"
        admitted = None
        key = None
        record: dict = {
            "decision_us": t,
            "role": role,
            "state": "missing_report",
            "outcome": {"state": "not_provided"},
        }
        for entry in choices:
            candidate: dict = {
                "decision_us": t,
                "role": role,
                "state": "missing_report",
                "outcome": {"state": "not_provided"},
            }
            semantic: dict
            arrays = None
            report_path = (
                locator(entry["report"]) if entry.get("report") is not None else None
            )
            if report_path is not None and report_path.is_file():
                charge(report_path)
                doc, report_hash = read_scan_report(report_path)
                if doc["decision_us"] != t:
                    raise ValueError("report origin differs from manifest")
                if study_model is not None and study_model != doc["model_id"]:
                    raise ValueError("incompatible models across temporal study")
                study_model = doc["model_id"]
                candidate.update(
                    report_sha256=report_hash,
                    members=len(doc["rows"]),
                    prediction_states=dict(
                        sorted(
                            Counter(r["state"] for r in doc["rows"].values()).items()
                        )
                    ),
                )
                if (doc.get("input_evidence") or {}).get("state") in {
                    "complete",
                    "partial",
                }:
                    sidecar = report_path.parent / "inputs.parquet"
                    if sidecar.is_file():
                        charge(sidecar)
                        _, _, _, symbols, matrix = snapshots._inputs(
                            report_path, admitted=(doc, report_hash)
                        )
                        contract = {
                            k: doc["input_evidence"][k]
                            for k in (
                                "model_id",
                                "recipe_sha256",
                                "implementation",
                                "capture_code_sha256",
                                "dtype",
                                "feature_count",
                            )
                        }
                        if baseline is not None and contract != baseline:
                            raise ValueError(
                                "incompatible input contracts across temporal study"
                            )
                        baseline = contract
                        candidate.update(
                            state="available",
                            captured=len(symbols),
                            input_state=doc["input_evidence"]["state"],
                        )
                        arrays = (symbols, matrix, doc)
                    else:
                        candidate["state"] = "missing_inputs"
                else:
                    candidate["state"] = "input_unavailable"
                if entry.get("outcome") is not None:
                    path = locator(entry["outcome"])
                    if path.is_file():
                        charge(path)
                        outcome = _outcome(path, doc, report_hash, as_of)
                        if target is not None and outcome["target"] != target:
                            raise ValueError("outcome targets differ across origins")
                        target = outcome["target"]
                        candidate["outcome"] = outcome
                    else:
                        candidate["outcome"] = {"state": "missing_report"}
                semantic = {
                    "doc": doc,
                    "state": candidate["state"],
                    "outcome": candidate["outcome"],
                }
            else:
                if entry.get("outcome") is not None:
                    candidate["outcome"] = {"state": "missing_prediction_report"}
                semantic = candidate
            identity = _artifacts.content_id(semantic)
            if key is not None:
                if identity != key:
                    raise ValueError(
                        "conflicting duplicate origin; no latest-wins selection"
                    )
                duplicate_entries += 1
                continue
            key = identity
            record = candidate
            admitted = arrays
        if role == "reference" and admitted is not None:
            reference_bytes += admitted[1].nbytes
            if reference_bytes > 64_000_000:
                raise ValueError("reference arrays exceed 64 MB")
            reference_arrays.append((t, *admitted))
        if role == "observation" and admitted is not None:
            symbols, matrix, doc = admitted
            natural, matched = [], []
            pairs = []
            ix = {s: i for i, s in enumerate(symbols)}
            for origin, base_symbols, x, base_doc in reference_arrays:
                work += matrix.shape[1]
                if work > MAX_COMPARISONS or time.monotonic() - started > 900:
                    raise ValueError(
                        "temporal comparison work/deadline budget exceeded"
                    )
                common = sorted(set(base_symbols) & set(symbols))
                base_ix = {s: i for i, s in enumerate(base_symbols)}
                natural.append(snapshots._distribution(x, matrix))
                matched.append(
                    snapshots._distribution(
                        x[[base_ix[s] for s in common]], matrix[[ix[s] for s in common]]
                    )
                )
                pairs.append(
                    {
                        "reference_origin_us": origin,
                        "matched_symbols": len(common),
                        "reference_captured": len(base_symbols),
                        "current_captured": len(symbols),
                        "added_members": len(set(doc["rows"]) - set(base_doc["rows"])),
                        "removed_members": len(
                            set(base_doc["rows"]) - set(doc["rows"])
                        ),
                    }
                )
            record["reference_pairs"] = pairs

            def aggregate(groups):
                valid = [g for g in groups if g]
                return (
                    [
                        {
                            "feature_index": i,
                            "reference_times": len(valid),
                            "mean_cdf_distance": float(
                                np.mean([g[i]["cdf_distance"] for g in valid])
                            ),
                        }
                        for i in range(len(valid[0]))
                    ]
                    if valid
                    else []
                )

            record.update(
                comparison_state="available" if pairs else "no_reference_inputs",
                natural=aggregate(natural),
                matched=aggregate(matched),
            )
        elif role == "observation":
            record["comparison_state"] = "no_current_inputs"
        records.append(record)
    summaries = {}
    for role in grids:
        group = [r for r in records if r["role"] == role]
        losses = [
            r["outcome"]["mean_pinball_loss"]
            for r in group
            if r["outcome"]["state"] == "verified"
        ]
        summaries[role] = {
            "expected_origins": len(group),
            "input_states": dict(sorted(Counter(r["state"] for r in group).items())),
            "outcome_states": dict(
                sorted(Counter(r["outcome"]["state"] for r in group).items())
            ),
            "origins_with_realized_loss": len(losses),
            "equal_origin_mean_loss": float(np.mean(losses)) if losses else None,
        }
    report = {
        "schema": "temporal-input-diagnostics-v1",
        "manifest_sha256": snapshots._hash(raw),
        "kind": manifest["kind"],
        "as_of_us": as_of,
        "reference_window": manifest["reference_window"],
        "observation_window": manifest["observation_window"],
        "step_minutes": step // MINUTE_US,
        "duplicate_entries_collapsed": duplicate_entries,
        "model_id": study_model,
        "input_contract": baseline,
        "outcome_target": target,
        "implementation": {
            name: _artifacts.file_identity(path)[0]
            for name, path in (
                ("window", Path(__file__)),
                ("inputs", Path(snapshots.__file__)),
            )
        },
        "summary": summaries,
        "timeline": records,
    }
    payload = _artifacts.canonical_json(report)
    if len(payload) > 32_000_000:
        raise ValueError("temporal output exceeds 32 MB")
    lines = [
        "# Temporal input and outcome diagnostics",
        "",
        f"Kind: {manifest['kind']}; duplicate entries collapsed: {duplicate_entries}.",
        "",
        "| Origin UTC | Role | Input state | Captured | Observed outcomes | Realized loss | Quantile coverage |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {datetime.fromtimestamp(r['decision_us'] / 1e6, UTC).isoformat()} | {r['role']} | {r['state']} | {r.get('captured', 'unavailable')} | {r['outcome'].get('observed', 'unavailable')} | {r['outcome'].get('mean_pinball_loss', 'unavailable')} | {r['outcome'].get('quantile_coverage', 'unavailable')} |"
        )
    lines.extend(
        [
            "",
            "Input distances are averaged equally over available reference origins, not pooled symbol rows. Counts and missing grid slots remain explicit. Matched subsets may vary between origin pairs. Period losses weight available origins equally; different realized populations and missing labels can change them. Origins and symbols are not necessarily independent. No p-values, alarm thresholds, causal attribution, training-baseline claim or model promotion is established.",
        ]
    )
    result = snapshots._publish(
        output,
        report,
        {
            "report.json": payload,
            "report.md": ("\n".join(lines) + "\n").encode(),
            "selection.json": raw,
        },
    )
    LOG.info("Temporal study published: %s; expected origins=%d", result, len(records))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = project_settings(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    result = study(
        args.manifest,
        output=args.output or settings.data_root / "runs/temporal-diagnostics",
    )
    print(json.dumps({"directory": str(result)}))


if __name__ == "__main__":
    main()
