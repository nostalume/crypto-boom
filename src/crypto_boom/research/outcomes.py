"""Bounded offline reconciliation of declared scan predictions and future prices."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from crypto_boom import _artifacts, bars
from crypto_boom.config import project_settings
from crypto_boom.research import path_targets as targets

LOG = logging.getLogger(__name__)
MAX_SOURCE_BYTES = 512_000_000


def read_scan_report(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        raw = stream.read(64_000_001)
    if len(raw) > 64_000_000:
        raise ValueError("report exceeds 64 MB")
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema")
        not in {"market-scan-v1", "market-scan-v2", "market-scan-v3"}
        or type(doc.get("decision_us")) is not int
        or doc["decision_us"] <= 0
        or doc["decision_us"] % bars.MINUTE_US
        or not isinstance(doc.get("rows"), list)
        or len(doc["rows"]) > 4000
        or not isinstance(doc.get("model_contract"), dict)
        or _artifacts.content_id(doc.get("model_contract", {}))
        != "sha256:" + str(doc.get("model_id"))
    ):
        raise ValueError("invalid scan report identity, time or row budget")
    rows = {}
    for row in doc["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("state"), str):
            raise ValueError("invalid report row")
        symbol = row.get("symbol")
        if (
            not isinstance(symbol, str)
            or not symbol
            or len(symbol) > 128
            or not symbol.isprintable()
        ):
            raise ValueError("invalid report symbol")
        # Product metadata is not prediction identity or another observation.
        evidence = {
            k: row.get(k)
            for k in ("state", "values", "cache_id", "source_sha256", "input_evidence")
        }
        if symbol in rows and rows[symbol] != evidence:
            rows[symbol] = {"state": "conflict"}
        else:
            rows[symbol] = evidence
    if len(rows) > 2000:
        raise ValueError("report exceeds 2000 members")
    return {
        k: doc[k] for k in ("schema", "decision_us", "model_id", "model_contract")
    } | {
        "rows": rows,
        "input_evidence": doc.get("input_evidence"),
    }, "sha256:" + hashlib.sha256(raw).hexdigest()


def reconcile(
    prediction_report: Path,
    outcome_report: Path,
    *,
    snapshots: Path,
    as_of_us: int,
    upside_output: str,
    output: Path,
    candidate_id: str | None = None,
    candidate_output: str | None = None,
    registry: Path | None = None,
    trusted: bool = False,
) -> Path:
    """Join a declared pair, optionally replay a frozen candidate; never fit/activate."""
    started = time.monotonic()
    if type(as_of_us) is not int or as_of_us <= 0:
        raise ValueError("as_of_us must be a positive UTC timestamp")
    prediction, prediction_hash = read_scan_report(prediction_report)
    future, future_hash = read_scan_report(outcome_report)
    descriptors = [
        d for d in prediction["model_contract"]["outputs"] if d["name"] == upside_output
    ]
    if len(descriptors) != 1:
        raise ValueError("choose one explicit upside output")
    descriptor = descriptors[0]
    horizon, quantile = descriptor.get("horizon_minutes"), descriptor.get("quantile")
    if (
        descriptor.get("statistic") != "quantile"
        or descriptor.get("unit") != "return_fraction"
        or type(horizon) is not int
        or not 1 <= horizon <= 1440
        or type(quantile) not in (int, float)
        or not 0 < quantile < 1
        or prediction["decision_us"] > as_of_us
    ):
        raise ValueError("unsupported upside quantile contract or as-of time")
    candidate = None
    if candidate_id is not None:
        if registry is None or candidate_output is None:
            raise ValueError("candidate requires registry and explicit upside output")
        from crypto_boom import features, model_runtime

        models, record, _ = model_runtime.load_model(
            registry, candidate_id, trusted=trusted
        )
        outputs = [
            d for d in record["identity"]["outputs"] if d["name"] == candidate_output
        ]
        if len(outputs) != 1 or any(
            outputs[0].get(k) != descriptor.get(k)
            for k in ("horizon_minutes", "quantile", "statistic", "unit")
        ):
            raise ValueError("candidate upside output is not comparable")
        if prediction["decision_us"] % (
            record["identity"]["decision_step_minutes"] * bars.MINUTE_US
        ):
            raise ValueError("candidate cadence does not admit this origin")
        candidate = {
            "model": record,
            "output": candidate_output,
            "runtime_sha256": _artifacts.file_identity(Path(model_runtime.__file__))[0],
            "features_sha256": _artifacts.file_identity(Path(features.__file__))[0],
        }
        LOG.info("Loaded frozen candidate %s; activation unchanged", candidate_id)
    elif candidate_output is not None or registry is not None or trusted:
        raise ValueError("candidate options require a candidate ID")
    bytes_read = 0

    def snapshot(row: dict, symbol: str, decision: int) -> pl.DataFrame:
        nonlocal bytes_read
        cache_id = row.get("cache_id")
        if not isinstance(cache_id, str) or not re.fullmatch(r"[0-9a-f]{64}", cache_id):
            raise ValueError("invalid snapshot identity")
        directory = snapshots / cache_id
        with (directory / "receipt.json").open("rb") as stream:
            receipt_raw = stream.read(4097)
        bytes_read += len(receipt_raw)
        if bytes_read > MAX_SOURCE_BYTES:
            raise RuntimeError("source reads exceed 512 MB")
        if len(receipt_raw) > 4096:
            raise ValueError("snapshot receipt exceeds budget")
        receipt = json.loads(receipt_raw)
        spec = receipt["spec"]
        history = spec.get("history_minutes")
        if (
            spec.get("schema") != "spot-minute-snapshot-v1"
            or spec.get("symbol") != symbol
            or spec.get("decision_us") != decision
            or type(history) is not int
            or not 1 <= history <= 1440
            or _artifacts.content_id(spec) != "sha256:" + cache_id
        ):
            raise ValueError("snapshot spec mismatch")
        with (directory / "bars.parquet").open("rb") as stream:
            raw = stream.read(2_000_001)
        bytes_read += len(raw)
        if bytes_read > MAX_SOURCE_BYTES:
            raise RuntimeError("source reads exceed 512 MB")
        if (
            len(raw) > 2_000_000
            or receipt["sha256"] != row.get("source_sha256")
            or receipt["sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("snapshot size/hash mismatch")
        metadata = pq.ParquetFile(io.BytesIO(raw)).metadata
        if (
            metadata.num_rows != history + 1
            or sum(
                metadata.row_group(i).total_byte_size
                for i in range(metadata.num_row_groups)
            )
            > 8_000_000
        ):
            raise ValueError("snapshot row budget mismatch")
        frame = bars.admit_bars(
            pl.read_parquet(io.BytesIO(raw), columns=list(bars.SOURCE_COLUMNS))
        )
        times = frame["open_time"].dt.epoch("us")
        if (
            len(frame) != history + 1
            or set(frame["symbol"]) != {symbol}
            or times[-1] + bars.MINUTE_US != decision
            or not (times.diff().drop_nulls() == bars.MINUTE_US).all()
            or not frame["quality_complete"].all()
            or not (frame["quality_state"] == "valid").all()
        ):
            raise ValueError("snapshot is not a complete valid window")
        return frame

    results: list[dict] = []
    t = prediction["decision_us"]
    LOG.info("Reconciling %d members at origin %d", len(prediction["rows"]), t)
    for symbol, row in sorted(prediction["rows"].items()):
        if time.monotonic() - started > 900:
            raise TimeoutError("reconciliation exceeds 900 seconds")
        result = {
            "symbol": symbol,
            "prediction_state": row["state"],
            "state": "unscored",
        }
        if candidate is not None:
            result["candidate"] = {"state": "not_evaluated"}
        results.append(result)
        if row["state"] == "conflict":
            result["state"] = "conflict"
            continue
        if row["state"] != "success":
            continue
        values = row.get("values")
        value = values.get(upside_output) if isinstance(values, dict) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            result["state"] = "invalid_prediction"
            continue
        result["prediction"] = value
        if as_of_us < t + horizon * bars.MINUTE_US:
            result["state"] = "pending"
            continue
        other = future["rows"].get(symbol)
        if other and other["state"] == "conflict":
            result["state"] = "conflict"
            continue
        if not other or not other.get("cache_id"):
            result["state"] = "missing_source"
            continue
        result["sources"] = [
            {k: source.get(k) for k in ("cache_id", "source_sha256")}
            for source in (row, other)
        ]
        if future["decision_us"] > as_of_us:
            result["state"] = "source_after_as_of"
            continue
        try:
            origin = snapshot(row, symbol, t)
            frame = snapshot(other, symbol, future["decision_us"])
            frame = frame.filter(
                (pl.col("open_time").dt.epoch("us") >= t - bars.MINUTE_US)
                & (pl.col("open_time").dt.epoch("us") < t + horizon * bars.MINUTE_US)
            )
            if len(frame) != horizon + 1:
                result["state"] = "incomplete_horizon"
                continue
            if not origin.tail(1).equals(frame.head(1)):
                result["state"] = "conflict"
                continue
            labels = targets.path_targets(
                frame,
                pl.DataFrame({"symbol": [symbol], "decision_us": [t]}),
                targets.PathTargetSpec((horizon,), ()),
            ).row(0, named=True)
            realized = labels[f"up_{horizon}"]
            if realized is None or not math.isfinite(realized):
                result["state"] = "incomplete_horizon"
                continue
            error = realized - value
            result.update(
                state="observed",
                realized=realized,
                pinball_loss=max(quantile * error, (quantile - 1) * error),
                below_quantile=realized <= value,
                path={
                    k: v if v is not None and math.isfinite(v) else None
                    for k, v in labels.items()
                    if k not in {"symbol", "decision_us"}
                },
            )
        except FileNotFoundError:
            result["state"] = "missing_source"
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            pl.exceptions.PolarsError,
        ) as exc:
            result.update(state="invalid_source", error_type=type(exc).__name__)
        if candidate is not None and result["state"] == "observed":
            try:
                predicted = model_runtime.predict_bars(
                    models, record, origin, decision_us=t
                )[candidate_output]
                residual = result["realized"] - predicted
                result["candidate"] = {
                    "state": "scored",
                    "prediction": predicted,
                    "pinball_loss": max(quantile * residual, (quantile - 1) * residual),
                    "below_quantile": result["realized"] <= predicted,
                }
            except (ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
                result["candidate"] = {
                    "state": "refused",
                    "error_type": type(exc).__name__,
                }
    if time.monotonic() - started > 900:
        raise TimeoutError("reconciliation exceeds 900 seconds")
    observed = [r for r in results if r["state"] == "observed"]
    counts = dict(sorted(Counter(r["state"] for r in results).items()))
    summary = {
        "members": len(results),
        "observed": len(observed),
        "counts": counts,
        "mean_pinball_loss": sum(r["pinball_loss"] for r in observed) / len(observed)
        if observed
        else None,
        "empirical_quantile_coverage": sum(r["below_quantile"] for r in observed)
        / len(observed)
        if observed
        else None,
    }
    report = {
        "schema": "scan-outcomes-v1",
        "prediction_report_sha256": prediction_hash,
        "outcome_report_sha256": future_hash,
        "model_id": prediction["model_id"],
        "decision_us": t,
        "as_of_us": as_of_us,
        "target": {
            "declaration": "maximum_minute_close_rise_including_origin",
            "output": upside_output,
            "horizon_minutes": horizon,
            "quantile": quantile,
        },
        "implementation": [
            _artifacts.file_identity(Path(f))[0]
            for f in (__file__, bars.__file__, targets.__file__)
        ],
        "summary": summary,
        "rows": results,
    }
    if candidate is not None:
        matched = [r for r in observed if r["candidate"]["state"] == "scored"]
        n = len(matched)
        incumbent_loss = sum(r["pinball_loss"] for r in matched) / n if n else None
        candidate_loss = (
            sum(r["candidate"]["pinball_loss"] for r in matched) / n if n else None
        )
        comparison = {
            "matched": n,
            "candidate_states": dict(
                sorted(Counter(r["candidate"]["state"] for r in results).items())
            ),
            "incumbent_matched_loss": incumbent_loss,
            "candidate_matched_loss": candidate_loss,
            "candidate_quantile_coverage": sum(
                r["candidate"]["below_quantile"] for r in matched
            )
            / n
            if n
            else None,
            "loss_delta_candidate_minus_incumbent": candidate_loss - incumbent_loss
            if candidate_loss is not None and incumbent_loss is not None
            else None,
            "gain_vs_incumbent": 1 - candidate_loss / incumbent_loss
            if n and incumbent_loss
            else None,
        }
        report.update(
            schema="scan-outcomes-v2", candidate=candidate, comparison=comparison
        )
    payload = _artifacts.canonical_json(report)
    text = (
        "# Offline prediction outcome review\n\n"
        + f"Members: {len(results)}; observed: {len(observed)}.\n\n"
        + f"States: {counts}\n\nMean pinball loss: {summary['mean_pinball_loss']}\n\nEmpirical quantile coverage: {summary['empirical_quantile_coverage']}\n\n"
        + "Metrics cover observed outcomes only. Missing outcomes are not negative labels. One market origin is not independent temporal evidence. No training-baseline gain, feature drift, trading return or promotion decision is established. The upside target mapping is explicitly caller-declared.\n"
    ).encode()
    if candidate is not None:
        text += (
            "\n## Frozen candidate comparison\n\n"
            + json.dumps(report["comparison"], sort_keys=True)
            + "\n\nBoth losses use the same matched observed subset. Refusals remain in the ledger. Model trust and compatibility do not establish an independent training/test split. No fit or activation occurred.\n"
        ).encode()
    target = output / _artifacts.content_id(report).removeprefix("sha256:")

    def verify_existing(directory: Path) -> None:
        if (directory / "report.json").read_bytes() != payload or (
            directory / "report.md"
        ).read_bytes() != text:
            raise ValueError("existing outcome publication differs")

    output.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output, prefix="outcomes-"
    ) as staging:
        _artifacts.write_exclusive_bytes(staging / "report.json", payload)
        _artifacts.write_exclusive_bytes(staging / "report.md", text)
        _artifacts.adopt_directory(staging, target, verify_existing=verify_existing)
    LOG.info("Outcome report published: %s; states: %s", target, counts)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--upside-output", required=True)
    parser.add_argument("--as-of", required=True, help="Timezone-aware ISO timestamp")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--candidate-output")
    parser.add_argument("--trust-model", action="store_true")
    args = parser.parse_args()
    when = datetime.fromisoformat(args.as_of)
    if when.tzinfo is None:
        parser.error("--as-of must include a timezone")
    settings = project_settings(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    directory = reconcile(
        args.predictions,
        args.outcomes,
        snapshots=settings.data_root / "snapshots",
        as_of_us=int(when.timestamp() * 1_000_000),
        upside_output=args.upside_output,
        output=args.output or settings.data_root / "runs/outcomes",
        candidate_id=args.candidate_id,
        candidate_output=args.candidate_output,
        registry=settings.data_root / "models" if args.candidate_id else None,
        trusted=args.trust_model,
    )
    print(json.dumps({"report_directory": str(directory)}))


if __name__ == "__main__":
    main()
