"""Full observed-universe scan with one model/time snapshot and explicit coverage."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_boom import _artifacts, features, model_runtime
from crypto_boom import bars as bar_module
from crypto_boom.bars import MINUTE_US
from crypto_boom.config import ProjectSettings
from crypto_boom.market_data import ScanStopped, SpotSnapshotClient
from crypto_boom.model_runtime import load_active_model, predict_bars
from crypto_boom.product_pool import attach_product_coverage, collect_product_pool

logger = logging.getLogger(__name__)
MAX_INPUT_BYTES = 16_000_000


async def scan_market(
    settings: ProjectSettings, *, record_inputs: bool = False
) -> dict:
    """No caller symbol list or model path: resolve activation, enumerate full scope."""
    models, manifest, recipe = load_active_model(settings.data_root / "models")
    logger.info("Active model loaded: %s", manifest["model_id"])
    identity = manifest["identity"]
    step_us = identity["decision_step_minutes"] * MINUTE_US
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = settings.data_root / "reports" / run_id
    results: dict[str, dict] = {}
    captured: dict[str, np.ndarray] = {}
    input_bytes = 0
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=12)
    ) as session:
        client = SpotSnapshotClient(session, seconds=settings.timeout_seconds)
        symbols, scope, raw, server_ms = await client.universe()
        logger.info(
            "Universe acquired: %d members; workers=%d", len(symbols), settings.workers
        )
        product_pool = await collect_product_pool(
            session, raw, deadline=client.deadline
        )
        if product_pool["status"] != "complete":
            logger.warning(
                "Product metadata is %s; spot predictions can continue",
                product_pool["status"],
            )
        decision = server_ms * 1000 // step_us * step_us
        refused = set(scope["metadata_refusals"])
        pending = iter(symbols)
        stop_reason = None

        async def worker() -> None:
            nonlocal stop_reason, input_bytes
            for symbol in pending:
                if symbol in refused:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "metadata_error",
                        "reason": "unsupported instrument metadata",
                    }
                    continue
                if client.stopped or time.monotonic() >= client.deadline:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "not_attempted",
                        "reason": stop_reason or "run budget exhausted",
                    }
                    continue
                try:
                    bars, receipt = await client.bars(
                        symbol,
                        decision_us=decision,
                        history_minutes=recipe.history_minutes,
                        cache=settings.data_root / "snapshots",
                    )
                    evidence = None
                    if record_inputs:
                        prediction = predict_bars(
                            models,
                            manifest,
                            bars,
                            decision_us=decision,
                            include_inputs=True,
                        )
                        values, evidence = prediction["values"], prediction["inputs"]
                    else:
                        values = predict_bars(
                            models, manifest, bars, decision_us=decision
                        )
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "success",
                        "values": values,
                        **receipt,
                    }
                    if evidence is not None:
                        vector = evidence["vector"]
                        reason = evidence["reason"]
                        if (
                            not isinstance(receipt.get("cache_id"), str)
                            or not _artifacts.is_sha256("sha256:" + receipt["cache_id"])
                            or not isinstance(receipt.get("source_sha256"), str)
                            or not _artifacts.is_sha256(receipt["source_sha256"])
                        ):
                            reason = "missing_source_binding"
                        elif (
                            vector is not None
                            and input_bytes + vector.nbytes > MAX_INPUT_BYTES
                        ):
                            reason = "input_budget_exceeded"
                        if vector is not None and reason is None:
                            captured[symbol] = vector
                            input_bytes += vector.nbytes
                        results[symbol]["input_evidence"] = {
                            "state": "available"
                            if symbol in captured
                            else "unavailable",
                            "reason": reason,
                        }
                except ScanStopped as exc:
                    client.stopped = True
                    stop_reason = str(exc)
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "not_completed",
                        "reason": str(exc) or type(exc).__name__,
                    }
                except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "fetch_error",
                        "reason": str(exc) or type(exc).__name__,
                    }
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "data_or_prediction_error",
                        "reason": str(exc) or type(exc).__name__,
                    }
                if len(results) % 50 == 0:
                    logger.info(
                        "Scan progress: %d/%d members processed",
                        len(results),
                        len(symbols),
                    )

        await asyncio.gather(*(worker() for _ in range(settings.workers)))
        elapsed = time.monotonic() - client.started
        requests = client.requests
    if set(results) != set(symbols):
        raise AssertionError("universe coverage ledger incomplete")
    rows = [results[s] for s in symbols]
    attach_product_coverage(product_pool, rows)
    counts = dict(Counter(r["state"] for r in rows))
    status = "complete" if counts.get("success", 0) == len(symbols) else "partial"
    logger.log(
        logging.INFO if status == "complete" else logging.WARNING,
        "Scan %s: %d/%d successful in %.1fs; states=%s",
        status,
        counts.get("success", 0),
        len(symbols),
        elapsed,
        counts,
    )
    rank_by = identity["rank_by"]
    ranking = sorted(
        (r for r in rows if r["state"] == "success"),
        key=lambda r: (-r["values"][rank_by], r["symbol"]),
    )
    result = {
        "schema": "market-scan-v2",
        "product_pool": product_pool,
        "product_metadata_status": product_pool["status"],
        "run_id": run_id,
        "status": status,
        "scope": scope,
        "model_id": manifest["model_id"],
        "model_contract": identity,
        "decision_us": decision,
        "exchange_snapshot_ms": server_ms,
        "elapsed_seconds": elapsed,
        "http_requests": requests + product_pool["http_requests"],
        "estimated_decision_age_seconds_at_finish": (server_ms * 1000 - decision) / 1e6
        + elapsed,
        "newer_decision_may_be_available": server_ms * 1000 + elapsed * 1e6
        >= decision + step_us,
        "eligible_symbols": len(symbols),
        "counts": counts,
        "stop_reason": stop_reason,
        "rank_by": rank_by,
        "rows": rows,
        "report_directory": str(output.resolve()),
    }
    when = datetime.fromtimestamp(decision / 1e6, UTC).isoformat()
    description = next(o for o in identity["outputs"] if o["name"] == rank_by)
    lines = [
        "# Market-wide prediction report",
        "",
        f"Status: **{status}**",
        f"Scope: {scope['scope']}",
        f"Shared prediction origin UTC: {when}",
        f"Model ID: `{manifest['model_id']}`",
        f"Decision step: {identity['decision_step_minutes']} minutes; history: {recipe.history_minutes} minutes",
        f"Members: {len(symbols)}; successful: {counts.get('success', 0)}; all other members remain in the status ledger",
        f"Elapsed: {elapsed:.1f} seconds; market requests: {requests}; product metadata requests: {product_pool['http_requests']}",
        f"Product metadata status: {product_pool['status']}; independent of prediction coverage",
        "Source: Binance Spot. Other products are ticker candidates, not perpetual-return forecasts.",
        "See products.csv for full product coverage. Metadata failure does not mean unlisted.",
        f"Newer origin may exist at completion: {result['newer_decision_may_be_available']}",
        "",
        "## Ranking preview (full results: predictions.csv / report.json)",
        f"Ranking field: {rank_by}; target: {description['label'] if description['label'].isascii() else rank_by}; horizon: {description['horizon_minutes']} minutes; unit: {description['unit']}",
        "",
        "| Symbol | Value | Product candidates (non-native mappings unverified) |",
        "|---|---:|---|",
    ]
    for row in ranking[:30]:
        value = row["values"][rank_by]
        formatted = (
            f"{value:.2%}"
            if description["unit"] in ("fraction", "return_fraction")
            else f"{value:.4f}"
        )
        candidates = "; ".join(
            f"{p['source']}:{p['instrument_id']} ({p['mapping_status']})"
            for p in row["product_candidates"]
        )
        lines.append(f"| {row['symbol']} | {formatted} | {candidates} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "Scope is the observed venue, market and quote universe, not all exchanges.",
            "All members share one completed boundary. Later minutes are excluded; newer-origin availability is flagged.",
            "Ranking is not trading advice. Quantiles are not rise probabilities; absent outputs do not mean zero risk. Experimental model.",
            "Listed assets may be outside training coverage. New assets, insufficient history, failures, rate limits and unattempted members are not negatives.",
            "",
            "## Status counts",
            json.dumps(counts, ensure_ascii=False),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output.parent, prefix="scan-"
    ) as staging:
        (staging / "exchange-info.json").write_bytes(raw)
        result["universe_sha256"] = _artifacts.file_identity(
            staging / "exchange-info.json"
        )[0]
        if record_inputs:
            result["schema"] = "market-scan-v3"
            evidence_record = {
                "schema": "inference-inputs-v1",
                "state": "unavailable",
                "rows": 0,
                "successful_predictions": counts.get("success", 0),
                "reference_kind": "deployment_observation_not_training_reference",
                "capture_code_sha256": _artifacts.file_identity(Path(__file__))[0],
                "model_id": manifest["model_id"],
                "recipe_sha256": _artifacts.content_id(identity["recipe"]),
                "implementation": {
                    name: _artifacts.file_identity(Path(module.__file__))[0]
                    for name, module in (
                        ("runtime", model_runtime),
                        ("features", features),
                        ("bars", bar_module),
                    )
                },
            }
            result["input_evidence"] = evidence_record
            if captured:
                sidecar = staging / "inputs.parquet"
                try:
                    members = sorted(captured)
                    matrix = np.stack([captured[s] for s in members])
                    table = pa.table(
                        {
                            "symbol": members,
                            "decision_us": [decision] * len(members),
                            "cache_id": [results[s]["cache_id"] for s in members],
                            "source_sha256": [
                                results[s]["source_sha256"] for s in members
                            ],
                            "features": pa.FixedSizeListArray.from_arrays(
                                pa.array(matrix.reshape(-1)), matrix.shape[1]
                            ),
                        }
                    )
                    pq.write_table(table, sidecar, compression="zstd")
                    digest, size = _artifacts.file_identity(sidecar)
                    evidence_record.update(
                        state="complete"
                        if len(members) == counts.get("success", 0)
                        else "partial",
                        rows=len(members),
                        file="inputs.parquet",
                        sha256=digest,
                        bytes=size,
                        dtype=str(matrix.dtype),
                        feature_count=matrix.shape[1],
                    )
                except (OSError, ValueError, pa.ArrowException) as exc:
                    sidecar.unlink(missing_ok=True)
                    evidence_record.update(
                        reason="publication_failed", error_type=type(exc).__name__
                    )
                    for symbol in captured:
                        results[symbol]["input_evidence"] = {
                            "state": "unavailable",
                            "reason": "publication_failed",
                        }
                    logger.warning(
                        "Input evidence unavailable; predictions retained: %s",
                        type(exc).__name__,
                    )
            else:
                evidence_record["reason"] = "no_captured_inputs"
            lines.extend(
                [
                    "",
                    "## Input evidence",
                    f"State: {evidence_record['state']}; captured: {evidence_record['rows']}/{evidence_record['successful_predictions']} successful predictions.",
                    "Deployment observations are not a training reference or a drift diagnosis.",
                ]
            )
            logger.info(
                "Input evidence: %s (%s rows)",
                evidence_record["state"],
                evidence_record["rows"],
            )
        (staging / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (staging / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "symbol",
                    "state",
                    "reason",
                    "prediction_data_source",
                    "product_candidates",
                    *models,
                ],
            )
            writer.writeheader()
            for row in [*ranking, *(r for r in rows if r["state"] != "success")]:
                writer.writerow(
                    {
                        "symbol": row["symbol"],
                        "state": row["state"],
                        "reason": row.get("reason", ""),
                        "prediction_data_source": row["prediction_data_source"],
                        "product_candidates": json.dumps(
                            row["product_candidates"], ensure_ascii=False
                        ),
                        **row.get("values", {}),
                    }
                )
        with (staging / "products.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            fields = [
                "source",
                "venue",
                "product",
                "instrument_id",
                "base_ticker",
                "quote",
                "asset_category",
                "contract_value",
                "contract_value_ccy",
                "mapping_status",
                "binance_spot_candidate",
                "binance_spot_prediction_state",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(product_pool["products"])
        staging.rename(output)
    logger.info("Report published: %s", output)
    return result
