"""Prediction CLI and on-demand dispatch to historical operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import aiohttp


def _add_prediction_commands(subparsers: argparse._SubParsersAction) -> None:
    scan = subparsers.add_parser(
        "scan",
        help="scan the entire observed Spot USDT universe using the active model",
    )
    scan.add_argument(
        "--config",
        type=Path,
        help="project crypto-boom.toml; no symbol/model-path arguments",
    )
    scan.add_argument(
        "--record-inputs",
        action="store_true",
        help="save optional exact inference inputs with the report",
    )
    models = subparsers.add_parser(
        "model", help="list or explicitly activate local model IDs"
    )
    models.add_argument("action", choices=("list", "activate"))
    models.add_argument("--id", dest="model_id")
    models.add_argument("--trust-model", action="store_true")
    models.add_argument("--config", type=Path)


def _emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Keep prediction independent of acquisition/qualification command imports."""
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not argv or argv[0] not in ("scan", "model"):
        from crypto_boom import _operations_cli

        return _operations_cli.main(argv)
    parser = argparse.ArgumentParser(prog="crypto-boom")
    _add_prediction_commands(parser.add_subparsers(dest="command", required=True))
    arguments = parser.parse_args(argv)
    from crypto_boom.config import project_settings

    try:
        settings = project_settings(arguments.config)
        logging.getLogger(__name__).info("Starting %s", arguments.command)
        if arguments.command == "scan":
            from crypto_boom.market_scan import scan_market

            result = asyncio.run(
                scan_market(settings, record_inputs=True)
                if arguments.record_inputs
                else scan_market(settings)
            )
            _emit_json(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "model_id",
                        "eligible_symbols",
                        "product_metadata_status",
                        "counts",
                        "report_directory",
                        "input_evidence",
                    )
                    if k in result
                }
            )
            return 0 if result["status"] == "complete" else 2
        registry = settings.data_root / "models"
        if arguments.action == "activate":
            from crypto_boom.model_runtime import activate_model

            if not arguments.model_id:
                raise ValueError("model activate requires --id")
            activate_model(registry, arguments.model_id, trusted=arguments.trust_model)
            _emit_json({"active_model_id": arguments.model_id})
        else:
            _emit_json(
                {
                    "models": sorted(
                        p.parent.name for p in registry.glob("*/manifest.json")
                    ),
                    "active": json.loads((registry / "active.json").read_text())
                    if (registry / "active.json").exists()
                    else None,
                }
            )
        return 0
    except (ValueError, OSError, aiohttp.ClientError, RuntimeError) as exc:
        print(f"Scan/model error: {exc}", file=sys.stderr)
        return 1
