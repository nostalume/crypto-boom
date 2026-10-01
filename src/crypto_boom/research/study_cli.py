"""Thin command adapters for reusable sample, feature and path-study interfaces."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _utc_us(value: str) -> int:
    return (
        int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        * 1_000_000
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Experimental reusable path research; no orders"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--start-month", required=True)
    acquire.add_argument("--end-month", required=True)
    acquire.add_argument("--output", type=Path)
    acquire.add_argument("--config", type=Path)
    acquire.add_argument("--count", type=int, default=24)
    acquire.add_argument("--seed", default="forward-path-v1")
    acquire.add_argument("--reuse-corpus", type=Path, action="append", default=[])
    acquire.add_argument("--exclude-symbol", action="append", default=[])
    acquire.add_argument(
        "--quarantine",
        action="append",
        default=[],
        help="SYMBOL:YYYY-MM:reason; excluded partition remains in ledger",
    )
    build = commands.add_parser("build")
    build.add_argument("--pool", required=True, type=Path)
    build.add_argument("--output", type=Path)
    build.add_argument("--config", type=Path)
    build.add_argument("--step-minutes", type=int, default=5)
    build.add_argument("--minimum-turnover", type=float, default=1_000_000)
    build.add_argument("--horizons", type=int, nargs="+", default=[120, 360])
    build.add_argument(
        "--amplitudes",
        type=float,
        nargs="*",
        default=[],
        help="optional amplitude queries; no fixed event thresholds by default",
    )
    audit = commands.add_parser("audit")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--start-month", required=True)
    audit.add_argument("--end-month", required=True)
    audit.add_argument("--output", type=Path, required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--dataset", type=Path, required=True)
    screen.add_argument(
        "--train-end", required=True, help="exclusive UTC date, YYYY-MM-DD"
    )
    screen.add_argument(
        "--selection-end", required=True, help="exclusive UTC date, YYYY-MM-DD"
    )
    screen.add_argument("--horizon", type=int, default=360)
    screen.add_argument(
        "--model",
        type=Path,
        required=True,
        help="new directory; never overwrite a prior study",
    )
    rolling = commands.add_parser("rolling")
    rolling.add_argument("--dataset", type=Path, required=True)
    rolling.add_argument("--output", type=Path, required=True)
    rolling.add_argument(
        "--window",
        nargs=3,
        action="append",
        type=_utc_us,
        required=True,
        metavar=("TRAIN_END", "SELECTION_END", "TEST_END"),
    )
    rolling.add_argument("--horizon", type=int, default=360)
    latest = commands.add_parser("latest")
    latest.add_argument("--model", type=Path, required=True)
    latest.add_argument("--symbol", required=True)
    latest.add_argument("--trust-model", action="store_true")
    latest.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("acquire", "build"):
        from crypto_boom.config import project_settings

        project = project_settings(args.config)
        if args.output is None:
            args.output = project.data_root / "runs" / f"{args.command}-{uuid4().hex}"
    if args.command == "acquire":
        from crypto_boom.history.availability import MonthlyAvailabilityRequest
        from crypto_boom.sample_pool import acquire_sample_pool

        result = asyncio.run(
            acquire_sample_pool(
                MonthlyAvailabilityRequest(
                    datetime.strptime(args.start_month, "%Y-%m").date(),
                    datetime.strptime(args.end_month, "%Y-%m").date(),
                ),
                output=args.output,
                count=args.count,
                seed=args.seed,
                reuse_corpora=tuple(args.reuse_corpus),
                data_root=project.data_root,
                excluded_symbols=tuple(args.exclude_symbol),
                quarantined_partitions=tuple(
                    tuple(s.split(":", 2)) for s in args.quarantine
                ),
            )
        )
        result = {
            k: v for k, v in result.items() if k not in ("selection", "partitions")
        }
    elif args.command == "audit":
        from crypto_boom import _artifacts
        from crypto_boom.history.availability import MonthlyAvailabilityRequest
        from crypto_boom.sample_pool import audit_local_corpus

        if args.output.exists():
            raise ValueError("audit receipt exists; do not overwrite prior evidence")
        result = audit_local_corpus(
            args.corpus,
            MonthlyAvailabilityRequest(
                datetime.strptime(args.start_month, "%Y-%m").date(),
                datetime.strptime(args.end_month, "%Y-%m").date(),
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _artifacts.write_exclusive_bytes(args.output, _artifacts.canonical_json(result))
        result = {
            k: v
            for k, v in result.items()
            if k not in ("partitions", "locally_absent_months")
        }
    elif args.command == "build":
        from crypto_boom.research.path_dataset import build_path_dataset
        from crypto_boom.research.path_targets import PathTargetSpec

        result = build_path_dataset(
            args.pool,
            output_root=args.output,
            spec=PathTargetSpec(tuple(args.horizons), tuple(args.amplitudes)),
            step_minutes=args.step_minutes,
            minimum_turnover=args.minimum_turnover,
            data_root=project.data_root,
        )
    elif args.command == "screen":
        from crypto_boom.feature_batch import read_feature_cache
        from crypto_boom.research.path_dataset import load_path_dataset
        from crypto_boom.research.path_model import save_path_model
        from crypto_boom.research.path_screen import screen_path_features

        if args.model.exists():
            raise ValueError(
                "model directory already exists; no repeated fit or overwrite"
            )
        rows, dataset = load_path_dataset(args.dataset)
        policy = {
            read_feature_cache(Path(b["features"]))[1]["spec"]["minimum_turnover"]
            for b in dataset["batches"]
        }
        if len(policy) != 1:
            raise ValueError("mixed population policies")
        models, metadata = screen_path_features(
            rows,
            train_end_us=_utc_us(args.train_end),
            selection_end_us=_utc_us(args.selection_end),
            horizon=args.horizon,
        )
        metadata.update(dataset_id=dataset["dataset_id"], minimum_turnover=policy.pop())
        result = save_path_model(args.model, models, metadata)
    elif args.command == "rolling":
        from crypto_boom.research.path_screen import rolling_path_audit

        result = rolling_path_audit(
            args.dataset,
            windows=tuple(tuple(w) for w in args.window),
            output=args.output,
            horizon=args.horizon,
        )
    else:
        from crypto_boom.latest_market import fetch_latest
        from crypto_boom.research.path_model import (
            forecast_path,
            load_path_model,
            render_path_forecast,
        )

        models, metadata = load_path_model(args.model, trusted=args.trust_model)
        source, server_ms = fetch_latest(args.symbol, minutes=1441)
        result = forecast_path(models, metadata, source, server_ms)
        if not args.json:
            print(render_path_forecast(result))
            return 0
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
