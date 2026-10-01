"""Model-specific research script: python -m crypto_boom.research.hourly_cli."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-hourly")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--train-end", required=True)
    train.add_argument("--model", type=Path, required=True)
    export = commands.add_parser("export-hourly")
    export.add_argument("--study", type=Path, required=True)
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--trust-model", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--model", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbol")
    source.add_argument("--bars", type=Path, action="append")
    report.add_argument("--trust-model", action="store_true")
    publish = commands.add_parser("publish")
    publish.add_argument("--model", type=Path, required=True)
    publish.add_argument("--registry", type=Path, default=Path("data/models"))
    publish.add_argument("--trust-model", action="store_true")
    publish.add_argument("--activate", action="store_true")
    args = parser.parse_args(argv)
    from crypto_boom.research.hourly import (
        RECIPE,
        export_hourly_study,
        fit_hourly_dataset,
        load_hourly_model,
    )

    if args.command == "train-hourly":
        end = (
            int(
                datetime.strptime(args.train_end, "%Y-%m-%d")
                .replace(tzinfo=UTC)
                .timestamp()
            )
            * 1_000_000
        )
        result = fit_hourly_dataset(
            args.dataset, train_end_us=end, destination=args.model
        )
    elif args.command == "export-hourly":
        result = export_hourly_study(args.study, args.model, trusted=args.trust_model)
    elif args.command == "report":
        from crypto_boom.research.prediction_report import generate_prediction_report

        result = generate_prediction_report(
            args.model,
            args.output,
            symbol=args.symbol,
            bar_paths=args.bars,
            trusted=args.trust_model,
        )
    else:
        from crypto_boom.model_runtime import activate_model, publish_model

        model, metadata = load_hourly_model(args.model, trusted=args.trust_model)
        result = publish_model(
            args.registry,
            {"upside_p90": model},
            recipe=RECIPE,
            decision_step_minutes=60,
            outputs=[
                {
                    "name": "upside_p90",
                    "label": "最大上涨空间 P90",
                    "horizon_minutes": 360,
                    "statistic": "quantile",
                    "quantile": 0.9,
                    "unit": "return_fraction",
                }
            ],
            rank_by="upside_p90",
            provenance=metadata["provenance"],
        )
        if args.activate:
            activate_model(args.registry, result["model_id"], trusted=args.trust_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
