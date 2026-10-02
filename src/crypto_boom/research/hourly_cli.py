"""Model-specific research script: python -m crypto_boom.research.hourly_cli."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def generate_prediction_report(
    model_directory: Path,
    output: Path,
    *,
    symbol: str | None = None,
    bar_paths: list[Path] | None = None,
    trusted: bool = False,
) -> dict:
    """Fresh directory only; single symbol, no silent network fallback in replay."""
    from crypto_boom import _artifacts
    from crypto_boom.bars import MINUTE_US, load_bar_files
    from crypto_boom.research.hourly import forecast_hourly, load_hourly_model

    if (symbol is None) == (not bar_paths):
        raise ValueError("choose exactly one of live symbol or replay bar paths")
    if output.exists():
        raise FileExistsError(output)
    model, metadata = load_hourly_model(model_directory, trusted=trusted)
    server_ms = None
    if symbol is not None:
        from crypto_boom.latest_market import fetch_latest

        source, server_ms = fetch_latest(symbol, minutes=1501)
        if (
            source["open_time"].dt.epoch("us")[-1] + MINUTE_US
            != server_ms // 60000 * MINUTE_US
        ):
            raise ValueError("latest source is stale")
        receipts = [{"source": "Binance public Spot minute endpoint"}]
    else:
        source, receipts = load_bar_files(bar_paths or [], start_us=0, end_us=2**63 - 1)
    result = forecast_hourly(model, metadata, source)
    if not re.fullmatch(r"[A-Z0-9]{3,32}", result["symbol"]):
        raise ValueError("unsupported report symbol")
    result.update(
        mode="live_hourly" if symbol is not None else "historical_replay",
        generated_at_utc=datetime.now(UTC).isoformat(),
        server_ms=server_ms,
        provenance=metadata["provenance"],
        sources=receipts,
    )
    decision = datetime.fromtimestamp(
        result["decision_us"] / 1_000_000, UTC
    ).isoformat()
    text = (
        f"# {result['symbol']} experimental prediction report\n\n"
        f"- Mode:{result['mode']}\n- Prediction origin (UTC):{decision}\n"
        f"- Lag behind latest completed minute:{result['hour_lag_minutes']} minutes (preserving training-time hourly alignment)\n"
        f"- Six-hour maximum upside P90:**{result['upside_p90']:.2%}**\n"
        f"- Traded-minute fraction over 24 hours:{result['observed_fraction_24h']:.2%}\n"
        f"- Model identity:`{result['model_sha256']}`\n\n"
        "## Interpretation\n\n"
        f"{result['interpretation']}\n\n"
        "Origin price is the last minute close before the hour. All hourly buckets are complete; "
        "this is not a new forecast for the current minute. Cross-symbol/time validity is not guaranteed.\n\n"
        "Retention, downside and terminal return are not supplied and must not be treated as zero risk. "
        "Experiments do not establish reliable sustained-rise identification. This is not an order or after-cost return forecast.\n\n"
        f"Training provenance:`{metadata['provenance'].get('kind', 'unknown')}`;"
        "new_fit_not_evaluated means an unevaluated fit. Historical replay is not live data.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output.parent, prefix="report-"
    ) as staging:
        source.sort("open_time").tail(1501).write_parquet(staging / "source.parquet")
        result["source_snapshot_sha256"] = _artifacts.file_identity(
            staging / "source.parquet"
        )[0]
        (staging / "prediction.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (staging / "report.md").write_text(text, encoding="utf-8")
        staging.rename(output)
    return result


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
    publish.add_argument("--registry", type=Path)
    publish.add_argument("--config", type=Path)
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
        result = generate_prediction_report(
            args.model,
            args.output,
            symbol=args.symbol,
            bar_paths=args.bars,
            trusted=args.trust_model,
        )
    else:
        from crypto_boom.model_runtime import activate_model, publish_model

        if args.registry is None:
            from crypto_boom.config import project_settings

            args.registry = project_settings(args.config).data_root / "models"
        model, metadata = load_hourly_model(args.model, trusted=args.trust_model)
        result = publish_model(
            args.registry,
            {"upside_p90": model},
            recipe=RECIPE,
            decision_step_minutes=60,
            outputs=[
                {
                    "name": "upside_p90",
                    "label": "Maximum upside P90",
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
