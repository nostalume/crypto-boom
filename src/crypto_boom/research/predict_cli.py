"""Experimental training and latest-data prediction (prediction extra)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError


def _utc_us(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(
            "dates require an explicit timezone, e.g. 2026-01-01T00:00:00Z"
        )
    return int(parsed.astimezone(UTC).timestamp() * 1_000_000)


def render_text(result: dict) -> str:
    lines = [
        f"{result['symbol']} | Origin {result['origin_utc']} | Close {result['origin_close']:.8g}",
        f"Origin age: {result['origin_age_seconds']:.1f} seconds; model is experimental, not forward-validated",
        "Target: maximum minute-close upside from the origin over the horizon, not terminal return",
    ]
    for entry in result["forecasts"]:
        lines.append(
            f"  {entry['horizon_minutes'] / 60:g} hours / "
            f"P{entry['quantile'] * 100:g} quantile estimate:"
            f"+{entry['max_close_rise_fraction'] * 100:.2f}%"
        )
    lines += [
        "Quantiles are not hit probabilities: P90 does not mean a 90% chance of that rise.",
        "All outputs beat simple baselines on historical validation: "
        + (
            "yes (not proof of future validity)"
            if result["all_coordinates_beat_baseline"]
            else "no"
        ),
        result["warning"],
        f"Model SHA256:{result['model_sha256']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="fit experimental path quantiles from canonical minute parquet"
    )
    train.add_argument("--bars", type=Path, nargs="+", required=True)
    train.add_argument(
        "--start", required=True, help="first origin UTC (history may precede it)"
    )
    train.add_argument(
        "--split",
        required=True,
        help="validation starts here, with maximum-horizon train purge",
    )
    train.add_argument(
        "--end", required=True, help="exclusive data/label observation cutoff UTC"
    )
    train.add_argument(
        "--model",
        type=Path,
        required=True,
        help="new output directory, never overwritten",
    )
    train.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[120, 360],
        help="increasing horizon minutes, max 4320",
    )
    train.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.9])
    replay = commands.add_parser(
        "backtest", help="evaluate a frozen model on later historical bars; no fit"
    )
    replay.add_argument("--bars", type=Path, nargs="+", required=True)
    replay.add_argument("--start", required=True)
    replay.add_argument("--end", required=True)
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--trust-model", action="store_true")
    latest = commands.add_parser(
        "latest", help="fetch latest completed bars and predict one trained symbol"
    )
    latest.add_argument("--symbol", required=True)
    latest.add_argument("--model", type=Path, required=True)
    latest.add_argument(
        "--trust-model",
        action="store_true",
        help="allow code-executing joblib load; only your own artifacts",
    )
    latest.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    try:
        import polars as pl

        from crypto_boom.bars import MINUTE_US, load_bar_files
        from crypto_boom.research.forward import (
            HISTORY_MINUTES,
            ForecastGrid,
            forecast,
            load_model,
            save_model,
        )

        if args.command in ("train", "backtest"):
            from crypto_boom.research.forward_training import (
                backtest,
                fit_forward,
                training_rows,
            )

            start, end = (_utc_us(value) for value in (args.start, args.end))
            if start >= end:
                raise ValueError("require start < end")
            if args.command == "train":
                grid = ForecastGrid(tuple(args.horizons), tuple(args.quantiles))
                split = _utc_us(args.split)
                if not start < split < end:
                    raise ValueError("require start < split < end")
                if args.model.exists():
                    raise ValueError(
                        "model directory already exists; choose a new path"
                    )
            else:
                models, metadata = load_model(args.model, trusted=args.trust_model)
                grid = ForecastGrid.from_metadata(metadata)
            bars, inputs = load_bar_files(
                args.bars, start_us=start - HISTORY_MINUTES * MINUTE_US, end_us=end
            )
            rows = pl.concat(
                [training_rows(frame, grid) for frame in bars.partition_by("symbol")]
            ).filter(pl.col("decision_us") >= start)
            provenance = {
                "inputs": inputs,
                "requested_start_us": start,
                "observation_end_exclusive_us": end,
            }
            if args.command == "train":
                models, metadata = fit_forward(rows, split, grid)
                metadata.update(provenance)
                result = save_model(args.model, models, metadata)
            else:
                result = backtest(rows, models, metadata)
                result.update(provenance)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "latest":
            from crypto_boom.latest_market import fetch_latest

            models, metadata = load_model(args.model, trusted=args.trust_model)
            if args.symbol not in metadata["symbols"]:
                raise ValueError("symbol outside fitted population; retrain explicitly")
            bars, server_ms = fetch_latest(args.symbol, minutes=HISTORY_MINUTES)
            result = forecast(bars, server_ms, models, metadata)
            result["source"] = "Binance Spot /api/v3/klines (closed 1m, UTC)"
            print(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                if args.format == "json"
                else render_text(result)
            )
    except (ValueError, OSError, URLError, ImportError) as exc:
        print(
            json.dumps({"status": "refused", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    except pl.exceptions.PolarsError as exc:
        print(
            json.dumps({"status": "refused", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
