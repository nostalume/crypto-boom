"""Reusable causal feature batches and source-quality receipts; no future labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from crypto_boom import _artifacts, bars, features
from crypto_boom.bars import MINUTE_US, admit_bars, load_bar_files
from crypto_boom.features import (
    DYNAMICS_FEATURES,
    FLOW_FEATURES,
    PATH_FEATURES,
    PRICE_FEATURES,
)

FEATURE_GROUPS = {
    "price": PRICE_FEATURES,
    "flow": FLOW_FEATURES,
    "path": PATH_FEATURES,
    "dynamics": DYNAMICS_FEATURES,
}
BATCH_FEATURES = tuple(name for group in FEATURE_GROUPS.values() for name in group)


def origin_observability(source: pl.DataFrame) -> pl.DataFrame:
    """Past-only selection evidence for every origin, including rejected origins.

    Admission is a technical minimum for the existing 1441-bar features, not a
    certificate of liquidity, historical trading eligibility or absence of noise.
    Bar age is measured in completed minutes, not exact last-trade timestamps.
    Existing models/population policies are not changed by this diagnostic API.
    """
    frame = admit_bars(source).with_columns(
        (pl.col("open_time").dt.epoch("us") + MINUTE_US).alias("decision_us"),
        (pl.col("quality_complete") & (pl.col("quality_state") == "valid")).alias(
            "valid"
        ),
    )
    frame = (
        frame.with_columns(
            (
                pl.col("valid")
                & (pl.col("trade_count") > 0)
                & (pl.col("quote_turnover") > 0)
            ).alias("price_observed"),
            (
                (pl.col("decision_us").diff() != MINUTE_US).fill_null(True)
                | ~pl.col("valid")
                | ~pl.col("valid").shift(1).fill_null(False)
            )
            .cum_sum()
            .alias("segment"),
        )
        .with_columns(
            pl.when(pl.col("valid"))
            .then(pl.col("decision_us").cum_count().over("segment"))
            .otherwise(0)
            .alias("contiguous_valid_minutes"),
            pl.when(pl.col("price_observed"))
            .then(pl.col("decision_us"))
            .otherwise(None)
            .forward_fill()
            .alias("last_traded_bar_us"),
        )
        .with_columns(
            ((pl.col("decision_us") - pl.col("last_traded_bar_us")) // MINUTE_US).alias(
                "minutes_since_traded_bar"
            ),
            pl.when(pl.col("contiguous_valid_minutes") >= 60)
            .then(pl.col("price_observed").cast(pl.Float64).rolling_mean(60))
            .otherwise(None)
            .alias("observed_fraction_60"),
            (pl.col("contiguous_valid_minutes") >= 1441).alias("feature_history_ready"),
        )
        .with_columns(
            (pl.col("feature_history_ready") & pl.col("price_observed")).alias(
                "origin_admissible"
            ),
            pl.when(~pl.col("valid"))
            .then(pl.lit("invalid_bar"))
            .when(~pl.col("feature_history_ready"))
            .then(pl.lit("insufficient_contiguous_history"))
            .when(~pl.col("price_observed"))
            .then(pl.lit("origin_without_trade"))
            .otherwise(pl.lit("admitted"))
            .alias("selection_reason"),
        )
    )
    return frame.select(
        "symbol",
        "decision_us",
        "valid",
        "price_observed",
        "contiguous_valid_minutes",
        "minutes_since_traded_bar",
        "observed_fraction_60",
        "feature_history_ready",
        "origin_admissible",
        "selection_reason",
    )


def read_feature_cache(path: Path) -> tuple[pl.DataFrame, dict]:
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "feature-cache-v1"
        or _artifacts.content_id(receipt["spec"]) != receipt["cache_id"]
    ):
        raise ValueError("invalid feature cache receipt")
    if (
        _artifacts.file_identity(path / "features.parquet")[0]
        != receipt["parquet_sha256"]
    ):
        raise ValueError("feature cache hash mismatch")
    frame = pl.read_parquet(path / "features.parquet")
    if len(frame) != receipt["rows"] or frame.select(
        pl.struct("symbol", "decision_us").n_unique()
    ).item() != len(frame):
        raise ValueError("invalid cached feature keys")
    return frame, receipt


def build_feature_cache(
    paths: list[Path],
    *,
    output_root: Path,
    step_minutes: int = 5,
    minimum_turnover: float = 1_000_000,
) -> tuple[Path, bool]:
    """Compute one symbol once; source/code/policy changes create a new cache."""
    if (
        type(step_minutes) is not int
        or not 1 <= step_minutes <= 60
        or not math.isfinite(minimum_turnover)
        or minimum_turnover < 0
    ):
        raise ValueError("invalid feature sampling policy")
    sources = [
        {"path": str(p.resolve()), "sha256": _artifacts.file_identity(p)[0]}
        for p in sorted(paths)
    ]
    spec: dict[str, object] = {
        "sources": sources,
        "step_minutes": step_minutes,
        "minimum_turnover": minimum_turnover,
        "feature_groups": FEATURE_GROUPS,
        "features_code": _artifacts.file_identity(Path(features.__file__))[0],
        "batch_code": _artifacts.file_identity(Path(__file__))[0],
        "bars_code": _artifacts.file_identity(Path(bars.__file__))[0],
    }
    cache_id = _artifacts.content_id(spec)
    target = output_root / cache_id.removeprefix("sha256:")
    if target.exists():
        read_feature_cache(target)
        return target, True
    source, _ = load_bar_files(paths, start_us=0, end_us=2**63 - 1)
    source = admit_bars(source)
    if len(source) > 2_000_000:
        raise ValueError("feature symbol exceeds two million minute rows")
    frames = []
    for segment in features.iter_feature_segments(source, rich=True):
        frames.append(
            segment.filter(
                (pl.col("decision_us") % (step_minutes * MINUTE_US) == 0)
                & (pl.col("turnover_1440") >= minimum_turnover)
                & pl.all_horizontal(pl.col(*BATCH_FEATURES).is_finite())
            ).select(
                "symbol", "decision_us", "close_price", "turnover_1440", *BATCH_FEATURES
            )
        )
    frame = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "decision_us": pl.Int64,
                "close_price": pl.Float64,
                "turnover_1440": pl.Float64,
                **{name: pl.Float64 for name in BATCH_FEATURES},
            }
        )
    )
    valid = source.filter(
        pl.col("quality_complete") & (pl.col("quality_state") == "valid")
    )
    times = source["open_time"].dt.epoch("us")
    quality = {
        "symbol": source["symbol"][0],
        "source_rows": len(source),
        "bad_quality_rows": len(source) - len(valid),
        "missing_minutes_inside_observed_span": int(
            (times[-1] - times[0]) // MINUTE_US + 1 - len(source)
        ),
        "zero_turnover_rows": int((source["quote_turnover"] == 0).sum()),
        "zero_trade_rows": int((source["trade_count"] == 0).sum()),
        "first_open_us": int(times[0]),
        "last_open_us": int(times[-1]),
        "eligible_origins": len(frame),
        "large_adjacent_return_rows": source.select(
            (
                (pl.col("close_price").pct_change().abs() > 0.5)
                & (pl.col("open_time").dt.epoch("us").diff() == MINUTE_US)
            ).sum()
        ).item(),
        "note": "Large jumps are flagged, not deleted; boundaries do not establish listing/delisting times. No imputation.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output_root, prefix="features-"
    ) as staging:
        frame.write_parquet(staging / "features.parquet")
        receipt: dict[str, object] = {
            "schema": "feature-cache-v1",
            "cache_id": cache_id,
            "spec": spec,
            "rows": len(frame),
            "quality": quality,
            "parquet_sha256": _artifacts.file_identity(staging / "features.parquet")[0],
        }
        _artifacts.write_exclusive_bytes(
            staging / "receipt.json", _artifacts.canonical_json(receipt)
        )

        def verify(path: Path) -> None:
            read_feature_cache(path)

        _artifacts.adopt_directory(staging, target, verify_existing=verify)
    return target, False
