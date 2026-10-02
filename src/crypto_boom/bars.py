"""Shared canonical minute bars: validation and bounded local loading, no models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from crypto_boom.market import TimeWindow

MINUTE_US = 60_000_000
SOURCE_COLUMNS = (
    "symbol",
    "open_time",
    "close_price",
    "high_price",
    "low_price",
    "quote_turnover",
    "taker_buy_quote_turnover",
    "trade_count",
    "quality_complete",
    "quality_state",
)


def admit_bars(source: pl.DataFrame) -> pl.DataFrame:
    """Canonical one-symbol, UTC minute data; bad quality breaks continuity."""
    source = source.select(SOURCE_COLUMNS).sort("open_time")
    if (
        source.is_empty()
        or source["symbol"].null_count()
        or source["symbol"].n_unique() != 1
    ):
        raise ValueError("expected nonempty, single-symbol bars")
    if source["open_time"].dtype != pl.Datetime("us", "UTC"):
        raise ValueError("open_time must be datetime[us, UTC]")
    times = source["open_time"].dt.epoch("us")
    if (
        times.null_count()
        or times.n_unique() != len(source)
        or (times % MINUTE_US != 0).any()
    ):
        raise ValueError("duplicate, missing or unaligned minute timestamp")
    numeric = (
        "close_price",
        "high_price",
        "low_price",
        "quote_turnover",
        "taker_buy_quote_turnover",
        "trade_count",
    )
    source = source.with_columns(pl.col(*numeric).cast(pl.Float64))
    valid = source.filter(
        pl.col("quality_complete") & (pl.col("quality_state") == "valid")
    )
    if valid.select(
        pl.any_horizontal(
            ~pl.col(*numeric).is_finite() | pl.col(*numeric).is_null(),
            pl.col("low_price") <= 0,
            pl.col("close_price") < pl.col("low_price"),
            pl.col("close_price") > pl.col("high_price"),
            pl.col("quote_turnover") < 0,
            pl.col("taker_buy_quote_turnover") < 0,
            pl.col("taker_buy_quote_turnover") > pl.col("quote_turnover") * (1 + 1e-10),
            pl.col("trade_count") < 0,
            pl.col("trade_count") != pl.col("trade_count").floor(),
        ).any()
    ).item():
        raise ValueError("invalid price, turnover or trade count")
    return source


def load_bar_files(
    paths: list[Path], *, start_us: int, end_us: int
) -> tuple[pl.DataFrame, list[dict]]:
    """Load an explicit UTC observation window; never fill gaps or drop duplicates.

    Caller owns warm-up and label horizons. Hashes identify input bytes, not their
    authenticity. All paths refer to canonical minute parquet (including the
    compatible columns emitted by storage.source).
    """
    return read_bar_window(paths, TimeWindow(start_us, end_us - MINUTE_US))


def read_bar_window(
    paths: list[Path], window: TimeWindow, *, columns: tuple[str, ...] = SOURCE_COLUMNS
) -> tuple[pl.DataFrame, list[dict]]:
    """Read minute-Parquet open times in [start, end), preserving order/duplicates.

    Data type is explicit. Instrument/partition selection happens before reading;
    resampling happens separately. No network, inference or hidden scale conversion.
    """
    if not {"symbol", "open_time"}.issubset(columns):
        raise ValueError("bar projection must retain symbol and open_time")
    paths = [path.resolve(strict=True) for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("empty or duplicate input paths")
    if len(paths) > 512 or sum(path.stat().st_size for path in paths) > 2_000_000_000:
        raise ValueError("bar input exceeds 512 files / 2 GB budget")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in paths) > 10_000_000:
        raise ValueError("bar input exceeds 10 million minute rows")
    frames, receipts = [], []
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        frame = pl.scan_parquet(path).select(columns)
        if frame.collect_schema()["open_time"] != pl.Datetime("us", "UTC"):
            raise ValueError("open_time must be datetime[us, UTC]")
        frames.append(frame)
        receipts.append({"name": path.name, "sha256": digest})
    # Preserve file/row order and duplicates, but materialize only the requested
    # window. Native timestamp predicates allow Parquet row-group pruning.
    bars = (
        pl.concat(frames)
        .filter(
            (
                pl.col("open_time")
                >= pl.lit(window.start_us).cast(pl.Datetime("us", "UTC"))
            )
            & (
                pl.col("open_time")
                < pl.lit(window.end_us).cast(pl.Datetime("us", "UTC"))
            )
        )
        .collect()
    )
    if bars.is_empty():
        raise ValueError("no bars in observation window")
    return bars, receipts


@dataclass(frozen=True)
class BarPeriod:
    """Fixed duration in minutes, aligned to the Unix UTC epoch."""

    minutes: int

    def __post_init__(self) -> None:
        if type(self.minutes) is not int or not 1 <= self.minutes <= 1440:
            raise ValueError("bar period must be 1..1440 minutes")


def aggregate_minute_bars(source: pl.DataFrame, period: BarPeriod) -> pl.DataFrame:
    """Aggregate genuine minute OHLC data, never infer opens from prior closes.

    Entire input must contain complete, valid, contiguous, aligned buckets.
    No partial edge buckets, imputation, upsampling, persistence or model policy.
    """
    if "open_price" not in source.columns or len(source) > 2_000_000:
        raise ValueError("aggregation requires genuine open_price and <=2M minute rows")
    opens = source.sort("open_time")["open_price"].cast(pl.Float64)
    frame = admit_bars(source).with_columns(opens)
    times = frame["open_time"].dt.epoch("us")
    step = period.minutes * MINUTE_US
    if (
        int(times[0]) % step
        or (int(times[-1]) + MINUTE_US) % step
        or not (times.diff().drop_nulls() == MINUTE_US).all()
        or not (frame["quality_complete"] & (frame["quality_state"] == "valid"))
        .fill_null(False)
        .all()
        or frame.select(
            (
                pl.col("open_price").is_null()
                | ~pl.col("open_price").is_finite()
                | (pl.col("open_price") < pl.col("low_price"))
                | (pl.col("open_price") > pl.col("high_price"))
            ).any()
        ).item()
    ):
        raise ValueError("aggregation needs complete aligned valid minute OHLC buckets")
    return (
        frame.with_columns(
            ((pl.col("open_time").dt.epoch("us") // step) * step)
            .cast(pl.Datetime("us", "UTC"))
            .alias("open_time")
        )
        .group_by("symbol", "open_time", maintain_order=True)
        .agg(
            pl.col("open_price").first(),
            pl.col("high_price").max(),
            pl.col("low_price").min(),
            pl.col("close_price").last(),
            pl.col("quote_turnover", "taker_buy_quote_turnover", "trade_count").sum(),
            pl.col("quality_complete", "quality_state").first(),
        )
    )
