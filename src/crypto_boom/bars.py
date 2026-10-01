"""Shared canonical minute bars: validation and bounded local loading, no models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

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
    if start_us >= end_us:
        raise ValueError("require start < end for bar observations")
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
        frame = pl.scan_parquet(path).select(SOURCE_COLUMNS)
        if frame.collect_schema()["open_time"] != pl.Datetime("us", "UTC"):
            raise ValueError("open_time must be datetime[us, UTC]")
        frames.append(frame)
        receipts.append({"name": path.name, "sha256": digest})
    # Preserve file/row order and duplicates, but materialize only the requested
    # window. Native timestamp predicates allow Parquet row-group pruning.
    bars = (
        pl.concat(frames)
        .filter(
            (pl.col("open_time") >= pl.lit(start_us).cast(pl.Datetime("us", "UTC")))
            & (
                pl.col("open_time")
                < pl.lit(end_us - MINUTE_US).cast(pl.Datetime("us", "UTC"))
            )
        )
        .collect()
    )
    if bars.is_empty():
        raise ValueError("no bars in observation window")
    return bars, receipts


def decode_minute_page(
    rows: object, *, symbol: str, page_start: int, count: int
) -> list[dict]:
    """Validate exact completed Binance minute page; no sorting, filling or repair."""
    records = []
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("missing minute history or unknown/new symbol")
    for index, bar in enumerate(rows):
        expected = page_start + index * 60_000
        if (
            not isinstance(bar, list)
            or len(bar) != 12
            or type(bar[0]) is not int
            or bar[0] != expected
            or type(bar[6]) is not int
            or bar[6] != expected + 59_999
            or type(bar[8]) is not int
        ):
            raise ValueError(
                "invalid, duplicate, out-of-order or unfinished minute bar"
            )
        records.append(
            {
                "symbol": symbol,
                "open_time": expected * 1000,
                "close_price": float(bar[4]),
                "high_price": float(bar[2]),
                "low_price": float(bar[3]),
                "quote_turnover": float(bar[7]),
                "taker_buy_quote_turnover": float(bar[10]),
                "trade_count": bar[8],
                "quality_complete": True,
                "quality_state": "valid",
            }
        )
    return records
