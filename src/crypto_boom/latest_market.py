"""Bounded, read-only Binance Spot minute-history acquisition for forecasting."""

from __future__ import annotations

import json
import re
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import polars as pl

from crypto_boom.bars import admit_bars

BASE_URL = "https://data-api.binance.vision"


def _get_json(path: str, params: dict) -> object:
    # At most four requests per one-symbol run; fail rather than retry on 429/418.
    with urlopen(f"{BASE_URL}{path}?{urlencode(params)}", timeout=12) as response:
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError("market response exceeds 2 MB")
    return json.loads(body)


def fetch_latest(symbol: str, *, minutes: int) -> tuple[pl.DataFrame, int]:
    """Fetch 1-2000 closed minutes; callers own quote and lookback policy."""
    if not re.fullmatch(r"[A-Z0-9]{3,32}", symbol):
        raise ValueError("expected uppercase ASCII Binance Spot symbol")
    if type(minutes) is not int or not 1 <= minutes <= 2000:
        raise ValueError("minute history must be an integer in [1, 2000]")
    started = time.monotonic()
    clock = _get_json("/api/v3/time", {})
    if not isinstance(clock, dict) or type(clock.get("serverTime")) is not int:
        raise ValueError("invalid exchange server clock")
    server_ms = clock["serverTime"]
    end_ms = server_ms // 60_000 * 60_000
    start_ms = end_ms - minutes * 60_000
    records = []
    for page_start in range(start_ms, end_ms, 1000 * 60_000):
        count = min(1000, (end_ms - page_start) // 60_000)
        rows = _get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": page_start,
                "endTime": page_start + count * 60_000 - 1,
                "limit": count,
            },
        )
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
    final_clock = _get_json("/api/v3/time", {})
    if (
        not isinstance(final_clock, dict)
        or type(final_clock.get("serverTime")) is not int
    ):
        raise ValueError("invalid final exchange server clock")
    final_ms = final_clock["serverTime"]
    # Crossing a minute boundary can leave a newer completed bar unobserved.
    # Refuse and let the caller rerun, rather than silently calling it latest.
    if (
        final_ms < server_ms
        or final_ms // 60_000 != server_ms // 60_000
        or time.monotonic() - started > 45
    ):
        raise ValueError("snapshot became stale during acquisition; rerun")
    return admit_bars(
        pl.DataFrame(records).with_columns(
            pl.col("open_time").cast(pl.Datetime("us", "UTC"))
        )
    ), final_ms
