"""Bounded whole-universe snapshot acquisition, independent of model or research."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from uuid import uuid4

import aiohttp
import polars as pl

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US, admit_bars, decode_minute_page
from crypto_boom.binance_source import (
    PUBLIC_REST_BASE,
    RestWeightBudget,
    decode_exchange_info,
    sampled_receipt,
    select_spot_usdt_universe,
)


class ScanStopped(RuntimeError):
    """Whole-run resource/rate boundary; never retry on another endpoint."""


class SpotSnapshotClient:
    """One shared session/rate/deadline owner; max four requests per second."""

    def __init__(self, session: aiohttp.ClientSession, *, seconds: int = 900):
        if not 1 <= seconds <= 900:
            raise ValueError("scan duration must be 1..900 seconds")
        self.session = session
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.lock = asyncio.Lock()
        self.next_request = 0.0
        self.stopped = False
        self.requests = 0
        self.budget = RestWeightBudget(venue_capacity=1200, interval_seconds=60)

    async def get(
        self, path: str, params: dict, *, weight: int, maximum_bytes: int = 2_000_000
    ) -> bytes:
        async with self.lock:
            if (
                self.stopped
                or time.monotonic() >= self.deadline
                or self.requests >= 5000
            ):
                self.stopped = True
                raise ScanStopped("scan request/time budget exhausted")
            await asyncio.sleep(max(0.0, self.next_request - time.monotonic()))
            while not self.budget.try_acquire(weight, now=time.monotonic()):
                if time.monotonic() >= self.deadline:
                    self.stopped = True
                    raise ScanStopped("scan rate-budget deadline reached")
                await asyncio.sleep(0.25)
            if self.stopped or time.monotonic() >= self.deadline:
                raise ScanStopped("scan stopped before request")
            self.next_request = time.monotonic() + 0.25
            self.requests += 1
        async with self.session.get(PUBLIC_REST_BASE + path, params=params) as response:
            if response.status in (418, 429):
                self.stopped = True
                raise ScanStopped(
                    f"HTTP {response.status}; no retry; Retry-After={response.headers.get('Retry-After', 'unknown')}"
                )
            response.raise_for_status()
            chunks = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                chunks.extend(chunk)
                if len(chunks) > maximum_bytes:
                    raise ValueError("response exceeds byte budget")
            return bytes(chunks)

    async def universe(self) -> tuple[list[str], dict, bytes, int]:
        raw = await self.get(
            "/api/v3/exchangeInfo", {}, weight=20, maximum_bytes=32 * 1024**2
        )
        payload = json.loads(raw)
        capture = decode_exchange_info(
            raw, ingestion_run_id=uuid4(), receipt=sampled_receipt()
        )
        canonical = set(select_spot_usdt_universe(capture.events))
        # Preserve eligible but unsupported names as explicit refusals, not disappearing members.
        scope = set(canonical)
        seen = set()
        for row in payload["symbols"]:
            if row["symbol"] in seen:
                raise ValueError("duplicate instrument metadata")
            seen.add(row["symbol"])
            if row.get("isSpotTradingAllowed") is False:
                scope.discard(row["symbol"])
            if (
                row.get("status") == "TRADING"
                and row.get("quoteAsset") == "USDT"
                and row.get("isSpotTradingAllowed") is True
            ):
                scope.add(row["symbol"])
        if not scope or len(scope) > 2000:
            raise ValueError(
                "whole market scope must contain 1..2000 members; not truncated"
            )
        limits = [
            r["limit"]
            for r in payload.get("rateLimits", [])
            if r.get("rateLimitType") == "REQUEST_WEIGHT"
            and r.get("interval") == "MINUTE"
            and r.get("intervalNum") == 1
        ]
        if limits and any(type(limit) is not int or limit < 40 for limit in limits):
            raise ValueError("unsupported venue rate budget")
        if limits:
            self.budget = RestWeightBudget(
                venue_capacity=min(limits), interval_seconds=60
            )
            self.budget.try_acquire(20, now=time.monotonic())
        clock = json.loads(
            await self.get("/api/v3/time", {}, weight=1, maximum_bytes=1024)
        )
        if type(clock.get("serverTime")) is not int or clock["serverTime"] <= 0:
            raise ValueError("invalid exchange clock")
        return (
            sorted(scope),
            {
                "scope": "Binance Spot TRADING USDT at snapshot",
                "metadata_refusals": sorted(scope - canonical),
            },
            raw,
            clock["serverTime"],
        )

    async def bars(
        self, symbol: str, *, decision_us: int, history_minutes: int, cache: Path
    ) -> tuple[pl.DataFrame, dict]:
        if (
            not re.fullmatch(r"[A-Z0-9]{3,32}", symbol)
            or decision_us % MINUTE_US
            or not 1 <= history_minutes <= 1440
        ):
            raise ValueError("invalid snapshot request")
        spec: dict[str, object] = {
            "schema": "spot-minute-snapshot-v1",
            "symbol": symbol,
            "decision_us": decision_us,
            "history_minutes": history_minutes,
        }
        cache_id = _artifacts.content_id(spec).removeprefix("sha256:")
        target = cache / cache_id
        if target.exists():
            if (target / "receipt.json").stat().st_size > 4096 or (
                target / "bars.parquet"
            ).stat().st_size > 2_000_000:
                raise ValueError("source cache exceeds size budget")
            receipt = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
            if (
                receipt["spec"] != spec
                or receipt["sha256"]
                != _artifacts.file_identity(target / "bars.parquet")[0]
            ):
                raise ValueError("source cache identity/hash mismatch")
            frame = admit_bars(pl.read_parquet(target / "bars.parquet"))
            reused = True
        else:
            end = decision_us // 1000
            start = end - (history_minutes + 1) * 60000
            records = []
            for page in range(start, end, 1000 * 60000):
                count = min(1000, (end - page) // 60000)
                rows = json.loads(
                    await self.get(
                        "/api/v3/klines",
                        {
                            "symbol": symbol,
                            "interval": "1m",
                            "startTime": page,
                            "endTime": page + count * 60000 - 1,
                            "limit": count,
                        },
                        weight=2,
                    )
                )
                records.extend(
                    decode_minute_page(
                        rows, symbol=symbol, page_start=page, count=count
                    )
                )
            frame = admit_bars(
                pl.DataFrame(records).with_columns(
                    pl.col("open_time").cast(pl.Datetime("us", "UTC"))
                )
            )
            cache.mkdir(parents=True, exist_ok=True)
            with _artifacts.publication_staging_directory(
                cache, prefix="snapshot-"
            ) as staging:
                frame.write_parquet(staging / "bars.parquet")
                receipt: dict[str, object] = {
                    "spec": spec,
                    "sha256": _artifacts.file_identity(staging / "bars.parquet")[0],
                }
                (staging / "receipt.json").write_bytes(
                    _artifacts.canonical_json(receipt)
                )
                staging.rename(target)
            reused = False
        times = frame["open_time"].dt.epoch("us")
        if (
            len(frame) != history_minutes + 1
            or set(frame["symbol"]) != {symbol}
            or int(times[-1]) + MINUTE_US != decision_us
            or not (times.diff().drop_nulls() == MINUTE_US).all()
        ):
            raise ValueError("snapshot does not match complete requested window")
        return frame, {
            "cache_id": cache_id,
            "source_sha256": receipt["sha256"],
            "reused": reused,
        }
