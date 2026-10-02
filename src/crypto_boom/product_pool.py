"""CEX product metadata and provisional links; never candles or trading permissions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from datetime import UTC, datetime

import aiohttp

ENDPOINTS = {
    "binance_perpetual": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "okx_spot": "https://www.okx.com/api/v5/public/instruments?instType=SPOT",
    "okx_perpetual": "https://www.okx.com/api/v5/public/instruments?instType=SWAP",
}


def decode_products(source: str, raw: bytes) -> list[dict]:
    """Explicit product variants; names are preserved, not inferred asset identities."""
    if source not in ("binance_spot", *ENDPOINTS):
        raise ValueError("unsupported product source")
    limit = (32 if source == "binance_spot" else 16) * 1024**2
    if len(raw) > limit:
        raise ValueError("product snapshot exceeds source byte budget")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("invalid product response")
    okx = source.startswith("okx_")
    if okx and payload.get("code") != "0":
        raise ValueError("OKX product request failed")
    items = payload["data" if okx else "symbols"]
    if not isinstance(items, list) or not 1 <= len(items) <= 10000:
        raise ValueError("require 1..10000 product records")
    result = []
    seen = set()
    perpetual = source.endswith("perpetual")
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("invalid product record")
        instrument = row["instId" if okx else "symbol"]
        if (
            not isinstance(instrument, str)
            or not 1 <= len(instrument) <= 200
            or not instrument[0].isalnum()
            or any(not (c.isalnum() or c in "-_.") for c in instrument)
            or instrument in seen
        ):
            raise ValueError("invalid or duplicate product identity")
        seen.add(instrument)
        if okx:
            if row["state"] != "live" or row["instCategory"] != "1":
                continue
            if perpetual:
                if (
                    row["instType"] != "SWAP"
                    or row["settleCcy"] != "USDT"
                    or row["ctType"] != "linear"
                ):
                    continue
                family = row["instFamily"]
                if not isinstance(family, str) or not family.endswith("-USDT"):
                    raise ValueError("unexpected perpetual family")
                base = family.removesuffix("-USDT")
            else:
                if row["instType"] != "SPOT" or row["quoteCcy"] != "USDT":
                    continue
                base = row["baseCcy"]
            category = "crypto"
        else:
            if row["status"] != "TRADING" or row["quoteAsset"] != "USDT":
                continue
            if perpetual:
                if row["contractType"] != "PERPETUAL" or row["marginAsset"] != "USDT":
                    continue
            elif row["isSpotTradingAllowed"] is not True:
                continue
            base = row["baseAsset"]
            category = row.get("underlyingType", "unclassified")
        if not isinstance(base, str) or not base or len(base) > 100:
            raise ValueError("invalid product base ticker")
        result.append(
            {
                "source": source,
                "venue": "okx" if okx else "binance",
                "product": "perpetual" if perpetual else "spot",
                "instrument_id": instrument,
                "base_ticker": base,
                "quote": "USDT",
                "asset_category": category,
                "contract_value": row.get("ctVal", "") if okx else "",
                "contract_value_ccy": row.get("ctValCcy", "") if okx else "",
            }
        )
    if source == "binance_spot" and len({r["base_ticker"] for r in result}) != len(
        result
    ):
        raise ValueError("ambiguous native spot base ticker")
    return result


async def collect_product_pool(
    session: aiohttp.ClientSession, spot_raw: bytes, *, deadline: float
) -> dict:
    """Three metadata requests maximum; each source fails independently, no fallback."""
    products, sources = [], {}
    stopped_hosts = set()
    requests = 0
    for name in ("binance_spot", *ENDPOINTS):
        raw = spot_raw if name == "binance_spot" else None
        endpoint = ENDPOINTS.get(name)
        host = endpoint.split("/")[2] if endpoint else ""
        observed = datetime.now(UTC).isoformat()
        try:
            if endpoint:
                if host in stopped_hosts or time.monotonic() >= deadline:
                    raise TimeoutError(
                        "metadata source not attempted: rate/time budget"
                    )
                requests += 1
                async with asyncio.timeout(min(10, deadline - time.monotonic())):
                    async with session.get(endpoint, allow_redirects=False) as response:
                        if response.status in (418, 429):
                            stopped_hosts.add(host)
                        response.raise_for_status()
                        if response.status != 200:
                            raise ValueError("unexpected metadata HTTP status")
                        chunks = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            chunks.extend(chunk)
                            if len(chunks) > 16 * 1024**2:
                                raise ValueError("product response exceeds 16 MiB")
                        raw = bytes(chunks)
            assert raw is not None
            admitted = decode_products(name, raw)
            products.extend(admitted)
            sources[name] = {"state": "success", "products": len(admitted)}
        except (
            aiohttp.ClientError,
            TimeoutError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            sources[name] = {"state": "error", "reason": str(exc) or type(exc).__name__}
        sources[name].update(
            observed_at_utc=observed,
            endpoint=endpoint or "/api/v3/exchangeInfo (scan snapshot)",
        )
        if raw is not None:
            sources[name]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
            # Store received bytes exactly, including malformed/error payloads, for replay.
            sources[name]["raw_base64"] = base64.b64encode(raw).decode("ascii")
    return {
        "schema": "cex-product-pool-v1",
        "status": "complete"
        if all(s["state"] == "success" for s in sources.values())
        else "partial",
        "sources": sources,
        "products": products,
        "http_requests": requests,
        "scope": "USDT CEX spot and linear perpetual; OKX crypto category; no delivery/DEX; no account permission or liquidity certification",
    }


def attach_product_coverage(pool: dict, predictions: list[dict]) -> None:
    """Annotate in place, without transferring a spot prediction onto a contract."""
    native = {
        p["base_ticker"]: p["instrument_id"]
        for p in pool["products"]
        if p["source"] == "binance_spot"
    }
    rows = {r["symbol"]: r for r in predictions}
    for row in predictions:
        row["prediction_data_source"] = "binance_spot"
        row["product_candidates"] = []
    native_known = pool["sources"]["binance_spot"]["state"] == "success"
    for product in pool["products"]:
        candidate = native.get(product["base_ticker"])
        native_product = product["source"] == "binance_spot"
        product["mapping_status"] = (
            "native"
            if native_product
            else (
                "ticker_match_unverified"
                if candidate
                else ("no_exact_ticker_match" if native_known else "source_unavailable")
            )
        )
        product["binance_spot_candidate"] = candidate
        row = rows.get(candidate)
        product["binance_spot_prediction_state"] = (
            row["state"] if row else "not_covered"
        )
        if row:
            row["product_candidates"].append(
                {k: product[k] for k in ("source", "instrument_id", "mapping_status")}
            )
    pool["prediction_contract"] = (
        "Binance spot prediction only; foreign products are provisional candidates, not verified mappings or derivative return forecasts. Missing/failed sources do not establish absence."
    )
