"""Metadata-only product coverage must never manufacture asset equivalence."""

import asyncio
import base64
import hashlib
import json
import time
from typing import cast

import aiohttp
import pytest

from crypto_boom.product_pool import (
    attach_product_coverage,
    collect_product_pool,
    decode_products,
)


def spot(base="AAA"):
    return {
        "symbol": base + "USDT",
        "baseAsset": base,
        "quoteAsset": "USDT",
        "status": "TRADING",
        "isSpotTradingAllowed": True,
    }


def okx(base="AAA", perpetual=False):
    return {
        "instId": base + ("-USDT-SWAP" if perpetual else "-USDT"),
        "baseCcy": base,
        "quoteCcy": "USDT",
        "instType": "SWAP" if perpetual else "SPOT",
        "state": "live",
        "instCategory": "1",
        "ctType": "linear",
        "settleCcy": "USDT",
        "instFamily": base + "-USDT",
        "ctVal": "10",
        "ctValCcy": base,
    }


def payload(rows, ok=False):
    return json.dumps({"code": "0", "data": rows} if ok else {"symbols": rows}).encode()


def test_product_types_categories_and_no_multiplier_guess():
    products = decode_products("binance_spot", payload([spot()]))
    products += decode_products(
        "okx_perpetual",
        payload(
            [
                okx("1000AAA", True),
                okx("AAA", True),
                {**okx("STOCK", True), "instCategory": "3"},
            ],
            True,
        ),
    )
    products += decode_products(
        "binance_perpetual",
        payload(
            [
                {**spot("BBB"), "contractType": "PERPETUAL", "marginAsset": "USDT"},
                {
                    **spot("DDD"),
                    "contractType": "CURRENT_QUARTER",
                    "marginAsset": "USDT",
                },
            ]
        ),
    )
    assert len(products) == 4
    pool = {"sources": {"binance_spot": {"state": "success"}}, "products": products}
    rows = [{"symbol": "AAAUSDT", "state": "fetch_error"}]
    attach_product_coverage(pool, rows)
    assert products[0]["mapping_status"] == "native"
    assert products[1]["mapping_status"] == "no_exact_ticker_match"
    assert products[1]["binance_spot_prediction_state"] == "not_covered"
    assert products[2]["mapping_status"] == "ticker_match_unverified"
    assert products[2]["binance_spot_prediction_state"] == "fetch_error"
    assert products[2]["contract_value"] == "10"
    assert len(rows[0]["product_candidates"]) == 2
    assert all("values" not in p for p in products)


@pytest.mark.parametrize(
    "raw",
    [b"[]", b"{}", b'{"symbols":[]}', payload([spot(), spot()]), b'{"symbols":[null]}'],
)
def test_invalid_snapshot_refused(raw):
    with pytest.raises((ValueError, KeyError)):
        decode_products("binance_spot", raw)


class Response:
    def __init__(self, raw, status=200):
        self.raw, self.status = raw, status
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status != 200:
            raise ValueError(f"HTTP {self.status}")

    async def iter_chunked(self, size):
        yield self.raw


class Session:
    def __init__(self, okx_failure=False):
        self.urls = []
        self.okx_failure = okx_failure

    def get(self, url, *, allow_redirects):
        assert not allow_redirects
        self.urls.append(url)
        if "fapi" in url:
            return Response(
                payload(
                    [
                        {
                            **spot("AAA"),
                            "contractType": "PERPETUAL",
                            "marginAsset": "USDT",
                        }
                    ]
                )
            )
        return Response(
            payload([okx("AAA", "SWAP" in url)], True), 429 if self.okx_failure else 200
        )


@pytest.mark.parametrize("failed", [False, True])
def test_collection_bounds_and_rate_failure_preserve_unknown(failed):
    session = Session(failed)
    raw = payload([spot()])
    pool = asyncio.run(
        collect_product_pool(
            cast(aiohttp.ClientSession, session), raw, deadline=time.monotonic() + 30
        )
    )
    assert pool["status"] == ("partial" if failed else "complete")
    assert len(session.urls) == (2 if failed else 3)
    assert all("instruments" in u or "exchangeInfo" in u for u in session.urls)
    source = pool["sources"]["binance_spot"]
    assert base64.b64decode(source["raw_base64"]) == raw
    assert source["sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    if failed:
        assert pool["sources"]["okx_perpetual"]["state"] == "error"
        assert "not attempted" in pool["sources"]["okx_perpetual"]["reason"]


def test_expired_budget_does_not_request_and_unknown_is_not_absence():
    session = Session()
    pool = asyncio.run(
        collect_product_pool(
            cast(aiohttp.ClientSession, session), b"{}", deadline=time.monotonic() - 1
        )
    )
    assert not session.urls and pool["status"] == "partial"
    pool["products"] = decode_products("okx_spot", payload([okx()], True))
    attach_product_coverage(pool, [])
    assert pool["products"][0]["mapping_status"] == "source_unavailable"


@pytest.mark.parametrize("name", ["=FORMULA", "AAA|USDT", "AAA\nUSDT"])
def test_unsafe_report_identifiers_refused(name):
    with pytest.raises(ValueError, match="identity"):
        decode_products("binance_spot", payload([{**spot(), "symbol": name}]))


def test_native_snapshot_keeps_existing_32_mib_admission_budget():
    raw = payload([spot()]) + b" " * (16 * 1024**2)
    assert len(decode_products("binance_spot", raw)) == 1
    with pytest.raises(ValueError, match="byte budget"):
        decode_products("binance_perpetual", raw)
