from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from crypto_boom.binance_source import decode_exchange_info
from crypto_boom.market import LocalReceipt
from crypto_boom.universe import (
    PoolIntegrityError,
    PoolPolicy,
    PoolResourceError,
    build_research_pool,
    load_published_research_pool,
    publish_research_pool,
)

RUN_ID = UUID("654de3b2-472f-46a9-9f27-97ba62fc72ac")
RECEIPT = LocalReceipt(1_795_027_260_000_001_000, 100)


def _symbol(
    symbol: str,
    base_asset: str,
    *,
    quote_asset: str = "USDT",
    status: str = "TRADING",
    permissions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "baseAsset": base_asset,
        "quoteAsset": quote_asset,
        "permissions": ["SPOT"] if permissions is None else permissions,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.0001"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
        ],
    }


def _metadata(*symbols: dict[str, object]):
    payload = json.dumps(
        {"symbols": list(symbols)},
        separators=(",", ":"),
    ).encode()
    return decode_exchange_info(
        payload,
        ingestion_run_id=RUN_ID,
        receipt=RECEIPT,
    )


def test_pool_is_observed_altcoin_spot_usdt_and_never_silently_truncated() -> None:
    metadata = _metadata(
        _symbol("SOLUSDT", "SOL"),
        _symbol("ETHUSDT", "ETH"),
        _symbol("BTCUSDT", "BTC"),
        _symbol("USDCUSDT", "USDC"),
        _symbol("ETHBTC", "ETH", quote_asset="BTC"),
        _symbol("OLDUSDT", "OLD", status="BREAK"),
        _symbol("NOSPOTUSDT", "NOSPOT", permissions=["MARGIN"]),
    )

    pool = build_research_pool(metadata)

    assert pool.symbols == ("ETHUSDT", "SOLUSDT")
    assert pool.maximum_instruments == 500
    assert dict((item.symbol, item.reason) for item in pool.exclusions) == {
        "BTCUSDT": "excluded_base_asset",
        "ETHBTC": "wrong_quote_asset",
        "NOSPOTUSDT": "missing_permission",
        "OLDUSDT": "not_trading",
        "USDCUSDT": "excluded_base_asset",
    }
    assert pool.pool_id == (
        "sha256:1fed44a82e28898a9e9ceae0ff3d5a6234e38684c7804f45d8e483f253ee98f0"
    )

    with pytest.raises(PoolResourceError, match="2 instruments"):
        build_research_pool(
            metadata,
            policy=PoolPolicy(maximum_instruments=1),
        )


def test_pool_publication_is_content_addressed_and_strictly_reloadable(
    tmp_path: Path,
) -> None:
    metadata = _metadata(
        _symbol("SOLUSDT", "SOL"),
        _symbol("ETHUSDT", "ETH"),
    )
    pool = build_research_pool(metadata)

    first = publish_research_pool(pool, metadata=metadata, output_root=tmp_path)
    second = publish_research_pool(pool, metadata=metadata, output_root=tmp_path)
    loaded = load_published_research_pool(first.path)

    assert not first.already_present
    assert second.already_present
    assert loaded == pool
    assert json.loads((first.path / "pool.json").read_text()) == pool.to_mapping()
    assert (first.path / "exchange-info.json").read_bytes() == metadata.raw_payload

    (first.path / "exchange-info.json").write_bytes(b"tampered")
    with pytest.raises(PoolIntegrityError, match="source payload"):
        load_published_research_pool(first.path)
