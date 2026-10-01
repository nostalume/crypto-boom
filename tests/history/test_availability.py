from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from crypto_boom.history.availability import (
    AvailabilityIntegrityError,
    AvailabilityLimits,
    AvailabilityResourceError,
    AvailabilityState,
    HistoricalArchiveSymbolCatalog,
    MonthlyAvailabilityRequest,
    discover_spot_archive_symbols,
    load_published_archive_availability,
    probe_monthly_archive_availability,
)
from crypto_boom.universe import ResearchInstrumentPool


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    body: bytes


def _pool() -> ResearchInstrumentPool:
    return ResearchInstrumentPool(
        schema_version=1,
        policy_version="binance-spot-altcoin-pool-v1",
        source_endpoint="https://data-api.binance.vision/api/v3/exchangeInfo",
        source_payload_sha256="sha256:" + "a" * 64,
        observed_at_ns=1_795_027_260_000_001_000,
        maximum_instruments=500,
        quote_asset="USDT",
        required_permission="SPOT",
        excluded_base_assets=(
            "BRL",
            "BTC",
            "BUSD",
            "DAI",
            "EUR",
            "FDUSD",
            "TRY",
            "TUSD",
            "USDC",
            "USDP",
        ),
        excluded_symbols=(),
        symbols=("ETHUSDT", "SOLUSDT"),
        exclusions=(),
        unsupported_symbols=(),
    )


def _checksum(symbol: str, month: str, character: str) -> bytes:
    filename = f"{symbol}-1m-{month}.zip"
    return f"{character * 64}  {filename}\n".encode()


def test_archive_symbol_discovery_keeps_historical_names_without_status_claim() -> None:
    prefix = "data/spot/monthly/klines/"

    def page(symbols: tuple[str, ...], *, truncated: bool, token: str = "") -> bytes:
        entries = "".join(
            f"<CommonPrefixes><Prefix>{prefix}{symbol}/</Prefix></CommonPrefixes>"
            for symbol in symbols
        )
        return (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Prefix>{prefix}</Prefix>"
            f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"
            f"<NextContinuationToken>{token}</NextContinuationToken>"
            f"{entries}</ListBucketResult>"
        ).encode()

    first_page = (
        f"/data.binance.vision?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000"
    )
    responses = {
        first_page: [_Response(200, page(("OLDUSDT",), truncated=True, token="next"))],
        first_page + "&continuation-token=next": [
            _Response(200, page(("NEWUSDT",), truncated=False))
        ],
    }
    observed: list[str] = []

    async def operation(url: str) -> HistoricalArchiveSymbolCatalog:
        return await discover_spot_archive_symbols(base_url=url)

    catalog = asyncio.run(
        _with_server(
            responses, operation, observed=observed, base_path="/data.binance.vision"
        )
    )
    assert catalog.symbols == ("NEWUSDT", "OLDUSDT")
    assert catalog.page_count == 2
    assert catalog.observed_at_ns > 0
    assert len(observed) == 2

    async def limited(url: str) -> HistoricalArchiveSymbolCatalog:
        return await discover_spot_archive_symbols(base_url=url, maximum_pages=1)

    with pytest.raises(AvailabilityResourceError, match="page limit"):
        asyncio.run(
            _with_server(
                responses, limited, observed=[], base_path="/data.binance.vision"
            )
        )


async def _with_server[T](
    responses: dict[str, list[_Response]],
    operation: Callable[[str], Awaitable[T]],
    *,
    observed: list[str],
    base_path: str = "/data/spot/monthly",
) -> T:
    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        parts = request_line.decode("ascii").strip().split(" ")
        path = parts[1] if len(parts) == 3 else ""
        observed.append(path)
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        choices = responses.get(path, [_Response(404, b"not found")])
        response = choices.pop(0) if len(choices) > 1 else choices[0]
        reason = "OK" if response.status == 200 else "Not Found"
        writer.write(
            (
                f"HTTP/1.1 {response.status} {reason}\r\n"
                f"Content-Length: {len(response.body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
        )
        writer.write(response.body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        return await operation(f"http://{host}:{port}{base_path}")
    finally:
        server.close()
        await server.wait_closed()


def test_monthly_availability_records_presence_absence_and_retry(
    tmp_path: Path,
) -> None:
    responses: dict[str, list[_Response]] = {}
    for symbol, month, response_list in (
        ("ETHUSDT", "2025-01", [_Response(200, _checksum("ETHUSDT", "2025-01", "a"))]),
        ("ETHUSDT", "2025-02", [_Response(404, b"not found")]),
        (
            "SOLUSDT",
            "2025-01",
            [
                _Response(500, b"temporary"),
                _Response(200, _checksum("SOLUSDT", "2025-01", "b")),
            ],
        ),
        ("SOLUSDT", "2025-02", [_Response(404, b"not found")]),
    ):
        path = f"/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip.CHECKSUM"
        responses[path] = response_list
    observed: list[str] = []
    request = MonthlyAvailabilityRequest(
        start_month=date(2025, 1, 1),
        end_month=date(2025, 2, 1),
    )

    async def probe(base_url: str):
        return await probe_monthly_archive_availability(
            _pool(),
            request,
            output_root=tmp_path,
            limits=AvailabilityLimits(
                maximum_probes=4,
                retry_attempts=2,
                retry_base_seconds=0.0,
                request_spacing_seconds=0.0,
            ),
            base_url=base_url,
        )

    published = asyncio.run(_with_server(responses, probe, observed=observed))

    assert published.report.available_count == 2
    assert published.report.not_found_count == 2
    assert published.report.unresolved_count == 0
    assert published.report.ready
    assert [
        (item.symbol, item.month, item.state) for item in published.report.probes
    ] == [
        ("ETHUSDT", "2025-01", AvailabilityState.AVAILABLE),
        ("ETHUSDT", "2025-02", AvailabilityState.NOT_FOUND),
        ("SOLUSDT", "2025-01", AvailabilityState.AVAILABLE),
        ("SOLUSDT", "2025-02", AvailabilityState.NOT_FOUND),
    ]
    assert len(observed) == 5
    document = json.loads((published.path / "availability.json").read_text())
    assert document == published.report.to_mapping()
    assert load_published_archive_availability(published.path) == published.report

    document["schema_version"] = True
    (published.path / "availability.json").write_text(json.dumps(document))
    with pytest.raises(
        AvailabilityIntegrityError, match="availability report is invalid"
    ):
        load_published_archive_availability(published.path)

    (published.path / "availability.json").write_text("{}")
    with pytest.raises(AvailabilityIntegrityError, match="availability report"):
        load_published_archive_availability(published.path)


def test_monthly_availability_bound_fails_before_network(tmp_path: Path) -> None:
    request = MonthlyAvailabilityRequest(
        start_month=date(2025, 1, 1),
        end_month=date(2025, 2, 1),
    )
    observed: list[str] = []

    async def probe(base_url: str):
        return await probe_monthly_archive_availability(
            _pool(),
            request,
            output_root=tmp_path,
            limits=AvailabilityLimits(maximum_probes=3),
            base_url=base_url,
        )

    with pytest.raises(AvailabilityResourceError, match="4 probes"):
        asyncio.run(_with_server({}, probe, observed=observed))
    assert observed == []


def test_monthly_availability_selects_known_symbols_before_network(
    tmp_path: Path,
) -> None:
    request = MonthlyAvailabilityRequest(
        start_month=date(2025, 1, 1),
        end_month=date(2025, 2, 1),
    )
    observed: list[str] = []

    async def reject_unknown(base_url: str):
        return await probe_monthly_archive_availability(
            _pool(),
            request,
            output_root=tmp_path,
            symbols=("UNKNOWNUSDT",),
            base_url=base_url,
        )

    with pytest.raises(AvailabilityIntegrityError, match="not present"):
        asyncio.run(_with_server({}, reject_unknown, observed=observed))
    assert observed == []

    async def reject_duplicate(base_url: str):
        return await probe_monthly_archive_availability(
            _pool(),
            request,
            output_root=tmp_path,
            symbols=("SOLUSDT", "SOLUSDT"),
            base_url=base_url,
        )

    with pytest.raises(AvailabilityIntegrityError, match="duplicates"):
        asyncio.run(_with_server({}, reject_duplicate, observed=observed))
    assert observed == []

    responses = {
        "/data/spot/monthly/klines/SOLUSDT/1m/SOLUSDT-1m-2025-01.zip.CHECKSUM": [
            _Response(200, _checksum("SOLUSDT", "2025-01", "b"))
        ],
        "/data/spot/monthly/klines/SOLUSDT/1m/SOLUSDT-1m-2025-02.zip.CHECKSUM": [
            _Response(404, b"not found")
        ],
    }

    async def probe_subset(base_url: str):
        return await probe_monthly_archive_availability(
            _pool(),
            request,
            output_root=tmp_path,
            symbols=("SOLUSDT",),
            limits=AvailabilityLimits(
                maximum_probes=2,
                request_spacing_seconds=0.0,
            ),
            base_url=base_url,
        )

    published = asyncio.run(_with_server(responses, probe_subset, observed=observed))
    assert [(item.symbol, item.month) for item in published.report.probes] == [
        ("SOLUSDT", "2025-01"),
        ("SOLUSDT", "2025-02"),
    ]
    assert len(observed) == 2
