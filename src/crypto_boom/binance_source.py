"""Shared Binance Spot source identity and REST evidence admission."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from time import monotonic, monotonic_ns, time_ns
from typing import Final
from uuid import UUID

import aiohttp
import msgspec

from crypto_boom.market import (
    Environment,
    EvidenceAdmissionError,
    InstrumentId,
    InstrumentMetadata,
    InstrumentStatus,
    LocalReceipt,
    MetadataObservation,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    VenueId,
)

PUBLIC_REST_BASE: Final = "https://data-api.binance.vision"
BINANCE_SOURCE_SCHEMA_VERSION: Final = 1
BINANCE_SPOT = VenueId(name="binance", market="spot")


class LiveCaptureError(RuntimeError):
    """Binance source evidence could not be admitted or captured safely."""


class LiveSchemaError(LiveCaptureError):
    """A Binance source payload does not satisfy its wire contract."""


class LiveResourceError(LiveCaptureError):
    """A Binance source operation exceeded a bounded resource contract."""


class LiveRateLimitError(LiveResourceError):
    """The venue instructed the client to stop sending requests."""

    def __init__(self, *, status: int, retry_after_seconds: int | None) -> None:
        super().__init__("venue rate limit requires backoff")
        self.status = status
        self.retry_after_seconds = retry_after_seconds


class _ExchangeFilter(msgspec.Struct, forbid_unknown_fields=False):
    filter_type: str = msgspec.field(name="filterType")
    tick_size: str | None = msgspec.field(default=None, name="tickSize")
    step_size: str | None = msgspec.field(default=None, name="stepSize")
    min_notional: str | None = msgspec.field(default=None, name="minNotional")


class _ExchangeSymbol(msgspec.Struct, forbid_unknown_fields=False):
    symbol: str
    status: str
    base_asset: str = msgspec.field(name="baseAsset")
    quote_asset: str = msgspec.field(name="quoteAsset")
    filters: list[_ExchangeFilter]
    permissions: list[str] = msgspec.field(default_factory=list)
    permission_sets: list[list[str]] = msgspec.field(
        default_factory=list,
        name="permissionSets",
    )


class _ExchangeInfo(msgspec.Struct, forbid_unknown_fields=False):
    symbols: list[_ExchangeSymbol]


class _ServerTime(msgspec.Struct, forbid_unknown_fields=False):
    server_time: int = msgspec.field(name="serverTime")


_EXCHANGE_INFO_DECODER = msgspec.json.Decoder(_ExchangeInfo)
_SERVER_TIME_DECODER = msgspec.json.Decoder(_ServerTime)


@dataclass(frozen=True, slots=True)
class ClockMeasurement:
    """One midpoint-adjusted exchange clock observation."""

    offset_ms: Decimal
    round_trip_ns: int
    sampled_at_ns: int

    def __post_init__(self) -> None:
        if self.round_trip_ns < 0 or self.sampled_at_ns <= 0:
            raise ValueError("clock measurement is invalid")


@dataclass(frozen=True, slots=True)
class MetadataCapture:
    """Exact REST snapshot paired with all canonical metadata rows."""

    raw_payload: bytes
    events: tuple[InstrumentMetadata, ...]
    unsupported_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.raw_payload or not self.events:
            raise ValueError("metadata capture must retain payload and rows")
        if not isinstance(self.unsupported_symbols, tuple):
            raise ValueError("unsupported metadata symbols must be a tuple")
        digest = PayloadDigest.sha256(self.raw_payload)
        if any(event.provenance.payload_digest != digest for event in self.events):
            raise ValueError("metadata rows do not match the retained payload")


class RestWeightBudget:
    """Deterministic token bucket reserving half the configured venue capacity."""

    def __init__(
        self,
        *,
        venue_capacity: int,
        interval_seconds: float,
        utilization: Decimal = Decimal("0.5"),
    ) -> None:
        if venue_capacity <= 0 or interval_seconds <= 0:
            raise ValueError("rate-limit capacity and interval must be positive")
        if not Decimal("0") < utilization <= Decimal("0.5"):
            raise ValueError("REST utilization must be in (0, 0.5]")
        self.capacity = Decimal(venue_capacity) * utilization
        self.refill_per_second = self.capacity / Decimal(str(interval_seconds))
        self._tokens = self.capacity
        self._updated_at = Decimal("0")

    @property
    def available(self) -> Decimal:
        return self._tokens

    def try_acquire(self, weight: int, *, now: float) -> bool:
        if weight <= 0:
            raise ValueError("request weight must be positive")
        current = Decimal(str(now))
        if current < self._updated_at:
            raise ValueError("rate-budget clock moved backwards")
        elapsed = current - self._updated_at
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.refill_per_second,
        )
        self._updated_at = current
        requested = Decimal(weight)
        if requested > self._tokens:
            return False
        self._tokens -= requested
        return True


async def fetch_exchange_info(
    session: aiohttp.ClientSession,
    *,
    ingestion_run_id: UUID,
    budget: RestWeightBudget,
    symbol: str | None = None,
    endpoint: str = f"{PUBLIC_REST_BASE}/api/v3/exchangeInfo",
    maximum_bytes: int = 32 * 1024 * 1024,
    budget_now: float | None = None,
) -> MetadataCapture:
    """Acquire one bounded, rate-budgeted public metadata snapshot."""

    if maximum_bytes <= 0:
        raise ValueError("metadata byte limit must be positive")
    params: dict[str, str] = {}
    if symbol is not None:
        params["symbol"] = _instrument(symbol).symbol
    now = monotonic() if budget_now is None else budget_now
    if not budget.try_acquire(20, now=now):
        raise LiveResourceError("REST request-weight budget is exhausted")
    try:
        async with session.get(
            endpoint,
            params=params,
            headers={"X-MBX-TIME-UNIT": "MICROSECOND"},
        ) as response:
            if response.status in {418, 429}:
                raise LiveRateLimitError(
                    status=response.status,
                    retry_after_seconds=_retry_after_seconds(response),
                )
            if response.status != 200:
                raise LiveCaptureError("exchange-info endpoint returned non-success")
            if (
                response.content_length is not None
                and response.content_length > maximum_bytes
            ):
                raise LiveResourceError("exchange-info response exceeds byte limit")
            body = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                body.extend(chunk)
                if len(body) > maximum_bytes:
                    raise LiveResourceError("exchange-info response exceeds byte limit")
            receipt = sampled_receipt()
    except LiveCaptureError:
        raise
    except (TimeoutError, aiohttp.ClientError) as error:
        raise LiveCaptureError("exchange-info request failed") from error
    payload = bytes(body)
    return decode_exchange_info(
        payload,
        ingestion_run_id=ingestion_run_id,
        receipt=receipt,
        endpoint=endpoint,
    )


async def measure_exchange_clock(
    session: aiohttp.ClientSession,
    *,
    budget: RestWeightBudget,
    endpoint: str = f"{PUBLIC_REST_BASE}/api/v3/time",
    budget_now: float | None = None,
) -> ClockMeasurement:
    """Measure exchange-to-local wall-clock offset with a monotonic midpoint."""

    now = monotonic() if budget_now is None else budget_now
    if not budget.try_acquire(1, now=now):
        raise LiveResourceError("REST request-weight budget is exhausted")
    wall_start = time_ns()
    monotonic_start = monotonic_ns()
    try:
        async with session.get(
            endpoint,
            headers={"X-MBX-TIME-UNIT": "MICROSECOND"},
        ) as response:
            if response.status in {418, 429}:
                raise LiveRateLimitError(
                    status=response.status,
                    retry_after_seconds=_retry_after_seconds(response),
                )
            if response.status != 200:
                raise LiveCaptureError("exchange-time endpoint returned non-success")
            payload = await response.read()
            if len(payload) > 1_024:
                raise LiveResourceError("exchange-time response exceeds byte limit")
    except LiveCaptureError:
        raise
    except (TimeoutError, aiohttp.ClientError) as error:
        raise LiveCaptureError("exchange-time request failed") from error
    wall_end = time_ns()
    monotonic_end = monotonic_ns()
    try:
        server_time = _SERVER_TIME_DECODER.decode(payload).server_time
    except msgspec.DecodeError as error:
        raise LiveSchemaError(
            "exchange-time payload failed schema admission"
        ) from error
    if server_time <= 0:
        raise LiveSchemaError("exchange-time value must be positive")
    multiplier = 1_000 if server_time >= 100_000_000_000_000 else 1_000_000
    server_time_ns = server_time * multiplier
    midpoint_ns = wall_start + (wall_end - wall_start) // 2
    return ClockMeasurement(
        offset_ms=Decimal(server_time_ns - midpoint_ns) / Decimal(1_000_000),
        round_trip_ns=monotonic_end - monotonic_start,
        sampled_at_ns=wall_end,
    )


def _retry_after_seconds(response: aiohttp.ClientResponse) -> int | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = int(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def decode_exchange_info(
    payload: bytes,
    *,
    ingestion_run_id: UUID,
    receipt: LocalReceipt,
    endpoint: str = f"{PUBLIC_REST_BASE}/api/v3/exchangeInfo",
) -> MetadataCapture:
    """Decode an observed snapshot and expose unsupported instrument refusals."""

    if not isinstance(payload, bytes) or not payload:
        raise LiveSchemaError("exchange-info payload must be non-empty bytes")
    digest = PayloadDigest.sha256(payload)
    provenance = Provenance(
        source=SourceDescriptor(
            endpoint=endpoint,
            channel="/api/v3/exchangeInfo",
            schema_version=BINANCE_SOURCE_SCHEMA_VERSION,
        ),
        ingestion_run_id=ingestion_run_id,
        receipt=receipt,
        payload_digest=digest,
        raw_payload_reference=digest.value,
    )
    quality = ObservationQuality(QualityState.VALID, complete=True)
    try:
        document = _EXCHANGE_INFO_DECODER.decode(payload)
        admitted: list[InstrumentMetadata] = []
        unsupported: list[str] = []
        for symbol in document.symbols:
            if any(
                not value.isascii()
                for value in (symbol.symbol, symbol.base_asset, symbol.quote_asset)
            ):
                unsupported.append(symbol.symbol)
                continue
            filters = {item.filter_type: item for item in symbol.filters}
            price_filter = filters.get("PRICE_FILTER")
            lot_filter = filters.get("LOT_SIZE")
            notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
            if (
                price_filter is None
                or price_filter.tick_size is None
                or lot_filter is None
                or lot_filter.step_size is None
                or notional_filter is None
                or notional_filter.min_notional is None
            ):
                raise LiveSchemaError("required exchange filter is absent")
            permissions = symbol.permissions or [
                value
                for permission_set in symbol.permission_sets
                for value in permission_set
            ]
            admitted.append(
                InstrumentMetadata(
                    instrument=_instrument(symbol.symbol),
                    raw_symbol=symbol.symbol,
                    status=InstrumentStatus(symbol.status),
                    base_asset=symbol.base_asset,
                    quote_asset=symbol.quote_asset,
                    price_tick=_decimal(price_filter.tick_size),
                    quantity_step=_decimal(lot_filter.step_size),
                    minimum_notional=_decimal(notional_filter.min_notional),
                    permissions=tuple(dict.fromkeys(permissions)),
                    observation=MetadataObservation.OBSERVED,
                    provenance=provenance,
                    quality=quality,
                )
            )
    except LiveCaptureError:
        raise
    except (
        msgspec.DecodeError,
        EvidenceAdmissionError,
        InvalidOperation,
        ValueError,
    ) as error:
        raise LiveSchemaError(
            "exchange-info payload failed schema admission"
        ) from error
    if not admitted:
        raise LiveSchemaError("exchange-info snapshot contains no supported symbols")
    return MetadataCapture(
        raw_payload=payload,
        events=tuple(admitted),
        unsupported_symbols=tuple(unsupported),
    )


def sampled_receipt() -> LocalReceipt:
    """Sample local clocks immediately before decoding a received payload."""

    return LocalReceipt(wall_time_ns=time_ns(), monotonic_ns=monotonic_ns())


def _instrument(symbol: str) -> InstrumentId:
    return InstrumentId(
        venue=BINANCE_SPOT,
        environment=Environment.PRODUCTION,
        symbol=symbol,
    )


def _decimal(value: str) -> Decimal:
    if not isinstance(value, str):
        raise LiveSchemaError("wire decimal must be encoded as text")
    return Decimal(value)


def select_spot_usdt_universe(
    metadata: Iterable[InstrumentMetadata],
) -> tuple[str, ...]:
    """Select observed, trading USDT Spot instruments in stable symbol order."""

    return tuple(
        sorted(
            event.instrument.symbol
            for event in metadata
            if event.status is InstrumentStatus.TRADING
            and event.quote_asset == "USDT"
            and "SPOT" in event.permissions
            and event.observation is MetadataObservation.OBSERVED
        )
    )


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
