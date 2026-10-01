"""Read-only prospective Binance Spot capture boundary."""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from time import monotonic, monotonic_ns, time_ns
from typing import Any, Final
from urllib.parse import quote
from uuid import UUID

import aiohttp
import msgspec

from crypto_boom import binance_source as _binance_source
from crypto_boom.market import (
    AggregateTradeEvent,
    BookTickerEvent,
    Environment,
    EpochTimestamp,
    EvidenceAdmissionError,
    InstrumentId,
    KlineEvent,
    LocalReceipt,
    MarketEvidence,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    TimeUnit,
    classify_reobservation,
)

# Finite compatibility aliases: live capture still consumes the shared source
# primitives, while new callers import their authoritative owner directly.
BINANCE_SPOT = _binance_source.BINANCE_SPOT
PUBLIC_REST_BASE = _binance_source.PUBLIC_REST_BASE
ClockMeasurement = _binance_source.ClockMeasurement
LiveCaptureError = _binance_source.LiveCaptureError
LiveRateLimitError = _binance_source.LiveRateLimitError
LiveResourceError = _binance_source.LiveResourceError
LiveSchemaError = _binance_source.LiveSchemaError
MetadataCapture = _binance_source.MetadataCapture
RestWeightBudget = _binance_source.RestWeightBudget
decode_exchange_info = _binance_source.decode_exchange_info
fetch_exchange_info = _binance_source.fetch_exchange_info
measure_exchange_clock = _binance_source.measure_exchange_clock
sampled_receipt = _binance_source.sampled_receipt
_decimal = _binance_source._decimal
_instrument = _binance_source._instrument

PUBLIC_WS_BASE: Final = "wss://data-stream.binance.vision"
LIVE_SCHEMA_VERSION: Final = 1
MAX_HOT_SYMBOLS: Final = 20
DEFAULT_BREADTH_SHARD_SIZE: Final = 250
CAPTURE_STREAMS_PER_CONNECTION: Final = 300
VENUE_STREAMS_PER_CONNECTION: Final = 1_024
VENUE_CONNECTION_ATTEMPTS_PER_5_MINUTES: Final = 300


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """Facts sampled before decoding one received payload."""

    ingestion_run_id: UUID
    receipt: LocalReceipt
    connection_generation: int
    endpoint: str = PUBLIC_WS_BASE
    environment: Environment = Environment.PRODUCTION
    time_unit: TimeUnit = TimeUnit.MICROSECOND

    def __post_init__(self) -> None:
        if self.connection_generation < 0:
            raise ValueError("connection generation must be non-negative")
        if self.environment is not Environment.PRODUCTION:
            raise ValueError("CAP-01 admits only the production public feed")


@dataclass(frozen=True, slots=True)
class CapturedEvidence:
    """Exact raw WebSocket message paired with its canonical event."""

    stream: str
    raw_payload: bytes
    event: MarketEvidence

    def __post_init__(self) -> None:
        if not self.stream or not self.raw_payload:
            raise ValueError("captured stream and payload must be non-empty")
        if self.event.provenance.payload_digest != PayloadDigest.sha256(
            self.raw_payload
        ):
            raise ValueError("raw payload does not match event provenance")


@dataclass(frozen=True, slots=True)
class RawMessageObservation:
    """Exact retained frame plus receive facts and retention disposition."""

    raw_payload: bytes
    ingestion_run_id: UUID
    receipt: LocalReceipt
    connection_generation: int
    retention_reason: str

    def __post_init__(self) -> None:
        if not self.raw_payload:
            raise ValueError("observed raw payload must be non-empty")
        if self.connection_generation < 0:
            raise ValueError("connection generation must be non-negative")
        if not self.retention_reason or len(self.retention_reason) > 64:
            raise ValueError("raw retention reason is invalid")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Downstream disposition without mutating immutable source evidence."""

    state: QualityState
    accepted_for_downstream: bool
    recovery_required: bool = False
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueueAdmission:
    """Observable result of a bounded queue offer."""

    enqueued: bool
    state: QualityState
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureStats:
    messages_received: int
    messages_enqueued: int
    messages_refused: int
    reconnects: int
    rotations: int
    duplicate_messages: int = 0
    conflict_messages: int = 0
    gap_messages: int = 0
    provisional_messages: int = 0
    schema_mismatches: int = 0
    queue_overflows: int = 0
    raw_messages_retained: int = 0
    raw_messages_suppressed: int = 0


class _CombinedMessage(msgspec.Struct, forbid_unknown_fields=False):
    stream: str
    data: msgspec.Raw


class _KlineBody(msgspec.Struct, forbid_unknown_fields=False):
    open_time: int = msgspec.field(name="t")
    close_time: int = msgspec.field(name="T")
    symbol: str = msgspec.field(name="s")
    interval: str = msgspec.field(name="i")
    first_trade_id: int = msgspec.field(name="f")
    last_trade_id: int = msgspec.field(name="L")
    open_price: str = msgspec.field(name="o")
    close_price: str = msgspec.field(name="c")
    high_price: str = msgspec.field(name="h")
    low_price: str = msgspec.field(name="l")
    base_volume: str = msgspec.field(name="v")
    trade_count: int = msgspec.field(name="n")
    closed: bool = msgspec.field(name="x")
    quote_turnover: str = msgspec.field(name="q")
    taker_buy_base_volume: str = msgspec.field(name="V")
    taker_buy_quote_turnover: str = msgspec.field(name="Q")


class _KlineMessage(msgspec.Struct, forbid_unknown_fields=False):
    event_type: str = msgspec.field(name="e")
    event_time: int = msgspec.field(name="E")
    symbol: str = msgspec.field(name="s")
    kline: _KlineBody = msgspec.field(name="k")


class _AggregateTradeMessage(msgspec.Struct, forbid_unknown_fields=False):
    event_type: str = msgspec.field(name="e")
    event_time: int = msgspec.field(name="E")
    symbol: str = msgspec.field(name="s")
    aggregate_trade_id: int = msgspec.field(name="a")
    price: str = msgspec.field(name="p")
    quantity: str = msgspec.field(name="q")
    first_trade_id: int = msgspec.field(name="f")
    last_trade_id: int = msgspec.field(name="l")
    trade_time: int = msgspec.field(name="T")
    buyer_is_maker: bool = msgspec.field(name="m")


class _BookTickerMessage(msgspec.Struct, forbid_unknown_fields=False):
    update_id: int = msgspec.field(name="u")
    symbol: str = msgspec.field(name="s")
    bid_price: str = msgspec.field(name="b")
    bid_quantity: str = msgspec.field(name="B")
    ask_price: str = msgspec.field(name="a")
    ask_quantity: str = msgspec.field(name="A")


_COMBINED_DECODER = msgspec.json.Decoder(_CombinedMessage)
_KLINE_DECODER = msgspec.json.Decoder(_KlineMessage)
_AGG_TRADE_DECODER = msgspec.json.Decoder(_AggregateTradeMessage)
_BOOK_TICKER_DECODER = msgspec.json.Decoder(_BookTickerMessage)


def decode_combined_message(
    payload: bytes,
    *,
    context: DecodeContext,
) -> CapturedEvidence:
    """Decode one combined-stream frame into one canonical event."""

    if not isinstance(payload, bytes) or not payload:
        raise LiveSchemaError("live payload must be non-empty bytes")
    try:
        envelope = _COMBINED_DECODER.decode(payload)
        stream = envelope.stream
        raw_event = bytes(envelope.data)
        if stream.endswith("@kline_1m"):
            event = _decode_kline(raw_event, stream, payload, context)
        elif stream.endswith("@aggTrade"):
            event = _decode_aggregate_trade(raw_event, stream, payload, context)
        elif stream.endswith("@bookTicker"):
            event = _decode_book_ticker(raw_event, stream, payload, context)
        else:
            raise LiveSchemaError("stream is not admitted by CAP-01")
    except LiveCaptureError:
        raise
    except (msgspec.DecodeError, EvidenceAdmissionError, InvalidOperation) as error:
        raise LiveSchemaError("live payload failed schema admission") from error
    return CapturedEvidence(stream=stream, raw_payload=payload, event=event)


def _decode_kline(
    raw: bytes,
    stream: str,
    payload: bytes,
    context: DecodeContext,
) -> KlineEvent:
    wire = _KLINE_DECODER.decode(raw)
    if wire.event_type != "kline" or wire.symbol != wire.kline.symbol:
        raise LiveSchemaError("kline discriminant or symbol is inconsistent")
    _require_stream_symbol(stream, wire.symbol)
    first_trade_id: int | None = wire.kline.first_trade_id
    last_trade_id: int | None = wire.kline.last_trade_id
    if wire.kline.trade_count == 0 and first_trade_id == -1 and last_trade_id == -1:
        # Binance uses -1/-1 on live klines that contain no trades. The
        # canonical contract represents unavailable bounds as a paired None.
        first_trade_id = None
        last_trade_id = None
    quality = (
        ObservationQuality(QualityState.VALID, complete=True)
        if wire.kline.closed
        else ObservationQuality(QualityState.PROVISIONAL, complete=False)
    )
    return KlineEvent(
        instrument=_instrument(wire.symbol),
        raw_symbol=wire.symbol,
        interval=wire.kline.interval,
        source_event_time=_timestamp(wire.event_time, context.time_unit),
        open_time=_timestamp(wire.kline.open_time, context.time_unit),
        close_time=_timestamp(wire.kline.close_time, context.time_unit),
        open_price=_decimal(wire.kline.open_price),
        high_price=_decimal(wire.kline.high_price),
        low_price=_decimal(wire.kline.low_price),
        close_price=_decimal(wire.kline.close_price),
        base_volume=_decimal(wire.kline.base_volume),
        quote_turnover=_decimal(wire.kline.quote_turnover),
        taker_buy_base_volume=_decimal(wire.kline.taker_buy_base_volume),
        taker_buy_quote_turnover=_decimal(wire.kline.taker_buy_quote_turnover),
        trade_count=wire.kline.trade_count,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        closed=wire.kline.closed,
        provenance=_provenance(stream, payload, context),
        quality=quality,
    )


def _decode_aggregate_trade(
    raw: bytes,
    stream: str,
    payload: bytes,
    context: DecodeContext,
) -> AggregateTradeEvent:
    wire = _AGG_TRADE_DECODER.decode(raw)
    if wire.event_type != "aggTrade":
        raise LiveSchemaError("aggregate-trade discriminant is invalid")
    _require_stream_symbol(stream, wire.symbol)
    return AggregateTradeEvent(
        instrument=_instrument(wire.symbol),
        raw_symbol=wire.symbol,
        aggregate_trade_id=wire.aggregate_trade_id,
        first_trade_id=wire.first_trade_id,
        last_trade_id=wire.last_trade_id,
        source_event_time=_timestamp(wire.event_time, context.time_unit),
        trade_time=_timestamp(wire.trade_time, context.time_unit),
        price=_decimal(wire.price),
        quantity=_decimal(wire.quantity),
        buyer_is_maker=wire.buyer_is_maker,
        provenance=_provenance(stream, payload, context),
        quality=ObservationQuality(QualityState.VALID, complete=True),
    )


def _decode_book_ticker(
    raw: bytes,
    stream: str,
    payload: bytes,
    context: DecodeContext,
) -> BookTickerEvent:
    wire = _BOOK_TICKER_DECODER.decode(raw)
    _require_stream_symbol(stream, wire.symbol)
    return BookTickerEvent(
        instrument=_instrument(wire.symbol),
        raw_symbol=wire.symbol,
        update_id=wire.update_id,
        source_event_time=None,
        bid_price=_decimal(wire.bid_price),
        bid_quantity=_decimal(wire.bid_quantity),
        ask_price=_decimal(wire.ask_price),
        ask_quantity=_decimal(wire.ask_quantity),
        provenance=_provenance(stream, payload, context),
        quality=ObservationQuality(QualityState.VALID, complete=True),
    )


def _timestamp(value: int, unit: TimeUnit) -> EpochTimestamp:
    return EpochTimestamp(value, unit)


def _provenance(
    stream: str,
    payload: bytes,
    context: DecodeContext,
) -> Provenance:
    digest = PayloadDigest.sha256(payload)
    return Provenance(
        source=SourceDescriptor(
            endpoint=context.endpoint,
            channel=stream,
            schema_version=LIVE_SCHEMA_VERSION,
        ),
        ingestion_run_id=context.ingestion_run_id,
        receipt=context.receipt,
        payload_digest=digest,
        connection_generation=context.connection_generation,
        raw_payload_reference=digest.value,
    )


def _require_stream_symbol(stream: str, symbol: str) -> None:
    expected_prefix = f"{symbol.lower()}@"
    if not stream.startswith(expected_prefix):
        raise LiveSchemaError("combined stream and payload symbol disagree")


@dataclass(frozen=True, slots=True)
class SubscriptionPlan:
    """Bounded breadth and hot-set streams with an always-on BTC reference."""

    breadth_symbols: tuple[str, ...]
    hot_symbols: tuple[str, ...]
    breadth_shard_size: int = DEFAULT_BREADTH_SHARD_SIZE
    btc_reference: str = "BTCUSDT"

    def __post_init__(self) -> None:
        if not 1 <= self.breadth_shard_size <= 300:
            raise ValueError("breadth shard size must be between 1 and 300")
        breadth = _unique_symbols(self.breadth_symbols)
        hot = _unique_symbols(self.hot_symbols)
        reference = _instrument(self.btc_reference).symbol
        if len(hot) > MAX_HOT_SYMBOLS:
            raise ValueError("hot symbol count exceeds the admitted bound")
        object.__setattr__(self, "breadth_symbols", breadth)
        object.__setattr__(self, "hot_symbols", hot)
        object.__setattr__(self, "btc_reference", reference)

    @property
    def breadth_shards(self) -> tuple[tuple[str, ...], ...]:
        streams = tuple(f"{symbol.lower()}@kline_1m" for symbol in self.breadth_symbols)
        size = self.breadth_shard_size
        return tuple(
            streams[index : index + size] for index in range(0, len(streams), size)
        )

    @property
    def hot_streams(self) -> tuple[str, ...]:
        symbols = _unique_symbols((*self.hot_symbols, self.btc_reference))
        return tuple(
            stream
            for symbol in symbols
            for stream in (
                f"{symbol.lower()}@aggTrade",
                f"{symbol.lower()}@bookTicker",
            )
        )


def _unique_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    admitted: dict[str, None] = {}
    for symbol in symbols:
        canonical = _instrument(symbol).symbol
        admitted.setdefault(canonical, None)
    return tuple(admitted)


def _utc_day(timestamp_ns: int) -> date:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).date()


def combined_stream_url(streams: tuple[str, ...]) -> str:
    if not streams:
        raise ValueError("at least one stream is required")
    if len(streams) > CAPTURE_STREAMS_PER_CONNECTION:
        raise ValueError("connection stream count exceeds the admitted bound")
    encoded = quote("/".join(streams), safe="/@_")
    return f"{PUBLIC_WS_BASE}/stream?streams={encoded}&timeUnit=MICROSECOND"


class LiveAdmissionState:
    """Bounded overlap deduplication and per-instrument continuity refusal."""

    def __init__(
        self,
        *,
        recent_identity_capacity: int = 100_000,
        daily_stats_capacity: int = 10,
    ) -> None:
        if recent_identity_capacity <= 0 or daily_stats_capacity <= 0:
            raise ValueError("admission capacities must be positive")
        self._recent_capacity = recent_identity_capacity
        self._daily_stats_capacity = daily_stats_capacity
        self._recent: OrderedDict[object, MarketEvidence] = OrderedDict()
        self._recent_high_water_mark = 0
        self._recent_evictions = 0
        self._daily_stats: dict[date, _AdmissionDayCounters] = {}
        self._last_final_kline: dict[InstrumentId, KlineEvent] = {}
        self._last_aggregate_id: dict[InstrumentId, int] = {}
        self._last_book_update_id: dict[InstrumentId, int] = {}

    @property
    def recent_identity_count(self) -> int:
        return len(self._recent)

    @property
    def recent_identity_high_water_mark(self) -> int:
        return self._recent_high_water_mark

    @property
    def recent_identity_evictions(self) -> int:
        return self._recent_evictions

    @property
    def tracked_instrument_count(self) -> int:
        return len(
            set(self._last_final_kline)
            | set(self._last_aggregate_id)
            | set(self._last_book_update_id)
        )

    def stats_for_day(self, day: date) -> AdmissionStats:
        stats = self._daily_stats.get(day)
        if stats is None:
            return AdmissionStats(self._recent_capacity, 0, 0)
        return AdmissionStats(
            self._recent_capacity,
            stats.recent_identity_high_water_mark,
            stats.recent_identity_evictions,
        )

    def discard_stats_before(self, day: date) -> None:
        for expired in tuple(value for value in self._daily_stats if value < day):
            del self._daily_stats[expired]

    def admit(self, event: MarketEvidence) -> AdmissionDecision:
        daily = self._daily_for(event)
        daily.recent_identity_high_water_mark = max(
            daily.recent_identity_high_water_mark,
            len(self._recent),
        )
        if isinstance(event, KlineEvent):
            return self._admit_kline(event)

        previous = self._recent.get(event.identity)
        if previous is not None:
            self._recent.move_to_end(event.identity)
            state = classify_reobservation(previous, event)
            return AdmissionDecision(
                state=state,
                accepted_for_downstream=False,
                diagnostics=("OVERLAP_REOBSERVATION",),
            )

        if isinstance(event, AggregateTradeEvent):
            return self._admit_aggregate_trade(event)
        if isinstance(event, BookTickerEvent):
            return self._admit_book_ticker(event)

        self._remember(event)
        return AdmissionDecision(QualityState.VALID, True)

    def _admit_kline(self, event: KlineEvent) -> AdmissionDecision:
        if not event.closed:
            return AdmissionDecision(
                state=QualityState.PROVISIONAL,
                accepted_for_downstream=False,
                diagnostics=("OPEN_KLINE",),
            )
        previous = self._last_final_kline.get(event.instrument)
        if previous is not None and previous.identity == event.identity:
            return AdmissionDecision(
                state=classify_reobservation(previous, event),
                accepted_for_downstream=False,
                diagnostics=("OVERLAP_REOBSERVATION",),
            )
        self._last_final_kline[event.instrument] = event
        return AdmissionDecision(QualityState.VALID, True)

    def _admit_aggregate_trade(
        self,
        event: AggregateTradeEvent,
    ) -> AdmissionDecision:
        previous_id = self._last_aggregate_id.get(event.instrument)
        self._remember(event)
        self._last_aggregate_id[event.instrument] = max(
            event.aggregate_trade_id,
            previous_id if previous_id is not None else event.aggregate_trade_id,
        )
        if previous_id is not None and event.aggregate_trade_id < previous_id:
            return AdmissionDecision(
                QualityState.CONFLICT,
                False,
                diagnostics=("AGGREGATE_ID_REGRESSION",),
            )
        if previous_id is not None and event.aggregate_trade_id > previous_id + 1:
            return AdmissionDecision(
                QualityState.GAP,
                False,
                recovery_required=True,
                diagnostics=("AGGREGATE_ID_GAP",),
            )
        return AdmissionDecision(QualityState.VALID, True)

    def _admit_book_ticker(self, event: BookTickerEvent) -> AdmissionDecision:
        previous_id = self._last_book_update_id.get(event.instrument)
        self._remember(event)
        if previous_id is not None and event.update_id < previous_id:
            return AdmissionDecision(
                QualityState.CONFLICT,
                False,
                diagnostics=("BOOK_UPDATE_REGRESSION",),
            )
        self._last_book_update_id[event.instrument] = event.update_id
        return AdmissionDecision(QualityState.VALID, True)

    def _remember(self, event: MarketEvidence) -> None:
        daily = self._daily_for(event)
        self._recent[event.identity] = event
        self._recent.move_to_end(event.identity)
        evictions = 0
        while len(self._recent) > self._recent_capacity:
            self._recent.popitem(last=False)
            self._recent_evictions += 1
            evictions += 1
        self._recent_high_water_mark = max(
            self._recent_high_water_mark,
            len(self._recent),
        )
        daily.recent_identity_high_water_mark = max(
            daily.recent_identity_high_water_mark,
            len(self._recent),
        )
        daily.recent_identity_evictions += evictions

    def _daily_for(self, event: MarketEvidence) -> _AdmissionDayCounters:
        day = _utc_day(event.provenance.receipt.wall_time_ns)
        stats = self._daily_stats.get(day)
        if stats is not None:
            return stats
        if len(self._daily_stats) >= self._daily_stats_capacity:
            raise RuntimeError("daily admission-stat capacity is exhausted")
        stats = _AdmissionDayCounters()
        self._daily_stats[day] = stats
        return stats


@dataclass(frozen=True, slots=True)
class AdmissionStats:
    recent_identity_capacity: int
    recent_identity_high_water_mark: int
    recent_identity_evictions: int


@dataclass(slots=True)
class _AdmissionDayCounters:
    recent_identity_high_water_mark: int = 0
    recent_identity_evictions: int = 0


@dataclass(frozen=True, slots=True)
class QueueStats:
    capacity: int
    high_water_mark: int
    overflows: int


@dataclass(slots=True)
class _QueueDayCounters:
    high_water_mark: int = 0
    overflows: int = 0


class _BoundedQueueMetrics:
    def __init__(self, capacity: int, daily_stats_capacity: int) -> None:
        if daily_stats_capacity <= 0:
            raise ValueError("daily queue-stat capacity must be positive")
        self.capacity = capacity
        self.high_water_mark = 0
        self.overflows = 0
        self._daily_stats_capacity = daily_stats_capacity
        self._daily: dict[date, _QueueDayCounters] = {}

    def record_enqueued(self, timestamp_ns: int, size: int) -> None:
        daily = self._daily_for(timestamp_ns)
        self.high_water_mark = max(self.high_water_mark, size)
        daily.high_water_mark = max(daily.high_water_mark, size)

    def record_overflow(self, timestamp_ns: int) -> None:
        daily = self._daily_for(timestamp_ns)
        self.overflows += 1
        daily.overflows += 1

    def prepare_day(self, timestamp_ns: int) -> None:
        self._daily_for(timestamp_ns)

    def stats_for_day(self, day: date) -> QueueStats:
        daily = self._daily.get(day)
        if daily is None:
            return QueueStats(self.capacity, 0, 0)
        return QueueStats(self.capacity, daily.high_water_mark, daily.overflows)

    def discard_stats_before(self, day: date) -> None:
        for expired in tuple(value for value in self._daily if value < day):
            del self._daily[expired]

    def _daily_for(self, timestamp_ns: int) -> _QueueDayCounters:
        day = _utc_day(timestamp_ns)
        stats = self._daily.get(day)
        if stats is not None:
            return stats
        if len(self._daily) >= self._daily_stats_capacity:
            raise RuntimeError("daily queue-stat capacity is exhausted")
        stats = _QueueDayCounters()
        self._daily[day] = stats
        return stats


class _BoundedQueue[T]:
    """Non-blocking producer boundary with explicit overflow refusal.

    The two queues below differ only in what they carry, how a wall-clock stamp
    is read from it, and the diagnostic an overflow reports. The capacity guard,
    the daily metric bookkeeping and the shape of every admission are shared
    here, so an overflow cannot be accounted for one way by the capture queue and
    another way by the raw queue.
    """

    _overflow_diagnostic: str

    def __init__(self, capacity: int, *, daily_stats_capacity: int = 10) -> None:
        if capacity <= 0:
            raise ValueError("queue capacity must be positive")
        self._queue: asyncio.Queue[T] = asyncio.Queue(capacity)
        self._capacity = capacity
        self._metrics = _BoundedQueueMetrics(capacity, daily_stats_capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def high_water_mark(self) -> int:
        return self._metrics.high_water_mark

    @property
    def overflow_count(self) -> int:
        return self._metrics.overflows

    def offer(self, item: T) -> QueueAdmission:
        timestamp_ns = self._wall_time_ns(item)
        self._metrics.prepare_day(timestamp_ns)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._metrics.record_overflow(timestamp_ns)
            return QueueAdmission(
                enqueued=False,
                state=QualityState.INCOMPLETE,
                diagnostics=(self._overflow_diagnostic,),
            )
        self._metrics.record_enqueued(timestamp_ns, self.size)
        return QueueAdmission(enqueued=True, state=QualityState.VALID)

    def stats_for_day(self, day: date) -> QueueStats:
        return self._metrics.stats_for_day(day)

    def discard_stats_before(self, day: date) -> None:
        self._metrics.discard_stats_before(day)

    async def get(self) -> T:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def _wall_time_ns(self, item: T) -> int:
        raise NotImplementedError("a bounded queue must know how to time its items")


class BoundedCaptureQueue(_BoundedQueue[CapturedEvidence]):
    """Non-blocking producer boundary for decoded capture evidence."""

    _overflow_diagnostic = "CAPTURE_QUEUE_OVERFLOW"

    def _wall_time_ns(self, item: CapturedEvidence) -> int:
        return item.event.provenance.receipt.wall_time_ns


class BoundedRawCaptureQueue(_BoundedQueue[RawMessageObservation]):
    """Non-blocking pre-decode audit boundary for exact received frames."""

    _overflow_diagnostic = "RAW_CAPTURE_QUEUE_OVERFLOW"

    def _wall_time_ns(self, item: RawMessageObservation) -> int:
        return item.receipt.wall_time_ns


class ConnectionAttemptBudget:
    """Process-wide conservative share of the venue connection-attempt ceiling."""

    def __init__(
        self,
        *,
        venue_capacity: int = VENUE_CONNECTION_ATTEMPTS_PER_5_MINUTES,
        interval_seconds: float = 5 * 60,
        utilization: Decimal = Decimal("0.5"),
    ) -> None:
        self._budget = RestWeightBudget(
            venue_capacity=venue_capacity,
            interval_seconds=interval_seconds,
            utilization=utilization,
        )

    @property
    def available(self) -> Decimal:
        return self._budget.available

    def try_acquire(self, *, now: float) -> bool:
        return self._budget.try_acquire(1, now=now)


_PROCESS_CONNECTION_ATTEMPTS = ConnectionAttemptBudget()


def _client_websocket_timeout() -> aiohttp.ClientWSTimeout:
    # aiohttp 3.14 exposes these runtime parameters, ahead of ty's constructor model.
    factory: Any = aiohttp.ClientWSTimeout
    return factory(ws_receive=90, ws_close=0.1)


@dataclass(slots=True)
class _CaptureCounters:
    received: int = 0
    enqueued: int = 0
    refused: int = 0
    reconnects: int = 0
    rotations: int = 0
    duplicates: int = 0
    conflicts: int = 0
    gaps: int = 0
    provisional: int = 0
    schema_mismatches: int = 0
    queue_overflows: int = 0
    raw_retained: int = 0
    raw_suppressed: int = 0
    _track_daily: bool = field(default=True, repr=False)
    _daily_capacity: int = field(default=10, repr=False)
    _daily: dict[date, _CaptureCounters] = field(default_factory=dict, repr=False)

    def reached(self, maximum: int | None) -> bool:
        return maximum is not None and self.received >= maximum

    def snapshot(self) -> CaptureStats:
        return CaptureStats(
            messages_received=self.received,
            messages_enqueued=self.enqueued,
            messages_refused=self.refused,
            reconnects=self.reconnects,
            rotations=self.rotations,
            duplicate_messages=self.duplicates,
            conflict_messages=self.conflicts,
            gap_messages=self.gaps,
            provisional_messages=self.provisional,
            schema_mismatches=self.schema_mismatches,
            queue_overflows=self.queue_overflows,
            raw_messages_retained=self.raw_retained,
            raw_messages_suppressed=self.raw_suppressed,
        )

    def snapshot_day(self, day: date) -> CaptureStats:
        counters = self._daily.get(day)
        if counters is None:
            return CaptureStats(0, 0, 0, 0, 0)
        return counters.snapshot()

    def discard_days_before(self, day: date) -> None:
        for expired in tuple(value for value in self._daily if value < day):
            del self._daily[expired]

    def add(
        self,
        timestamp_ns: int,
        *,
        received: int = 0,
        enqueued: int = 0,
        refused: int = 0,
        reconnects: int = 0,
        rotations: int = 0,
        duplicates: int = 0,
        conflicts: int = 0,
        gaps: int = 0,
        provisional: int = 0,
        schema_mismatches: int = 0,
        queue_overflows: int = 0,
        raw_retained: int = 0,
        raw_suppressed: int = 0,
    ) -> None:
        increments = (
            received,
            enqueued,
            refused,
            reconnects,
            rotations,
            duplicates,
            conflicts,
            gaps,
            provisional,
            schema_mismatches,
            queue_overflows,
            raw_retained,
            raw_suppressed,
        )
        if timestamp_ns <= 0 or any(value < 0 for value in increments):
            raise ValueError("capture counter observation is invalid")
        self.received += received
        self.enqueued += enqueued
        self.refused += refused
        self.reconnects += reconnects
        self.rotations += rotations
        self.duplicates += duplicates
        self.conflicts += conflicts
        self.gaps += gaps
        self.provisional += provisional
        self.schema_mismatches += schema_mismatches
        self.queue_overflows += queue_overflows
        self.raw_retained += raw_retained
        self.raw_suppressed += raw_suppressed
        if not self._track_daily:
            return
        day = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).date()
        counters = self._daily.get(day)
        if counters is None:
            if len(self._daily) >= self._daily_capacity:
                raise RuntimeError("daily capture-stat capacity is exhausted")
            counters = _CaptureCounters(_track_daily=False)
            self._daily[day] = counters
        counters.add(
            timestamp_ns,
            received=received,
            enqueued=enqueued,
            refused=refused,
            reconnects=reconnects,
            rotations=rotations,
            duplicates=duplicates,
            conflicts=conflicts,
            gaps=gaps,
            provisional=provisional,
            schema_mismatches=schema_mismatches,
            queue_overflows=queue_overflows,
            raw_retained=raw_retained,
            raw_suppressed=raw_suppressed,
        )

    def observe_refusal(
        self,
        decision: AdmissionDecision,
        *,
        timestamp_ns: int,
    ) -> None:
        duplicates = 0
        conflicts = 0
        gaps = 0
        provisional = 0
        if decision.state is QualityState.DUPLICATE:
            duplicates = 1
        elif decision.state is QualityState.CONFLICT:
            conflicts = 1
        elif decision.state is QualityState.GAP:
            gaps = 1
        elif decision.state is QualityState.PROVISIONAL:
            provisional = 1
        self.add(
            timestamp_ns,
            refused=1,
            duplicates=duplicates,
            conflicts=conflicts,
            gaps=gaps,
            provisional=provisional,
        )


@dataclass(frozen=True, slots=True)
class _SocketWait:
    message: aiohttp.WSMessage | None
    rotation_due: bool


class BinanceLiveCollector:
    """Combined-stream reader with proactive overlap rotation and reconnects."""

    def __init__(
        self,
        *,
        streams: tuple[str, ...],
        ingestion_run_id: UUID,
        queue: BoundedCaptureQueue,
        admission: LiveAdmissionState | None = None,
        raw_queue: BoundedRawCaptureQueue | None = None,
        reconnect_delay_seconds: float = 1.0,
        reconnect_cap_seconds: float = 30.0,
        rotation_seconds: float = 23 * 60 * 60,
        overlap_ready_seconds: float = 15.0,
        connection_attempt_budget: ConnectionAttemptBudget | None = None,
        daily_stats_capacity: int = 10,
    ) -> None:
        if reconnect_delay_seconds <= 0:
            raise ValueError("reconnect delay must be positive")
        if reconnect_cap_seconds < reconnect_delay_seconds:
            raise ValueError("reconnect cap must not be below its base delay")
        if rotation_seconds <= 0 or overlap_ready_seconds <= 0:
            raise ValueError("rotation bounds must be positive")
        if rotation_seconds >= 24 * 60 * 60:
            raise ValueError("connections must rotate before the venue limit")
        if daily_stats_capacity <= 0:
            raise ValueError("daily stats capacity must be positive")
        self.url = combined_stream_url(streams)
        self.ingestion_run_id = ingestion_run_id
        self.queue = queue
        self.admission = admission or LiveAdmissionState()
        self.raw_queue = raw_queue
        self._last_retained_provisional: dict[InstrumentId, int] = {}
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.reconnect_cap_seconds = reconnect_cap_seconds
        self.rotation_seconds = rotation_seconds
        self.connection_attempt_budget = (
            connection_attempt_budget or _PROCESS_CONNECTION_ATTEMPTS
        )
        self.overlap_ready_seconds = overlap_ready_seconds
        self.daily_stats_capacity = daily_stats_capacity
        self._counters = _CaptureCounters(_daily_capacity=daily_stats_capacity)

    def stats_for_day(self, day: date) -> CaptureStats:
        """Return an exact receive-UTC-day snapshot without stopping the collector."""

        return self._counters.snapshot_day(day)

    def discard_stats_before(self, day: date) -> None:
        """Release finalized daily counters without changing live socket state."""

        self._counters.discard_days_before(day)

    async def run(
        self,
        *,
        stop: asyncio.Event,
        max_messages: int | None = None,
        ready: asyncio.Event | None = None,
    ) -> CaptureStats:
        if max_messages is not None and max_messages <= 0:
            raise ValueError("maximum messages must be positive")

        counters = _CaptureCounters(_daily_capacity=self.daily_stats_capacity)
        self._counters = counters
        generation = 0
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not stop.is_set() and not counters.reached(max_messages):
                try:
                    generation = await self._run_connection(
                        session,
                        stop=stop,
                        generation=generation,
                        counters=counters,
                        max_messages=max_messages,
                        ready=ready,
                    )
                except (TimeoutError, aiohttp.ClientError, LiveCaptureError):
                    if stop.is_set():
                        break

                if stop.is_set() or counters.reached(max_messages):
                    break
                generation += 1
                counters.add(time_ns(), reconnects=1)
                await self._wait_before_reconnect(
                    stop,
                    attempt=counters.reconnects,
                )

        return counters.snapshot()

    async def _run_connection(
        self,
        session: aiohttp.ClientSession,
        *,
        stop: asyncio.Event,
        generation: int,
        counters: _CaptureCounters,
        max_messages: int | None,
        ready: asyncio.Event | None,
    ) -> int:
        socket = await self._connect(session)
        if ready is not None:
            ready.set()
        rotation_due = self._next_rotation()
        try:
            while not stop.is_set() and not counters.reached(max_messages):
                wait = await _wait_for_socket(
                    socket,
                    stop,
                    rotate_after=self._rotation_delay(rotation_due),
                )
                if wait.rotation_due:
                    replacement = await self._open_replacement(
                        session,
                        socket,
                        stop=stop,
                        generation=generation + 1,
                        counters=counters,
                    )
                    if replacement is None:
                        break
                    socket = replacement
                    generation += 1
                    counters.add(time_ns(), rotations=1)
                    rotation_due = self._next_rotation()
                    continue

                if wait.message is None or _socket_finished(wait.message):
                    break
                if wait.message.type is aiohttp.WSMsgType.TEXT:
                    self._admit_text(
                        wait.message,
                        generation=generation,
                        counters=counters,
                    )
        finally:
            if not socket.closed:
                await socket.close()
        return generation

    async def _open_replacement(
        self,
        session: aiohttp.ClientSession,
        old_socket: aiohttp.ClientWebSocketResponse,
        *,
        stop: asyncio.Event,
        generation: int,
        counters: _CaptureCounters,
    ) -> aiohttp.ClientWebSocketResponse | None:
        replacement = await self._connect(session)
        try:
            wait = await _wait_for_socket(
                replacement,
                stop,
                rotate_after=self.overlap_ready_seconds,
            )
            if stop.is_set():
                await replacement.close()
                return None
            if wait.rotation_due or wait.message is None:
                raise LiveResourceError("replacement connection did not become ready")
            if wait.message.type is not aiohttp.WSMsgType.TEXT:
                raise LiveCaptureError("replacement connection closed before evidence")
            self._admit_text(
                wait.message,
                generation=generation,
                counters=counters,
            )
            await old_socket.close()
            return replacement
        except BaseException:
            if not replacement.closed:
                await replacement.close()
            raise

    async def _connect(
        self,
        session: aiohttp.ClientSession,
    ) -> aiohttp.ClientWebSocketResponse:
        if not self.connection_attempt_budget.try_acquire(now=monotonic()):
            raise LiveResourceError("WebSocket connection-attempt budget is exhausted")
        return await session.ws_connect(
            self.url,
            timeout=_client_websocket_timeout(),
            autoping=True,
            heartbeat=30,
            max_msg_size=256 * 1024,
        )

    def _admit_text(
        self,
        message: aiohttp.WSMessage,
        *,
        generation: int,
        counters: _CaptureCounters,
    ) -> None:
        receipt = LocalReceipt(time_ns(), monotonic_ns())
        payload = message.data.encode("utf-8")
        counters.add(receipt.wall_time_ns, received=1)
        context = DecodeContext(
            ingestion_run_id=self.ingestion_run_id,
            receipt=receipt,
            connection_generation=generation,
        )
        try:
            captured = decode_combined_message(payload, context=context)
        except LiveSchemaError:
            counters.add(
                receipt.wall_time_ns,
                refused=1,
                schema_mismatches=1,
            )
            self._retain_raw(
                payload,
                receipt=receipt,
                generation=generation,
                reason="SCHEMA_MISMATCH",
                counters=counters,
            )
            return

        decision = self.admission.admit(captured.event)
        self._retain_decoded_raw(captured, decision=decision, counters=counters)
        if not decision.accepted_for_downstream:
            counters.observe_refusal(decision, timestamp_ns=receipt.wall_time_ns)
            return
        if self.queue.offer(captured).enqueued:
            counters.add(receipt.wall_time_ns, enqueued=1)
            return
        counters.add(receipt.wall_time_ns, refused=1, queue_overflows=1)

    def _retain_decoded_raw(
        self,
        captured: CapturedEvidence,
        *,
        decision: AdmissionDecision,
        counters: _CaptureCounters,
    ) -> None:
        event = captured.event
        reason = decision.state.value.upper()
        if isinstance(event, KlineEvent) and not event.closed:
            open_time = event.open_time.epoch_microseconds
            previous = self._last_retained_provisional.get(event.instrument)
            if previous == open_time:
                if self.raw_queue is not None:
                    counters.add(
                        event.provenance.receipt.wall_time_ns,
                        raw_suppressed=1,
                    )
                return
            self._last_retained_provisional[event.instrument] = open_time
            reason = "PROVISIONAL_SAMPLE"
        self._retain_raw(
            captured.raw_payload,
            receipt=event.provenance.receipt,
            generation=event.provenance.connection_generation or 0,
            reason=reason,
            counters=counters,
        )

    def _retain_raw(
        self,
        payload: bytes,
        *,
        receipt: LocalReceipt,
        generation: int,
        reason: str,
        counters: _CaptureCounters,
    ) -> None:
        if self.raw_queue is None:
            return
        admission = self.raw_queue.offer(
            RawMessageObservation(
                raw_payload=payload,
                ingestion_run_id=self.ingestion_run_id,
                receipt=receipt,
                connection_generation=generation,
                retention_reason=reason,
            )
        )
        if admission.enqueued:
            counters.add(receipt.wall_time_ns, raw_retained=1)
        else:
            counters.add(receipt.wall_time_ns, queue_overflows=1)

    async def _wait_before_reconnect(
        self,
        stop: asyncio.Event,
        *,
        attempt: int,
    ) -> None:
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=self._reconnect_delay(attempt),
            )
        except TimeoutError:
            pass

    def _reconnect_delay(self, attempt: int) -> float:
        exponent = min(max(attempt - 1, 0), 10)
        ceiling = min(
            self.reconnect_delay_seconds * 2**exponent,
            self.reconnect_cap_seconds,
        )
        seed = (
            self.ingestion_run_id.bytes
            + self.url.encode("utf-8")
            + attempt.to_bytes(8, "big")
        )
        fraction = int.from_bytes(hashlib.sha256(seed).digest()[:2], "big") / 65_535
        return ceiling * (0.5 + fraction / 2)

    def _next_rotation(self) -> float:
        return asyncio.get_running_loop().time() + self.rotation_seconds

    @staticmethod
    def _rotation_delay(rotation_due: float) -> float:
        return max(0.0, rotation_due - asyncio.get_running_loop().time())


async def _wait_for_socket(
    socket: aiohttp.ClientWebSocketResponse,
    stop: asyncio.Event,
    *,
    rotate_after: float,
) -> _SocketWait:
    receive_task = asyncio.create_task(socket.receive())
    stop_task = asyncio.create_task(stop.wait())
    rotation_task = asyncio.create_task(asyncio.sleep(rotate_after))
    done, pending = await asyncio.wait(
        {receive_task, stop_task, rotation_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if stop_task in done and stop_task.result():
        if receive_task in done:
            receive_task.result()
        return _SocketWait(message=None, rotation_due=False)
    if receive_task in done:
        return _SocketWait(message=receive_task.result(), rotation_due=False)
    return _SocketWait(message=None, rotation_due=True)


def _socket_finished(message: aiohttp.WSMessage) -> bool:
    return message.type in {
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.ERROR,
    }


def monotonic_seconds() -> float:
    """Expose the operational monotonic clock for rate-budget owners."""

    return monotonic()
