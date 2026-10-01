"""Canonical, transport-independent market evidence contracts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Protocol
from uuid import UUID

_CANONICAL_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._-]{1,39}")
_ASSET_TOKEN = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,39}")
_LOWER_TOKEN = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_DIAGNOSTIC_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_INTERVAL = re.compile(r"[1-9][0-9]*(?:s|m|h|d|w|M)")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_SIGNED_64 = 2**63 - 1


class EvidenceAdmissionError(ValueError):
    """A value cannot enter the canonical market-evidence domain."""


class Environment(StrEnum):
    """Venue environment that produced an observation."""

    PRODUCTION = "production"
    TESTNET = "testnet"


class TimeUnit(StrEnum):
    """Source timestamp units admitted by the initial venue contract."""

    MILLISECOND = "ms"
    MICROSECOND = "us"


class EventKind(StrEnum):
    """Stable discriminants for canonical evidence variants."""

    KLINE = "kline"
    AGGREGATE_TRADE = "aggregate_trade"
    BOOK_TICKER = "book_ticker"
    INSTRUMENT_METADATA = "instrument_metadata"


class QualityState(StrEnum):
    """Exactly one quality disposition for an observation or window."""

    PROVISIONAL = "provisional"
    VALID = "valid"
    DUPLICATE = "duplicate"
    LATE = "late"
    GAP = "gap"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    SCHEMA_MISMATCH = "schema_mismatch"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"
    UNAVAILABLE = "unavailable"


class MetadataObservation(StrEnum):
    """Whether point-in-time metadata was seen or reconstructed."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class InstrumentStatus(StrEnum):
    """Venue instrument states admitted by metadata schema version 1."""

    PRE_TRADING = "PRE_TRADING"
    TRADING = "TRADING"
    POST_TRADING = "POST_TRADING"
    END_OF_DAY = "END_OF_DAY"
    HALT = "HALT"
    AUCTION_MATCH = "AUCTION_MATCH"
    BREAK = "BREAK"


@dataclass(frozen=True, slots=True)
class VenueId:
    """Canonical venue and market identity, independent of an endpoint."""

    name: str
    market: str

    def __post_init__(self) -> None:
        _require_pattern(self.name, _LOWER_TOKEN, "venue name")
        _require_pattern(self.market, _LOWER_TOKEN, "venue market")


@dataclass(frozen=True, slots=True)
class InstrumentId:
    """A venue-native symbol in one environment."""

    venue: VenueId
    environment: Environment
    symbol: str

    def __post_init__(self) -> None:
        _require_pattern(self.symbol, _CANONICAL_SYMBOL, "canonical symbol")


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """Wire/archive route and decoder schema that supplied evidence."""

    endpoint: str
    channel: str
    schema_version: int

    def __post_init__(self) -> None:
        _require_text(self.endpoint, "source endpoint", maximum=512)
        _require_text(self.channel, "source channel", maximum=128)
        _require_int(self.schema_version, "schema version", minimum=1)


@dataclass(frozen=True, slots=True)
class EpochTimestamp:
    """A raw source timestamp whose unit is preserved and normalized once."""

    raw_value: int
    unit: TimeUnit

    def __post_init__(self) -> None:
        raw_value = _require_int(
            self.raw_value,
            "timestamp",
            minimum=0,
            maximum=_MAX_SIGNED_64,
        )
        if not isinstance(self.unit, TimeUnit):
            raise EvidenceAdmissionError("timestamp unit is not supported")
        if raw_value > _MAX_SIGNED_64 // self._microsecond_factor:
            raise EvidenceAdmissionError("normalized timestamp exceeds int64")

    @classmethod
    def from_raw(cls, raw_value: int, unit: str) -> EpochTimestamp:
        """Admit a raw integer and fail closed on an unknown unit."""

        try:
            admitted_unit = TimeUnit(unit)
        except (TypeError, ValueError) as error:
            raise EvidenceAdmissionError("timestamp unit is not supported") from error
        return cls(raw_value=raw_value, unit=admitted_unit)

    @property
    def epoch_microseconds(self) -> int:
        """Return the single canonical comparison representation."""

        return self.raw_value * self._microsecond_factor

    @property
    def _microsecond_factor(self) -> int:
        if self.unit is TimeUnit.MILLISECOND:
            return 1_000
        return 1


@dataclass(frozen=True, slots=True)
class LocalReceipt:
    """Local wall and monotonic clocks sampled before expensive decoding."""

    wall_time_ns: int
    monotonic_ns: int

    def __post_init__(self) -> None:
        _require_int(
            self.wall_time_ns,
            "receipt wall time",
            minimum=1,
            maximum=_MAX_SIGNED_64,
        )
        _require_int(
            self.monotonic_ns,
            "receipt monotonic time",
            minimum=0,
            maximum=_MAX_SIGNED_64,
        )


@dataclass(frozen=True, slots=True)
class PayloadDigest:
    """Content identity of the exact source payload."""

    value: str

    def __post_init__(self) -> None:
        _require_pattern(self.value, _SHA256, "payload digest")

    @classmethod
    def sha256(cls, payload: bytes) -> PayloadDigest:
        if not isinstance(payload, bytes):
            raise EvidenceAdmissionError("payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        return cls(f"sha256:{digest}")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Source and local-observation facts retained with every payload."""

    source: SourceDescriptor
    ingestion_run_id: UUID
    receipt: LocalReceipt
    payload_digest: PayloadDigest
    connection_generation: int | None = None
    source_revision: str | None = None
    raw_payload_reference: str | None = None

    def __post_init__(self) -> None:
        if self.connection_generation is not None:
            _require_int(
                self.connection_generation,
                "connection generation",
                minimum=0,
            )
        if self.source_revision is not None:
            _require_text(self.source_revision, "source revision", maximum=256)
        if self.raw_payload_reference is not None:
            _require_text(
                self.raw_payload_reference,
                "raw payload reference",
                maximum=512,
            )


@dataclass(frozen=True, slots=True)
class ObservationQuality:
    """One explicit quality state plus independent grain completeness."""

    state: QualityState
    complete: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, QualityState):
            raise EvidenceAdmissionError("quality state is not supported")
        if type(self.complete) is not bool:
            raise EvidenceAdmissionError("quality completeness must be boolean")
        if not isinstance(self.diagnostics, tuple):
            raise EvidenceAdmissionError("quality diagnostics must be a tuple")
        for diagnostic in self.diagnostics:
            _require_pattern(diagnostic, _DIAGNOSTIC_CODE, "quality diagnostic")

        necessarily_incomplete = {
            QualityState.PROVISIONAL,
            QualityState.GAP,
            QualityState.INCOMPLETE,
            QualityState.SCHEMA_MISMATCH,
            QualityState.CONFLICT,
            QualityState.UNAVAILABLE,
        }
        if self.state is QualityState.VALID and not self.complete:
            raise EvidenceAdmissionError("valid evidence must be complete")
        if self.state in necessarily_incomplete and self.complete:
            raise EvidenceAdmissionError(
                "the quality state cannot describe complete evidence"
            )

    @property
    def usable_for_final_transition(self) -> bool:
        """Only complete VALID evidence may authorize a final transition."""

        return self.state is QualityState.VALID and self.complete


@dataclass(frozen=True, slots=True)
class KlineIdentity:
    instrument: InstrumentId
    interval: str
    open_time_us: int


@dataclass(frozen=True, slots=True)
class AggregateTradeIdentity:
    instrument: InstrumentId
    aggregate_trade_id: int


@dataclass(frozen=True, slots=True)
class BookTickerIdentity:
    instrument: InstrumentId
    update_id: int


@dataclass(frozen=True, slots=True)
class MetadataIdentity:
    instrument: InstrumentId
    capture_time_ns: int


type EvidenceIdentity = (
    KlineIdentity | AggregateTradeIdentity | BookTickerIdentity | MetadataIdentity
)


@dataclass(frozen=True, slots=True)
class KlineEvent:
    """One provisional or final venue kline revision."""

    instrument: InstrumentId
    raw_symbol: str
    interval: str
    source_event_time: EpochTimestamp | None
    open_time: EpochTimestamp
    close_time: EpochTimestamp
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    base_volume: Decimal
    quote_turnover: Decimal
    taker_buy_base_volume: Decimal
    taker_buy_quote_turnover: Decimal
    trade_count: int
    first_trade_id: int | None
    last_trade_id: int | None
    closed: bool
    provenance: Provenance
    quality: ObservationQuality

    def __post_init__(self) -> None:
        _require_text(self.raw_symbol, "raw symbol", maximum=128)
        _require_pattern(self.interval, _INTERVAL, "kline interval")
        if self.close_time.epoch_microseconds < self.open_time.epoch_microseconds:
            raise EvidenceAdmissionError("kline close time precedes open time")

        for name, price in (
            ("open price", self.open_price),
            ("high price", self.high_price),
            ("low price", self.low_price),
            ("close price", self.close_price),
        ):
            _require_decimal(price, name, positive=True)
        for name, quantity in (
            ("base volume", self.base_volume),
            ("quote turnover", self.quote_turnover),
            ("taker-buy base volume", self.taker_buy_base_volume),
            ("taker-buy quote turnover", self.taker_buy_quote_turnover),
        ):
            _require_decimal(quantity, name, positive=False)

        if self.high_price < max(self.open_price, self.close_price):
            raise EvidenceAdmissionError("kline high is below open or close")
        if self.low_price > min(self.open_price, self.close_price):
            raise EvidenceAdmissionError("kline low is above open or close")
        if self.high_price < self.low_price:
            raise EvidenceAdmissionError("kline high is below low")
        if self.taker_buy_quote_turnover > self.quote_turnover:
            raise EvidenceAdmissionError("taker-buy turnover exceeds total turnover")
        if self.taker_buy_base_volume > self.base_volume:
            raise EvidenceAdmissionError(
                "taker-buy base volume exceeds total base volume"
            )

        _require_int(self.trade_count, "trade count", minimum=0)
        _require_trade_range(self.first_trade_id, self.last_trade_id)
        if type(self.closed) is not bool:
            raise EvidenceAdmissionError("kline closed flag must be boolean")
        if (self.quality.state is QualityState.PROVISIONAL) is self.closed:
            raise EvidenceAdmissionError(
                "kline finality and provisional quality disagree"
            )

    @property
    def identity(self) -> KlineIdentity:
        return KlineIdentity(
            instrument=self.instrument,
            interval=self.interval,
            open_time_us=self.open_time.epoch_microseconds,
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.KLINE


@dataclass(frozen=True, slots=True)
class AggregateTradeEvent:
    """One venue aggregate-trade observation."""

    instrument: InstrumentId
    raw_symbol: str
    aggregate_trade_id: int
    first_trade_id: int
    last_trade_id: int
    source_event_time: EpochTimestamp
    trade_time: EpochTimestamp
    price: Decimal
    quantity: Decimal
    buyer_is_maker: bool
    provenance: Provenance
    quality: ObservationQuality

    def __post_init__(self) -> None:
        _require_text(self.raw_symbol, "raw symbol", maximum=128)
        _require_int(
            self.aggregate_trade_id,
            "aggregate trade identity",
            minimum=0,
        )
        first_trade_id = _require_int(
            self.first_trade_id,
            "first trade identity",
            minimum=0,
        )
        last_trade_id = _require_int(
            self.last_trade_id,
            "last trade identity",
            minimum=0,
        )
        if first_trade_id > last_trade_id:
            raise EvidenceAdmissionError("aggregate trade identity range is invalid")
        _require_decimal(self.price, "trade price", positive=True)
        _require_decimal(self.quantity, "trade quantity", positive=True)
        if type(self.buyer_is_maker) is not bool:
            raise EvidenceAdmissionError("buyer-maker flag must be boolean")
        if self.quality.state is QualityState.PROVISIONAL:
            raise EvidenceAdmissionError("aggregate trades cannot be provisional")

    @property
    def identity(self) -> AggregateTradeIdentity:
        return AggregateTradeIdentity(
            instrument=self.instrument,
            aggregate_trade_id=self.aggregate_trade_id,
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.AGGREGATE_TRADE


@dataclass(frozen=True, slots=True)
class BookTickerEvent:
    """One best-bid/ask update with optional venue event time."""

    instrument: InstrumentId
    raw_symbol: str
    update_id: int
    source_event_time: EpochTimestamp | None
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    provenance: Provenance
    quality: ObservationQuality

    def __post_init__(self) -> None:
        _require_text(self.raw_symbol, "raw symbol", maximum=128)
        _require_int(self.update_id, "book ticker update identity", minimum=0)
        _require_decimal(self.bid_price, "bid price", positive=True)
        _require_decimal(self.ask_price, "ask price", positive=True)
        _require_decimal(self.bid_quantity, "bid quantity", positive=False)
        _require_decimal(self.ask_quantity, "ask quantity", positive=False)
        if self.bid_price >= self.ask_price:
            raise EvidenceAdmissionError("bid price must be below ask price")
        if self.quality.state is QualityState.PROVISIONAL:
            raise EvidenceAdmissionError("book ticker updates cannot be provisional")

    @property
    def identity(self) -> BookTickerIdentity:
        return BookTickerIdentity(
            instrument=self.instrument,
            update_id=self.update_id,
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.BOOK_TICKER


@dataclass(frozen=True, slots=True)
class InstrumentMetadata:
    """Append-only point-in-time venue metadata for one instrument."""

    instrument: InstrumentId
    raw_symbol: str
    status: InstrumentStatus
    base_asset: str
    quote_asset: str
    price_tick: Decimal
    quantity_step: Decimal
    minimum_notional: Decimal
    permissions: tuple[str, ...]
    observation: MetadataObservation
    provenance: Provenance
    quality: ObservationQuality

    def __post_init__(self) -> None:
        _require_text(self.raw_symbol, "raw symbol", maximum=128)
        if not isinstance(self.status, InstrumentStatus):
            raise EvidenceAdmissionError("instrument status is not supported")
        _require_pattern(self.base_asset, _ASSET_TOKEN, "base asset")
        _require_pattern(self.quote_asset, _ASSET_TOKEN, "quote asset")
        _require_decimal(self.price_tick, "price tick", positive=True)
        _require_decimal(self.quantity_step, "quantity step", positive=True)
        _require_decimal(
            self.minimum_notional,
            "minimum notional",
            positive=False,
        )
        if not isinstance(self.permissions, tuple):
            raise EvidenceAdmissionError("instrument permissions must be a tuple")
        for permission in self.permissions:
            _require_pattern(permission, _CANONICAL_SYMBOL, "instrument permission")
        if not isinstance(self.observation, MetadataObservation):
            raise EvidenceAdmissionError("metadata observation status is invalid")
        if self.quality.state is QualityState.PROVISIONAL:
            raise EvidenceAdmissionError("metadata snapshots cannot be provisional")

    @property
    def identity(self) -> MetadataIdentity:
        return MetadataIdentity(
            instrument=self.instrument,
            capture_time_ns=self.provenance.receipt.wall_time_ns,
        )

    @property
    def kind(self) -> EventKind:
        return EventKind.INSTRUMENT_METADATA


type MarketEvidence = (
    KlineEvent | AggregateTradeEvent | BookTickerEvent | InstrumentMetadata
)


class _IdentifiedEvidence(Protocol):
    @property
    def identity(self) -> EvidenceIdentity: ...

    @property
    def provenance(self) -> Provenance: ...


def classify_reobservation(
    admitted: _IdentifiedEvidence,
    observed: _IdentifiedEvidence,
) -> QualityState:
    """Classify a second payload for one canonical identity."""

    if admitted.identity != observed.identity:
        raise EvidenceAdmissionError(
            "reobservation must have the same canonical identity"
        )
    if (isinstance(admitted, KlineEvent) and not admitted.closed) or (
        isinstance(observed, KlineEvent) and not observed.closed
    ):
        raise EvidenceAdmissionError(
            "provisional kline revisions are not final reobservations"
        )
    if admitted.provenance.payload_digest == observed.provenance.payload_digest:
        return QualityState.DUPLICATE
    return QualityState.CONFLICT


def _require_trade_range(
    first_trade_id: int | None,
    last_trade_id: int | None,
) -> None:
    if first_trade_id is None and last_trade_id is None:
        return
    if first_trade_id is None or last_trade_id is None:
        raise EvidenceAdmissionError("kline trade identities must be paired")
    first = _require_int(first_trade_id, "first trade identity", minimum=0)
    last = _require_int(last_trade_id, "last trade identity", minimum=0)
    if first > last:
        raise EvidenceAdmissionError("kline trade identity range is invalid")


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvidenceAdmissionError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise EvidenceAdmissionError(f"{name} is outside the admitted range")
    return value


def _require_decimal(value: object, name: str, *, positive: bool) -> Decimal:
    if not isinstance(value, Decimal):
        raise EvidenceAdmissionError(f"{name} must be an exact decimal")
    if not value.is_finite():
        raise EvidenceAdmissionError(f"{name} must be finite")
    if positive and value <= 0:
        raise EvidenceAdmissionError(f"{name} must be positive")
    if not positive and value < 0:
        raise EvidenceAdmissionError(f"{name} must be non-negative")
    return value


def _require_text(value: object, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise EvidenceAdmissionError(f"{name} is invalid")
    return value


def _require_pattern(value: object, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvidenceAdmissionError(f"{name} is invalid")
    return value
