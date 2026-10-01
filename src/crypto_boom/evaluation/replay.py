"""Deterministic event-time replay with explicit watermarks and fault evidence."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import StrEnum

from crypto_boom import _artifacts
from crypto_boom.market import (
    AggregateTradeEvent,
    BookTickerEvent,
    InstrumentMetadata,
    KlineEvent,
    MarketEvidence,
    QualityState,
)
from crypto_boom.storage.canonical import (
    PublishedCanonicalPartition,
    load_published_canonical_klines,
)

__all__ = (
    "FaultInjection",
    "FaultKind",
    "ManualReplayClock",
    "ReplayError",
    "ReplayInput",
    "ReplayRecord",
    "ReplayResult",
    "ReplaySpec",
    "replay_events",
    "replay_partition",
)

_REPLAY_SCHEMA_VERSION = 1


class ReplayError(RuntimeError):
    """Replay input or execution violates the deterministic replay contract."""


class FaultKind(StrEnum):
    """Bounded faults supported by the shared replay harness."""

    DUPLICATE = "duplicate"
    GAP = "gap"
    DISCONNECT = "disconnect"


@dataclass(frozen=True, slots=True)
class ReplaySpec:
    """Versioned replay policy independent of detector implementation."""

    specification_version: str
    allowed_lateness_us: int
    watermark_policy: str = "event-time-v1"

    def __post_init__(self) -> None:
        if not self.specification_version:
            raise ValueError("replay specification version must not be empty")
        if type(self.allowed_lateness_us) is not int or self.allowed_lateness_us < 0:
            raise ValueError("allowed lateness must be a non-negative integer")
        if not self.watermark_policy:
            raise ValueError("watermark policy must not be empty")

    @property
    def spec_id(self) -> str:
        return _artifacts.content_id(
            {
                "allowed_lateness_us": self.allowed_lateness_us,
                "schema_version": _REPLAY_SCHEMA_VERSION,
                "specification_version": self.specification_version,
                "watermark_policy": self.watermark_policy,
            }
        )


@dataclass(frozen=True, slots=True)
class ReplayInput:
    """One admitted observation plus its controlled replay arrival."""

    event: MarketEvidence
    arrival_time_us: int
    partition_manifest_id: str
    decoder_version: str

    def __post_init__(self) -> None:
        if type(self.arrival_time_us) is not int or self.arrival_time_us < 0:
            raise ValueError("replay arrival time must be a non-negative integer")
        if not self.partition_manifest_id:
            raise ValueError("partition manifest identity must not be empty")
        if not self.decoder_version:
            raise ValueError("decoder version must not be empty")


@dataclass(frozen=True, slots=True)
class FaultInjection:
    """A deterministic fault applied to indices in the supplied arrival stream."""

    kind: FaultKind
    event_index: int
    length: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FaultKind):
            raise ValueError("fault kind is not supported")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("fault event index must be a non-negative integer")
        if type(self.length) is not int or self.length < 1:
            raise ValueError("fault length must be a positive integer")
        if self.kind is not FaultKind.DISCONNECT and self.length != 1:
            raise ValueError("only disconnect faults may span multiple events")


@dataclass(slots=True)
class ManualReplayClock:
    """A monotonic, explicitly advanced replay clock."""

    now_us: int = 0

    def __post_init__(self) -> None:
        if type(self.now_us) is not int or self.now_us < 0:
            raise ValueError("replay clock must start at a non-negative integer")

    def advance_to(self, instant_us: int) -> None:
        if type(instant_us) is not int or instant_us < self.now_us:
            raise ReplayError("replay clock cannot move backwards")
        self.now_us = instant_us


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One observable replay disposition in deterministic processing order."""

    sequence: int
    event: MarketEvidence
    partition_manifest_id: str
    decoder_version: str
    event_time_us: int
    arrival_time_us: int
    emitted_at_us: int
    watermark_us: int
    quality_state: QualityState
    authoritative: bool
    diagnostics: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "arrival_time_us": self.arrival_time_us,
            "authoritative": self.authoritative,
            "decoder_version": self.decoder_version,
            "diagnostics": list(self.diagnostics),
            "emitted_at_us": self.emitted_at_us,
            "event_identity": _event_identity_mapping(self.event),
            "event_kind": self.event.kind.value,
            "event_time_us": self.event_time_us,
            "partition_manifest_id": self.partition_manifest_id,
            "payload_digest": self.event.provenance.payload_digest.value,
            "quality_state": self.quality_state.value,
            "sequence": self.sequence,
            "watermark_us": self.watermark_us,
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Stable replay evidence suitable for equality and content comparison."""

    spec_id: str
    partition_manifest_ids: tuple[str, ...]
    records: tuple[ReplayRecord, ...]

    @property
    def authoritative_events(self) -> tuple[MarketEvidence, ...]:
        return tuple(record.event for record in self.records if record.authoritative)

    @property
    def result_id(self) -> str:
        return _artifacts.content_id(self._content_mapping())

    def _content_mapping(self) -> dict[str, object]:
        return {
            "partition_manifest_ids": list(self.partition_manifest_ids),
            "records": [record.to_mapping() for record in self.records],
            "schema_version": _REPLAY_SCHEMA_VERSION,
            "spec_id": self.spec_id,
        }

    def to_bytes(self) -> bytes:
        return _artifacts.canonical_json(
            {"result_id": self.result_id, **self._content_mapping()}
        )


def replay_partition(
    partition: PublishedCanonicalPartition,
    spec: ReplaySpec,
    *,
    faults: tuple[FaultInjection, ...] = (),
    clock: ManualReplayClock | None = None,
) -> ReplayResult:
    """Reverify and replay one qualified canonical partition."""

    rows = load_published_canonical_klines(partition)
    inputs = tuple(
        ReplayInput(
            event=row.event,
            arrival_time_us=_event_time_us(row.event),
            partition_manifest_id=partition.manifest.manifest_id,
            decoder_version=row.decoder_version,
        )
        for row in rows
    )
    return replay_events(inputs, spec, faults=faults, clock=clock)


def replay_events(
    inputs: tuple[ReplayInput, ...],
    spec: ReplaySpec,
    *,
    faults: tuple[FaultInjection, ...] = (),
    clock: ManualReplayClock | None = None,
) -> ReplayResult:
    """Replay an arrival-ordered stream through one event-time watermark policy."""

    replay_clock = clock if clock is not None else ManualReplayClock()
    _validate_arrivals(inputs, replay_clock)
    fault_by_index = _fault_schedule(len(inputs), faults)

    records: list[ReplayRecord] = []
    buffered: list[tuple[int, str, str, int, ReplayInput]] = []
    seen: dict[str, str] = {}
    pending: dict[str, ReplayInput] = {}
    conflicted: set[str] = set()
    watermark_us = -1
    serial = 0

    def record(
        item: ReplayInput,
        state: QualityState,
        *,
        authoritative: bool,
        diagnostics: tuple[str, ...],
    ) -> None:
        records.append(
            ReplayRecord(
                sequence=len(records),
                event=item.event,
                partition_manifest_id=item.partition_manifest_id,
                decoder_version=item.decoder_version,
                event_time_us=_event_time_us(item.event),
                arrival_time_us=item.arrival_time_us,
                emitted_at_us=replay_clock.now_us,
                watermark_us=watermark_us,
                quality_state=state,
                authoritative=authoritative,
                diagnostics=diagnostics,
            )
        )

    def flush(closed_through_us: int) -> None:
        while buffered and buffered[0][0] <= closed_through_us:
            _, _, identity_key, _, item = heapq.heappop(buffered)
            if identity_key in conflicted:
                continue
            pending.pop(identity_key, None)
            record(
                item,
                QualityState.VALID,
                authoritative=True,
                diagnostics=item.event.quality.diagnostics,
            )

    def advance_watermark() -> None:
        nonlocal watermark_us
        candidate = replay_clock.now_us - spec.allowed_lateness_us
        if candidate > watermark_us:
            watermark_us = candidate
            flush(watermark_us)

    def observe(item: ReplayInput) -> None:
        nonlocal serial
        event = item.event
        event_time_us = _event_time_us(event)
        identity_key = _event_identity_key(event)
        payload_digest = event.provenance.payload_digest.value

        prior_digest = seen.get(identity_key)
        if prior_digest is not None:
            if identity_key in conflicted or prior_digest != payload_digest:
                prior = pending.pop(identity_key, None)
                conflicted.add(identity_key)
                if prior is not None:
                    record(
                        prior,
                        QualityState.CONFLICT,
                        authoritative=False,
                        diagnostics=("CONFLICTING_REOBSERVATION",),
                    )
                record(
                    item,
                    QualityState.CONFLICT,
                    authoritative=False,
                    diagnostics=("CONFLICTING_REOBSERVATION",),
                )
            else:
                record(
                    item,
                    QualityState.DUPLICATE,
                    authoritative=False,
                    diagnostics=("DUPLICATE_EVENT",),
                )
            advance_watermark()
            return
        seen[identity_key] = payload_digest

        if not event.quality.usable_for_final_transition:
            record(
                item,
                event.quality.state,
                authoritative=False,
                diagnostics=event.quality.diagnostics,
            )
        elif event_time_us <= watermark_us:
            record(
                item,
                QualityState.LATE,
                authoritative=False,
                diagnostics=("LATE_AFTER_WATERMARK",),
            )
        else:
            pending[identity_key] = item
            heapq.heappush(
                buffered,
                (
                    event_time_us,
                    event.kind.value,
                    identity_key,
                    serial,
                    item,
                ),
            )
            serial += 1
        advance_watermark()

    for index, item in enumerate(inputs):
        replay_clock.advance_to(item.arrival_time_us)
        injected = fault_by_index.get(index)
        if injected is FaultKind.GAP:
            record(
                item,
                QualityState.GAP,
                authoritative=False,
                diagnostics=("INJECTED_GAP",),
            )
            advance_watermark()
            continue
        if injected is FaultKind.DISCONNECT:
            record(
                item,
                QualityState.INCOMPLETE,
                authoritative=False,
                diagnostics=("INJECTED_DISCONNECT",),
            )
            advance_watermark()
            continue

        observe(item)
        if injected is FaultKind.DUPLICATE:
            observe(item)

    if buffered:
        watermark_us = max(watermark_us, max(item[0] for item in buffered))
        flush(watermark_us)

    return ReplayResult(
        spec_id=spec.spec_id,
        partition_manifest_ids=tuple(
            sorted({item.partition_manifest_id for item in inputs})
        ),
        records=tuple(records),
    )


def _validate_arrivals(
    inputs: tuple[ReplayInput, ...],
    clock: ManualReplayClock,
) -> None:
    previous = clock.now_us
    for item in inputs:
        if item.arrival_time_us < previous:
            raise ReplayError(
                "replay inputs must be ordered by controlled arrival time"
            )
        previous = item.arrival_time_us


def _fault_schedule(
    input_count: int,
    faults: tuple[FaultInjection, ...],
) -> dict[int, FaultKind]:
    schedule: dict[int, FaultKind] = {}
    for fault in faults:
        end = fault.event_index + fault.length
        if fault.event_index >= input_count or end > input_count:
            raise ReplayError("fault injection is outside the replay input")
        for index in range(fault.event_index, end):
            if index in schedule:
                raise ReplayError("fault injections overlap")
            schedule[index] = fault.kind
    return schedule


def _event_time_us(event: MarketEvidence) -> int:
    if isinstance(event, KlineEvent):
        return event.close_time.epoch_microseconds
    if isinstance(event, AggregateTradeEvent):
        return event.trade_time.epoch_microseconds
    if isinstance(event, BookTickerEvent) and event.source_event_time is not None:
        return event.source_event_time.epoch_microseconds
    return event.provenance.receipt.wall_time_ns // 1_000


def _event_identity_mapping(event: MarketEvidence) -> dict[str, object]:
    instrument = event.instrument
    identity: dict[str, object] = {
        "environment": instrument.environment.value,
        "market": instrument.venue.market,
        "symbol": instrument.symbol,
        "venue": instrument.venue.name,
    }
    if isinstance(event, KlineEvent):
        identity.update(
            interval=event.interval,
            open_time_us=event.open_time.epoch_microseconds,
        )
    elif isinstance(event, AggregateTradeEvent):
        identity["aggregate_trade_id"] = event.aggregate_trade_id
    elif isinstance(event, BookTickerEvent):
        identity["update_id"] = event.update_id
    elif isinstance(event, InstrumentMetadata):
        identity["capture_time_ns"] = event.provenance.receipt.wall_time_ns
    return identity


def _event_identity_key(event: MarketEvidence) -> str:
    return _artifacts.canonical_json(_event_identity_mapping(event)).decode("utf-8")
