from __future__ import annotations

import hashlib
from decimal import Decimal
from uuid import UUID

import pytest

from crypto_boom.evaluation.replay import (
    FaultInjection,
    FaultKind,
    ManualReplayClock,
    ReplayError,
    ReplayInput,
    ReplaySpec,
    replay_events,
)
from crypto_boom.market import (
    Environment,
    EpochTimestamp,
    InstrumentId,
    KlineEvent,
    LocalReceipt,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    TimeUnit,
    VenueId,
)

INSTRUMENT = InstrumentId(
    venue=VenueId("binance", "spot"),
    environment=Environment.PRODUCTION,
    symbol="ETHUSDT",
)
QUALITY = ObservationQuality(QualityState.VALID, complete=True)
RUN_ID = UUID("00000000-0000-0000-0000-000000000456")


def _kline(open_time_us: int, label: str) -> KlineEvent:
    return KlineEvent(
        instrument=INSTRUMENT,
        raw_symbol="ETHUSDT",
        interval="1m",
        source_event_time=None,
        open_time=EpochTimestamp(open_time_us, TimeUnit.MICROSECOND),
        close_time=EpochTimestamp(open_time_us + 59, TimeUnit.MICROSECOND),
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("99"),
        close_price=Decimal("101"),
        base_volume=Decimal("3"),
        quote_turnover=Decimal("303"),
        taker_buy_base_volume=Decimal("2"),
        taker_buy_quote_turnover=Decimal("202"),
        trade_count=2,
        first_trade_id=None,
        last_trade_id=None,
        closed=True,
        provenance=Provenance(
            source=SourceDescriptor("archive", "klines", 1),
            ingestion_run_id=RUN_ID,
            receipt=LocalReceipt(1, 1),
            payload_digest=PayloadDigest.sha256(label.encode()),
            source_revision="revision-1",
            raw_payload_reference=label,
        ),
        quality=QUALITY,
    )


def _input(event: KlineEvent, arrival_time_us: int) -> ReplayInput:
    return ReplayInput(
        event=event,
        arrival_time_us=arrival_time_us,
        partition_manifest_id="partition-1",
        decoder_version="decoder-1",
    )


def test_equal_source_and_spec_produce_byte_equal_event_time_order() -> None:
    later = _input(_kline(200, "later"), 300)
    earlier = _input(_kline(100, "earlier"), 301)
    spec = ReplaySpec("test-v1", allowed_lateness_us=1_000)

    first = replay_events((later, earlier), spec)
    second = replay_events((later, earlier), spec)

    assert first.to_bytes() == second.to_bytes()
    assert first.result_id == (
        "sha256:9573c352a0cb8b0aa78c1fbeca1d3244a96597f49c3a914351a53b6fce70b66b"
    )
    assert hashlib.sha256(first.to_bytes()).hexdigest() == (
        "70f8c4a3ea8f6a92a102e265bf687b4d9637390bacb484eebb88d98a28060eaf"
    )
    assert [
        event.open_time.epoch_microseconds
        for event in first.authoritative_events
        if isinstance(event, KlineEvent)
    ] == [100, 200]


def test_event_after_closed_watermark_is_retained_but_not_authoritative() -> None:
    first = _input(_kline(100, "first"), 159)
    late = _input(_kline(0, "late"), 160)

    result = replay_events(
        (first, late),
        ReplaySpec("test-v1", allowed_lateness_us=0),
    )

    assert result.authoritative_events == (first.event,)
    late_record = next(
        record for record in result.records if record.quality_state is QualityState.LATE
    )
    assert not late_record.authoritative
    assert late_record.diagnostics == ("LATE_AFTER_WATERMARK",)


def test_fault_injection_exposes_duplicate_gap_and_disconnect_quality() -> None:
    inputs = tuple(
        _input(_kline(index * 100, f"event-{index}"), index * 100 + 59)
        for index in range(4)
    )

    result = replay_events(
        inputs,
        ReplaySpec("test-v1", allowed_lateness_us=0),
        faults=(
            FaultInjection(FaultKind.DUPLICATE, 0),
            FaultInjection(FaultKind.GAP, 1),
            FaultInjection(FaultKind.DISCONNECT, 2, length=2),
        ),
    )

    states = [record.quality_state for record in result.records]
    assert states == [
        QualityState.VALID,
        QualityState.DUPLICATE,
        QualityState.GAP,
        QualityState.INCOMPLETE,
        QualityState.INCOMPLETE,
    ]
    assert result.authoritative_events == (inputs[0].event,)


def test_conflicting_reobservation_never_reaches_authoritative_stream() -> None:
    original = _input(_kline(100, "original"), 159)
    conflict = _input(_kline(100, "conflict"), 160)

    result = replay_events(
        (original, conflict),
        ReplaySpec("test-v1", allowed_lateness_us=1_000),
    )

    assert result.authoritative_events == ()
    assert [record.quality_state for record in result.records] == [
        QualityState.CONFLICT,
        QualityState.CONFLICT,
    ]


def test_clock_and_fault_schedule_fail_closed() -> None:
    first = _input(_kline(100, "first"), 200)
    second = _input(_kline(200, "second"), 199)
    with pytest.raises(ReplayError, match="arrival"):
        replay_events((first, second), ReplaySpec("test-v1", 0))

    clock = ManualReplayClock(201)
    with pytest.raises(ReplayError, match="arrival"):
        replay_events((first,), ReplaySpec("test-v1", 0), clock=clock)

    with pytest.raises(ReplayError, match="outside"):
        replay_events(
            (first,),
            ReplaySpec("test-v1", 0),
            faults=(FaultInjection(FaultKind.GAP, 1),),
        )
