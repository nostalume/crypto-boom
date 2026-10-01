from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from crypto_boom.market import (
    AggregateTradeEvent,
    BookTickerEvent,
    Environment,
    EpochTimestamp,
    EventKind,
    EvidenceAdmissionError,
    InstrumentId,
    InstrumentMetadata,
    InstrumentStatus,
    KlineEvent,
    LocalReceipt,
    MetadataObservation,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    TimeUnit,
    VenueId,
    classify_reobservation,
)

RUN_ID = UUID("b64ac706-e9e1-44bd-b606-e0051b5ee452")


def _instrument(symbol: str = "ETHUSDT") -> InstrumentId:
    return InstrumentId(
        venue=VenueId(name="binance", market="spot"),
        environment=Environment.PRODUCTION,
        symbol=symbol,
    )


def _provenance(
    payload: bytes = b"payload-a",
    *,
    generation: int | None = 3,
    wall_time_ns: int = 1_758_307_200_000_000_000,
) -> Provenance:
    return Provenance(
        source=SourceDescriptor(
            endpoint="wss://data-stream.binance.vision",
            channel="ethusdt@kline_1m",
            schema_version=1,
        ),
        ingestion_run_id=RUN_ID,
        receipt=LocalReceipt(
            wall_time_ns=wall_time_ns,
            monotonic_ns=123_456_789,
        ),
        payload_digest=PayloadDigest.sha256(payload),
        connection_generation=generation,
    )


def _quality(state: QualityState = QualityState.VALID) -> ObservationQuality:
    necessarily_incomplete = {
        QualityState.PROVISIONAL,
        QualityState.GAP,
        QualityState.INCOMPLETE,
        QualityState.SCHEMA_MISMATCH,
        QualityState.CONFLICT,
        QualityState.UNAVAILABLE,
    }
    return ObservationQuality(
        state=state,
        complete=state not in necessarily_incomplete,
        diagnostics=() if state is QualityState.VALID else ("TEST_EVIDENCE",),
    )


def _kline(
    *,
    payload: bytes = b"payload-a",
    generation: int | None = 3,
    open_time: EpochTimestamp | None = None,
    quality: ObservationQuality | None = None,
    closed: bool = True,
) -> KlineEvent:
    return KlineEvent(
        instrument=_instrument(),
        raw_symbol="ETHUSDT",
        interval="1m",
        source_event_time=EpochTimestamp.from_raw(1_758_307_260_010, "ms"),
        open_time=open_time or EpochTimestamp.from_raw(1_758_307_200_000, "ms"),
        close_time=EpochTimestamp.from_raw(1_758_307_259_999, "ms"),
        open_price=Decimal("0.00000001"),
        high_price=Decimal("0.00000003"),
        low_price=Decimal("0.00000001"),
        close_price=Decimal("0.00000002"),
        base_volume=Decimal("123456789.00000001"),
        quote_turnover=Decimal("2.3456789012345678"),
        taker_buy_base_volume=Decimal("50000000.00000001"),
        taker_buy_quote_turnover=Decimal("1.0000000000000001"),
        trade_count=2,
        first_trade_id=10,
        last_trade_id=11,
        closed=closed,
        provenance=_provenance(payload, generation=generation),
        quality=quality or _quality(),
    )


def test_canonical_identity_ignores_connection_generation_and_timestamp_unit() -> None:
    millisecond = _kline(generation=3)
    microsecond = _kline(
        generation=4,
        open_time=EpochTimestamp.from_raw(1_758_307_200_000_000, "us"),
    )

    assert millisecond.identity == microsecond.identity
    assert millisecond.open_time.raw_value == 1_758_307_200_000
    assert millisecond.open_time.unit is TimeUnit.MILLISECOND
    assert millisecond.kind is EventKind.KLINE


def test_same_identity_and_payload_is_duplicate_across_socket_generations() -> None:
    first = _kline(generation=3)
    overlap_copy = _kline(generation=4)

    assert classify_reobservation(first, overlap_copy) is QualityState.DUPLICATE


def test_same_identity_and_different_payload_is_conflict() -> None:
    first = _kline(payload=b"first-final")
    conflicting_final = _kline(payload=b"changed-final")

    assert classify_reobservation(first, conflicting_final) is QualityState.CONFLICT


def test_provisional_kline_revisions_are_not_final_conflicts() -> None:
    first = _kline(
        payload=b"provisional-a",
        quality=_quality(QualityState.PROVISIONAL),
        closed=False,
    )
    revision = _kline(
        payload=b"provisional-b",
        quality=_quality(QualityState.PROVISIONAL),
        closed=False,
    )

    with pytest.raises(EvidenceAdmissionError, match="provisional kline"):
        classify_reobservation(first, revision)


def test_reobservation_rejects_a_different_identity() -> None:
    another_instrument = replace(
        _kline(),
        instrument=_instrument("SOLUSDT"),
        raw_symbol="SOLUSDT",
    )

    with pytest.raises(
        EvidenceAdmissionError,
        match="same canonical identity",
    ):
        classify_reobservation(_kline(), another_instrument)


@pytest.mark.parametrize("state", list(QualityState))
def test_every_quality_state_is_explicit_and_only_valid_is_usable(
    state: QualityState,
) -> None:
    quality = _quality(state)

    assert quality.state is state
    assert quality.usable_for_final_transition is (state is QualityState.VALID)


def test_valid_quality_cannot_hide_incomplete_evidence() -> None:
    with pytest.raises(EvidenceAdmissionError, match="valid evidence"):
        ObservationQuality(state=QualityState.VALID, complete=False)


def test_quality_diagnostics_are_bounded_codes_not_payload_text() -> None:
    with pytest.raises(EvidenceAdmissionError, match="diagnostic"):
        ObservationQuality(
            state=QualityState.SCHEMA_MISMATCH,
            complete=False,
            diagnostics=("unexpected secret payload value",),
        )


@given(st.integers(min_value=0, max_value=(2**63 - 1) // 1_000))
def test_millisecond_and_microsecond_timestamps_normalize_identically(
    milliseconds: int,
) -> None:
    as_milliseconds = EpochTimestamp.from_raw(milliseconds, "ms")
    as_microseconds = EpochTimestamp.from_raw(milliseconds * 1_000, "us")

    assert as_milliseconds.epoch_microseconds == as_microseconds.epoch_microseconds


@pytest.mark.parametrize("unit", ["s", "ns", "unknown", ""])
def test_unknown_timestamp_units_fail_closed(unit: str) -> None:
    with pytest.raises(EvidenceAdmissionError, match="unit is not supported"):
        EpochTimestamp.from_raw(1_758_307_200, unit)


def test_boolean_is_not_admitted_as_an_integer_timestamp() -> None:
    with pytest.raises(EvidenceAdmissionError, match="must be an integer"):
        EpochTimestamp.from_raw(True, "ms")


@given(st.binary(max_size=1_024))
def test_payload_digest_is_deterministic(payload: bytes) -> None:
    digest = PayloadDigest.sha256(payload)

    assert digest == PayloadDigest.sha256(payload)
    assert digest.value.startswith("sha256:")
    assert len(digest.value) == 71


def test_exact_decimal_boundaries_survive_event_admission() -> None:
    event = _kline()

    assert event.open_price == Decimal("0.00000001")
    assert event.base_volume == Decimal("123456789.00000001")
    assert event.quote_turnover == Decimal("2.3456789012345678")


def test_archive_kline_may_omit_unavailable_trade_identity_bounds() -> None:
    event = replace(_kline(), first_trade_id=None, last_trade_id=None)

    assert event.trade_count == 2
    assert event.first_trade_id is None
    assert event.last_trade_id is None


def test_kline_trade_identity_bounds_must_be_supplied_together() -> None:
    with pytest.raises(EvidenceAdmissionError, match="must be paired"):
        replace(_kline(), first_trade_id=None)


@pytest.mark.parametrize(
    "price",
    [0.1, Decimal("NaN"), Decimal("Infinity"), Decimal("0")],
)
def test_non_exact_or_non_positive_prices_are_rejected(price: object) -> None:
    with pytest.raises(EvidenceAdmissionError):
        replace(_kline(), open_price=price)


def test_unclosed_kline_must_be_provisional() -> None:
    provisional = _kline(
        quality=_quality(QualityState.PROVISIONAL),
        closed=False,
    )

    assert provisional.quality.state is QualityState.PROVISIONAL
    assert not provisional.quality.usable_for_final_transition

    with pytest.raises(EvidenceAdmissionError, match="finality"):
        _kline(closed=False, quality=_quality(QualityState.VALID))


def test_aggregate_trade_has_venue_identity_and_exact_values() -> None:
    event = AggregateTradeEvent(
        instrument=_instrument(),
        raw_symbol="ETHUSDT",
        aggregate_trade_id=900,
        first_trade_id=1_000,
        last_trade_id=1_002,
        source_event_time=EpochTimestamp.from_raw(1_758_307_200_010, "ms"),
        trade_time=EpochTimestamp.from_raw(1_758_307_200_008, "ms"),
        price=Decimal("0.00000002"),
        quantity=Decimal("100000000.00000001"),
        buyer_is_maker=False,
        provenance=_provenance(),
        quality=_quality(),
    )

    assert event.identity.instrument.venue == VenueId("binance", "spot")
    assert event.identity.aggregate_trade_id == 900
    assert event.kind is EventKind.AGGREGATE_TRADE


def test_aggregate_trade_rejects_a_reversed_underlying_trade_range() -> None:
    with pytest.raises(EvidenceAdmissionError, match="range is invalid"):
        AggregateTradeEvent(
            instrument=_instrument(),
            raw_symbol="ETHUSDT",
            aggregate_trade_id=900,
            first_trade_id=1_002,
            last_trade_id=1_000,
            source_event_time=EpochTimestamp.from_raw(1_758_307_200_010, "ms"),
            trade_time=EpochTimestamp.from_raw(1_758_307_200_008, "ms"),
            price=Decimal("1"),
            quantity=Decimal("1"),
            buyer_is_maker=False,
            provenance=_provenance(),
            quality=_quality(),
        )


def test_book_ticker_uses_update_identity_without_claiming_sequence_contiguity() -> (
    None
):
    first = BookTickerEvent(
        instrument=_instrument(),
        raw_symbol="ETHUSDT",
        update_id=10,
        source_event_time=None,
        bid_price=Decimal("1999.99999999"),
        bid_quantity=Decimal("0"),
        ask_price=Decimal("2000.00000001"),
        ask_quantity=Decimal("3.00000001"),
        provenance=_provenance(),
        quality=_quality(),
    )
    later = replace(first, update_id=25, provenance=_provenance(b"later"))

    assert first.identity.update_id == 10
    assert later.identity.update_id == 25
    assert first.kind is EventKind.BOOK_TICKER


def test_crossed_or_locked_book_ticker_is_rejected() -> None:
    with pytest.raises(EvidenceAdmissionError, match="below ask"):
        BookTickerEvent(
            instrument=_instrument(),
            raw_symbol="ETHUSDT",
            update_id=10,
            source_event_time=None,
            bid_price=Decimal("2000"),
            bid_quantity=Decimal("1"),
            ask_price=Decimal("2000"),
            ask_quantity=Decimal("1"),
            provenance=_provenance(),
            quality=_quality(),
        )


def test_metadata_is_append_only_point_in_time_evidence() -> None:
    observed = InstrumentMetadata(
        instrument=_instrument(),
        raw_symbol="ETHUSDT",
        status=InstrumentStatus.TRADING,
        base_asset="ETH",
        quote_asset="USDT",
        price_tick=Decimal("0.00000001"),
        quantity_step=Decimal("0.00000001"),
        minimum_notional=Decimal("5.00000000"),
        permissions=("SPOT",),
        observation=MetadataObservation.OBSERVED,
        provenance=_provenance(wall_time_ns=1_758_307_200_000_000_000),
        quality=_quality(),
    )
    changed_later = replace(
        observed,
        status=InstrumentStatus.BREAK,
        provenance=_provenance(
            b"metadata-later",
            wall_time_ns=1_758_310_800_000_000_000,
        ),
    )

    assert observed.identity != changed_later.identity
    assert observed.status is InstrumentStatus.TRADING
    assert changed_later.status is InstrumentStatus.BREAK
    assert observed.observation is MetadataObservation.OBSERVED
    assert observed.kind is EventKind.INSTRUMENT_METADATA


def test_unknown_instrument_status_is_not_admitted_as_valid_metadata() -> None:
    observed = InstrumentMetadata(
        instrument=_instrument(),
        raw_symbol="ETHUSDT",
        status=InstrumentStatus.TRADING,
        base_asset="ETH",
        quote_asset="USDT",
        price_tick=Decimal("0.00000001"),
        quantity_step=Decimal("0.00000001"),
        minimum_notional=Decimal("5.00000000"),
        permissions=("SPOT",),
        observation=MetadataObservation.OBSERVED,
        provenance=_provenance(),
        quality=_quality(),
    )

    with pytest.raises(EvidenceAdmissionError, match="status is not supported"):
        replace(observed, status="NEW_UNKNOWN_STATUS")
