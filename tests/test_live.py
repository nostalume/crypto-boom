from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import time_ns
from uuid import UUID

import aiohttp
import pytest
from aiohttp import web

import crypto_boom.binance_source as binance_source
import crypto_boom.history.daily as daily
import crypto_boom.live as live
from crypto_boom.binance_source import (
    LiveRateLimitError,
    LiveResourceError,
    LiveSchemaError,
    RestWeightBudget,
    decode_exchange_info,
    fetch_exchange_info,
    measure_exchange_clock,
)
from crypto_boom.live import (
    AdmissionDecision,
    BinanceLiveCollector,
    BoundedCaptureQueue,
    BoundedRawCaptureQueue,
    ConnectionAttemptBudget,
    DecodeContext,
    LiveAdmissionState,
    SubscriptionPlan,
    decode_combined_message,
)
from crypto_boom.market import (
    AggregateTradeEvent,
    BookTickerEvent,
    KlineEvent,
    LocalReceipt,
    QualityState,
    TimeUnit,
)


def test_root_modules_reexport_shared_binance_source_objects() -> None:
    assert daily.BINANCE_SPOT is binance_source.BINANCE_SPOT

    names = (
        "BINANCE_SPOT",
        "PUBLIC_REST_BASE",
        "ClockMeasurement",
        "LiveCaptureError",
        "LiveRateLimitError",
        "LiveResourceError",
        "LiveSchemaError",
        "MetadataCapture",
        "RestWeightBudget",
        "decode_exchange_info",
        "fetch_exchange_info",
        "measure_exchange_clock",
        "sampled_receipt",
    )
    for name in names:
        assert getattr(live, name) is getattr(binance_source, name)


RUN_ID = UUID("8386f2c3-b925-40ce-8c6d-4785ea0f37ce")
RECEIPT = LocalReceipt(
    wall_time_ns=1_795_027_200_000_000_000,
    monotonic_ns=100,
)


def _context(generation: int = 0) -> DecodeContext:
    return DecodeContext(
        ingestion_run_id=RUN_ID,
        receipt=RECEIPT,
        connection_generation=generation,
    )


def _combined(stream: str, data: dict[str, object]) -> bytes:
    return json.dumps(
        {"stream": stream, "data": data},
        separators=(",", ":"),
    ).encode()


def _agg_payload(
    aggregate_id: int = 100,
    *,
    symbol: str = "ETHUSDT",
) -> bytes:
    return _combined(
        f"{symbol.lower()}@aggTrade",
        {
            "e": "aggTrade",
            "E": 1_795_027_200_000_001,
            "s": symbol,
            "a": aggregate_id,
            "p": "2000.00000001",
            "q": "1.00000001",
            "f": aggregate_id + 1000,
            "l": aggregate_id + 1001,
            "T": 1_795_027_200_000_000,
            "m": False,
            "M": True,
        },
    )


def _book_payload(update_id: int = 200) -> bytes:
    return _combined(
        "ethusdt@bookTicker",
        {
            "u": update_id,
            "s": "ETHUSDT",
            "b": "1999.99999999",
            "B": "2.0",
            "a": "2000.00000001",
            "A": "3.0",
        },
    )


def _kline_payload(
    *,
    closed: bool = True,
    trade_count: int = 2,
    first_trade_id: int = 10,
    last_trade_id: int = 11,
) -> bytes:
    return _combined(
        "ethusdt@kline_1m",
        {
            "e": "kline",
            "E": 1_795_027_260_000_001,
            "s": "ETHUSDT",
            "k": {
                "t": 1_795_027_200_000_000,
                "T": 1_795_027_259_999_999,
                "s": "ETHUSDT",
                "i": "1m",
                "f": first_trade_id,
                "L": last_trade_id,
                "o": "1900.0",
                "c": "2000.0",
                "h": "2100.0",
                "l": "1800.0",
                "v": "10.0",
                "n": trade_count,
                "x": closed,
                "q": "20000.0",
                "V": "4.0",
                "Q": "8000.0",
                "B": "0",
            },
        },
    )


def test_combined_decoders_preserve_exact_values_receipt_and_raw_payload() -> None:
    aggregate = decode_combined_message(_agg_payload(), context=_context())
    quote = decode_combined_message(_book_payload(), context=_context())
    kline = decode_combined_message(_kline_payload(), context=_context())

    assert isinstance(aggregate.event, AggregateTradeEvent)
    assert aggregate.event.price == Decimal("2000.00000001")
    assert aggregate.event.source_event_time.unit is TimeUnit.MICROSECOND
    assert aggregate.event.provenance.receipt is RECEIPT
    assert aggregate.raw_payload == _agg_payload()

    assert isinstance(quote.event, BookTickerEvent)
    assert quote.event.source_event_time is None
    assert quote.event.bid_quantity == Decimal("2.0")
    assert kline.event.quality.state is QualityState.VALID


def test_open_kline_is_explicitly_provisional() -> None:
    captured = decode_combined_message(
        _kline_payload(closed=False),
        context=_context(),
    )

    assert captured.event.quality.state is QualityState.PROVISIONAL
    assert not captured.event.quality.complete


def test_zero_trade_kline_normalizes_binance_identity_sentinels() -> None:
    captured = decode_combined_message(
        _kline_payload(trade_count=0, first_trade_id=-1, last_trade_id=-1),
        context=_context(),
    )

    assert isinstance(captured.event, KlineEvent)
    assert captured.event.first_trade_id is None
    assert captured.event.last_trade_id is None


def test_negative_kline_trade_identity_is_not_admitted_when_trades_exist() -> None:
    with pytest.raises(LiveSchemaError, match="schema admission"):
        decode_combined_message(
            _kline_payload(trade_count=1, first_trade_id=-1, last_trade_id=-1),
            context=_context(),
        )


def test_unknown_stream_missing_field_and_symbol_disagreement_fail_closed() -> None:
    with pytest.raises(LiveSchemaError, match="not admitted"):
        decode_combined_message(
            _combined("ethusdt@depth", {"s": "ETHUSDT"}),
            context=_context(),
        )
    with pytest.raises(LiveSchemaError, match="schema admission"):
        decode_combined_message(
            _combined("ethusdt@bookTicker", {"s": "ETHUSDT"}),
            context=_context(),
        )
    with pytest.raises(LiveSchemaError, match="symbol disagree"):
        decode_combined_message(
            _combined(
                "btcusdt@bookTicker",
                json.loads(_book_payload())["data"],
            ),
            context=_context(),
        )


def test_exchange_info_decoder_requires_authoritative_filters() -> None:
    payload = json.dumps(
        {
            "timezone": "UTC",
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "permissionSets": [["SPOT"]],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.00000001"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.00010000"},
                        {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                    ],
                }
            ],
        },
        separators=(",", ":"),
    ).encode()

    metadata = decode_exchange_info(
        payload,
        ingestion_run_id=RUN_ID,
        receipt=RECEIPT,
    )

    assert len(metadata.events) == 1
    assert metadata.events[0].permissions == ("SPOT",)
    assert metadata.events[0].price_tick == Decimal("0.00000001")
    assert metadata.events[0].provenance.receipt is RECEIPT

    document = json.loads(payload)
    unsupported = dict(document["symbols"][0])
    unsupported.update(
        {
            "symbol": "币安人生USDT",
            "baseAsset": "币安人生",
        }
    )
    document["symbols"].append(unsupported)
    mixed = json.dumps(document, separators=(",", ":")).encode()
    mixed_metadata = decode_exchange_info(
        mixed,
        ingestion_run_id=RUN_ID,
        receipt=RECEIPT,
    )
    assert len(mixed_metadata.events) == 1
    assert mixed_metadata.unsupported_symbols == ("币安人生USDT",)

    missing_filter = payload.replace(b'"NOTIONAL"', b'"UNKNOWN_"')
    with pytest.raises(LiveSchemaError, match="filter is absent"):
        decode_exchange_info(
            missing_filter,
            ingestion_run_id=RUN_ID,
            receipt=RECEIPT,
        )


@pytest.mark.asyncio
async def test_metadata_acquisition_is_bounded_and_rate_budgeted() -> None:
    payload = json.dumps(
        {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.00001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5.0"},
                    ],
                }
            ]
        },
        separators=(",", ":"),
    ).encode()

    async def handler(request: web.Request) -> web.Response:
        assert request.query["symbol"] == "BTCUSDT"
        return web.Response(body=payload, content_type="application/json")

    app = web.Application()
    app.router.add_get("/exchangeInfo", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    budget = RestWeightBudget(venue_capacity=40, interval_seconds=60)
    try:
        async with aiohttp.ClientSession() as session:
            captured = await fetch_exchange_info(
                session,
                ingestion_run_id=RUN_ID,
                budget=budget,
                symbol="BTCUSDT",
                endpoint=f"{site.name}/exchangeInfo",
                budget_now=0,
            )
            with pytest.raises(LiveResourceError, match="budget is exhausted"):
                await fetch_exchange_info(
                    session,
                    ingestion_run_id=RUN_ID,
                    budget=budget,
                    symbol="BTCUSDT",
                    endpoint=f"{site.name}/exchangeInfo",
                    budget_now=0,
                )
    finally:
        await runner.cleanup()

    assert captured.raw_payload == payload
    assert len(captured.events) == 1
    assert captured.events[0].instrument.symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_exchange_clock_measurement_uses_midpoint_and_shared_budget() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"serverTime": time_ns() // 1_000})

    app = web.Application()
    app.router.add_get("/time", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    budget = RestWeightBudget(venue_capacity=2, interval_seconds=60)
    try:
        async with aiohttp.ClientSession() as session:
            measurement = await measure_exchange_clock(
                session,
                budget=budget,
                endpoint=f"{site.name}/time",
                budget_now=0,
            )
            with pytest.raises(LiveResourceError, match="budget is exhausted"):
                await measure_exchange_clock(
                    session,
                    budget=budget,
                    endpoint=f"{site.name}/time",
                    budget_now=0,
                )
    finally:
        await runner.cleanup()

    assert abs(measurement.offset_ms) < Decimal(500)
    assert measurement.round_trip_ns >= 0


def test_subscription_plan_bounds_shards_and_always_adds_btc_reference() -> None:
    plan = SubscriptionPlan(
        breadth_symbols=("ETHUSDT", "SOLUSDT", "ETHUSDT"),
        hot_symbols=("ETHUSDT",),
        breadth_shard_size=1,
    )

    assert plan.breadth_shards == (
        ("ethusdt@kline_1m",),
        ("solusdt@kline_1m",),
    )
    assert plan.hot_streams == (
        "ethusdt@aggTrade",
        "ethusdt@bookTicker",
        "btcusdt@aggTrade",
        "btcusdt@bookTicker",
    )

    with pytest.raises(ValueError, match="hot symbol count"):
        SubscriptionPlan(
            breadth_symbols=(),
            hot_symbols=tuple(f"A{index}USDT" for index in range(21)),
        )


def test_overlap_deduplication_and_aggregate_gap_refusal_cross_generations() -> None:
    state = LiveAdmissionState()
    first = decode_combined_message(_agg_payload(100), context=_context(0)).event
    overlap = decode_combined_message(_agg_payload(100), context=_context(1)).event
    gap = decode_combined_message(_agg_payload(103), context=_context(1)).event

    assert state.admit(first) == AdmissionDecision(QualityState.VALID, True)
    duplicate = state.admit(overlap)
    assert duplicate.state is QualityState.DUPLICATE
    assert not duplicate.accepted_for_downstream
    gap_decision = state.admit(gap)
    assert gap_decision.state is QualityState.GAP
    assert gap_decision.recovery_required
    assert not gap_decision.accepted_for_downstream


def test_overlap_identity_memory_is_bounded_without_losing_last_final_kline() -> None:
    state = LiveAdmissionState(recent_identity_capacity=2)
    final_kline = decode_combined_message(_kline_payload(), context=_context()).event
    trades = tuple(
        decode_combined_message(_agg_payload(value), context=_context()).event
        for value in (100, 101, 102)
    )

    assert state.admit(final_kline).accepted_for_downstream
    assert all(state.admit(trade).accepted_for_downstream for trade in trades)
    assert state.recent_identity_count == 2
    assert state.recent_identity_high_water_mark == 2
    assert state.recent_identity_evictions == 1
    assert state.tracked_instrument_count == 1

    recent_duplicate = decode_combined_message(
        _agg_payload(101),
        context=_context(1),
    ).event
    evicted_regression = decode_combined_message(
        _agg_payload(100),
        context=_context(1),
    ).event
    repeated_kline = decode_combined_message(
        _kline_payload(),
        context=_context(1),
    ).event
    assert state.admit(recent_duplicate).state is QualityState.DUPLICATE
    assert state.admit(evicted_regression).state is QualityState.CONFLICT
    assert state.admit(repeated_kline).state is QualityState.DUPLICATE
    assert state.recent_identity_count == 2


def test_book_ticker_ids_need_not_be_contiguous_but_cannot_regress() -> None:
    state = LiveAdmissionState()
    first = decode_combined_message(_book_payload(10), context=_context()).event
    later = decode_combined_message(_book_payload(25), context=_context()).event
    regressed = decode_combined_message(_book_payload(20), context=_context()).event

    assert state.admit(first).accepted_for_downstream
    assert state.admit(later).accepted_for_downstream
    refusal = state.admit(regressed)
    assert refusal.state is QualityState.CONFLICT
    assert not refusal.accepted_for_downstream


@pytest.mark.asyncio
async def test_bounded_queue_never_blocks_and_marks_overflow_incomplete() -> None:
    first = decode_combined_message(_agg_payload(100), context=_context())
    second = decode_combined_message(_agg_payload(101), context=_context())
    queue = BoundedCaptureQueue(capacity=1)

    assert queue.offer(first).enqueued
    overflow = queue.offer(second)

    assert not overflow.enqueued
    assert overflow.state is QualityState.INCOMPLETE
    assert queue.capacity == 1
    assert queue.high_water_mark == 1
    assert queue.overflow_count == 1
    assert queue.size == 1
    assert await queue.get() == first
    queue.task_done()


def test_connection_attempt_budget_reserves_half_the_ip_ceiling() -> None:
    budget = ConnectionAttemptBudget(
        venue_capacity=4,
        interval_seconds=10,
    )

    assert budget.available == Decimal("2.0")
    assert budget.try_acquire(now=0)
    assert budget.try_acquire(now=0)
    assert not budget.try_acquire(now=0)


@pytest.mark.asyncio
async def test_rest_rate_limit_retains_retry_after_without_retrying() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=429, headers={"Retry-After": "7"})

    app = web.Application()
    app.router.add_get("/exchangeInfo", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    budget = RestWeightBudget(venue_capacity=40, interval_seconds=60)
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(LiveRateLimitError) as captured:
                await fetch_exchange_info(
                    session,
                    ingestion_run_id=RUN_ID,
                    budget=budget,
                    endpoint=f"{site.name}/exchangeInfo",
                    budget_now=0,
                )
    finally:
        await runner.cleanup()

    assert captured.value.status == 429
    assert captured.value.retry_after_seconds == 7


def test_rest_budget_reserves_half_capacity_and_refills_deterministically() -> None:
    budget = RestWeightBudget(venue_capacity=100, interval_seconds=10)

    assert budget.available == Decimal("50.0")
    assert budget.try_acquire(40, now=0)
    assert not budget.try_acquire(11, now=0)
    assert budget.try_acquire(20, now=2)
    assert not budget.try_acquire(1, now=2)

    with pytest.raises(ValueError, match="backwards"):
        budget.try_acquire(1, now=1)


async def _start_server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"{site.name}/stream"


@pytest.mark.asyncio
async def test_collector_reconnects_and_deduplicates_rotation_overlap() -> None:
    connections = 0
    first_payload = _agg_payload(100).decode()
    next_payload = _agg_payload(101).decode()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        if connections == 1:
            await socket.send_str(first_payload)
        else:
            await socket.send_str(first_payload)
            await socket.send_str(next_payload)
        await socket.close()
        return socket

    runner, url = await _start_server(handler)
    try:
        queue = BoundedCaptureQueue(capacity=4)
        collector = BinanceLiveCollector(
            streams=("ethusdt@aggTrade",),
            ingestion_run_id=RUN_ID,
            queue=queue,
            reconnect_delay_seconds=0.01,
        )
        collector.url = url
        stats = await asyncio.wait_for(
            collector.run(stop=asyncio.Event(), max_messages=3),
            timeout=2,
        )
    finally:
        await runner.cleanup()

    assert stats.messages_received == 3
    assert stats.messages_enqueued == 2
    assert stats.messages_refused == 1
    assert stats.duplicate_messages == 1
    assert stats.reconnects == 1
    assert queue.size == 2


@pytest.mark.asyncio
async def test_collector_audits_schema_mismatch_without_reconnecting() -> None:
    connections = 0
    invalid_payload = '{"stream":"ethusdt@bookTicker","data":{"s":"ETHUSDT"}}'
    valid_payload = _agg_payload(100).decode()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str(invalid_payload)
        await socket.send_str(valid_payload)
        await socket.close()
        return socket

    runner, url = await _start_server(handler)
    try:
        queue = BoundedCaptureQueue(capacity=2)
        raw_queue = BoundedRawCaptureQueue(capacity=2)
        collector = BinanceLiveCollector(
            streams=("ethusdt@aggTrade",),
            ingestion_run_id=RUN_ID,
            queue=queue,
            raw_queue=raw_queue,
            reconnect_delay_seconds=0.01,
        )
        collector.url = url
        stats = await asyncio.wait_for(
            collector.run(stop=asyncio.Event(), max_messages=2),
            timeout=2,
        )
    finally:
        await runner.cleanup()

    first = await raw_queue.get()
    second = await raw_queue.get()
    assert first.raw_payload == invalid_payload.encode()
    assert first.retention_reason == "SCHEMA_MISMATCH"
    assert second.raw_payload == valid_payload.encode()
    assert second.retention_reason == "VALID"
    assert stats.schema_mismatches == 1
    assert stats.messages_enqueued == 1
    assert stats.reconnects == 0
    assert connections == 1
    assert raw_queue.high_water_mark == 2


@pytest.mark.asyncio
async def test_collector_retains_one_provisional_sample_per_symbol_minute() -> None:
    payload = _kline_payload(closed=False).decode()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        for _ in range(3):
            await socket.send_str(payload)
        await socket.close()
        return socket

    runner, url = await _start_server(handler)
    try:
        queue = BoundedCaptureQueue(capacity=2)
        raw_queue = BoundedRawCaptureQueue(capacity=2)
        collector = BinanceLiveCollector(
            streams=("ethusdt@kline_1m",),
            ingestion_run_id=RUN_ID,
            queue=queue,
            raw_queue=raw_queue,
            reconnect_delay_seconds=0.01,
        )
        collector.url = url
        stats = await asyncio.wait_for(
            collector.run(stop=asyncio.Event(), max_messages=3),
            timeout=2,
        )
    finally:
        await runner.cleanup()

    retained = await raw_queue.get()
    assert retained.raw_payload == payload.encode()
    assert retained.retention_reason == "PROVISIONAL_SAMPLE"
    assert raw_queue.size == 0
    assert stats.provisional_messages == 3
    assert stats.raw_messages_retained == 1
    assert stats.raw_messages_suppressed == 2


@pytest.mark.asyncio
async def test_collector_rotates_with_live_overlap_before_closing_old_socket() -> None:
    connections = 0
    first_payload = _agg_payload(100).decode()
    next_payload = _agg_payload(101).decode()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str(first_payload)
        if connections > 1:
            await socket.send_str(next_payload)
        await socket.receive()
        return socket

    runner, url = await _start_server(handler)
    try:
        queue = BoundedCaptureQueue(capacity=4)
        collector = BinanceLiveCollector(
            streams=("ethusdt@aggTrade",),
            ingestion_run_id=RUN_ID,
            queue=queue,
            reconnect_delay_seconds=0.01,
            rotation_seconds=0.05,
            overlap_ready_seconds=0.5,
        )
        collector.url = url
        stats = await asyncio.wait_for(
            collector.run(stop=asyncio.Event(), max_messages=3),
            timeout=2,
        )
    finally:
        await runner.cleanup()

    assert connections == 2
    assert stats.messages_received == 3
    assert stats.messages_enqueued == 2
    assert stats.messages_refused == 1
    assert stats.reconnects == 0
    assert stats.rotations == 1
    assert queue.size == 2


@pytest.mark.asyncio
async def test_collector_stop_event_cancels_an_idle_connection_cleanly() -> None:
    connected = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        connected.set()
        await release.wait()
        await socket.close()
        return socket

    runner, url = await _start_server(handler)
    stop = asyncio.Event()
    queue = BoundedCaptureQueue(capacity=1)
    collector = BinanceLiveCollector(
        streams=("ethusdt@aggTrade",),
        ingestion_run_id=RUN_ID,
        queue=queue,
        reconnect_delay_seconds=0.01,
    )
    collector.url = url
    ready = asyncio.Event()
    task = asyncio.create_task(collector.run(stop=stop, ready=ready))
    try:
        await asyncio.wait_for(connected.wait(), timeout=1)
        await asyncio.wait_for(ready.wait(), timeout=1)
        stop.set()
        stats = await asyncio.wait_for(task, timeout=1)
    finally:
        release.set()
        await runner.cleanup()

    assert stats.messages_received == 0
    assert stats.reconnects == 0


@pytest.mark.asyncio
async def test_collector_exposes_receive_utc_day_stats_without_restart(
    monkeypatch,
) -> None:
    first_receipt = datetime(2026, 9, 20, 23, 59, 59, tzinfo=UTC)
    second_receipt = first_receipt + timedelta(seconds=2)
    receipt_times = iter(
        (
            int(first_receipt.timestamp()) * 1_000_000_000,
            int(second_receipt.timestamp()) * 1_000_000_000,
        )
    )
    monkeypatch.setattr("crypto_boom.live.time_ns", lambda: next(receipt_times))

    async def handler(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_str(_agg_payload(100).decode())
        await socket.send_str(_agg_payload(101).decode())
        await socket.receive()
        return socket

    runner, url = await _start_server(handler)
    queue = BoundedCaptureQueue(capacity=2)
    collector = BinanceLiveCollector(
        streams=("ethusdt@aggTrade",),
        ingestion_run_id=RUN_ID,
        queue=queue,
    )
    collector.url = url
    try:
        total = await asyncio.wait_for(
            collector.run(stop=asyncio.Event(), max_messages=2),
            timeout=1,
        )
    finally:
        await runner.cleanup()

    first = collector.stats_for_day(first_receipt.date())
    second = collector.stats_for_day(second_receipt.date())
    assert total.messages_received == 2
    assert first.messages_received == first.messages_enqueued == 1
    assert second.messages_received == second.messages_enqueued == 1
    assert first.reconnects == second.reconnects == 0
    assert queue.stats_for_day(first_receipt.date()).high_water_mark == 1
    assert queue.stats_for_day(second_receipt.date()).high_water_mark == 2
    assert (
        collector.admission.stats_for_day(
            first_receipt.date()
        ).recent_identity_high_water_mark
        == 1
    )
    assert (
        collector.admission.stats_for_day(
            second_receipt.date()
        ).recent_identity_high_water_mark
        == 2
    )

    collector.discard_stats_before(second_receipt.date())
    queue.discard_stats_before(second_receipt.date())
    collector.admission.discard_stats_before(second_receipt.date())
    assert collector.stats_for_day(first_receipt.date()).messages_received == 0
    assert collector.stats_for_day(second_receipt.date()).messages_received == 1
    assert queue.stats_for_day(first_receipt.date()).high_water_mark == 0
    assert (
        collector.admission.stats_for_day(
            first_receipt.date()
        ).recent_identity_high_water_mark
        == 0
    )
