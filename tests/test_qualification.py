from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import UUID

import aiohttp
import pytest

import crypto_boom.qualification as qualification
from crypto_boom.binance_source import (
    ClockMeasurement,
    MetadataCapture,
    RestWeightBudget,
    decode_exchange_info,
    select_spot_usdt_universe,
)
from crypto_boom.live import (
    BoundedCaptureQueue,
    BoundedRawCaptureQueue,
    CapturedEvidence,
    CaptureStats,
    DecodeContext,
    LiveAdmissionState,
    RawMessageObservation,
    SubscriptionPlan,
    decode_combined_message,
)
from crypto_boom.market import LocalReceipt
from crypto_boom.qualification import (
    CampaignFacts,
    QualificationError,
    QualificationLimits,
    QualificationPublicationError,
    publish_qualification_campaign,
)

RUN_ID = UUID("8386f2c3-b925-40ce-8c6d-4785ea0f37ce")
RECEIPT_NS = 1_795_027_260_000_001_000
RECEIPT = LocalReceipt(RECEIPT_NS, 100)
DAY = datetime.fromtimestamp(RECEIPT_NS / 1_000_000_000, tz=UTC).date()


def _metadata() -> MetadataCapture:
    payload = json.dumps(
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.0001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                },
                {
                    "symbol": "ETHBTC",
                    "status": "TRADING",
                    "baseAsset": "ETH",
                    "quoteAsset": "BTC",
                    "permissions": ["SPOT"],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.000001"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.0001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "0.0001"},
                    ],
                },
                {
                    "symbol": "OLDUSDT",
                    "status": "BREAK",
                    "baseAsset": "OLD",
                    "quoteAsset": "USDT",
                    "permissions": ["SPOT"],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "1"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                },
            ]
        },
        separators=(",", ":"),
    ).encode()
    return decode_exchange_info(
        payload,
        ingestion_run_id=RUN_ID,
        receipt=RECEIPT,
    )


def _combined(stream: str, data: dict[str, object]) -> bytes:
    return json.dumps(
        {"stream": stream, "data": data},
        separators=(",", ":"),
    ).encode()


def _kline() -> bytes:
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
                "f": 10,
                "L": 11,
                "o": "1900",
                "c": "2000",
                "h": "2100",
                "l": "1800",
                "v": "10",
                "n": 2,
                "x": True,
                "q": "20000",
                "V": "4",
                "Q": "8000",
            },
        },
    )


def _aggregate_trade() -> bytes:
    return _combined(
        "ethusdt@aggTrade",
        {
            "e": "aggTrade",
            "E": 1_795_027_259_999_999,
            "s": "ETHUSDT",
            "a": 100,
            "p": "2000",
            "q": "1",
            "f": 1000,
            "l": 1001,
            "T": 1_795_027_259_999_998,
            "m": False,
        },
    )


def _captures():
    context = DecodeContext(RUN_ID, RECEIPT, 0)
    return (
        decode_combined_message(_kline(), context=context),
        decode_combined_message(_aggregate_trade(), context=context),
    )


def _facts(*, stats: CaptureStats | None = None) -> CampaignFacts:
    return CampaignFacts(
        day=DAY,
        started_at_ns=RECEIPT_NS - 60_000_000_000,
        ended_at_ns=RECEIPT_NS,
        expected_closed_klines=1,
        queue_capacity=8,
        queue_high_water_mark=2,
        queue_overflows=0,
        collector_stats=(stats or CaptureStats(2, 2, 0, 0, 1),),
        clock_offset_ms=Decimal("0"),
        representative=True,
    )


def _full_day_publications(
    output_root: Path,
) -> tuple[
    qualification.PublishedQualification,
    qualification.PublishedQualification,
]:
    next_day = DAY + timedelta(days=1)
    day_start_ns, boundary_ns = qualification._utc_day_bounds(DAY)
    _, third_day_ns = qualification._utc_day_bounds(next_day)
    sink = qualification.RotatingQualificationSink(
        metadata=_metadata(),
        output_root=output_root,
        limits=QualificationLimits(minimum_free_disk_bytes=1),
        clock_offset_ms=Decimal("0"),
    )
    sink.open_day(DAY, started_at_ns=day_start_ns)
    sink.open_day(next_day, started_at_ns=boundary_ns)
    empty_stats = (CaptureStats(0, 0, 0, 0, 0),)
    first = sink.finalize_day(
        CampaignFacts(
            day=DAY,
            started_at_ns=day_start_ns,
            ended_at_ns=boundary_ns,
            expected_closed_klines=1_440,
            queue_capacity=2,
            queue_high_water_mark=0,
            queue_overflows=0,
            collector_stats=empty_stats,
            clock_offset_ms=Decimal("0"),
        )
    )
    second = sink.finalize_day(
        CampaignFacts(
            day=next_day,
            started_at_ns=boundary_ns,
            ended_at_ns=third_day_ns,
            expected_closed_klines=1_440,
            queue_capacity=2,
            queue_high_water_mark=0,
            queue_overflows=0,
            collector_stats=empty_stats,
            clock_offset_ms=Decimal("0"),
        )
    )
    return first, second


def _full_day_publication(
    output_root: Path,
    observed_day: date,
) -> qualification.PublishedQualification:
    started_at_ns, ended_at_ns = qualification._utc_day_bounds(observed_day)
    return publish_qualification_campaign(
        (),
        metadata=_metadata(),
        facts=CampaignFacts(
            day=observed_day,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            expected_closed_klines=1_440,
            queue_capacity=2,
            queue_high_water_mark=0,
            queue_overflows=0,
            collector_stats=(CaptureStats(0, 0, 0, 0, 0),),
            clock_offset_ms=Decimal("0"),
        ),
        output_root=output_root,
        limits=QualificationLimits(minimum_free_disk_bytes=1),
    )


def test_selects_only_observed_trading_spot_usdt_symbols() -> None:
    assert select_spot_usdt_universe(_metadata().events) == ("ETHUSDT",)


def test_publication_segments_exact_raw_and_is_content_idempotent(
    tmp_path: Path,
) -> None:
    limits = QualificationLimits(
        queue_capacity=8,
        segment_records=1,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
    )

    first = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
        limits=limits,
    )
    second = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
        limits=limits,
    )

    assert not first.already_present
    assert second.already_present
    assert first.path == second.path
    assert first.manifest == second.manifest
    loaded = qualification.load_published_qualification(first.path)
    assert loaded.already_present
    assert loaded.path == first.path.resolve()
    assert loaded.manifest == first.manifest
    assert loaded.report == first.report
    assert first.manifest.schema_version == 3
    assert first.manifest.capture_record_schema_version == 1
    assert first.manifest.quality_report_schema_version == 1
    assert len(first.manifest.segments) == 2
    assert {segment.content_encoding for segment in first.manifest.segments} == {"gzip"}
    assert all(
        segment.name.endswith(".jsonl.gz") for segment in first.manifest.segments
    )
    assert first.report.capture_compressed_bytes < (
        first.report.capture_uncompressed_bytes
    )
    assert first.report.ready
    assert first.report.closed_kline_coverage == "1"
    assert first.report.rotations == 1

    report_payload = json.loads((first.path / "quality-report.json").read_bytes())
    assert report_payload["schema_version"] == 1
    assert report_payload["day_basis"] == "receipt_wall_time_utc"
    assert report_payload["window_started_at_ns"] == _facts().started_at_ns
    assert report_payload["window_ended_at_ns"] == _facts().ended_at_ns
    assert not report_payload["window_complete_utc_day"]
    assert sum(report_payload["raw_event_latency_histogram"]["counts"]) == 2
    assert sum(report_payload["event_latency_histogram"]["counts"]) == 2
    assert sum(report_payload["raw_closed_kline_latency_histogram"]["counts"]) == 1
    assert sum(report_payload["closed_kline_latency_histogram"]["counts"]) == 1
    for name in (
        "raw_event_latency_histogram",
        "event_latency_histogram",
        "raw_closed_kline_latency_histogram",
        "closed_kline_latency_histogram",
    ):
        histogram = report_payload[name]
        assert histogram["schema_version"] == 1
        assert histogram["unit"] == "nanoseconds"
        assert histogram["bucket_rule"] == "bit_length_power_of_two_upper_bound"
        assert len(histogram["counts"]) == 65

    lines: list[str] = []
    for segment in first.manifest.segments:
        path = first.path / segment.name
        with gzip.open(path, mode="rb") as stream:
            uncompressed = stream.read()
        assert segment.bytes == path.stat().st_size
        assert segment.uncompressed_bytes == len(uncompressed)
        lines.extend(uncompressed.decode("utf-8").splitlines())
    decoded_lines = [json.loads(line) for line in lines]
    assert {line["schema_version"] for line in decoded_lines} == {1}
    persisted = [base64.b64decode(line["raw_payload_base64"]) for line in decoded_lines]
    assert persisted == [_kline(), _aggregate_trade()]
    assert {line["retention_reason"] for line in decoded_lines} == {"VALID"}
    assert (first.path / "exchange-info.json").read_bytes() == _metadata().raw_payload


def test_report_marks_only_an_exact_utc_day_window_as_complete(tmp_path: Path) -> None:
    day_start_ns = int(
        datetime(DAY.year, DAY.month, DAY.day, tzinfo=UTC).timestamp() * 1_000_000_000
    )
    facts = CampaignFacts(
        day=DAY,
        started_at_ns=day_start_ns,
        ended_at_ns=day_start_ns + 24 * 60 * 60 * 1_000_000_000,
        expected_closed_klines=1,
        queue_capacity=8,
        queue_high_water_mark=2,
        queue_overflows=0,
        collector_stats=(CaptureStats(2, 2, 0, 0, 1),),
        clock_offset_ms=Decimal("0"),
        representative=False,
    )

    result = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=facts,
        output_root=tmp_path,
    )

    assert result.report.window_complete_utc_day
    assert not result.report.representative


def test_cross_day_aggregation_merges_histograms_instead_of_daily_percentiles(
    tmp_path: Path,
) -> None:
    report = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
    ).report
    low_counts = (0, 1, *(0 for _ in range(63)))
    high_counts = (0, *(0 for _ in range(9)), 1, *(0 for _ in range(54)))
    first_start = datetime(DAY.year, DAY.month, DAY.day, tzinfo=UTC)
    second_start = first_start + timedelta(days=1)
    first = replace(
        report,
        report_id="day-1",
        day=DAY.isoformat(),
        window_started_at_ns=int(first_start.timestamp()) * 1_000_000_000,
        window_ended_at_ns=int(second_start.timestamp()) * 1_000_000_000,
        window_complete_utc_day=True,
        event_latency_p95_ns=2,
        event_latency_histogram=replace(
            report.event_latency_histogram,
            counts=low_counts,
        ),
        representative=False,
        ready=False,
        critical_failures=("REPRESENTATIVE_WINDOW_PENDING",),
    )
    third_start = second_start + timedelta(days=1)
    second = replace(
        report,
        report_id="day-2",
        day=(DAY + timedelta(days=1)).isoformat(),
        window_started_at_ns=int(second_start.timestamp()) * 1_000_000_000,
        window_ended_at_ns=int(third_start.timestamp()) * 1_000_000_000,
        window_complete_utc_day=True,
        event_latency_p95_ns=1_024,
        event_latency_histogram=replace(
            report.event_latency_histogram,
            counts=high_counts,
        ),
        representative=False,
        ready=False,
        critical_failures=("REPRESENTATIVE_WINDOW_PENDING",),
    )

    aggregate = qualification.aggregate_quality_reports(
        (second, first),
        minimum_complete_days=2,
    )
    repeated = qualification.aggregate_quality_reports(
        (first, second),
        minimum_complete_days=2,
    )

    assert aggregate == repeated
    assert aggregate.days == 2
    assert aggregate.report_ids == ("day-1", "day-2")
    assert aggregate.event_latency_p95_ns == 1_024
    assert sum(aggregate.event_latency_histogram.counts) == 2
    assert aggregate.representative
    assert aggregate.ready
    assert not aggregate.critical_failures


def test_cross_day_aggregation_refuses_a_missing_utc_day(tmp_path: Path) -> None:
    report = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
    ).report
    later = replace(report, day=(DAY + timedelta(days=2)).isoformat())

    with pytest.raises(QualificationError, match="contiguous UTC days"):
        qualification.aggregate_quality_reports((report, later))


def test_cross_day_aggregation_keeps_partial_windows_non_representative(
    tmp_path: Path,
) -> None:
    report = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
    ).report

    aggregate = qualification.aggregate_quality_reports(
        (report,),
        minimum_complete_days=1,
    )

    assert not aggregate.representative
    assert not aggregate.ready
    assert set(aggregate.critical_failures) >= {
        "INCOMPLETE_UTC_DAY_WINDOW",
        "REPRESENTATIVE_WINDOW_PENDING",
    }


def test_cross_day_publication_is_idempotent_and_links_daily_sources(
    tmp_path: Path,
) -> None:
    first_day, second_day = _full_day_publications(tmp_path)

    first = qualification.publish_cross_day_quality_report(
        (second_day, first_day),
        output_root=tmp_path,
        minimum_complete_days=2,
    )
    repeated = qualification.publish_cross_day_quality_report(
        (first_day, second_day),
        output_root=tmp_path,
        minimum_complete_days=2,
    )

    assert not first.already_present
    assert repeated.already_present
    assert repeated.path == first.path
    assert first.report.aggregate_id == first.manifest.aggregate_id
    assert tuple(source.day for source in first.manifest.sources) == (
        DAY.isoformat(),
        (DAY + timedelta(days=1)).isoformat(),
    )
    assert tuple(source.manifest_id for source in first.manifest.sources) == (
        first_day.manifest.manifest_id,
        second_day.manifest.manifest_id,
    )
    assert (
        json.loads((first.path / "aggregate-quality-report.json").read_bytes())[
            "aggregate_id"
        ]
        == first.report.aggregate_id
    )


def test_publication_refuses_a_tampered_existing_target(tmp_path: Path) -> None:
    limits = QualificationLimits(
        queue_capacity=8,
        segment_records=1,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
    )
    first = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=_facts(),
        output_root=tmp_path,
        limits=limits,
    )
    (first.path / first.manifest.segments[0].name).write_bytes(b"tampered\n")

    with pytest.raises(
        QualificationPublicationError,
        match="existing qualification file conflicts with its manifest",
    ):
        publish_qualification_campaign(
            _captures(),
            metadata=_metadata(),
            facts=_facts(),
            output_root=tmp_path,
            limits=limits,
        )


def test_cross_day_publication_refuses_a_tampered_existing_target(
    tmp_path: Path,
) -> None:
    first_day, second_day = _full_day_publications(tmp_path)
    first = qualification.publish_cross_day_quality_report(
        (first_day, second_day),
        output_root=tmp_path,
        minimum_complete_days=2,
    )
    (first.path / "aggregate-quality-report.json").write_bytes(b"{}\n")

    with pytest.raises(
        QualificationPublicationError,
        match="existing cross-day report conflicts with its manifest",
    ):
        qualification.publish_cross_day_quality_report(
            (first_day, second_day),
            output_root=tmp_path,
            minimum_complete_days=2,
        )


def test_publication_verifies_what_a_lost_replace_race_left_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery branch must verify the directory it finds, not merely adopt it.

    `os.replace` is made to lose the race and leave tampered content behind, so
    this fails if the recover-after-OSError branch ever stops verifying: the
    publication would be reported as already present without its contents ever
    having been checked.
    """

    limits = QualificationLimits(
        queue_capacity=8,
        segment_records=1,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
    )

    def losing_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        shutil.copytree(source, target)
        (Path(target) / "quality-report.json").write_bytes(b"{}\n")
        raise OSError("lost the publication race")

    monkeypatch.setattr(qualification.os, "replace", losing_replace)

    with pytest.raises(
        QualificationPublicationError,
        match="existing qualification file conflicts with its manifest",
    ):
        publish_qualification_campaign(
            _captures(),
            metadata=_metadata(),
            facts=_facts(),
            output_root=tmp_path,
            limits=limits,
        )


def test_cross_day_publication_refuses_tampered_daily_source(tmp_path: Path) -> None:
    first_day, second_day = _full_day_publications(tmp_path)
    (first_day.path / "quality-report.json").write_bytes(b"{}\n")

    with pytest.raises(
        QualificationPublicationError,
        match="conflicts with its manifest",
    ):
        qualification.publish_cross_day_quality_report(
            (first_day, second_day),
            output_root=tmp_path,
            minimum_complete_days=2,
        )


def test_rolling_cross_day_publisher_waits_for_its_complete_window(
    tmp_path: Path,
) -> None:
    first_day, second_day = _full_day_publications(tmp_path)
    publisher = qualification.RollingCrossDayQualityPublisher(
        tmp_path,
        window_days=2,
    )

    assert publisher.observe(first_day) is None
    published = publisher.observe(second_day)

    assert published is not None
    assert published.report.days == 2
    assert published.report.representative


def test_rolling_cross_day_publisher_hydrates_before_next_day(
    tmp_path: Path,
) -> None:
    first_day, second_day = _full_day_publications(tmp_path)
    publisher = qualification.RollingCrossDayQualityPublisher(
        tmp_path,
        window_days=3,
    )

    assert publisher.hydrate_from_disk() == 2
    third_day = _full_day_publication(tmp_path, DAY + timedelta(days=2))
    published = publisher.observe(third_day)

    assert published is not None
    assert published.report.report_ids == (
        first_day.report.report_id,
        second_day.report.report_id,
        third_day.report.report_id,
    )


def test_hydration_refuses_tampered_daily_artifact(tmp_path: Path) -> None:
    first_day, _ = _full_day_publications(tmp_path)
    (first_day.path / "quality-report.json").write_bytes(b"{}\n")
    publisher = qualification.RollingCrossDayQualityPublisher(
        tmp_path,
        window_days=3,
    )

    with pytest.raises(QualificationPublicationError, match="invalid typed content"):
        publisher.hydrate_from_disk()


def test_rotating_sink_publishes_days_independently_and_refuses_late_writes(
    tmp_path: Path,
) -> None:
    next_receipt_ns = RECEIPT_NS + 24 * 60 * 60 * 1_000_000_000
    next_receipt = LocalReceipt(next_receipt_ns, 200)
    next_capture = decode_combined_message(
        _aggregate_trade(),
        context=DecodeContext(RUN_ID, next_receipt, 0),
    )
    next_day = DAY + timedelta(days=1)
    sink = qualification.RotatingQualificationSink(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(),
        clock_offset_ms=Decimal("0"),
    )
    sink.open_day(DAY, started_at_ns=_facts().started_at_ns)
    sink.open_day(next_day, started_at_ns=next_receipt_ns - 1)
    sink.append(_captures()[0])
    sink.append(next_capture)

    first = sink.finalize_day(_facts())
    second_facts = replace(
        _facts(),
        day=next_day,
        started_at_ns=next_receipt_ns - 1,
        ended_at_ns=next_receipt_ns,
        expected_closed_klines=None,
        collector_stats=(CaptureStats(1, 1, 0, 0, 0),),
        representative=False,
    )
    second = sink.finalize_day(second_facts)

    assert first.report.day == DAY.isoformat()
    assert second.report.day == next_day.isoformat()
    assert first.path != second.path
    with pytest.raises(QualificationPublicationError, match="already finalized"):
        sink.append(_captures()[1])


def test_campaign_facts_use_only_the_requested_receive_utc_day() -> None:
    capture = _captures()[1]
    queue = BoundedCaptureQueue(capacity=2)
    raw_queue = BoundedRawCaptureQueue(capacity=2)
    admission = LiveAdmissionState(recent_identity_capacity=2)
    assert queue.offer(capture).enqueued
    assert admission.admit(capture.event).accepted_for_downstream
    provenance = capture.event.provenance
    assert raw_queue.offer(
        RawMessageObservation(
            raw_payload=capture.raw_payload,
            ingestion_run_id=provenance.ingestion_run_id,
            receipt=provenance.receipt,
            connection_generation=provenance.connection_generation or 0,
            retention_reason="VALID",
        )
    ).enqueued

    class Collector:
        def stats_for_day(self, day: date) -> CaptureStats:
            assert day == DAY
            return CaptureStats(3, 2, 1, 0, 0, raw_messages_retained=1)

    facts = qualification.campaign_facts_for_day(
        DAY,
        started_at_ns=RECEIPT_NS - 1,
        ended_at_ns=RECEIPT_NS,
        expected_closed_klines=1,
        collectors=(Collector(),),
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
        clock_offset_ms=Decimal("0"),
    )

    assert facts.collector_stats == (
        CaptureStats(3, 2, 1, 0, 0, raw_messages_retained=1),
    )
    assert facts.queue_capacity == 2
    assert facts.queue_high_water_mark == 1
    assert facts.queue_overflows == 0
    assert facts.recent_identity_capacity == 2
    assert facts.recent_identity_high_water_mark == 1
    assert facts.recent_identity_evictions == 0


def test_rollover_controller_opens_next_day_before_finalizing_and_discards_stats(
    tmp_path: Path,
) -> None:
    next_day = DAY + timedelta(days=1)
    boundary_ns = (
        int(
            datetime(
                next_day.year, next_day.month, next_day.day, tzinfo=UTC
            ).timestamp()
        )
        * 1_000_000_000
    )
    queue = BoundedCaptureQueue(capacity=4)
    raw_queue = BoundedRawCaptureQueue(capacity=4)
    admission = LiveAdmissionState(recent_identity_capacity=4)

    class Collector:
        def __init__(self) -> None:
            self.discarded_before: date | None = None

        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(1, 1, 0, 0, 0, raw_messages_retained=1)

        def discard_stats_before(self, day: date) -> None:
            self.discarded_before = day

    collector = Collector()
    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(),
        clock_offset_ms=Decimal("0"),
        collectors=(collector,),
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=_facts().started_at_ns)

    capture = _captures()[1]
    assert queue.offer(capture).enqueued
    assert admission.admit(capture.event).accepted_for_downstream
    provenance = capture.event.provenance
    observed = RawMessageObservation(
        raw_payload=capture.raw_payload,
        ingestion_run_id=provenance.ingestion_run_id,
        receipt=provenance.receipt,
        connection_generation=provenance.connection_generation or 0,
        retention_reason="VALID",
    )
    assert raw_queue.offer(observed).enqueued
    controller.append_raw(observed)
    controller.observe(capture)

    controller.prepare_next_day(
        boundary_ns,
        context=qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("1"),
        ),
    )
    published = controller.finalize_oldest_day(ended_at_ns=boundary_ns)

    assert published.report.day == DAY.isoformat()
    assert collector.discarded_before == next_day
    assert queue.stats_for_day(DAY).high_water_mark == 0
    assert raw_queue.stats_for_day(DAY).high_water_mark == 0
    assert admission.stats_for_day(DAY).recent_identity_high_water_mark == 0


def test_rollover_publication_failure_preserves_daily_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue = BoundedCaptureQueue(capacity=2)
    raw_queue = BoundedRawCaptureQueue(capacity=2)
    admission = LiveAdmissionState(recent_identity_capacity=2)

    class Collector:
        discarded = False

        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(1, 1, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            self.discarded = True

    collector = Collector()
    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(),
        clock_offset_ms=Decimal("0"),
        collectors=(collector,),
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=RECEIPT_NS - 1)
    capture = _captures()[1]
    assert queue.offer(capture).enqueued

    def fail_publication(facts: CampaignFacts):
        raise QualificationPublicationError("injected publication failure")

    monkeypatch.setattr(
        controller.sink,
        "finalize_day",
        fail_publication,
    )

    with pytest.raises(
        QualificationPublicationError,
        match="injected publication failure",
    ):
        controller.finalize_oldest_day(ended_at_ns=RECEIPT_NS)

    assert not collector.discarded
    assert queue.stats_for_day(DAY).high_water_mark == 1


@pytest.mark.asyncio
async def test_scheduled_rollover_prefetches_context_drains_and_publishes(
    tmp_path: Path,
) -> None:
    next_day = DAY + timedelta(days=1)
    boundary_ns = (
        int(
            datetime(
                next_day.year, next_day.month, next_day.day, tzinfo=UTC
            ).timestamp()
        )
        * 1_000_000_000
    )
    queue = BoundedCaptureQueue(capacity=2)
    raw_queue = BoundedRawCaptureQueue(capacity=2)
    admission = LiveAdmissionState(recent_identity_capacity=2)

    class Collector:
        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(1, 1, 0, 0, 0, raw_messages_retained=1)

        def discard_stats_before(self, day: date) -> None:
            pass

    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(),
        clock_offset_ms=Decimal("0"),
        collectors=(Collector(),),
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=RECEIPT_NS - 1)
    capture = _captures()[1]
    assert admission.admit(capture.event).accepted_for_downstream
    assert queue.offer(capture).enqueued
    provenance = capture.event.provenance
    assert raw_queue.offer(
        RawMessageObservation(
            raw_payload=capture.raw_payload,
            ingestion_run_id=provenance.ingestion_run_id,
            receipt=provenance.receipt,
            connection_generation=provenance.connection_generation or 0,
            retention_reason="VALID",
        )
    ).enqueued
    deadlines: list[int] = []

    async def wait_until(deadline_ns: int) -> None:
        deadlines.append(deadline_ns)

    async def context_for(day: date) -> qualification.QualificationDayContext:
        assert day == next_day
        return qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("2"),
        )

    published = await qualification.rollover_qualification_day(
        controller,
        context_for=context_for,
        sink_lock=asyncio.Lock(),
        wait_until=wait_until,
        now_ns=lambda: boundary_ns - 1,
        context_lead_seconds=30,
        drain_grace_seconds=600,
    )

    assert deadlines == [
        boundary_ns - 30_000_000_000,
        boundary_ns + 600_000_000_000,
    ]
    assert published.report.day == DAY.isoformat()
    assert controller.open_days == (next_day,)
    assert queue.size == raw_queue.size == 0


@pytest.mark.asyncio
async def test_continuous_consumer_holds_sink_lock_before_dequeueing() -> None:
    queue = BoundedCaptureQueue(capacity=2)
    raw_queue = BoundedRawCaptureQueue(capacity=2)
    capture = _captures()[1]
    assert queue.offer(capture).enqueued

    class Sink:
        def __init__(self) -> None:
            self.observed: list[CapturedEvidence] = []

        def append_raw_many(
            self, observations: Sequence[RawMessageObservation]
        ) -> None:
            pass

        def observe_many(self, captures: Sequence[CapturedEvidence]) -> None:
            self.observed.extend(captures)

    sink = Sink()
    collectors_done = asyncio.Event()
    collectors_done.set()
    sink_lock = asyncio.Lock()
    await sink_lock.acquire()
    consumer = asyncio.create_task(
        qualification._drain_capture_queues(
            queue,
            raw_queue,
            sink,
            collectors_done=collectors_done,
            sink_lock=sink_lock,
        )
    )
    await asyncio.sleep(0)

    assert queue.size == 1
    assert not sink.observed

    sink_lock.release()
    await consumer

    assert queue.size == 0
    assert sink.observed == [capture]


@pytest.mark.asyncio
async def test_scheduled_rollover_refuses_context_that_arrives_at_midnight(
    tmp_path: Path,
) -> None:
    next_day = DAY + timedelta(days=1)
    _, boundary_ns = qualification._utc_day_bounds(DAY)

    class Collector:
        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(0, 0, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            pass

    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(),
        clock_offset_ms=Decimal("0"),
        collectors=(Collector(),),
        queue=BoundedCaptureQueue(capacity=2),
        raw_queue=BoundedRawCaptureQueue(capacity=2),
        admission=LiveAdmissionState(recent_identity_capacity=2),
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=RECEIPT_NS - 1)

    async def context_for(day: date) -> qualification.QualificationDayContext:
        assert day == next_day
        return qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("0"),
        )

    async def wait_until(deadline_ns: int) -> None:
        pass

    with pytest.raises(
        QualificationError,
        match="not ready before UTC midnight",
    ):
        await qualification.rollover_qualification_day(
            controller,
            context_for=context_for,
            sink_lock=asyncio.Lock(),
            wait_until=wait_until,
            now_ns=lambda: boundary_ns,
        )

    assert controller.open_days == (DAY,)


@pytest.mark.asyncio
async def test_binance_context_provider_refreshes_metadata_and_clock(
    monkeypatch,
) -> None:
    session = cast(aiohttp.ClientSession, object())
    budget = RestWeightBudget(venue_capacity=100, interval_seconds=60)
    calls: list[str] = []

    async def fetch_metadata(
        requested_session: aiohttp.ClientSession,
        *,
        ingestion_run_id: UUID,
        budget: RestWeightBudget,
    ) -> MetadataCapture:
        assert requested_session is session
        assert ingestion_run_id == RUN_ID
        calls.append("metadata")
        return _metadata()

    async def fetch_clock(
        requested_session: aiohttp.ClientSession,
        *,
        budget: RestWeightBudget,
    ) -> ClockMeasurement:
        assert requested_session is session
        calls.append("clock")
        return ClockMeasurement(Decimal("3"), 1, RECEIPT_NS)

    monkeypatch.setattr(qualification, "fetch_exchange_info", fetch_metadata)
    monkeypatch.setattr(qualification, "measure_exchange_clock", fetch_clock)
    provider = qualification.BinanceQualificationContextProvider(
        session=session,
        ingestion_run_id=RUN_ID,
        budget=budget,
    )

    context = await provider(DAY + timedelta(days=1))

    assert calls == ["metadata", "clock"]
    assert context.metadata == _metadata()
    assert context.clock_offset_ms == Decimal("3")


@pytest.mark.asyncio
async def test_continuous_campaign_rolls_without_restarting_collectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, boundary_ns = qualification._utc_day_bounds(DAY)
    now = {"value": RECEIPT_NS}

    class Collector:
        def __init__(
            self,
            *,
            streams: tuple[str, ...],
            ingestion_run_id: UUID,
            queue: BoundedCaptureQueue,
            admission: LiveAdmissionState,
            raw_queue: BoundedRawCaptureQueue,
        ) -> None:
            self.run_calls = 0
            instances.append(self)

        async def run(
            self,
            *,
            stop: asyncio.Event,
            ready: asyncio.Event,
        ) -> CaptureStats:
            self.run_calls += 1
            ready.set()
            await stop.wait()
            return CaptureStats(0, 0, 0, 0, 0)

        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(0, 0, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            pass

    instances: list[Collector] = []
    monkeypatch.setattr(qualification, "BinanceLiveCollector", Collector)
    stop = asyncio.Event()
    deadlines: list[int] = []
    published_days: list[str] = []
    statuses: list[qualification.QualificationServiceStatus] = []

    async def wait_until(deadline_ns: int) -> None:
        deadlines.append(deadline_ns)

    async def context_for(day: date) -> qualification.QualificationDayContext:
        assert day == DAY + timedelta(days=1)
        return qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("1"),
        )

    async def on_published(result: qualification.PublishedQualification) -> None:
        published_days.append(result.report.day)
        if result.report.day == DAY.isoformat():
            now["value"] = boundary_ns + 1
            stop.set()

    async def on_status(status: qualification.QualificationServiceStatus) -> None:
        statuses.append(status)

    shutdown_publications = await qualification.run_continuous_qualification_campaign(
        SubscriptionPlan(breadth_symbols=("ETHUSDT",), hot_symbols=()),
        initial_context=qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("0"),
        ),
        context_for=context_for,
        output_root=tmp_path,
        stop=stop,
        limits=QualificationLimits(minimum_free_disk_bytes=1),
        context_lead_seconds=30,
        drain_grace_seconds=600,
        on_published=on_published,
        on_status=on_status,
        wait_until=wait_until,
        now_ns=lambda: now["value"],
    )

    assert deadlines == [
        boundary_ns - 30_000_000_000,
        boundary_ns + 600_000_000_000,
    ]
    assert published_days == [DAY.isoformat(), (DAY + timedelta(days=1)).isoformat()]
    assert tuple(result.report.day for result in shutdown_publications) == (
        (DAY + timedelta(days=1)).isoformat(),
    )
    assert instances
    assert all(instance.run_calls == 1 for instance in instances)
    assert tuple(status.phase for status in statuses) == (
        qualification.QualificationServicePhase.STARTING,
        qualification.QualificationServicePhase.READY,
        qualification.QualificationServicePhase.STOPPING,
        qualification.QualificationServicePhase.STOPPED,
    )
    assert tuple(status.ready for status in statuses) == (False, True, False, False)


@pytest.mark.asyncio
async def test_shutdown_aborts_preopened_future_day_before_partial_publication(
    tmp_path: Path,
) -> None:
    _, boundary_ns = qualification._utc_day_bounds(DAY)

    class Collector:
        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(0, 0, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            pass

    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(minimum_free_disk_bytes=1),
        clock_offset_ms=Decimal("0"),
        collectors=(Collector(),),
        queue=BoundedCaptureQueue(capacity=2),
        raw_queue=BoundedRawCaptureQueue(capacity=2),
        admission=LiveAdmissionState(recent_identity_capacity=2),
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=RECEIPT_NS)
    controller.prepare_next_day(
        boundary_ns,
        context=qualification.QualificationDayContext(
            metadata=_metadata(),
            clock_offset_ms=Decimal("1"),
        ),
    )

    published = await qualification._finalize_shutdown_days(
        controller,
        sink_lock=asyncio.Lock(),
        ended_at_ns=RECEIPT_NS + 1,
    )

    assert tuple(result.report.day for result in published) == (DAY.isoformat(),)
    assert not controller.open_days


@pytest.mark.asyncio
async def test_shutdown_can_publish_zero_duration_initial_window(
    tmp_path: Path,
) -> None:
    class Collector:
        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(0, 0, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            pass

    controller = qualification.QualificationRolloverController(
        metadata=_metadata(),
        output_root=tmp_path,
        limits=QualificationLimits(minimum_free_disk_bytes=1),
        clock_offset_ms=Decimal("0"),
        collectors=(Collector(),),
        queue=BoundedCaptureQueue(capacity=2),
        raw_queue=BoundedRawCaptureQueue(capacity=2),
        admission=LiveAdmissionState(recent_identity_capacity=2),
        breadth_symbol_count=1,
    )
    controller.open_initial_day(DAY, started_at_ns=RECEIPT_NS)

    published = await qualification._finalize_shutdown_days(
        controller,
        sink_lock=asyncio.Lock(),
        ended_at_ns=RECEIPT_NS,
    )

    assert tuple(result.report.day for result in published) == (DAY.isoformat(),)


@pytest.mark.asyncio
async def test_continuous_campaign_revokes_readiness_on_worker_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release = asyncio.Event()

    class Collector:
        def __init__(
            self,
            *,
            streams: tuple[str, ...],
            ingestion_run_id: UUID,
            queue: BoundedCaptureQueue,
            admission: LiveAdmissionState,
            raw_queue: BoundedRawCaptureQueue,
        ) -> None:
            pass

        async def run(
            self,
            *,
            stop: asyncio.Event,
            ready: asyncio.Event,
        ) -> CaptureStats:
            ready.set()
            await release.wait()
            raise RuntimeError("injected collector failure")

        def stats_for_day(self, day: date) -> CaptureStats:
            return CaptureStats(0, 0, 0, 0, 0)

        def discard_stats_before(self, day: date) -> None:
            pass

    monkeypatch.setattr(qualification, "BinanceLiveCollector", Collector)
    statuses: list[qualification.QualificationServiceStatus] = []

    async def on_status(status: qualification.QualificationServiceStatus) -> None:
        statuses.append(status)
        if status.phase is qualification.QualificationServicePhase.READY:
            release.set()

    async def context_for(day: date) -> qualification.QualificationDayContext:
        raise AssertionError("rollover context should not be requested")

    with pytest.raises(RuntimeError, match="injected collector failure"):
        await qualification.run_continuous_qualification_campaign(
            SubscriptionPlan(breadth_symbols=("ETHUSDT",), hot_symbols=()),
            initial_context=qualification.QualificationDayContext(
                metadata=_metadata(),
                clock_offset_ms=Decimal("0"),
            ),
            context_for=context_for,
            output_root=tmp_path,
            stop=asyncio.Event(),
            limits=QualificationLimits(minimum_free_disk_bytes=1),
            on_status=on_status,
        )

    assert statuses[-1] == qualification.QualificationServiceStatus(
        qualification.QualificationServicePhase.FAILED,
        False,
        "supervisor_failed",
    )


def test_streaming_publication_rolls_many_records_at_the_declared_bound(
    tmp_path: Path,
) -> None:
    total = 257
    limits = QualificationLimits(
        queue_capacity=64,
        segment_records=32,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
        recent_identity_capacity=64,
    )
    facts = CampaignFacts(
        day=DAY,
        started_at_ns=RECEIPT_NS - 1,
        ended_at_ns=RECEIPT_NS,
        expected_closed_klines=None,
        queue_capacity=64,
        queue_high_water_mark=32,
        queue_overflows=0,
        collector_stats=(
            CaptureStats(
                total,
                total,
                0,
                0,
                0,
                raw_messages_retained=total,
            ),
        ),
        clock_offset_ms=Decimal("0"),
        recent_identity_capacity=64,
        recent_identity_high_water_mark=64,
        recent_identity_evictions=total - 64,
    )
    captures = (
        decode_combined_message(
            _aggregate_trade().replace(b'"a":100', f'"a":{index}'.encode()),
            context=DecodeContext(RUN_ID, RECEIPT, 0),
        )
        for index in range(total)
    )

    result = publish_qualification_campaign(
        captures,
        metadata=_metadata(),
        facts=facts,
        output_root=tmp_path,
        limits=limits,
    )

    assert len(result.manifest.segments) == 9
    assert sum(segment.records for segment in result.manifest.segments) == total
    assert max(segment.records for segment in result.manifest.segments) == 32
    assert result.report.raw_messages_retained == total
    assert result.report.capture_compressed_bytes < (
        result.report.capture_uncompressed_bytes
    )
    assert result.report.recent_identity_evictions == total - 64


def test_critical_capture_faults_force_readiness_false(tmp_path: Path) -> None:
    stats = CaptureStats(
        messages_received=4,
        messages_enqueued=2,
        messages_refused=2,
        reconnects=1,
        rotations=0,
        conflict_messages=1,
        gap_messages=1,
        schema_mismatches=1,
        queue_overflows=1,
    )
    facts = CampaignFacts(
        day=DAY,
        started_at_ns=RECEIPT_NS - 60_000_000_000,
        ended_at_ns=RECEIPT_NS,
        expected_closed_klines=2,
        queue_capacity=2,
        queue_high_water_mark=2,
        queue_overflows=1,
        collector_stats=(stats,),
        clock_offset_ms=None,
        representative=False,
    )

    result = publish_qualification_campaign(
        _captures(),
        metadata=_metadata(),
        facts=facts,
        output_root=tmp_path,
    )

    assert not result.report.ready
    assert set(result.report.critical_failures) >= {
        "CLOSED_KLINE_COVERAGE_BELOW_99_9_PERCENT",
        "CONFLICTING_EVIDENCE",
        "UNRESOLVED_GAP",
        "SCHEMA_MISMATCH",
        "CAPTURE_QUEUE_OVERFLOW",
        "CLOCK_QUALIFICATION_UNAVAILABLE",
        "REPRESENTATIVE_WINDOW_PENDING",
    }
    assert result.report.queue_overflows == 1
    assert result.report.clock_offset_ms is None
    assert result.report.unresolved_recoveries == 0


def test_clock_adjustment_separates_transport_latency_from_wall_clock_skew(
    tmp_path: Path,
) -> None:
    event_time_ns = 1_795_027_260_000_001_000
    delayed_receipt = LocalReceipt(event_time_ns + 1_100_000_000, 200)
    captured = decode_combined_message(
        _kline(),
        context=DecodeContext(RUN_ID, delayed_receipt, 0),
    )
    facts = CampaignFacts(
        day=datetime.fromtimestamp(
            delayed_receipt.wall_time_ns / 1_000_000_000,
            tz=UTC,
        ).date(),
        started_at_ns=delayed_receipt.wall_time_ns - 1,
        ended_at_ns=delayed_receipt.wall_time_ns,
        expected_closed_klines=1,
        queue_capacity=2,
        queue_high_water_mark=1,
        queue_overflows=0,
        collector_stats=(CaptureStats(1, 1, 0, 0, 0),),
        clock_offset_ms=Decimal("-1000"),
        representative=True,
    )

    result = publish_qualification_campaign(
        (captured,),
        metadata=_metadata(),
        facts=facts,
        output_root=tmp_path,
    )

    assert result.report.raw_event_latency_p95_ns == 2_147_483_648
    assert result.report.event_latency_p95_ns == 134_217_728
    assert "EVENT_LATENCY_P95_EXCEEDS_500_MS" not in result.report.critical_failures
    assert "CLOCK_OFFSET_EXCEEDS_100_MS" in result.report.critical_failures


def test_events_received_before_all_collectors_are_ready_are_startup_only(
    tmp_path: Path,
) -> None:
    capture = _captures()[0]
    facts = CampaignFacts(
        day=DAY,
        started_at_ns=RECEIPT_NS + 1,
        ended_at_ns=RECEIPT_NS + 2,
        expected_closed_klines=1,
        queue_capacity=2,
        queue_high_water_mark=1,
        queue_overflows=0,
        collector_stats=(CaptureStats(1, 1, 0, 0, 0),),
        clock_offset_ms=Decimal("0"),
        collector_startup_ns=500_000_000,
        representative=True,
    )

    result = publish_qualification_campaign(
        (capture,),
        metadata=_metadata(),
        facts=facts,
        output_root=tmp_path,
    )

    assert result.report.persisted_messages == 1
    assert result.report.startup_messages == 1
    assert result.report.closed_klines == 0
    assert result.report.collector_startup_seconds == "0.5"
    assert "CLOSED_KLINE_COVERAGE_BELOW_99_9_PERCENT" in (
        result.report.critical_failures
    )


def test_capture_outside_report_day_is_refused(tmp_path: Path) -> None:
    wrong_day = date.fromordinal(DAY.toordinal() + 1)
    wrong_day_start_ns = int(
        datetime(wrong_day.year, wrong_day.month, wrong_day.day, tzinfo=UTC).timestamp()
        * 1_000_000_000
    )
    facts = CampaignFacts(
        day=wrong_day,
        started_at_ns=wrong_day_start_ns,
        ended_at_ns=wrong_day_start_ns + 1,
        expected_closed_klines=1,
        queue_capacity=1,
        queue_high_water_mark=0,
        queue_overflows=0,
        collector_stats=(),
        clock_offset_ms=Decimal("0"),
        representative=True,
    )

    from pytest import raises

    with raises(Exception, match="outside the report UTC day"):
        publish_qualification_campaign(
            _captures(),
            metadata=_metadata(),
            facts=facts,
            output_root=tmp_path,
        )


def test_campaign_uncompressed_byte_limit_aborts_without_publication(
    tmp_path: Path,
) -> None:
    from pytest import raises

    limits = QualificationLimits(
        queue_capacity=8,
        segment_records=10,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
        maximum_campaign_uncompressed_bytes=1,
        minimum_free_disk_bytes=1,
    )

    with raises(Exception, match="campaign uncompressed-byte limit"):
        publish_qualification_campaign(
            _captures(),
            metadata=_metadata(),
            facts=_facts(),
            output_root=tmp_path,
            limits=limits,
        )

    assert not (tmp_path / "prospective").exists()
    assert not any((tmp_path / ".staging").iterdir())


def test_low_disk_refuses_before_opening_a_capture_segment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pytest import raises

    class DiskUsage:
        free = 1

    monkeypatch.setattr(
        "crypto_boom.qualification.shutil.disk_usage",
        lambda _: DiskUsage(),
    )
    limits = QualificationLimits(
        queue_capacity=8,
        segment_records=10,
        segment_bytes=1024 * 1024,
        maximum_raw_payload_bytes=256 * 1024,
        minimum_free_disk_bytes=1,
    )

    with raises(Exception, match="insufficient free disk"):
        publish_qualification_campaign(
            _captures(),
            metadata=_metadata(),
            facts=_facts(),
            output_root=tmp_path,
            limits=limits,
        )

    assert not (tmp_path / "prospective").exists()
    assert not any((tmp_path / ".staging").iterdir())
