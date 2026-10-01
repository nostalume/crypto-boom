"""Bounded prospective capture persistence and QLT-01 quality reporting."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import types
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence, Set
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from time import time_ns
from typing import (
    Any,
    BinaryIO,
    Final,
    Protocol,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

import aiohttp

from crypto_boom import _artifacts
from crypto_boom.binance_source import (
    MetadataCapture,
    RestWeightBudget,
    fetch_exchange_info,
    measure_exchange_clock,
)
from crypto_boom.live import (
    BinanceLiveCollector,
    BoundedCaptureQueue,
    BoundedRawCaptureQueue,
    CapturedEvidence,
    CaptureStats,
    LiveAdmissionState,
    RawMessageObservation,
    SubscriptionPlan,
)
from crypto_boom.market import (
    AggregateTradeEvent,
    BookTickerEvent,
    KlineEvent,
    MarketEvidence,
)

QUALIFICATION_SCHEMA_VERSION: Final = 3
CAPTURE_RECORD_SCHEMA_VERSION: Final = 1
QUALITY_REPORT_SCHEMA_VERSION: Final = 1
LATENCY_HISTOGRAM_SCHEMA_VERSION: Final = 1
CROSS_DAY_QUALITY_SCHEMA_VERSION: Final = 1
CROSS_DAY_ARTIFACT_SCHEMA_VERSION: Final = 1
DEFAULT_MAXIMUM_CAMPAIGN_UNCOMPRESSED_BYTES: Final = 32 * 1024**3
_MINIMUM_FREE_DISK_BYTES: Final = 1024**3
_MINUTE_NS: Final = 60_000_000_000
_SHA256_PREFIX: Final = "sha256:"
_LATENCY_HISTOGRAM_BUCKETS: Final = 65
_LATENCY_HISTOGRAM_BUCKET_RULE: Final = "bit_length_power_of_two_upper_bound"
_MAX_MANIFEST_BYTES: Final = 1024 * 1024
_MAX_QUALITY_REPORT_BYTES: Final = 16 * 1024 * 1024
_MAX_PUBLICATIONS_PER_DAY: Final = 128


class QualificationError(RuntimeError):
    """Prospective qualification could not be completed safely."""


class QualificationPublicationError(QualificationError):
    """A qualification artifact could not be published atomically."""


class QualificationServicePhase(StrEnum):
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class QualificationServiceStatus:
    phase: QualificationServicePhase
    ready: bool
    detail: str


class DailyCaptureStatsProvider(Protocol):
    """Collector-like owner exposing immutable receive-day statistics."""

    def stats_for_day(self, day: date) -> CaptureStats: ...


class DailyCaptureStatsOwner(DailyCaptureStatsProvider, Protocol):
    """Receive-day statistics that can release successfully published days."""

    def discard_stats_before(self, day: date) -> None: ...


class QualificationEvidenceSink(Protocol):
    """Non-blocking consumer target for retained and admitted live evidence."""

    def append_raw_many(
        self, observations: Sequence[RawMessageObservation]
    ) -> None: ...

    def observe_many(self, captures: Sequence[CapturedEvidence]) -> None: ...


@dataclass(frozen=True, slots=True)
class QualificationLimits:
    """Resource limits for one bounded capture campaign."""

    queue_capacity: int = 8_192
    segment_records: int = 10_000
    segment_bytes: int = 64 * 1024 * 1024
    maximum_raw_payload_bytes: int = 256 * 1024
    maximum_campaign_uncompressed_bytes: int = (
        DEFAULT_MAXIMUM_CAMPAIGN_UNCOMPRESSED_BYTES
    )
    minimum_free_disk_bytes: int = _MINIMUM_FREE_DISK_BYTES
    recent_identity_capacity: int = 100_000

    def __post_init__(self) -> None:
        if (
            min(
                self.queue_capacity,
                self.segment_records,
                self.segment_bytes,
                self.maximum_raw_payload_bytes,
                self.maximum_campaign_uncompressed_bytes,
                self.minimum_free_disk_bytes,
            )
            <= 0
        ):
            raise ValueError("qualification limits must be positive")
        if self.segment_bytes < self.maximum_raw_payload_bytes * 2:
            raise ValueError("segment byte limit is too small for one encoded payload")


@dataclass(frozen=True, slots=True)
class CampaignFacts:
    """Observed campaign facts that cannot be reconstructed from accepted records."""

    day: date
    started_at_ns: int
    ended_at_ns: int
    expected_closed_klines: int | None
    queue_capacity: int
    queue_high_water_mark: int
    queue_overflows: int
    collector_stats: tuple[CaptureStats, ...]
    clock_offset_ms: Decimal | None = None
    unresolved_recoveries: int = 0
    collector_startup_ns: int = 0
    recent_identity_capacity: int = 0
    recent_identity_high_water_mark: int = 0
    recent_identity_evictions: int = 0
    representative: bool = False

    def __post_init__(self) -> None:
        if self.started_at_ns <= 0 or self.ended_at_ns < self.started_at_ns:
            raise ValueError("campaign time range is invalid")
        day_start_ns, next_day_start_ns = _utc_day_bounds(self.day)
        if not (
            day_start_ns <= self.started_at_ns < next_day_start_ns
            and self.started_at_ns <= self.ended_at_ns <= next_day_start_ns
        ):
            raise ValueError("campaign time range falls outside the report UTC day")
        if self.expected_closed_klines is not None and self.expected_closed_klines < 0:
            raise ValueError("expected closed-kline count must be non-negative")
        if self.queue_capacity <= 0:
            raise ValueError("queue capacity must be positive")
        if not 0 <= self.queue_high_water_mark <= self.queue_capacity:
            raise ValueError("queue high-water mark is invalid")
        if (
            self.queue_overflows < 0
            or self.unresolved_recoveries < 0
            or self.collector_startup_ns < 0
            or self.recent_identity_capacity < 0
            or self.recent_identity_high_water_mark < 0
            or self.recent_identity_evictions < 0
            or self.recent_identity_high_water_mark > self.recent_identity_capacity
        ):
            raise ValueError("campaign fault counts must be non-negative")


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    name: str
    sha256: str
    records: int
    bytes: int
    uncompressed_bytes: int
    content_encoding: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "content_encoding": self.content_encoding,
            "name": self.name,
            "records": self.records,
            "sha256": self.sha256,
            "uncompressed_bytes": self.uncompressed_bytes,
        }


@dataclass(frozen=True, slots=True)
class LatencyHistogramRecord:
    """Mergeable sufficient statistics for one latency population."""

    schema_version: int
    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != LATENCY_HISTOGRAM_SCHEMA_VERSION:
            raise ValueError("latency histogram schema version is unsupported")
        if len(self.counts) != _LATENCY_HISTOGRAM_BUCKETS:
            raise ValueError("latency histogram bucket count is invalid")
        if any(count < 0 for count in self.counts):
            raise ValueError("latency histogram counts must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return {
            "bucket_rule": _LATENCY_HISTOGRAM_BUCKET_RULE,
            "counts": list(self.counts),
            "schema_version": self.schema_version,
            "unit": "nanoseconds",
        }

    def percentile(self, numerator: int, denominator: int = 100) -> int | None:
        if not 0 < numerator <= denominator:
            raise ValueError("latency percentile fraction is invalid")
        total = sum(self.counts)
        if total == 0:
            return None
        target = (total * numerator + denominator - 1) // denominator
        seen = 0
        for index, count in enumerate(self.counts):
            seen += count
            if seen >= target:
                return 0 if index == 0 else 1 << index
        raise AssertionError("latency histogram count is inconsistent")


@dataclass(frozen=True, slots=True)
class ProspectiveQualityReport:
    """Reproducible release-gate result for one UTC qualification window."""

    report_id: str
    schema_version: int
    day: str
    day_basis: str
    run_id: str
    window_started_at_ns: int
    window_ended_at_ns: int
    window_complete_utc_day: bool
    input_messages: int
    persisted_messages: int
    refused_messages: int
    provisional_messages: int
    raw_messages_retained: int
    raw_messages_suppressed: int
    capture_uncompressed_bytes: int
    capture_compressed_bytes: int
    capture_compression_ratio: str | None
    startup_messages: int
    closed_klines: int
    aggregate_trades: int
    book_tickers: int
    expected_closed_klines: int | None
    closed_kline_coverage: str | None
    duplicates: int
    conflicts: int
    gaps: int
    schema_mismatches: int
    queue_capacity: int
    queue_high_water_mark: int
    queue_overflows: int
    reconnects: int
    rotations: int
    raw_event_latency_p50_ns: int | None
    raw_event_latency_p95_ns: int | None
    raw_event_latency_p99_ns: int | None
    raw_event_latency_histogram: LatencyHistogramRecord
    event_latency_p50_ns: int | None
    event_latency_p95_ns: int | None
    event_latency_p99_ns: int | None
    event_latency_histogram: LatencyHistogramRecord
    raw_closed_kline_latency_p95_ns: int | None
    raw_closed_kline_latency_histogram: LatencyHistogramRecord
    closed_kline_latency_p95_ns: int | None
    closed_kline_latency_histogram: LatencyHistogramRecord
    collector_startup_seconds: str
    recent_identity_capacity: int
    recent_identity_high_water_mark: int
    recent_identity_evictions: int
    metadata_age_seconds: str
    clock_offset_ms: str | None
    unresolved_recoveries: int
    unsupported_metadata_symbols: tuple[str, ...]
    unknown_symbols: tuple[str, ...]
    critical_failures: tuple[str, ...]
    representative: bool
    ready: bool
    by_symbol: tuple[dict[str, object], ...]

    def to_mapping(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregate_trades": self.aggregate_trades,
            "book_tickers": self.book_tickers,
            "by_symbol": list(self.by_symbol),
            "capture_compressed_bytes": self.capture_compressed_bytes,
            "capture_compression_ratio": self.capture_compression_ratio,
            "capture_uncompressed_bytes": self.capture_uncompressed_bytes,
            "closed_kline_coverage": self.closed_kline_coverage,
            "closed_kline_latency_p95_ns": self.closed_kline_latency_p95_ns,
            "closed_kline_latency_histogram": (
                self.closed_kline_latency_histogram.to_mapping()
            ),
            "raw_closed_kline_latency_p95_ns": (self.raw_closed_kline_latency_p95_ns),
            "raw_closed_kline_latency_histogram": (
                self.raw_closed_kline_latency_histogram.to_mapping()
            ),
            "collector_startup_seconds": self.collector_startup_seconds,
            "recent_identity_capacity": self.recent_identity_capacity,
            "recent_identity_high_water_mark": (self.recent_identity_high_water_mark),
            "recent_identity_evictions": self.recent_identity_evictions,
            "closed_klines": self.closed_klines,
            "conflicts": self.conflicts,
            "critical_failures": list(self.critical_failures),
            "day": self.day,
            "day_basis": self.day_basis,
            "duplicates": self.duplicates,
            "event_latency_p50_ns": self.event_latency_p50_ns,
            "event_latency_p95_ns": self.event_latency_p95_ns,
            "event_latency_p99_ns": self.event_latency_p99_ns,
            "event_latency_histogram": self.event_latency_histogram.to_mapping(),
            "raw_event_latency_p50_ns": self.raw_event_latency_p50_ns,
            "raw_event_latency_p95_ns": self.raw_event_latency_p95_ns,
            "raw_event_latency_p99_ns": self.raw_event_latency_p99_ns,
            "raw_event_latency_histogram": (
                self.raw_event_latency_histogram.to_mapping()
            ),
            "expected_closed_klines": self.expected_closed_klines,
            "gaps": self.gaps,
            "input_messages": self.input_messages,
            "metadata_age_seconds": self.metadata_age_seconds,
            "clock_offset_ms": self.clock_offset_ms,
            "persisted_messages": self.persisted_messages,
            "provisional_messages": self.provisional_messages,
            "raw_messages_retained": self.raw_messages_retained,
            "raw_messages_suppressed": self.raw_messages_suppressed,
            "refused_messages": self.refused_messages,
            "startup_messages": self.startup_messages,
            "queue_capacity": self.queue_capacity,
            "queue_high_water_mark": self.queue_high_water_mark,
            "queue_overflows": self.queue_overflows,
            "ready": self.ready,
            "reconnects": self.reconnects,
            "representative": self.representative,
            "rotations": self.rotations,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "schema_mismatches": self.schema_mismatches,
            "unknown_symbols": list(self.unknown_symbols),
            "unsupported_metadata_symbols": list(self.unsupported_metadata_symbols),
            "unresolved_recoveries": self.unresolved_recoveries,
            "window_complete_utc_day": self.window_complete_utc_day,
            "window_ended_at_ns": self.window_ended_at_ns,
            "window_started_at_ns": self.window_started_at_ns,
        }
        if include_id:
            payload["report_id"] = self.report_id
        return payload


@dataclass(frozen=True, slots=True)
class CrossDayQualityReport:
    """Deterministic aggregation of contiguous daily qualification reports."""

    aggregate_id: str
    schema_version: int
    start_day: str
    end_day: str
    days: int
    report_ids: tuple[str, ...]
    input_messages: int
    persisted_messages: int
    capture_uncompressed_bytes: int
    capture_compressed_bytes: int
    closed_klines: int
    expected_closed_klines: int | None
    closed_kline_coverage: str | None
    duplicates: int
    conflicts: int
    gaps: int
    schema_mismatches: int
    queue_overflows: int
    reconnects: int
    rotations: int
    unresolved_recoveries: int
    raw_event_latency_histogram: LatencyHistogramRecord
    raw_event_latency_p50_ns: int | None
    raw_event_latency_p95_ns: int | None
    raw_event_latency_p99_ns: int | None
    event_latency_histogram: LatencyHistogramRecord
    event_latency_p50_ns: int | None
    event_latency_p95_ns: int | None
    event_latency_p99_ns: int | None
    raw_closed_kline_latency_histogram: LatencyHistogramRecord
    raw_closed_kline_latency_p95_ns: int | None
    closed_kline_latency_histogram: LatencyHistogramRecord
    closed_kline_latency_p95_ns: int | None
    critical_failures: tuple[str, ...]
    representative: bool
    ready: bool

    def to_mapping(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "capture_compressed_bytes": self.capture_compressed_bytes,
            "capture_uncompressed_bytes": self.capture_uncompressed_bytes,
            "closed_kline_coverage": self.closed_kline_coverage,
            "closed_kline_latency_histogram": (
                self.closed_kline_latency_histogram.to_mapping()
            ),
            "closed_kline_latency_p95_ns": self.closed_kline_latency_p95_ns,
            "closed_klines": self.closed_klines,
            "conflicts": self.conflicts,
            "critical_failures": list(self.critical_failures),
            "days": self.days,
            "duplicates": self.duplicates,
            "end_day": self.end_day,
            "event_latency_histogram": self.event_latency_histogram.to_mapping(),
            "event_latency_p50_ns": self.event_latency_p50_ns,
            "event_latency_p95_ns": self.event_latency_p95_ns,
            "event_latency_p99_ns": self.event_latency_p99_ns,
            "expected_closed_klines": self.expected_closed_klines,
            "gaps": self.gaps,
            "input_messages": self.input_messages,
            "persisted_messages": self.persisted_messages,
            "queue_overflows": self.queue_overflows,
            "raw_closed_kline_latency_histogram": (
                self.raw_closed_kline_latency_histogram.to_mapping()
            ),
            "raw_closed_kline_latency_p95_ns": (self.raw_closed_kline_latency_p95_ns),
            "raw_event_latency_histogram": (
                self.raw_event_latency_histogram.to_mapping()
            ),
            "raw_event_latency_p50_ns": self.raw_event_latency_p50_ns,
            "raw_event_latency_p95_ns": self.raw_event_latency_p95_ns,
            "raw_event_latency_p99_ns": self.raw_event_latency_p99_ns,
            "ready": self.ready,
            "reconnects": self.reconnects,
            "report_ids": list(self.report_ids),
            "representative": self.representative,
            "rotations": self.rotations,
            "schema_mismatches": self.schema_mismatches,
            "schema_version": self.schema_version,
            "start_day": self.start_day,
            "unresolved_recoveries": self.unresolved_recoveries,
        }
        if include_id:
            payload["aggregate_id"] = self.aggregate_id
        return payload


@dataclass(frozen=True, slots=True)
class CrossDaySourceRecord:
    day: str
    run_id: str
    manifest_id: str
    report_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "day": self.day,
            "manifest_id": self.manifest_id,
            "report_id": self.report_id,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class CrossDayQualityManifest:
    schema_version: int
    manifest_id: str
    aggregate_report_schema_version: int
    aggregate_report_sha256: str
    aggregate_id: str
    start_day: str
    end_day: str
    sources: tuple[CrossDaySourceRecord, ...]

    def to_mapping(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregate_id": self.aggregate_id,
            "aggregate_report_schema_version": (self.aggregate_report_schema_version),
            "aggregate_report_sha256": self.aggregate_report_sha256,
            "end_day": self.end_day,
            "schema_version": self.schema_version,
            "sources": [source.to_mapping() for source in self.sources],
            "start_day": self.start_day,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


@dataclass(frozen=True, slots=True)
class QualificationManifest:
    schema_version: int
    manifest_id: str
    run_id: str
    day: str
    capture_record_schema_version: int
    quality_report_schema_version: int
    metadata_sha256: str
    quality_report_sha256: str
    report_id: str
    segments: tuple[SegmentRecord, ...]

    def to_mapping(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "capture_record_schema_version": self.capture_record_schema_version,
            "day": self.day,
            "metadata_sha256": self.metadata_sha256,
            "quality_report_sha256": self.quality_report_sha256,
            "report_id": self.report_id,
            "run_id": self.run_id,
            "quality_report_schema_version": self.quality_report_schema_version,
            "schema_version": self.schema_version,
            "segments": [segment.to_mapping() for segment in self.segments],
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


@dataclass(frozen=True, slots=True)
class PublishedQualification:
    path: Path
    manifest: QualificationManifest
    report: ProspectiveQualityReport
    already_present: bool


@dataclass(frozen=True, slots=True)
class PublishedCrossDayQuality:
    path: Path
    manifest: CrossDayQualityManifest
    report: CrossDayQualityReport
    already_present: bool


class RollingCrossDayQualityPublisher:
    """Publish one bounded rolling aggregate after each complete UTC day."""

    def __init__(self, output_root: Path, *, window_days: int = 7) -> None:
        if window_days <= 0:
            raise ValueError("rolling aggregate window must be positive")
        self.output_root = output_root
        self.window_days = window_days
        self._daily: deque[PublishedQualification] = deque(maxlen=window_days)

    def hydrate(self, publications: Iterable[PublishedQualification]) -> int:
        """Restore only the bounded prefix needed before the next daily result."""

        if self._daily:
            raise QualificationError("rolling aggregate publisher is already hydrated")
        ordered = sorted(publications, key=lambda item: item.report.day)
        history_size = max(0, self.window_days - 1)
        retained = ordered[-history_size:] if history_size else []
        for previous, current in pairwise(retained):
            previous_day = date.fromisoformat(previous.report.day)
            current_day = date.fromisoformat(current.report.day)
            if current_day != previous_day + timedelta(days=1):
                raise QualificationError(
                    "rolling aggregate hydration requires contiguous complete UTC days"
                )
        for publication in retained:
            if not publication.report.window_complete_utc_day:
                raise QualificationError(
                    "rolling aggregate hydration requires complete UTC days"
                )
            _verify_daily_publication(publication)
            self._daily.append(publication)
        return len(retained)

    def hydrate_from_disk(self) -> int:
        """Restore the latest verified contiguous suffix from durable artifacts."""

        publications = discover_recent_complete_qualifications(
            self.output_root,
            limit=max(0, self.window_days - 1),
        )
        return self.hydrate(publications)

    def observe(
        self, publication: PublishedQualification
    ) -> PublishedCrossDayQuality | None:
        if not publication.report.window_complete_utc_day:
            return None
        _verify_daily_publication(publication)
        current_day = date.fromisoformat(publication.report.day)
        if self._daily:
            previous_day = date.fromisoformat(self._daily[-1].report.day)
            if current_day != previous_day + timedelta(days=1):
                raise QualificationError(
                    "rolling aggregate requires contiguous complete UTC days"
                )
        self._daily.append(publication)
        if len(self._daily) < self.window_days:
            return None
        return publish_cross_day_quality_report(
            tuple(self._daily),
            output_root=self.output_root,
            minimum_complete_days=self.window_days,
        )


class _LatencyHistogram:
    """Fixed-memory, deterministic power-of-two latency histogram."""

    def __init__(self) -> None:
        self._counts = [0] * _LATENCY_HISTOGRAM_BUCKETS
        self._total = 0

    def observe(self, value: int) -> None:
        if value < 0:
            return
        bucket = min(value.bit_length(), 64)
        self._counts[bucket] += 1
        self._total += 1

    def percentile(self, numerator: int, denominator: int = 100) -> int | None:
        if self._total == 0:
            return None
        target = (self._total * numerator + denominator - 1) // denominator
        seen = 0
        for index, count in enumerate(self._counts):
            seen += count
            if seen >= target:
                return 0 if index == 0 else 1 << index
        raise AssertionError("latency histogram count is inconsistent")

    def snapshot(self) -> LatencyHistogramRecord:
        return LatencyHistogramRecord(
            schema_version=LATENCY_HISTOGRAM_SCHEMA_VERSION,
            counts=tuple(self._counts),
        )


class _QualityAccumulator:
    def __init__(
        self,
        metadata: MetadataCapture,
        *,
        clock_offset_ms: Decimal | None,
        window_start_ns: int | None,
    ) -> None:
        self.run_id = metadata.events[0].provenance.ingestion_run_id
        self.metadata_by_symbol = {
            event.instrument.symbol: event for event in metadata.events
        }
        self.persisted = 0
        self.closed_klines = 0
        self.aggregate_trades = 0
        self.book_tickers = 0
        self.clock_offset_ns = (
            None
            if clock_offset_ms is None
            else int(clock_offset_ms * Decimal(1_000_000))
        )
        self.window_start_ns = window_start_ns
        self.startup_messages = 0
        self.negative_latency = 0
        self.unknown_symbols: set[str] = set()
        self.by_symbol: dict[str, dict[str, int]] = {}
        self.last_kline_open_us: dict[str, int] = {}
        self.observed_kline_gaps = 0
        self.raw_latency = _LatencyHistogram()
        self.latency = _LatencyHistogram()
        self.raw_kline_latency = _LatencyHistogram()
        self.kline_latency = _LatencyHistogram()

    def begin_window(self, started_at_ns: int) -> None:
        if started_at_ns <= 0:
            raise ValueError("qualification window start must be positive")
        self.window_start_ns = started_at_ns

    def observe(self, captured: CapturedEvidence) -> None:
        event = captured.event
        provenance = event.provenance
        if provenance.ingestion_run_id != self.run_id:
            raise QualificationError("capture and metadata run identities disagree")
        if event.instrument.symbol not in self.metadata_by_symbol:
            self.unknown_symbols.add(event.instrument.symbol)
        if (
            self.window_start_ns is None
            or provenance.receipt.wall_time_ns < self.window_start_ns
        ):
            self.startup_messages += 1
            return

        counts = self.by_symbol.setdefault(
            event.instrument.symbol,
            {"aggregate_trades": 0, "book_tickers": 0, "closed_klines": 0},
        )
        if isinstance(event, KlineEvent):
            self.closed_klines += 1
            counts["closed_klines"] += 1
            previous = self.last_kline_open_us.get(event.instrument.symbol)
            if (
                previous is not None
                and event.open_time.epoch_microseconds != previous + 60_000_000
            ):
                self.observed_kline_gaps += 1
            self.last_kline_open_us[event.instrument.symbol] = (
                event.open_time.epoch_microseconds
            )
        elif isinstance(event, AggregateTradeEvent):
            self.aggregate_trades += 1
            counts["aggregate_trades"] += 1
        elif isinstance(event, BookTickerEvent):
            self.book_tickers += 1
            counts["book_tickers"] += 1

        source_time = _source_time_us(event)
        if source_time is not None:
            raw_latency = provenance.receipt.wall_time_ns - source_time * 1_000
            if raw_latency >= 0:
                self.raw_latency.observe(raw_latency)
                if isinstance(event, KlineEvent):
                    self.raw_kline_latency.observe(raw_latency)
            if self.clock_offset_ns is not None:
                latency = raw_latency + self.clock_offset_ns
                if latency < 0:
                    self.negative_latency += 1
                else:
                    self.latency.observe(latency)
                    if isinstance(event, KlineEvent):
                        self.kline_latency.observe(latency)


class _QualificationStager:
    def __init__(
        self,
        *,
        metadata: MetadataCapture,
        output_root: Path,
        limits: QualificationLimits,
        clock_offset_ms: Decimal | None,
        window_start_ns: int | None,
    ) -> None:
        self.metadata = metadata
        self.output_root = output_root.resolve()
        self.limits = limits
        self.accumulator = _QualityAccumulator(
            metadata,
            clock_offset_ms=clock_offset_ms,
            window_start_ns=window_start_ns,
        )
        self._segments: list[SegmentRecord] = []
        self._stream: gzip.GzipFile | None = None
        self._raw_stream: BinaryIO | None = None
        self._segment_name = ""
        self._segment_records = 0
        self._segment_bytes = 0
        self._total_uncompressed_bytes = 0
        self._closed = False
        try:
            staging_root = self.output_root / ".staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            self.staging = Path(tempfile.mkdtemp(prefix="qlt-", dir=staging_root))
            _write_bytes(self.staging / "exchange-info.json", metadata.raw_payload)
        except OSError as error:
            raise QualificationPublicationError(
                "qualification staging root is unavailable"
            ) from error

    def append_many(self, captures: Sequence[CapturedEvidence]) -> None:
        for captured in captures:
            self.append(captured)

    def append(self, captured: CapturedEvidence) -> None:
        provenance = captured.event.provenance
        generation = provenance.connection_generation
        if generation is None:
            raise QualificationError("live capture has no connection generation")
        self.append_raw(
            RawMessageObservation(
                raw_payload=captured.raw_payload,
                ingestion_run_id=provenance.ingestion_run_id,
                receipt=provenance.receipt,
                connection_generation=generation,
                retention_reason="VALID",
            )
        )
        self.accumulator.observe(captured)

    def begin_window(self, started_at_ns: int) -> None:
        self.accumulator.begin_window(started_at_ns)

    def append_raw_many(self, observations: Sequence[RawMessageObservation]) -> None:
        for observed in observations:
            self.append_raw(observed)

    def observe_many(self, captures: Sequence[CapturedEvidence]) -> None:
        for captured in captures:
            self.accumulator.observe(captured)

    def append_raw(self, observed: RawMessageObservation) -> None:
        if self._closed:
            raise QualificationPublicationError("qualification stager is closed")
        if observed.ingestion_run_id != self.accumulator.run_id:
            raise QualificationError("raw capture and metadata run identities disagree")
        if len(observed.raw_payload) > self.limits.maximum_raw_payload_bytes:
            raise QualificationError("captured payload exceeds the qualification limit")
        line = _capture_line(observed)
        if len(line) > self.limits.segment_bytes:
            raise QualificationError("encoded capture exceeds the segment byte limit")
        if (
            self._total_uncompressed_bytes + len(line)
            > self.limits.maximum_campaign_uncompressed_bytes
        ):
            raise QualificationError(
                "capture exceeds the campaign uncompressed-byte limit"
            )
        if self._must_roll(len(line)):
            self._close_segment()
        if self._stream is None:
            self._open_segment()
        assert self._stream is not None
        try:
            self._stream.write(line)
        except OSError as error:
            raise QualificationPublicationError(
                "capture segment write failed"
            ) from error
        self._segment_records += 1
        self._segment_bytes += len(line)
        self._total_uncompressed_bytes += len(line)
        self.accumulator.persisted += 1

    def finalize(self, facts: CampaignFacts) -> PublishedQualification:
        if self._closed:
            raise QualificationPublicationError("qualification stager is closed")
        try:
            self._close_segment()
            report = _quality_report(
                self.metadata,
                self.accumulator,
                facts,
                segments=tuple(self._segments),
            )
            report_bytes = _artifacts.canonical_json(report.to_mapping())
            report_path = self.staging / "quality-report.json"
            _write_bytes(report_path, report_bytes)
            metadata_path = self.staging / "exchange-info.json"
            unsigned_manifest = QualificationManifest(
                schema_version=QUALIFICATION_SCHEMA_VERSION,
                manifest_id="",
                run_id=str(self.accumulator.run_id),
                day=facts.day.isoformat(),
                capture_record_schema_version=CAPTURE_RECORD_SCHEMA_VERSION,
                quality_report_schema_version=QUALITY_REPORT_SCHEMA_VERSION,
                metadata_sha256=_file_id(metadata_path),
                quality_report_sha256=_file_id(report_path),
                report_id=report.report_id,
                segments=tuple(self._segments),
            )
            manifest_id = _artifacts.content_id(
                unsigned_manifest.to_mapping(include_id=False)
            )
            manifest = replace(unsigned_manifest, manifest_id=manifest_id)
            manifest_bytes = _artifacts.canonical_json(manifest.to_mapping())
            _write_bytes(self.staging / "manifest.json", manifest_bytes)
            target = (
                self.output_root
                / "prospective"
                / facts.day.isoformat()
                / str(self.accumulator.run_id)
                / manifest_id.removeprefix(_SHA256_PREFIX)
            )
            already_present = _publish_directory(
                self.staging,
                target,
                manifest_bytes=manifest_bytes,
                manifest=manifest,
            )
            self._closed = True
            return PublishedQualification(target, manifest, report, already_present)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        if self._raw_stream is not None:
            try:
                self._raw_stream.close()
            except OSError:
                pass
            self._raw_stream = None
        if not self._closed:
            shutil.rmtree(self.staging, ignore_errors=True)
            self._closed = True

    def _must_roll(self, next_bytes: int) -> bool:
        return self._stream is not None and (
            self._segment_records >= self.limits.segment_records
            or self._segment_bytes + next_bytes > self.limits.segment_bytes
        )

    def _open_segment(self) -> None:
        required_free = self.limits.minimum_free_disk_bytes + self.limits.segment_bytes
        try:
            free_disk = shutil.disk_usage(self.staging).free
        except OSError as error:
            raise QualificationPublicationError(
                "capture free-disk check failed"
            ) from error
        if free_disk < required_free:
            raise QualificationPublicationError(
                "insufficient free disk for another capture segment"
            )
        self._segment_name = f"capture-{len(self._segments):06d}.jsonl.gz"
        try:
            raw_stream = (self.staging / self._segment_name).open("xb")
            try:
                stream = gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=6,
                    fileobj=raw_stream,
                    mtime=0,
                )
            except BaseException:
                raw_stream.close()
                raise
        except OSError as error:
            raise QualificationPublicationError(
                "capture segment open failed"
            ) from error
        self._raw_stream = raw_stream
        self._stream = stream
        self._segment_records = 0
        self._segment_bytes = 0

    def _close_segment(self) -> None:
        if self._stream is None:
            return
        raw_stream = self._raw_stream
        assert raw_stream is not None
        try:
            self._stream.close()
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
            raw_stream.close()
        except OSError as error:
            raise QualificationPublicationError(
                "capture segment flush failed"
            ) from error
        finally:
            if not raw_stream.closed:
                try:
                    raw_stream.close()
                except OSError:
                    pass
            self._stream = None
            self._raw_stream = None
        path = self.staging / self._segment_name
        try:
            compressed_bytes = path.stat().st_size
        except OSError as error:
            raise QualificationPublicationError(
                "capture segment is unavailable after flush"
            ) from error
        self._segments.append(
            SegmentRecord(
                name=self._segment_name,
                sha256=_file_id(path),
                records=self._segment_records,
                bytes=compressed_bytes,
                uncompressed_bytes=self._segment_bytes,
                content_encoding="gzip",
            )
        )


@dataclass(frozen=True, slots=True)
class QualificationDayContext:
    metadata: MetadataCapture
    clock_offset_ms: Decimal | None


@dataclass(frozen=True, slots=True)
class BinanceQualificationContextProvider:
    """Refresh one next-day qualification context through the shared REST budget."""

    session: aiohttp.ClientSession
    ingestion_run_id: UUID
    budget: RestWeightBudget

    async def __call__(self, day: date) -> QualificationDayContext:
        metadata = await fetch_exchange_info(
            self.session,
            ingestion_run_id=self.ingestion_run_id,
            budget=self.budget,
        )
        receipt_day = _utc_day(metadata.events[0].provenance.receipt.wall_time_ns)
        if receipt_day + timedelta(days=1) != day:
            raise QualificationError(
                "next-day context was not sampled on the preceding UTC day"
            )
        clock = await measure_exchange_clock(self.session, budget=self.budget)
        return QualificationDayContext(metadata, clock.offset_ms)


class RotatingQualificationSink:
    """Own bounded per-receipt-day stagers without owning live connections."""

    def __init__(
        self,
        *,
        metadata: MetadataCapture,
        output_root: Path,
        limits: QualificationLimits,
        clock_offset_ms: Decimal | None,
        maximum_open_days: int = 2,
    ) -> None:
        if maximum_open_days <= 0:
            raise ValueError("maximum open qualification days must be positive")
        self.output_root = output_root
        self.limits = limits
        self.default_context = QualificationDayContext(metadata, clock_offset_ms)
        self.maximum_open_days = maximum_open_days
        self._stagers: dict[date, _QualificationStager] = {}
        self._finalized_through: date | None = None
        self._aborted = False

    def open_day(
        self,
        day: date,
        *,
        started_at_ns: int | None = None,
        context: QualificationDayContext | None = None,
    ) -> None:
        if self._aborted:
            raise QualificationPublicationError("qualification sink is aborted")
        if self._finalized_through is not None and day <= self._finalized_through:
            raise QualificationPublicationError(
                "qualification receipt day is already finalized"
            )
        if day in self._stagers:
            raise QualificationPublicationError("qualification receipt day is open")
        if self._stagers and day != max(self._stagers) + timedelta(days=1):
            raise QualificationError("qualification receipt days must be contiguous")
        if len(self._stagers) >= self.maximum_open_days:
            raise QualificationPublicationError(
                "open qualification-day capacity is exhausted"
            )
        if started_at_ns is not None:
            day_start_ns, next_day_start_ns = _utc_day_bounds(day)
            if not day_start_ns <= started_at_ns < next_day_start_ns:
                raise ValueError("qualification day start falls outside its UTC day")
        selected = context or self.default_context
        self._stagers[day] = _QualificationStager(
            metadata=selected.metadata,
            output_root=self.output_root,
            limits=self.limits,
            clock_offset_ms=selected.clock_offset_ms,
            window_start_ns=started_at_ns,
        )

    def begin_day(self, day: date, started_at_ns: int) -> None:
        self._require_open(day).begin_window(started_at_ns)

    def append(self, captured: CapturedEvidence) -> None:
        day = _utc_day(captured.event.provenance.receipt.wall_time_ns)
        self._require_open(day).append(captured)

    def append_raw(self, observed: RawMessageObservation) -> None:
        day = _utc_day(observed.receipt.wall_time_ns)
        self._require_open(day).append_raw(observed)

    def append_raw_many(self, observations: Sequence[RawMessageObservation]) -> None:
        for observed in observations:
            self.append_raw(observed)

    def observe(self, captured: CapturedEvidence) -> None:
        day = _utc_day(captured.event.provenance.receipt.wall_time_ns)
        self._require_open(day).accumulator.observe(captured)

    def observe_many(self, captures: Sequence[CapturedEvidence]) -> None:
        for captured in captures:
            self.observe(captured)

    def finalize_day(self, facts: CampaignFacts) -> PublishedQualification:
        if not self._stagers or facts.day != min(self._stagers):
            raise QualificationPublicationError(
                "qualification days must finalize in UTC order"
            )
        stager = self._require_open(facts.day)
        try:
            published = stager.finalize(facts)
        except BaseException:
            self.abort()
            raise
        del self._stagers[facts.day]
        self._finalized_through = facts.day
        return published

    def abort_latest_day(self, day: date) -> None:
        if not self._stagers or day != max(self._stagers):
            raise QualificationPublicationError(
                "only the latest open qualification day can be aborted"
            )
        self._stagers.pop(day).abort()

    def abort(self) -> None:
        for stager in self._stagers.values():
            stager.abort()
        self._stagers.clear()
        self._aborted = True

    def _require_open(self, day: date) -> _QualificationStager:
        if self._finalized_through is not None and day <= self._finalized_through:
            raise QualificationPublicationError(
                "qualification receipt day is already finalized"
            )
        try:
            return self._stagers[day]
        except KeyError as error:
            raise QualificationPublicationError(
                "qualification receipt day is not open"
            ) from error


class QualificationRolloverController:
    """Coordinate daily evidence owners without owning collector socket lifecycle."""

    def __init__(
        self,
        *,
        metadata: MetadataCapture,
        output_root: Path,
        limits: QualificationLimits,
        clock_offset_ms: Decimal | None,
        collectors: Sequence[DailyCaptureStatsOwner],
        queue: BoundedCaptureQueue,
        raw_queue: BoundedRawCaptureQueue,
        admission: LiveAdmissionState,
        breadth_symbol_count: int,
    ) -> None:
        if not collectors or breadth_symbol_count <= 0:
            raise ValueError("rollover requires collectors and breadth symbols")
        self.sink = RotatingQualificationSink(
            metadata=metadata,
            output_root=output_root,
            limits=limits,
            clock_offset_ms=clock_offset_ms,
        )
        self.collectors = tuple(collectors)
        self.queue = queue
        self.raw_queue = raw_queue
        self.admission = admission
        self.breadth_symbol_count = breadth_symbol_count
        self._default_context = QualificationDayContext(metadata, clock_offset_ms)
        self._started_at: dict[date, int] = {}
        self._contexts: dict[date, QualificationDayContext] = {}
        self._startup_ns: dict[date, int] = {}

    @property
    def open_days(self) -> tuple[date, ...]:
        return tuple(sorted(self._started_at))

    def open_initial_day(
        self,
        day: date,
        *,
        started_at_ns: int,
        collector_startup_ns: int = 0,
    ) -> None:
        if self._started_at:
            raise QualificationPublicationError("initial qualification day is open")
        self.sink.open_day(day, started_at_ns=started_at_ns)
        self._started_at[day] = started_at_ns
        self._contexts[day] = self._default_context
        self._startup_ns[day] = collector_startup_ns

    def prepare_next_day(
        self,
        boundary_ns: int,
        *,
        context: QualificationDayContext,
    ) -> date:
        if not self._started_at:
            raise QualificationPublicationError("initial qualification day is not open")
        previous = max(self._started_at)
        _, expected_boundary_ns = _utc_day_bounds(previous)
        if boundary_ns != expected_boundary_ns:
            raise QualificationError("rollover boundary is not the next UTC midnight")
        next_day = previous + timedelta(days=1)
        self.sink.open_day(
            next_day,
            started_at_ns=boundary_ns,
            context=context,
        )
        self._started_at[next_day] = boundary_ns
        self._contexts[next_day] = context
        self._startup_ns[next_day] = 0
        return next_day

    def set_collector_startup_ns(self, day: date, startup_ns: int) -> None:
        if startup_ns < 0:
            raise ValueError("collector startup duration must be non-negative")
        if day not in self._startup_ns:
            raise QualificationPublicationError("qualification day is not open")
        self._startup_ns[day] = startup_ns

    def abort_future_day(self, at_ns: int) -> date | None:
        if at_ns <= 0:
            raise ValueError("shutdown time must be positive")
        if not self._started_at:
            return None
        latest = max(self._started_at)
        if self._started_at[latest] < at_ns:
            return None
        if latest == min(self._started_at):
            return None
        self.sink.abort_latest_day(latest)
        del self._started_at[latest]
        del self._contexts[latest]
        del self._startup_ns[latest]
        return latest

    def append_raw(self, observed: RawMessageObservation) -> None:
        self.sink.append_raw(observed)

    def append_raw_many(self, observations: Sequence[RawMessageObservation]) -> None:
        self.sink.append_raw_many(observations)

    def observe(self, captured: CapturedEvidence) -> None:
        self.sink.observe(captured)

    def observe_many(self, captures: Sequence[CapturedEvidence]) -> None:
        self.sink.observe_many(captures)

    def finalize_oldest_day(
        self,
        *,
        ended_at_ns: int,
        unresolved_recoveries: int = 0,
    ) -> PublishedQualification:
        if not self._started_at:
            raise QualificationPublicationError(
                "qualification rollover has no open day"
            )
        day = min(self._started_at)
        started_at_ns = self._started_at[day]
        expected = _expected_closed_klines(
            started_at_ns,
            ended_at_ns,
            self.breadth_symbol_count,
        )
        facts = campaign_facts_for_day(
            day,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            expected_closed_klines=expected,
            collectors=self.collectors,
            queue=self.queue,
            raw_queue=self.raw_queue,
            admission=self.admission,
            clock_offset_ms=self._contexts[day].clock_offset_ms,
            unresolved_recoveries=unresolved_recoveries,
            collector_startup_ns=self._startup_ns[day],
            representative=False,
        )
        published = self.sink.finalize_day(facts)
        retain_from = day + timedelta(days=1)
        for collector in self.collectors:
            collector.discard_stats_before(retain_from)
        self.queue.discard_stats_before(retain_from)
        self.raw_queue.discard_stats_before(retain_from)
        self.admission.discard_stats_before(retain_from)
        del self._started_at[day]
        del self._contexts[day]
        del self._startup_ns[day]
        return published

    def abort(self) -> None:
        self.sink.abort()


def campaign_facts_for_day(
    day: date,
    *,
    started_at_ns: int,
    ended_at_ns: int,
    expected_closed_klines: int | None,
    collectors: Sequence[DailyCaptureStatsProvider],
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    admission: LiveAdmissionState,
    clock_offset_ms: Decimal | None,
    unresolved_recoveries: int = 0,
    collector_startup_ns: int = 0,
    representative: bool = False,
) -> CampaignFacts:
    """Snapshot one receipt UTC day without stopping any live owner."""

    if not collectors:
        raise ValueError("daily campaign facts require collectors")
    if queue.capacity != raw_queue.capacity:
        raise QualificationError("capture queue capacities disagree")
    queue_stats = queue.stats_for_day(day)
    raw_queue_stats = raw_queue.stats_for_day(day)
    admission_stats = admission.stats_for_day(day)
    return CampaignFacts(
        day=day,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        expected_closed_klines=expected_closed_klines,
        queue_capacity=queue.capacity,
        queue_high_water_mark=max(
            queue_stats.high_water_mark,
            raw_queue_stats.high_water_mark,
        ),
        queue_overflows=queue_stats.overflows + raw_queue_stats.overflows,
        collector_stats=tuple(collector.stats_for_day(day) for collector in collectors),
        clock_offset_ms=clock_offset_ms,
        unresolved_recoveries=unresolved_recoveries,
        collector_startup_ns=collector_startup_ns,
        recent_identity_capacity=admission_stats.recent_identity_capacity,
        recent_identity_high_water_mark=(
            admission_stats.recent_identity_high_water_mark
        ),
        recent_identity_evictions=admission_stats.recent_identity_evictions,
        representative=representative,
    )


def aggregate_quality_reports(
    reports: Iterable[ProspectiveQualityReport],
    *,
    minimum_complete_days: int = 7,
) -> CrossDayQualityReport:
    """Merge compatible contiguous UTC-day reports without averaging percentiles."""

    if minimum_complete_days <= 0:
        raise ValueError("minimum complete days must be positive")
    admitted = tuple(reports)
    if not admitted:
        raise QualificationError("cross-day aggregation requires daily reports")
    try:
        ordered = tuple(
            sorted(admitted, key=lambda report: date.fromisoformat(report.day))
        )
        days = tuple(date.fromisoformat(report.day) for report in ordered)
    except ValueError as error:
        raise QualificationError("quality report day is invalid") from error
    if any(
        report.schema_version != QUALITY_REPORT_SCHEMA_VERSION for report in ordered
    ):
        raise QualificationError("quality report schema versions are incompatible")
    if any(report.day_basis != "receipt_wall_time_utc" for report in ordered):
        raise QualificationError("quality report day bases are incompatible")
    if any(
        current != previous + timedelta(days=1) for previous, current in pairwise(days)
    ):
        raise QualificationError("cross-day aggregation requires contiguous UTC days")
    for report, report_day in zip(ordered, days, strict=True):
        day_start_ns, next_day_start_ns = _utc_day_bounds(report_day)
        if not (
            day_start_ns <= report.window_started_at_ns < next_day_start_ns
            and report.window_started_at_ns
            <= report.window_ended_at_ns
            <= next_day_start_ns
        ):
            raise QualificationError(
                "quality report window falls outside its receipt UTC day"
            )
        exact_day = (
            report.window_started_at_ns == day_start_ns
            and report.window_ended_at_ns == next_day_start_ns
        )
        if report.window_complete_utc_day != exact_day:
            raise QualificationError("quality report full-day claim is inconsistent")

    raw_event_histogram = _merge_latency_histograms(
        report.raw_event_latency_histogram for report in ordered
    )
    event_histogram = _merge_latency_histograms(
        report.event_latency_histogram for report in ordered
    )
    raw_kline_histogram = _merge_latency_histograms(
        report.raw_closed_kline_latency_histogram for report in ordered
    )
    kline_histogram = _merge_latency_histograms(
        report.closed_kline_latency_histogram for report in ordered
    )
    expected_values = tuple(report.expected_closed_klines for report in ordered)
    expected = (
        None
        if any(value is None for value in expected_values)
        else sum(value for value in expected_values if value is not None)
    )
    closed_klines = sum(report.closed_klines for report in ordered)
    coverage = (
        None
        if expected is None or expected == 0
        else str(Decimal(closed_klines) / Decimal(expected))
    )
    complete_days = all(report.window_complete_utc_day for report in ordered)
    representative = complete_days and len(ordered) >= minimum_complete_days
    failures = {
        failure
        for report in ordered
        for failure in report.critical_failures
        if failure != "REPRESENTATIVE_WINDOW_PENDING"
    }
    if not complete_days:
        failures.add("INCOMPLETE_UTC_DAY_WINDOW")
    if not representative:
        failures.add("REPRESENTATIVE_WINDOW_PENDING")
    ordered_failures = tuple(sorted(failures))
    unsigned = CrossDayQualityReport(
        aggregate_id="",
        schema_version=CROSS_DAY_QUALITY_SCHEMA_VERSION,
        start_day=days[0].isoformat(),
        end_day=days[-1].isoformat(),
        days=len(ordered),
        report_ids=tuple(report.report_id for report in ordered),
        input_messages=sum(report.input_messages for report in ordered),
        persisted_messages=sum(report.persisted_messages for report in ordered),
        capture_uncompressed_bytes=sum(
            report.capture_uncompressed_bytes for report in ordered
        ),
        capture_compressed_bytes=sum(
            report.capture_compressed_bytes for report in ordered
        ),
        closed_klines=closed_klines,
        expected_closed_klines=expected,
        closed_kline_coverage=coverage,
        duplicates=sum(report.duplicates for report in ordered),
        conflicts=sum(report.conflicts for report in ordered),
        gaps=sum(report.gaps for report in ordered),
        schema_mismatches=sum(report.schema_mismatches for report in ordered),
        queue_overflows=sum(report.queue_overflows for report in ordered),
        reconnects=sum(report.reconnects for report in ordered),
        rotations=sum(report.rotations for report in ordered),
        unresolved_recoveries=sum(report.unresolved_recoveries for report in ordered),
        raw_event_latency_histogram=raw_event_histogram,
        raw_event_latency_p50_ns=raw_event_histogram.percentile(50),
        raw_event_latency_p95_ns=raw_event_histogram.percentile(95),
        raw_event_latency_p99_ns=raw_event_histogram.percentile(99),
        event_latency_histogram=event_histogram,
        event_latency_p50_ns=event_histogram.percentile(50),
        event_latency_p95_ns=event_histogram.percentile(95),
        event_latency_p99_ns=event_histogram.percentile(99),
        raw_closed_kline_latency_histogram=raw_kline_histogram,
        raw_closed_kline_latency_p95_ns=raw_kline_histogram.percentile(95),
        closed_kline_latency_histogram=kline_histogram,
        closed_kline_latency_p95_ns=kline_histogram.percentile(95),
        critical_failures=ordered_failures,
        representative=representative,
        ready=representative and not ordered_failures,
    )
    return replace(
        unsigned,
        aggregate_id=_artifacts.content_id(unsigned.to_mapping(include_id=False)),
    )


def publish_cross_day_quality_report(
    publications: Iterable[PublishedQualification],
    *,
    output_root: Path,
    minimum_complete_days: int = 7,
) -> PublishedCrossDayQuality:
    """Verify daily artifacts and atomically publish their cross-day quality report."""

    admitted = tuple(publications)
    if not admitted:
        raise QualificationError("cross-day publication requires daily artifacts")
    ordered = tuple(sorted(admitted, key=lambda item: item.report.day))
    for publication in ordered:
        _verify_daily_publication(publication)
    report = aggregate_quality_reports(
        (publication.report for publication in ordered),
        minimum_complete_days=minimum_complete_days,
    )
    sources = tuple(
        CrossDaySourceRecord(
            day=publication.report.day,
            run_id=publication.report.run_id,
            manifest_id=publication.manifest.manifest_id,
            report_id=publication.report.report_id,
        )
        for publication in ordered
    )
    resolved_root = output_root.resolve()
    staging: Path | None = None
    try:
        staging_root = resolved_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="qlt-aggregate-", dir=staging_root))
        report_bytes = _artifacts.canonical_json(report.to_mapping())
        report_path = staging / "aggregate-quality-report.json"
        _write_bytes(report_path, report_bytes)
        unsigned_manifest = CrossDayQualityManifest(
            schema_version=CROSS_DAY_ARTIFACT_SCHEMA_VERSION,
            manifest_id="",
            aggregate_report_schema_version=CROSS_DAY_QUALITY_SCHEMA_VERSION,
            aggregate_report_sha256=_file_id(report_path),
            aggregate_id=report.aggregate_id,
            start_day=report.start_day,
            end_day=report.end_day,
            sources=sources,
        )
        manifest = replace(
            unsigned_manifest,
            manifest_id=_artifacts.content_id(
                unsigned_manifest.to_mapping(include_id=False)
            ),
        )
        manifest_bytes = _artifacts.canonical_json(manifest.to_mapping())
        _write_bytes(staging / "manifest.json", manifest_bytes)
        target = (
            resolved_root
            / "quality-aggregates"
            / f"{report.start_day}_{report.end_day}"
            / manifest.manifest_id.removeprefix(_SHA256_PREFIX)
        )
        already_present = _publish_cross_day_directory(
            staging,
            target,
            manifest_bytes=manifest_bytes,
            manifest=manifest,
        )
        staging = None
        return PublishedCrossDayQuality(
            target,
            manifest,
            report,
            already_present,
        )
    except QualificationError:
        raise
    except OSError as error:
        raise QualificationPublicationError(
            "cross-day quality staging is unavailable"
        ) from error
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def publish_qualification_campaign(
    captures: Iterable[CapturedEvidence],
    *,
    metadata: MetadataCapture,
    facts: CampaignFacts,
    output_root: Path,
    limits: QualificationLimits | None = None,
) -> PublishedQualification:
    """Stream captures into bounded segments and atomically publish one report."""

    sink = RotatingQualificationSink(
        metadata=metadata,
        output_root=output_root,
        limits=limits or QualificationLimits(),
        clock_offset_ms=facts.clock_offset_ms,
    )
    sink.open_day(facts.day, started_at_ns=facts.started_at_ns)
    try:
        for captured in captures:
            _require_capture_day(captured, facts.day)
            sink.append(captured)
        return sink.finalize_day(facts)
    except BaseException:
        sink.abort()
        raise


async def rollover_qualification_day(
    controller: QualificationRolloverController,
    *,
    context_for: Callable[[date], Awaitable[QualificationDayContext]],
    sink_lock: asyncio.Lock,
    context_lead_seconds: float = 30.0,
    drain_grace_seconds: float = 10 * 60,
    wait_until: Callable[[int], Awaitable[None]] | None = None,
    now_ns: Callable[[], int] = time_ns,
) -> PublishedQualification:
    """Prepare the next UTC day, drain after grace, and publish the oldest day."""

    if not 0 < context_lead_seconds < 24 * 60 * 60:
        raise ValueError("rollover context lead must be within one UTC day")
    if drain_grace_seconds < 0:
        raise ValueError("rollover drain grace must be non-negative")
    if len(controller.open_days) != 1:
        raise QualificationPublicationError(
            "scheduled rollover requires exactly one open qualification day"
        )
    current_day = controller.open_days[0]
    _, boundary_ns = _utc_day_bounds(current_day)
    next_day = current_day + timedelta(days=1)
    waiter = wait_until or _wait_until_ns
    await waiter(boundary_ns - int(context_lead_seconds * 1_000_000_000))
    context = await context_for(next_day)
    if now_ns() >= boundary_ns:
        raise QualificationError("next-day context was not ready before UTC midnight")
    async with sink_lock:
        await asyncio.to_thread(
            controller.prepare_next_day,
            boundary_ns,
            context=context,
        )
    await waiter(boundary_ns + int(drain_grace_seconds * 1_000_000_000))
    async with sink_lock:
        await _drain_available_capture_queues(
            controller.queue,
            controller.raw_queue,
            controller,
        )
        return await asyncio.to_thread(
            controller.finalize_oldest_day,
            ended_at_ns=boundary_ns,
        )


async def run_continuous_qualification_campaign(
    plan: SubscriptionPlan,
    *,
    initial_context: QualificationDayContext,
    context_for: Callable[[date], Awaitable[QualificationDayContext]],
    output_root: Path,
    stop: asyncio.Event,
    limits: QualificationLimits | None = None,
    startup_timeout_seconds: float = 30.0,
    context_lead_seconds: float = 30.0,
    drain_grace_seconds: float = 10 * 60,
    on_published: Callable[[PublishedQualification], Awaitable[None]] | None = None,
    on_status: Callable[[QualificationServiceStatus], Awaitable[None]] | None = None,
    wait_until: Callable[[int], Awaitable[None]] | None = None,
    now_ns: Callable[[], int] = time_ns,
) -> tuple[PublishedQualification, ...]:
    """Run collectors across UTC days until clean stop and publish the final window."""

    if stop.is_set():
        raise ValueError("continuous qualification stop is already set")
    if startup_timeout_seconds <= 0:
        raise ValueError("collector startup timeout must be positive")
    admitted_limits = limits or QualificationLimits()
    queue = BoundedCaptureQueue(admitted_limits.queue_capacity)
    raw_queue = BoundedRawCaptureQueue(admitted_limits.queue_capacity)
    admission = LiveAdmissionState(
        recent_identity_capacity=admitted_limits.recent_identity_capacity
    )
    run_id = initial_context.metadata.events[0].provenance.ingestion_run_id
    collectors = _qualification_collectors(
        plan,
        ingestion_run_id=run_id,
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
    )
    controller = QualificationRolloverController(
        metadata=initial_context.metadata,
        output_root=output_root,
        limits=admitted_limits,
        clock_offset_ms=initial_context.clock_offset_ms,
        collectors=collectors,
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
        breadth_symbol_count=len(plan.breadth_symbols),
    )
    launched_at_ns = now_ns()
    initial_day = _utc_day(launched_at_ns)
    controller.open_initial_day(initial_day, started_at_ns=launched_at_ns)
    sink_lock = asyncio.Lock()
    collectors_done = asyncio.Event()
    ready_events = tuple(asyncio.Event() for _ in collectors)
    collector_tasks = [
        asyncio.create_task(collector.run(stop=stop, ready=ready))
        for collector, ready in zip(collectors, ready_events, strict=True)
    ]
    consumer = asyncio.create_task(
        _drain_capture_queues(
            queue,
            raw_queue,
            controller,
            collectors_done=collectors_done,
            sink_lock=sink_lock,
        )
    )
    stop_wait = asyncio.create_task(stop.wait())
    rollover: asyncio.Task[PublishedQualification] | None = None
    try:
        await _notify_service_status(
            on_status,
            QualificationServicePhase.STARTING,
            ready=False,
            detail="collectors_starting",
        )
        await _await_collectors_ready(
            ready_events,
            collector_tasks,
            consumer,
            timeout_seconds=startup_timeout_seconds,
        )
        controller.set_collector_startup_ns(
            initial_day,
            now_ns() - launched_at_ns,
        )
        await _notify_service_status(
            on_status,
            QualificationServicePhase.READY,
            ready=True,
            detail="collectors_ready",
        )
        workers = {consumer, *collector_tasks}
        while not stop.is_set():
            rollover = asyncio.create_task(
                rollover_qualification_day(
                    controller,
                    context_for=context_for,
                    sink_lock=sink_lock,
                    context_lead_seconds=context_lead_seconds,
                    drain_grace_seconds=drain_grace_seconds,
                    wait_until=wait_until,
                    now_ns=now_ns,
                )
            )
            completed, _ = await asyncio.wait(
                {stop_wait, rollover, *workers},
                return_when=asyncio.FIRST_COMPLETED,
            )
            _raise_for_stopped_qualification_worker(completed, workers)
            if rollover in completed:
                published = await rollover
                rollover = None
                if on_published is not None:
                    await on_published(published)
                continue
            break

        await _notify_service_status(
            on_status,
            QualificationServicePhase.STOPPING,
            ready=False,
            detail="shutdown_requested",
        )
        if rollover is not None:
            rollover.cancel()
            await asyncio.gather(rollover, return_exceptions=True)
            rollover = None
        stop.set()
        await asyncio.gather(*collector_tasks)
        collectors_done.set()
        await consumer
        published = await _finalize_shutdown_days(
            controller,
            sink_lock=sink_lock,
            ended_at_ns=now_ns(),
        )
        if on_published is not None:
            for result in published:
                await on_published(result)
        await _notify_service_status(
            on_status,
            QualificationServicePhase.STOPPED,
            ready=False,
            detail="shutdown_complete",
        )
        return published
    except BaseException:
        try:
            await _notify_service_status(
                on_status,
                QualificationServicePhase.FAILED,
                ready=False,
                detail="supervisor_failed",
            )
        except Exception:
            pass
        stop.set()
        if rollover is not None:
            rollover.cancel()
        for task in (*collector_tasks, consumer):
            task.cancel()
        await asyncio.gather(
            *collector_tasks,
            consumer,
            *((rollover,) if rollover is not None else ()),
            return_exceptions=True,
        )
        controller.abort()
        raise
    finally:
        stop_wait.cancel()
        await asyncio.gather(stop_wait, return_exceptions=True)


async def run_qualification_campaign(
    plan: SubscriptionPlan,
    *,
    metadata: MetadataCapture,
    output_root: Path,
    duration_seconds: float,
    limits: QualificationLimits | None = None,
    clock_offset_ms: Decimal | None = None,
    startup_timeout_seconds: float = 30.0,
) -> PublishedQualification:
    """Run every planned shard and persist the bounded UTC campaign off-reader."""

    if duration_seconds <= 0:
        raise ValueError("qualification duration must be positive")
    if startup_timeout_seconds <= 0:
        raise ValueError("collector startup timeout must be positive")
    admitted_limits = limits or QualificationLimits()
    queue = BoundedCaptureQueue(admitted_limits.queue_capacity)
    raw_queue = BoundedRawCaptureQueue(admitted_limits.queue_capacity)
    admission = LiveAdmissionState(
        recent_identity_capacity=admitted_limits.recent_identity_capacity
    )
    collectors = _qualification_collectors(
        plan,
        ingestion_run_id=metadata.events[0].provenance.ingestion_run_id,
        queue=queue,
        raw_queue=raw_queue,
        admission=admission,
    )

    stager = _QualificationStager(
        metadata=metadata,
        output_root=output_root,
        limits=admitted_limits,
        clock_offset_ms=clock_offset_ms,
        window_start_ns=None,
    )
    stop = asyncio.Event()
    collectors_done = asyncio.Event()
    launched_at_ns = time_ns()
    ready_events = tuple(asyncio.Event() for _ in collectors)
    collector_tasks = [
        asyncio.create_task(collector.run(stop=stop, ready=ready))
        for collector, ready in zip(collectors, ready_events, strict=True)
    ]
    consumer = asyncio.create_task(
        _drain_capture_queues(
            queue,
            raw_queue,
            stager,
            collectors_done=collectors_done,
        )
    )
    deadline: asyncio.Task[None] | None = None
    try:
        await _await_collectors_ready(
            ready_events,
            collector_tasks,
            consumer,
            timeout_seconds=startup_timeout_seconds,
        )
        started_at_ns = time_ns()
        campaign_day = _utc_day(started_at_ns)
        ending_at_ns = started_at_ns + int(duration_seconds * 1_000_000_000)
        if _utc_day(ending_at_ns) != campaign_day:
            raise QualificationError(
                "one qualification publication cannot cross a UTC day"
            )
        await asyncio.to_thread(stager.begin_window, started_at_ns)
        deadline = asyncio.create_task(asyncio.sleep(duration_seconds))
        completed, _ = await asyncio.wait(
            {deadline, consumer, *collector_tasks},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if deadline not in completed:
            for task in completed:
                exception = task.exception()
                if exception is not None:
                    raise exception
            raise QualificationError("qualification worker stopped before its deadline")
        stop.set()
        stats = tuple(await asyncio.gather(*collector_tasks))
        collectors_done.set()
        await consumer
        ended_at_ns = time_ns()
        expected = _expected_closed_klines(
            started_at_ns,
            ended_at_ns,
            len(plan.breadth_symbols),
        )
        facts = CampaignFacts(
            day=campaign_day,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            expected_closed_klines=expected,
            queue_capacity=queue.capacity,
            queue_high_water_mark=max(
                queue.high_water_mark,
                raw_queue.high_water_mark,
            ),
            queue_overflows=queue.overflow_count + raw_queue.overflow_count,
            collector_stats=stats,
            clock_offset_ms=clock_offset_ms,
            collector_startup_ns=started_at_ns - launched_at_ns,
            recent_identity_capacity=admitted_limits.recent_identity_capacity,
            recent_identity_high_water_mark=(admission.recent_identity_high_water_mark),
            recent_identity_evictions=admission.recent_identity_evictions,
            # Cross-day aggregation, not one daily artifact, owns this claim.
            representative=False,
        )
        return await asyncio.to_thread(stager.finalize, facts)
    except BaseException:
        stop.set()
        if deadline is not None:
            deadline.cancel()
        for task in collector_tasks:
            task.cancel()
        consumer.cancel()
        await asyncio.gather(
            *collector_tasks,
            consumer,
            *((deadline,) if deadline is not None else ()),
            return_exceptions=True,
        )
        await asyncio.to_thread(stager.abort)
        raise


def _qualification_collectors(
    plan: SubscriptionPlan,
    *,
    ingestion_run_id: UUID,
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    admission: LiveAdmissionState,
) -> tuple[BinanceLiveCollector, ...]:
    stream_sets = (*plan.breadth_shards, plan.hot_streams)
    collectors = tuple(
        BinanceLiveCollector(
            streams=streams,
            ingestion_run_id=ingestion_run_id,
            queue=queue,
            admission=admission,
            raw_queue=raw_queue,
        )
        for streams in stream_sets
        if streams
    )
    if not collectors:
        raise QualificationError("qualification plan contains no streams")
    return collectors


def _raise_for_stopped_qualification_worker(
    completed: Set[asyncio.Task[object]],
    workers: Set[asyncio.Task[object]],
) -> None:
    stopped = completed & workers
    for task in stopped:
        exception = task.exception()
        if exception is not None:
            raise exception
    if stopped:
        raise QualificationError("qualification worker stopped unexpectedly")


async def _notify_service_status(
    callback: Callable[[QualificationServiceStatus], Awaitable[None]] | None,
    phase: QualificationServicePhase,
    *,
    ready: bool,
    detail: str,
) -> None:
    if callback is not None:
        await callback(QualificationServiceStatus(phase, ready, detail))


async def _finalize_shutdown_days(
    controller: QualificationRolloverController,
    *,
    sink_lock: asyncio.Lock,
    ended_at_ns: int,
) -> tuple[PublishedQualification, ...]:
    published: list[PublishedQualification] = []
    async with sink_lock:
        await asyncio.to_thread(controller.abort_future_day, ended_at_ns)
        while controller.open_days:
            day = controller.open_days[0]
            _, boundary_ns = _utc_day_bounds(day)
            published.append(
                await asyncio.to_thread(
                    controller.finalize_oldest_day,
                    ended_at_ns=min(ended_at_ns, boundary_ns),
                )
            )
    return tuple(published)


async def _await_collectors_ready(
    ready_events: tuple[asyncio.Event, ...],
    collector_tasks: Sequence[asyncio.Task[CaptureStats]],
    consumer: asyncio.Task[None],
    *,
    timeout_seconds: float,
) -> None:
    async def wait_all() -> None:
        await asyncio.gather(*(ready.wait() for ready in ready_events))

    readiness = asyncio.create_task(wait_all())
    completed, _ = await asyncio.wait(
        {readiness, consumer, *collector_tasks},
        timeout=timeout_seconds,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if readiness in completed:
        await readiness
        return
    readiness.cancel()
    await asyncio.gather(readiness, return_exceptions=True)
    for task in completed:
        exception = task.exception()
        if exception is not None:
            raise exception
    if completed:
        raise QualificationError("qualification worker stopped during startup")
    raise QualificationError("collectors did not become ready before the timeout")


async def _drain_capture_queues(
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    stager: QualificationEvidenceSink,
    *,
    collectors_done: asyncio.Event,
    sink_lock: asyncio.Lock | None = None,
) -> None:
    while not collectors_done.is_set() or queue.size or raw_queue.size:
        if sink_lock is None:
            await _drain_capture_batch(queue, raw_queue, stager)
            continue
        async with sink_lock:
            await _drain_capture_batch(queue, raw_queue, stager)


async def _drain_capture_batch(
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    sink: QualificationEvidenceSink,
) -> None:
    raw_batch: list[RawMessageObservation] = []
    accepted_batch: list[CapturedEvidence] = []
    if raw_queue.size:
        raw_batch.append(await raw_queue.get())
    elif not queue.size:
        try:
            raw_batch.append(await asyncio.wait_for(raw_queue.get(), timeout=0.25))
        except TimeoutError:
            pass
    while raw_queue.size and len(raw_batch) < 256:
        raw_batch.append(await raw_queue.get())
    while queue.size and len(accepted_batch) < 256:
        accepted_batch.append(await queue.get())
    await _persist_capture_batches(
        queue,
        raw_queue,
        sink,
        raw_batch=raw_batch,
        accepted_batch=accepted_batch,
    )


async def _drain_available_capture_queues(
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    sink: QualificationEvidenceSink,
) -> None:
    raw_remaining = raw_queue.size
    accepted_remaining = queue.size
    while raw_remaining or accepted_remaining:
        raw_batch = [await raw_queue.get() for _ in range(min(raw_remaining, 256))]
        accepted_batch = [
            await queue.get() for _ in range(min(accepted_remaining, 256))
        ]
        raw_remaining -= len(raw_batch)
        accepted_remaining -= len(accepted_batch)
        await _persist_capture_batches(
            queue,
            raw_queue,
            sink,
            raw_batch=raw_batch,
            accepted_batch=accepted_batch,
        )


async def _persist_capture_batches(
    queue: BoundedCaptureQueue,
    raw_queue: BoundedRawCaptureQueue,
    sink: QualificationEvidenceSink,
    *,
    raw_batch: Sequence[RawMessageObservation],
    accepted_batch: Sequence[CapturedEvidence],
) -> None:
    if raw_batch:
        await asyncio.to_thread(sink.append_raw_many, tuple(raw_batch))
        for _ in raw_batch:
            raw_queue.task_done()
    if accepted_batch:
        await asyncio.to_thread(sink.observe_many, tuple(accepted_batch))
        for _ in accepted_batch:
            queue.task_done()


async def _wait_until_ns(deadline_ns: int) -> None:
    while True:
        remaining_seconds = (deadline_ns - time_ns()) / 1_000_000_000
        if remaining_seconds <= 0:
            return
        await asyncio.sleep(min(remaining_seconds, 60.0))


def _quality_report(
    metadata: MetadataCapture,
    accumulator: _QualityAccumulator,
    facts: CampaignFacts,
    *,
    segments: Sequence[SegmentRecord],
) -> ProspectiveQualityReport:
    stats = facts.collector_stats
    capture_uncompressed_bytes = sum(item.uncompressed_bytes for item in segments)
    capture_compressed_bytes = sum(item.bytes for item in segments)
    capture_compression_ratio = (
        None
        if capture_uncompressed_bytes == 0
        else str(
            Decimal(capture_compressed_bytes) / Decimal(capture_uncompressed_bytes)
        )
    )
    input_messages = sum(item.messages_received for item in stats)
    refused_messages = sum(item.messages_refused for item in stats)
    provisional_messages = sum(item.provisional_messages for item in stats)
    raw_messages_retained = max(
        accumulator.persisted,
        sum(item.raw_messages_retained for item in stats),
    )
    raw_messages_suppressed = sum(item.raw_messages_suppressed for item in stats)
    duplicates = sum(item.duplicate_messages for item in stats)
    conflicts = sum(item.conflict_messages for item in stats)
    gaps = accumulator.observed_kline_gaps + sum(item.gap_messages for item in stats)
    schema_mismatches = sum(item.schema_mismatches for item in stats)
    queue_overflows = max(
        facts.queue_overflows,
        sum(item.queue_overflows for item in stats),
    )
    expected = facts.expected_closed_klines
    coverage = None
    if expected is not None and expected > 0:
        coverage = str(Decimal(accumulator.closed_klines) / Decimal(expected))

    metadata_receipt_ns = metadata.events[0].provenance.receipt.wall_time_ns
    metadata_age_ns = max(0, facts.ended_at_ns - metadata_receipt_ns)
    failures: list[str] = []
    if accumulator.persisted == 0:
        failures.append("NO_CAPTURED_MESSAGES")
    if expected is None or expected == 0:
        failures.append("CLOSED_KLINE_EXPECTATION_UNAVAILABLE")
    elif accumulator.closed_klines * 1_000 < expected * 999:
        failures.append("CLOSED_KLINE_COVERAGE_BELOW_99_9_PERCENT")
    if conflicts:
        failures.append("CONFLICTING_EVIDENCE")
    if gaps:
        failures.append("UNRESOLVED_GAP")
    if schema_mismatches:
        failures.append("SCHEMA_MISMATCH")
    if queue_overflows:
        failures.append("CAPTURE_QUEUE_OVERFLOW")
    if facts.unresolved_recoveries:
        failures.append("UNRESOLVED_RECOVERY")
    if accumulator.unknown_symbols:
        failures.append("UNKNOWN_INSTRUMENT")
    if metadata_age_ns > 24 * 60 * 60 * 1_000_000_000:
        failures.append("STALE_METADATA")
    if facts.clock_offset_ms is None:
        failures.append("CLOCK_QUALIFICATION_UNAVAILABLE")
    elif abs(facts.clock_offset_ms) > Decimal(100):
        failures.append("CLOCK_OFFSET_EXCEEDS_100_MS")
    if accumulator.negative_latency:
        failures.append("NEGATIVE_EVENT_LATENCY")
    raw_latency_p50 = accumulator.raw_latency.percentile(50)
    raw_latency_p95 = accumulator.raw_latency.percentile(95)
    raw_latency_p99 = accumulator.raw_latency.percentile(99)
    latency_p50 = accumulator.latency.percentile(50)
    latency_p95 = accumulator.latency.percentile(95)
    latency_p99 = accumulator.latency.percentile(99)
    raw_kline_latency_p95 = accumulator.raw_kline_latency.percentile(95)
    kline_latency_p95 = accumulator.kline_latency.percentile(95)
    if latency_p95 is not None and latency_p95 > 500_000_000:
        failures.append("EVENT_LATENCY_P95_EXCEEDS_500_MS")
    if latency_p99 is not None and latency_p99 > 1_500_000_000:
        failures.append("EVENT_LATENCY_P99_EXCEEDS_1500_MS")
    if kline_latency_p95 is not None and kline_latency_p95 > 3_000_000_000:
        failures.append("CLOSED_KLINE_LATENCY_P95_EXCEEDS_3_S")
    if not facts.representative:
        failures.append("REPRESENTATIVE_WINDOW_PENDING")

    by_symbol: tuple[dict[str, object], ...] = tuple(
        {"symbol": symbol, **counts}
        for symbol, counts in sorted(accumulator.by_symbol.items())
    )
    day_start_ns, next_day_start_ns = _utc_day_bounds(facts.day)
    unsigned_report = ProspectiveQualityReport(
        report_id="",
        schema_version=QUALITY_REPORT_SCHEMA_VERSION,
        day=facts.day.isoformat(),
        day_basis="receipt_wall_time_utc",
        run_id=str(accumulator.run_id),
        window_started_at_ns=facts.started_at_ns,
        window_ended_at_ns=facts.ended_at_ns,
        window_complete_utc_day=(
            facts.started_at_ns == day_start_ns
            and facts.ended_at_ns == next_day_start_ns
        ),
        input_messages=input_messages,
        persisted_messages=accumulator.persisted,
        refused_messages=refused_messages,
        provisional_messages=provisional_messages,
        raw_messages_retained=raw_messages_retained,
        raw_messages_suppressed=raw_messages_suppressed,
        capture_uncompressed_bytes=capture_uncompressed_bytes,
        capture_compressed_bytes=capture_compressed_bytes,
        capture_compression_ratio=capture_compression_ratio,
        startup_messages=accumulator.startup_messages,
        closed_klines=accumulator.closed_klines,
        aggregate_trades=accumulator.aggregate_trades,
        book_tickers=accumulator.book_tickers,
        expected_closed_klines=expected,
        closed_kline_coverage=coverage,
        duplicates=duplicates,
        conflicts=conflicts,
        gaps=gaps,
        schema_mismatches=schema_mismatches,
        queue_capacity=facts.queue_capacity,
        queue_high_water_mark=facts.queue_high_water_mark,
        queue_overflows=queue_overflows,
        reconnects=sum(item.reconnects for item in stats),
        rotations=sum(item.rotations for item in stats),
        raw_event_latency_p50_ns=raw_latency_p50,
        raw_event_latency_p95_ns=raw_latency_p95,
        raw_event_latency_p99_ns=raw_latency_p99,
        raw_event_latency_histogram=accumulator.raw_latency.snapshot(),
        event_latency_p50_ns=latency_p50,
        event_latency_p95_ns=latency_p95,
        event_latency_p99_ns=latency_p99,
        event_latency_histogram=accumulator.latency.snapshot(),
        raw_closed_kline_latency_p95_ns=raw_kline_latency_p95,
        raw_closed_kline_latency_histogram=accumulator.raw_kline_latency.snapshot(),
        closed_kline_latency_p95_ns=kline_latency_p95,
        closed_kline_latency_histogram=accumulator.kline_latency.snapshot(),
        collector_startup_seconds=str(
            Decimal(facts.collector_startup_ns) / Decimal(1_000_000_000)
        ),
        recent_identity_capacity=facts.recent_identity_capacity,
        recent_identity_high_water_mark=facts.recent_identity_high_water_mark,
        recent_identity_evictions=facts.recent_identity_evictions,
        metadata_age_seconds=str(Decimal(metadata_age_ns) / Decimal(1_000_000_000)),
        clock_offset_ms=(
            None if facts.clock_offset_ms is None else str(facts.clock_offset_ms)
        ),
        unresolved_recoveries=facts.unresolved_recoveries,
        unsupported_metadata_symbols=metadata.unsupported_symbols,
        unknown_symbols=tuple(sorted(accumulator.unknown_symbols)),
        critical_failures=tuple(failures),
        representative=facts.representative,
        ready=not failures,
        by_symbol=by_symbol,
    )
    return replace(
        unsigned_report,
        report_id=_artifacts.content_id(unsigned_report.to_mapping(include_id=False)),
    )


def _capture_line(observed: RawMessageObservation) -> bytes:
    payload = {
        "connection_generation": observed.connection_generation,
        "ingestion_run_id": str(observed.ingestion_run_id),
        "payload_digest": _SHA256_PREFIX
        + hashlib.sha256(observed.raw_payload).hexdigest(),
        "raw_payload_base64": base64.b64encode(observed.raw_payload).decode("ascii"),
        "raw_payload_bytes": len(observed.raw_payload),
        "receipt_monotonic_ns": observed.receipt.monotonic_ns,
        "receipt_wall_time_ns": observed.receipt.wall_time_ns,
        "retention_reason": observed.retention_reason,
        "schema_version": CAPTURE_RECORD_SCHEMA_VERSION,
    }
    return _artifacts.canonical_json(payload)


def _source_time_us(event: MarketEvidence) -> int | None:
    if not isinstance(event, (KlineEvent, AggregateTradeEvent, BookTickerEvent)):
        return None
    if event.source_event_time is None:
        return None
    return event.source_event_time.epoch_microseconds


def _require_capture_day(captured: CapturedEvidence, expected: date) -> None:
    if _utc_day(captured.event.provenance.receipt.wall_time_ns) != expected:
        raise QualificationError("capture receipt falls outside the report UTC day")


def _utc_day(timestamp_ns: int) -> date:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).date()


def _utc_day_bounds(day: date) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start_ns = int(start.timestamp()) * 1_000_000_000
    next_start_ns = int((start + timedelta(days=1)).timestamp()) * 1_000_000_000
    return start_ns, next_start_ns


def _expected_closed_klines(start_ns: int, end_ns: int, symbols: int) -> int:
    closed_minutes = max(0, end_ns // _MINUTE_NS - start_ns // _MINUTE_NS)
    return closed_minutes * symbols


def _publish_directory(
    staging: Path,
    target: Path,
    *,
    manifest_bytes: bytes,
    manifest: QualificationManifest,
) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _verify_existing(target, manifest_bytes, manifest)
            shutil.rmtree(staging)
            return True
        os.replace(staging, target)
    except QualificationError:
        raise
    except OSError as error:
        if target.exists():
            _verify_existing(target, manifest_bytes, manifest)
            shutil.rmtree(staging, ignore_errors=True)
            return True
        raise QualificationPublicationError(
            "atomic qualification publication failed"
        ) from error
    return False


def load_published_qualification(path: Path) -> PublishedQualification:
    """Load and verify one content-addressed daily qualification publication."""

    resolved = path.resolve()
    manifest_payload = _read_json_mapping(
        resolved / "manifest.json",
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    report_payload = _read_json_mapping(
        resolved / "quality-report.json",
        maximum_bytes=_MAX_QUALITY_REPORT_BYTES,
    )
    try:
        manifest = _qualification_manifest_from_mapping(manifest_payload)
        report = _prospective_report_from_mapping(report_payload)
    except (TypeError, ValueError) as error:
        raise QualificationPublicationError(
            "daily qualification publication contains invalid typed content"
        ) from error

    if (
        manifest.schema_version != QUALIFICATION_SCHEMA_VERSION
        or manifest.capture_record_schema_version != CAPTURE_RECORD_SCHEMA_VERSION
        or manifest.quality_report_schema_version != QUALITY_REPORT_SCHEMA_VERSION
    ):
        raise QualificationPublicationError(
            "daily qualification manifest schema version is unsupported"
        )
    if report.schema_version != QUALITY_REPORT_SCHEMA_VERSION:
        raise QualificationPublicationError(
            "daily quality report schema version is unsupported"
        )
    if manifest.manifest_id != _artifacts.content_id(
        manifest.to_mapping(include_id=False)
    ):
        raise QualificationPublicationError(
            "daily qualification manifest identity is invalid"
        )
    if report.report_id != _artifacts.content_id(report.to_mapping(include_id=False)):
        raise QualificationPublicationError("daily quality report identity is invalid")
    if (
        resolved.name != manifest.manifest_id.removeprefix(_SHA256_PREFIX)
        or resolved.parent.name != manifest.run_id
        or resolved.parent.parent.name != manifest.day
    ):
        raise QualificationPublicationError(
            "daily qualification path disagrees with its manifest"
        )

    publication = PublishedQualification(
        path=resolved,
        manifest=manifest,
        report=report,
        already_present=True,
    )
    _verify_daily_publication(publication)
    return publication


def discover_recent_complete_qualifications(
    output_root: Path,
    *,
    limit: int,
) -> tuple[PublishedQualification, ...]:
    """Return the newest verified contiguous suffix of complete UTC days."""

    if limit < 0:
        raise ValueError("qualification discovery limit must be non-negative")
    if limit == 0:
        return ()
    prospective = output_root.resolve() / "prospective"
    if not prospective.exists():
        return ()
    try:
        day_entries = tuple(path for path in prospective.iterdir() if path.is_dir())
    except OSError as error:
        raise QualificationPublicationError(
            "daily qualification root is unreadable"
        ) from error

    dated_entries: list[tuple[date, Path]] = []
    for path in day_entries:
        try:
            observed_day = date.fromisoformat(path.name)
        except ValueError as error:
            raise QualificationPublicationError(
                "daily qualification root contains an invalid day directory"
            ) from error
        if observed_day.isoformat() != path.name:
            raise QualificationPublicationError(
                "daily qualification day directory is not canonical"
            )
        dated_entries.append((observed_day, path))

    selected: list[PublishedQualification] = []
    latest_day: date | None = None
    for observed_day, day_path in sorted(dated_entries, reverse=True):
        if latest_day is not None and observed_day != latest_day - timedelta(days=1):
            break
        complete = _complete_publication_for_day(day_path)
        if complete is None:
            if latest_day is not None:
                break
            continue
        selected.append(complete)
        latest_day = observed_day
        if len(selected) == limit:
            break
    selected.reverse()
    return tuple(selected)


def _complete_publication_for_day(day_path: Path) -> PublishedQualification | None:
    manifest_paths: list[Path] = []
    try:
        for run_path in day_path.iterdir():
            if not run_path.is_dir():
                raise QualificationPublicationError(
                    "daily qualification day contains a non-directory entry"
                )
            for publication_path in run_path.iterdir():
                if not publication_path.is_dir():
                    raise QualificationPublicationError(
                        "daily qualification run contains a non-directory entry"
                    )
                manifest_paths.append(publication_path / "manifest.json")
                if len(manifest_paths) > _MAX_PUBLICATIONS_PER_DAY:
                    raise QualificationPublicationError(
                        "daily qualification publication count exceeds its bound"
                    )
    except QualificationError:
        raise
    except OSError as error:
        raise QualificationPublicationError(
            "daily qualification directory is unreadable"
        ) from error

    complete: list[PublishedQualification] = []
    for manifest_path in sorted(manifest_paths):
        publication = load_published_qualification(manifest_path.parent)
        if publication.report.window_complete_utc_day:
            complete.append(publication)
    if len(complete) > 1:
        raise QualificationPublicationError(
            "daily qualification day has ambiguous complete publications"
        )
    return complete[0] if complete else None


def _read_json_mapping(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
    except OSError as error:
        raise QualificationPublicationError(
            "daily qualification publication is unreadable"
        ) from error
    if len(payload) > maximum_bytes:
        raise QualificationPublicationError(
            "daily qualification JSON exceeds its byte bound"
        )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationPublicationError(
            "daily qualification JSON is invalid"
        ) from error
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise QualificationPublicationError(
            "daily qualification JSON must be an object"
        )
    return cast(dict[str, object], decoded)


def _qualification_manifest_from_mapping(
    payload: dict[str, object],
) -> QualificationManifest:
    expected = set(get_type_hints(QualificationManifest))
    _require_exact_keys(payload, expected, "daily qualification manifest")
    segments_payload = payload["segments"]
    if not isinstance(segments_payload, list):
        raise TypeError("daily qualification manifest segments must be a list")
    segments = tuple(_segment_from_mapping(item) for item in segments_payload)
    normalized = {**payload, "segments": segments}
    _require_runtime_types(QualificationManifest, normalized)
    manifest = QualificationManifest(**cast(dict[str, Any], normalized))
    if manifest.to_mapping() != payload:
        raise ValueError("daily qualification manifest is not canonical")
    return manifest


def _segment_from_mapping(payload: object) -> SegmentRecord:
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise TypeError("daily qualification segment must be an object")
    normalized = cast(dict[str, object], payload)
    _require_exact_keys(
        normalized,
        set(get_type_hints(SegmentRecord)),
        "daily qualification segment",
    )
    _require_runtime_types(SegmentRecord, normalized)
    segment = SegmentRecord(**cast(dict[str, Any], normalized))
    if segment.to_mapping() != normalized:
        raise ValueError("daily qualification segment is not canonical")
    return segment


def _prospective_report_from_mapping(
    payload: dict[str, object],
) -> ProspectiveQualityReport:
    expected = set(get_type_hints(ProspectiveQualityReport))
    _require_exact_keys(payload, expected, "daily quality report")
    normalized = dict(payload)
    histogram_fields = (
        "raw_event_latency_histogram",
        "event_latency_histogram",
        "raw_closed_kline_latency_histogram",
        "closed_kline_latency_histogram",
    )
    for name in histogram_fields:
        normalized[name] = _latency_histogram_from_mapping(normalized[name])
    for name in (
        "unsupported_metadata_symbols",
        "unknown_symbols",
        "critical_failures",
    ):
        value = normalized[name]
        if not isinstance(value, list):
            raise TypeError(f"daily quality report {name} must be a list")
        normalized[name] = tuple(value)
    by_symbol = normalized["by_symbol"]
    if not isinstance(by_symbol, list) or not all(
        isinstance(item, dict) and all(isinstance(key, str) for key in item)
        for item in by_symbol
    ):
        raise TypeError("daily quality report by_symbol must be a list of objects")
    normalized["by_symbol"] = tuple(by_symbol)
    _require_runtime_types(ProspectiveQualityReport, normalized)
    report = ProspectiveQualityReport(**cast(dict[str, Any], normalized))
    if report.to_mapping() != payload:
        raise ValueError("daily quality report is not canonical")
    return report


def _latency_histogram_from_mapping(payload: object) -> LatencyHistogramRecord:
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise TypeError("latency histogram must be an object")
    mapping = cast(dict[str, object], payload)
    _require_exact_keys(
        mapping,
        {"bucket_rule", "counts", "schema_version", "unit"},
        "latency histogram",
    )
    if mapping["bucket_rule"] != _LATENCY_HISTOGRAM_BUCKET_RULE:
        raise ValueError("latency histogram bucket rule is unsupported")
    if mapping["unit"] != "nanoseconds":
        raise ValueError("latency histogram unit is unsupported")
    counts = mapping["counts"]
    if not isinstance(counts, list):
        raise TypeError("latency histogram counts must be a list")
    normalized = {
        "schema_version": mapping["schema_version"],
        "counts": tuple(counts),
    }
    _require_runtime_types(LatencyHistogramRecord, normalized)
    record = LatencyHistogramRecord(**cast(dict[str, Any], normalized))
    if record.to_mapping() != mapping:
        raise ValueError("latency histogram is not canonical")
    return record


def _require_exact_keys(
    payload: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields do not match its schema")


def _require_runtime_types(
    record_type: type[object],
    values: dict[str, object],
) -> None:
    for name, annotation in get_type_hints(record_type).items():
        if not _matches_runtime_type(values[name], annotation):
            raise TypeError(f"{record_type.__name__}.{name} has an invalid type")


def _matches_runtime_type(value: object, annotation: object) -> bool:
    if annotation is object or annotation is Any:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is types.UnionType:
        return any(_matches_runtime_type(value, item) for item in arguments)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches_runtime_type(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches_runtime_type(item, item_type)
            for item, item_type in zip(value, arguments, strict=True)
        )
    if origin is dict:
        return isinstance(value, dict) and all(
            _matches_runtime_type(key, arguments[0])
            and _matches_runtime_type(item, arguments[1])
            for key, item in value.items()
        )
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    return isinstance(value, cast(type[object], annotation))


def _verify_daily_publication(publication: PublishedQualification) -> None:
    manifest = publication.manifest
    report = publication.report
    if (
        manifest.day != report.day
        or manifest.run_id != report.run_id
        or manifest.report_id != report.report_id
        or manifest.quality_report_schema_version != report.schema_version
    ):
        raise QualificationError(
            "daily qualification manifest and report identities disagree"
        )
    _verify_existing(
        publication.path,
        _artifacts.canonical_json(manifest.to_mapping()),
        manifest,
    )


def _publish_cross_day_directory(
    staging: Path,
    target: Path,
    *,
    manifest_bytes: bytes,
    manifest: CrossDayQualityManifest,
) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _verify_existing_cross_day(target, manifest_bytes, manifest)
            shutil.rmtree(staging)
            return True
        os.replace(staging, target)
    except QualificationError:
        raise
    except OSError as error:
        if target.exists():
            _verify_existing_cross_day(target, manifest_bytes, manifest)
            shutil.rmtree(staging, ignore_errors=True)
            return True
        raise QualificationPublicationError(
            "atomic cross-day quality publication failed"
        ) from error
    return False


def _verify_existing_cross_day(
    target: Path,
    manifest_bytes: bytes,
    manifest: CrossDayQualityManifest,
) -> None:
    try:
        if (target / "manifest.json").read_bytes() != manifest_bytes:
            raise QualificationPublicationError(
                "existing cross-day manifest conflicts with staged content"
            )
        if (
            _file_id(target / "aggregate-quality-report.json")
            != manifest.aggregate_report_sha256
        ):
            raise QualificationPublicationError(
                "existing cross-day report conflicts with its manifest"
            )
    except QualificationError:
        raise
    except OSError as error:
        raise QualificationPublicationError(
            "existing cross-day quality publication is unreadable"
        ) from error


def _verify_existing(
    target: Path,
    manifest_bytes: bytes,
    manifest: QualificationManifest,
) -> None:
    try:
        if (target / "manifest.json").read_bytes() != manifest_bytes:
            raise QualificationPublicationError(
                "existing qualification manifest conflicts with staged content"
            )
        expected = {
            "exchange-info.json": manifest.metadata_sha256,
            "quality-report.json": manifest.quality_report_sha256,
            **{segment.name: segment.sha256 for segment in manifest.segments},
        }
        if any(_file_id(target / name) != digest for name, digest in expected.items()):
            raise QualificationPublicationError(
                "existing qualification file conflicts with its manifest"
            )
    except QualificationError:
        raise
    except OSError as error:
        raise QualificationPublicationError(
            "existing qualification publication is unreadable"
        ) from error


def _write_bytes(path: Path, payload: bytes) -> None:
    """Publish staging bytes, naming this module's failure if the write cannot.

    The write itself is the shared one: an exclusive create followed by an
    fsync. It is delegated rather than repeated so that the durability rule is
    stated once, and every staging write in the package keeps or loses it
    together.
    """

    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise QualificationPublicationError(
            "qualification staging write failed"
        ) from error


def _file_id(path: Path) -> str:
    try:
        digest, _size = _artifacts.file_identity(path)
    except OSError as error:
        raise QualificationPublicationError(
            "qualification staged file is unavailable"
        ) from error
    return digest


def _merge_latency_histograms(
    records: Iterable[LatencyHistogramRecord],
) -> LatencyHistogramRecord:
    counts = [0] * _LATENCY_HISTOGRAM_BUCKETS
    observed = False
    for record in records:
        if record.schema_version != LATENCY_HISTOGRAM_SCHEMA_VERSION:
            raise QualificationError("latency histogram schemas are incompatible")
        observed = True
        for index, count in enumerate(record.counts):
            counts[index] += count
    if not observed:
        raise QualificationError("latency histogram aggregation requires records")
    return LatencyHistogramRecord(
        schema_version=LATENCY_HISTOGRAM_SCHEMA_VERSION,
        counts=tuple(counts),
    )
