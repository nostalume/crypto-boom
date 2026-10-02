"""Verified monthly kline archives admitted by an availability report."""

from __future__ import annotations

import csv
import hashlib
import logging
import multiprocessing as mp
import zipfile
from asyncio import (
    CancelledError,
    create_task,
    gather,
    get_running_loop,
    sleep,
    to_thread,
)
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from time import monotonic, monotonic_ns, time_ns
from uuid import UUID

import aiohttp
import msgspec

from crypto_boom import _artifacts
from crypto_boom.binance_source import BINANCE_SPOT
from crypto_boom.history._transport import download_file
from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
)
from crypto_boom.history.codec import (
    DECODER_VERSION,
    ArchiveBatchError,
    ArchiveError,
    ArchiveIntegrityError,
    ArchivePublicationError,
    ArchiveResourceError,
    ArchiveSchemaError,
    ArchiveTransportError,
    admit_single_archive_member,
    archive_timestamp_unit,
    decode_binance_kline_row,
    validate_one_minute_kline_time,
)
from crypto_boom.market import (
    Environment,
    EvidenceAdmissionError,
    InstrumentId,
    KlineEvent,
    LocalReceipt,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    VenueId,
)

_MICROSECONDS_PER_MINUTE = 60_000_000
_MICROSECONDS_PER_DAY = 86_400_000_000


@dataclass(frozen=True, slots=True)
class MonthlyArchiveLimits:
    """Resource and scheduling limits for verified monthly acquisition."""

    maximum_archives: int = 500
    maximum_concurrency: int = 4
    maximum_decode_processes: int = 1
    maximum_total_compressed_bytes: int = 1 * 1_024 * 1_024 * 1_024
    maximum_elapsed_seconds: float = 300.0
    compressed_bytes: int = 64 * 1_024 * 1_024
    uncompressed_bytes: int = 256 * 1_024 * 1_024
    rows: int = 44_640
    request_timeout_seconds: float = 120.0
    decode_timeout_seconds: float = 60.0
    chunk_bytes: int = 64 * 1_024
    retry_attempts: int = 4
    retry_base_seconds: float = 1.0
    request_spacing_seconds: float = 0.25

    def __post_init__(self) -> None:
        counts = (
            self.maximum_archives,
            self.maximum_concurrency,
            self.maximum_decode_processes,
            self.maximum_total_compressed_bytes,
            self.compressed_bytes,
            self.uncompressed_bytes,
            self.rows,
            self.chunk_bytes,
            self.retry_attempts,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("monthly archive counts and byte limits must be positive")
        if (
            self.request_timeout_seconds <= 0
            or self.decode_timeout_seconds <= 0
            or self.maximum_elapsed_seconds <= 0
        ):
            raise ValueError("monthly archive timeouts must be positive")
        if self.retry_base_seconds < 0 or self.request_spacing_seconds < 0:
            raise ValueError("monthly archive delays cannot be negative")
        if type(self.maximum_decode_processes) is not int or not (
            1 <= self.maximum_decode_processes <= 2
        ):
            raise ValueError("monthly archive decode process count must be 1 or 2")


DEFAULT_MONTHLY_ARCHIVE_LIMITS = MonthlyArchiveLimits()


@dataclass(frozen=True, slots=True)
class MonthlyGap:
    """One missing run strictly between decoded one-minute observations."""

    first_missing_open_time_us: int
    last_missing_open_time_us: int
    missing_minutes: int

    def __post_init__(self) -> None:
        if (
            self.first_missing_open_time_us <= 0
            or self.last_missing_open_time_us < self.first_missing_open_time_us
            or self.missing_minutes <= 0
        ):
            raise ArchiveIntegrityError("monthly archive gap is invalid")
        expected = (
            self.last_missing_open_time_us - self.first_missing_open_time_us
        ) // _MICROSECONDS_PER_MINUTE + 1
        if expected != self.missing_minutes:
            raise ArchiveIntegrityError("monthly archive gap count is inconsistent")

    def to_mapping(self) -> dict[str, object]:
        return {
            "first_missing_open_time_us": self.first_missing_open_time_us,
            "last_missing_open_time_us": self.last_missing_open_time_us,
            "missing_minutes": self.missing_minutes,
        }


@dataclass(frozen=True, slots=True)
class MonthlyArchiveRequest:
    """One AVAILABLE source observation converted into a download request."""

    instrument: InstrumentId
    month: date
    interval: str
    base_url: str
    expected_sha256: str

    def __post_init__(self) -> None:
        if self.instrument.venue != BINANCE_SPOT:
            raise ArchiveSchemaError("monthly archive venue must be Binance Spot")
        if self.instrument.environment is not Environment.PRODUCTION:
            raise ArchiveSchemaError("monthly archive environment must be production")
        if self.month.day != 1:
            raise ArchiveSchemaError("monthly archive month must be its first day")
        if self.interval != "1m":
            raise ArchiveSchemaError("monthly archive supports only one-minute klines")
        if not self.base_url:
            raise ArchiveSchemaError("monthly archive base URL is empty")
        if not _is_digest(self.expected_sha256):
            raise ArchiveIntegrityError("monthly archive checksum is invalid")

    @property
    def month_text(self) -> str:
        return self.month.strftime("%Y-%m")

    @property
    def stem(self) -> str:
        return f"{self.instrument.symbol}-{self.interval}-{self.month_text}"

    @property
    def archive_filename(self) -> str:
        return f"{self.stem}.zip"

    @property
    def member_filename(self) -> str:
        return f"{self.stem}.csv"

    @property
    def source_url(self) -> str:
        return (
            f"{self.base_url.rstrip('/')}/klines/{self.instrument.symbol}/"
            f"{self.interval}/{self.archive_filename}"
        )


@dataclass(frozen=True, slots=True)
class MonthlyArchiveManifest:
    """Content and timestamp coverage of one monthly source revision."""

    schema_version: int
    decoder_version: str
    venue: str
    market: str
    environment: str
    dataset: str
    symbol: str
    interval: str
    month: str
    timestamp_unit: str
    source_url: str
    checksum_url: str
    archive_filename: str
    member_filename: str
    source_revision: str
    archive_sha256: str
    member_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    row_count: int
    first_open_time_us: int
    last_open_time_us: int
    observed_first_day: str
    observed_last_day: str
    observed_days: tuple[str, ...]
    missing_days: tuple[str, ...]
    internal_gaps: tuple[MonthlyGap, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.decoder_version != DECODER_VERSION:
            raise ArchiveIntegrityError("monthly archive manifest version is invalid")
        if self.interval != "1m" or self.dataset != "klines":
            raise ArchiveIntegrityError("monthly archive manifest dataset is invalid")
        digests = self.source_revision, self.archive_sha256, self.member_sha256
        if any(not _artifacts.is_sha256(value) for value in digests):
            raise ArchiveIntegrityError("monthly archive manifest digest is invalid")
        if (
            self.compressed_bytes <= 0
            or self.uncompressed_bytes <= 0
            or self.row_count <= 0
            or self.first_open_time_us <= 0
            or self.last_open_time_us < self.first_open_time_us
        ):
            raise ArchiveIntegrityError("monthly archive manifest extent is invalid")
        if (
            not self.observed_days
            or tuple(sorted(self.observed_days)) != self.observed_days
        ):
            raise ArchiveIntegrityError("monthly observed days are invalid")
        if len(set(self.observed_days)) != len(self.observed_days):
            raise ArchiveIntegrityError("monthly observed days contain duplicates")
        if tuple(sorted(self.missing_days)) != self.missing_days:
            raise ArchiveIntegrityError("monthly missing days are invalid")
        if set(self.observed_days) & set(self.missing_days):
            raise ArchiveIntegrityError("monthly observed and missing days overlap")
        if (
            self.observed_first_day != self.observed_days[0]
            or self.observed_last_day != self.observed_days[-1]
        ):
            raise ArchiveIntegrityError("monthly observed bounds are inconsistent")

    @property
    def internal_gap_count(self) -> int:
        return len(self.internal_gaps)

    @property
    def internal_missing_minutes(self) -> int:
        return sum(item.missing_minutes for item in self.internal_gaps)

    def _content_mapping(self) -> dict[str, object]:
        return {
            "archive_filename": self.archive_filename,
            "archive_sha256": self.archive_sha256,
            "checksum_url": self.checksum_url,
            "compressed_bytes": self.compressed_bytes,
            "dataset": self.dataset,
            "decoder_version": self.decoder_version,
            "environment": self.environment,
            "first_open_time_us": self.first_open_time_us,
            "internal_gap_count": self.internal_gap_count,
            "internal_gaps": [item.to_mapping() for item in self.internal_gaps],
            "internal_missing_minutes": self.internal_missing_minutes,
            "interval": self.interval,
            "last_open_time_us": self.last_open_time_us,
            "market": self.market,
            "member_filename": self.member_filename,
            "member_sha256": self.member_sha256,
            "missing_days": list(self.missing_days),
            "month": self.month,
            "observed_days": list(self.observed_days),
            "observed_first_day": self.observed_first_day,
            "observed_last_day": self.observed_last_day,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "source_url": self.source_url,
            "symbol": self.symbol,
            "timestamp_unit": self.timestamp_unit,
            "uncompressed_bytes": self.uncompressed_bytes,
            "venue": self.venue,
        }

    @property
    def manifest_id(self) -> str:
        digest = hashlib.sha256(
            _artifacts.canonical_json(self._content_mapping())
        ).hexdigest()
        return f"sha256:{digest}"

    def to_mapping(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self._content_mapping()}


_MANIFEST_DECODER = msgspec.json.Decoder(MonthlyArchiveManifest)


@dataclass(frozen=True, slots=True)
class PublishedMonthlyArchive:
    path: Path
    manifest: MonthlyArchiveManifest
    availability_report_id: str
    already_present: bool


@dataclass(frozen=True, slots=True)
class MonthlyArchiveBatchResult:
    availability_report_id: str
    publications: tuple[PublishedMonthlyArchive, ...]

    @property
    def archive_count(self) -> int:
        return len(self.publications)

    @property
    def downloaded_count(self) -> int:
        return sum(not item.already_present for item in self.publications)

    @property
    def reused_count(self) -> int:
        return sum(item.already_present for item in self.publications)

    @property
    def compressed_bytes(self) -> int:
        return sum(item.manifest.compressed_bytes for item in self.publications)


@dataclass(frozen=True, slots=True)
class MonthlyKlineDay:
    """An in-memory daily view decoded from one monthly ZIP."""

    day: date
    events: tuple[KlineEvent, ...]


@dataclass(frozen=True, slots=True)
class _DecodedMonthlyArchive:
    manifest: MonthlyArchiveManifest
    events: tuple[KlineEvent, ...]


@dataclass(frozen=True, slots=True)
class _Acquisition:
    receipt: LocalReceipt
    ingestion_run_id: UUID
    availability_report_id: str
    pool_id: str
    availability_probed_at_ns: int


class _AcquisitionDocument(msgspec.Struct):
    schema_version: int
    wall_time_unit: str
    download_completed_wall_time_ns: int
    download_completed_monotonic_ns: int
    ingestion_run_id: str
    availability_report_id: str
    pool_id: str
    availability_probed_at_ns: int


_ACQUISITION_DECODER = msgspec.json.Decoder(_AcquisitionDocument)


async def acquire_monthly_archives(
    report: ArchiveAvailabilityReport,
    *,
    output_root: Path,
    ingestion_run_id: UUID,
    limits: MonthlyArchiveLimits = DEFAULT_MONTHLY_ARCHIVE_LIMITS,
    symbols: tuple[str, ...] = (),
    on_published: Callable[[PublishedMonthlyArchive], None] | None = None,
) -> MonthlyArchiveBatchResult:
    """Download AVAILABLE observations with deterministic bounded concurrency."""

    selected = _admit_symbol_filter(report, symbols)
    requests = tuple(
        _request_from_probe(report, probe)
        for probe in report.probes
        if probe.state is AvailabilityState.AVAILABLE
        and (selected is None or probe.symbol in selected)
    )
    if len(requests) > limits.maximum_archives:
        raise ArchiveResourceError(
            f"monthly acquisition has {len(requests)} archives; "
            f"limit is {limits.maximum_archives}"
        )
    logging.getLogger(__name__).info(
        "Monthly acquisition: %d selected archives", len(requests)
    )
    if not requests:
        return MonthlyArchiveBatchResult(report.report_id, ())

    output_root = output_root.resolve()
    _prepare_root(output_root)
    timeout = aiohttp.ClientTimeout(total=limits.request_timeout_seconds)
    publications: list[PublishedMonthlyArchive] = []
    batch_started = monotonic()
    admitted_compressed_bytes = 0
    decoder = (
        ProcessPoolExecutor(
            max_workers=limits.maximum_decode_processes,
            mp_context=mp.get_context("spawn"),
            max_tasks_per_child=1,
        )
        if limits.maximum_decode_processes > 1
        else None
    )
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, auto_decompress=False
        ) as session:
            for group_start in range(0, len(requests), limits.maximum_concurrency):
                group = requests[group_start : group_start + limits.maximum_concurrency]
                request = group[0]
                if monotonic() - batch_started >= limits.maximum_elapsed_seconds:
                    raise ArchiveBatchError(
                        "monthly acquisition elapsed-time budget was exhausted before "
                        f"{request.instrument.symbol} {request.month_text}"
                    )
                remaining_bytes = (
                    limits.maximum_total_compressed_bytes - admitted_compressed_bytes
                )
                if remaining_bytes <= 0:
                    raise ArchiveBatchError(
                        "monthly acquisition compressed-byte budget was exhausted before "
                        f"{request.instrument.symbol} {request.month_text}"
                    )
                per_archive_budget = min(
                    limits.compressed_bytes,
                    remaining_bytes // len(group),
                )
                if per_archive_budget <= 0:
                    raise ArchiveBatchError(
                        "monthly acquisition remaining byte budget cannot admit "
                        "the next concurrent group"
                    )
                tasks = []
                for group_index, grouped_request in enumerate(group):
                    tasks.append(
                        create_task(
                            _acquire_with_retries(
                                grouped_request,
                                report=report,
                                output_root=output_root,
                                ingestion_run_id=ingestion_run_id,
                                session=session,
                                decoder=decoder,
                                limits=limits,
                                maximum_compressed_bytes=per_archive_budget,
                            )
                        )
                    )
                    if (
                        group_index + 1 < len(group)
                        and limits.request_spacing_seconds > 0
                    ):
                        await sleep(limits.request_spacing_seconds)
                results = await gather(*tasks, return_exceptions=True)
                first_error: tuple[MonthlyArchiveRequest, ArchiveError] | None = None
                for grouped_request, result in zip(group, results, strict=True):
                    if isinstance(result, CancelledError):
                        raise result
                    if isinstance(result, BaseException):
                        if not isinstance(result, ArchiveError):
                            raise result
                        if first_error is None:
                            first_error = grouped_request, result
                        continue
                    publications.append(result)
                    logging.getLogger(__name__).info(
                        "Monthly acquisition progress: %d/%d; %s %s reused=%s",
                        len(publications),
                        len(requests),
                        result.manifest.symbol,
                        result.manifest.month,
                        result.already_present,
                    )
                    admitted_compressed_bytes += result.manifest.compressed_bytes
                    if on_published is not None:
                        on_published(result)
                if first_error is not None:
                    failed_request, error = first_error
                    raise ArchiveBatchError(
                        "monthly acquisition failed at "
                        f"{failed_request.instrument.symbol} "
                        f"{failed_request.month_text}: {error}"
                    ) from error
        return MonthlyArchiveBatchResult(report.report_id, tuple(publications))
    finally:
        if decoder is not None:
            decoder.shutdown(wait=True, cancel_futures=True)


def load_published_monthly_klines_by_day(
    published: PublishedMonthlyArchive,
    *,
    limits: MonthlyArchiveLimits = DEFAULT_MONTHLY_ARCHIVE_LIMITS,
) -> tuple[MonthlyKlineDay, ...]:
    """Reverify one monthly ZIP and expose ordered daily canonical inputs."""

    manifest = published.manifest
    _verify_existing_manifest(
        published.path,
        _artifacts.canonical_json(manifest.to_mapping()),
    )
    archive_path = published.path / manifest.archive_filename
    archive_sha256, compressed_bytes = _hash_file(
        archive_path,
        maximum=limits.compressed_bytes,
    )
    if (
        f"sha256:{archive_sha256}" != manifest.archive_sha256
        or compressed_bytes != manifest.compressed_bytes
    ):
        raise ArchiveIntegrityError("published monthly archive bytes are inconsistent")
    acquisition = _read_acquisition(published.path / "acquisition.json")
    decoded = _inspect_archive(
        _request_from_manifest(manifest),
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        compressed_bytes=compressed_bytes,
        receipt=acquisition.receipt,
        ingestion_run_id=acquisition.ingestion_run_id,
        limits=limits,
        collect_events=True,
    )
    if decoded.manifest != manifest:
        raise ArchiveIntegrityError(
            "published monthly archive no longer matches manifest"
        )

    grouped: list[MonthlyKlineDay] = []
    current_day: date | None = None
    current_events: list[KlineEvent] = []
    for event in decoded.events:
        event_day = date(1970, 1, 1) + timedelta(
            days=event.open_time.epoch_microseconds // _MICROSECONDS_PER_DAY
        )
        if current_day is not None and event_day != current_day:
            grouped.append(MonthlyKlineDay(current_day, tuple(current_events)))
            current_events = []
        current_day = event_day
        current_events.append(event)
    if current_day is not None:
        grouped.append(MonthlyKlineDay(current_day, tuple(current_events)))
    return tuple(grouped)


def load_published_monthly_archive(
    path: Path,
    *,
    limits: MonthlyArchiveLimits = DEFAULT_MONTHLY_ARCHIVE_LIMITS,
) -> PublishedMonthlyArchive:
    """Strictly reload one monthly publication after a process restart."""

    path = path.resolve()
    manifest = _read_manifest(path / "manifest.json")
    expected_tail = (
        manifest.symbol,
        manifest.interval,
        manifest.month,
        manifest.archive_sha256.removeprefix("sha256:"),
    )
    actual_tail = (*(part.name for part in path.parents[:3][::-1]), path.name)
    if actual_tail != expected_tail:
        raise ArchiveIntegrityError("monthly archive path does not match its identity")
    digest, size = _hash_file(
        path / manifest.archive_filename,
        maximum=limits.compressed_bytes,
    )
    if (
        f"sha256:{digest}" != manifest.archive_sha256
        or size != manifest.compressed_bytes
    ):
        raise ArchiveIntegrityError("published monthly archive bytes are inconsistent")
    acquisition = _read_acquisition(path / "acquisition.json")
    return PublishedMonthlyArchive(
        path,
        manifest,
        acquisition.availability_report_id,
        True,
    )


def _admit_symbol_filter(
    report: ArchiveAvailabilityReport,
    symbols: tuple[str, ...],
) -> frozenset[str] | None:
    if not symbols:
        return None
    if len(symbols) != len(set(symbols)):
        raise ArchiveSchemaError("monthly symbol filter contains duplicates")
    report_symbols = {probe.symbol for probe in report.probes}
    unknown = sorted(set(symbols) - report_symbols)
    if unknown:
        raise ArchiveSchemaError(
            "monthly symbol filter is not present in the availability report: "
            + ", ".join(unknown)
        )
    return frozenset(symbols)


def _request_from_probe(
    report: ArchiveAvailabilityReport,
    probe: MonthlyArchiveProbe,
) -> MonthlyArchiveRequest:
    if probe.checksum_sha256 is None:
        raise ArchiveIntegrityError("AVAILABLE monthly probe has no checksum")
    try:
        month = date.fromisoformat(f"{probe.month}-01")
        instrument = InstrumentId(
            venue=VenueId("binance", "spot"),
            environment=Environment.PRODUCTION,
            symbol=probe.symbol,
        )
    except (ValueError, EvidenceAdmissionError) as error:
        raise ArchiveSchemaError("monthly availability identity is invalid") from error
    if month.strftime("%Y-%m") != probe.month:
        raise ArchiveSchemaError("monthly availability month is invalid")
    return MonthlyArchiveRequest(
        instrument=instrument,
        month=month,
        interval=report.interval,
        base_url=report.base_url,
        expected_sha256=probe.checksum_sha256.removeprefix("sha256:"),
    )


def _request_from_manifest(manifest: MonthlyArchiveManifest) -> MonthlyArchiveRequest:
    try:
        month = date.fromisoformat(f"{manifest.month}-01")
        instrument = InstrumentId(
            venue=VenueId(manifest.venue, manifest.market),
            environment=Environment(manifest.environment),
            symbol=manifest.symbol,
        )
    except (ValueError, EvidenceAdmissionError) as error:
        raise ArchiveSchemaError("published monthly identity is invalid") from error
    base_url = manifest.source_url.rsplit("/klines/", maxsplit=1)[0]
    return MonthlyArchiveRequest(
        instrument=instrument,
        month=month,
        interval=manifest.interval,
        base_url=base_url,
        expected_sha256=manifest.archive_sha256.removeprefix("sha256:"),
    )


async def _acquire_with_retries(
    request: MonthlyArchiveRequest,
    *,
    report: ArchiveAvailabilityReport,
    output_root: Path,
    ingestion_run_id: UUID,
    session: aiohttp.ClientSession,
    decoder: ProcessPoolExecutor | None,
    limits: MonthlyArchiveLimits,
    maximum_compressed_bytes: int,
) -> PublishedMonthlyArchive:
    for attempt in range(1, limits.retry_attempts + 1):
        try:
            return await _acquire_one(
                request,
                report=report,
                output_root=output_root,
                ingestion_run_id=ingestion_run_id,
                session=session,
                decoder=decoder,
                limits=limits,
                maximum_compressed_bytes=maximum_compressed_bytes,
            )
        except ArchiveTransportError:
            if attempt == limits.retry_attempts:
                raise
            delay = limits.retry_base_seconds * (2 ** (attempt - 1))
            if delay > 0:
                await sleep(delay)
    raise AssertionError("monthly retry loop did not return or raise")


async def _acquire_one(
    request: MonthlyArchiveRequest,
    *,
    report: ArchiveAvailabilityReport,
    output_root: Path,
    ingestion_run_id: UUID,
    session: aiohttp.ClientSession,
    decoder: ProcessPoolExecutor | None,
    limits: MonthlyArchiveLimits,
    maximum_compressed_bytes: int,
) -> PublishedMonthlyArchive:
    target = _publication_path(output_root, request)
    if target.exists():
        manifest = await to_thread(
            _load_existing,
            target,
            request=request,
            limits=limits,
        )
        if manifest.compressed_bytes > maximum_compressed_bytes:
            raise ArchiveResourceError(
                "monthly archive exceeds remaining batch byte budget"
            )
        return PublishedMonthlyArchive(target, manifest, report.report_id, True)

    with _artifacts.publication_staging_directory(
        output_root / ".staging",
        prefix="monthly-archive-",
    ) as staging:
        archive_path = staging / request.archive_filename
        archive_sha256, compressed_bytes = await download_file(
            session,
            request.source_url,
            archive_path,
            maximum=maximum_compressed_bytes,
            chunk_bytes=limits.chunk_bytes,
            description="monthly archive",
            limit_description="monthly archive",
        )
        if archive_sha256 != request.expected_sha256:
            raise ArchiveIntegrityError(
                "monthly archive SHA-256 does not match availability CHECKSUM"
            )

        receipt = LocalReceipt(time_ns(), monotonic_ns())
        inspect = partial(
            _inspect_archive,
            request,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            compressed_bytes=compressed_bytes,
            receipt=receipt,
            ingestion_run_id=ingestion_run_id,
            limits=limits,
            collect_events=False,
        )
        if decoder is None:
            decoded = await to_thread(inspect)
        else:
            decoded = await get_running_loop().run_in_executor(decoder, inspect)
        manifest_bytes = _artifacts.canonical_json(decoded.manifest.to_mapping())
        _write_bytes(staging / "manifest.json", manifest_bytes)
        _write_bytes(
            staging / "acquisition.json",
            _artifacts.canonical_json(
                {
                    "availability_probed_at_ns": report.probed_at_ns,
                    "availability_report_id": report.report_id,
                    "download_completed_monotonic_ns": receipt.monotonic_ns,
                    "download_completed_wall_time_ns": receipt.wall_time_ns,
                    "ingestion_run_id": str(ingestion_run_id),
                    "pool_id": report.pool_id,
                    "schema_version": 1,
                    "wall_time_unit": "ns",
                }
            ),
        )
        already_present = _publish_directory(
            staging,
            target,
            expected_manifest=manifest_bytes,
        )
        return PublishedMonthlyArchive(
            target,
            decoded.manifest,
            report.report_id,
            already_present,
        )


def _inspect_archive(
    request: MonthlyArchiveRequest,
    *,
    archive_path: Path,
    archive_sha256: str,
    compressed_bytes: int,
    receipt: LocalReceipt,
    ingestion_run_id: UUID,
    limits: MonthlyArchiveLimits,
    collect_events: bool,
) -> _DecodedMonthlyArchive:
    started = monotonic()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = admit_single_archive_member(
                archive,
                expected_filename=request.member_filename,
                maximum_bytes=limits.uncompressed_bytes,
                subject="monthly archive",
            )
            member_sha256, uncompressed_bytes = _hash_member(
                archive,
                member,
                maximum=limits.uncompressed_bytes,
            )
            _admit_decode_time(started, limits)
            (
                events,
                row_count,
                first_open_time_us,
                last_open_time_us,
                observed_days,
                gaps,
            ) = _decode_member(
                archive,
                member,
                request=request,
                member_sha256=member_sha256,
                receipt=receipt,
                ingestion_run_id=ingestion_run_id,
                archive_sha256=archive_sha256,
                limits=limits,
                started=started,
                collect_events=collect_events,
            )
    except ArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ArchiveIntegrityError("monthly archive ZIP is invalid") from error

    observed = set(observed_days)
    missing_days = tuple(
        day for day in _calendar_days(request.month) if day not in observed
    )
    manifest = MonthlyArchiveManifest(
        schema_version=1,
        decoder_version=DECODER_VERSION,
        venue=request.instrument.venue.name,
        market=request.instrument.venue.market,
        environment=request.instrument.environment.value,
        dataset="klines",
        symbol=request.instrument.symbol,
        interval=request.interval,
        month=request.month_text,
        timestamp_unit=archive_timestamp_unit(request.month).value,
        source_url=request.source_url,
        checksum_url=f"{request.source_url}.CHECKSUM",
        archive_filename=request.archive_filename,
        member_filename=request.member_filename,
        source_revision=f"sha256:{archive_sha256}",
        archive_sha256=f"sha256:{archive_sha256}",
        member_sha256=f"sha256:{member_sha256}",
        compressed_bytes=compressed_bytes,
        uncompressed_bytes=uncompressed_bytes,
        row_count=row_count,
        first_open_time_us=first_open_time_us,
        last_open_time_us=last_open_time_us,
        observed_first_day=observed_days[0],
        observed_last_day=observed_days[-1],
        observed_days=observed_days,
        missing_days=missing_days,
        internal_gaps=gaps,
    )
    return _DecodedMonthlyArchive(manifest, events)


def _decode_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    request: MonthlyArchiveRequest,
    member_sha256: str,
    receipt: LocalReceipt,
    ingestion_run_id: UUID,
    archive_sha256: str,
    limits: MonthlyArchiveLimits,
    started: float,
    collect_events: bool,
) -> tuple[
    tuple[KlineEvent, ...],
    int,
    int,
    int,
    tuple[str, ...],
    tuple[MonthlyGap, ...],
]:
    timestamp_unit = archive_timestamp_unit(request.month)
    source = SourceDescriptor(
        endpoint=request.source_url,
        channel=f"spot/monthly/klines/{request.interval}",
        schema_version=1,
    )
    provenance = Provenance(
        source=source,
        ingestion_run_id=ingestion_run_id,
        receipt=receipt,
        payload_digest=PayloadDigest(f"sha256:{member_sha256}"),
        source_revision=f"sha256:{archive_sha256}",
        raw_payload_reference=request.member_filename,
    )
    quality = ObservationQuality(QualityState.VALID, complete=True)
    window_start_us = (request.month - date(1970, 1, 1)).days * _MICROSECONDS_PER_DAY
    window_end_us = (
        _next_month(request.month) - date(1970, 1, 1)
    ).days * _MICROSECONDS_PER_DAY

    first_open_time_us: int | None = None
    last_open_time_us: int | None = None
    row_count = 0
    observed_days: list[str] = []
    gaps: list[MonthlyGap] = []
    events: list[KlineEvent] = []
    try:
        with archive.open(member) as stream:
            for row_number, raw_line in enumerate(stream, start=1):
                if row_number > limits.rows:
                    raise ArchiveSchemaError("monthly archive row count exceeds limit")
                try:
                    text = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ArchiveSchemaError(
                        "monthly archive member is not UTF-8"
                    ) from error
                decoded_rows = list(csv.reader([text], strict=True))
                if len(decoded_rows) != 1:
                    raise ArchiveSchemaError(
                        "monthly archive row has an invalid physical grain"
                    )
                event = decode_binance_kline_row(
                    decoded_rows[0],
                    row_number=row_number,
                    instrument=request.instrument,
                    interval=request.interval,
                    member_filename=request.member_filename,
                    timestamp_unit=timestamp_unit,
                    provenance=provenance,
                    payload_digest=PayloadDigest.sha256(raw_line),
                    quality=quality,
                )
                open_time_us = event.open_time.epoch_microseconds
                validate_one_minute_kline_time(
                    event,
                    window_start_us=window_start_us,
                    window_end_us=window_end_us,
                    previous_open_time_us=last_open_time_us,
                    window_name="requested month",
                )
                if last_open_time_us is not None:
                    first_missing = last_open_time_us + _MICROSECONDS_PER_MINUTE
                    if open_time_us > first_missing:
                        gaps.append(
                            MonthlyGap(
                                first_missing,
                                open_time_us - _MICROSECONDS_PER_MINUTE,
                                (open_time_us - last_open_time_us)
                                // _MICROSECONDS_PER_MINUTE
                                - 1,
                            )
                        )
                if first_open_time_us is None:
                    first_open_time_us = open_time_us
                last_open_time_us = open_time_us
                row_count = row_number
                day_text = (
                    date(1970, 1, 1)
                    + timedelta(days=open_time_us // _MICROSECONDS_PER_DAY)
                ).isoformat()
                if not observed_days or observed_days[-1] != day_text:
                    observed_days.append(day_text)
                if collect_events:
                    events.append(event)
                if row_number % 256 == 0:
                    _admit_decode_time(started, limits)
    except csv.Error as error:
        raise ArchiveSchemaError("monthly archive CSV is malformed") from error
    if row_count == 0 or first_open_time_us is None or last_open_time_us is None:
        raise ArchiveSchemaError("monthly archive contains no kline rows")
    _admit_decode_time(started, limits)
    return (
        tuple(events),
        row_count,
        first_open_time_us,
        last_open_time_us,
        tuple(observed_days),
        tuple(gaps),
    )


def _hash_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    maximum: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member) as stream:
        while chunk := stream.read(64 * 1_024):
            size += len(chunk)
            if size > maximum:
                raise ArchiveIntegrityError("monthly archive member exceeds byte limit")
            digest.update(chunk)
    if size != member.file_size:
        raise ArchiveIntegrityError("monthly archive member size is inconsistent")
    return digest.hexdigest(), size


def _load_existing(
    target: Path,
    *,
    request: MonthlyArchiveRequest,
    limits: MonthlyArchiveLimits,
) -> MonthlyArchiveManifest:
    manifest = _read_manifest(target / "manifest.json")
    if (
        manifest.venue != request.instrument.venue.name
        or manifest.market != request.instrument.venue.market
        or manifest.environment != request.instrument.environment.value
        or manifest.symbol != request.instrument.symbol
        or manifest.interval != request.interval
        or manifest.month != request.month_text
        or manifest.source_url != request.source_url
        or manifest.checksum_url != f"{request.source_url}.CHECKSUM"
        or manifest.archive_filename != request.archive_filename
        or manifest.member_filename != request.member_filename
        or manifest.archive_sha256 != f"sha256:{request.expected_sha256}"
        or manifest.source_revision != f"sha256:{request.expected_sha256}"
    ):
        raise ArchiveIntegrityError(
            "existing monthly archive does not match the requested revision"
        )
    if (
        manifest.compressed_bytes > limits.compressed_bytes
        or manifest.uncompressed_bytes > limits.uncompressed_bytes
        or manifest.row_count > limits.rows
    ):
        raise ArchiveIntegrityError("existing monthly archive exceeds resource limits")
    digest, size = _hash_file(
        target / manifest.archive_filename,
        maximum=limits.compressed_bytes,
    )
    if digest != request.expected_sha256 or size != manifest.compressed_bytes:
        raise ArchiveIntegrityError(
            "existing monthly archive bytes do not match the requested revision"
        )
    _read_acquisition(target / "acquisition.json")
    return manifest


def _read_manifest(path: Path) -> MonthlyArchiveManifest:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "monthly archive manifest is unavailable"
        ) from error
    if len(payload) > 8 * 1024 * 1024:
        raise ArchiveIntegrityError("monthly archive manifest exceeds byte limit")
    try:
        manifest = _MANIFEST_DECODER.decode(payload)
        if payload != _artifacts.canonical_json(manifest.to_mapping()):
            raise ValueError
    except (msgspec.DecodeError, ValueError) as error:
        raise ArchiveIntegrityError("monthly archive manifest is invalid") from error
    return manifest


def _read_acquisition(path: Path) -> _Acquisition:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "monthly archive acquisition record is unavailable"
        ) from error
    if len(payload) > 8_192:
        raise ArchiveIntegrityError("monthly acquisition record exceeds byte limit")
    try:
        document = _ACQUISITION_DECODER.decode(payload)
        if document.schema_version != 1 or document.wall_time_unit != "ns":
            raise ValueError
        acquisition = _Acquisition(
            receipt=LocalReceipt(
                document.download_completed_wall_time_ns,
                document.download_completed_monotonic_ns,
            ),
            ingestion_run_id=UUID(document.ingestion_run_id),
            availability_report_id=document.availability_report_id,
            pool_id=document.pool_id,
            availability_probed_at_ns=document.availability_probed_at_ns,
        )
        expected: dict[str, object] = {
            "availability_probed_at_ns": acquisition.availability_probed_at_ns,
            "availability_report_id": acquisition.availability_report_id,
            "download_completed_monotonic_ns": acquisition.receipt.monotonic_ns,
            "download_completed_wall_time_ns": acquisition.receipt.wall_time_ns,
            "ingestion_run_id": str(acquisition.ingestion_run_id),
            "pool_id": acquisition.pool_id,
            "schema_version": 1,
            "wall_time_unit": "ns",
        }
        if (
            not _artifacts.is_sha256(acquisition.availability_report_id)
            or not _artifacts.is_sha256(acquisition.pool_id)
            or payload != _artifacts.canonical_json(expected)
        ):
            raise ValueError
    except (EvidenceAdmissionError, msgspec.DecodeError, ValueError) as error:
        raise ArchiveIntegrityError(
            "monthly archive acquisition record is invalid"
        ) from error
    return acquisition


def _prepare_root(output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / ".staging").mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArchivePublicationError(
            "monthly archive staging root is unavailable"
        ) from error


def _publication_path(output_root: Path, request: MonthlyArchiveRequest) -> Path:
    return (
        output_root
        / "binance"
        / "spot"
        / "monthly-klines"
        / request.instrument.symbol
        / request.interval
        / request.month_text
        / request.expected_sha256
    )


def _publish_directory(
    staging: Path,
    target: Path,
    *,
    expected_manifest: bytes,
) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArchivePublicationError("monthly publication path failed") from error
    try:
        return _artifacts.adopt_directory(
            staging,
            target,
            verify_existing=lambda existing: _verify_existing_manifest(
                existing, expected_manifest
            ),
        )
    except OSError as error:
        raise ArchivePublicationError(
            "atomic monthly archive publication failed"
        ) from error


def _verify_existing_manifest(target: Path, expected_manifest: bytes) -> None:
    try:
        existing = (target / "manifest.json").read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "monthly archive has no readable manifest"
        ) from error
    if existing != expected_manifest:
        raise ArchivePublicationError("monthly archive manifest conflicts with content")


def _hash_file(path: Path, *, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1_024):
                size += len(chunk)
                if size > maximum:
                    raise ArchiveIntegrityError(
                        "published monthly archive exceeds byte limit"
                    )
                digest.update(chunk)
    except OSError as error:
        raise ArchivePublicationError("monthly archive is unavailable") from error
    return digest.hexdigest(), size


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise ArchivePublicationError("monthly metadata write failed") from error


def _calendar_days(month: date) -> tuple[str, ...]:
    end = _next_month(month)
    values: list[str] = []
    current = month
    while current < end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def _next_month(month: date) -> date:
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)


def _admit_decode_time(started: float, limits: MonthlyArchiveLimits) -> None:
    if monotonic() - started > limits.decode_timeout_seconds:
        raise ArchiveResourceError("monthly archive decode exceeds time limit")


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
