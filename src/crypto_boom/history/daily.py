"""Bounded acquisition and atomic publication of official kline archives."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import zipfile
from asyncio import sleep
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import monotonic, monotonic_ns, time_ns
from uuid import UUID

import aiohttp
import msgspec

from crypto_boom import _artifacts
from crypto_boom.binance_source import BINANCE_SPOT
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

OFFICIAL_ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot/daily"
MAXIMUM_ARCHIVE_RANGE_INSTRUMENTS = 500

_CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+[*]?([^\r\n]+)")
_MICROSECONDS_PER_DAY = 86_400_000_000


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Resource bounds for one daily one-minute kline archive."""

    checksum_bytes: int = 1_024
    compressed_bytes: int = 8 * 1_024 * 1_024
    uncompressed_bytes: int = 32 * 1_024 * 1_024
    rows: int = 1_440
    request_timeout_seconds: float = 60.0
    decode_timeout_seconds: float = 10.0
    chunk_bytes: int = 64 * 1_024

    def __post_init__(self) -> None:
        values = (
            self.checksum_bytes,
            self.compressed_bytes,
            self.uncompressed_bytes,
            self.rows,
            self.chunk_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("archive byte and row limits must be positive")
        if self.request_timeout_seconds <= 0 or self.decode_timeout_seconds <= 0:
            raise ValueError("archive request and decode timeouts must be positive")


DEFAULT_ARCHIVE_LIMITS = ArchiveLimits()


@dataclass(frozen=True, slots=True)
class ArchiveBatchLimits:
    """Scheduling bounds for a deterministic daily-archive range."""

    maximum_partitions: int = 10_000
    retry_attempts: int = 4
    retry_base_seconds: float = 1.0
    request_spacing_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.maximum_partitions <= 0 or self.retry_attempts <= 0:
            raise ValueError("archive batch counts must be positive")
        if self.retry_base_seconds < 0 or self.request_spacing_seconds < 0:
            raise ValueError("archive batch delays cannot be negative")


DEFAULT_ARCHIVE_BATCH_LIMITS = ArchiveBatchLimits()


@dataclass(frozen=True, slots=True)
class ArchiveDayRequest:
    """One supported immutable historical-source request."""

    instrument: InstrumentId
    day: date
    interval: str = "1m"

    def __post_init__(self) -> None:
        if self.instrument.venue != BINANCE_SPOT:
            raise ArchiveSchemaError("daily archive venue must be Binance Spot")
        if self.instrument.environment is not Environment.PRODUCTION:
            raise ArchiveSchemaError("daily archive environment must be production")
        if self.interval != "1m":
            raise ArchiveSchemaError("ARC-01 admits only one-minute kline archives")

    @property
    def stem(self) -> str:
        return f"{self.instrument.symbol}-{self.interval}-{self.day.isoformat()}"

    @property
    def archive_filename(self) -> str:
        return f"{self.stem}.zip"

    @property
    def member_filename(self) -> str:
        return f"{self.stem}.csv"

    def source_url(self, base_url: str) -> str:
        base = base_url.rstrip("/")
        return (
            f"{base}/klines/{self.instrument.symbol}/{self.interval}/"
            f"{self.archive_filename}"
        )


@dataclass(frozen=True, slots=True)
class ArchiveRangeRequest:
    """A bounded symbol set and inclusive daily range."""

    instruments: tuple[InstrumentId, ...]
    start_day: date
    end_day: date
    interval: str = "1m"

    def __post_init__(self) -> None:
        if not self.instruments:
            raise ArchiveSchemaError("archive range requires at least one instrument")
        if len(self.instruments) > MAXIMUM_ARCHIVE_RANGE_INSTRUMENTS:
            raise ArchiveResourceError("archive range cannot exceed 500 instruments")
        if self.end_day < self.start_day:
            raise ArchiveSchemaError("archive range end precedes its start")
        symbols: set[str] = set()
        for instrument in self.instruments:
            ArchiveDayRequest(instrument, self.start_day, self.interval)
            if instrument.symbol in symbols:
                raise ArchiveSchemaError("archive range instruments must be unique")
            symbols.add(instrument.symbol)

    @property
    def partition_count(self) -> int:
        day_count = (self.end_day - self.start_day).days + 1
        return len(self.instruments) * day_count

    def daily_requests(self) -> Iterator[ArchiveDayRequest]:
        for instrument in sorted(
            self.instruments,
            key=lambda item: item.symbol,
        ):
            day = self.start_day
            while day <= self.end_day:
                yield ArchiveDayRequest(instrument, day, self.interval)
                if day == self.end_day:
                    break
                day += timedelta(days=1)


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """Deterministic content/source revision produced by one validated archive."""

    schema_version: int
    decoder_version: str
    venue: str
    market: str
    environment: str
    dataset: str
    symbol: str
    interval: str
    day: str
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

    def _content_mapping(self) -> dict[str, object]:
        return {
            "archive_filename": self.archive_filename,
            "archive_sha256": self.archive_sha256,
            "checksum_url": self.checksum_url,
            "compressed_bytes": self.compressed_bytes,
            "dataset": self.dataset,
            "day": self.day,
            "decoder_version": self.decoder_version,
            "environment": self.environment,
            "first_open_time_us": self.first_open_time_us,
            "interval": self.interval,
            "last_open_time_us": self.last_open_time_us,
            "market": self.market,
            "member_filename": self.member_filename,
            "member_sha256": self.member_sha256,
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
        return (
            "sha256:"
            + hashlib.sha256(
                _artifacts.canonical_json(self._content_mapping())
            ).hexdigest()
        )

    def to_mapping(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self._content_mapping()}


_MANIFEST_DECODER = msgspec.json.Decoder(ArchiveManifest)


@dataclass(frozen=True, slots=True)
class PublishedArchive:
    """Observed result of one acquisition attempt."""

    path: Path
    manifest: ArchiveManifest
    already_present: bool


@dataclass(frozen=True, slots=True)
class ArchiveRangeResult:
    """Completed publications from one bounded range."""

    publications: tuple[PublishedArchive, ...]

    @property
    def partition_count(self) -> int:
        return len(self.publications)

    @property
    def downloaded_count(self) -> int:
        return sum(not item.already_present for item in self.publications)

    @property
    def reused_count(self) -> int:
        return sum(item.already_present for item in self.publications)


@dataclass(frozen=True, slots=True)
class _DecodedArchive:
    manifest: ArchiveManifest
    events: tuple[KlineEvent, ...]


async def acquire_archive_day(
    request: ArchiveDayRequest,
    *,
    output_root: Path,
    ingestion_run_id: UUID,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
    base_url: str = OFFICIAL_ARCHIVE_BASE_URL,
) -> PublishedArchive:
    """Download, validate, and atomically publish one official archive revision."""

    output_root = output_root.resolve()
    _prepare_archive_root(output_root)
    timeout = aiohttp.ClientTimeout(total=limits.request_timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout,
        auto_decompress=False,
    ) as session:
        return await _acquire_archive_day_with_session(
            request,
            output_root=output_root,
            ingestion_run_id=ingestion_run_id,
            session=session,
            limits=limits,
            base_url=base_url,
        )


async def acquire_archive_range(
    request: ArchiveRangeRequest,
    *,
    output_root: Path,
    ingestion_run_id: UUID,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
    batch_limits: ArchiveBatchLimits = DEFAULT_ARCHIVE_BATCH_LIMITS,
    base_url: str = OFFICIAL_ARCHIVE_BASE_URL,
    on_published: Callable[[PublishedArchive], None] | None = None,
) -> ArchiveRangeResult:
    """Acquire a deterministic range sequentially with bounded retries."""

    partition_count = request.partition_count
    if partition_count > batch_limits.maximum_partitions:
        raise ArchiveResourceError(
            f"archive range has {partition_count} partitions; "
            f"limit is {batch_limits.maximum_partitions}"
        )

    output_root = output_root.resolve()
    _prepare_archive_root(output_root)
    timeout = aiohttp.ClientTimeout(total=limits.request_timeout_seconds)
    publications: list[PublishedArchive] = []
    async with aiohttp.ClientSession(
        timeout=timeout,
        auto_decompress=False,
    ) as session:
        for partition_index, day_request in enumerate(request.daily_requests()):
            try:
                publication = await _acquire_with_retries(
                    day_request,
                    output_root=output_root,
                    ingestion_run_id=ingestion_run_id,
                    session=session,
                    limits=limits,
                    batch_limits=batch_limits,
                    base_url=base_url,
                )
            except ArchiveError as error:
                raise ArchiveBatchError(
                    "archive range failed at "
                    f"{day_request.instrument.symbol} {day_request.day.isoformat()}: "
                    f"{error}"
                ) from error
            publications.append(publication)
            if on_published is not None:
                on_published(publication)
            if (
                partition_index + 1 < partition_count
                and batch_limits.request_spacing_seconds > 0
            ):
                await sleep(batch_limits.request_spacing_seconds)
    return ArchiveRangeResult(tuple(publications))


async def _acquire_with_retries(
    request: ArchiveDayRequest,
    *,
    output_root: Path,
    ingestion_run_id: UUID,
    session: aiohttp.ClientSession,
    limits: ArchiveLimits,
    batch_limits: ArchiveBatchLimits,
    base_url: str,
) -> PublishedArchive:
    for attempt in range(1, batch_limits.retry_attempts + 1):
        try:
            return await _acquire_archive_day_with_session(
                request,
                output_root=output_root,
                ingestion_run_id=ingestion_run_id,
                session=session,
                limits=limits,
                base_url=base_url,
            )
        except ArchiveTransportError:
            if attempt == batch_limits.retry_attempts:
                raise
            await sleep(batch_limits.retry_base_seconds * (2 ** (attempt - 1)))
    raise AssertionError("archive retry loop did not return or raise")


def _prepare_archive_root(output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / ".staging").mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArchivePublicationError("archive staging root is unavailable") from error


async def _acquire_archive_day_with_session(
    request: ArchiveDayRequest,
    *,
    output_root: Path,
    ingestion_run_id: UUID,
    session: aiohttp.ClientSession,
    limits: ArchiveLimits,
    base_url: str,
) -> PublishedArchive:
    source_url = request.source_url(base_url)
    checksum_url = f"{source_url}.CHECKSUM"
    checksum_document = await _download_bytes(
        session,
        checksum_url,
        maximum=limits.checksum_bytes,
    )
    expected_sha256 = _parse_checksum(
        checksum_document,
        expected_filename=request.archive_filename,
    )
    target = _publication_path(output_root, request, expected_sha256)
    if target.exists():
        return _load_existing_archive(
            target,
            request=request,
            expected_sha256=expected_sha256,
            source_url=source_url,
            checksum_url=checksum_url,
            limits=limits,
        )

    staging_parent = output_root / ".staging"
    with _artifacts.publication_staging_directory(
        staging_parent, prefix="arc-"
    ) as staging:
        archive_path = staging / request.archive_filename
        actual_sha256, compressed_bytes = await _download_file(
            session,
            source_url,
            archive_path,
            maximum=limits.compressed_bytes,
            chunk_bytes=limits.chunk_bytes,
        )

        if actual_sha256 != expected_sha256:
            raise ArchiveIntegrityError("archive SHA-256 does not match CHECKSUM")

        receipt = LocalReceipt(
            wall_time_ns=time_ns(),
            monotonic_ns=monotonic_ns(),
        )
        decoded = _inspect_kline_archive(
            request,
            archive_path=archive_path,
            source_url=source_url,
            checksum_url=checksum_url,
            archive_sha256=actual_sha256,
            compressed_bytes=compressed_bytes,
            ingestion_run_id=ingestion_run_id,
            receipt=receipt,
            limits=limits,
        )
        manifest = decoded.manifest
        manifest_bytes = _artifacts.canonical_json(manifest.to_mapping())
        _write_bytes(staging / "manifest.json", manifest_bytes)
        _write_bytes(
            staging / "acquisition.json",
            _artifacts.canonical_json(
                {
                    "download_completed_monotonic_ns": receipt.monotonic_ns,
                    "download_completed_wall_time_ns": receipt.wall_time_ns,
                    "ingestion_run_id": str(ingestion_run_id),
                    "schema_version": 1,
                    "wall_time_unit": "ns",
                }
            ),
        )

        target = _publication_path(output_root, request, actual_sha256)
        already_present = _publish_directory(
            staging,
            target,
            expected_manifest=manifest_bytes,
        )
        return PublishedArchive(
            path=target,
            manifest=manifest,
            already_present=already_present,
        )


def _load_existing_archive(
    target: Path,
    *,
    request: ArchiveDayRequest,
    expected_sha256: str,
    source_url: str,
    checksum_url: str,
    limits: ArchiveLimits,
) -> PublishedArchive:
    manifest = _read_archive_manifest(target / "manifest.json")
    expected_identity = (
        manifest.schema_version == 1
        and manifest.decoder_version == DECODER_VERSION
        and manifest.venue == request.instrument.venue.name
        and manifest.market == request.instrument.venue.market
        and manifest.environment == request.instrument.environment.value
        and manifest.dataset == "klines"
        and manifest.symbol == request.instrument.symbol
        and manifest.interval == request.interval
        and manifest.day == request.day.isoformat()
        and manifest.timestamp_unit == archive_timestamp_unit(request.day).value
        and manifest.source_url == source_url
        and manifest.checksum_url == checksum_url
        and manifest.archive_filename == request.archive_filename
        and manifest.member_filename == request.member_filename
        and manifest.source_revision == f"sha256:{expected_sha256}"
        and manifest.archive_sha256 == f"sha256:{expected_sha256}"
    )
    if not expected_identity:
        raise ArchiveIntegrityError(
            "existing archive revision does not match the requested partition"
        )
    if (
        manifest.compressed_bytes <= 0
        or manifest.compressed_bytes > limits.compressed_bytes
        or manifest.uncompressed_bytes <= 0
        or manifest.uncompressed_bytes > limits.uncompressed_bytes
        or manifest.row_count <= 0
        or manifest.row_count > limits.rows
    ):
        raise ArchiveIntegrityError("existing archive manifest exceeds resource limits")

    archive_sha256, compressed_bytes = _hash_published_archive(
        target / manifest.archive_filename,
        maximum=limits.compressed_bytes,
    )
    if (
        archive_sha256 != expected_sha256
        or compressed_bytes != manifest.compressed_bytes
    ):
        raise ArchiveIntegrityError(
            "existing archive bytes do not match the official revision"
        )
    _read_acquisition(target / "acquisition.json")
    return PublishedArchive(
        path=target,
        manifest=manifest,
        already_present=True,
    )


def _read_archive_manifest(path: Path) -> ArchiveManifest:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "existing archive revision has no readable manifest"
        ) from error
    if len(payload) > 65_536:
        raise ArchiveIntegrityError("archive manifest exceeds the byte limit")
    try:
        manifest = _MANIFEST_DECODER.decode(payload)
        if payload != _artifacts.canonical_json(manifest.to_mapping()):
            raise ValueError
    except (msgspec.DecodeError, ValueError) as error:
        raise ArchiveIntegrityError("archive manifest is invalid") from error
    return manifest


def load_published_klines(
    published: PublishedArchive,
    *,
    limits: ArchiveLimits = DEFAULT_ARCHIVE_LIMITS,
) -> tuple[KlineEvent, ...]:
    """Reverify and decode one immutable raw partition for downstream storage."""

    manifest = published.manifest
    _verify_existing_manifest(
        published.path,
        _artifacts.canonical_json(manifest.to_mapping()),
    )
    archive_path = published.path / manifest.archive_filename
    archive_sha256, compressed_bytes = _hash_published_archive(
        archive_path,
        maximum=limits.compressed_bytes,
    )
    if f"sha256:{archive_sha256}" != manifest.archive_sha256:
        raise ArchiveIntegrityError("published archive SHA-256 does not match manifest")
    receipt, ingestion_run_id = _read_acquisition(published.path / "acquisition.json")
    try:
        request = ArchiveDayRequest(
            instrument=InstrumentId(
                venue=VenueId(manifest.venue, manifest.market),
                environment=Environment(manifest.environment),
                symbol=manifest.symbol,
            ),
            day=date.fromisoformat(manifest.day),
            interval=manifest.interval,
        )
    except (ValueError, EvidenceAdmissionError) as error:
        raise ArchiveSchemaError("published manifest identity is invalid") from error

    decoded = _inspect_kline_archive(
        request,
        archive_path=archive_path,
        source_url=manifest.source_url,
        checksum_url=manifest.checksum_url,
        archive_sha256=archive_sha256,
        compressed_bytes=compressed_bytes,
        ingestion_run_id=ingestion_run_id,
        receipt=receipt,
        limits=limits,
    )
    if decoded.manifest != manifest:
        raise ArchiveIntegrityError("published archive no longer matches its manifest")
    return decoded.events


async def _download_bytes(
    session: aiohttp.ClientSession,
    url: str,
    *,
    maximum: int,
) -> bytes:
    try:
        async with session.get(url) as response:
            _admit_response(response, maximum=maximum)
            payload = bytearray()
            async for chunk in response.content.iter_chunked(min(maximum, 16_384)):
                payload.extend(chunk)
                if len(payload) > maximum:
                    raise ArchiveTransportError("archive response exceeds byte limit")
            return bytes(payload)
    except ArchiveError:
        raise
    except (TimeoutError, aiohttp.ClientError) as error:
        raise ArchiveTransportError("archive request failed") from error


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    destination: Path,
    *,
    maximum: int,
    chunk_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        async with session.get(url) as response:
            _admit_response(response, maximum=maximum)
            with destination.open("xb") as stream:
                async for chunk in response.content.iter_chunked(chunk_bytes):
                    size += len(chunk)
                    if size > maximum:
                        raise ArchiveTransportError(
                            "archive response exceeds byte limit"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except ArchiveError:
        raise
    except (TimeoutError, aiohttp.ClientError) as error:
        raise ArchiveTransportError("archive request failed") from error
    except OSError as error:
        raise ArchivePublicationError("archive staging write failed") from error
    return digest.hexdigest(), size


def _admit_response(response: aiohttp.ClientResponse, *, maximum: int) -> None:
    if response.status != 200:
        raise ArchiveTransportError("archive endpoint returned a non-success status")
    if response.content_length is not None and response.content_length > maximum:
        raise ArchiveTransportError("archive response exceeds byte limit")


def _parse_checksum(document: bytes, *, expected_filename: str) -> str:
    try:
        text = document.decode("ascii")
    except UnicodeDecodeError as error:
        raise ArchiveIntegrityError("CHECKSUM is not ASCII") from error

    match = _CHECKSUM_LINE.fullmatch(text.strip())
    if match is None:
        raise ArchiveIntegrityError("CHECKSUM has an invalid format")
    digest, filename = match.groups()
    if filename != expected_filename:
        raise ArchiveIntegrityError("CHECKSUM filename does not match archive")
    return digest.lower()


def _inspect_kline_archive(
    request: ArchiveDayRequest,
    *,
    archive_path: Path,
    source_url: str,
    checksum_url: str,
    archive_sha256: str,
    compressed_bytes: int,
    ingestion_run_id: UUID,
    receipt: LocalReceipt,
    limits: ArchiveLimits,
) -> _DecodedArchive:
    decode_started = monotonic()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            member = admit_single_archive_member(
                archive,
                expected_filename=request.member_filename,
                maximum_bytes=limits.uncompressed_bytes,
                subject="archive",
            )
            with archive.open(member) as stream:
                member_bytes = stream.read(limits.uncompressed_bytes + 1)
    except ArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ArchiveIntegrityError("archive ZIP is invalid") from error

    if len(member_bytes) > limits.uncompressed_bytes:
        raise ArchiveIntegrityError("archive member exceeds byte limit")
    if len(member_bytes) != member.file_size:
        raise ArchiveIntegrityError("archive member size is inconsistent")
    _admit_decode_time(decode_started, limits)

    try:
        member_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArchiveSchemaError("archive member is not UTF-8") from error

    member_sha256 = hashlib.sha256(member_bytes).hexdigest()
    timestamp_unit = archive_timestamp_unit(request.day)
    source = SourceDescriptor(
        endpoint=source_url,
        channel=f"spot/daily/klines/{request.interval}",
        schema_version=1,
    )
    partition_provenance = Provenance(
        source=source,
        ingestion_run_id=ingestion_run_id,
        receipt=receipt,
        payload_digest=PayloadDigest(f"sha256:{member_sha256}"),
        source_revision=f"sha256:{archive_sha256}",
        raw_payload_reference=request.member_filename,
    )
    quality = ObservationQuality(state=QualityState.VALID, complete=True)

    first_open_time_us: int | None = None
    last_open_time_us: int | None = None
    row_count = 0
    events: list[KlineEvent] = []
    day_start_us = (request.day - date(1970, 1, 1)).days * _MICROSECONDS_PER_DAY
    try:
        for row_number, raw_line in enumerate(
            member_bytes.splitlines(keepends=True),
            start=1,
        ):
            if row_number > limits.rows:
                raise ArchiveSchemaError("archive row count exceeds limit")
            decoded_rows = list(
                csv.reader(
                    [raw_line.decode("utf-8")],
                    strict=True,
                )
            )
            if len(decoded_rows) != 1:
                raise ArchiveSchemaError("archive row has an invalid physical grain")
            event = decode_binance_kline_row(
                decoded_rows[0],
                row_number=row_number,
                instrument=request.instrument,
                interval=request.interval,
                member_filename=request.member_filename,
                timestamp_unit=timestamp_unit,
                provenance=partition_provenance,
                payload_digest=PayloadDigest.sha256(raw_line),
                quality=quality,
            )
            open_time_us = event.open_time.epoch_microseconds
            validate_one_minute_kline_time(
                event,
                window_start_us=day_start_us,
                window_end_us=day_start_us + _MICROSECONDS_PER_DAY,
                previous_open_time_us=last_open_time_us,
                window_name="requested day",
            )
            if first_open_time_us is None:
                first_open_time_us = open_time_us
            last_open_time_us = open_time_us
            row_count = row_number
            events.append(event)
            if row_number % 128 == 0:
                _admit_decode_time(decode_started, limits)
    except csv.Error as error:
        raise ArchiveSchemaError("archive CSV is malformed") from error

    if row_count == 0 or first_open_time_us is None or last_open_time_us is None:
        raise ArchiveSchemaError("archive contains no kline rows")

    _admit_decode_time(decode_started, limits)
    return _DecodedArchive(
        manifest=ArchiveManifest(
            schema_version=1,
            decoder_version=DECODER_VERSION,
            venue=request.instrument.venue.name,
            market=request.instrument.venue.market,
            environment=request.instrument.environment.value,
            dataset="klines",
            symbol=request.instrument.symbol,
            interval=request.interval,
            day=request.day.isoformat(),
            timestamp_unit=timestamp_unit.value,
            source_url=source_url,
            checksum_url=checksum_url,
            archive_filename=request.archive_filename,
            member_filename=request.member_filename,
            source_revision=f"sha256:{archive_sha256}",
            archive_sha256=f"sha256:{archive_sha256}",
            member_sha256=f"sha256:{member_sha256}",
            compressed_bytes=compressed_bytes,
            uncompressed_bytes=len(member_bytes),
            row_count=row_count,
            first_open_time_us=first_open_time_us,
            last_open_time_us=last_open_time_us,
        ),
        events=tuple(events),
    )


def _admit_decode_time(started: float, limits: ArchiveLimits) -> None:
    if monotonic() - started > limits.decode_timeout_seconds:
        raise ArchiveResourceError("archive decode exceeds time limit")


def _publication_path(
    output_root: Path,
    request: ArchiveDayRequest,
    archive_sha256: str,
) -> Path:
    return (
        output_root
        / request.instrument.venue.name
        / request.instrument.venue.market
        / "klines"
        / request.instrument.symbol
        / request.interval
        / request.day.isoformat()
        / archive_sha256
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
        raise ArchivePublicationError("archive publication path failed") from error
    try:
        return _artifacts.adopt_directory(
            staging,
            target,
            verify_existing=lambda existing: _verify_existing_manifest(
                existing, expected_manifest
            ),
        )
    except ArchiveError:
        raise
    except OSError as error:
        raise ArchivePublicationError("atomic archive publication failed") from error


def _verify_existing_manifest(target: Path, expected_manifest: bytes) -> None:
    try:
        existing_manifest = (target / "manifest.json").read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "existing archive revision has no readable manifest"
        ) from error
    if existing_manifest != expected_manifest:
        raise ArchivePublicationError(
            "existing archive revision manifest conflicts with staged content"
        )


def _hash_published_archive(path: Path, *, maximum: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1_024):
                size += len(chunk)
                if size > maximum:
                    raise ArchiveIntegrityError(
                        "published archive exceeds the byte limit"
                    )
                digest.update(chunk)
    except ArchiveError:
        raise
    except OSError as error:
        raise ArchivePublicationError("published archive is unavailable") from error
    return digest.hexdigest(), size


class _AcquisitionDocument(msgspec.Struct):
    schema_version: int
    wall_time_unit: str
    download_completed_wall_time_ns: int
    download_completed_monotonic_ns: int
    ingestion_run_id: str


_ACQUISITION_DECODER = msgspec.json.Decoder(_AcquisitionDocument)


def _read_acquisition(path: Path) -> tuple[LocalReceipt, UUID]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArchivePublicationError(
            "archive acquisition record is unavailable"
        ) from error
    if len(payload) > 4_096:
        raise ArchiveIntegrityError("archive acquisition record exceeds the byte limit")
    try:
        document = _ACQUISITION_DECODER.decode(payload)
        if document.schema_version != 1 or document.wall_time_unit != "ns":
            raise ValueError
        receipt = LocalReceipt(
            document.download_completed_wall_time_ns,
            document.download_completed_monotonic_ns,
        )
        ingestion_run_id = UUID(document.ingestion_run_id)
    except (EvidenceAdmissionError, msgspec.DecodeError, ValueError) as error:
        raise ArchiveIntegrityError("archive acquisition record is invalid") from error
    return receipt, ingestion_run_id


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise ArchivePublicationError(
            "archive staging metadata write failed"
        ) from error
