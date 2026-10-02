"""Source-faithful Parquet materialization for bounded historical research."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Final

import msgspec
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_boom import _artifacts
from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
)
from crypto_boom.history.codec import ArchiveError
from crypto_boom.history.monthly import (
    PublishedMonthlyArchive,
    load_published_monthly_archive,
    load_published_monthly_klines_by_day,
)
from crypto_boom.market import (
    Environment,
    InstrumentId,
    KlineEvent,
    TimeWindow,
    VenueId,
)

RESEARCH_STORAGE_VERSION: Final = "research-source-kline-parquet-v1"
_SHA256_PREFIX = "sha256:"
_DECIMAL_TYPE = pa.decimal128(38, 18)


class ResearchStorageError(RuntimeError):
    """Verified source data could not be materialized for research safely."""


class ResearchStorageIntegrityError(ResearchStorageError):
    """A research source or derived partition is inconsistent."""


class ResearchStorageResourceError(ResearchStorageError):
    """Research materialization exceeded its admitted resource envelope."""


class ResearchStoragePublicationError(ResearchStorageError):
    """A research partition could not be published atomically."""


RESEARCH_KLINE_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("environment", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("close_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("raw_open_time", pa.int64(), nullable=False),
        pa.field("raw_close_time", pa.int64(), nullable=False),
        pa.field("timestamp_unit", pa.string(), nullable=False),
        pa.field("open_price", _DECIMAL_TYPE, nullable=False),
        pa.field("high_price", _DECIMAL_TYPE, nullable=False),
        pa.field("low_price", _DECIMAL_TYPE, nullable=False),
        pa.field("close_price", _DECIMAL_TYPE, nullable=False),
        pa.field("base_volume", _DECIMAL_TYPE, nullable=False),
        pa.field("quote_turnover", _DECIMAL_TYPE, nullable=False),
        pa.field("taker_buy_base_volume", _DECIMAL_TYPE, nullable=False),
        pa.field("taker_buy_quote_turnover", _DECIMAL_TYPE, nullable=False),
        pa.field("trade_count", pa.int64(), nullable=False),
        pa.field("first_trade_id", pa.int64()),
        pa.field("last_trade_id", pa.int64()),
        pa.field("quality_state", pa.string(), nullable=False),
        pa.field("quality_complete", pa.bool_(), nullable=False),
        pa.field("diagnostics", pa.list_(pa.string()), nullable=False),
        pa.field("payload_digest", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("raw_payload_reference", pa.string(), nullable=False),
        pa.field("source_manifest_id", pa.string(), nullable=False),
        pa.field("decoder_version", pa.string(), nullable=False),
    ],
    metadata={
        b"crypto_boom.dataset": b"research_source_klines",
        b"crypto_boom.evidence_class": b"source_only",
        b"crypto_boom.schema_version": b"1",
    },
)


@dataclass(frozen=True, slots=True)
class ResearchCorpusLimits:
    """Bounds for one local materialization campaign."""

    maximum_archives: int = 500
    maximum_rows: int = 25_000_000
    maximum_source_uncompressed_bytes: int = 8 * 1_024 * 1_024 * 1_024
    maximum_parquet_bytes: int = 8 * 1_024 * 1_024 * 1_024
    maximum_elapsed_seconds: float = 300.0
    maximum_concurrency: int = 1

    def __post_init__(self) -> None:
        if (
            self.maximum_archives <= 0
            or self.maximum_rows <= 0
            or self.maximum_source_uncompressed_bytes <= 0
            or self.maximum_parquet_bytes <= 0
            or self.maximum_elapsed_seconds <= 0
        ):
            raise ValueError("research corpus limits must be positive")
        if type(self.maximum_concurrency) is not int or not (
            1 <= self.maximum_concurrency <= 2
        ):
            raise ValueError("research corpus concurrency must be 1 or 2")


DEFAULT_RESEARCH_CORPUS_LIMITS = ResearchCorpusLimits()


@dataclass(frozen=True, slots=True)
class ResearchPartitionManifest:
    schema_version: int
    storage_version: str
    evidence_class: str
    venue: str
    market: str
    environment: str
    dataset: str
    symbol: str
    interval: str
    month: str
    source_manifest_id: str
    source_revision: str
    decoder_version: str
    row_count: int
    first_open_time_us: int
    last_open_time_us: int
    parquet_sha256: str
    parquet_bytes: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.storage_version != RESEARCH_STORAGE_VERSION
            or self.evidence_class != "source_only"
            or self.venue != "binance"
            or self.market != "spot"
            or self.environment != "production"
            or self.dataset != "research_source_klines"
            or self.interval != "1m"
            or not self.symbol
            or not self.decoder_version
        ):
            raise ResearchStorageIntegrityError(
                "research partition manifest version is invalid"
            )
        try:
            month = date.fromisoformat(f"{self.month}-01")
        except ValueError as error:
            raise ResearchStorageIntegrityError(
                "research partition month is invalid"
            ) from error
        if month.strftime("%Y-%m") != self.month:
            raise ResearchStorageIntegrityError("research partition month is invalid")
        if not _artifacts.is_sha256(
            self.source_manifest_id
        ) or not _artifacts.is_sha256(self.source_revision):
            raise ResearchStorageIntegrityError(
                "research partition source identity is invalid"
            )
        if not _artifacts.is_sha256(self.parquet_sha256):
            raise ResearchStorageIntegrityError(
                "research partition Parquet identity is invalid"
            )
        if (
            self.row_count <= 0
            or self.first_open_time_us <= 0
            or self.last_open_time_us < self.first_open_time_us
            or self.parquet_bytes <= 0
        ):
            raise ResearchStorageIntegrityError("research partition extent is invalid")

    @property
    def manifest_id(self) -> str:
        return _artifacts.content_id(asdict(self))

    def to_mapping(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **asdict(self)}


_MANIFEST_DECODER = msgspec.json.Decoder(ResearchPartitionManifest)


@dataclass(frozen=True, slots=True)
class PublishedResearchPartition:
    path: Path
    manifest: ResearchPartitionManifest
    already_present: bool


@dataclass(frozen=True, slots=True)
class ResearchCorpusResult:
    availability_report_id: str
    publications: tuple[PublishedResearchPartition, ...]
    #: Symbol-months the caller declared undecodable, carried into the receipt so a reader can tell a
    #: corpus that is smaller than its audit from one whose audit was wrong.
    excluded_symbol_months: tuple[tuple[str, str], ...] = ()

    @property
    def archive_count(self) -> int:
        return len(self.publications)

    @property
    def row_count(self) -> int:
        return sum(item.manifest.row_count for item in self.publications)

    @property
    def parquet_bytes(self) -> int:
        return sum(item.manifest.parquet_bytes for item in self.publications)

    @property
    def created_count(self) -> int:
        return sum(not item.already_present for item in self.publications)

    @property
    def reused_count(self) -> int:
        return sum(item.already_present for item in self.publications)


@dataclass(frozen=True, slots=True)
class _MaterializationJob:
    index: int
    source: PublishedMonthlyArchive
    output_root: Path
    maximum_parquet_bytes: int


def materialize_research_corpus(
    report: ArchiveAvailabilityReport,
    *,
    monthly_root: Path,
    output_root: Path,
    symbols: tuple[str, ...] = (),
    excluded: frozenset[tuple[str, str]] = frozenset(),
    limits: ResearchCorpusLimits = DEFAULT_RESEARCH_CORPUS_LIMITS,
    on_published: Callable[[PublishedResearchPartition], None] | None = None,
) -> ResearchCorpusResult:
    """Materialize selected AVAILABLE monthly sources as immutable Parquet.

    `excluded` names symbol-months to leave undecoded although the availability audit reports them
    AVAILABLE. The audit describes what the venue publishes and stays untouched; a pair belongs here
    when the published archive exists but cannot be admitted at all - the venue serves an archive
    whose contents violate the kline contract, which the audit has no vocabulary for (protocol
    section 24.7). The two records are kept apart on purpose: writing the exclusion into the audit
    would make an availability observation that is not true.
    """

    logging.getLogger(__name__).info("Materializing canonical archive partitions")
    if report.unresolved_count:
        raise ResearchStorageIntegrityError(
            "research corpus cannot consume unresolved availability observations"
        )
    selected = _select_symbols(report, symbols)
    excluded = _select_excluded(report, excluded)
    probes = tuple(
        probe
        for probe in report.probes
        if probe.symbol in selected
        and probe.state is AvailabilityState.AVAILABLE
        and (probe.symbol, probe.month) not in excluded
    )
    if len(probes) > limits.maximum_archives:
        raise ResearchStorageResourceError(
            f"research corpus requires {len(probes)} archives; "
            f"limit is {limits.maximum_archives}"
        )

    started = monotonic()
    sources: list[PublishedMonthlyArchive] = []
    total_rows = 0
    total_source_bytes = 0
    monthly_root = monthly_root.resolve()
    for probe in probes:
        _check_elapsed(started, limits)
        if probe.checksum_sha256 is None:
            raise ResearchStorageIntegrityError(
                "available observation has no archive checksum"
            )
        path = (
            monthly_root
            / "binance"
            / "spot"
            / "monthly-klines"
            / probe.symbol
            / report.interval
            / probe.month
            / probe.checksum_sha256.removeprefix(_SHA256_PREFIX)
        )
        try:
            source = load_published_monthly_archive(path)
        except ArchiveError as error:
            raise ResearchStorageIntegrityError(
                f"verified monthly source is unavailable for "
                f"{probe.symbol} {probe.month}"
            ) from error
        manifest = source.manifest
        if (
            manifest.symbol != probe.symbol
            or manifest.month != probe.month
            or manifest.interval != report.interval
            or manifest.archive_sha256 != probe.checksum_sha256
        ):
            raise ResearchStorageIntegrityError(
                "monthly source conflicts with availability observation"
            )
        total_rows += manifest.row_count
        total_source_bytes += manifest.uncompressed_bytes
        if total_rows > limits.maximum_rows:
            raise ResearchStorageResourceError(
                f"research corpus requires {total_rows} rows; "
                f"limit is {limits.maximum_rows}"
            )
        if total_source_bytes > limits.maximum_source_uncompressed_bytes:
            raise ResearchStorageResourceError(
                "research corpus source bytes exceed the admitted limit"
            )
        sources.append(source)

    output_root = output_root.resolve()
    publications: list[PublishedResearchPartition] = []
    admitted_parquet_bytes = 0
    for group_start in range(0, len(sources), limits.maximum_concurrency):
        _check_elapsed(started, limits)
        group = sources[group_start : group_start + limits.maximum_concurrency]
        remaining_parquet_bytes = limits.maximum_parquet_bytes - admitted_parquet_bytes
        if remaining_parquet_bytes <= 0:
            raise ResearchStorageResourceError(
                "research corpus Parquet byte budget was exhausted before next partition"
            )
        per_partition_budget = remaining_parquet_bytes // len(group)
        if per_partition_budget <= 0:
            raise ResearchStorageResourceError(
                "research corpus remaining Parquet byte budget cannot admit "
                "the next concurrent group"
            )
        jobs = tuple(
            _MaterializationJob(
                group_start + offset,
                source,
                output_root,
                per_partition_budget,
            )
            for offset, source in enumerate(group)
        )
        if len(jobs) == 1:
            completed = (_materialize_job(jobs[0]),)
        else:
            completed = _materialize_parallel_group(
                jobs,
                deadline=started + limits.maximum_elapsed_seconds,
            )
        for _, publication in completed:
            publications.append(publication)
            admitted_parquet_bytes += publication.manifest.parquet_bytes
            if on_published is not None:
                on_published(publication)
    logging.getLogger(__name__).info(
        "Canonical materialization complete: %d partitions", len(publications)
    )
    return ResearchCorpusResult(
        report.report_id,
        tuple(publications),
        tuple(sorted(excluded)),
    )


def _materialize_parallel_group(
    jobs: tuple[_MaterializationJob, ...],
    *,
    deadline: float,
) -> tuple[tuple[int, PublishedResearchPartition], ...]:
    context = mp.get_context("spawn")
    pool = context.Pool(processes=len(jobs), maxtasksperchild=1)
    finished = False
    try:
        pending = [pool.apply_async(_materialize_job, (job,)) for job in jobs]
        pool.close()
        completed: list[tuple[int, PublishedResearchPartition]] = []
        while pending:
            if monotonic() >= deadline:
                raise ResearchStorageResourceError(
                    "research corpus elapsed-time budget was exhausted during "
                    "concurrent materialization"
                )
            ready = [result for result in pending if result.ready()]
            if not ready:
                sleep(min(0.01, max(0.0, deadline - monotonic())))
                continue
            for result in ready:
                completed.append(result.get())
                pending.remove(result)
        pool.join()
        finished = True
        return tuple(sorted(completed, key=lambda item: item[0]))
    finally:
        if not finished:
            pool.terminate()
            pool.join()


def _materialize_job(
    job: _MaterializationJob,
) -> tuple[int, PublishedResearchPartition]:
    return (
        job.index,
        _materialize_partition(
            job.source,
            output_root=job.output_root,
            maximum_parquet_bytes=job.maximum_parquet_bytes,
        ),
    )


def load_published_research_partition(path: Path) -> PublishedResearchPartition:
    """Strictly reload one source-only research Parquet partition."""

    path = path.resolve()
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        if len(manifest_bytes) > 64 * 1024:
            raise ResearchStorageIntegrityError(
                "research partition manifest exceeds its byte limit"
            )
        manifest = _MANIFEST_DECODER.decode(manifest_bytes)
        if manifest_bytes != _artifacts.canonical_json(manifest.to_mapping()):
            raise ValueError
    except ResearchStorageError:
        raise
    except (OSError, msgspec.DecodeError, ValueError) as error:
        raise ResearchStorageIntegrityError(
            "research partition manifest is invalid"
        ) from error

    expected_tail = (
        manifest.symbol,
        manifest.interval,
        manifest.month,
        manifest.source_manifest_id.removeprefix(_SHA256_PREFIX),
    )
    actual_tail = (*(part.name for part in path.parents[:3][::-1]), path.name)
    if actual_tail != expected_tail:
        raise ResearchStorageIntegrityError(
            "research partition path does not match its identity"
        )
    parquet_path = path / "klines.parquet"
    parquet_sha256, parquet_bytes = _file_identity(parquet_path)
    if (
        parquet_sha256 != manifest.parquet_sha256
        or parquet_bytes != manifest.parquet_bytes
    ):
        raise ResearchStorageIntegrityError(
            "research partition Parquet bytes are inconsistent"
        )
    try:
        parquet = pq.ParquetFile(parquet_path)
        if parquet.schema_arrow != RESEARCH_KLINE_SCHEMA:
            raise ResearchStorageIntegrityError(
                "research partition Parquet schema is inconsistent"
            )
        if parquet.metadata.num_rows != manifest.row_count:
            raise ResearchStorageIntegrityError(
                "research partition Parquet row count is inconsistent"
            )
    except ResearchStorageError:
        raise
    except (OSError, pa.ArrowException) as error:
        raise ResearchStorageIntegrityError(
            "research partition Parquet metadata is invalid"
        ) from error
    return PublishedResearchPartition(path, manifest, True)


def _materialize_partition(
    source: PublishedMonthlyArchive,
    *,
    output_root: Path,
    maximum_parquet_bytes: int,
) -> PublishedResearchPartition:
    source_manifest = source.manifest
    target = (
        output_root
        / source_manifest.venue
        / source_manifest.market
        / "research-source-klines"
        / source_manifest.symbol
        / source_manifest.interval
        / source_manifest.month
        / source_manifest.manifest_id.removeprefix(_SHA256_PREFIX)
    )
    if target.exists():
        existing = load_published_research_partition(target)
        if (
            existing.manifest.source_manifest_id != source_manifest.manifest_id
            or existing.manifest.source_revision != source_manifest.source_revision
            or existing.manifest.row_count != source_manifest.row_count
        ):
            raise ResearchStoragePublicationError(
                "existing research partition conflicts with monthly source"
            )
        if existing.manifest.parquet_bytes > maximum_parquet_bytes:
            raise ResearchStorageResourceError(
                "research partition exceeds the remaining Parquet byte budget"
            )
        return existing

    days = load_published_monthly_klines_by_day(source)
    table = _research_table(
        (event for day in days for event in day.events),
        source_manifest_id=source_manifest.manifest_id,
        source_revision=source_manifest.source_revision,
        decoder_version=source_manifest.decoder_version,
    )
    if table.num_rows != source_manifest.row_count:
        raise ResearchStorageIntegrityError(
            "decoded monthly rows disagree with the source manifest"
        )
    staging_parent = output_root / ".staging"
    try:
        staging_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ResearchStoragePublicationError(
            "research staging root is unavailable"
        ) from error

    try:
        with _artifacts.publication_staging_directory(
            staging_parent,
            prefix="research-source-",
        ) as staging:
            parquet_path = staging / "klines.parquet"
            pq.write_table(
                table,
                parquet_path,
                compression="zstd",
                version="2.6",
                data_page_version="2.0",
            )
            parquet_sha256, parquet_bytes = _file_identity(parquet_path)
            if parquet_bytes > maximum_parquet_bytes:
                raise ResearchStorageResourceError(
                    "research partition exceeds the remaining Parquet byte budget"
                )
            manifest = ResearchPartitionManifest(
                schema_version=1,
                storage_version=RESEARCH_STORAGE_VERSION,
                evidence_class="source_only",
                venue=source_manifest.venue,
                market=source_manifest.market,
                environment=source_manifest.environment,
                dataset="research_source_klines",
                symbol=source_manifest.symbol,
                interval=source_manifest.interval,
                month=source_manifest.month,
                source_manifest_id=source_manifest.manifest_id,
                source_revision=source_manifest.source_revision,
                decoder_version=source_manifest.decoder_version,
                row_count=source_manifest.row_count,
                first_open_time_us=source_manifest.first_open_time_us,
                last_open_time_us=source_manifest.last_open_time_us,
                parquet_sha256=parquet_sha256,
                parquet_bytes=parquet_bytes,
            )
            manifest_bytes = _artifacts.canonical_json(manifest.to_mapping())
            _write_bytes(staging / "manifest.json", manifest_bytes)
            already_present = _publish_directory(
                staging,
                target,
                expected_manifest=manifest_bytes,
                expected_parquet_sha256=parquet_sha256,
            )
    except ResearchStorageError:
        raise
    except (OSError, pa.ArrowException) as error:
        raise ResearchStoragePublicationError(
            "research Parquet publication failed"
        ) from error
    return PublishedResearchPartition(target, manifest, already_present)


def _research_table(
    events: Iterable[KlineEvent],
    *,
    source_manifest_id: str,
    source_revision: str,
    decoder_version: str,
) -> pa.Table:
    try:
        return pa.Table.from_pylist(
            [
                _event_mapping(
                    event,
                    source_manifest_id=source_manifest_id,
                    source_revision=source_revision,
                    decoder_version=decoder_version,
                )
                for event in events
            ],
            schema=RESEARCH_KLINE_SCHEMA,
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as error:
        raise ResearchStorageIntegrityError(
            "research values do not fit source schema version 1"
        ) from error


def _event_mapping(
    event: KlineEvent,
    *,
    source_manifest_id: str,
    source_revision: str,
    decoder_version: str,
) -> dict[str, object]:
    provenance = event.provenance
    if (
        provenance.source_revision != source_revision
        or provenance.raw_payload_reference is None
    ):
        raise ResearchStorageIntegrityError(
            "research row provenance conflicts with its monthly source"
        )
    return {
        "venue": event.instrument.venue.name,
        "market": event.instrument.venue.market,
        "environment": event.instrument.environment.value,
        "symbol": event.instrument.symbol,
        "interval": event.interval,
        "open_time": event.open_time.epoch_microseconds,
        "close_time": event.close_time.epoch_microseconds,
        "raw_open_time": event.open_time.raw_value,
        "raw_close_time": event.close_time.raw_value,
        "timestamp_unit": event.open_time.unit.value,
        "open_price": event.open_price,
        "high_price": event.high_price,
        "low_price": event.low_price,
        "close_price": event.close_price,
        "base_volume": event.base_volume,
        "quote_turnover": event.quote_turnover,
        "taker_buy_base_volume": event.taker_buy_base_volume,
        "taker_buy_quote_turnover": event.taker_buy_quote_turnover,
        "trade_count": event.trade_count,
        "first_trade_id": event.first_trade_id,
        "last_trade_id": event.last_trade_id,
        "quality_state": event.quality.state.value,
        "quality_complete": event.quality.complete,
        "diagnostics": list(event.quality.diagnostics),
        "payload_digest": provenance.payload_digest.value,
        "source_revision": provenance.source_revision,
        "raw_payload_reference": provenance.raw_payload_reference,
        "source_manifest_id": source_manifest_id,
        "decoder_version": decoder_version,
    }


def _select_symbols(
    report: ArchiveAvailabilityReport,
    requested: tuple[str, ...],
) -> frozenset[str]:
    report_symbols = {probe.symbol for probe in report.probes}
    if not requested:
        return frozenset(report_symbols)
    if len(requested) != len(set(requested)):
        raise ResearchStorageIntegrityError(
            "research corpus symbol filter contains duplicates"
        )
    unknown = sorted(set(requested) - report_symbols)
    if unknown:
        raise ResearchStorageIntegrityError(
            "research corpus symbol filter is not present in availability report: "
            + ", ".join(unknown)
        )
    return frozenset(requested)


def _select_excluded(
    report: ArchiveAvailabilityReport,
    excluded: frozenset[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """Validate the declared symbol-month exclusions against the availability report.

    A pair the report never mentions is a typo rather than an exclusion, and is refused instead of
    being silently ignored - an exclusion that matches nothing would look like a protection while
    protecting nothing.
    """
    if not excluded:
        return frozenset()
    observed = {(probe.symbol, probe.month) for probe in report.probes}
    unknown = sorted(pair for pair in excluded if pair not in observed)
    if unknown:
        raise ResearchStorageIntegrityError(
            "research corpus exclusion is not present in availability report: "
            + ", ".join(f"{symbol} {month}" for symbol, month in unknown)
        )
    return frozenset(excluded)


def _check_elapsed(started: float, limits: ResearchCorpusLimits) -> None:
    if monotonic() - started >= limits.maximum_elapsed_seconds:
        raise ResearchStorageResourceError(
            "research corpus elapsed-time budget was exhausted before next partition"
        )


def _publish_directory(
    staging: Path,
    target: Path,
    *,
    expected_manifest: bytes,
    expected_parquet_sha256: str,
) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ResearchStoragePublicationError(
            "research publication path failed"
        ) from error
    try:
        return _artifacts.adopt_directory(
            staging,
            target,
            verify_existing=lambda existing: _verify_existing(
                existing,
                expected_manifest,
                expected_parquet_sha256,
            ),
        )
    except ResearchStorageError:
        raise
    except OSError as error:
        raise ResearchStoragePublicationError(
            "atomic research publication failed"
        ) from error


def _verify_existing(
    target: Path,
    expected_manifest: bytes,
    expected_parquet_sha256: str,
) -> None:
    try:
        manifest = (target / "manifest.json").read_bytes()
    except OSError as error:
        raise ResearchStoragePublicationError(
            "existing research partition has no readable manifest"
        ) from error
    parquet_sha256, _ = _file_identity(target / "klines.parquet")
    if manifest != expected_manifest or parquet_sha256 != expected_parquet_sha256:
        raise ResearchStoragePublicationError(
            "existing research partition conflicts with staged content"
        )


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise ResearchStoragePublicationError(
            "research metadata staging write failed"
        ) from error


def _file_identity(path: Path) -> tuple[str, int]:
    try:
        return _artifacts.file_identity(path)
    except OSError as error:
        raise ResearchStoragePublicationError(
            "research partition file is unavailable"
        ) from error


@dataclass(frozen=True)
class MinutePartitionSelection:
    partitions: tuple[PublishedResearchPartition, ...]
    missing_months: tuple[str, ...]


def select_minute_partitions(
    corpora: tuple[Path, ...], instrument: InstrumentId, window: TimeWindow
) -> MinutePartitionSelection:
    """Locate native Binance Spot minute archives; no network or universal reader.

    Mounts are explicit. Same manifest across mounts is reused; different revisions
    for a month are ambiguous, never resolved by directory order or newest mtime.
    Missing months remain explicit and do not establish historical listing status.
    """
    if (
        instrument.venue != VenueId("binance", "spot")
        or instrument.environment is not Environment.PRODUCTION
    ):
        raise ValueError(
            "minute archive selector supports Binance Spot production only"
        )
    if not 1 <= len(corpora) <= 32:
        raise ValueError("require 1..32 corpus roots")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    first = (epoch + timedelta(microseconds=window.start_us)).date().replace(day=1)
    last = (epoch + timedelta(microseconds=window.end_us - 1)).date().replace(day=1)
    months = []
    while first <= last:
        if len(months) >= 240:
            raise ValueError("partition selection exceeds 240 months")
        months.append(first.strftime("%Y-%m"))
        if first == last:
            break
        first = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    selected = {}
    files, total_bytes = 0, 0
    for root in dict.fromkeys(p.resolve() for p in corpora):
        base = root / "binance/spot/research-source-klines" / instrument.symbol / "1m"
        for month in months:
            for path in (base / month).glob("*/manifest.json"):
                files += 1
                total_bytes += (path.parent / "klines.parquet").stat().st_size
                if files > 512 or total_bytes > 8 * 1024**3:
                    raise ValueError("partition selection exceeds 512 files / 8 GiB")
                part = load_published_research_partition(path.parent)
                if (
                    part.manifest.symbol != instrument.symbol
                    or part.manifest.month != month
                ):
                    raise ValueError("partition path and metadata disagree")
                if (
                    month in selected
                    and selected[month].manifest.manifest_id
                    != part.manifest.manifest_id
                ):
                    raise ValueError(
                        f"ambiguous archive revision for {instrument.symbol} {month}; pin one corpus/version"
                    )
                selected[month] = part
    return MinutePartitionSelection(
        tuple(selected[m] for m in months if m in selected),
        tuple(m for m in months if m not in selected),
    )
