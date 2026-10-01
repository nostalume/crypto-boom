"""Canonical historical storage, reconciliation, and quality reporting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from crypto_boom import _artifacts
from crypto_boom.history.daily import PublishedArchive, load_published_klines
from crypto_boom.market import (
    AggregateTradeEvent,
    Environment,
    EpochTimestamp,
    InstrumentId,
    InstrumentMetadata,
    InstrumentStatus,
    KlineEvent,
    KlineIdentity,
    LocalReceipt,
    MetadataObservation,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    TimeUnit,
    VenueId,
)

STORAGE_VERSION = "canonical-kline-parquet-v1"
_MINUTE_US = 60_000_000
_DECIMAL_TYPE = pa.decimal128(38, 18)
_SHA256_PREFIX = "sha256:"


class StorageError(RuntimeError):
    """Canonical evidence could not be safely prepared or published."""


class StorageSchemaError(StorageError):
    """Canonical evidence cannot be represented by storage schema version 1."""


class StoragePublicationError(StorageError):
    """A canonical partition could not be atomically published."""


KLINE_SCHEMA = pa.schema(
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
        pa.field("metadata_capture_time", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("metadata_observation", pa.string(), nullable=False),
        pa.field("instrument_status", pa.string(), nullable=False),
        pa.field("base_asset", pa.string(), nullable=False),
        pa.field("quote_asset", pa.string(), nullable=False),
        pa.field("price_tick", _DECIMAL_TYPE, nullable=False),
        pa.field("quantity_step", _DECIMAL_TYPE, nullable=False),
        pa.field("minimum_notional", _DECIMAL_TYPE, nullable=False),
        pa.field("permissions", pa.list_(pa.string()), nullable=False),
        pa.field("quality_state", pa.string(), nullable=False),
        pa.field("quality_complete", pa.bool_(), nullable=False),
        pa.field("diagnostics", pa.list_(pa.string()), nullable=False),
        pa.field("source_endpoint", pa.string(), nullable=False),
        pa.field("source_channel", pa.string(), nullable=False),
        pa.field("source_schema_version", pa.int32(), nullable=False),
        pa.field("ingestion_run_id", pa.string(), nullable=False),
        pa.field("receipt_wall_time_ns", pa.int64(), nullable=False),
        pa.field("receipt_monotonic_ns", pa.int64(), nullable=False),
        pa.field("payload_digest", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("raw_payload_reference", pa.string(), nullable=False),
        pa.field("source_manifest_id", pa.string(), nullable=False),
        pa.field("decoder_version", pa.string(), nullable=False),
    ],
    metadata={
        b"crypto_boom.dataset": b"canonical_klines",
        b"crypto_boom.schema_version": b"1",
    },
)

QUARANTINE_SCHEMA = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("environment", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open_time_us", pa.int64(), nullable=False),
        pa.field("payload_digest", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("raw_payload_reference", pa.string(), nullable=False),
        pa.field("source_manifest_id", pa.string(), nullable=False),
        pa.field("decoder_version", pa.string(), nullable=False),
        pa.field("quarantine_reason", pa.string(), nullable=False),
    ],
    metadata={
        b"crypto_boom.dataset": b"quarantined_klines",
        b"crypto_boom.schema_version": b"1",
    },
)


class QuarantineReason(StrEnum):
    CONFLICT = "CONFLICT"
    MISSING_METADATA = "MISSING_METADATA"
    INVALID_QUALITY = "INVALID_QUALITY"


@dataclass(frozen=True, slots=True)
class SourceKline:
    event: KlineEvent
    source_manifest_id: str
    decoder_version: str


@dataclass(frozen=True, slots=True)
class CanonicalKline:
    source: SourceKline
    metadata: InstrumentMetadata


@dataclass(frozen=True, slots=True)
class QuarantinedKline:
    source: SourceKline
    reason: QuarantineReason


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    matches: bool
    complete_trade_coverage: bool
    mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationSample:
    kline: KlineEvent
    aggregate_trades: tuple[AggregateTradeEvent, ...]
    tolerance: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class CanonicalQualityReport:
    input_rows: int
    canonical_rows: int
    duplicate_rows: int
    conflict_keys: int
    quarantined_rows: int
    missing_metadata_rows: int
    missing_grid_rows: int
    reconciliation_samples: int
    reconciliation_failures: int
    inferred_metadata_rows: int
    ready: bool

    @property
    def report_id(self) -> str:
        return _artifacts.content_id(asdict(self))

    def to_mapping(self) -> dict[str, object]:
        return {
            "report_id": self.report_id,
            "schema_version": 1,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    rows: tuple[CanonicalKline, ...]
    quarantined: tuple[QuarantinedKline, ...]
    report: CanonicalQualityReport


@dataclass(frozen=True, slots=True)
class CanonicalManifest:
    schema_version: int
    storage_version: str
    source_manifest_id: str
    decoder_version: str
    parquet_sha256: str
    quarantine_sha256: str
    quality_report_sha256: str
    report_id: str
    canonical_rows: int
    quarantined_rows: int

    def _content_mapping(self) -> dict[str, object]:
        return asdict(self)

    @property
    def manifest_id(self) -> str:
        return _artifacts.content_id(self._content_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self._content_mapping()}


@dataclass(frozen=True, slots=True)
class PublishedCanonicalPartition:
    path: Path
    manifest: CanonicalManifest
    report: CanonicalQualityReport
    already_present: bool


def load_published_canonical_klines(
    partition: PublishedCanonicalPartition,
) -> tuple[SourceKline, ...]:
    """Reverify and rehydrate one qualified canonical kline partition."""

    if not partition.report.ready:
        raise StorageSchemaError("canonical partition is not quality-ready")
    expected_manifest = _artifacts.canonical_json(partition.manifest.to_mapping())
    expected_report = _artifacts.canonical_json(partition.report.to_mapping())
    expected_files = {
        "klines.parquet": partition.manifest.parquet_sha256,
        "quarantine.parquet": partition.manifest.quarantine_sha256,
        "quality-report.json": partition.manifest.quality_report_sha256,
    }
    try:
        _verify_existing_partition(
            partition.path,
            expected_manifest,
            expected_files,
        )
        if (partition.path / "quality-report.json").read_bytes() != expected_report:
            raise StorageSchemaError(
                "canonical quality report conflicts with admitted result"
            )
        table = pq.read_table(partition.path / "klines.parquet")
    except StorageError:
        raise
    except (OSError, pa.ArrowException) as error:
        raise StorageSchemaError("canonical replay input is unreadable") from error

    if not _schema_matches(table.schema, KLINE_SCHEMA):
        raise StorageSchemaError("canonical replay schema is not version 1")
    if table.num_rows != partition.manifest.canonical_rows:
        raise StorageSchemaError("canonical replay row count conflicts with manifest")

    try:
        rows = tuple(
            _source_kline_from_mapping(row)
            for row in table.select(
                [
                    "venue",
                    "market",
                    "environment",
                    "symbol",
                    "interval",
                    "raw_open_time",
                    "raw_close_time",
                    "timestamp_unit",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "base_volume",
                    "quote_turnover",
                    "taker_buy_base_volume",
                    "taker_buy_quote_turnover",
                    "trade_count",
                    "first_trade_id",
                    "last_trade_id",
                    "quality_state",
                    "quality_complete",
                    "diagnostics",
                    "source_endpoint",
                    "source_channel",
                    "source_schema_version",
                    "ingestion_run_id",
                    "receipt_wall_time_ns",
                    "receipt_monotonic_ns",
                    "payload_digest",
                    "source_revision",
                    "raw_payload_reference",
                    "source_manifest_id",
                    "decoder_version",
                ]
            ).to_pylist()
        )
    except (TypeError, ValueError) as error:
        raise StorageSchemaError("canonical replay row is invalid") from error
    if any(
        row.source_manifest_id != partition.manifest.source_manifest_id
        or row.decoder_version != partition.manifest.decoder_version
        for row in rows
    ):
        raise StorageSchemaError("canonical replay provenance conflicts with manifest")
    return rows


def _schema_matches(actual: pa.Schema, expected: pa.Schema) -> bool:
    if actual.metadata != expected.metadata or len(actual) != len(expected):
        return False
    return all(
        observed.name == declared.name
        and observed.type == declared.type
        and observed.nullable == declared.nullable
        for observed, declared in zip(actual, expected, strict=True)
    )


def _required_text_value(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} is not text")
    return value


def _required_int_value(row: dict[str, object], name: str) -> int:
    value = row.get(name)
    if type(value) is not int:
        raise ValueError(f"{name} is not an integer")
    return value


def _optional_int_value(row: dict[str, object], name: str) -> int | None:
    value = row.get(name)
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{name} is not an integer")
    return value


def _required_decimal_value(row: dict[str, object], name: str) -> Decimal:
    value = row.get(name)
    if not isinstance(value, Decimal):
        raise ValueError(f"{name} is not decimal")
    return value


def _required_bool_value(row: dict[str, object], name: str) -> bool:
    value = row.get(name)
    if type(value) is not bool:
        raise ValueError(f"{name} is not boolean")
    return value


def _required_text_sequence(row: dict[str, object], name: str) -> tuple[str, ...]:
    value = row.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} is not a text sequence")
    return tuple(value)


def _source_kline_from_mapping(row: dict[str, object]) -> SourceKline:
    unit = TimeUnit(_required_text_value(row, "timestamp_unit"))
    instrument = InstrumentId(
        venue=VenueId(
            _required_text_value(row, "venue"),
            _required_text_value(row, "market"),
        ),
        environment=Environment(_required_text_value(row, "environment")),
        symbol=_required_text_value(row, "symbol"),
    )
    provenance = Provenance(
        source=SourceDescriptor(
            endpoint=_required_text_value(row, "source_endpoint"),
            channel=_required_text_value(row, "source_channel"),
            schema_version=_required_int_value(row, "source_schema_version"),
        ),
        ingestion_run_id=UUID(_required_text_value(row, "ingestion_run_id")),
        receipt=LocalReceipt(
            wall_time_ns=_required_int_value(row, "receipt_wall_time_ns"),
            monotonic_ns=_required_int_value(row, "receipt_monotonic_ns"),
        ),
        payload_digest=PayloadDigest(_required_text_value(row, "payload_digest")),
        source_revision=_required_text_value(row, "source_revision"),
        raw_payload_reference=_required_text_value(row, "raw_payload_reference"),
    )
    event = KlineEvent(
        instrument=instrument,
        raw_symbol=instrument.symbol,
        interval=_required_text_value(row, "interval"),
        source_event_time=None,
        open_time=EpochTimestamp(_required_int_value(row, "raw_open_time"), unit),
        close_time=EpochTimestamp(_required_int_value(row, "raw_close_time"), unit),
        open_price=_required_decimal_value(row, "open_price"),
        high_price=_required_decimal_value(row, "high_price"),
        low_price=_required_decimal_value(row, "low_price"),
        close_price=_required_decimal_value(row, "close_price"),
        base_volume=_required_decimal_value(row, "base_volume"),
        quote_turnover=_required_decimal_value(row, "quote_turnover"),
        taker_buy_base_volume=_required_decimal_value(row, "taker_buy_base_volume"),
        taker_buy_quote_turnover=_required_decimal_value(
            row, "taker_buy_quote_turnover"
        ),
        trade_count=_required_int_value(row, "trade_count"),
        first_trade_id=_optional_int_value(row, "first_trade_id"),
        last_trade_id=_optional_int_value(row, "last_trade_id"),
        closed=True,
        provenance=provenance,
        quality=ObservationQuality(
            state=QualityState(_required_text_value(row, "quality_state")),
            complete=_required_bool_value(row, "quality_complete"),
            diagnostics=_required_text_sequence(row, "diagnostics"),
        ),
    )
    return SourceKline(
        event=event,
        source_manifest_id=_required_text_value(row, "source_manifest_id"),
        decoder_version=_required_text_value(row, "decoder_version"),
    )


def canonicalize_klines(
    sources: Iterable[SourceKline],
    metadata: Iterable[InstrumentMetadata],
    *,
    expected_open_times_us: Iterable[int] = (),
    reconciliation_samples: Sequence[ReconciliationSample] = (),
) -> CanonicalizationResult:
    """Deduplicate, join, quarantine, and report canonical kline evidence."""

    source_rows = tuple(sources)
    metadata_rows = tuple(metadata)
    grouped: dict[KlineIdentity, list[SourceKline]] = defaultdict(list)
    for source in source_rows:
        grouped[source.event.identity].append(source)

    rows: list[CanonicalKline] = []
    quarantined: list[QuarantinedKline] = []
    duplicate_rows = 0
    conflict_keys = 0
    missing_metadata_rows = 0

    for identity in sorted(
        grouped,
        key=lambda value: value.open_time_us,
    ):
        revisions = grouped[identity]
        digests = {source.event.provenance.payload_digest.value for source in revisions}
        if len(digests) > 1:
            conflict_keys += 1
            quarantined.extend(
                QuarantinedKline(source, QuarantineReason.CONFLICT)
                for source in revisions
            )
            continue

        source = revisions[0]
        duplicate_rows += len(revisions) - 1
        if not source.event.quality.usable_for_final_transition:
            quarantined.append(
                QuarantinedKline(source, QuarantineReason.INVALID_QUALITY)
            )
            continue

        selected = _point_in_time_metadata(
            source.event.instrument,
            source.event.open_time.epoch_microseconds,
            metadata_rows,
        )
        if selected is None:
            missing_metadata_rows += 1
            quarantined.append(
                QuarantinedKline(source, QuarantineReason.MISSING_METADATA)
            )
            continue
        rows.append(CanonicalKline(source, selected))

    observed_times = {
        source.event.open_time.epoch_microseconds for source in source_rows
    }
    missing_grid_rows = len(set(expected_open_times_us) - observed_times)
    reconciliation_failures = sum(
        not reconcile_kline(
            sample.kline,
            sample.aggregate_trades,
            tolerance=sample.tolerance,
        ).matches
        for sample in reconciliation_samples
    )
    inferred_metadata_rows = sum(
        row.metadata.observation is MetadataObservation.INFERRED for row in rows
    )
    report = CanonicalQualityReport(
        input_rows=len(source_rows),
        canonical_rows=len(rows),
        duplicate_rows=duplicate_rows,
        conflict_keys=conflict_keys,
        quarantined_rows=len(quarantined),
        missing_metadata_rows=missing_metadata_rows,
        missing_grid_rows=missing_grid_rows,
        reconciliation_samples=len(reconciliation_samples),
        reconciliation_failures=reconciliation_failures,
        inferred_metadata_rows=inferred_metadata_rows,
        ready=bool(rows)
        and not quarantined
        and missing_grid_rows == 0
        and reconciliation_failures == 0,
    )
    return CanonicalizationResult(tuple(rows), tuple(quarantined), report)


def _point_in_time_metadata(
    instrument: InstrumentId,
    event_time_us: int,
    metadata: Sequence[InstrumentMetadata],
) -> InstrumentMetadata | None:
    event_time_ns = event_time_us * 1_000
    candidates = [
        snapshot
        for snapshot in metadata
        if snapshot.instrument == instrument
        and snapshot.provenance.receipt.wall_time_ns <= event_time_ns
        and snapshot.quality.usable_for_final_transition
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda snapshot: snapshot.provenance.receipt.wall_time_ns,
    )


def reconcile_kline(
    kline: KlineEvent,
    aggregate_trades: Sequence[AggregateTradeEvent],
    *,
    tolerance: Decimal = Decimal(0),
) -> ReconciliationResult:
    """Reconstruct one kline from complete aggregate-trade coverage."""

    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("reconciliation tolerance must be finite and non-negative")
    if not aggregate_trades:
        return ReconciliationResult(
            matches=False,
            complete_trade_coverage=False,
            mismatches=("NO_AGGREGATE_TRADES",),
        )

    ordered = sorted(
        aggregate_trades,
        key=lambda trade: (
            trade.trade_time.epoch_microseconds,
            trade.aggregate_trade_id,
        ),
    )
    mismatches: list[str] = []
    start_us = kline.open_time.epoch_microseconds
    end_us = kline.close_time.epoch_microseconds
    if any(trade.instrument != kline.instrument for trade in ordered):
        mismatches.append("INSTRUMENT_MISMATCH")
    if any(
        not start_us <= trade.trade_time.epoch_microseconds <= end_us
        for trade in ordered
    ):
        mismatches.append("TRADE_OUTSIDE_KLINE")
    if any(not trade.quality.usable_for_final_transition for trade in ordered):
        mismatches.append("INVALID_TRADE_QUALITY")

    ranges = sorted((trade.first_trade_id, trade.last_trade_id) for trade in ordered)
    range_gap = any(
        current_first != previous_last + 1
        for (_, previous_last), (current_first, _) in pairwise(ranges)
    )
    aggregate_id_gap = any(
        current.aggregate_trade_id != previous.aggregate_trade_id + 1
        for previous, current in pairwise(ordered)
    )
    underlying_trade_count = sum(last - first + 1 for first, last in ranges)
    boundary_mismatch = kline.first_trade_id is not None and (
        ranges[0][0] != kline.first_trade_id or ranges[-1][1] != kline.last_trade_id
    )
    complete_trade_coverage = (
        not range_gap
        and not aggregate_id_gap
        and not boundary_mismatch
        and underlying_trade_count == kline.trade_count
    )
    if range_gap:
        mismatches.append("TRADE_ID_GAP")
    if aggregate_id_gap:
        mismatches.append("AGGREGATE_TRADE_ID_GAP")
    if boundary_mismatch:
        mismatches.append("TRADE_ID_BOUNDARY_MISMATCH")
    if underlying_trade_count != kline.trade_count:
        mismatches.append("UNDERLYING_TRADE_COUNT_MISMATCH")

    reconstructed = {
        "OPEN_PRICE": ordered[0].price,
        "HIGH_PRICE": max(trade.price for trade in ordered),
        "LOW_PRICE": min(trade.price for trade in ordered),
        "CLOSE_PRICE": ordered[-1].price,
        "BASE_VOLUME": sum(
            (trade.quantity for trade in ordered),
            start=Decimal(0),
        ),
        "QUOTE_TURNOVER": sum(
            (trade.price * trade.quantity for trade in ordered),
            start=Decimal(0),
        ),
        "TAKER_BUY_BASE_VOLUME": sum(
            (trade.quantity for trade in ordered if not trade.buyer_is_maker),
            start=Decimal(0),
        ),
        "TAKER_BUY_QUOTE_TURNOVER": sum(
            (
                trade.price * trade.quantity
                for trade in ordered
                if not trade.buyer_is_maker
            ),
            start=Decimal(0),
        ),
    }
    expected = {
        "OPEN_PRICE": kline.open_price,
        "HIGH_PRICE": kline.high_price,
        "LOW_PRICE": kline.low_price,
        "CLOSE_PRICE": kline.close_price,
        "BASE_VOLUME": kline.base_volume,
        "QUOTE_TURNOVER": kline.quote_turnover,
        "TAKER_BUY_BASE_VOLUME": kline.taker_buy_base_volume,
        "TAKER_BUY_QUOTE_TURNOVER": kline.taker_buy_quote_turnover,
    }
    mismatches.extend(
        name
        for name, value in reconstructed.items()
        if abs(value - expected[name]) > tolerance
    )
    return ReconciliationResult(
        matches=complete_trade_coverage and not mismatches,
        complete_trade_coverage=complete_trade_coverage,
        mismatches=tuple(mismatches),
    )


def _canonical_mapping(row: CanonicalKline) -> dict[str, object]:
    event = row.source.event
    provenance = event.provenance
    metadata = row.metadata
    if provenance.source_revision is None or provenance.raw_payload_reference is None:
        raise StorageSchemaError("canonical provenance is incomplete")
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
        "metadata_capture_time": metadata.provenance.receipt.wall_time_ns,
        "metadata_observation": metadata.observation.value,
        "instrument_status": metadata.status.value,
        "base_asset": metadata.base_asset,
        "quote_asset": metadata.quote_asset,
        "price_tick": metadata.price_tick,
        "quantity_step": metadata.quantity_step,
        "minimum_notional": metadata.minimum_notional,
        "permissions": list(metadata.permissions),
        "quality_state": event.quality.state.value,
        "quality_complete": event.quality.complete,
        "diagnostics": list(event.quality.diagnostics),
        "source_endpoint": provenance.source.endpoint,
        "source_channel": provenance.source.channel,
        "source_schema_version": provenance.source.schema_version,
        "ingestion_run_id": str(provenance.ingestion_run_id),
        "receipt_wall_time_ns": provenance.receipt.wall_time_ns,
        "receipt_monotonic_ns": provenance.receipt.monotonic_ns,
        "payload_digest": provenance.payload_digest.value,
        "source_revision": provenance.source_revision,
        "raw_payload_reference": provenance.raw_payload_reference,
        "source_manifest_id": row.source.source_manifest_id,
        "decoder_version": row.source.decoder_version,
    }


def _quarantine_mapping(row: QuarantinedKline) -> dict[str, object]:
    event = row.source.event
    provenance = event.provenance
    if provenance.source_revision is None or provenance.raw_payload_reference is None:
        raise StorageSchemaError("quarantine provenance is incomplete")
    return {
        "venue": event.instrument.venue.name,
        "market": event.instrument.venue.market,
        "environment": event.instrument.environment.value,
        "symbol": event.instrument.symbol,
        "interval": event.interval,
        "open_time_us": event.open_time.epoch_microseconds,
        "payload_digest": provenance.payload_digest.value,
        "source_revision": provenance.source_revision,
        "raw_payload_reference": provenance.raw_payload_reference,
        "source_manifest_id": row.source.source_manifest_id,
        "decoder_version": row.source.decoder_version,
        "quarantine_reason": row.reason.value,
    }


def _table(
    rows: Iterable[dict[str, object]],
    schema: pa.Schema,
) -> pa.Table:
    try:
        return pa.Table.from_pylist(list(rows), schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as error:
        raise StorageSchemaError(
            "canonical values do not fit storage schema version 1"
        ) from error


def publish_canonical_archive(
    archive: PublishedArchive,
    *,
    metadata: Iterable[InstrumentMetadata],
    output_root: Path,
    reconciliation_samples: Sequence[ReconciliationSample] = (),
) -> PublishedCanonicalPartition:
    """Publish one verified archive revision as an immutable Parquet partition."""

    events = load_published_klines(archive)
    manifest = archive.manifest
    sources = tuple(
        SourceKline(
            event=event,
            source_manifest_id=manifest.manifest_id,
            decoder_version=manifest.decoder_version,
        )
        for event in events
    )
    try:
        partition_day = date.fromisoformat(manifest.day)
    except ValueError as error:
        raise StorageSchemaError("archive day is invalid") from error
    day_start_us = (partition_day - date(1970, 1, 1)).days * 86_400_000_000
    metadata_rows = tuple(metadata)
    daily_grid = range(
        day_start_us,
        day_start_us + 86_400_000_000,
        _MINUTE_US,
    )
    expected_open_times = (
        open_time_us
        for open_time_us in daily_grid
        if (
            snapshot := _point_in_time_metadata(
                events[0].instrument,
                open_time_us,
                metadata_rows,
            )
        )
        is not None
        and snapshot.status is InstrumentStatus.TRADING
    )
    canonical = canonicalize_klines(
        sources,
        metadata_rows,
        expected_open_times_us=expected_open_times,
        reconciliation_samples=reconciliation_samples,
    )
    canonical_table = _table(
        (_canonical_mapping(row) for row in canonical.rows),
        KLINE_SCHEMA,
    )
    quarantine_table = _table(
        (_quarantine_mapping(row) for row in canonical.quarantined),
        QUARANTINE_SCHEMA,
    )

    output_root = output_root.resolve()
    target = (
        output_root
        / manifest.venue
        / manifest.market
        / manifest.dataset
        / manifest.symbol
        / manifest.interval
        / manifest.day
        / manifest.manifest_id.removeprefix(_SHA256_PREFIX)
    )
    try:
        staging_parent = output_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StoragePublicationError(
            "canonical staging root is unavailable"
        ) from error

    try:
        with _artifacts.publication_staging_directory(
            staging_parent,
            prefix="sto-",
        ) as staging:
            parquet_path = staging / "klines.parquet"
            quarantine_path = staging / "quarantine.parquet"
            pq.write_table(
                canonical_table,
                parquet_path,
                compression="zstd",
                version="2.6",
                data_page_version="2.0",
            )
            pq.write_table(
                quarantine_table,
                quarantine_path,
                compression="zstd",
                version="2.6",
                data_page_version="2.0",
            )
            report_bytes = _artifacts.canonical_json(canonical.report.to_mapping())
            report_path = staging / "quality-report.json"
            _write_bytes(report_path, report_bytes)
            canonical_manifest = CanonicalManifest(
                schema_version=1,
                storage_version=STORAGE_VERSION,
                source_manifest_id=manifest.manifest_id,
                decoder_version=manifest.decoder_version,
                parquet_sha256=_file_id(parquet_path),
                quarantine_sha256=_file_id(quarantine_path),
                quality_report_sha256=_file_id(report_path),
                report_id=canonical.report.report_id,
                canonical_rows=canonical.report.canonical_rows,
                quarantined_rows=canonical.report.quarantined_rows,
            )
            manifest_bytes = _artifacts.canonical_json(canonical_manifest.to_mapping())
            _write_bytes(staging / "manifest.json", manifest_bytes)
            already_present = _publish_directory(
                staging,
                target,
                expected_manifest=manifest_bytes,
                expected_files={
                    "klines.parquet": canonical_manifest.parquet_sha256,
                    "quarantine.parquet": canonical_manifest.quarantine_sha256,
                    "quality-report.json": canonical_manifest.quality_report_sha256,
                },
            )
    except StorageError:
        raise
    except (OSError, pa.ArrowException) as error:
        raise StoragePublicationError("canonical Parquet publication failed") from error

    return PublishedCanonicalPartition(
        path=target,
        manifest=canonical_manifest,
        report=canonical.report,
        already_present=already_present,
    )


def _publish_directory(
    staging: Path,
    target: Path,
    *,
    expected_manifest: bytes,
    expected_files: dict[str, str],
) -> bool:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StoragePublicationError("canonical publication path failed") from error
    try:
        return _artifacts.adopt_directory(
            staging,
            target,
            verify_existing=lambda existing: _verify_existing_partition(
                existing,
                expected_manifest,
                expected_files,
            ),
        )
    except StorageError:
        raise
    except OSError as error:
        raise StoragePublicationError("atomic canonical publication failed") from error


def _verify_existing_partition(
    target: Path,
    expected_manifest: bytes,
    expected_files: dict[str, str],
) -> None:
    try:
        existing_manifest = (target / "manifest.json").read_bytes()
    except OSError as error:
        raise StoragePublicationError(
            "existing canonical partition has no readable manifest"
        ) from error
    if existing_manifest != expected_manifest:
        raise StoragePublicationError(
            "existing canonical partition conflicts with staged content"
        )
    if any(
        _file_id(target / name) != digest for name, digest in expected_files.items()
    ):
        raise StoragePublicationError(
            "existing canonical partition file hash conflicts with manifest"
        )


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise StoragePublicationError(
            "canonical metadata staging write failed"
        ) from error


def _file_id(path: Path) -> str:
    try:
        digest, _ = _artifacts.file_identity(path)
    except OSError as error:
        raise StoragePublicationError("canonical staged file is unavailable") from error
    return digest
