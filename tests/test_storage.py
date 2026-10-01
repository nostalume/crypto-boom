from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest

import crypto_boom.storage as storage_api
import crypto_boom.storage.canonical as canonical_storage
from crypto_boom.evaluation.replay import ReplaySpec, replay_partition
from crypto_boom.history.daily import (
    ArchiveIntegrityError,
    ArchiveManifest,
    PublishedArchive,
)
from crypto_boom.market import (
    AggregateTradeEvent,
    Environment,
    EpochTimestamp,
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
)
from crypto_boom.storage.canonical import (
    QuarantineReason,
    ReconciliationSample,
    SourceKline,
    StoragePublicationError,
    canonicalize_klines,
    load_published_canonical_klines,
    publish_canonical_archive,
    reconcile_kline,
)

OPEN_TIME_US = 1_735_776_000_000_000
MINUTE_US = 60_000_000
RUN_ID = UUID("00000000-0000-0000-0000-000000000321")
INSTRUMENT = InstrumentId(
    venue=VenueId("binance", "spot"),
    environment=Environment.PRODUCTION,
    symbol="ETHUSDT",
)
QUALITY = ObservationQuality(QualityState.VALID, complete=True)


def test_documented_storage_facade_is_identity_preserving() -> None:
    expected = {
        "KLINE_SCHEMA",
        "QUARANTINE_SCHEMA",
        "STORAGE_VERSION",
        "CanonicalKline",
        "CanonicalManifest",
        "CanonicalQualityReport",
        "CanonicalizationResult",
        "PublishedCanonicalPartition",
        "QuarantineReason",
        "QuarantinedKline",
        "ReconciliationResult",
        "ReconciliationSample",
        "SourceKline",
        "StorageError",
        "StoragePublicationError",
        "StorageSchemaError",
        "canonicalize_klines",
        "load_published_canonical_klines",
        "publish_canonical_archive",
        "reconcile_kline",
    }

    assert set(storage_api.__all__) == expected
    for name in expected:
        assert getattr(storage_api, name) is getattr(canonical_storage, name)


def _provenance(label: str, *, wall_time_ns: int = 1) -> Provenance:
    return Provenance(
        source=SourceDescriptor(
            endpoint="https://data.binance.vision",
            channel="spot/daily/klines/1m",
            schema_version=1,
        ),
        ingestion_run_id=RUN_ID,
        receipt=LocalReceipt(wall_time_ns, wall_time_ns),
        payload_digest=PayloadDigest.sha256(label.encode()),
        source_revision="sha256:" + "a" * 64,
        raw_payload_reference=f"ETHUSDT-1m-2025-01-02.csv#{label}",
    )


def _kline(
    *,
    open_time_us: int = OPEN_TIME_US,
    close: Decimal = Decimal("101"),
    label: str = "row-1",
    trade_count: int = 2,
    first_trade_id: int | None = None,
    last_trade_id: int | None = None,
) -> KlineEvent:
    return KlineEvent(
        instrument=INSTRUMENT,
        raw_symbol="ETHUSDT",
        interval="1m",
        source_event_time=None,
        open_time=EpochTimestamp(open_time_us, TimeUnit.MICROSECOND),
        close_time=EpochTimestamp(
            open_time_us + MINUTE_US - 1,
            TimeUnit.MICROSECOND,
        ),
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("99"),
        close_price=close,
        base_volume=Decimal("3"),
        quote_turnover=Decimal("303"),
        taker_buy_base_volume=Decimal("2"),
        taker_buy_quote_turnover=Decimal("202"),
        trade_count=trade_count,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        closed=True,
        provenance=_provenance(label),
        quality=QUALITY,
    )


def _metadata(
    capture_time_ns: int,
    *,
    observation: MetadataObservation = MetadataObservation.OBSERVED,
    status: InstrumentStatus = InstrumentStatus.TRADING,
) -> InstrumentMetadata:
    return InstrumentMetadata(
        instrument=INSTRUMENT,
        raw_symbol="ETHUSDT",
        status=status,
        base_asset="ETH",
        quote_asset="USDT",
        price_tick=Decimal("0.01"),
        quantity_step=Decimal("0.0001"),
        minimum_notional=Decimal("5"),
        permissions=("SPOT",),
        observation=observation,
        provenance=_provenance(
            f"metadata-{capture_time_ns}",
            wall_time_ns=capture_time_ns,
        ),
        quality=QUALITY,
    )


def _source(event: KlineEvent) -> SourceKline:
    return SourceKline(
        event=event,
        source_manifest_id="sha256:" + "b" * 64,
        decoder_version="binance-spot-kline-csv-v1",
    )


def test_same_identity_and_payload_deduplicates_to_one_canonical_row() -> None:
    event = _kline()
    result = canonicalize_klines(
        [_source(event), _source(event)],
        [_metadata(OPEN_TIME_US * 1_000 - 1)],
        expected_open_times_us=[OPEN_TIME_US],
    )

    assert len(result.rows) == 1
    assert result.report.duplicate_rows == 1
    assert result.report.conflict_keys == 0
    assert result.report.ready


def test_conflicting_finals_quarantine_every_revision() -> None:
    first = _kline(label="first")
    conflicting = _kline(close=Decimal("100.5"), label="conflicting")

    result = canonicalize_klines(
        [_source(first), _source(conflicting)],
        [_metadata(OPEN_TIME_US * 1_000 - 1)],
        expected_open_times_us=[OPEN_TIME_US],
    )

    assert result.rows == ()
    assert result.report.conflict_keys == 1
    assert result.report.quarantined_rows == 2
    assert {item.reason for item in result.quarantined} == {QuarantineReason.CONFLICT}
    assert not result.report.ready


def test_point_in_time_join_uses_latest_snapshot_not_future_state() -> None:
    before = _metadata(
        OPEN_TIME_US * 1_000 - 10,
        observation=MetadataObservation.INFERRED,
    )
    future = _metadata(
        OPEN_TIME_US * 1_000 + 10,
        status=InstrumentStatus.HALT,
    )

    result = canonicalize_klines(
        [_source(_kline())],
        [future, before],
        expected_open_times_us=[OPEN_TIME_US],
    )

    assert result.rows[0].metadata == before
    assert result.report.inferred_metadata_rows == 1
    assert result.report.ready


def test_missing_metadata_and_grid_gap_are_explicit_refusals() -> None:
    result = canonicalize_klines(
        [_source(_kline())],
        [],
        expected_open_times_us=[OPEN_TIME_US, OPEN_TIME_US + MINUTE_US],
    )

    assert result.rows == ()
    assert result.quarantined[0].reason is QuarantineReason.MISSING_METADATA
    assert result.report.missing_metadata_rows == 1
    assert result.report.missing_grid_rows == 1
    assert not result.report.ready


def _aggregate_trade(
    aggregate_trade_id: int,
    first_trade_id: int,
    last_trade_id: int,
    *,
    time_offset_us: int,
    price: str,
    quantity: str,
    buyer_is_maker: bool,
) -> AggregateTradeEvent:
    timestamp = EpochTimestamp(
        OPEN_TIME_US + time_offset_us,
        TimeUnit.MICROSECOND,
    )
    return AggregateTradeEvent(
        instrument=INSTRUMENT,
        raw_symbol="ETHUSDT",
        aggregate_trade_id=aggregate_trade_id,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        source_event_time=timestamp,
        trade_time=timestamp,
        price=Decimal(price),
        quantity=Decimal(quantity),
        buyer_is_maker=buyer_is_maker,
        provenance=_provenance(f"trade-{aggregate_trade_id}"),
        quality=QUALITY,
    )


def test_grain_aware_reconstruction_uses_underlying_trade_ranges() -> None:
    kline = replace(
        _kline(first_trade_id=10, last_trade_id=11),
        high_price=Decimal("101"),
        low_price=Decimal("100"),
        quote_turnover=Decimal("302"),
    )
    trades = (
        _aggregate_trade(
            7,
            10,
            10,
            time_offset_us=1,
            price="100",
            quantity="1",
            buyer_is_maker=True,
        ),
        _aggregate_trade(
            8,
            11,
            11,
            time_offset_us=2,
            price="101",
            quantity="2",
            buyer_is_maker=False,
        ),
    )

    result = reconcile_kline(kline, trades)

    assert result.matches
    assert result.complete_trade_coverage
    assert result.mismatches == ()


def test_reconstruction_refuses_incomplete_or_mismatched_trade_evidence() -> None:
    kline = _kline(first_trade_id=10, last_trade_id=12, trade_count=3)
    trades = (
        _aggregate_trade(
            7,
            10,
            10,
            time_offset_us=1,
            price="100",
            quantity="1",
            buyer_is_maker=True,
        ),
        _aggregate_trade(
            8,
            12,
            12,
            time_offset_us=2,
            price="101",
            quantity="2",
            buyer_is_maker=False,
        ),
    )

    result = reconcile_kline(kline, trades)

    assert not result.matches
    assert not result.complete_trade_coverage
    assert "TRADE_ID_GAP" in result.mismatches

    quality = canonicalize_klines(
        [_source(_kline())],
        [_metadata(OPEN_TIME_US * 1_000 - 1)],
        expected_open_times_us=[OPEN_TIME_US],
        reconciliation_samples=[ReconciliationSample(kline, trades)],
    )
    assert quality.report.reconciliation_failures == 1
    assert not quality.report.ready


def _published_archive(root: Path, *, minutes: int = 1_440) -> PublishedArchive:
    day = date(2025, 1, 2)
    member_filename = "ETHUSDT-1m-2025-01-02.csv"
    archive_filename = f"{member_filename[:-4]}.zip"
    rows = []
    for minute in range(minutes):
        open_time = OPEN_TIME_US + minute * MINUTE_US
        close_time = open_time + MINUTE_US - 1
        rows.append(f"{open_time},100,102,99,101,3,{close_time},303,2,2,202,0\n")
    member_bytes = "".join(rows).encode()
    archive_path = root / archive_filename
    root.mkdir(parents=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        member = zipfile.ZipInfo(member_filename, (2025, 1, 2, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(member, member_bytes)
    archive_bytes = archive_path.read_bytes()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    member_hash = hashlib.sha256(member_bytes).hexdigest()
    manifest = ArchiveManifest(
        schema_version=1,
        decoder_version="binance-spot-kline-csv-v1",
        venue="binance",
        market="spot",
        environment="production",
        dataset="klines",
        symbol="ETHUSDT",
        interval="1m",
        day=day.isoformat(),
        timestamp_unit="us",
        source_url=(
            "https://data.binance.vision/data/spot/daily/klines/"
            "ETHUSDT/1m/ETHUSDT-1m-2025-01-02.zip"
        ),
        checksum_url=(
            "https://data.binance.vision/data/spot/daily/klines/"
            "ETHUSDT/1m/ETHUSDT-1m-2025-01-02.zip.CHECKSUM"
        ),
        archive_filename=archive_filename,
        member_filename=member_filename,
        source_revision=f"sha256:{archive_hash}",
        archive_sha256=f"sha256:{archive_hash}",
        member_sha256=f"sha256:{member_hash}",
        compressed_bytes=len(archive_bytes),
        uncompressed_bytes=len(member_bytes),
        row_count=minutes,
        first_open_time_us=OPEN_TIME_US,
        last_open_time_us=OPEN_TIME_US + (minutes - 1) * MINUTE_US,
    )
    (root / "manifest.json").write_bytes(
        (
            json.dumps(
                manifest.to_mapping(),
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    (root / "acquisition.json").write_bytes(
        (
            json.dumps(
                {
                    "download_completed_monotonic_ns": 1,
                    "download_completed_wall_time_ns": OPEN_TIME_US * 1_000 - 2,
                    "ingestion_run_id": str(RUN_ID),
                    "schema_version": 1,
                    "wall_time_unit": "ns",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    return PublishedArchive(root, manifest, already_present=False)


def test_verified_archive_publishes_deterministic_inspectable_parquet(
    tmp_path: Path,
) -> None:
    archive = _published_archive(tmp_path / "raw")
    metadata = [_metadata(OPEN_TIME_US * 1_000 - 1)]

    first = publish_canonical_archive(
        archive,
        metadata=metadata,
        output_root=tmp_path / "canonical",
    )
    second = publish_canonical_archive(
        archive,
        metadata=metadata,
        output_root=tmp_path / "canonical",
    )

    assert not first.already_present
    assert second.already_present
    assert second.manifest == first.manifest
    assert first.manifest.manifest_id == (
        "sha256:c1e1e4f2113b3fdcd0c317dcc1a981401f0994e4267f404c0431ea441a520d59"
    )
    assert first.manifest.parquet_sha256 == (
        "sha256:d464e07b074af80d8e2e5c6e1c05625ad54e0211a7d7df750de5aa6cf901e7ba"
    )
    assert (first.path / "klines.parquet").stat().st_size == 95_133
    assert first.report.ready
    assert first.report.canonical_rows == 1_440
    assert first.report.quarantined_rows == 0
    table = pq.read_table(first.path / "klines.parquet")
    assert table.num_rows == 1_440
    assert str(table.schema.field("open_price").type) == "decimal128(38, 18)"
    assert table.column("payload_digest")[0] != table.column("payload_digest")[1]
    assert table.column("source_manifest_id")[0].as_py() == archive.manifest.manifest_id
    assert table.column("decoder_version")[0].as_py() == "binance-spot-kline-csv-v1"
    assert pq.read_table(first.path / "quarantine.parquet").num_rows == 0
    replayed = replay_partition(first, ReplaySpec("rpl-01-v1", 3_000_000))
    assert len(replayed.authoritative_events) == 1_440
    first_event = replayed.authoritative_events[0]
    assert isinstance(first_event, KlineEvent)
    assert first_event.identity.open_time_us == OPEN_TIME_US
    assert (
        first_event.provenance.payload_digest.value
        == table.column("payload_digest")[0].as_py()
    )


def test_raw_archive_tampering_fails_before_canonical_publication(
    tmp_path: Path,
) -> None:
    archive = _published_archive(tmp_path / "raw", minutes=1)
    archive_file = archive.path / archive.manifest.archive_filename
    archive_file.write_bytes(archive_file.read_bytes() + b"tampered")

    with pytest.raises(ArchiveIntegrityError, match="SHA-256"):
        publish_canonical_archive(
            archive,
            metadata=[_metadata(OPEN_TIME_US * 1_000 - 1)],
            output_root=tmp_path / "canonical",
        )

    assert not (tmp_path / "canonical").exists()


def test_existing_canonical_file_tampering_is_not_accepted_as_idempotent(
    tmp_path: Path,
) -> None:
    archive = _published_archive(tmp_path / "raw", minutes=1)
    metadata = [
        _metadata(
            OPEN_TIME_US * 1_000 - 1,
            status=InstrumentStatus.HALT,
        )
    ]
    published = publish_canonical_archive(
        archive,
        metadata=metadata,
        output_root=tmp_path / "canonical",
    )
    parquet_path = published.path / "klines.parquet"
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tampered")

    with pytest.raises(StoragePublicationError, match="file hash"):
        load_published_canonical_klines(published)

    with pytest.raises(StoragePublicationError, match="file hash"):
        publish_canonical_archive(
            archive,
            metadata=metadata,
            output_root=tmp_path / "canonical",
        )


def test_partial_existing_canonical_partition_is_rejected_at_storage_boundary(
    tmp_path: Path,
) -> None:
    archive = _published_archive(tmp_path / "raw", minutes=1)
    metadata = [_metadata(OPEN_TIME_US * 1_000 - 1)]
    published = publish_canonical_archive(
        archive,
        metadata=metadata,
        output_root=tmp_path / "canonical",
    )
    (published.path / "manifest.json").unlink()

    with pytest.raises(StoragePublicationError, match="no readable manifest"):
        publish_canonical_archive(
            archive,
            metadata=metadata,
            output_root=tmp_path / "canonical",
        )


def test_daily_grid_expects_only_point_in_time_trading_minutes(
    tmp_path: Path,
) -> None:
    archive = _published_archive(tmp_path / "raw", minutes=1)

    halted = publish_canonical_archive(
        archive,
        metadata=[
            _metadata(
                OPEN_TIME_US * 1_000 - 1,
                status=InstrumentStatus.HALT,
            )
        ],
        output_root=tmp_path / "halted",
    )
    trading = publish_canonical_archive(
        archive,
        metadata=[_metadata(OPEN_TIME_US * 1_000 - 1)],
        output_root=tmp_path / "trading",
    )

    assert halted.report.missing_grid_rows == 0
    assert halted.report.ready
    assert trading.report.missing_grid_rows == 1_439
    assert not trading.report.ready
