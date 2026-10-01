from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest

from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
)
from crypto_boom.history.monthly import (
    MonthlyArchiveManifest,
    MonthlyKlineDay,
    PublishedMonthlyArchive,
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
from crypto_boom.storage import source as research_storage
from crypto_boom.storage.source import (
    RESEARCH_KLINE_SCHEMA,
    ResearchCorpusLimits,
    ResearchStorageIntegrityError,
    ResearchStorageResourceError,
    load_published_research_partition,
    materialize_research_corpus,
)

OPEN_TIME_US = 1_735_776_000_000_000
MINUTE_US = 60_000_000
RUN_ID = UUID("00000000-0000-0000-0000-000000000654")
SOURCE_REVISION = "sha256:" + "b" * 64


def _report(*, unresolved: bool = False) -> ArchiveAvailabilityReport:
    state = AvailabilityState.UNRESOLVED if unresolved else AvailabilityState.AVAILABLE
    return ArchiveAvailabilityReport(
        schema_version=1,
        pool_id="sha256:" + "a" * 64,
        base_url="https://data.binance.vision/data/spot/monthly",
        interval="1m",
        start_month="2025-01",
        end_month="2025-01",
        probed_at_ns=1_795_027_260_000_002_000,
        probes=(
            MonthlyArchiveProbe(
                "ETHUSDT",
                "2025-01",
                state,
                SOURCE_REVISION if not unresolved else None,
                1,
                200 if not unresolved else 500,
            ),
        ),
    )


def _manifest() -> MonthlyArchiveManifest:
    return MonthlyArchiveManifest(
        schema_version=1,
        decoder_version="binance-spot-kline-csv-v1",
        venue="binance",
        market="spot",
        environment="production",
        dataset="klines",
        symbol="ETHUSDT",
        interval="1m",
        month="2025-01",
        timestamp_unit="us",
        source_url="https://data.binance.vision/ETHUSDT-1m-2025-01.zip",
        checksum_url="https://data.binance.vision/ETHUSDT-1m-2025-01.zip.CHECKSUM",
        archive_filename="ETHUSDT-1m-2025-01.zip",
        member_filename="ETHUSDT-1m-2025-01.csv",
        source_revision=SOURCE_REVISION,
        archive_sha256=SOURCE_REVISION,
        member_sha256="sha256:" + "c" * 64,
        compressed_bytes=100,
        uncompressed_bytes=1_000,
        row_count=2,
        first_open_time_us=OPEN_TIME_US,
        last_open_time_us=OPEN_TIME_US + MINUTE_US,
        observed_first_day="2025-01-02",
        observed_last_day="2025-01-02",
        observed_days=("2025-01-02",),
        missing_days=(),
        internal_gaps=(),
    )


def _event(minute: int) -> KlineEvent:
    open_time = OPEN_TIME_US + minute * MINUTE_US
    instrument = InstrumentId(
        venue=VenueId("binance", "spot"),
        environment=Environment.PRODUCTION,
        symbol="ETHUSDT",
    )
    return KlineEvent(
        instrument=instrument,
        raw_symbol="ETHUSDT",
        interval="1m",
        source_event_time=None,
        open_time=EpochTimestamp(open_time, TimeUnit.MICROSECOND),
        close_time=EpochTimestamp(open_time + MINUTE_US - 1, TimeUnit.MICROSECOND),
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("99"),
        close_price=Decimal("101"),
        base_volume=Decimal("3"),
        quote_turnover=Decimal("303"),
        taker_buy_base_volume=Decimal("2"),
        taker_buy_quote_turnover=Decimal("202"),
        trade_count=2,
        first_trade_id=minute * 2,
        last_trade_id=minute * 2 + 1,
        closed=True,
        provenance=Provenance(
            source=SourceDescriptor(
                endpoint="https://data.binance.vision",
                channel="spot/monthly/klines/1m",
                schema_version=1,
            ),
            ingestion_run_id=RUN_ID,
            receipt=LocalReceipt(1, 1),
            payload_digest=PayloadDigest.sha256(f"row-{minute}".encode()),
            source_revision=SOURCE_REVISION,
            raw_payload_reference=f"ETHUSDT-1m-2025-01.csv#{minute + 1}",
        ),
        quality=ObservationQuality(QualityState.VALID, complete=True),
    )


def _install_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    published = PublishedMonthlyArchive(
        tmp_path / "monthly-source",
        _manifest(),
        _report().report_id,
        True,
    )
    monkeypatch.setattr(
        research_storage,
        "load_published_monthly_archive",
        lambda path: published,
    )
    monkeypatch.setattr(
        research_storage,
        "load_published_monthly_klines_by_day",
        lambda source: (MonthlyKlineDay(date(2025, 1, 2), (_event(0), _event(1))),),
    )


def test_materializes_source_only_parquet_and_reuses_verified_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    first = materialize_research_corpus(
        _report(),
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "research",
        limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
    )
    second = materialize_research_corpus(
        _report(),
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "research",
        limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
    )

    assert first.archive_count == 1
    assert first.row_count == 2
    assert first.created_count == 1
    assert second.reused_count == 1
    publication = first.publications[0]
    assert publication.manifest.manifest_id == (
        "sha256:9fae591c94a8796ec6a3ca05ccc7397b23b790ac5cab16ca752e377ce7792307"
    )
    assert publication.manifest.parquet_sha256 == (
        "sha256:6a23d7206320e4ee469b01e37d10b81a2fe93723e0158fb8203702e76c06c724"
    )
    assert publication.manifest.parquet_bytes == 10_930
    assert publication.manifest.evidence_class == "source_only"
    assert publication.manifest.source_manifest_id == _manifest().manifest_id
    assert load_published_research_partition(publication.path).manifest == (
        publication.manifest
    )
    table = pq.read_table(publication.path / "klines.parquet")
    assert table.schema == RESEARCH_KLINE_SCHEMA
    assert table.column("symbol").to_pylist() == ["ETHUSDT", "ETHUSDT"]
    assert "instrument_status" not in table.column_names
    assert "price_tick" not in table.column_names


def test_research_partition_reload_rejects_modified_manifest_and_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    result = materialize_research_corpus(
        _report(),
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "research",
        limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
    )
    manifest_path = result.publications[0].path / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    document = json.loads(manifest_bytes)
    document["row_count"] = True
    manifest_path.write_bytes(
        (json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n").encode()
    )
    with pytest.raises(ResearchStorageIntegrityError, match="manifest is invalid"):
        load_published_research_partition(result.publications[0].path)
    manifest_path.write_bytes(manifest_bytes)

    parquet = result.publications[0].path / "klines.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tampered")

    with pytest.raises(ResearchStorageIntegrityError, match="inconsistent"):
        load_published_research_partition(result.publications[0].path)


def test_partial_existing_research_partition_is_rejected_at_storage_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    result = materialize_research_corpus(
        _report(),
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "research",
        limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
    )
    (result.publications[0].path / "manifest.json").unlink()

    with pytest.raises(ResearchStorageIntegrityError, match="manifest is invalid"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path / "monthly",
            output_root=tmp_path / "research",
            limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
        )


def test_research_corpus_refuses_unresolved_unknown_and_oversized_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="concurrency"):
        ResearchCorpusLimits(maximum_concurrency=3)
    with pytest.raises(ResearchStorageIntegrityError, match="unresolved"):
        materialize_research_corpus(
            _report(unresolved=True),
            monthly_root=tmp_path,
            output_root=tmp_path,
        )
    with pytest.raises(ResearchStorageIntegrityError, match="not present"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path,
            output_root=tmp_path,
            symbols=("UNKNOWNUSDT",),
        )
    with pytest.raises(ResearchStorageResourceError, match="2 rows"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path,
            output_root=tmp_path,
            limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=1),
        )
    assert not (tmp_path / "binance").exists()

    with pytest.raises(ResearchStorageResourceError, match="Parquet byte budget"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path,
            output_root=tmp_path / "bounded-output",
            limits=ResearchCorpusLimits(
                maximum_archives=1,
                maximum_rows=2,
                maximum_parquet_bytes=1,
            ),
        )
    assert not list((tmp_path / "bounded-output").rglob("manifest.json"))


def test_research_corpus_elapsed_budget_stops_before_next_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    clock = iter((0.0, 31.0))
    monkeypatch.setattr(research_storage, "monotonic", lambda: next(clock))

    with pytest.raises(ResearchStorageResourceError, match="elapsed-time"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path,
            output_root=tmp_path / "research",
            limits=ResearchCorpusLimits(
                maximum_archives=1,
                maximum_rows=2,
                maximum_elapsed_seconds=30,
            ),
        )
    assert not (tmp_path / "research").exists()


def test_research_corpus_rejects_row_source_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_source(monkeypatch, tmp_path)
    mismatched = replace(
        _event(0),
        provenance=replace(
            _event(0).provenance,
            source_revision="sha256:" + "d" * 64,
        ),
    )
    monkeypatch.setattr(
        research_storage,
        "load_published_monthly_klines_by_day",
        lambda source: (MonthlyKlineDay(date(2025, 1, 2), (mismatched, _event(1))),),
    )

    with pytest.raises(ResearchStorageIntegrityError, match="provenance"):
        materialize_research_corpus(
            _report(),
            monthly_root=tmp_path,
            output_root=tmp_path / "research",
            limits=ResearchCorpusLimits(maximum_archives=1, maximum_rows=2),
        )
