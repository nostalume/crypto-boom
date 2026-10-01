from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import crypto_boom.history.coverage as historical_coverage
from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
)
from crypto_boom.history.coverage import (
    CoverageLimits,
    HistoricalCoverageIntegrityError,
    HistoricalCoverageResourceError,
    build_historical_coverage,
    load_published_historical_coverage,
)
from crypto_boom.history.monthly import MonthlyArchiveManifest, MonthlyGap


def _availability(*, unresolved: bool = False) -> ArchiveAvailabilityReport:
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
                "sha256:" + "b" * 64 if not unresolved else None,
                1,
                200 if not unresolved else 500,
            ),
            MonthlyArchiveProbe(
                "SOLUSDT",
                "2025-01",
                AvailabilityState.NOT_FOUND,
                None,
                1,
                404,
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
        source_revision="sha256:" + "b" * 64,
        archive_sha256="sha256:" + "b" * 64,
        member_sha256="sha256:" + "c" * 64,
        compressed_bytes=100,
        uncompressed_bytes=1_000,
        row_count=2_880,
        first_open_time_us=1_735_776_000_000_000,
        last_open_time_us=1_736_035_140_000_000,
        observed_first_day="2025-01-02",
        observed_last_day="2025-01-05",
        observed_days=("2025-01-02", "2025-01-03", "2025-01-05"),
        missing_days=("2025-01-01", "2025-01-04"),
        internal_gaps=(
            MonthlyGap(
                1_735_948_800_000_000,
                1_735_948_800_000_000 + (1_439 - 1) * 60_000_000,
                1_439,
            ),
        ),
    )


def test_coverage_preserves_presence_absence_intervals_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    assert manifest.manifest_id == (
        "sha256:b6c20be9c0543ec177dc6555eecedd9dd8fe3e029cbac1f0ed07f2f668769547"
    )
    observed_paths: list[Path] = []

    def load(path: Path):
        observed_paths.append(path)
        return SimpleNamespace(manifest=manifest)

    monkeypatch.setattr(historical_coverage, "load_published_monthly_archive", load)
    availability = _availability()
    assert availability.report_id == (
        "sha256:2f09ca559cfb19cbdc50f7345d9ca436cc68d8a1e9b8e4c63113bdcdf695cbe4"
    )
    first = build_historical_coverage(
        availability,
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "coverage",
    )
    second = build_historical_coverage(
        availability,
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "coverage",
    )

    assert not first.already_present
    assert second.already_present
    assert first.report.report_id == (
        "sha256:570905c9fb2e69867ae802d9ff6b4b0116cde69011490b99373ff57850a2cb29"
    )
    assert first.report.available_archive_count == 1
    assert first.report.not_found_archive_count == 1
    eth, sol = first.report.coverage
    assert eth.symbol == "ETHUSDT"
    assert [(item.start_day, item.end_day) for item in eth.observed_intervals] == [
        ("2025-01-02", "2025-01-03"),
        ("2025-01-05", "2025-01-05"),
    ]
    assert eth.missing_days == ("2025-01-01", "2025-01-04")
    assert eth.internal_gap_count == 1
    assert eth.internal_missing_minutes == 1_439
    assert eth.source_manifest_ids == (manifest.manifest_id,)
    assert sol.symbol == "SOLUSDT"
    assert sol.not_found_months == ("2025-01",)
    assert sol.observed_intervals == ()
    expected_path = (
        (tmp_path / "monthly").resolve()
        / "binance/spot/monthly-klines/ETHUSDT/1m/2025-01"
        / ("b" * 64)
    )
    assert observed_paths == [expected_path, expected_path]
    assert load_published_historical_coverage(first.path) == first.report
    document = json.loads((first.path / "coverage.json").read_text())
    assert document["observation"] == "inferred"


def test_coverage_refuses_unresolved_unknown_oversized_and_tampered_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        historical_coverage,
        "load_published_monthly_archive",
        lambda path: SimpleNamespace(manifest=_manifest()),
    )
    with pytest.raises(HistoricalCoverageIntegrityError, match="unresolved"):
        build_historical_coverage(
            _availability(unresolved=True),
            monthly_root=tmp_path,
            output_root=tmp_path,
        )
    with pytest.raises(HistoricalCoverageIntegrityError, match="not present"):
        build_historical_coverage(
            _availability(),
            monthly_root=tmp_path,
            output_root=tmp_path,
            symbols=("UNKNOWNUSDT",),
        )
    availability = _availability()
    two_available = replace(
        availability,
        probes=(
            availability.probes[0],
            MonthlyArchiveProbe(
                "SOLUSDT",
                "2025-01",
                AvailabilityState.AVAILABLE,
                "sha256:" + "d" * 64,
                1,
                200,
            ),
        ),
    )
    with pytest.raises(HistoricalCoverageResourceError, match="2 archives"):
        build_historical_coverage(
            two_available,
            monthly_root=tmp_path,
            output_root=tmp_path,
            limits=CoverageLimits(maximum_archives=1),
        )


def test_coverage_reload_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        historical_coverage,
        "load_published_monthly_archive",
        lambda path: SimpleNamespace(manifest=_manifest()),
    )
    published = build_historical_coverage(
        _availability(),
        monthly_root=tmp_path / "monthly",
        output_root=tmp_path / "coverage",
    )
    document = json.loads((published.path / "coverage.json").read_text())
    document["coverage"][0]["internal_gap_count"] = True
    (published.path / "coverage.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(HistoricalCoverageIntegrityError, match="report is invalid"):
        load_published_historical_coverage(published.path)

    (published.path / "coverage.json").write_text("{}")
    with pytest.raises(HistoricalCoverageIntegrityError, match="coverage report"):
        load_published_historical_coverage(published.path)
