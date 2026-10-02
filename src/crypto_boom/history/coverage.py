"""Inferred historical source coverage from availability and decoded archives."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import msgspec

from crypto_boom import _artifacts
from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
)
from crypto_boom.history.codec import ArchiveError
from crypto_boom.history.monthly import (
    MonthlyArchiveManifest,
    load_published_monthly_archive,
)
from crypto_boom.market import MetadataObservation

COVERAGE_POLICY_VERSION: Final = "binance-archive-coverage-v1"
_MAXIMUM_SYMBOLS = 500


class HistoricalCoverageError(RuntimeError):
    """Historical source coverage could not be established safely."""


class HistoricalCoverageResourceError(HistoricalCoverageError):
    """Coverage work exceeded its admitted resource envelope."""


class HistoricalCoverageIntegrityError(HistoricalCoverageError):
    """Coverage inputs or a published report are inconsistent."""


class HistoricalCoveragePublicationError(HistoricalCoverageError):
    """A complete coverage report could not be atomically published."""


@dataclass(frozen=True, slots=True)
class CoverageLimits:
    """Bounds for synchronous local coverage aggregation."""

    maximum_archives: int = 500
    maximum_report_bytes: int = 16 * 1_024 * 1_024

    def __post_init__(self) -> None:
        if self.maximum_archives <= 0 or self.maximum_report_bytes <= 0:
            raise ValueError("coverage limits must be positive")


DEFAULT_COVERAGE_LIMITS = CoverageLimits()


@dataclass(frozen=True, slots=True)
class CoverageInterval:
    """Inclusive contiguous UTC-day interval with decoded rows."""

    start_day: str
    end_day: str

    def __post_init__(self) -> None:
        start = _parse_day(self.start_day)
        end = _parse_day(self.end_day)
        if end < start:
            raise HistoricalCoverageIntegrityError("coverage interval is reversed")

    def to_mapping(self) -> dict[str, object]:
        return {"end_day": self.end_day, "start_day": self.start_day}


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    """Inferred source-presence evidence for one canonical symbol."""

    symbol: str
    available_months: tuple[str, ...]
    not_found_months: tuple[str, ...]
    observed_intervals: tuple[CoverageInterval, ...]
    missing_days: tuple[str, ...]
    internal_gap_count: int
    internal_missing_minutes: int
    source_manifest_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise HistoricalCoverageIntegrityError("coverage symbol is empty")
        if tuple(sorted(self.available_months)) != self.available_months:
            raise HistoricalCoverageIntegrityError("available months are not ordered")
        if tuple(sorted(self.not_found_months)) != self.not_found_months:
            raise HistoricalCoverageIntegrityError("not-found months are not ordered")
        if set(self.available_months) & set(self.not_found_months):
            raise HistoricalCoverageIntegrityError("coverage month states overlap")
        if len(self.available_months) != len(self.source_manifest_ids):
            raise HistoricalCoverageIntegrityError(
                "coverage source lineage is incomplete"
            )
        if any(not _artifacts.is_sha256(value) for value in self.source_manifest_ids):
            raise HistoricalCoverageIntegrityError(
                "coverage source identity is invalid"
            )
        if tuple(sorted(self.missing_days)) != self.missing_days:
            raise HistoricalCoverageIntegrityError(
                "missing coverage days are not ordered"
            )
        if self.internal_gap_count < 0 or self.internal_missing_minutes < 0:
            raise HistoricalCoverageIntegrityError("coverage gap totals are invalid")
        previous_end: date | None = None
        for interval in self.observed_intervals:
            start = _parse_day(interval.start_day)
            end = _parse_day(interval.end_day)
            if previous_end is not None and start <= previous_end + timedelta(days=1):
                raise HistoricalCoverageIntegrityError(
                    "coverage intervals overlap or are not maximally coalesced"
                )
            previous_end = end

    @property
    def observed_first_day(self) -> str | None:
        if not self.observed_intervals:
            return None
        return self.observed_intervals[0].start_day

    @property
    def observed_last_day(self) -> str | None:
        if not self.observed_intervals:
            return None
        return self.observed_intervals[-1].end_day

    def to_mapping(self) -> dict[str, object]:
        return {
            "available_months": list(self.available_months),
            "internal_gap_count": self.internal_gap_count,
            "internal_missing_minutes": self.internal_missing_minutes,
            "missing_days": list(self.missing_days),
            "not_found_months": list(self.not_found_months),
            "observation": MetadataObservation.INFERRED.value,
            "observed_first_day": self.observed_first_day,
            "observed_intervals": [
                item.to_mapping() for item in self.observed_intervals
            ],
            "observed_last_day": self.observed_last_day,
            "source_manifest_ids": list(self.source_manifest_ids),
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class HistoricalCoverageReport:
    """Content-addressed inferred coverage for an explicit report subset."""

    schema_version: int
    policy_version: str
    availability_report_id: str
    pool_id: str
    interval: str
    start_month: str
    end_month: str
    symbols: tuple[str, ...]
    coverage: tuple[SymbolCoverage, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.policy_version != COVERAGE_POLICY_VERSION:
            raise HistoricalCoverageIntegrityError("coverage version is invalid")
        if not _artifacts.is_sha256(
            self.availability_report_id
        ) or not _artifacts.is_sha256(self.pool_id):
            raise HistoricalCoverageIntegrityError(
                "coverage source identity is invalid"
            )
        if self.interval != "1m" or not self.symbols:
            raise HistoricalCoverageIntegrityError("coverage scope is invalid")
        if len(self.symbols) > _MAXIMUM_SYMBOLS:
            raise HistoricalCoverageResourceError("coverage cannot exceed 500 symbols")
        if tuple(sorted(self.symbols)) != self.symbols or len(set(self.symbols)) != len(
            self.symbols
        ):
            raise HistoricalCoverageIntegrityError("coverage symbols are invalid")
        if tuple(item.symbol for item in self.coverage) != self.symbols:
            raise HistoricalCoverageIntegrityError("coverage rows do not match scope")

    @property
    def available_archive_count(self) -> int:
        return sum(len(item.available_months) for item in self.coverage)

    @property
    def not_found_archive_count(self) -> int:
        return sum(len(item.not_found_months) for item in self.coverage)

    def _content_mapping(self) -> dict[str, object]:
        return {
            "availability_report_id": self.availability_report_id,
            "available_archive_count": self.available_archive_count,
            "coverage": [item.to_mapping() for item in self.coverage],
            "end_month": self.end_month,
            "interval": self.interval,
            "not_found_archive_count": self.not_found_archive_count,
            "observation": MetadataObservation.INFERRED.value,
            "policy_version": self.policy_version,
            "pool_id": self.pool_id,
            "schema_version": self.schema_version,
            "start_month": self.start_month,
            "symbols": list(self.symbols),
        }

    @property
    def report_id(self) -> str:
        digest = hashlib.sha256(
            _artifacts.canonical_json(self._content_mapping())
        ).hexdigest()
        return f"sha256:{digest}"

    def to_mapping(self) -> dict[str, object]:
        return {"report_id": self.report_id, **self._content_mapping()}


_COVERAGE_DECODER = msgspec.json.Decoder(HistoricalCoverageReport)


@dataclass(frozen=True, slots=True)
class PublishedHistoricalCoverage:
    path: Path
    report: HistoricalCoverageReport
    already_present: bool


def build_historical_coverage(
    availability: ArchiveAvailabilityReport,
    *,
    monthly_root: Path,
    output_root: Path,
    symbols: tuple[str, ...] = (),
    limits: CoverageLimits = DEFAULT_COVERAGE_LIMITS,
) -> PublishedHistoricalCoverage:
    """Verify selected monthly publications and aggregate inferred coverage."""

    if availability.unresolved_count:
        raise HistoricalCoverageIntegrityError(
            "coverage cannot consume unresolved availability observations"
        )
    selected = _select_symbols(availability, symbols)
    selected_probes = tuple(
        probe for probe in availability.probes if probe.symbol in selected
    )
    available_count = sum(
        probe.state is AvailabilityState.AVAILABLE for probe in selected_probes
    )
    if available_count > limits.maximum_archives:
        raise HistoricalCoverageResourceError(
            f"coverage requires {available_count} archives; "
            f"limit is {limits.maximum_archives}"
        )

    monthly_root = monthly_root.resolve()
    grouped: dict[
        str,
        list[tuple[MonthlyArchiveProbe, MonthlyArchiveManifest | None]],
    ] = {symbol: [] for symbol in selected}
    for probe in selected_probes:
        manifest: MonthlyArchiveManifest | None = None
        if probe.state is AvailabilityState.AVAILABLE:
            if probe.checksum_sha256 is None:
                raise HistoricalCoverageIntegrityError(
                    "available observation has no checksum"
                )
            path = (
                monthly_root
                / "binance"
                / "spot"
                / "monthly-klines"
                / probe.symbol
                / availability.interval
                / probe.month
                / probe.checksum_sha256.removeprefix("sha256:")
            )
            try:
                published = load_published_monthly_archive(path)
            except ArchiveError as error:
                raise HistoricalCoverageIntegrityError(
                    f"verified monthly publication is unavailable for "
                    f"{probe.symbol} {probe.month}"
                ) from error
            manifest = published.manifest
            if (
                manifest.symbol != probe.symbol
                or manifest.month != probe.month
                or manifest.interval != availability.interval
                or manifest.archive_sha256 != probe.checksum_sha256
            ):
                raise HistoricalCoverageIntegrityError(
                    "monthly publication conflicts with availability observation"
                )
        grouped[probe.symbol].append((probe, manifest))

    rows = tuple(
        _aggregate_symbol(symbol, grouped[symbol]) for symbol in sorted(selected)
    )
    report = HistoricalCoverageReport(
        schema_version=1,
        policy_version=COVERAGE_POLICY_VERSION,
        availability_report_id=availability.report_id,
        pool_id=availability.pool_id,
        interval=availability.interval,
        start_month=availability.start_month,
        end_month=availability.end_month,
        symbols=tuple(sorted(selected)),
        coverage=rows,
    )
    return _publish_report(report, output_root=output_root, limits=limits)


def load_published_historical_coverage(path: Path) -> HistoricalCoverageReport:
    """Strictly reload one content-addressed coverage report."""

    path = path.resolve()
    try:
        payload = (path / "coverage.json").read_bytes()
    except OSError as error:
        raise HistoricalCoveragePublicationError(
            "published coverage report is unavailable"
        ) from error
    if len(payload) > DEFAULT_COVERAGE_LIMITS.maximum_report_bytes:
        raise HistoricalCoverageResourceError("coverage report exceeds byte limit")
    try:
        report = _COVERAGE_DECODER.decode(payload)
        if payload != _artifacts.canonical_json(report.to_mapping()):
            raise ValueError
    except (msgspec.DecodeError, ValueError) as error:
        raise HistoricalCoverageIntegrityError("coverage report is invalid") from error
    if path.name != report.report_id.removeprefix("sha256:"):
        raise HistoricalCoverageIntegrityError(
            "coverage report path does not match its identity"
        )
    return report


def _select_symbols(
    availability: ArchiveAvailabilityReport,
    requested: tuple[str, ...],
) -> frozenset[str]:
    report_symbols = {probe.symbol for probe in availability.probes}
    if not requested:
        selected = report_symbols
    else:
        if len(requested) != len(set(requested)):
            raise HistoricalCoverageIntegrityError(
                "coverage symbol filter contains duplicates"
            )
        unknown = sorted(set(requested) - report_symbols)
        if unknown:
            raise HistoricalCoverageIntegrityError(
                "coverage symbol filter is not present in availability report: "
                + ", ".join(unknown)
            )
        selected = set(requested)
    if not selected:
        raise HistoricalCoverageIntegrityError("coverage requires at least one symbol")
    if len(selected) > _MAXIMUM_SYMBOLS:
        raise HistoricalCoverageResourceError("coverage cannot exceed 500 symbols")
    return frozenset(selected)


def _aggregate_symbol(
    symbol: str,
    observations: list[tuple[MonthlyArchiveProbe, MonthlyArchiveManifest | None]],
) -> SymbolCoverage:
    available_months: list[str] = []
    not_found_months: list[str] = []
    observed_days: set[str] = set()
    missing_days: set[str] = set()
    manifest_ids: list[str] = []
    internal_gap_count = 0
    internal_missing_minutes = 0
    for probe, manifest in observations:
        if probe.state is AvailabilityState.NOT_FOUND:
            not_found_months.append(probe.month)
            continue
        if probe.state is not AvailabilityState.AVAILABLE or manifest is None:
            raise HistoricalCoverageIntegrityError(
                "coverage encountered an inadmissible availability state"
            )
        available_months.append(probe.month)
        observed_days.update(manifest.observed_days)
        missing_days.update(manifest.missing_days)
        manifest_ids.append(manifest.manifest_id)
        internal_gap_count += manifest.internal_gap_count
        internal_missing_minutes += manifest.internal_missing_minutes
    return SymbolCoverage(
        symbol=symbol,
        available_months=tuple(available_months),
        not_found_months=tuple(not_found_months),
        observed_intervals=_coalesce_days(observed_days),
        missing_days=tuple(sorted(missing_days)),
        internal_gap_count=internal_gap_count,
        internal_missing_minutes=internal_missing_minutes,
        source_manifest_ids=tuple(manifest_ids),
    )


def _coalesce_days(values: set[str]) -> tuple[CoverageInterval, ...]:
    days = sorted(_parse_day(value) for value in values)
    if not days:
        return ()
    intervals: list[CoverageInterval] = []
    start = days[0]
    end = days[0]
    for current in days[1:]:
        if current == end + timedelta(days=1):
            end = current
            continue
        intervals.append(CoverageInterval(start.isoformat(), end.isoformat()))
        start = current
        end = current
    intervals.append(CoverageInterval(start.isoformat(), end.isoformat()))
    return tuple(intervals)


def _publish_report(
    report: HistoricalCoverageReport,
    *,
    output_root: Path,
    limits: CoverageLimits,
) -> PublishedHistoricalCoverage:
    payload = _artifacts.canonical_json(report.to_mapping())
    if len(payload) > limits.maximum_report_bytes:
        raise HistoricalCoverageResourceError("coverage report exceeds byte limit")
    output_root = output_root.resolve()
    target = (
        output_root
        / "binance"
        / "spot"
        / "historical-coverage"
        / report.pool_id.removeprefix("sha256:")
        / f"{report.start_month}_{report.end_month}"
        / report.report_id.removeprefix("sha256:")
    )
    staging_parent = output_root / ".staging"
    try:
        staging_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise HistoricalCoveragePublicationError(
            "coverage staging root is unavailable"
        ) from error
    with _artifacts.publication_staging_directory(
        staging_parent,
        prefix="historical-coverage-",
    ) as staging:
        _write_bytes(staging / "coverage.json", payload)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                _verify_existing(target, payload)
                return PublishedHistoricalCoverage(target, report, True)
            os.replace(staging, target)
        except HistoricalCoverageError:
            raise
        except OSError as error:
            if target.exists():
                _verify_existing(target, payload)
                return PublishedHistoricalCoverage(target, report, True)
            raise HistoricalCoveragePublicationError(
                "atomic coverage publication failed"
            ) from error
    return PublishedHistoricalCoverage(target, report, False)


def _verify_existing(target: Path, expected: bytes) -> None:
    try:
        actual = (target / "coverage.json").read_bytes()
    except OSError as error:
        raise HistoricalCoveragePublicationError(
            "existing coverage report is unavailable"
        ) from error
    if actual != expected:
        raise HistoricalCoveragePublicationError(
            "existing coverage report conflicts with content"
        )


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise HistoricalCoveragePublicationError(
            "coverage staging write failed"
        ) from error


def _parse_day(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise HistoricalCoverageIntegrityError("coverage day is invalid") from error
    if parsed.isoformat() != value:
        raise HistoricalCoverageIntegrityError("coverage day is invalid")
    return parsed
