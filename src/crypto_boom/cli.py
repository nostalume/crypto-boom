"""Command-line adapter for Crypto Boom."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import aiohttp

from crypto_boom.binance_source import (
    LiveCaptureError,
    MetadataCapture,
    RestWeightBudget,
    fetch_exchange_info,
    measure_exchange_clock,
)
from crypto_boom.config import ConfigAdmissionError, admit_run_config
from crypto_boom.history.availability import (
    AvailabilityError,
    AvailabilityLimits,
    MonthlyAvailabilityRequest,
    PublishedArchiveAvailability,
    load_published_archive_availability,
    probe_monthly_archive_availability,
)
from crypto_boom.history.coverage import (
    CoverageLimits,
    HistoricalCoverageError,
    build_historical_coverage,
)
from crypto_boom.history.daily import (
    ArchiveBatchLimits,
    ArchiveDayRequest,
    ArchiveError,
    ArchiveRangeRequest,
    acquire_archive_day,
    acquire_archive_range,
)
from crypto_boom.history.monthly import (
    MonthlyArchiveLimits,
    acquire_monthly_archives,
)
from crypto_boom.live import SubscriptionPlan
from crypto_boom.market import (
    Environment,
    EvidenceAdmissionError,
    InstrumentId,
    VenueId,
)
from crypto_boom.qualification import (
    DEFAULT_MAXIMUM_CAMPAIGN_UNCOMPRESSED_BYTES,
    BinanceQualificationContextProvider,
    PublishedQualification,
    QualificationDayContext,
    QualificationError,
    QualificationLimits,
    QualificationServiceStatus,
    RollingCrossDayQualityPublisher,
    run_continuous_qualification_campaign,
    run_qualification_campaign,
    select_spot_usdt_universe,
)
from crypto_boom.storage.source import (
    ResearchCorpusLimits,
    ResearchStorageError,
    materialize_research_corpus,
)
from crypto_boom.universe import (
    PoolError,
    PoolPolicy,
    PublishedResearchPool,
    build_research_pool,
    load_published_research_pool,
    publish_research_pool,
)

MAX_CONFIG_BYTES = 65_536


class ConfigFileError(ValueError):
    """The configuration file could not be safely read."""


def _add_qualification_arguments(
    parser: argparse.ArgumentParser,
    *,
    bounded: bool,
) -> None:
    if bounded:
        parser.add_argument(
            "--duration-seconds",
            required=True,
            type=_positive_float,
        )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--maximum-uncompressed-capture-bytes",
        default=DEFAULT_MAXIMUM_CAMPAIGN_UNCOMPRESSED_BYTES,
        type=_positive_int,
        help="fail before retained capture exceeds this logical byte limit",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        default=30.0,
        type=_positive_float,
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="restrict breadth capture to a repeated observed USDT Spot symbol",
    )
    parser.add_argument(
        "--hot-symbol",
        action="append",
        default=[],
        help="add a repeated symbol to bounded trade/quote capture",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-boom")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke",
        help="validate configuration and emit a read-only run context",
    )
    smoke.add_argument("--config", required=True, type=Path)
    archive_day = subparsers.add_parser(
        "archive-day",
        help="acquire one verified Binance Spot one-minute kline archive",
    )
    archive_day.add_argument("--symbol", required=True)
    archive_day.add_argument("--day", required=True, type=_iso_day)
    archive_day.add_argument("--output", required=True, type=Path)
    archive_range = subparsers.add_parser(
        "archive-range",
        help="acquire a bounded Binance Spot symbol/date archive range",
    )
    range_source = archive_range.add_mutually_exclusive_group(required=True)
    range_source.add_argument("--symbol", action="append")
    range_source.add_argument("--pool", type=Path)
    archive_range.add_argument("--start-day", required=True, type=_iso_day)
    archive_range.add_argument("--end-day", required=True, type=_iso_day)
    archive_range.add_argument("--output", required=True, type=Path)
    archive_range.add_argument(
        "--maximum-partitions",
        default=10_000,
        type=_positive_int,
        help="fail before scheduling more daily symbol partitions than this bound",
    )
    instrument_pool = subparsers.add_parser(
        "instrument-pool",
        help="publish a current observed Binance Spot altcoin research pool",
    )
    instrument_pool.add_argument("--output", required=True, type=Path)
    instrument_pool.add_argument(
        "--maximum-instruments",
        default=500,
        type=_pool_limit,
    )
    instrument_pool.add_argument(
        "--exclude-symbol",
        action="append",
        default=[],
        help="exclude a repeated canonical symbol from the versioned pool",
    )
    archive_availability = subparsers.add_parser(
        "archive-availability",
        help="audit monthly Binance archive availability for a verified pool",
    )
    archive_availability.add_argument("--pool", required=True, type=Path)
    archive_availability.add_argument(
        "--start-month",
        required=True,
        type=_iso_month,
    )
    archive_availability.add_argument(
        "--end-month",
        required=True,
        type=_iso_month,
    )
    archive_availability.add_argument("--output", required=True, type=Path)
    archive_availability.add_argument(
        "--maximum-probes",
        default=12_000,
        type=_positive_int,
    )
    archive_availability.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="restrict the audit to a repeated symbol from the verified pool",
    )
    archive_monthly = subparsers.add_parser(
        "archive-monthly",
        help="acquire AVAILABLE monthly archives from a verified audit",
    )
    archive_monthly.add_argument("--availability", required=True, type=Path)
    archive_monthly.add_argument("--output", required=True, type=Path)
    archive_monthly.add_argument(
        "--maximum-archives",
        default=500,
        type=_positive_int,
    )
    archive_monthly.add_argument(
        "--maximum-concurrency",
        default=4,
        type=_positive_int,
    )
    archive_monthly.add_argument(
        "--maximum-decode-processes",
        default=1,
        type=_positive_int,
        choices=(1, 2),
    )
    archive_monthly.add_argument(
        "--maximum-total-compressed-bytes",
        default=1 * 1_024 * 1_024 * 1_024,
        type=_positive_int,
    )
    archive_monthly.add_argument(
        "--maximum-elapsed-seconds",
        default=300.0,
        type=_positive_float,
    )
    archive_monthly.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="restrict acquisition to a repeated symbol from the audit",
    )
    historical_coverage = subparsers.add_parser(
        "historical-coverage",
        help="aggregate inferred decoded coverage without inventing historical rules",
    )
    historical_coverage.add_argument("--availability", required=True, type=Path)
    historical_coverage.add_argument("--monthly-root", required=True, type=Path)
    historical_coverage.add_argument("--output", required=True, type=Path)
    historical_coverage.add_argument(
        "--maximum-archives",
        default=500,
        type=_positive_int,
    )
    historical_coverage.add_argument("--symbol", action="append", default=[])
    research_corpus = subparsers.add_parser(
        "research-corpus",
        help="materialize verified monthly sources as source-only research Parquet",
    )
    research_corpus.add_argument("--availability", required=True, type=Path)
    research_corpus.add_argument("--monthly-root", required=True, type=Path)
    research_corpus.add_argument("--output", required=True, type=Path)
    research_corpus.add_argument("--symbol", action="append", default=[])
    research_corpus.add_argument(
        "--exclude-symbol-month",
        action="append",
        default=[],
        metavar="SYMBOL:MONTH",
        help=(
            "leave one audited-available symbol-month undecoded, e.g. REDUSDT:2025-03, because the "
            "published archive violates the kline contract; repeatable"
        ),
    )
    research_corpus.add_argument(
        "--maximum-archives",
        default=500,
        type=_positive_int,
    )
    research_corpus.add_argument(
        "--maximum-rows",
        default=25_000_000,
        type=_positive_int,
    )
    research_corpus.add_argument(
        "--maximum-source-uncompressed-bytes",
        default=8 * 1_024 * 1_024 * 1_024,
        type=_positive_int,
    )
    research_corpus.add_argument(
        "--maximum-parquet-bytes",
        default=8 * 1_024 * 1_024 * 1_024,
        type=_positive_int,
    )
    research_corpus.add_argument(
        "--maximum-elapsed-seconds",
        default=300.0,
        type=_positive_float,
    )
    research_corpus.add_argument(
        "--maximum-concurrency",
        default=1,
        type=_positive_int,
        choices=(1, 2),
    )
    qualify_live = subparsers.add_parser(
        "qualify-live",
        help="run a bounded read-only prospective qualification campaign",
    )
    _add_qualification_arguments(qualify_live, bounded=True)
    qualify_service = subparsers.add_parser(
        "qualify-live-service",
        help="run continuous read-only qualification until SIGINT or SIGTERM",
    )
    _add_qualification_arguments(qualify_service, bounded=False)
    return parser


def _iso_day(raw: str) -> date:
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("day must use YYYY-MM-DD") from error
    if parsed.isoformat() != raw:
        raise argparse.ArgumentTypeError("day must use YYYY-MM-DD")
    return parsed


def _iso_month(raw: str) -> date:
    try:
        parsed = date.fromisoformat(f"{raw}-01")
    except ValueError as error:
        raise argparse.ArgumentTypeError("month must use YYYY-MM") from error
    if parsed.strftime("%Y-%m") != raw:
        raise argparse.ArgumentTypeError("month must use YYYY-MM")
    return parsed


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return value


def _symbol_month_pairs(values: Sequence[str]) -> frozenset[tuple[str, str]]:
    """Parse `SYMBOL:MONTH` exclusions, refusing anything that is not a symbol-month pair.

    Parsed strictly rather than leniently: an exclusion that silently matched nothing would look
    like a protection while protecting nothing, and the run it was meant to unblock would fail
    later, far from this argument. `ValueError` rather than `argparse.ArgumentTypeError` because this
    is called from the command handler, whose `except ... ValueError` turns it into the documented
    status 2 instead of a traceback.
    """
    pairs: set[tuple[str, str]] = set()
    for value in values:
        symbol, separator, month = value.partition(":")
        if not separator or not symbol or not month:
            raise ValueError(f"exclusion must be SYMBOL:MONTH, not {value!r}")
        pairs.add((symbol, month))
    return frozenset(pairs)


def _pool_limit(raw: str) -> int:
    value = _positive_int(raw)
    if value > 500:
        raise argparse.ArgumentTypeError("instrument pool limit cannot exceed 500")
    return value


async def _capture_research_pool(
    *,
    run_id: UUID,
    output: Path,
    maximum_instruments: int,
    excluded_symbols: tuple[str, ...],
) -> PublishedResearchPool:
    timeout = aiohttp.ClientTimeout(total=30)
    budget = RestWeightBudget(venue_capacity=6_000, interval_seconds=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        metadata = await fetch_exchange_info(
            session,
            ingestion_run_id=run_id,
            budget=budget,
        )
    policy = PoolPolicy(
        maximum_instruments=maximum_instruments,
        excluded_symbols=tuple(sorted(set(excluded_symbols))),
    )
    pool = build_research_pool(metadata, policy=policy)
    return publish_research_pool(pool, metadata=metadata, output_root=output)


async def _audit_archive_availability(
    *,
    pool_path: Path,
    start_month: date,
    end_month: date,
    maximum_probes: int,
    symbols: tuple[str, ...],
    output: Path,
) -> PublishedArchiveAvailability:
    pool = load_published_research_pool(pool_path)
    request = MonthlyAvailabilityRequest(start_month, end_month)
    return await probe_monthly_archive_availability(
        pool,
        request,
        output_root=output,
        symbols=symbols,
        limits=AvailabilityLimits(maximum_probes=maximum_probes),
    )


async def _qualify_live(
    *,
    run_id: UUID,
    output: Path,
    duration_seconds: float,
    requested_symbols: tuple[str, ...],
    hot_symbols: tuple[str, ...],
    startup_timeout_seconds: float,
    maximum_uncompressed_capture_bytes: int,
) -> PublishedQualification:
    budget = RestWeightBudget(venue_capacity=6_000, interval_seconds=60)
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=15, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        metadata = await fetch_exchange_info(
            session,
            ingestion_run_id=run_id,
            budget=budget,
        )
        clock = await measure_exchange_clock(session, budget=budget)

    plan = _qualification_plan(metadata, requested_symbols, hot_symbols)
    return await run_qualification_campaign(
        plan,
        metadata=metadata,
        output_root=output,
        duration_seconds=duration_seconds,
        limits=QualificationLimits(
            maximum_campaign_uncompressed_bytes=(maximum_uncompressed_capture_bytes)
        ),
        clock_offset_ms=clock.offset_ms,
        startup_timeout_seconds=startup_timeout_seconds,
    )


async def _qualify_live_service(
    *,
    run_id: UUID,
    output: Path,
    requested_symbols: tuple[str, ...],
    hot_symbols: tuple[str, ...],
    startup_timeout_seconds: float,
    maximum_uncompressed_capture_bytes: int,
) -> tuple[PublishedQualification, ...]:
    aggregate_publisher = RollingCrossDayQualityPublisher(output)
    hydrated_days = await asyncio.to_thread(aggregate_publisher.hydrate_from_disk)
    _emit_json(
        {
            "event": "quality_aggregate_window_hydrated",
            "hydrated_days": hydrated_days,
            "run_id": str(run_id),
        }
    )
    budget = RestWeightBudget(venue_capacity=6_000, interval_seconds=60)
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=15, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        metadata = await fetch_exchange_info(
            session,
            ingestion_run_id=run_id,
            budget=budget,
        )
        clock = await measure_exchange_clock(session, budget=budget)
        plan = _qualification_plan(metadata, requested_symbols, hot_symbols)
        provider = BinanceQualificationContextProvider(
            session=session,
            ingestion_run_id=run_id,
            budget=budget,
        )
        stop = asyncio.Event()
        restore_signals = _install_shutdown_signals(stop)
        try:

            async def emit_status(status: QualificationServiceStatus) -> None:
                _emit_json(
                    {
                        "detail": status.detail,
                        "event": "service_status",
                        "phase": status.phase.value,
                        "ready": status.ready,
                        "run_id": str(run_id),
                    }
                )

            async def emit_publication(result: PublishedQualification) -> None:
                _emit_json(
                    {
                        "day": result.report.day,
                        "event": "qualification_published",
                        "manifest_id": result.manifest.manifest_id,
                        "path": str(result.path),
                        "ready": result.report.ready,
                        "report_id": result.report.report_id,
                        "run_id": str(run_id),
                    }
                )
                aggregate = await asyncio.to_thread(
                    aggregate_publisher.observe,
                    result,
                )
                if aggregate is not None:
                    _emit_json(
                        {
                            "aggregate_id": aggregate.report.aggregate_id,
                            "end_day": aggregate.report.end_day,
                            "event": "quality_aggregate_published",
                            "manifest_id": aggregate.manifest.manifest_id,
                            "path": str(aggregate.path),
                            "ready": aggregate.report.ready,
                            "run_id": str(run_id),
                            "start_day": aggregate.report.start_day,
                        }
                    )

            return await run_continuous_qualification_campaign(
                plan,
                initial_context=QualificationDayContext(
                    metadata=metadata,
                    clock_offset_ms=clock.offset_ms,
                ),
                context_for=provider,
                output_root=output,
                stop=stop,
                limits=QualificationLimits(
                    maximum_campaign_uncompressed_bytes=(
                        maximum_uncompressed_capture_bytes
                    )
                ),
                startup_timeout_seconds=startup_timeout_seconds,
                on_published=emit_publication,
                on_status=emit_status,
            )
        finally:
            restore_signals()


def _qualification_plan(
    metadata: MetadataCapture,
    requested_symbols: tuple[str, ...],
    hot_symbols: tuple[str, ...],
) -> SubscriptionPlan:
    universe = select_spot_usdt_universe(metadata.events)
    selected = requested_symbols or universe
    unknown = sorted((set(selected) | set(hot_symbols)) - set(universe))
    if unknown:
        raise QualificationError(
            "requested symbol is not in the observed trading USDT Spot universe"
        )
    return SubscriptionPlan(
        breadth_symbols=selected,
        hot_symbols=hot_symbols,
    )


def _install_shutdown_signals(stop: asyncio.Event) -> Callable[[], None]:
    loop = asyncio.get_running_loop()
    loop_handlers: list[signal.Signals] = []
    fallback_handlers: list[tuple[signal.Signals, Any]] = []

    def request_stop(*_: object) -> None:
        loop.call_soon_threadsafe(stop.set)

    shutdown_signals = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if isinstance(sigbreak, signal.Signals):
        shutdown_signals.append(sigbreak)
    for signum in shutdown_signals:
        try:
            loop.add_signal_handler(signum, stop.set)
            loop_handlers.append(signum)
        except (NotImplementedError, RuntimeError):
            previous = signal.getsignal(signum)
            signal.signal(signum, request_stop)
            fallback_handlers.append((signum, previous))

    def restore() -> None:
        for signum in loop_handlers:
            loop.remove_signal_handler(signum)
        for signum, previous in fallback_handlers:
            signal.signal(signum, previous)

    return restore


def _emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True), flush=True)


def _read_config_document(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_CONFIG_BYTES + 1)
    except OSError as error:
        raise ConfigFileError("configuration file is unavailable") from error

    if len(payload) > MAX_CONFIG_BYTES:
        raise ConfigFileError(f"configuration file exceeds {MAX_CONFIG_BYTES} bytes")

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigFileError("configuration file is not UTF-8") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line adapter and return its process status."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "research-corpus":

        def emit_partition(publication: Any) -> None:
            manifest = publication.manifest
            _emit_json(
                {
                    "already_present": publication.already_present,
                    "event": "research_partition",
                    "manifest_id": manifest.manifest_id,
                    "month": manifest.month,
                    "parquet_bytes": manifest.parquet_bytes,
                    "path": str(publication.path),
                    "row_count": manifest.row_count,
                    "source_manifest_id": manifest.source_manifest_id,
                    "symbol": manifest.symbol,
                }
            )

        try:
            availability = load_published_archive_availability(
                cast(Path, arguments.availability)
            )
            result = materialize_research_corpus(
                availability,
                monthly_root=cast(Path, arguments.monthly_root),
                output_root=cast(Path, arguments.output),
                symbols=tuple(cast(list[str], arguments.symbol)),
                excluded=_symbol_month_pairs(
                    cast(list[str], arguments.exclude_symbol_month)
                ),
                limits=ResearchCorpusLimits(
                    maximum_archives=cast(int, arguments.maximum_archives),
                    maximum_rows=cast(int, arguments.maximum_rows),
                    maximum_source_uncompressed_bytes=cast(
                        int,
                        arguments.maximum_source_uncompressed_bytes,
                    ),
                    maximum_parquet_bytes=cast(
                        int,
                        arguments.maximum_parquet_bytes,
                    ),
                    maximum_elapsed_seconds=cast(
                        float,
                        arguments.maximum_elapsed_seconds,
                    ),
                    maximum_concurrency=cast(
                        int,
                        arguments.maximum_concurrency,
                    ),
                ),
                on_published=emit_partition,
            )
        except (AvailabilityError, ResearchStorageError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        _emit_json(
            {
                "archive_count": result.archive_count,
                "availability_report_id": result.availability_report_id,
                "created_count": result.created_count,
                "event": "research_corpus_complete",
                "excluded_symbol_months": [
                    f"{symbol}:{month}"
                    for symbol, month in result.excluded_symbol_months
                ],
                "parquet_bytes": result.parquet_bytes,
                "reused_count": result.reused_count,
                "row_count": result.row_count,
                "status": "ok",
            }
        )
        return 0

    if arguments.command == "historical-coverage":
        try:
            availability = load_published_archive_availability(
                cast(Path, arguments.availability)
            )
            published_coverage = build_historical_coverage(
                availability,
                monthly_root=cast(Path, arguments.monthly_root),
                output_root=cast(Path, arguments.output),
                symbols=tuple(cast(list[str], arguments.symbol)),
                limits=CoverageLimits(
                    maximum_archives=cast(int, arguments.maximum_archives)
                ),
            )
        except (AvailabilityError, HistoricalCoverageError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        report = published_coverage.report
        _emit_json(
            {
                "already_present": published_coverage.already_present,
                "available_archive_count": report.available_archive_count,
                "event": "historical_coverage",
                "not_found_archive_count": report.not_found_archive_count,
                "path": str(published_coverage.path),
                "report_id": report.report_id,
                "status": "ok",
                "symbol_count": len(report.symbols),
            }
        )
        return 0

    if arguments.command == "archive-monthly":
        run_id = uuid4()

        def emit_month(publication: Any) -> None:
            manifest = publication.manifest
            _emit_json(
                {
                    "already_present": publication.already_present,
                    "event": "monthly_archive",
                    "internal_missing_minutes": manifest.internal_missing_minutes,
                    "manifest_id": manifest.manifest_id,
                    "month": manifest.month,
                    "observed_first_day": manifest.observed_first_day,
                    "observed_last_day": manifest.observed_last_day,
                    "path": str(publication.path),
                    "row_count": manifest.row_count,
                    "run_id": str(run_id),
                    "symbol": manifest.symbol,
                }
            )

        try:
            report = load_published_archive_availability(
                cast(Path, arguments.availability)
            )
            result = asyncio.run(
                acquire_monthly_archives(
                    report,
                    output_root=cast(Path, arguments.output),
                    ingestion_run_id=run_id,
                    limits=MonthlyArchiveLimits(
                        maximum_archives=cast(int, arguments.maximum_archives),
                        maximum_concurrency=cast(int, arguments.maximum_concurrency),
                        maximum_decode_processes=cast(
                            int,
                            arguments.maximum_decode_processes,
                        ),
                        maximum_total_compressed_bytes=cast(
                            int,
                            arguments.maximum_total_compressed_bytes,
                        ),
                        maximum_elapsed_seconds=cast(
                            float,
                            arguments.maximum_elapsed_seconds,
                        ),
                    ),
                    symbols=tuple(cast(list[str], arguments.symbol)),
                    on_published=emit_month,
                )
            )
        except (ArchiveError, AvailabilityError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        _emit_json(
            {
                "archive_count": result.archive_count,
                "availability_report_id": result.availability_report_id,
                "compressed_bytes": result.compressed_bytes,
                "downloaded_count": result.downloaded_count,
                "event": "monthly_archive_complete",
                "reused_count": result.reused_count,
                "run_id": str(run_id),
                "status": "ok",
            }
        )
        return 0

    if arguments.command == "archive-availability":
        try:
            published_availability = asyncio.run(
                _audit_archive_availability(
                    pool_path=cast(Path, arguments.pool),
                    start_month=cast(date, arguments.start_month),
                    end_month=cast(date, arguments.end_month),
                    maximum_probes=cast(int, arguments.maximum_probes),
                    symbols=tuple(cast(list[str], arguments.symbol)),
                    output=cast(Path, arguments.output),
                )
            )
        except (AvailabilityError, PoolError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        report = published_availability.report
        _emit_json(
            {
                "already_present": published_availability.already_present,
                "available_count": report.available_count,
                "not_found_count": report.not_found_count,
                "path": str(published_availability.path),
                "pool_id": report.pool_id,
                "ready": report.ready,
                "report_id": report.report_id,
                "status": "ok",
                "unresolved_count": report.unresolved_count,
            }
        )
        return 0

    if arguments.command == "instrument-pool":
        run_id = uuid4()
        try:
            published_pool = asyncio.run(
                _capture_research_pool(
                    run_id=run_id,
                    output=cast(Path, arguments.output),
                    maximum_instruments=cast(int, arguments.maximum_instruments),
                    excluded_symbols=tuple(cast(list[str], arguments.exclude_symbol)),
                )
            )
        except (
            EvidenceAdmissionError,
            LiveCaptureError,
            PoolError,
            ValueError,
        ) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        pool = published_pool.pool
        _emit_json(
            {
                "already_present": published_pool.already_present,
                "instrument_count": len(pool.symbols),
                "observed_at_ns": pool.observed_at_ns,
                "path": str(published_pool.path),
                "pool_id": pool.pool_id,
                "run_id": str(run_id),
                "status": "ok",
            }
        )
        return 0

    if arguments.command == "archive-range":
        run_id = uuid4()

        def emit_partition(publication: Any) -> None:
            _emit_json(
                {
                    "already_present": publication.already_present,
                    "day": publication.manifest.day,
                    "event": "archive_partition",
                    "manifest_id": publication.manifest.manifest_id,
                    "path": str(publication.path),
                    "row_count": publication.manifest.row_count,
                    "run_id": str(run_id),
                    "symbol": publication.manifest.symbol,
                }
            )

        try:
            pool_path = cast(Path | None, arguments.pool)
            if pool_path is not None:
                instruments = load_published_research_pool(pool_path).instruments
            else:
                instruments = tuple(
                    InstrumentId(
                        venue=VenueId("binance", "spot"),
                        environment=Environment.PRODUCTION,
                        symbol=symbol,
                    )
                    for symbol in cast(list[str], arguments.symbol)
                )
            request = ArchiveRangeRequest(
                instruments=instruments,
                start_day=cast(date, arguments.start_day),
                end_day=cast(date, arguments.end_day),
            )
            result = asyncio.run(
                acquire_archive_range(
                    request,
                    output_root=cast(Path, arguments.output),
                    ingestion_run_id=run_id,
                    batch_limits=ArchiveBatchLimits(
                        maximum_partitions=cast(int, arguments.maximum_partitions)
                    ),
                    on_published=emit_partition,
                )
            )
        except (ArchiveError, EvidenceAdmissionError, PoolError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        _emit_json(
            {
                "downloaded_count": result.downloaded_count,
                "event": "archive_range_complete",
                "partition_count": result.partition_count,
                "reused_count": result.reused_count,
                "run_id": str(run_id),
                "status": "ok",
            }
        )
        return 0

    if arguments.command == "archive-day":
        symbol = cast(str, arguments.symbol)
        day = cast(date, arguments.day)
        output = cast(Path, arguments.output)
        run_id = uuid4()
        try:
            request = ArchiveDayRequest(
                instrument=InstrumentId(
                    venue=VenueId("binance", "spot"),
                    environment=Environment.PRODUCTION,
                    symbol=symbol,
                ),
                day=day,
            )
            result = asyncio.run(
                acquire_archive_day(
                    request,
                    output_root=output,
                    ingestion_run_id=run_id,
                )
            )
        except (ArchiveError, EvidenceAdmissionError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

        payload = {
            "already_present": result.already_present,
            "archive_sha256": result.manifest.archive_sha256,
            "manifest_id": result.manifest.manifest_id,
            "path": str(result.path),
            "row_count": result.manifest.row_count,
            "run_id": str(run_id),
            "status": "ok",
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return 0

    if arguments.command == "qualify-live":
        run_id = uuid4()
        try:
            result = asyncio.run(
                _qualify_live(
                    run_id=run_id,
                    output=cast(Path, arguments.output),
                    duration_seconds=cast(float, arguments.duration_seconds),
                    requested_symbols=tuple(
                        cast(list[str] | None, arguments.symbol) or ()
                    ),
                    hot_symbols=tuple(cast(list[str], arguments.hot_symbol)),
                    startup_timeout_seconds=cast(
                        float,
                        arguments.startup_timeout_seconds,
                    ),
                    maximum_uncompressed_capture_bytes=cast(
                        int,
                        arguments.maximum_uncompressed_capture_bytes,
                    ),
                )
            )
        except (
            LiveCaptureError,
            QualificationError,
            EvidenceAdmissionError,
            ValueError,
        ) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        payload = {
            "critical_failures": list(result.report.critical_failures),
            "manifest_id": result.manifest.manifest_id,
            "path": str(result.path),
            "ready": result.report.ready,
            "report_id": result.report.report_id,
            "run_id": str(run_id),
            "status": "ok",
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return 0

    if arguments.command == "qualify-live-service":
        run_id = uuid4()
        try:
            asyncio.run(
                _qualify_live_service(
                    run_id=run_id,
                    output=cast(Path, arguments.output),
                    requested_symbols=tuple(
                        cast(list[str] | None, arguments.symbol) or ()
                    ),
                    hot_symbols=tuple(cast(list[str], arguments.hot_symbol)),
                    startup_timeout_seconds=cast(
                        float,
                        arguments.startup_timeout_seconds,
                    ),
                    maximum_uncompressed_capture_bytes=cast(
                        int,
                        arguments.maximum_uncompressed_capture_bytes,
                    ),
                )
            )
        except KeyboardInterrupt:
            print("error: interrupted before clean shutdown", file=sys.stderr)
            return 130
        except (
            LiveCaptureError,
            QualificationError,
            EvidenceAdmissionError,
            ValueError,
        ) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return 0

    if arguments.command != "smoke":
        raise RuntimeError("argparse admitted an unknown command")

    config_path = cast(Path, arguments.config)
    try:
        document = _read_config_document(config_path)
        config = admit_run_config(document)
    except (ConfigFileError, ConfigAdmissionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    payload = {
        "config_id": config.config_id,
        "read_only": True,
        "run_id": str(uuid4()),
        "schema_version": config.schema_version,
        "status": "ok",
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
