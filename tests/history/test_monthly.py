from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
)
from crypto_boom.history.codec import (
    ArchiveBatchError,
    ArchiveResourceError,
    ArchiveSchemaError,
)
from crypto_boom.history.monthly import (
    MonthlyArchiveLimits,
    acquire_monthly_archives,
    load_published_monthly_archive,
    load_published_monthly_klines_by_day,
)

RUN_ID = UUID("c782ef24-45fb-4264-a61e-659224f869e1")
MONTH = date(2025, 1, 1)
MICROSECONDS_PER_DAY = 86_400_000_000
MICROSECONDS_PER_MINUTE = 60_000_000


@dataclass(frozen=True, slots=True)
class _Response:
    body: bytes
    status: int = 200
    delay_seconds: float = 0.0


def _raw_time(day: date, minute: int = 0) -> int:
    return (
        day - date(1970, 1, 1)
    ).days * MICROSECONDS_PER_DAY + minute * MICROSECONDS_PER_MINUTE


def _row(day: date, minute: int = 0) -> str:
    open_time = _raw_time(day, minute)
    return (
        ",".join(
            (
                str(open_time),
                "1.0",
                "2.0",
                "0.5",
                "1.5",
                "10.0",
                str(open_time + MICROSECONDS_PER_MINUTE - 1),
                "15.0",
                "2",
                "6.0",
                "9.0",
                "0",
            )
        )
        + "\n"
    )


def _archive(symbol: str, month: str, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    member = f"{symbol}-1m-{month}.csv"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(member, csv_text.encode())
    return buffer.getvalue()


def _report(
    base_url: str,
    archive_bytes: bytes,
    *,
    state: AvailabilityState = AvailabilityState.AVAILABLE,
    checksum: str | None = None,
) -> ArchiveAvailabilityReport:
    digest = checksum or hashlib.sha256(archive_bytes).hexdigest()
    return ArchiveAvailabilityReport(
        schema_version=1,
        pool_id="sha256:" + "a" * 64,
        base_url=base_url,
        interval="1m",
        start_month="2025-01",
        end_month="2025-01",
        probed_at_ns=1_795_027_260_000_002_000,
        probes=(
            MonthlyArchiveProbe(
                "ETHUSDT",
                "2025-01",
                state,
                f"sha256:{digest}" if state is AvailabilityState.AVAILABLE else None,
                1,
                200 if state is AvailabilityState.AVAILABLE else 404,
            ),
        ),
    )


async def _with_server[T](
    responses: dict[str, _Response],
    operation: Callable[[str], Awaitable[T]],
    *,
    observed: list[str] | None = None,
    activity: list[int] | None = None,
) -> T:
    paths = observed if observed is not None else []

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        parts = request_line.decode("ascii").strip().split(" ")
        path = parts[1] if len(parts) == 3 else ""
        paths.append(path)
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        response = responses.get(path, _Response(b"not found", 404))
        if activity is not None:
            activity[0] += 1
            activity[1] = max(activity[1], activity[0])
        if response.delay_seconds:
            await asyncio.sleep(response.delay_seconds)
        reason = "OK" if response.status == 200 else "Not Found"
        writer.write(
            (
                f"HTTP/1.1 {response.status} {reason}\r\n"
                f"Content-Length: {len(response.body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
        )
        writer.write(response.body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        if activity is not None:
            activity[0] -= 1

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    base_url = f"http://{host}:{port}/data/spot/monthly"
    try:
        return await operation(base_url)
    finally:
        server.close()
        await server.wait_closed()


def test_available_month_is_published_once_and_exposed_as_daily_inputs(
    tmp_path: Path,
) -> None:
    symbol = "ETHUSDT"
    month = "2025-01"
    first_day = MONTH + timedelta(days=1)
    second_day = first_day + timedelta(days=1)
    archive_bytes = _archive(symbol, month, _row(first_day) + _row(second_day))
    path = f"/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
    observed: list[str] = []

    async def acquire(base_url: str):
        report = _report(base_url, archive_bytes)
        first = await acquire_monthly_archives(
            report,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=1,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )
        second = await acquire_monthly_archives(
            report,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=1,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )
        return first, second

    first, second = asyncio.run(
        _with_server(
            {path: _Response(archive_bytes)},
            acquire,
            observed=observed,
        )
    )

    assert first.downloaded_count == 1
    assert second.reused_count == 1
    publication = first.publications[0]
    manifest = publication.manifest
    assert manifest.row_count == 2
    assert manifest.observed_first_day == first_day.isoformat()
    assert manifest.observed_last_day == second_day.isoformat()
    assert manifest.observed_days == (first_day.isoformat(), second_day.isoformat())
    assert manifest.internal_gap_count == 1
    assert manifest.internal_missing_minutes == 1_439
    assert len(manifest.missing_days) == 29
    assert observed == [path]
    assert len(list(tmp_path.rglob("*.zip"))) == 1

    reloaded = load_published_monthly_archive(publication.path)
    assert reloaded.manifest == publication.manifest
    assert reloaded.availability_report_id == first.availability_report_id
    daily = load_published_monthly_klines_by_day(reloaded)
    assert [item.day for item in daily] == [first_day, second_day]
    assert [len(item.events) for item in daily] == [1, 1]
    assert daily[0].events[0].provenance.source.channel == "spot/monthly/klines/1m"


def test_monthly_bound_fails_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decode process"):
        MonthlyArchiveLimits(maximum_decode_processes=3)
    archive_bytes = _archive("ETHUSDT", "2025-01", _row(MONTH))
    observed: list[str] = []

    async def acquire(base_url: str):
        report = _report(base_url, archive_bytes)
        return await acquire_monthly_archives(
            report,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(maximum_archives=1),
            symbols=("ETHUSDT", "SOLUSDT"),
        )

    with pytest.raises(ArchiveSchemaError, match="not present"):
        asyncio.run(_with_server({}, acquire, observed=observed))
    assert observed == []

    report = _report("http://127.0.0.1:1/data/spot/monthly", archive_bytes)
    duplicate = replace(
        report,
        probes=(
            report.probes[0],
            replace(report.probes[0], month="2025-02"),
        ),
        end_month="2025-02",
    )
    with pytest.raises(ArchiveResourceError, match="2 archives"):
        asyncio.run(
            acquire_monthly_archives(
                duplicate,
                output_root=tmp_path,
                ingestion_run_id=RUN_ID,
                limits=MonthlyArchiveLimits(maximum_archives=1),
            )
        )


def test_monthly_checksum_and_month_window_fail_closed(tmp_path: Path) -> None:
    archive_bytes = _archive("ETHUSDT", "2025-01", _row(MONTH))
    path = "/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2025-01.zip"

    async def bad_checksum(base_url: str):
        return await acquire_monthly_archives(
            _report(base_url, archive_bytes, checksum="0" * 64),
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=1,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )

    with pytest.raises(ArchiveBatchError, match="SHA-256"):
        asyncio.run(_with_server({path: _Response(archive_bytes)}, bad_checksum))
    assert list(tmp_path.rglob("manifest.json")) == []

    outside = _archive("ETHUSDT", "2025-01", _row(date(2025, 2, 1)))

    async def bad_window(base_url: str):
        return await acquire_monthly_archives(
            _report(base_url, outside),
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=1,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )

    with pytest.raises(ArchiveBatchError, match="requested month"):
        asyncio.run(_with_server({path: _Response(outside)}, bad_window))
    assert list(tmp_path.rglob("manifest.json")) == []


def test_non_available_observations_are_not_downloaded(tmp_path: Path) -> None:
    archive_bytes = _archive("ETHUSDT", "2025-01", _row(MONTH))
    observed: list[str] = []

    async def acquire(base_url: str):
        return await acquire_monthly_archives(
            _report(
                base_url,
                archive_bytes,
                state=AvailabilityState.NOT_FOUND,
            ),
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(maximum_archives=1),
        )

    result = asyncio.run(_with_server({}, acquire, observed=observed))
    assert result.publications == ()
    assert observed == []


def test_batch_byte_and_elapsed_budgets_preserve_completed_progress(
    tmp_path: Path,
) -> None:
    first_bytes = _archive("ETHUSDT", "2025-01", _row(MONTH))
    second_bytes = _archive("SOLUSDT", "2025-01", _row(MONTH))
    first_path = "/data/spot/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2025-01.zip"
    second_path = "/data/spot/monthly/klines/SOLUSDT/1m/SOLUSDT-1m-2025-01.zip"

    def report(base_url: str) -> ArchiveAvailabilityReport:
        first = _report(base_url, first_bytes)
        return replace(
            first,
            probes=(
                first.probes[0],
                MonthlyArchiveProbe(
                    "SOLUSDT",
                    "2025-01",
                    AvailabilityState.AVAILABLE,
                    "sha256:" + hashlib.sha256(second_bytes).hexdigest(),
                    1,
                    200,
                ),
            ),
        )

    observed_bytes: list[str] = []

    async def exhaust_bytes(base_url: str):
        return await acquire_monthly_archives(
            report(base_url),
            output_root=tmp_path / "bytes",
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=2,
                maximum_concurrency=1,
                maximum_total_compressed_bytes=(
                    len(first_bytes) + len(second_bytes) - 1
                ),
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )

    with pytest.raises(ArchiveBatchError, match="byte limit"):
        asyncio.run(
            _with_server(
                {
                    first_path: _Response(first_bytes),
                    second_path: _Response(second_bytes),
                },
                exhaust_bytes,
                observed=observed_bytes,
            )
        )
    assert observed_bytes == [first_path, second_path]
    assert len(list((tmp_path / "bytes").rglob("manifest.json"))) == 1

    observed_time: list[str] = []

    async def exhaust_time(base_url: str):
        return await acquire_monthly_archives(
            report(base_url),
            output_root=tmp_path / "time",
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=2,
                maximum_concurrency=1,
                maximum_elapsed_seconds=0.01,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )

    with pytest.raises(ArchiveBatchError, match="elapsed-time budget"):
        asyncio.run(
            _with_server(
                {
                    first_path: _Response(first_bytes, delay_seconds=0.03),
                    second_path: _Response(second_bytes),
                },
                exhaust_time,
                observed=observed_time,
            )
        )
    assert observed_time == [first_path]
    assert len(list((tmp_path / "time").rglob("manifest.json"))) == 1

    activity = [0, 0]

    async def acquire_concurrently(base_url: str):
        return await acquire_monthly_archives(
            report(base_url),
            output_root=tmp_path / "concurrent",
            ingestion_run_id=RUN_ID,
            limits=MonthlyArchiveLimits(
                maximum_archives=2,
                maximum_concurrency=2,
                retry_attempts=1,
                retry_base_seconds=0,
                request_spacing_seconds=0,
            ),
        )

    concurrent = asyncio.run(
        _with_server(
            {
                first_path: _Response(first_bytes, delay_seconds=0.03),
                second_path: _Response(second_bytes, delay_seconds=0.03),
            },
            acquire_concurrently,
            activity=activity,
        )
    )
    assert activity[1] == 2
    assert [item.manifest.symbol for item in concurrent.publications] == [
        "ETHUSDT",
        "SOLUSDT",
    ]
