from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

import pytest

import crypto_boom.history.daily as daily
from crypto_boom.history.daily import (
    ArchiveBatchError,
    ArchiveBatchLimits,
    ArchiveDayRequest,
    ArchiveIntegrityError,
    ArchiveLimits,
    ArchivePublicationError,
    ArchiveRangeRequest,
    ArchiveResourceError,
    ArchiveSchemaError,
    ArchiveTransportError,
    acquire_archive_day,
    acquire_archive_range,
)
from crypto_boom.market import Environment, InstrumentId, VenueId

RUN_ID = UUID("c782ef24-45fb-4264-a61e-659224f869e1")
DAY = date(2025, 1, 2)
MICROSECONDS_PER_DAY = 86_400_000_000


@dataclass(frozen=True, slots=True)
class _Response:
    body: bytes
    status: int = 200
    advertised_length: int | None = None
    delay_seconds: float = 0.0


def _request(day: date = DAY, *, symbol: str = "ETHUSDT") -> ArchiveDayRequest:
    return ArchiveDayRequest(
        instrument=InstrumentId(
            venue=VenueId("binance", "spot"),
            environment=Environment.PRODUCTION,
            symbol=symbol,
        ),
        day=day,
    )


def _day_start_raw(day: date, *, microseconds: bool = True) -> int:
    day_start_us = (day - date(1970, 1, 1)).days * MICROSECONDS_PER_DAY
    return day_start_us if microseconds else day_start_us // 1_000


def _row(
    day: date = DAY,
    *,
    microseconds: bool = True,
    column_count: int = 12,
    open_price: str = "1.00000000",
) -> str:
    open_time = _day_start_raw(day, microseconds=microseconds)
    minute = 60_000_000 if microseconds else 60_000
    values = [
        str(open_time),
        open_price,
        "2.00000000",
        "0.50000000",
        "1.50000000",
        "10.00000000",
        str(open_time + minute - 1),
        "15.00000000",
        "2",
        "6.00000000",
        "9.00000000",
        "0",
    ]
    return ",".join(values[:column_count]) + "\n"


def _archive_bytes(request: ArchiveDayRequest, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(request.member_filename, csv_text.encode("utf-8"))
    return buffer.getvalue()


def _responses(
    request: ArchiveDayRequest,
    archive_bytes: bytes,
    *,
    checksum: str | None = None,
    checksum_filename: str | None = None,
    advertised_archive_length: int | None = None,
) -> dict[str, _Response]:
    digest = checksum or hashlib.sha256(archive_bytes).hexdigest()
    filename = checksum_filename or request.archive_filename
    source_path = (
        f"/data/spot/daily/klines/{request.instrument.symbol}/"
        f"{request.interval}/{request.archive_filename}"
    )
    return {
        f"{source_path}.CHECKSUM": _Response(f"{digest}  {filename}\n".encode("ascii")),
        source_path: _Response(
            archive_bytes,
            advertised_length=advertised_archive_length,
        ),
    }


async def _with_server[T](
    responses: dict[str, _Response],
    operation: Callable[[str], Awaitable[T]],
    *,
    observed_requests: list[str] | None = None,
) -> T:
    requests = observed_requests if observed_requests is not None else []

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_line = await reader.readline()
        parts = request_line.decode("ascii").strip().split(" ")
        path = parts[1] if len(parts) == 3 else ""
        requests.append(path)
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass

        response = responses.get(path, _Response(b"not found", status=404))
        if response.delay_seconds:
            await asyncio.sleep(response.delay_seconds)
        reason = "OK" if response.status == 200 else "Not Found"
        length = response.advertised_length
        if length is None:
            length = len(response.body)
        writer.write(
            (
                f"HTTP/1.1 {response.status} {reason}\r\n"
                f"Content-Length: {length}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
        )
        writer.write(response.body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    socket = server.sockets[0]
    host, port = socket.getsockname()[:2]
    base_url = f"http://{host}:{port}/data/spot/daily"
    try:
        return await operation(base_url)
    finally:
        server.close()
        await server.wait_closed()


def _run[T](
    responses: dict[str, _Response],
    operation: Callable[[str], Awaitable[T]],
    *,
    observed_requests: list[str] | None = None,
) -> T:
    return asyncio.run(
        _with_server(
            responses,
            operation,
            observed_requests=observed_requests,
        )
    )


def test_clean_download_is_reproducible_and_published_atomically(
    tmp_path: Path,
) -> None:
    request = _request()
    archive_bytes = _archive_bytes(
        request,
        _row()
        + _row()
        .replace(
            str(_day_start_raw(DAY)),
            str(_day_start_raw(DAY) + 60_000_000),
            1,
        )
        .replace(
            str(_day_start_raw(DAY) + 59_999_999),
            str(_day_start_raw(DAY) + 119_999_999),
            1,
        ),
    )
    responses = _responses(request, archive_bytes)

    async def acquire(
        base_url: str,
    ) -> tuple[daily.PublishedArchive, daily.PublishedArchive]:
        first = await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )
        second = await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=UUID("510b766e-ce49-4277-b035-9b49f54d580d"),
            base_url=base_url,
        )
        return first, second

    observed_requests: list[str] = []
    first, second = _run(
        responses,
        acquire,
        observed_requests=observed_requests,
    )

    assert not first.already_present
    assert second.already_present
    assert first.path == second.path
    assert first.manifest == second.manifest
    assert first.manifest.row_count == 2
    assert first.manifest.timestamp_unit == "us"
    assert first.manifest.archive_sha256 == (
        "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
    )
    assert (first.path / request.archive_filename).read_bytes() == archive_bytes
    manifest_document = json.loads((first.path / "manifest.json").read_text())
    assert manifest_document == first.manifest.to_mapping()
    assert list((tmp_path / ".staging").iterdir()) == []
    assert observed_requests == [
        request.source_url("/data/spot/daily") + ".CHECKSUM",
        request.source_url("/data/spot/daily"),
        request.source_url("/data/spot/daily") + ".CHECKSUM",
    ]


def test_range_acquisition_is_bounded_and_deterministic(tmp_path: Path) -> None:
    next_day = DAY + timedelta(days=1)
    requests = [
        _request(day, symbol=symbol)
        for symbol in ("ETHUSDT", "BTCUSDT")
        for day in (DAY, next_day)
    ]
    responses: dict[str, _Response] = {}
    for request in requests:
        responses.update(
            _responses(request, _archive_bytes(request, _row(request.day)))
        )

    instruments = tuple(request.instrument for request in (requests[0], requests[2]))
    range_request = ArchiveRangeRequest(
        instruments=instruments,
        start_day=DAY,
        end_day=next_day,
    )

    async def acquire(base_url: str) -> daily.ArchiveRangeResult:
        return await acquire_archive_range(
            range_request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            batch_limits=ArchiveBatchLimits(
                maximum_partitions=4,
                retry_attempts=1,
                retry_base_seconds=0.01,
                request_spacing_seconds=0.0,
            ),
            base_url=base_url,
        )

    result = _run(responses, acquire)

    assert [
        (item.manifest.symbol, item.manifest.day) for item in result.publications
    ] == [
        ("BTCUSDT", DAY.isoformat()),
        ("BTCUSDT", next_day.isoformat()),
        ("ETHUSDT", DAY.isoformat()),
        ("ETHUSDT", next_day.isoformat()),
    ]
    assert result.partition_count == 4
    assert result.downloaded_count == 4
    assert result.reused_count == 0


def test_range_limit_rejects_before_network(tmp_path: Path) -> None:
    next_day = DAY + timedelta(days=1)
    first = _request(symbol="BTCUSDT").instrument
    second = _request(symbol="ETHUSDT").instrument
    range_request = ArchiveRangeRequest(
        instruments=(first, second),
        start_day=DAY,
        end_day=next_day,
    )
    observed_requests: list[str] = []

    async def acquire(base_url: str) -> daily.ArchiveRangeResult:
        return await acquire_archive_range(
            range_request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            batch_limits=ArchiveBatchLimits(maximum_partitions=3),
            base_url=base_url,
        )

    with pytest.raises(ArchiveResourceError, match="4 partitions"):
        _run({}, acquire, observed_requests=observed_requests)

    assert observed_requests == []


def test_range_rejects_more_than_500_instruments() -> None:
    instruments = tuple(
        _request(symbol=f"A{index:03d}USDT").instrument for index in range(501)
    )

    with pytest.raises(ArchiveResourceError, match="500 instruments"):
        ArchiveRangeRequest(
            instruments=instruments,
            start_day=DAY,
            end_day=DAY,
        )


def test_range_failure_preserves_progress_for_a_resumable_rerun(
    tmp_path: Path,
) -> None:
    next_day = DAY + timedelta(days=1)
    first = _request(DAY)
    second = _request(next_day)
    first_bytes = _archive_bytes(first, _row(first.day))
    second_bytes = _archive_bytes(second, _row(second.day))
    responses = _responses(first, first_bytes)
    second_responses = _responses(second, second_bytes)
    second_source = next(
        path for path in second_responses if not path.endswith(".CHECKSUM")
    )
    valid_second_response = second_responses[second_source]
    second_responses[second_source] = _Response(b"not found", status=404)
    responses.update(second_responses)
    range_request = ArchiveRangeRequest(
        instruments=(first.instrument,),
        start_day=DAY,
        end_day=next_day,
    )
    batch_limits = ArchiveBatchLimits(
        maximum_partitions=2,
        retry_attempts=1,
        retry_base_seconds=0.01,
        request_spacing_seconds=0.0,
    )

    async def fail_then_resume(base_url: str) -> daily.ArchiveRangeResult:
        with pytest.raises(ArchiveBatchError, match=next_day.isoformat()):
            await acquire_archive_range(
                range_request,
                output_root=tmp_path,
                ingestion_run_id=RUN_ID,
                batch_limits=batch_limits,
                base_url=base_url,
            )
        responses[second_source] = valid_second_response
        return await acquire_archive_range(
            range_request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            batch_limits=batch_limits,
            base_url=base_url,
        )

    observed_requests: list[str] = []
    result = _run(
        responses,
        fail_then_resume,
        observed_requests=observed_requests,
    )

    assert result.reused_count == 1
    assert result.downloaded_count == 1
    first_source = first.source_url("/data/spot/daily")
    second_source_expected = second.source_url("/data/spot/daily")
    assert observed_requests == [
        first_source + ".CHECKSUM",
        first_source,
        second_source_expected + ".CHECKSUM",
        second_source_expected,
        first_source + ".CHECKSUM",
        second_source_expected + ".CHECKSUM",
        second_source_expected,
    ]


def test_bad_checksum_fails_before_authoritative_publication(tmp_path: Path) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    responses = _responses(request, archive_bytes, checksum="0" * 64)

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    with pytest.raises(ArchiveIntegrityError, match="does not match"):
        _run(responses, acquire)

    assert list(tmp_path.rglob("manifest.json")) == []
    assert list((tmp_path / ".staging").iterdir()) == []


def test_checksum_must_name_the_requested_archive(tmp_path: Path) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    responses = _responses(
        request,
        archive_bytes,
        checksum_filename="another-file.zip",
    )

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    with pytest.raises(ArchiveIntegrityError, match="filename"):
        _run(responses, acquire)


@pytest.mark.parametrize(
    "csv_text",
    [
        _row(column_count=11),
        _row(open_price="NaN"),
        _row(microseconds=False),
    ],
)
def test_schema_decimal_and_timestamp_unit_mismatches_fail_closed(
    tmp_path: Path,
    csv_text: str,
) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, csv_text)
    responses = _responses(request, archive_bytes)

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    with pytest.raises(ArchiveSchemaError):
        _run(responses, acquire)

    assert list(tmp_path.rglob("manifest.json")) == []


def test_advertised_download_size_is_rejected_before_body_read(tmp_path: Path) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    limits = replace(ArchiveLimits(), compressed_bytes=len(archive_bytes) - 1)
    responses = _responses(
        request,
        archive_bytes,
        advertised_archive_length=len(archive_bytes),
    )

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            limits=limits,
            base_url=base_url,
        )

    with pytest.raises(ArchiveTransportError, match="byte limit"):
        _run(responses, acquire)

    assert list(tmp_path.rglob("manifest.json")) == []


def test_decode_time_limit_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    responses = _responses(request, archive_bytes)
    elapsed = iter((10.0, 10.0, 21.0))
    monkeypatch.setattr(daily, "monotonic", lambda: next(elapsed))

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    with pytest.raises(ArchiveResourceError, match="time limit"):
        _run(responses, acquire)

    assert list(tmp_path.rglob("manifest.json")) == []


def test_failed_atomic_replace_leaves_no_authoritative_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    responses = _responses(request, archive_bytes)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(daily._artifacts.os, "replace", fail_replace)

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    with pytest.raises(ArchivePublicationError, match="atomic"):
        _run(responses, acquire)

    assert list(tmp_path.rglob("manifest.json")) == []
    assert list((tmp_path / ".staging").iterdir()) == []


def test_cancelled_download_leaves_no_authoritative_partition(tmp_path: Path) -> None:
    request = _request()
    archive_bytes = _archive_bytes(request, _row())
    responses = _responses(request, archive_bytes)
    checksum_path = next(path for path in responses if path.endswith(".CHECKSUM"))
    responses[checksum_path] = replace(
        responses[checksum_path],
        delay_seconds=0.25,
    )

    async def cancel(base_url: str) -> None:
        task = asyncio.create_task(
            acquire_archive_day(
                request,
                output_root=tmp_path,
                ingestion_run_id=RUN_ID,
                base_url=base_url,
            )
        )
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(responses, cancel)

    assert list(tmp_path.rglob("manifest.json")) == []
    assert list((tmp_path / ".staging").iterdir()) == []


def test_pre_2025_archive_declares_millisecond_timestamps(tmp_path: Path) -> None:
    day = date(2024, 12, 31)
    request = _request(day)
    archive_bytes = _archive_bytes(
        request,
        _row(day, microseconds=False),
    )
    responses = _responses(request, archive_bytes)

    async def acquire(base_url: str) -> daily.PublishedArchive:
        return await acquire_archive_day(
            request,
            output_root=tmp_path,
            ingestion_run_id=RUN_ID,
            base_url=base_url,
        )

    result = _run(responses, acquire)

    assert result.manifest.timestamp_unit == "ms"
