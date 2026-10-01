from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from crypto_boom import cli
from crypto_boom.history.availability import (
    ArchiveAvailabilityReport,
    AvailabilityState,
    MonthlyArchiveProbe,
    PublishedArchiveAvailability,
)
from crypto_boom.history.daily import (
    ArchiveManifest,
    ArchiveRangeRequest,
    ArchiveRangeResult,
    PublishedArchive,
)
from crypto_boom.universe import (
    PublishedResearchPool,
    ResearchInstrumentPool,
)

FIXED_RUN_ID = UUID("4b0de96f-c5c9-40ce-9747-b558f84821bb")


def _archive_manifest(
    *,
    symbol: str = "ETHUSDT",
    day: str = "2025-01-02",
) -> ArchiveManifest:
    return ArchiveManifest(
        schema_version=1,
        decoder_version="binance-spot-kline-csv-v1",
        venue="binance",
        market="spot",
        environment="production",
        dataset="klines",
        symbol=symbol,
        interval="1m",
        day=day,
        timestamp_unit="us",
        source_url=f"https://data.binance.vision/{symbol}-{day}.zip",
        checksum_url=f"https://data.binance.vision/{symbol}-{day}.zip.CHECKSUM",
        archive_filename=f"{symbol}-1m-{day}.zip",
        member_filename=f"{symbol}-1m-{day}.csv",
        source_revision="sha256:" + "2" * 64,
        archive_sha256="sha256:" + "2" * 64,
        member_sha256="sha256:" + "3" * 64,
        compressed_bytes=100,
        uncompressed_bytes=1_000,
        row_count=1_440,
        first_open_time_us=1_735_776_000_000_000,
        last_open_time_us=1_735_862_340_000_000,
    )


def _research_pool() -> ResearchInstrumentPool:
    return ResearchInstrumentPool(
        schema_version=1,
        policy_version="binance-spot-altcoin-pool-v1",
        source_endpoint="https://data-api.binance.vision/api/v3/exchangeInfo",
        source_payload_sha256="sha256:" + "a" * 64,
        observed_at_ns=1_795_027_260_000_001_000,
        maximum_instruments=500,
        quote_asset="USDT",
        required_permission="SPOT",
        excluded_base_assets=(
            "BRL",
            "BTC",
            "BUSD",
            "DAI",
            "EUR",
            "FDUSD",
            "TRY",
            "TUSD",
            "USDC",
            "USDP",
        ),
        excluded_symbols=(),
        symbols=("ETHUSDT", "SOLUSDT"),
        exclusions=(),
        unsupported_symbols=(),
    )


def test_root_help_preserves_the_command_grammar(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    commands = (
        "{smoke,archive-day,archive-range,instrument-pool,archive-availability,"
        "archive-monthly,historical-coverage,research-corpus,qualify-live,"
        "qualify-live-service}"
    )
    assert raised.value.code == 0
    assert captured.err == ""
    assert normalized.count(commands) == 2


def test_smoke_emits_machine_readable_run_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("schema_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(["smoke", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "config_id": "sha256:a9d5f6d002d956b8af5787a05e0ca000d45c03977ffa54ee8fbed719fed5fd23",
        "read_only": True,
        "run_id": str(FIXED_RUN_ID),
        "schema_version": 1,
        "status": "ok",
    }


def test_smoke_rejects_invalid_config_without_echoing_secret(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    secret_value = "super-secret-value"
    config_path.write_text(
        f'schema_version = 1\napi_secret = "{secret_value}"\n',
        encoding="utf-8",
    )

    exit_code = cli.main(["smoke", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "error: configuration fields must be exactly: schema_version\n"
    )
    assert secret_value not in captured.err


def test_smoke_reports_unavailable_config_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.toml"

    exit_code = cli.main(["smoke", "--config", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: configuration file is unavailable\n"


def test_smoke_rejects_non_utf8_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b"schema_version = 1\n\xff")

    exit_code = cli.main(["smoke", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: configuration file is not UTF-8\n"


def test_smoke_rejects_oversized_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b" " * 65_537)

    exit_code = cli.main(["smoke", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: configuration file exceeds 65536 bytes\n"


def test_archive_day_emits_published_manifest_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published_path = tmp_path / "published"
    manifest = _archive_manifest()
    assert manifest.manifest_id == (
        "sha256:5120548a06349f36de953fd5894e4c67eec5b8e5368f74917c89d9e7848dd36c"
    )

    async def fake_acquire(*args: object, **kwargs: object) -> PublishedArchive:
        return PublishedArchive(
            path=published_path,
            manifest=manifest,
            already_present=False,
        )

    monkeypatch.setattr(cli, "acquire_archive_day", fake_acquire)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "archive-day",
            "--symbol",
            "ETHUSDT",
            "--day",
            "2025-01-02",
            "--output",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "already_present": False,
        "archive_sha256": manifest.archive_sha256,
        "manifest_id": manifest.manifest_id,
        "path": str(published_path),
        "row_count": 1_440,
        "run_id": str(FIXED_RUN_ID),
        "status": "ok",
    }


def test_archive_day_rejects_invalid_symbol_without_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "archive-day",
            "--symbol",
            "eth/usdt",
            "--day",
            "2025-01-02",
            "--output",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: canonical symbol is invalid\n"


def test_archive_range_emits_partition_progress_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = (
        PublishedArchive(tmp_path / "btc", _archive_manifest(symbol="BTCUSDT"), False),
        PublishedArchive(tmp_path / "eth", _archive_manifest(), True),
    )
    received: dict[str, object] = {}

    async def fake_acquire(*args: object, **kwargs: object) -> ArchiveRangeResult:
        received["request"] = args[0]
        received.update(kwargs)
        callback = kwargs["on_published"]
        assert callable(callback)
        for publication in publications:
            callback(publication)
        return ArchiveRangeResult(publications)

    monkeypatch.setattr(cli, "acquire_archive_range", fake_acquire)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "archive-range",
            "--symbol",
            "ETHUSDT",
            "--symbol",
            "BTCUSDT",
            "--start-day",
            "2025-01-02",
            "--end-day",
            "2025-01-02",
            "--maximum-partitions",
            "2",
            "--output",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert exit_code == 0
    assert captured.err == ""
    assert [line["event"] for line in lines] == [
        "archive_partition",
        "archive_partition",
        "archive_range_complete",
    ]
    assert [(line["symbol"], line["already_present"]) for line in lines[:2]] == [
        ("BTCUSDT", False),
        ("ETHUSDT", True),
    ]
    assert lines[-1] == {
        "downloaded_count": 1,
        "event": "archive_range_complete",
        "partition_count": 2,
        "reused_count": 1,
        "run_id": str(FIXED_RUN_ID),
        "status": "ok",
    }
    assert received["output_root"] == tmp_path
    assert received["ingestion_run_id"] == FIXED_RUN_ID


def test_archive_range_accepts_a_verified_pool_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _research_pool()
    received: dict[str, object] = {}

    async def fake_acquire(*args: object, **kwargs: object) -> ArchiveRangeResult:
        received["request"] = args[0]
        return ArchiveRangeResult(())

    monkeypatch.setattr(cli, "load_published_research_pool", lambda path: pool)
    monkeypatch.setattr(cli, "acquire_archive_range", fake_acquire)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "archive-range",
            "--pool",
            str(tmp_path / "pool"),
            "--start-day",
            "2025-01-02",
            "--end-day",
            "2025-01-02",
            "--output",
            str(tmp_path / "archives"),
        ]
    )

    captured = capsys.readouterr()
    request = received["request"]
    assert isinstance(request, ArchiveRangeRequest)
    assert exit_code == 0
    assert captured.err == ""
    assert [item.symbol for item in request.instruments] == ["ETHUSDT", "SOLUSDT"]


def test_instrument_pool_publishes_current_observed_selection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _research_pool()
    published = PublishedResearchPool(tmp_path / "pool", pool, False)
    received: dict[str, object] = {}

    async def fake_capture(**kwargs: object) -> PublishedResearchPool:
        received.update(kwargs)
        return published

    monkeypatch.setattr(cli, "_capture_research_pool", fake_capture)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "instrument-pool",
            "--maximum-instruments",
            "500",
            "--exclude-symbol",
            "TESTUSDT",
            "--output",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "already_present": False,
        "instrument_count": 2,
        "observed_at_ns": pool.observed_at_ns,
        "path": str(published.path),
        "pool_id": pool.pool_id,
        "run_id": str(FIXED_RUN_ID),
        "status": "ok",
    }
    assert received == {
        "excluded_symbols": ("TESTUSDT",),
        "maximum_instruments": 500,
        "output": tmp_path,
        "run_id": FIXED_RUN_ID,
    }


def test_archive_availability_emits_an_inspectable_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _research_pool()
    report = ArchiveAvailabilityReport(
        schema_version=1,
        pool_id=pool.pool_id,
        base_url="https://data.binance.vision/data/spot/monthly",
        interval="1m",
        start_month="2025-01",
        end_month="2025-01",
        probed_at_ns=1_795_027_260_000_002_000,
        probes=(
            MonthlyArchiveProbe(
                "ETHUSDT",
                "2025-01",
                AvailabilityState.AVAILABLE,
                "sha256:" + "b" * 64,
                1,
                200,
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
    published = PublishedArchiveAvailability(tmp_path / "availability", report, False)
    received: dict[str, object] = {}

    async def fake_audit(**kwargs: object) -> PublishedArchiveAvailability:
        received.update(kwargs)
        return published

    monkeypatch.setattr(cli, "_audit_archive_availability", fake_audit)

    exit_code = cli.main(
        [
            "archive-availability",
            "--pool",
            str(tmp_path / "pool"),
            "--start-month",
            "2025-01",
            "--end-month",
            "2025-01",
            "--maximum-probes",
            "2",
            "--symbol",
            "ETHUSDT",
            "--output",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "already_present": False,
        "available_count": 1,
        "not_found_count": 1,
        "path": str(published.path),
        "pool_id": pool.pool_id,
        "ready": True,
        "report_id": report.report_id,
        "status": "ok",
        "unresolved_count": 0,
    }
    assert received == {
        "end_month": date(2025, 1, 1),
        "maximum_probes": 2,
        "output": tmp_path,
        "pool_path": tmp_path / "pool",
        "start_month": date(2025, 1, 1),
        "symbols": ("ETHUSDT",),
    }


def test_archive_monthly_consumes_verified_availability_and_emits_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = ArchiveAvailabilityReport(
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
                AvailabilityState.AVAILABLE,
                "sha256:" + "b" * 64,
                1,
                200,
            ),
        ),
    )
    manifest = SimpleNamespace(
        internal_missing_minutes=3,
        manifest_id="sha256:" + "c" * 64,
        month="2025-01",
        observed_first_day="2025-01-02",
        observed_last_day="2025-01-31",
        row_count=43_197,
        symbol="ETHUSDT",
    )
    publication = SimpleNamespace(
        already_present=False,
        manifest=manifest,
        path=tmp_path / "monthly",
    )
    result = SimpleNamespace(
        archive_count=1,
        availability_report_id=report.report_id,
        compressed_bytes=2_005_311,
        downloaded_count=1,
        reused_count=0,
    )
    received: dict[str, object] = {}

    async def fake_acquire(*args: object, **kwargs: object):
        received["report"] = args[0]
        received.update(kwargs)
        callback = kwargs["on_published"]
        assert callable(callback)
        callback(publication)
        return result

    monkeypatch.setattr(
        cli,
        "load_published_archive_availability",
        lambda path: report,
    )
    monkeypatch.setattr(cli, "acquire_monthly_archives", fake_acquire)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "archive-monthly",
            "--availability",
            str(tmp_path / "availability"),
            "--symbol",
            "ETHUSDT",
            "--maximum-archives",
            "1",
            "--maximum-concurrency",
            "2",
            "--maximum-total-compressed-bytes",
            "4096",
            "--maximum-elapsed-seconds",
            "12",
            "--maximum-decode-processes",
            "2",
            "--output",
            str(tmp_path / "archives"),
        ]
    )

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines()]
    assert exit_code == 0
    assert captured.err == ""
    assert [line["event"] for line in lines] == [
        "monthly_archive",
        "monthly_archive_complete",
    ]
    assert lines[0]["internal_missing_minutes"] == 3
    assert lines[-1]["archive_count"] == 1
    assert lines[-1]["downloaded_count"] == 1
    assert received["report"] == report
    assert received["symbols"] == ("ETHUSDT",)
    limits = received["limits"]
    assert isinstance(limits, cli.MonthlyArchiveLimits)
    assert limits.maximum_archives == 1
    assert limits.maximum_concurrency == 2
    assert limits.maximum_total_compressed_bytes == 4096
    assert limits.maximum_elapsed_seconds == 12
    assert limits.maximum_decode_processes == 2


def test_historical_coverage_dispatches_verified_offline_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability = _research_pool()
    report = SimpleNamespace(
        available_archive_count=1,
        not_found_archive_count=1,
        report_id="sha256:" + "e" * 64,
        symbols=("ETHUSDT", "SOLUSDT"),
    )
    published = SimpleNamespace(
        already_present=False,
        path=tmp_path / "coverage",
        report=report,
    )
    received: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object):
        received["availability"] = args[0]
        received.update(kwargs)
        return published

    monkeypatch.setattr(
        cli,
        "load_published_archive_availability",
        lambda path: availability,
    )
    monkeypatch.setattr(cli, "build_historical_coverage", fake_build)

    exit_code = cli.main(
        [
            "historical-coverage",
            "--availability",
            str(tmp_path / "availability"),
            "--monthly-root",
            str(tmp_path / "monthly"),
            "--symbol",
            "ETHUSDT",
            "--maximum-archives",
            "1",
            "--output",
            str(tmp_path / "output"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["event"] == "historical_coverage"
    assert payload["symbol_count"] == 2
    assert received["availability"] == availability
    assert received["symbols"] == ("ETHUSDT",)
    limits = received["limits"]
    assert isinstance(limits, cli.CoverageLimits)
    assert limits.maximum_archives == 1


def test_research_corpus_dispatches_bounded_offline_materialization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability = _research_pool()
    manifest = SimpleNamespace(
        manifest_id="sha256:" + "d" * 64,
        month="2025-01",
        parquet_bytes=1234,
        row_count=2,
        source_manifest_id="sha256:" + "e" * 64,
        symbol="ETHUSDT",
    )
    publication = SimpleNamespace(
        already_present=False,
        manifest=manifest,
        path=tmp_path / "partition",
    )
    result = SimpleNamespace(
        archive_count=1,
        availability_report_id="sha256:" + "f" * 64,
        created_count=1,
        excluded_symbol_months=(("REDUSDT", "2025-03"),),
        parquet_bytes=1234,
        reused_count=0,
        row_count=2,
    )
    received: dict[str, object] = {}

    def fake_materialize(*args: object, **kwargs: object):
        received["availability"] = args[0]
        received.update(kwargs)
        callback = kwargs["on_published"]
        assert callable(callback)
        callback(publication)
        return result

    monkeypatch.setattr(
        cli,
        "load_published_archive_availability",
        lambda path: availability,
    )
    monkeypatch.setattr(cli, "materialize_research_corpus", fake_materialize)

    exit_code = cli.main(
        [
            "research-corpus",
            "--availability",
            str(tmp_path / "availability"),
            "--monthly-root",
            str(tmp_path / "monthly"),
            "--symbol",
            "ETHUSDT",
            "--exclude-symbol-month",
            "REDUSDT:2025-03",
            "--maximum-archives",
            "1",
            "--maximum-rows",
            "2",
            "--maximum-source-uncompressed-bytes",
            "4096",
            "--maximum-parquet-bytes",
            "2048",
            "--maximum-elapsed-seconds",
            "12",
            "--maximum-concurrency",
            "2",
            "--output",
            str(tmp_path / "research"),
        ]
    )

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert [line["event"] for line in lines] == [
        "research_partition",
        "research_corpus_complete",
    ]
    assert lines[-1]["row_count"] == 2
    assert lines[-1]["excluded_symbol_months"] == ["REDUSDT:2025-03"]
    assert received["availability"] == availability
    assert received["monthly_root"] == tmp_path / "monthly"
    assert received["output_root"] == tmp_path / "research"
    assert received["symbols"] == ("ETHUSDT",)
    assert received["excluded"] == frozenset({("REDUSDT", "2025-03")})
    limits = received["limits"]
    assert isinstance(limits, cli.ResearchCorpusLimits)
    assert limits.maximum_archives == 1
    assert limits.maximum_rows == 2
    assert limits.maximum_source_uncompressed_bytes == 4096
    assert limits.maximum_parquet_bytes == 2048
    assert limits.maximum_elapsed_seconds == 12
    assert limits.maximum_concurrency == 2


def test_qualify_live_service_dispatches_without_bounded_duration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    async def fake_service(**kwargs: object) -> tuple[object, ...]:
        received.update(kwargs)
        cli._emit_json(
            {
                "event": "service_status",
                "phase": "stopped",
                "ready": False,
                "run_id": str(FIXED_RUN_ID),
            }
        )
        return ()

    monkeypatch.setattr(cli, "_qualify_live_service", fake_service)
    monkeypatch.setattr(cli, "uuid4", lambda: FIXED_RUN_ID)

    exit_code = cli.main(
        [
            "qualify-live-service",
            "--output",
            str(tmp_path),
            "--symbol",
            "ETHUSDT",
            "--hot-symbol",
            "ETHUSDT",
            "--startup-timeout-seconds",
            "12",
            "--maximum-uncompressed-capture-bytes",
            "4096",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "event": "service_status",
        "phase": "stopped",
        "ready": False,
        "run_id": str(FIXED_RUN_ID),
    }
    assert received == {
        "hot_symbols": ("ETHUSDT",),
        "maximum_uncompressed_capture_bytes": 4096,
        "output": tmp_path,
        "requested_symbols": ("ETHUSDT",),
        "run_id": FIXED_RUN_ID,
        "startup_timeout_seconds": 12.0,
    }


def test_shutdown_signal_stops_a_real_child_process() -> None:
    script = textwrap.dedent(
        """
        import asyncio

        from crypto_boom.cli import _install_shutdown_signals


        async def main() -> None:
            stop = asyncio.Event()
            restore = _install_shutdown_signals(stop)
            print("ready", flush=True)
            await stop.wait()
            restore()
            print("stopped", flush=True)


        asyncio.run(main())
        """
    )
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == "ready\n"
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    assert process.returncode == 0
    assert stdout == "stopped\n"
    assert stderr == ""
