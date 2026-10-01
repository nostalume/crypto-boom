"""The study acquisition runner consumes library operations, not CLI globals."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_boom.history.availability import AvailabilityState
from crypto_boom.history.monthly import ArchiveError
from crypto_boom.storage.source import ResearchStorageError

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "src" / "crypto_boom" / "research" / "acquisition.py"


def test_research_package_does_not_mutate_module_search_paths():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; before = sys.path.copy(); "
            "import crypto_boom.research as research; "
            "assert sys.path == before; assert len(research.__path__) == 1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def acquisition():
    spec = importlib.util.spec_from_file_location("m1_acquisition_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.research
def test_probe_calls_typed_availability_operation(acquisition, monkeypatch, tmp_path):
    seen = {}
    acquisition.AVAILABILITY = tmp_path / "availability"
    acquisition.SELECTION = tmp_path / "selection.json"
    acquisition.SELECTION.write_text(
        json.dumps(
            {
                "selected_symbols": ["BBBUSD"],
                "window": {"start_month": "2025-09", "end_month": "2025-10"},
            }
        ),
        encoding="utf-8",
    )
    acquisition.EXTRA = ["AAAUSDT"]
    monkeypatch.setattr(acquisition, "pool_directory", lambda: tmp_path / "pool")
    monkeypatch.setattr(acquisition, "load_published_research_pool", lambda path: path)

    async def fake_probe(pool, request, **options):
        seen.update(pool=pool, request=request, options=options)
        return SimpleNamespace(path=tmp_path / "published")

    monkeypatch.setattr(acquisition, "probe_monthly_archive_availability", fake_probe)
    original_argv = sys.argv
    acquisition.probe()
    assert sys.argv is original_argv
    assert seen["pool"] == tmp_path / "pool"
    assert seen["request"].start_month == date(2025, 9, 1)
    assert seen["request"].end_month == date(2025, 10, 1)
    assert seen["options"]["symbols"] == ("AAAUSDT",)
    assert seen["options"]["output_root"] == acquisition.AVAILABILITY
    assert seen["options"]["limits"].maximum_probes == acquisition.MAXIMUM_PROBES

    async def stall_probe(pool, request, **options):
        await acquisition.asyncio.sleep(0.1)

    monkeypatch.setattr(acquisition, "probe_monthly_archive_availability", stall_probe)
    acquisition.MAXIMUM_ELAPSED_SECONDS = 0.001
    with pytest.raises(TimeoutError):
        acquisition.probe()


@pytest.mark.research
def test_download_bisects_failure_and_records_unpublished_months(
    acquisition, monkeypatch, tmp_path
):
    acquisition.STUDY = tmp_path
    acquisition.MONTHLY_ROOT = tmp_path / "monthly"
    acquisition.EXTRA = ["AAAUSDT", "BBBUSDT"]
    acquisition.EXCLUDED_SYMBOL_MONTHS = ()
    report = SimpleNamespace(
        probes=tuple(
            SimpleNamespace(
                symbol=symbol, month="2025-09", state=AvailabilityState.AVAILABLE
            )
            for symbol in ("AAAUSDT", "BBBUSDT")
        )
    )
    monkeypatch.setattr(acquisition, "availability_directory", lambda: tmp_path)
    monkeypatch.setattr(
        acquisition, "load_published_archive_availability", lambda path: report
    )
    monkeypatch.setattr(
        acquisition, "published_symbol_months", lambda: {("AAAUSDT", "2025-09")}
    )
    calls = []

    async def fake_acquire(received, **options):
        assert received is report
        calls.append(options)
        if "BBBUSDT" in options["symbols"]:
            raise ArchiveError("upstream archive is invalid")
        return object()

    monkeypatch.setattr(acquisition, "acquire_monthly_archives", fake_acquire)
    original_argv = sys.argv
    with pytest.raises(SystemExit, match="neither published nor declared excluded"):
        acquisition.download()
    assert sys.argv is original_argv
    assert [call["symbols"] for call in calls] == [
        ("AAAUSDT", "BBBUSDT"),
        ("AAAUSDT",),
        ("BBBUSDT",),
    ]
    assert all(
        call["limits"].maximum_archives == acquisition.MAXIMUM_ARCHIVES
        for call in calls
    )
    receipt = json.loads((tmp_path / "download-failures.json").read_text())
    assert receipt["failures"][0]["symbols"] == ["BBBUSDT"]
    assert receipt["missing_symbol_months"] == [
        {"symbol": "BBBUSDT", "month": "2025-09"}
    ]


@pytest.mark.research
def test_download_budget_stops_bisection_after_partial_write(
    acquisition, monkeypatch, tmp_path
):
    acquisition.MONTHLY_ROOT = tmp_path / "monthly"
    acquisition.MONTHLY_ROOT.mkdir()
    acquisition.EXTRA = ["AAAUSDT", "BBBUSDT"]
    acquisition.MAXIMUM_COMPRESSED_BYTES = 10
    monkeypatch.setattr(acquisition, "availability_directory", lambda: tmp_path)
    report = SimpleNamespace(
        probes=tuple(
            SimpleNamespace(
                symbol=symbol, month="2025-09", state=AvailabilityState.AVAILABLE
            )
            for symbol in acquisition.EXTRA
        )
    )
    monkeypatch.setattr(
        acquisition, "load_published_archive_availability", lambda path: report
    )
    calls = []

    async def partial_acquire(received, **options):
        calls.append(options)
        (acquisition.MONTHLY_ROOT / "partial.zip").write_bytes(b"0123456789")
        raise ArchiveError("partial publication")

    monkeypatch.setattr(acquisition, "acquire_monthly_archives", partial_acquire)
    with pytest.raises(SystemExit, match="exhausted its byte or elapsed-time budget"):
        acquisition.download()
    assert len(calls) == 1
    assert calls[0]["limits"].maximum_total_compressed_bytes == 10


@pytest.mark.research
def test_download_reconciliation_rejects_unverified_manifest(
    acquisition, monkeypatch, tmp_path
):
    acquisition.MONTHLY_ROOT = tmp_path
    (tmp_path / "published").mkdir()
    (tmp_path / "published" / "manifest.json").write_text("{}")

    def reject(path):
        raise ArchiveError(f"corrupt publication at {path}")

    monkeypatch.setattr(acquisition, "load_published_monthly_archive", reject)
    with pytest.raises(ArchiveError, match="corrupt publication"):
        acquisition.published_symbol_months()


@pytest.mark.research
def test_corpus_preserves_exclusion_and_fails_closed(
    acquisition, monkeypatch, tmp_path
):
    acquisition.MONTHLY_ROOT = tmp_path / "monthly"
    acquisition.CORPUS_ROOT = tmp_path / "corpus"
    monkeypatch.setattr(acquisition, "availability_directory", lambda: tmp_path)
    report = SimpleNamespace(
        probes=(
            SimpleNamespace(
                symbol="REDUSDT", month="2025-03", state=AvailabilityState.AVAILABLE
            ),
        )
    )
    monkeypatch.setattr(
        acquisition, "load_published_archive_availability", lambda path: report
    )
    seen = {}

    def fake_materialize(received, **options):
        assert received is report
        seen.update(options)
        return SimpleNamespace(archive_count=2, row_count=10)

    monkeypatch.setattr(acquisition, "materialize_research_corpus", fake_materialize)
    acquisition.corpus()
    assert seen["excluded"] == frozenset(acquisition.EXCLUDED_SYMBOL_MONTHS)
    assert seen["monthly_root"] == acquisition.MONTHLY_ROOT
    assert seen["output_root"] == acquisition.CORPUS_ROOT
    assert seen["limits"].maximum_rows == acquisition.MAXIMUM_ROWS

    # The same runner also serves windows where the declared REDUSDT exclusion
    # is outside the availability report; it must not pass an unknown pair.
    other_report = SimpleNamespace(probes=())
    monkeypatch.setattr(
        acquisition,
        "load_published_archive_availability",
        lambda path: other_report,
    )

    def other_materialize(received, **options):
        assert received is other_report
        seen.update(options)
        return SimpleNamespace(archive_count=0, row_count=0)

    monkeypatch.setattr(acquisition, "materialize_research_corpus", other_materialize)
    acquisition.corpus()
    assert seen["excluded"] == frozenset()

    monkeypatch.setattr(
        acquisition, "load_published_archive_availability", lambda path: report
    )

    def fail_materialize(received, **options):
        raise ResearchStorageError("partial source")

    monkeypatch.setattr(acquisition, "materialize_research_corpus", fail_materialize)
    with pytest.raises(SystemExit, match="partial source"):
        acquisition.corpus()
