"""Reproducible archive-observed cohorts using public acquisition interfaces.

Directory presence is not historical listing/status metadata. No model, future
return or current exchange trading status determines membership.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, replace
from datetime import date
from itertools import islice
from pathlib import Path
from time import monotonic
from uuid import uuid4

from crypto_boom import _artifacts
from crypto_boom.history.availability import (
    AvailabilityLimits,
    AvailabilityState,
    HistoricalArchiveSymbolCatalog,
    MonthlyAvailabilityRequest,
    discover_spot_archive_symbols,
    load_published_archive_availability,
    probe_historical_archive_availability,
)
from crypto_boom.history.monthly import MonthlyArchiveLimits, acquire_monthly_archives
from crypto_boom.storage.source import (
    ResearchCorpusLimits,
    ResearchStorageError,
    load_published_research_partition,
    materialize_research_corpus,
)


def select_archive_sample(
    catalog: HistoricalArchiveSymbolCatalog,
    *,
    count: int,
    seed: str,
    excluded_symbols: tuple[str, ...] = (),
) -> dict:
    """Stable hash ordering, never filter for future full-period survival."""
    if (
        type(count) is not int
        or not 1 <= count <= 128
        or not isinstance(seed, str)
        or not seed
    ):
        raise ValueError("sample needs a seed and 1-128 symbols")
    candidates = sorted(
        s
        for s in catalog.symbols
        if re.fullmatch(r"[A-Z0-9]{1,24}USDT", s) and s not in excluded_symbols
    )
    if len(candidates) < count:
        raise ValueError("sample exceeds observed candidate universe")
    selected = sorted(
        sorted(
            candidates, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
        )[:count]
    )
    return {
        "schema": "archive-sample-v1",
        "catalog": asdict(catalog),
        "seed": seed,
        "candidate_count": len(candidates),
        "symbols": selected,
        "excluded_symbols": sorted(set(excluded_symbols)),
        "membership": "historical_archive_presence_not_verified_historical_trading_status",
    }


async def acquire_sample_pool(
    request: MonthlyAvailabilityRequest,
    *,
    output: Path,
    count: int = 24,
    seed: str = "forward-path-v1",
    reuse_corpora: tuple[Path, ...] = (),
    data_root: Path | None = None,
    excluded_symbols: tuple[str, ...] = (),
    maximum_download_bytes: int = 1_073_741_824,
    quarantined_partitions: tuple[tuple[str, str, str], ...] = (),
) -> dict:
    """Resume a pinned cohort; reuse verified partitions, acquire only missing ones.

    Output pool.json is a completion receipt, never a claim that every sampled
    symbol was trading. Missing monthly archives remain explicit not_found rows.
    A failed campaign preserves source publications but has no completion marker.
    """
    if len(request.months) * count > 512:
        raise ValueError("campaign exceeds 512 symbol-months")
    if not 1 <= maximum_download_bytes <= 4_294_967_296:
        raise ValueError("download budget must be in (0, 4 GiB]")
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection["seed"],
            len(selection["symbols"]),
            selection["start_month"],
            selection["end_month"],
            selection["excluded_symbols"],
        ) != (
            seed,
            count,
            request.months[0],
            request.months[-1],
            sorted(set(excluded_symbols)),
        ):
            raise ValueError("campaign configuration differs; use a new output path")
        raw = selection["catalog"]
        catalog = HistoricalArchiveSymbolCatalog(
            source_url=raw["source_url"],
            prefix=raw["prefix"],
            observed_at_ns=raw["observed_at_ns"],
            page_count=raw["page_count"],
            symbols=tuple(raw["symbols"]),
        )
    else:
        catalog = await discover_spot_archive_symbols()
        selection = select_archive_sample(
            catalog, count=count, seed=seed, excluded_symbols=excluded_symbols
        )
        selection.update(start_month=request.months[0], end_month=request.months[-1])
        _artifacts.write_exclusive_bytes(
            selection_path, _artifacts.canonical_json(selection)
        )
    symbols = tuple(selection["symbols"])
    pointer = output / "availability-path.json"
    if pointer.exists():
        saved = json.loads(pointer.read_text(encoding="utf-8"))
        report_path = (
            _artifacts.resolve_reference(saved["path"], base=output)
            if saved.get("schema") == "relative-path-v1"
            else Path(saved["path"])
        )
        report = load_published_archive_availability(report_path)
    else:
        published = await asyncio.wait_for(
            probe_historical_archive_availability(
                catalog,
                request,
                symbols=symbols,
                output_root=output / "availability",
                limits=AvailabilityLimits(
                    maximum_probes=512, retry_attempts=2, request_timeout_seconds=12
                ),
            ),
            timeout=1800,
        )
        report = published.report
        _artifacts.write_exclusive_bytes(
            pointer,
            _artifacts.canonical_json(
                {
                    "schema": "relative-path-v1",
                    "path": _artifacts.relative_reference(published.path, base=output),
                }
            ),
        )
    if report.unresolved_count:
        raise ValueError(
            "unresolved source availability; preserve audit and use a new campaign after resolving source access"
        )
    if {(p.symbol, p.month) for p in report.probes} != {
        (s, m) for s in symbols for m in request.months
    }:
        raise ValueError("availability does not cover the pinned selection")
    if report.pool_id != _artifacts.content_id(asdict(catalog)):
        raise ValueError("availability catalog identity mismatch")
    quarantine = {(s, m): reason for s, m, reason in quarantined_partitions}
    available_keys = {
        (p.symbol, p.month)
        for p in report.probes
        if p.state is AvailabilityState.AVAILABLE
    }
    if (
        len(quarantine) != len(quarantined_partitions)
        or not set(quarantine).issubset(available_keys)
        or any(not r.strip() for r in quarantine.values())
    ):
        raise ValueError(
            "quarantine requires unique available partitions and explicit reasons"
        )
    corpus = (
        data_root / "canonical/archives" if data_root is not None else output / "corpus"
    )
    monthly = data_root / "raw/monthly" if data_root is not None else output / "monthly"
    roots = (
        (corpus, *reuse_corpora, output / "corpus")
        if data_root is not None
        else (*reuse_corpora, corpus)
    )
    partitions: dict[tuple[str, str], Path] = {}
    missing = []
    for probe in report.probes:
        if (
            probe.state is not AvailabilityState.AVAILABLE
            or (probe.symbol, probe.month) in quarantine
        ):
            continue
        for root in roots:
            directory = (
                root
                / "binance/spot/research-source-klines"
                / probe.symbol
                / "1m"
                / probe.month
            )
            for path in sorted(directory.glob("*/manifest.json")):
                published = load_published_research_partition(path.parent)
                if published.manifest.source_revision == probe.checksum_sha256:
                    partitions[(probe.symbol, probe.month)] = path.parent.resolve()
            if (probe.symbol, probe.month) in partitions:
                break
        if (probe.symbol, probe.month) not in partitions:
            missing.append(probe)
    if missing:
        missing_report = replace(report, probes=tuple(missing))
        await acquire_monthly_archives(
            missing_report,
            output_root=monthly,
            ingestion_run_id=uuid4(),
            limits=MonthlyArchiveLimits(
                maximum_archives=512,
                maximum_concurrency=2,
                maximum_total_compressed_bytes=maximum_download_bytes,
                maximum_elapsed_seconds=1800,
                retry_attempts=2,
            ),
        )
        materialized = materialize_research_corpus(
            missing_report,
            monthly_root=monthly,
            output_root=corpus,
            limits=ResearchCorpusLimits(
                maximum_archives=512, maximum_elapsed_seconds=900
            ),
        )
        for part in materialized.publications:
            partitions[(part.manifest.symbol, part.manifest.month)] = (
                part.path.resolve()
            )
    ledger = []
    for probe in report.probes:
        key = (probe.symbol, probe.month)
        row = probe.to_mapping()
        if key in quarantine:
            row.update(state="quarantined", reason=quarantine[key])
        elif probe.state is AvailabilityState.AVAILABLE:
            part = load_published_research_partition(partitions[key])
            row.update(
                path=str(part.path / "klines.parquet"),
                parquet_sha256=part.manifest.parquet_sha256,
                rows=part.manifest.row_count,
                first_open_us=part.manifest.first_open_time_us,
                last_open_us=part.manifest.last_open_time_us,
            )
        ledger.append(row)
    result = {
        "schema": "sample-pool-v1",
        "selection_id": _artifacts.content_id(selection),
        "selection": selection,
        "availability_id": report.report_id,
        "partitions": ledger,
        "available_symbol_months": len(partitions),
        "quarantined_symbol_months": len(quarantine),
        "coverage_status": "partial_quality_quarantine"
        if quarantine
        else "all_available_archives_admitted",
        "not_found_symbol_months": report.not_found_count,
        "symbols_with_data": sorted({symbol for symbol, _ in partitions}),
        "reused_symbol_months": len(partitions) - len(missing),
    }
    target = output / "pool.json"
    result["schema"] = "sample-pool-v2"
    result["pool_id"] = sample_pool_identity(result)
    payload = _artifacts.canonical_json(_portable_pool(result, base=output))
    if target.exists():
        # Reuse count describes this invocation, not dataset identity.
        old = load_sample_pool(target)
        if old["selection_id"] != result["selection_id"] or old["partitions"] != ledger:
            raise ValueError("completed pool conflicts with current verified sources")
    else:
        _artifacts.write_exclusive_bytes(target, payload)
    return result


def load_sample_pool(path: Path) -> dict:
    pool: dict = json.loads(path.read_text(encoding="utf-8"))
    if (
        pool.get("schema") not in ("sample-pool-v1", "sample-pool-v2")
        or _artifacts.content_id(pool["selection"]) != pool["selection_id"]
    ):
        raise ValueError("invalid sample pool")
    if pool["schema"] == "sample-pool-v2":
        if pool.get("pool_id") != sample_pool_identity(pool):
            raise ValueError("sample pool identity mismatch")
        pool = {
            **pool,
            "partitions": [
                {
                    **row,
                    "path": str(
                        _artifacts.resolve_reference(row["path"], base=path.parent)
                    ),
                }
                if "path" in row
                else row
                for row in pool["partitions"]
            ],
        }
    selection: dict = pool["selection"]
    months = MonthlyAvailabilityRequest(
        date.fromisoformat(selection["start_month"] + "-01"),
        date.fromisoformat(selection["end_month"] + "-01"),
    ).months
    keys = [(row["symbol"], row["month"]) for row in pool["partitions"]]
    if len(keys) != len(set(keys)) or set(keys) != {
        (s, m) for s in selection["symbols"] for m in months
    }:
        raise ValueError("sample pool ledger coverage mismatch")
    available = [r for r in pool["partitions"] if r["state"] == "available"]
    if (
        len(available) != pool["available_symbol_months"]
        or sorted({r["symbol"] for r in available}) != pool["symbols_with_data"]
    ):
        raise ValueError("sample pool admitted coverage mismatch")
    for row in pool["partitions"]:
        if row["state"] not in ("available", "not_found", "quarantined") or (
            row["state"] == "quarantined" and not row.get("reason")
        ):
            raise ValueError("invalid sample pool admission state")
        if row["state"] == "available":
            part = load_published_research_partition(Path(row["path"]).parent)
            if (
                part.manifest.symbol,
                part.manifest.month,
                part.manifest.parquet_sha256,
                part.manifest.source_revision,
            ) != (
                row["symbol"],
                row["month"],
                row["parquet_sha256"],
                row["checksum_sha256"],
            ):
                raise ValueError("sample pool source identity mismatch")
    return pool


def audit_local_corpus(corpus: Path, request: MonthlyAvailabilityRequest) -> dict:
    """Audit existing local partitions, never select on future returns or survival.

    Local absence is not evidence of archive unavailability or historical trading
    status. No network, repair, deletion, feature computation or training occurs.
    """
    import calendar
    from collections import Counter
    from datetime import UTC, datetime

    import polars as pl

    from crypto_boom.bars import MINUTE_US, SOURCE_COLUMNS, admit_bars

    root = corpus / "binance/spot/research-source-klines"
    if not root.is_dir():
        raise ValueError("research corpus directory does not exist")
    paths = sorted(
        islice(
            (
                p
                for p in root.glob("*/1m/*/*/manifest.json")
                if p.parent.parent.name in request.months
            ),
            2049,
        )
    )
    if not paths or len(paths) > 2048:
        raise ValueError("audit requires 1-2048 local partitions")
    start, total_bytes, ledger = monotonic(), 0, []
    for path in paths:
        if monotonic() - start > 900:
            raise ValueError("audit exceeded fifteen minutes; no complete receipt")
        row = {
            "path": str(path.parent.resolve()),
            "symbol": path.parents[3].name,
            "month": path.parents[1].name,
        }
        try:
            size = (path.parent / "klines.parquet").stat().st_size
        except OSError as error:
            ledger.append({**row, "status": "rejected", "reason": str(error)})
            continue
        total_bytes += size
        if total_bytes > 16 * 1024**3:
            raise ValueError("audit exceeds 16 GiB; no complete receipt")
        try:
            part = load_published_research_partition(path.parent)
            if part.manifest.row_count > 50_000:
                raise ValueError("one-minute monthly partition exceeds row budget")
            source = admit_bars(
                pl.read_parquet(
                    path.parent / "klines.parquet", columns=list(SOURCE_COLUMNS)
                )
            )
            times = source["open_time"].dt.epoch("us")
            year, month = map(int, row["month"].split("-"))
            calendar_rows = calendar.monthrange(year, month)[1] * 1440
            month_start = (
                int(datetime(year, month, 1, tzinfo=UTC).timestamp()) * 1_000_000
            )
            if (
                times[0] < month_start
                or times[-1] >= month_start + calendar_rows * MINUTE_US
            ):
                raise ValueError("bar timestamp outside partition month")
            invalid = (
                ~pl.col("quality_complete") | (pl.col("quality_state") != "valid")
            ).sum()
            ledger.append(
                {
                    **row,
                    "status": "verified",
                    "source_revision": part.manifest.source_revision,
                    "parquet_sha256": part.manifest.parquet_sha256,
                    "rows": len(source),
                    "first_open_us": int(times[0]),
                    "last_open_us": int(times[-1]),
                    "quality_invalid_rows": source.select(invalid).item(),
                    "internal_missing_minutes": int(
                        (times[-1] - times[0]) // MINUTE_US + 1 - len(source)
                    ),
                    "calendar_unobserved_minutes": calendar_rows - len(source),
                    "zero_quote_rows": int((source["quote_turnover"] == 0).sum()),
                    "zero_trade_rows": int((source["trade_count"] == 0).sum()),
                }
            )
        except (
            ResearchStorageError,
            OSError,
            ValueError,
            pl.exceptions.PolarsError,
        ) as error:
            ledger.append(
                {
                    **row,
                    "status": "rejected",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    verified = [r for r in ledger if r["status"] == "verified"]
    keys = Counter((r["symbol"], r["month"]) for r in ledger)
    symbols = sorted({r["symbol"] for r in ledger})
    return {
        "schema": "local-corpus-quality-v1",
        "corpus": str(corpus.resolve()),
        "start_month": request.months[0],
        "end_month": request.months[-1],
        "symbols_present": len(symbols),
        "partitions_present": len(ledger),
        "verified_partitions": len(verified),
        "rejected_partitions": len(ledger) - len(verified),
        "multiple_revision_symbol_months": sum(n > 1 for n in keys.values()),
        "parquet_bytes_examined": total_bytes,
        "partitions": ledger,
        "locally_absent_months": {
            s: [m for m in request.months if (s, m) not in keys] for s in symbols
        },
        "interpretation": "Inventory and integrity only; no historical eligibility certification. Partial months and zero trading are observations, not automatic rejection. No full-period survival filter.",
    }


def sample_pool_identity(pool: dict) -> str:
    """Selection and partition facts identify a pool; physical locators do not."""
    return _artifacts.content_id(
        {
            "schema": "sample-pool-identity-v2",
            "selection_id": pool["selection_id"],
            "availability_id": pool["availability_id"],
            "partitions": [
                {k: v for k, v in row.items() if k != "path"}
                for row in sorted(
                    pool["partitions"], key=lambda r: (r["symbol"], r["month"])
                )
            ],
        }
    )


def _portable_pool(pool: dict, *, base: Path) -> dict:
    return {
        **pool,
        "schema": "sample-pool-v2",
        "pool_id": sample_pool_identity(pool),
        "partitions": [
            {**row, "path": _artifacts.relative_reference(Path(row["path"]), base=base)}
            if "path" in row
            else row
            for row in pool["partitions"]
        ],
    }


def export_sample_pool(source: Path, destination: Path) -> dict:
    """Publish a verified portable manifest; never move data or overwrite history."""
    if destination.exists():
        raise FileExistsError(destination)
    pool = load_sample_pool(source)
    portable = _portable_pool(pool, base=destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _artifacts.write_exclusive_bytes(destination, _artifacts.canonical_json(portable))
    return portable
