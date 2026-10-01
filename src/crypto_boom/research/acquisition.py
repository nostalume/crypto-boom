"""Universe expansion acquisition: the bounded Phase-2 pipeline for one study.

This implementation draws a systematic sample from the observed
Binance Spot USDT pool, audits archive availability for the admitted window, downloads under a byte
and an elapsed budget, and materialises the decoded Parquet. It performs no measurement and no fit;
the occurrence and independence axes must be evaluated separately.

Run one stage at a time, from the repository root:

    .venv\\Scripts\\python.exe -m crypto_boom.research.acquisition select
    .venv\\Scripts\\python.exe -m crypto_boom.research.acquisition probe
    .venv\\Scripts\\python.exe -m crypto_boom.research.acquisition download
    .venv\\Scripts\\python.exe -m crypto_boom.research.acquisition corpus

The stages are idempotent in the sense that each writes its own receipt under the study directory and
re-running a stage re-reads the previous receipt rather than re-deriving it, so a partially completed
download is resumed by the venue client's own cache rather than by this script.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from crypto_boom.history.availability import (
    AvailabilityLimits,
    AvailabilityState,
    MonthlyAvailabilityRequest,
    load_published_archive_availability,
    probe_monthly_archive_availability,
)
from crypto_boom.history.monthly import (
    ArchiveError,
    MonthlyArchiveLimits,
    acquire_monthly_archives,
    load_published_monthly_archive,
)
from crypto_boom.storage.source import (
    ResearchCorpusLimits,
    ResearchStorageError,
    materialize_research_corpus,
)
from crypto_boom.universe import load_published_research_pool

# This is a repository study CLI, not an installed-package data locator. Its
# documented invocation is from the repository root; avoid deriving the data
# directory from the installed module's location.
ROOT = Path.cwd().resolve()

#: Default study: the pre-registered 2025-09..2026-04 expansion. `--study`, `--start-month` and
#: `--end-month` retarget the whole pipeline at another window, which is how the frozen evaluator's
#: confirm block (2026-05..08, `universe-expansion-evaluator-protocol.md` section 5) is acquired
#: without overwriting any receipt of the original study.
STUDY = ROOT / "data/universe-expansion-20260927-v1"
POOL_ROOT = STUDY / "pool.json"
SELECTION = STUDY / "selection.json"
AVAILABILITY = STUDY / "availability.json"
MONTHLY_ROOT = STUDY / "monthly"
CORPUS_ROOT = STUDY / "corpus"

# Fixed in the protocol before any probe: sorted(pool) minus the current cohort, then every third.
STRIDE = 3
START_MONTH = "2025-09"
END_MONTH = "2026-04"

#: Positional arguments after the stage name, for stages that accept symbols.
EXTRA: list[str] = []

#: Symbol-months the venue publishes but this repository's frozen schema contract refuses, so they can
#: never be decoded. Declared rather than repaired: the one-minute kline contract has no tolerance, and
#: inventing one after seeing a failure would be an undeclared adjustment to the data a gate reads.
#: See `universe-expansion-successor-protocol.md` section 24.
EXCLUDED_SYMBOL_MONTHS: tuple[tuple[str, str], ...] = (
    # Binance's own 2025-03 REDUSDT archive holds one row in 44,220 whose close_time is
    # 1741251599999000 where the contract requires open_time + 59_999_999 = 1741251599999999 - a
    # one-character upstream typo, not a grain error, which `codec.py:157` raises on.
    ("REDUSDT", "2025-03"),
)


def configure(
    study: str | None,
    start_month: str | None,
    end_month: str | None,
    pool_study: str | None = None,
) -> None:
    """Retarget the module constants at another study directory and window.

    Every stage reads these names at call time rather than binding them at import, so assigning the
    globals before a stage runs is sufficient. A relative `--study` resolves against the repository
    root, and a `--study` value is used verbatim as the receipt's study id. `--pool-study` points the
    observed-pool lookup at another study, so a second window reuses the frozen pool observation
    instead of duplicating it.
    """
    global STUDY, POOL_ROOT, SELECTION, AVAILABILITY, MONTHLY_ROOT, CORPUS_ROOT
    global START_MONTH, END_MONTH
    if study:
        path = Path(study)
        STUDY = path if path.is_absolute() else (ROOT / path)
        POOL_ROOT = STUDY / "pool.json"
        SELECTION = STUDY / "selection.json"
        AVAILABILITY = STUDY / "availability.json"
        MONTHLY_ROOT = STUDY / "monthly"
        CORPUS_ROOT = STUDY / "corpus"
    if pool_study:
        path = Path(pool_study)
        POOL_ROOT = (path if path.is_absolute() else (ROOT / path)) / "pool.json"
    if start_month:
        START_MONTH = start_month
    if end_month:
        END_MONTH = end_month
    for label, value in (("--start-month", START_MONTH), ("--end-month", END_MONTH)):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
            raise SystemExit(f"{label} must look like YYYY-MM, got {value!r}")
    if START_MONTH > END_MONTH:
        raise SystemExit(f"start month {START_MONTH} is after end month {END_MONTH}")


# The user's standing authorisation: at most 5 GB of new data on disk and at most 60 minutes per
# acquisition invocation. The compressed budget is set below 5 GB so the decoded Parquet also fits.
MAXIMUM_PROBES = 1_600
MAXIMUM_ARCHIVES = 1_000
MAXIMUM_COMPRESSED_BYTES = 4 * 1024**3
MAXIMUM_ELAPSED_SECONDS = 3_300
MAXIMUM_CONCURRENCY = 8
MAXIMUM_DECODE_PROCESSES = 2
MAXIMUM_PARQUET_BYTES = 4 * 1024**3
# `research-corpus` defaults `--maximum-rows` to 25,000,000 and refuses silent truncation. The
# failure message reports a RUNNING total, not the requirement: `src/crypto_boom/storage/source.py:302`
# accumulates `manifest.row_count` and trips as soon as the running total exceeds the bound, so each
# raised bound got further before failing (25,040,250 at 25,000,000, then 40,014,450 at
# 40,000,000). The ceiling for the admitted set is 985 symbol-months x 44,640 minutes =
# 43,970,400 rows, so the bound is set above that with headroom for a re-run.
MAXIMUM_ROWS = 50_000_000
MAXIMUM_SOURCE_UNCOMPRESSED_BYTES = 16 * 1024**3
DOWNLOAD_CHUNK = 12

CURRENT_COHORT = (
    "ADAUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ETCUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "XRPUSDT",
)


def pool_directory() -> Path:
    """The content-addressed directory holding `pool.json` and `exchange-info.json`.

    `--pool` takes the directory whose *name* is the pool id, not the publication root: the loader
    verifies `path.name == pool_id.removeprefix("sha256:")`.
    """
    records = sorted(POOL_ROOT.rglob("pool.json"))
    if len(records) != 1:
        raise ValueError(
            f"expected exactly one pool record under {POOL_ROOT}, found {len(records)}"
        )
    directory = records[0].parent
    record = json.loads(records[0].read_text(encoding="utf-8"))
    if directory.name != record["pool_id"].removeprefix("sha256:"):
        raise ValueError(f"{directory} does not match pool id {record['pool_id']}")
    return directory


def availability_directory() -> Path:
    """The content-addressed directory holding `availability.json`.

    Like `--pool`, `--availability` names the artifact directory, not the publication root.
    """
    records = sorted(AVAILABILITY.rglob("availability.json"))
    if len(records) != 1:
        raise ValueError(
            f"expected exactly one availability artifact, found {len(records)}"
        )
    return records[0].parent


def select() -> None:
    pool = json.loads((pool_directory() / "pool.json").read_text(encoding="utf-8"))
    candidates = sorted(
        symbol for symbol in pool["symbols"] if symbol not in CURRENT_COHORT
    )
    selected = candidates[::STRIDE]
    STUDY.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(
        json.dumps(
            {
                "study": STUDY.name,
                "derived_from_pool_id": pool["pool_id"],
                "derived_from_pool_record": str(POOL_ROOT.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "observed_at_ns": pool["observed_at_ns"],
                "source_endpoint": pool["source_endpoint"],
                "selection_rule": (
                    "sorted(pool symbols) minus the current 12-symbol cohort, then every third symbol"
                ),
                "stride": STRIDE,
                "window": {"start_month": START_MONTH, "end_month": END_MONTH},
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "excluded_current_cohort": list(CURRENT_COHORT),
                "selected_symbols": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"candidates {len(candidates)} selected {len(selected)}")
    print(" ".join(selected))


def probe() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    symbols = EXTRA or selection["selected_symbols"]
    window = selection["window"]
    pool = load_published_research_pool(pool_directory())
    request = MonthlyAvailabilityRequest(
        date.fromisoformat(window["start_month"] + "-01"),
        date.fromisoformat(window["end_month"] + "-01"),
    )
    published = asyncio.run(
        asyncio.wait_for(
            probe_monthly_archive_availability(
                pool,
                request,
                output_root=AVAILABILITY,
                symbols=tuple(symbols),
                limits=AvailabilityLimits(maximum_probes=MAXIMUM_PROBES),
            ),
            timeout=MAXIMUM_ELAPSED_SECONDS,
        )
    )
    print(f"availability published: {published.path}")


def published_symbol_months() -> set[tuple[str, str]]:
    """Every verified published symbol-month, never just a manifest filename."""
    found: set[tuple[str, str]] = set()
    for manifest in MONTHLY_ROOT.rglob("manifest.json"):
        published = load_published_monthly_archive(manifest.parent)
        found.add((published.manifest.symbol, published.manifest.month))
    return found


def download() -> None:
    """Download the audited archives in symbol chunks, retrying a failed chunk by bisection.

    Monthly acquisition aborts its whole invocation when one archive fails to publish atomically
    (`src/crypto_boom/history/monthly.py:1196`), and that failure recurs: measured on 2026-09-28 it
    had already hit `ENSUSDT 2026-01`, `LUMIAUSDT 2025-12` and `OGUSDT 2026-02`, and an immediate
    retry of `ENSUSDT` succeeded, so it is a filesystem race rather than a bad archive. A chunk that
    fails with twelve symbols therefore loses eleven good symbols with it. This splits a failing
    chunk in half and recurses, so one bad archive costs exactly one symbol, and every other symbol
    is retried once at no extra download cost because the visitor cache and the reuse sweep make a
    repeated invocation resume. Symbols that fail even alone are recorded in
    `download-failures.json`, so an unresolved archive stays visible instead of being retried to
    convergence.
    """
    report = load_published_archive_availability(availability_directory())
    audited = {
        (probe.symbol, probe.month)
        for probe in report.probes
        if probe.state is AvailabilityState.AVAILABLE
    }
    symbols = EXTRA or sorted({symbol for symbol, _month in audited})
    failures: list[dict[str, Any]] = []
    started = monotonic()
    initial_bytes = sum(path.stat().st_size for path in MONTHLY_ROOT.rglob("*.zip"))

    def run(block: list[str]) -> None:
        written_bytes = (
            sum(path.stat().st_size for path in MONTHLY_ROOT.rglob("*.zip"))
            - initial_bytes
        )
        remaining_bytes = MAXIMUM_COMPRESSED_BYTES - max(written_bytes, 0)
        remaining_seconds = MAXIMUM_ELAPSED_SECONDS - (monotonic() - started)
        if remaining_bytes <= 0 or remaining_seconds <= 0:
            raise SystemExit(
                "download invocation exhausted its byte or elapsed-time budget"
            )
        try:
            asyncio.run(
                acquire_monthly_archives(
                    report,
                    output_root=MONTHLY_ROOT,
                    ingestion_run_id=uuid4(),
                    limits=MonthlyArchiveLimits(
                        maximum_archives=MAXIMUM_ARCHIVES,
                        maximum_concurrency=MAXIMUM_CONCURRENCY,
                        maximum_decode_processes=MAXIMUM_DECODE_PROCESSES,
                        maximum_total_compressed_bytes=remaining_bytes,
                        maximum_elapsed_seconds=remaining_seconds,
                    ),
                    symbols=tuple(block),
                )
            )
            return
        except (ArchiveError, ValueError) as error:
            failure = str(error)
        if len(block) == 1:
            failures.append({"symbols": block, "exit_code": 2})
            print(f"symbol failed alone: {block[0]}: {failure}")
            return
        middle = len(block) // 2
        print(
            f"chunk failed at {block[0]}..{block[-1]}, splitting {len(block)} symbols"
        )
        run(block[:middle])
        run(block[middle:])

    for start in range(0, len(symbols), DOWNLOAD_CHUNK):
        run(symbols[start : start + DOWNLOAD_CHUNK])

    # The audit says what the venue holds; the monthly root says what landed. Comparing the two is the
    # check whose absence let the 2026-09-29 run lose 23 symbol-months while printing "0 repeated
    # failures": the old CLI relay discarded `main()`'s returned status, so no chunk reported a failure and
    # nothing reconciled the audit against the disk. A symbol-month the audit called available and the
    # monthly root does not hold is recorded here whether or not any chunk mentioned it.
    excluded = set(EXCLUDED_SYMBOL_MONTHS)
    published = published_symbol_months()
    declared = sorted(audited & excluded)
    missing = sorted(audited - published - excluded)
    (STUDY / "download-failures.json").write_text(
        json.dumps(
            {
                "study": STUDY.name,
                "chunk_size": DOWNLOAD_CHUNK,
                "symbol_count": len(symbols),
                "failures": failures,
                "audited_symbol_months": len(audited),
                "published_symbol_months": len(published),
                "declared_excluded_symbol_months": [
                    {"symbol": symbol, "month": month} for symbol, month in declared
                ],
                "missing_symbol_months": [
                    {"symbol": symbol, "month": month} for symbol, month in missing
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"download finished: {len(symbols)} symbols, {len(failures)} failed alone, "
        f"{len(audited)} audited, {len(published)} published, "
        f"{len(declared)} declared excluded, {len(missing)} unexplained missing"
    )
    for symbol, month in missing:
        print(f"  UNEXPLAINED MISSING: {symbol} {month}")
    if missing:
        raise SystemExit(
            f"{len(missing)} audited symbol-months are neither published nor declared excluded; "
            "see download-failures.json"
        )


def corpus() -> None:
    """Decode the published monthly archives into the corpus the surface runners read.

    The corpus operation defaults to 500 archives and 25,000,000 rows and
    refuses silent truncation, so both bounds are passed explicitly here. The first two executions
    failed with `error: research corpus requires 985 archives; limit is 500` and
    `error: research corpus requires 25040250 rows; limit is 25000000` because these arguments were
    missing. The bounds are set to the module constants rather than to the exact counts, so the same
    invocation also covers a re-run after the available set grows.

    Applicable `EXCLUDED_SYMBOL_MONTHS` are passed separately: the availability audit legitimately
    reports those symbol-months as AVAILABLE, because the venue does serve an archive for them, and the
    audit is left saying so.  What the audit cannot express is that the archive's CONTENTS cannot be
    admitted, so the decode list carries that separately rather than the audit being rewritten to a
    falsehood.  Without this the stage fails on the first excluded pair, naming it
    (`verified monthly source is unavailable for REDUSDT 2025-03`) even though every published archive
    is present.
    """
    report = load_published_archive_availability(availability_directory())
    observed = {(probe.symbol, probe.month) for probe in report.probes}
    excluded = frozenset(set(EXCLUDED_SYMBOL_MONTHS) & observed)
    try:
        result = materialize_research_corpus(
            report,
            monthly_root=MONTHLY_ROOT,
            output_root=CORPUS_ROOT,
            excluded=excluded,
            limits=ResearchCorpusLimits(
                maximum_archives=MAXIMUM_ARCHIVES,
                maximum_rows=MAXIMUM_ROWS,
                maximum_source_uncompressed_bytes=MAXIMUM_SOURCE_UNCOMPRESSED_BYTES,
                maximum_concurrency=2,
                maximum_parquet_bytes=MAXIMUM_PARQUET_BYTES,
                maximum_elapsed_seconds=MAXIMUM_ELAPSED_SECONDS,
            ),
        )
    except (ResearchStorageError, ValueError) as error:
        raise SystemExit(
            f"research-corpus failed: {error}; the corpus was not published"
        ) from error
    print(
        f"research corpus published: {result.archive_count} archives, {result.row_count} rows"
    )


def symbols() -> None:
    """Record which symbols the decoded corpus can actually admit, from the availability audit.

    A symbol's available months are admitted only when they form a contiguous run ending at the
    last probed month: a missing interior month would concatenate two non-adjacent periods into one
    minute series and a forward window would silently jump the gap.
    """
    first_year, last_year = int(START_MONTH[:4]), int(END_MONTH[:4])
    months = [
        f"{year:04d}-{month:02d}"
        for year in range(first_year, last_year + 1)
        for month in range(1, 13)
        if START_MONTH <= f"{year:04d}-{month:02d}" <= END_MONTH
    ]
    order = {month: index for index, month in enumerate(months)}
    excluded = set(EXCLUDED_SYMBOL_MONTHS)
    report = load_published_archive_availability(availability_directory())
    have: dict[str, set[str]] = {}
    for probe in report.probes:
        if probe.state is not AvailabilityState.AVAILABLE:
            continue
        # A declared exclusion is audited AVAILABLE - the venue does serve the archive, and raising here
        # is purely a decode-side contract failure - so it must not count towards contiguity either. Left
        # in, the month looks present in this receipt and is absent from the corpus, which is precisely
        # the silently jumped gap this function exists to prevent. Measured on 2026-09-29: with the
        # exclusion not subtracted here, `REDUSDT` reported `{2025-02, 2025-03, 2025-04}`, read as
        # contiguous, and was admitted with a month that can never be decoded.
        if (probe.symbol, probe.month) in excluded:
            continue
        have.setdefault(probe.symbol, set()).add(probe.month)

    contiguous: list[tuple[str, int, str, str]] = []
    dropped: list[dict[str, Any]] = []
    gapped: list[str] = []
    for symbol, available in sorted(have.items()):
        index = sorted(order[month] for month in available)
        if index == list(range(index[0], index[-1] + 1)):
            contiguous.append((symbol, len(index), months[index[0]], months[index[-1]]))
            continue
        # An interior gap disqualifies the symbol: a missing month would concatenate two non-adjacent
        # periods into one minute series and a forward window would silently jump the gap. A gap made
        # entirely of DECLARED exclusions is explained, so the symbol is dropped and recorded; any
        # other gap is unexplained and stays a hard error.
        span = months[index[0] : index[-1] + 1]
        absent = [month for month in span if order[month] not in index]
        unexplained = [month for month in absent if (symbol, month) not in excluded]
        if unexplained:
            gapped.append(symbol)
        else:
            dropped.append(
                {
                    "symbol": symbol,
                    "reason": "interior month gap entirely accounted for by a declared exclusion",
                    "absent_months": absent,
                }
            )
    if gapped:
        raise ValueError(
            f"symbols with an interior month gap cannot be admitted: {gapped}"
        )

    complete = [symbol for symbol, count, _, _ in contiguous if count == len(months)]
    all_new = [symbol for symbol, _, _, _ in contiguous]
    merged = sorted(set(all_new) | set(CURRENT_COHORT))
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "symbols.json").write_text(
        json.dumps(
            {
                "study": STUDY.name,
                "derived_from_availability": str(
                    AVAILABILITY.relative_to(ROOT)
                ).replace("\\", "/"),
                "window": {"start_month": START_MONTH, "end_month": END_MONTH},
                "probed_months": len(months),
                "available_symbol_months": sum(count for _, count, _, _ in contiguous),
                "symbols_new_all": all_new,
                "symbols_new_complete": complete,
                "symbols_current_cohort": list(CURRENT_COHORT),
                "symbols_merged": merged,
                "dropped_symbols": dropped,
                "per_symbol": [
                    {
                        "symbol": symbol,
                        "months": count,
                        "first_month": first,
                        "last_month": last,
                    }
                    for symbol, count, first, last in contiguous
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, listing in (
        ("symbols_new_all.txt", all_new),
        ("symbols_new_complete.txt", complete),
        ("symbols_merged.txt", merged),
    ):
        (STUDY / name).write_text("\n".join(listing) + "\n", encoding="utf-8")
    print(
        f"available symbol-months {sum(count for _, count, _, _ in contiguous)}; "
        f"new symbols {len(all_new)} (complete {len(complete)}); merged {len(merged)}; "
        f"dropped {len(dropped)}"
    )
    for entry in dropped:
        print(f"  dropped {entry['symbol']}: absent {entry['absent_months']}")


STAGES = {
    "select": select,
    "probe": probe,
    "symbols": symbols,
    "download": download,
    "corpus": corpus,
}


def parse_arguments() -> str:
    """Split `--study`, `--start-month` and `--end-month` out of the argument vector.

    The stage name stays the first positional argument, so every documented invocation keeps
    working; the remaining positionals are published as `EXTRA`.
    """
    global EXTRA
    tokens = sys.argv[1:]
    options: dict[str, str] = {}
    positional: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in (
            "--study",
            "--start-month",
            "--end-month",
            "--pool-study",
            "--symbols-file",
        ):
            if index + 1 >= len(tokens):
                raise SystemExit(f"{token} needs a value")
            options[token] = tokens[index + 1]
            index += 2
            continue
        positional.append(token)
        index += 1
    if not positional or positional[0] not in STAGES:
        raise SystemExit(
            f"usage: {Path(__file__).name} {{{'|'.join(STAGES)}}} [symbol ...] "
            "[--study PATH] [--pool-study PATH] [--symbols-file PATH] "
            "[--start-month YYYY-MM] [--end-month YYYY-MM]"
        )
    configure(
        options.get("--study"),
        options.get("--start-month"),
        options.get("--end-month"),
        options.get("--pool-study"),
    )
    if "--symbols-file" in options:
        listing = Path(options["--symbols-file"])
        if not listing.is_absolute():
            listing = ROOT / listing
        if not listing.is_file():
            raise SystemExit(f"--symbols-file not found: {listing}")
        EXTRA = [
            line.strip()
            for line in listing.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        EXTRA = positional[1:]
    return positional[0]


def main() -> None:
    STAGES[parse_arguments()]()


if __name__ == "__main__":
    main()
