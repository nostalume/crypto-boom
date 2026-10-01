"""Research-only Polars challenger for the frozen BEN-01 label semantics."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from crypto_boom.evaluation.benchmark import (
    BarrierDefinition,
    BenchmarkError,
    BenchmarkResult,
    BenchmarkSpec,
    EpisodeLabel,
    LabelExit,
    LabelOutcome,
    PrevalenceRow,
    _assemble_benchmark_result,
    _cost_factors,
    _executable_return_from_factors,
    _prevalence_row_from_counts,
)
from crypto_boom.market import Environment, InstrumentId, VenueId
from crypto_boom.storage.source import (
    PublishedResearchPartition,
    ResearchStorageError,
    load_published_research_partition,
)

if TYPE_CHECKING:
    import polars as pl

_MINUTE_US = 60_000_000
_MAX_PARTITION_CANDIDATE_ROWS = 50_000
_MAX_SOURCE_PARTITION_ROWS = 50_000
_MAX_RESEARCH_INSTRUMENTS = 500
_IDENTITY_COLUMNS = ("venue", "market", "environment", "symbol")
_KLINE_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "interval",
    "open_time",
    "close_time",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "quality_state",
    "quality_complete",
)
_ELIGIBILITY_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "effective_time_us",
    "eligible",
    "evidence_id",
    "reason",
)
_LABEL_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "interval",
    "spec_id",
    "numerical_policy_id",
    "barrier_name",
    "candidate_time_us",
    "horizon_end_us",
    "outcome",
    "exit",
    "exit_time_us",
    "entry_time_us",
    "volatility_end_time_us",
    "sigma",
    "executable_return",
    "maximum_favorable_return",
    "maximum_adverse_return",
    "eligibility_effective_time_us",
    "eligibility_evidence_id",
)
_COVERAGE_LABEL_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "interval",
    "spec_id",
    "numerical_policy_id",
    "barrier_name",
    "candidate_time_us",
    "horizon_end_us",
    "outcome",
    "exit",
    "exit_time_us",
    "sample_scope",
)
_CANDIDATE_KEY_COLUMNS = (*_IDENTITY_COLUMNS, "interval", "candidate_time_us")
_COVERAGE_SCOPE = "coverage_conditioned"
_SOURCE_COVERAGE_CLASS = "coverage_conditioned_source_only"


class ColumnarDependencyError(BenchmarkError):
    """The optional research execution dependency is unavailable."""


class ColumnarNumericalPolicy(StrEnum):
    """Versioned numerical execution policy for columnar BEN labels."""

    EXACT_DECIMAL_V1 = "exact-decimal-v1"
    FLOAT64_ULP16_V1 = "float64-ulp16-unavailable-v1"


@dataclass(frozen=True, slots=True)
class ColumnarResearchLimits:
    """Fail-closed envelope; defaults match the largest measured fresh process."""

    maximum_instruments: int = 4
    maximum_source_partitions: int = 8
    maximum_rows: int = 400_000
    maximum_parquet_bytes: int = 64 * 1_024 * 1_024
    maximum_elapsed_seconds: float = 30.0

    def __post_init__(self) -> None:
        integer_limits = (
            self.maximum_instruments,
            self.maximum_source_partitions,
            self.maximum_rows,
            self.maximum_parquet_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integer_limits) or (
            type(self.maximum_elapsed_seconds) not in {int, float}
            or self.maximum_elapsed_seconds <= 0
        ):
            raise ValueError("columnar research limits must be positive")
        if self.maximum_instruments > _MAX_RESEARCH_INSTRUMENTS:
            raise ValueError("columnar research instrument limit exceeds 500")


DEFAULT_COLUMNAR_RESEARCH_LIMITS = ColumnarResearchLimits()


@dataclass(frozen=True, slots=True)
class _AdmittedInstrument:
    """One normalized instrument whose external invariants were checked once."""

    bars: pl.DataFrame
    history: tuple[tuple[int, bool, str], ...]
    open_times: list[int] | None
    numerical_policy: ColumnarNumericalPolicy
    eligibility_required: bool = True


@dataclass(frozen=True, slots=True)
class _ExactInstrumentContext:
    """Admitted arrays reused by every exact barrier for one instrument."""

    bars: pl.DataFrame
    spec_id: str
    interval_us: int
    cost_factors: tuple[Decimal, Decimal]
    candidate_times: list[int]
    open_times: list[int]
    entry_prices: list[Decimal | None]
    closes: list[Decimal]
    highs: pl.Series
    lows: pl.Series
    sigmas: list[Decimal | None]
    eligibility_times: list[int | None]
    eligibility_states: list[bool | None]
    eligibility_ids: list[str | None]
    high_blocks: list[pl.Series]
    low_blocks: list[pl.Series]
    eligibility_required: bool


@dataclass(frozen=True, slots=True)
class _FloatInstrumentContext:
    """Native arrays shared by every float64 barrier for one instrument."""

    numeric: pl.DataFrame
    spec_id: str
    interval_us: int
    count: int
    close_times: pl.Series
    indices: pl.Series
    highs: pl.Series
    lows: pl.Series
    high_blocks: list[pl.Series]
    low_blocks: list[pl.Series]
    entry_factor: float
    exit_factor: float
    eligibility_required: bool


@dataclass(frozen=True, slots=True)
class ColumnarExitCount:
    """One bounded aggregate cell from partitioned columnar execution."""

    barrier_name: str
    outcome: LabelOutcome
    exit: LabelExit
    count: int


@dataclass(frozen=True, slots=True)
class ColumnarResearchSummary:
    """Non-materializing BEN counts; not an authoritative benchmark report."""

    spec_id: str
    numerical_policy_id: str
    instrument_count: int
    partition_count: int
    label_count: int
    exit_counts: tuple[ColumnarExitCount, ...]
    prevalence: tuple[PrevalenceRow, ...]
    source_manifest_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CoverageConditionedLabels:
    """Price-path labels; cohort identity is supplied, not source-verified here."""

    cohort_id: str
    spec_id: str
    numerical_policy_id: str
    labels: pl.DataFrame
    evidence_class: str = field(default=_COVERAGE_SCOPE, init=False)

    def __post_init__(self) -> None:
        if not self.cohort_id:
            raise BenchmarkError("coverage-conditioned cohort identity is required")


@dataclass(frozen=True, slots=True)
class CoverageConditionedSummary:
    """Verified-source event counts, not eligibility or BEN prevalence evidence."""

    spec_id: str
    numerical_policy_id: str
    instrument_count: int
    partition_count: int
    label_count: int
    exit_counts: tuple[ColumnarExitCount, ...]
    source_manifest_ids: tuple[str, ...]
    evidence_class: str = field(default=_SOURCE_COVERAGE_CLASS, init=False)


def build_columnar_labels(
    klines: pl.DataFrame | pl.LazyFrame,
    eligibility: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
    *,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.EXACT_DECIMAL_V1
    ),
) -> pl.DataFrame:
    """Build versioned BEN labels for one or more instruments with Polars.

    The input column names match the source-only research Parquet schema. Times may
    be microsecond integers or Polars ``Datetime`` values. The default exact policy
    preserves BEN-v1 parity. The explicit float64 policy keeps the scalar benchmark
    as its oracle and refuses decisions inside its versioned ambiguity band.
    """

    _require_numerical_policy(numerical_policy)
    _require_polars()

    instruments = _admit_instruments(klines, eligibility, spec, numerical_policy)
    return _build_label_frame(instruments, spec, numerical_policy)


def build_coverage_conditioned_labels(
    klines: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
    *,
    cohort_id: str,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    ),
) -> CoverageConditionedLabels:
    """Label observed price paths without asserting historical eligibility.

    Outcomes retain the BEN cost/barrier assumptions, but output omits modeled
    execution returns and eligibility fields. It cannot become a BEN report.
    """

    if not cohort_id:
        raise BenchmarkError("coverage-conditioned cohort identity is required")
    _require_numerical_policy(numerical_policy)
    _require_polars()
    import polars as pl

    instruments = _admit_instruments(
        klines,
        None,
        spec,
        numerical_policy,
        eligibility_required=False,
    )
    labels = _build_label_frame(instruments, spec, numerical_policy).with_columns(
        pl.lit(_COVERAGE_SCOPE).alias("sample_scope")
    )
    return CoverageConditionedLabels(
        cohort_id=cohort_id,
        spec_id=spec.spec_id,
        numerical_policy_id=numerical_policy.value,
        labels=labels.select(_COVERAGE_LABEL_COLUMNS),
    )


def build_coverage_conditioned_candidate_labels(
    klines: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
    *,
    cohort_id: str,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    ),
) -> CoverageConditionedLabels:
    """Evaluate admitted candidate keys against the full observed price path.

    Volatility and future-path inputs retain their full causal context, while
    barrier decisions and returned rows are restricted to selected candidates.
    """

    if not cohort_id:
        raise BenchmarkError("coverage-conditioned cohort identity is required")
    _require_numerical_policy(numerical_policy)
    _require_polars()
    import polars as pl

    candidate_keys = _normalize_candidate_keys(candidates)
    instruments = _admit_instruments(
        klines,
        None,
        spec,
        numerical_policy,
        eligibility_required=False,
    )
    selected_frames: list[pl.DataFrame] = []
    for instrument in instruments:
        identity = instrument.bars.select(*_IDENTITY_COLUMNS, "interval").row(0)
        selected_keys = candidate_keys.filter(
            pl.all_horizontal(
                *(
                    pl.col(column) == value
                    for column, value in zip(
                        (*_IDENTITY_COLUMNS, "interval"), identity, strict=True
                    )
                )
            )
        )
        if selected_keys.is_empty():
            continue
        if numerical_policy is ColumnarNumericalPolicy.EXACT_DECIMAL_V1:
            selected_frames.append(
                _label_instrument(instrument, spec, numerical_policy).join(
                    selected_keys,
                    on=list(_CANDIDATE_KEY_COLUMNS),
                    how="semi",
                )
            )
            continue
        indices = instrument.bars.select(
            pl.int_range(pl.len(), dtype=pl.Int64).alias("_row_index"),
            pl.col("close_time_us").alias("candidate_time_us"),
        ).join(
            selected_keys.select("candidate_time_us"),
            on="candidate_time_us",
            how="semi",
        )["_row_index"]
        if indices.is_empty():
            raise BenchmarkError(
                "coverage-conditioned candidates must match one cohort row each"
            )
        selected_frames.append(
            _label_instrument_float64(
                instrument,
                spec,
                candidate_indices=indices,
                coverage_only=True,
            )
        )

    if not selected_frames:
        raise BenchmarkError("coverage-conditioned candidates do not match the cohort")
    labels = pl.concat(selected_frames, how="diagonal_relaxed")
    expected = len(candidate_keys) * len(spec.barriers)
    if len(labels) != expected:
        raise BenchmarkError(
            "coverage-conditioned candidates must match one cohort row each"
        )
    labels = (
        labels.sort([*_IDENTITY_COLUMNS, "candidate_time_us", "_barrier_index"])
        .drop("_barrier_index")
        .with_columns(pl.lit(_COVERAGE_SCOPE).alias("sample_scope"))
        .select(_COVERAGE_LABEL_COLUMNS)
    )
    return CoverageConditionedLabels(
        cohort_id=cohort_id,
        spec_id=spec.spec_id,
        numerical_policy_id=numerical_policy.value,
        labels=labels,
    )


def _build_label_frame(
    instruments: tuple[_AdmittedInstrument, ...],
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
) -> pl.DataFrame:
    import polars as pl

    labelled = [
        _label_instrument(instrument, spec, numerical_policy)
        for instrument in instruments
    ]
    result = pl.concat(labelled, how="diagonal_relaxed")
    return (
        result.sort([*_IDENTITY_COLUMNS, "candidate_time_us", "_barrier_index"])
        .drop("_barrier_index")
        .select(_LABEL_COLUMNS)
    )


def summarize_columnar_labels(
    klines: pl.DataFrame | pl.LazyFrame,
    eligibility: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
    *,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    ),
    partition_candidate_rows: int = 10_000,
) -> ColumnarResearchSummary:
    """Aggregate BEN label counts while retaining only one halo partition."""

    _require_numerical_policy(numerical_policy)
    _require_partition_candidate_rows(partition_candidate_rows)
    _require_polars()
    instruments = _admit_instruments(klines, eligibility, spec, numerical_policy)
    counts: dict[tuple[str, LabelOutcome, LabelExit], int] = {}
    partition_count = 0
    label_count = 0

    for instrument in instruments:
        added_partitions, added_labels = _summarize_instrument(
            instrument,
            spec,
            numerical_policy,
            partition_candidate_rows,
            counts,
        )
        partition_count += added_partitions
        label_count += added_labels

    return _research_summary(
        spec,
        numerical_policy,
        instrument_count=len(instruments),
        partition_count=partition_count,
        label_count=label_count,
        counts=counts,
    )


def summarize_research_partitions(
    paths: Sequence[Path],
    eligibility: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
    *,
    limits: ColumnarResearchLimits = DEFAULT_COLUMNAR_RESEARCH_LIMITS,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    ),
    partition_candidate_rows: int = 10_000,
) -> ColumnarResearchSummary:
    """Verify and summarize monthly source partitions without collecting the corpus."""

    return _summarize_source_partitions(
        paths,
        eligibility,
        spec,
        limits=limits,
        numerical_policy=numerical_policy,
        partition_candidate_rows=partition_candidate_rows,
        eligibility_required=True,
    )


def summarize_coverage_conditioned_partitions(
    paths: Sequence[Path],
    spec: BenchmarkSpec,
    *,
    limits: ColumnarResearchLimits = DEFAULT_COLUMNAR_RESEARCH_LIMITS,
    numerical_policy: ColumnarNumericalPolicy = (
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    ),
    partition_candidate_rows: int = 10_000,
) -> CoverageConditionedSummary:
    """Count bounded source-backed event labels without eligibility assertions."""

    summary = _summarize_source_partitions(
        paths,
        None,
        spec,
        limits=limits,
        numerical_policy=numerical_policy,
        partition_candidate_rows=partition_candidate_rows,
        eligibility_required=False,
    )
    return CoverageConditionedSummary(
        spec_id=summary.spec_id,
        numerical_policy_id=summary.numerical_policy_id,
        instrument_count=summary.instrument_count,
        partition_count=summary.partition_count,
        label_count=summary.label_count,
        exit_counts=summary.exit_counts,
        source_manifest_ids=summary.source_manifest_ids,
    )


def _summarize_source_partitions(
    paths: Sequence[Path],
    eligibility: pl.DataFrame | pl.LazyFrame | None,
    spec: BenchmarkSpec,
    *,
    limits: ColumnarResearchLimits,
    numerical_policy: ColumnarNumericalPolicy,
    partition_candidate_rows: int,
    eligibility_required: bool,
) -> ColumnarResearchSummary:
    _require_numerical_policy(numerical_policy)
    _require_partition_candidate_rows(partition_candidate_rows)
    if not isinstance(limits, ColumnarResearchLimits):
        raise BenchmarkError("columnar research limits are not supported")
    if spec.interval_us != _MINUTE_US:
        raise BenchmarkError("research source partitions require a one-minute spec")

    _require_polars()
    started = monotonic()
    publications = _admit_source_publications(paths, spec, limits, started)
    keys = {_manifest_key(publication) for publication in publications}
    histories = (
        _eligibility_histories(_normalize_eligibility(eligibility), keys)
        if eligibility_required
        else {}
    )
    grouped: dict[tuple[object, ...], list[PublishedResearchPartition]] = {}
    for publication in publications:
        grouped.setdefault(_manifest_key(publication), []).append(publication)

    counts: dict[tuple[str, LabelOutcome, LabelExit], int] = {}
    partition_count = 0
    label_count = 0
    left_halo, right_halo = _halo_rows(spec)
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: item.manifest.first_open_time_us)
        _require_contiguous_publications(group, spec)
        previous_tail = None
        current = _load_source_bars(group[0], spec, numerical_policy)
        for index, _publication in enumerate(group):
            if current is None:
                raise AssertionError("source partition lookahead was exhausted early")
            _check_research_elapsed(started, limits)
            following = (
                _load_source_bars(group[index + 1], spec, numerical_policy)
                if index + 1 < len(group)
                else None
            )
            if (
                following is not None
                and numerical_policy is ColumnarNumericalPolicy.FLOAT64_ULP16_V1
            ):
                _require_float64_boundary_resolution(current, following)
            next_head = following.head(right_halo) if following is not None else None
            instrument = _source_window_instrument(
                current,
                previous_tail,
                next_head,
                histories.get(key, ()),
                numerical_policy,
                eligibility_required=eligibility_required,
            )
            core_offset = 0 if previous_tail is None else len(previous_tail)
            added_partitions, added_labels = _summarize_instrument(
                instrument,
                spec,
                numerical_policy,
                partition_candidate_rows,
                counts,
                core_start=core_offset,
                core_end=core_offset + len(current),
            )
            partition_count += added_partitions
            label_count += added_labels
            previous_tail = _owned_tail(current, left_halo)
            current = following
            _check_research_elapsed(started, limits)

    return _research_summary(
        spec,
        numerical_policy,
        instrument_count=len(grouped),
        partition_count=partition_count,
        label_count=label_count,
        counts=counts,
        source_manifest_ids=tuple(
            publication.manifest.manifest_id for publication in publications
        ),
    )


def _summarize_instrument(
    instrument: _AdmittedInstrument,
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
    partition_candidate_rows: int,
    counts: dict[tuple[str, LabelOutcome, LabelExit], int],
    *,
    core_start: int = 0,
    core_end: int | None = None,
) -> tuple[int, int]:
    import polars as pl

    bars = instrument.bars
    core_end = len(bars) if core_end is None else core_end
    left_halo, right_halo = _halo_rows(spec)
    partition_count = 0
    label_count = 0
    for start in range(core_start, core_end, partition_candidate_rows):
        end = min(start + partition_candidate_rows, core_end)
        window_start = max(0, start - left_halo)
        window_end = min(len(bars), end + right_halo)
        open_times = (
            None
            if instrument.open_times is None
            else instrument.open_times[window_start:window_end]
        )
        window = _AdmittedInstrument(
            bars=bars.slice(window_start, window_end - window_start),
            history=instrument.history,
            open_times=open_times,
            numerical_policy=instrument.numerical_policy,
            eligibility_required=instrument.eligibility_required,
        )
        labels = (
            _label_instrument_float64(window, spec, summary_only=True)
            if numerical_policy is ColumnarNumericalPolicy.FLOAT64_ULP16_V1
            else _label_instrument(window, spec, numerical_policy)
        )
        first_time = bars["close_time_us"][start]
        last_time = bars["close_time_us"][end - 1]
        core = labels.filter(
            (pl.col("candidate_time_us") >= first_time)
            & (pl.col("candidate_time_us") <= last_time)
        )
        expected = (end - start) * len(spec.barriers)
        if len(core) != expected:
            raise AssertionError("partitioned BEN labels do not cover the core")
        for barrier_name, outcome, exit_reason, count in (
            core.group_by("barrier_name", "outcome", "exit").len().iter_rows()
        ):
            key = (
                barrier_name,
                LabelOutcome(outcome),
                LabelExit(exit_reason),
            )
            counts[key] = counts.get(key, 0) + count
        partition_count += 1
        label_count += len(core)
    return partition_count, label_count


def _research_summary(
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
    *,
    instrument_count: int,
    partition_count: int,
    label_count: int,
    counts: dict[tuple[str, LabelOutcome, LabelExit], int],
    source_manifest_ids: tuple[str, ...] = (),
) -> ColumnarResearchSummary:
    barrier_order = {barrier.name: index for index, barrier in enumerate(spec.barriers)}
    exit_counts = tuple(
        ColumnarExitCount(barrier_name, outcome, exit_reason, count)
        for (barrier_name, outcome, exit_reason), count in sorted(
            counts.items(),
            key=lambda item: (
                barrier_order[item[0][0]],
                item[0][1].value,
                item[0][2].value,
            ),
        )
    )
    return ColumnarResearchSummary(
        spec_id=spec.spec_id,
        numerical_policy_id=numerical_policy.value,
        instrument_count=instrument_count,
        partition_count=partition_count,
        label_count=label_count,
        exit_counts=exit_counts,
        prevalence=tuple(
            _prevalence_row(barrier.name, exit_counts) for barrier in spec.barriers
        ),
        source_manifest_ids=source_manifest_ids,
    )


def _require_partition_candidate_rows(value: object) -> None:
    if type(value) is not int or value < 1:
        raise BenchmarkError("columnar partition candidate rows must be positive")
    if value > _MAX_PARTITION_CANDIDATE_ROWS:
        raise BenchmarkError("columnar partition candidate rows exceed 50000")


def _halo_rows(spec: BenchmarkSpec) -> tuple[int, int]:
    return (
        spec.volatility.window_intervals,
        max(
            barrier.horizon_minutes * _MINUTE_US // spec.interval_us
            for barrier in spec.barriers
        ),
    )


def _prevalence_row(
    barrier_name: str,
    counts: tuple[ColumnarExitCount, ...],
) -> PrevalenceRow:
    selected = tuple(row for row in counts if row.barrier_name == barrier_name)
    positive = sum(
        row.count for row in selected if row.outcome is LabelOutcome.POSITIVE
    )
    negative = sum(
        row.count for row in selected if row.outcome is LabelOutcome.NEGATIVE
    )
    total = sum(row.count for row in selected)
    return _prevalence_row_from_counts(
        barrier_name,
        total=total,
        positive=positive,
        negative=negative,
    )


def _require_numerical_policy(policy: object) -> None:
    if not isinstance(policy, ColumnarNumericalPolicy):
        raise BenchmarkError("columnar numerical policy is not supported")


def _admit_source_publications(
    paths: Sequence[Path],
    spec: BenchmarkSpec,
    limits: ColumnarResearchLimits,
    started: float,
) -> tuple[PublishedResearchPartition, ...]:
    requested = tuple(Path(path).resolve() for path in paths)
    if not requested:
        raise BenchmarkError("columnar research requires at least one source partition")
    if len(requested) > limits.maximum_source_partitions:
        raise BenchmarkError("columnar research source-partition limit exceeded")
    if len(set(requested)) != len(requested):
        raise BenchmarkError("columnar research source partitions must be unique")

    publications = []
    manifest_ids: set[str] = set()
    instrument_keys: set[tuple[object, ...]] = set()
    total_rows = 0
    total_bytes = 0
    for path in requested:
        _check_research_elapsed(started, limits)
        try:
            publication = load_published_research_partition(path)
        except ResearchStorageError as error:
            raise BenchmarkError(
                f"research source partition is not verified: {path}"
            ) from error
        manifest = publication.manifest
        if manifest.manifest_id in manifest_ids:
            raise BenchmarkError("columnar research source manifests must be unique")
        if manifest.interval != "1m" or manifest.row_count > _MAX_SOURCE_PARTITION_ROWS:
            raise BenchmarkError(
                "research source partition is outside its row contract"
            )
        manifest_ids.add(manifest.manifest_id)
        instrument_keys.add(_manifest_key(publication))
        total_rows += manifest.row_count
        total_bytes += manifest.parquet_bytes
        if len(instrument_keys) > limits.maximum_instruments:
            raise BenchmarkError("columnar research instrument limit exceeded")
        if total_rows > limits.maximum_rows:
            raise BenchmarkError("columnar research row limit exceeded")
        if total_bytes > limits.maximum_parquet_bytes:
            raise BenchmarkError("columnar research Parquet byte limit exceeded")
        publications.append(publication)
        _check_research_elapsed(started, limits)
    return tuple(
        sorted(
            publications,
            key=lambda item: (
                *_manifest_key(item),
                item.manifest.first_open_time_us,
            ),
        )
    )


def _manifest_key(publication: PublishedResearchPartition) -> tuple[object, ...]:
    manifest = publication.manifest
    return (
        manifest.venue,
        manifest.market,
        manifest.environment,
        manifest.symbol,
    )


def _require_contiguous_publications(
    publications: Sequence[PublishedResearchPartition],
    spec: BenchmarkSpec,
) -> None:
    for previous, current in pairwise(publications):
        if (
            previous.manifest.last_open_time_us + spec.interval_us
            != current.manifest.first_open_time_us
        ):
            raise BenchmarkError(
                "research source partitions are not a complete interval grid"
            )


def _load_source_bars(
    publication: PublishedResearchPartition,
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
) -> pl.DataFrame:
    import polars as pl

    try:
        raw = pl.read_parquet(
            publication.path / "klines.parquet",
            columns=list(_KLINE_COLUMNS),
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise BenchmarkError("verified research source could not be read") from error
    bars = _normalize_klines(raw)
    manifest = publication.manifest
    identities = bars.select(_IDENTITY_COLUMNS).unique()
    if (
        len(bars) != manifest.row_count
        or len(identities) != 1
        or _key(bars) != _manifest_key(publication)
        or bars["interval"].unique().to_list() != [manifest.interval]
        or bars["open_time_us"][0] != manifest.first_open_time_us
        or bars["open_time_us"][-1] != manifest.last_open_time_us
    ):
        raise BenchmarkError("research source values conflict with their manifest")
    _require_complete_grid(bars, spec)
    return _prepare_numerical_bars(bars, numerical_policy)


def _source_window_instrument(
    current: pl.DataFrame,
    previous_tail: pl.DataFrame | None,
    next_head: pl.DataFrame | None,
    history: tuple[tuple[int, bool, str], ...],
    numerical_policy: ColumnarNumericalPolicy,
    *,
    eligibility_required: bool = True,
) -> _AdmittedInstrument:
    import polars as pl

    frames = [
        frame for frame in (previous_tail, current, next_head) if frame is not None
    ]
    bars = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]
    return _AdmittedInstrument(
        bars=bars,
        history=history,
        open_times=(
            bars["open_time_us"].to_list()
            if numerical_policy is ColumnarNumericalPolicy.EXACT_DECIMAL_V1
            else None
        ),
        numerical_policy=numerical_policy,
        eligibility_required=eligibility_required,
    )


def _owned_tail(frame: pl.DataFrame, rows: int) -> pl.DataFrame:
    """Copy a bounded halo so an Arrow slice cannot pin the prior full partition."""

    import polars as pl

    start = max(0, len(frame) - rows)
    return frame.gather(pl.int_range(start, len(frame), eager=True))


def _check_research_elapsed(started: float, limits: ColumnarResearchLimits) -> None:
    if monotonic() - started > limits.maximum_elapsed_seconds:
        raise BenchmarkError("columnar research elapsed-time limit exceeded")


def _admit_instruments(
    klines: object,
    eligibility: object | None,
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
    *,
    eligibility_required: bool = True,
) -> tuple[_AdmittedInstrument, ...]:
    bars = _prepare_numerical_bars(_normalize_klines(klines), numerical_policy)
    histories: dict[tuple[object, ...], tuple[tuple[int, bool, str], ...]] = {}
    if eligibility_required:
        snapshots = _normalize_eligibility(eligibility)
        instrument_keys = {
            tuple(row) for row in bars.select(_IDENTITY_COLUMNS).unique().iter_rows()
        }
        histories = _eligibility_histories(snapshots, instrument_keys)
    elif eligibility is not None:
        raise BenchmarkError("coverage-conditioned labels reject eligibility input")
    admitted = []
    for group in bars.partition_by(list(_IDENTITY_COLUMNS), maintain_order=True):
        if group["interval"].n_unique() != 1:
            raise BenchmarkError("benchmark input must contain one instrument/interval")
        _require_complete_grid(group, spec)
        admitted.append(
            _AdmittedInstrument(
                bars=group,
                history=histories.get(_key(group), ()),
                open_times=(
                    group["open_time_us"].to_list()
                    if numerical_policy is ColumnarNumericalPolicy.EXACT_DECIMAL_V1
                    else None
                ),
                numerical_policy=numerical_policy,
                eligibility_required=eligibility_required,
            )
        )
    return tuple(admitted)


def _prepare_numerical_bars(
    bars: pl.DataFrame,
    numerical_policy: ColumnarNumericalPolicy,
) -> pl.DataFrame:
    import polars as pl

    if numerical_policy is ColumnarNumericalPolicy.EXACT_DECIMAL_V1:
        return bars
    price_columns = ("open_price", "high_price", "low_price", "close_price")
    numeric = bars.with_columns(
        pl.col("close_price").alias("_exact_close_price"),
        *(pl.col(name).cast(pl.Float64).alias(name) for name in price_columns),
    )
    invalid = numeric.select(
        *(
            (~pl.col(name).is_finite() | (pl.col(name) <= 0.0)).any().alias(name)
            for name in price_columns
        )
    )
    if any(invalid.row(0)):
        raise BenchmarkError("float64 benchmark prices must be positive and finite")
    collapsed_return = numeric.select(
        (
            (pl.col("_exact_close_price") != pl.col("_exact_close_price").shift(1))
            & (pl.col("close_price") == pl.col("close_price").shift(1))
        )
        .fill_null(False)
        .any()
        .alias("collapsed")
    )["collapsed"][0]
    if collapsed_return:
        raise BenchmarkError(
            "nonzero close return is below admitted binary64 resolution"
        )
    return numeric


def _require_float64_boundary_resolution(
    previous: pl.DataFrame, current: pl.DataFrame
) -> None:
    if (
        previous["_exact_close_price"][-1] != current["_exact_close_price"][0]
        and previous["close_price"][-1] == current["close_price"][0]
    ):
        raise BenchmarkError(
            "nonzero close return is below admitted binary64 resolution"
        )


def _require_complete_grid(bars: pl.DataFrame, spec: BenchmarkSpec) -> None:
    import polars as pl

    if (
        len(bars) > 1
        and bars.select(
            (pl.col("open_time_us").diff().drop_nulls() != spec.interval_us)
            .any()
            .alias("broken")
        )["broken"][0]
    ):
        raise BenchmarkError("benchmark input is not a complete interval grid")


def _to_episode_labels(labels: pl.DataFrame | pl.LazyFrame) -> tuple[EpisodeLabel, ...]:
    """Materialize a bounded columnar result as scalar labels for parity checks."""

    _require_polars()
    frame = _collect(labels, "columnar labels")
    if "sample_scope" in frame.columns:
        raise BenchmarkError("coverage-conditioned labels cannot become BEN reports")
    missing = set(_LABEL_COLUMNS) - set(frame.columns)
    if missing:
        raise BenchmarkError(
            "columnar labels are missing required columns: "
            + ", ".join(sorted(missing))
        )
    if frame["numerical_policy_id"].unique().to_list() != [
        ColumnarNumericalPolicy.EXACT_DECIMAL_V1.value
    ]:
        raise BenchmarkError("scalar parity conversion requires the exact policy")
    return tuple(
        EpisodeLabel(
            spec_id=row["spec_id"],
            barrier_name=row["barrier_name"],
            instrument=InstrumentId(
                venue=VenueId(row["venue"], row["market"]),
                environment=Environment(row["environment"]),
                symbol=row["symbol"],
            ),
            candidate_time_us=row["candidate_time_us"],
            horizon_end_us=row["horizon_end_us"],
            outcome=LabelOutcome(row["outcome"]),
            exit=LabelExit(row["exit"]),
            exit_time_us=row["exit_time_us"],
            entry_time_us=row["entry_time_us"],
            volatility_end_time_us=row["volatility_end_time_us"],
            sigma=row["sigma"],
            executable_return=row["executable_return"],
            maximum_favorable_return=row["maximum_favorable_return"],
            maximum_adverse_return=row["maximum_adverse_return"],
            eligibility_effective_time_us=row["eligibility_effective_time_us"],
            eligibility_evidence_id=row["eligibility_evidence_id"],
        )
        for row in frame.select(_LABEL_COLUMNS).iter_rows(named=True)
    )


def _to_benchmark_results(
    labels: pl.DataFrame | pl.LazyFrame,
    spec: BenchmarkSpec,
) -> tuple[BenchmarkResult, ...]:
    """Assemble bounded scalar results from columnar labels for exact parity."""

    materialized = _to_episode_labels(labels)
    grouped: dict[InstrumentId, list[EpisodeLabel]] = {}
    for label in materialized:
        if label.spec_id != spec.spec_id:
            raise BenchmarkError("columnar label specification does not match")
        grouped.setdefault(label.instrument, []).append(label)

    results = []
    expected_barriers = {barrier.name for barrier in spec.barriers}
    for instrument_labels in grouped.values():
        candidate_times = tuple(
            sorted({label.candidate_time_us for label in instrument_labels})
        )
        pairs = {
            (label.candidate_time_us, label.barrier_name) for label in instrument_labels
        }
        if len(instrument_labels) != len(candidate_times) * len(spec.barriers) or len(
            pairs
        ) != len(instrument_labels):
            raise BenchmarkError("columnar labels do not cover every candidate barrier")
        if {label.barrier_name for label in instrument_labels} != expected_barriers:
            raise BenchmarkError("columnar labels do not match benchmark barriers")
        results.append(
            _assemble_benchmark_result(
                spec,
                tuple(instrument_labels),
                candidate_times,
            )
        )
    return tuple(results)


def _require_polars() -> None:
    try:
        import_module("polars")
    except ImportError as error:
        raise ColumnarDependencyError(
            "columnar BEN-01 execution requires the research dependency group"
        ) from error


def _collect(value: object, name: str) -> pl.DataFrame:
    import polars as pl

    try:
        if isinstance(value, pl.LazyFrame):
            return value.collect()
        if isinstance(value, pl.DataFrame):
            return value.clone()
    except pl.exceptions.PolarsError as error:
        raise BenchmarkError(f"{name} could not be collected") from error
    raise BenchmarkError(f"{name} must be a Polars DataFrame or LazyFrame")


def _normalize_klines(value: object) -> pl.DataFrame:
    import polars as pl

    frame = _collect(value, "columnar klines")
    if frame.height == 0:
        raise BenchmarkError("benchmark requires at least one kline")
    _require_columns(frame, _KLINE_COLUMNS, "columnar klines")
    _refuse_nulls(frame, _KLINE_COLUMNS, "columnar klines")
    for column in ("open_price", "high_price", "low_price", "close_price"):
        if frame.schema[column] != pl.Decimal(38, 18):
            raise BenchmarkError(
                "columnar kline prices must use Decimal(38, 18) columns"
            )

    open_time = _time_expression(frame.schema["open_time"], "open_time")
    close_time = _time_expression(frame.schema["close_time"], "close_time")
    normalized = frame.select(
        *(
            pl.col(column)
            for column in _KLINE_COLUMNS
            if column not in {"open_time", "close_time"}
        ),
        open_time.alias("open_time_us"),
        close_time.alias("close_time_us"),
    ).sort([*_IDENTITY_COLUMNS, "open_time_us"])
    if normalized.filter(
        (pl.col("quality_state") != "valid") | ~pl.col("quality_complete")
    ).height:
        raise BenchmarkError("benchmark input contains non-final evidence")
    return normalized


def _normalize_candidate_keys(value: object) -> pl.DataFrame:
    import polars as pl

    frame = _collect(value, "coverage-conditioned candidates")
    if frame.is_empty():
        raise BenchmarkError("coverage-conditioned candidates must not be empty")
    _require_columns(frame, _CANDIDATE_KEY_COLUMNS, "coverage-conditioned candidates")
    _refuse_nulls(frame, _CANDIDATE_KEY_COLUMNS, "coverage-conditioned candidates")
    normalized = frame.select(
        *(pl.col(column) for column in _CANDIDATE_KEY_COLUMNS[:-1]),
        _time_expression(frame.schema["candidate_time_us"], "candidate_time_us").alias(
            "candidate_time_us"
        ),
    )
    if normalized.select(
        pl.struct(*_CANDIDATE_KEY_COLUMNS).is_duplicated().any()
    ).item():
        raise BenchmarkError("coverage-conditioned candidate keys must be unique")
    return normalized.sort(list(_CANDIDATE_KEY_COLUMNS))


def _normalize_eligibility(value: object) -> pl.DataFrame:
    import polars as pl

    frame = _collect(value, "columnar eligibility")
    _require_columns(frame, _ELIGIBILITY_COLUMNS, "columnar eligibility")
    _refuse_nulls(frame, _ELIGIBILITY_COLUMNS, "columnar eligibility")
    if not frame.schema["effective_time_us"].is_integer():
        raise BenchmarkError("eligibility times must be microsecond integers")
    if frame.schema["eligible"] != pl.Boolean:
        raise BenchmarkError("eligibility state must be boolean")
    return frame.with_columns(pl.col("effective_time_us").cast(pl.Int64)).sort(
        [*_IDENTITY_COLUMNS, "effective_time_us"]
    )


def _require_columns(frame: pl.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise BenchmarkError(
            f"{name} are missing required columns: " + ", ".join(sorted(missing))
        )


def _refuse_nulls(frame: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    null_columns = [column for column in columns if frame[column].null_count()]
    if null_columns:
        raise BenchmarkError(
            f"{name} contain null required values: " + ", ".join(null_columns)
        )


def _time_expression(dtype: pl.DataType, column: str) -> pl.Expr:
    import polars as pl

    if dtype.is_integer():
        return pl.col(column).cast(pl.Int64)
    if isinstance(dtype, pl.Datetime):
        return pl.col(column).dt.epoch("us")
    raise BenchmarkError(f"{column} must be a microsecond integer or Datetime")


def _eligibility_histories(
    frame: pl.DataFrame,
    instrument_keys: set[tuple[object, ...]],
) -> dict[tuple[object, ...], tuple[tuple[int, bool, str], ...]]:
    histories: dict[tuple[object, ...], list[tuple[int, bool, str]]] = {}
    for row in frame.iter_rows(named=True):
        key = tuple(row[column] for column in _IDENTITY_COLUMNS)
        if key not in instrument_keys:
            raise BenchmarkError("eligibility history contains another instrument")
        time_us = row["effective_time_us"]
        evidence_id = row["evidence_id"]
        reason = row["reason"]
        if time_us < 0 or not evidence_id or not reason:
            raise BenchmarkError("eligibility history contains an invalid snapshot")
        history = histories.setdefault(key, [])
        if history and history[-1][0] >= time_us:
            raise BenchmarkError("eligibility history is not strictly chronological")
        history.append((time_us, row["eligible"], evidence_id))
    return {key: tuple(history) for key, history in histories.items()}


def _key(frame: pl.DataFrame) -> tuple[object, ...]:
    return tuple(frame[column][0] for column in _IDENTITY_COLUMNS)


def _label_instrument(
    instrument: _AdmittedInstrument,
    spec: BenchmarkSpec,
    numerical_policy: ColumnarNumericalPolicy,
) -> pl.DataFrame:
    import polars as pl

    if instrument.numerical_policy is not numerical_policy:
        raise AssertionError("columnar numerical admission does not match execution")
    if numerical_policy is ColumnarNumericalPolicy.FLOAT64_ULP16_V1:
        return _label_instrument_float64(instrument, spec)

    bars = instrument.bars
    open_times = instrument.open_times
    if open_times is None:
        raise AssertionError("exact columnar execution requires admitted open times")
    close_times = bars["close_time_us"].to_list()
    opens = bars["open_price"].to_list()
    highs = bars["high_price"]
    lows = bars["low_price"]
    closes = bars["close_price"].to_list()
    sigmas = _causal_sigmas(closes, spec)
    eligibility_times, eligibility_states, eligibility_ids = _point_in_time_eligibility(
        close_times,
        instrument.history,
    )
    maximum_intervals = max(
        barrier.horizon_minutes * _MINUTE_US // spec.interval_us
        for barrier in spec.barriers
    )
    context = _ExactInstrumentContext(
        bars=bars,
        spec_id=spec.spec_id,
        interval_us=spec.interval_us,
        cost_factors=_cost_factors(spec.costs),
        candidate_times=close_times,
        open_times=open_times,
        entry_prices=[*opens[1:], None],
        closes=closes,
        highs=highs,
        lows=lows,
        sigmas=sigmas,
        eligibility_times=eligibility_times,
        eligibility_states=eligibility_states,
        eligibility_ids=eligibility_ids,
        high_blocks=_extrema_blocks(highs, maximum_intervals, maximum=True),
        low_blocks=_extrema_blocks(lows, maximum_intervals, maximum=False),
        eligibility_required=instrument.eligibility_required,
    )
    frames = [
        _barrier_labels(context, barrier_index, barrier)
        for barrier_index, barrier in enumerate(spec.barriers)
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _label_instrument_float64(
    instrument: _AdmittedInstrument,
    spec: BenchmarkSpec,
    *,
    summary_only: bool = False,
    candidate_indices: pl.Series | None = None,
    coverage_only: bool = False,
) -> pl.DataFrame:
    """Build candidate BEN-v2 labels without leaving the columnar data path."""

    import polars as pl

    context = _prepare_float_context(
        instrument, spec, candidate_indices=candidate_indices
    )
    frames = [
        _float_barrier_labels(
            context,
            barrier_index,
            barrier,
            summary_only=summary_only,
            coverage_only=coverage_only,
        )
        for barrier_index, barrier in enumerate(spec.barriers)
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _prepare_float_context(
    instrument: _AdmittedInstrument,
    spec: BenchmarkSpec,
    *,
    candidate_indices: pl.Series | None = None,
) -> _FloatInstrumentContext:
    import polars as pl

    numeric = instrument.bars
    count = len(numeric)

    model = spec.volatility
    numeric = numeric.with_columns(
        (pl.col("close_price") / pl.col("close_price").shift(1) - 1.0)
        .pow(2)
        .rolling_mean(
            window_size=model.window_intervals,
            min_samples=model.minimum_intervals,
        )
        .alias("_mean_square"),
        pl.col("open_price").shift(-1).alias("_entry_price"),
        pl.col("open_time_us").shift(-1).alias("_entry_time_us"),
    ).with_columns(pl.col("_mean_square").clip(lower_bound=0.0).sqrt().alias("_sigma"))
    sigmas = numeric["_sigma"]
    invalid_sigma = sigmas.is_not_null() & ~sigmas.is_finite()
    if invalid_sigma.any():
        raise BenchmarkError("float64 benchmark volatility is not finite")

    close_times = numeric["close_time_us"]
    if instrument.eligibility_required:
        eligibility_times, eligibility_states, eligibility_ids = (
            _point_in_time_eligibility(close_times.to_list(), instrument.history)
        )
        numeric = numeric.with_columns(
            pl.Series("_eligibility_time_us", eligibility_times, dtype=pl.Int64),
            pl.Series("_eligibility_state", eligibility_states, dtype=pl.Boolean),
            pl.Series("_eligibility_id", eligibility_ids, dtype=pl.String),
        )
    else:
        numeric = numeric.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("_eligibility_time_us"),
            pl.lit(None, dtype=pl.Boolean).alias("_eligibility_state"),
            pl.lit(None, dtype=pl.String).alias("_eligibility_id"),
        )

    maximum_intervals = max(
        barrier.horizon_minutes * _MINUTE_US // spec.interval_us
        for barrier in spec.barriers
    )
    highs = numeric["high_price"]
    lows = numeric["low_price"]
    high_blocks = _extrema_blocks(highs, maximum_intervals, maximum=True)
    low_blocks = _extrema_blocks(lows, maximum_intervals, maximum=False)
    entry_factor, exit_factor = (float(value) for value in _cost_factors(spec.costs))
    indices = (
        pl.int_range(0, count, eager=True, dtype=pl.Int64)
        if candidate_indices is None
        else candidate_indices
    )

    return _FloatInstrumentContext(
        numeric=numeric if candidate_indices is None else numeric.gather(indices),
        spec_id=spec.spec_id,
        interval_us=spec.interval_us,
        count=count,
        close_times=close_times,
        indices=indices,
        highs=highs,
        lows=lows,
        high_blocks=high_blocks,
        low_blocks=low_blocks,
        entry_factor=entry_factor,
        exit_factor=exit_factor,
        eligibility_required=instrument.eligibility_required,
    )


def _float_barrier_labels(
    context: _FloatInstrumentContext,
    barrier_index: int,
    barrier: BarrierDefinition,
    *,
    summary_only: bool,
    coverage_only: bool,
) -> pl.DataFrame:
    import polars as pl

    numeric = context.numeric
    count = context.count
    close_times = context.close_times
    indices = context.indices
    highs = context.highs
    lows = context.lows
    high_blocks = context.high_blocks
    low_blocks = context.low_blocks
    entry_factor = context.entry_factor
    exit_factor = context.exit_factor

    horizon_intervals = barrier.horizon_minutes * _MINUTE_US // context.interval_us
    upper = numeric.select(
        (
            pl.col("_entry_price")
            * entry_factor
            * (1.0 + float(barrier.upper_sigma) * pl.col("_sigma"))
            / exit_factor
        ).alias("threshold")
    )["threshold"]
    lower = numeric.select(
        (
            pl.col("_entry_price")
            * entry_factor
            * (1.0 - float(barrier.lower_sigma) * pl.col("_sigma"))
            / exit_factor
        ).alias("threshold")
    )["threshold"]
    if (upper.is_not_null() & (~upper.is_finite() | (upper <= 0.0))).any() or (
        lower.is_not_null() & ~lower.is_finite()
    ).any():
        raise BenchmarkError("float64 benchmark threshold is outside its range")

    upper_band = _positive_ulp_band(upper, 16)
    lower_band = _positive_ulp_band(lower, 16)
    first_high = _first_threshold_hit(
        highs,
        upper,
        high_blocks,
        horizon_intervals,
        above=True,
        candidate_indices=indices,
    )
    first_low = _first_threshold_hit(
        lows,
        lower,
        low_blocks,
        horizon_intervals,
        above=False,
        candidate_indices=indices,
    )
    near_high = _first_threshold_hit(
        highs,
        upper - upper_band,
        high_blocks,
        horizon_intervals,
        above=True,
        candidate_indices=indices,
    )
    near_low = _first_threshold_hit(
        lows,
        lower + lower_band,
        low_blocks,
        horizon_intervals,
        above=False,
        candidate_indices=indices,
    )

    high_near_value = highs.gather(near_high.fill_null(0))
    low_near_value = lows.gather(near_low.fill_null(0))
    high_is_near = near_high.is_not_null() & (high_near_value <= upper + upper_band)
    low_is_near = near_low.is_not_null() & (low_near_value >= lower - lower_band)
    low_wins = first_low.is_not_null() & (
        first_high.is_null() | (first_low <= first_high)
    )
    high_wins = ~low_wins & first_high.is_not_null()
    decision_index = (
        pl.when(low_wins)
        .then(first_low)
        .when(high_wins)
        .then(first_high)
        .otherwise(indices + horizon_intervals)
    )
    ambiguous = (high_is_near & (near_high <= decision_index)) | (
        low_is_near & (near_low <= decision_index)
    )

    terminal_indices = indices + horizon_intervals
    terminal_times = close_times.gather(terminal_indices, null_on_oob=True)
    horizon_end = numeric["close_time_us"] + barrier.horizon_minutes * _MINUTE_US
    complete = (indices + horizon_intervals < count) & (terminal_times == horizon_end)
    eligibility_path = (
        pl.col("_eligibility_state").fill_null(False)
        if context.eligibility_required
        else pl.lit(True)
    )
    missing_eligibility = (
        pl.col("_eligibility_state").is_null()
        if context.eligibility_required
        else pl.lit(False)
    )
    ineligible = (
        ~pl.col("_eligibility_state") if context.eligibility_required else pl.lit(False)
    )

    work = (
        numeric.with_columns(
            horizon_end.alias("horizon_end_us"),
            complete.alias("_complete"),
            ambiguous.alias("_ambiguous"),
            low_wins.alias("_low_wins"),
            high_wins.alias("_high_wins"),
        )
        .with_columns(
            (
                eligibility_path
                & pl.col("_sigma").is_not_null()
                & (pl.col("_sigma") != 0.0)
                & pl.col("_complete")
            ).alias("_eligible_path")
        )
        .with_columns(
            (pl.col("_eligible_path") & ~pl.col("_ambiguous")).alias("_available")
        )
    )

    exit_reason = (
        pl.when(missing_eligibility)
        .then(pl.lit(LabelExit.MISSING_ELIGIBILITY.value))
        .when(ineligible)
        .then(pl.lit(LabelExit.INELIGIBLE.value))
        .when(pl.col("_sigma").is_null())
        .then(pl.lit(LabelExit.INSUFFICIENT_VOLATILITY_HISTORY.value))
        .when(pl.col("_sigma") == 0.0)
        .then(pl.lit(LabelExit.ZERO_VOLATILITY.value))
        .when(~pl.col("_complete"))
        .then(pl.lit(LabelExit.HORIZON_INCOMPLETE.value))
        .when(pl.col("_ambiguous"))
        .then(pl.lit(LabelExit.NUMERICAL_AMBIGUITY.value))
        .when(pl.col("_low_wins"))
        .then(pl.lit(LabelExit.LOWER_BARRIER.value))
        .when(pl.col("_high_wins"))
        .then(pl.lit(LabelExit.UPPER_BARRIER.value))
        .otherwise(pl.lit(LabelExit.EXPIRY.value))
    )
    outcome = (
        pl.when(~pl.col("_available"))
        .then(pl.lit(LabelOutcome.UNAVAILABLE.value))
        .when(pl.col("_high_wins"))
        .then(pl.lit(LabelOutcome.POSITIVE.value))
        .otherwise(pl.lit(LabelOutcome.NEGATIVE.value))
    )
    if summary_only:
        return work.select(
            pl.lit(barrier.name).alias("barrier_name"),
            pl.col("close_time_us").alias("candidate_time_us"),
            outcome.alias("outcome"),
            exit_reason.alias("exit"),
        )

    time_at_high = close_times.gather(first_high.fill_null(0))
    time_at_low = close_times.gather(first_low.fill_null(0))
    work = work.with_columns(
        time_at_low.alias("_time_at_low"),
        time_at_high.alias("_time_at_high"),
        terminal_times.alias("_terminal_time"),
    )
    exit_time = (
        pl.when(~pl.col("_available"))
        .then(None)
        .when(pl.col("_low_wins"))
        .then(pl.col("_time_at_low"))
        .when(pl.col("_high_wins"))
        .then(pl.col("_time_at_high"))
        .otherwise(pl.col("_terminal_time"))
    )
    if coverage_only:
        return work.select(
            *_IDENTITY_COLUMNS,
            "interval",
            pl.lit(context.spec_id).alias("spec_id"),
            pl.lit(ColumnarNumericalPolicy.FLOAT64_ULP16_V1.value).alias(
                "numerical_policy_id"
            ),
            pl.lit(barrier.name).alias("barrier_name"),
            pl.lit(barrier_index).alias("_barrier_index"),
            pl.col("close_time_us").alias("candidate_time_us"),
            "horizon_end_us",
            outcome.alias("outcome"),
            exit_reason.alias("exit"),
            exit_time.alias("exit_time_us"),
        )

    high_at_hit = highs.gather(first_high.fill_null(0))
    low_at_hit = lows.gather(first_low.fill_null(0))
    terminal_prices = numeric["close_price"].shift(-horizon_intervals)
    maximum_highs = _range_extreme(
        high_blocks,
        horizon_intervals,
        maximum=True,
    )
    minimum_lows = _range_extreme(
        low_blocks,
        horizon_intervals,
        maximum=False,
    )
    entry_prices = numeric["_entry_price"]
    maximum_favorable = (
        maximum_highs * exit_factor / (entry_prices * entry_factor) - 1.0
    )
    maximum_adverse = minimum_lows * exit_factor / (entry_prices * entry_factor) - 1.0
    high_return = high_at_hit * exit_factor / (entry_prices * entry_factor) - 1.0
    low_return = low_at_hit * exit_factor / (entry_prices * entry_factor) - 1.0
    expiry_return = terminal_prices * exit_factor / (entry_prices * entry_factor) - 1.0
    work = work.with_columns(
        maximum_favorable.alias("_maximum_favorable"),
        maximum_adverse.alias("_maximum_adverse"),
        low_return.alias("_low_return"),
        high_return.alias("_high_return"),
        expiry_return.alias("_expiry_return"),
    )
    executable_return = (
        pl.when(~pl.col("_available"))
        .then(None)
        .when(pl.col("_low_wins"))
        .then(pl.col("_low_return"))
        .when(pl.col("_high_wins"))
        .then(pl.col("_high_return"))
        .otherwise(pl.col("_expiry_return"))
    )

    return work.select(
        *_IDENTITY_COLUMNS,
        "interval",
        pl.lit(context.spec_id).alias("spec_id"),
        pl.lit(ColumnarNumericalPolicy.FLOAT64_ULP16_V1.value).alias(
            "numerical_policy_id"
        ),
        pl.lit(barrier.name).alias("barrier_name"),
        pl.lit(barrier_index).alias("_barrier_index"),
        pl.col("close_time_us").alias("candidate_time_us"),
        "horizon_end_us",
        outcome.alias("outcome"),
        exit_reason.alias("exit"),
        exit_time.alias("exit_time_us"),
        pl.when("_available")
        .then("_entry_time_us")
        .otherwise(None)
        .alias("entry_time_us"),
        pl.when(pl.col("_sigma").is_not_null())
        .then("close_time_us")
        .otherwise(None)
        .alias("volatility_end_time_us"),
        pl.col("_sigma").alias("sigma"),
        executable_return.alias("executable_return"),
        pl.when("_available")
        .then("_maximum_favorable")
        .otherwise(None)
        .alias("maximum_favorable_return"),
        pl.when("_available")
        .then("_maximum_adverse")
        .otherwise(None)
        .alias("maximum_adverse_return"),
        pl.col("_eligibility_time_us").alias("eligibility_effective_time_us"),
        pl.col("_eligibility_id").alias("eligibility_evidence_id"),
    )


def _positive_ulp_band(values: pl.Series, ulps: int) -> pl.Series:
    """Return an exact positive-float ULP band without Python materialization."""

    import polars as pl

    next_values = (values.reinterpret(dtype=pl.UInt64) + ulps).reinterpret(
        dtype=pl.Float64
    )
    band = next_values - values
    return band.zip_with((values > 0.0).fill_null(False), values * 0.0)


def _causal_sigmas(closes: list[Decimal], spec: BenchmarkSpec) -> list[Decimal | None]:
    """Preserve the BEN-v1 reference oracle's exact Decimal operation order.

    Polars 1.44.2 cannot roll or square-root Decimal values without rejection or
    Float64 conversion. BEN-NUM-01 evaluates that native float path as a separately
    identified policy; this implementation remains the comparison/fallback path.
    """

    model = spec.volatility
    window: deque[Decimal] = deque()
    sigmas: list[Decimal | None] = []
    for index, close in enumerate(closes):
        if index:
            value = close / closes[index - 1] - Decimal(1)
            window.append(value * value)
            if len(window) > model.window_intervals:
                window.popleft()
        sigmas.append(
            (sum(window, Decimal(0)) / Decimal(len(window))).sqrt()
            if len(window) >= model.minimum_intervals
            else None
        )
    return sigmas


def _point_in_time_eligibility(
    candidate_times: list[int],
    history: tuple[tuple[int, bool, str], ...],
) -> tuple[list[int | None], list[bool | None], list[str | None]]:
    times: list[int | None] = []
    states: list[bool | None] = []
    evidence_ids: list[str | None] = []
    position = 0
    current: tuple[int, bool, str] | None = None
    for candidate_time in candidate_times:
        while position < len(history) and history[position][0] <= candidate_time:
            current = history[position]
            position += 1
        times.append(current[0] if current else None)
        states.append(current[1] if current else None)
        evidence_ids.append(current[2] if current else None)
    return times, states, evidence_ids


def _barrier_thresholds(
    entries: list[Decimal | None],
    sigmas: list[Decimal | None],
    upper_sigma: Decimal,
    lower_sigma: Decimal,
    cost_factors: tuple[Decimal, Decimal],
) -> tuple[list[Decimal | None], list[Decimal | None]]:
    entry_factor, exit_factor = cost_factors
    upper: list[Decimal | None] = []
    lower: list[Decimal | None] = []
    for entry, sigma in zip(entries, sigmas, strict=True):
        if entry is None or sigma is None:
            upper.append(None)
            lower.append(None)
            continue
        admitted_entry = entry * entry_factor
        upper.append(admitted_entry * (Decimal(1) + upper_sigma * sigma) / exit_factor)
        lower.append(admitted_entry * (Decimal(1) - lower_sigma * sigma) / exit_factor)
    return upper, lower


def _extrema_blocks(
    series: pl.Series, maximum_size: int, *, maximum: bool
) -> list[pl.Series]:
    blocks = [series]
    size = 1
    while size * 2 <= maximum_size:
        previous = blocks[-1]
        right = previous.shift(-size)
        comparison = previous >= right if maximum else previous <= right
        blocks.append(previous.zip_with(comparison.fill_null(False), right))
        size *= 2
    return blocks


def _range_extreme(
    blocks: list[pl.Series],
    length: int,
    *,
    maximum: bool,
) -> pl.Series:
    selected = None
    offset = 1
    bit = 0
    remaining = length
    while remaining:
        if remaining & 1:
            block = blocks[bit].shift(-offset)
            if selected is None:
                selected = block
            else:
                comparison = selected >= block if maximum else selected <= block
                selected = selected.zip_with(comparison.fill_null(False), block)
            offset += 1 << bit
        remaining >>= 1
        bit += 1
    if selected is None:
        raise AssertionError("positive benchmark horizons must select an extremum")
    return selected


def _first_threshold_hit(
    values: pl.Series,
    thresholds: pl.Series,
    blocks: list[pl.Series],
    horizon: int,
    *,
    above: bool,
    candidate_indices: pl.Series | None = None,
) -> pl.Series:
    import polars as pl

    count = len(values)
    indices = (
        pl.int_range(0, count, eager=True, dtype=pl.Int64)
        if candidate_indices is None
        else candidate_indices
    )
    cursor = indices + 1
    end = cursor + horizon
    for bit in range(horizon.bit_length() - 1, -1, -1):
        size = 1 << bit
        if size > horizon:
            continue
        within = (cursor + size <= end) & (cursor + size <= count)
        gathered = blocks[bit].gather(cursor.clip(0, count - 1))
        no_hit = gathered < thresholds if above else gathered > thresholds
        advance = within & no_hit.fill_null(False)
        cursor = cursor + advance.cast(pl.Int64) * size
    gathered = values.gather(cursor.clip(0, count - 1))
    crossed = gathered >= thresholds if above else gathered <= thresholds
    hit = (cursor < end) & (cursor < count) & crossed.fill_null(False)
    return (
        pl.DataFrame({"cursor": cursor, "hit": hit})
        .select(pl.when("hit").then("cursor").otherwise(None).alias("index"))
        .to_series()
    )


def _barrier_labels(
    context: _ExactInstrumentContext,
    barrier_index: int,
    barrier: BarrierDefinition,
) -> pl.DataFrame:
    import polars as pl

    horizon_intervals = barrier.horizon_minutes * _MINUTE_US // context.interval_us
    upper_thresholds, lower_thresholds = _barrier_thresholds(
        context.entry_prices,
        context.sigmas,
        barrier.upper_sigma,
        barrier.lower_sigma,
        context.cost_factors,
    )
    first_high = _first_threshold_hit(
        context.highs,
        pl.Series(upper_thresholds),
        context.high_blocks,
        horizon_intervals,
        above=True,
    ).to_list()
    first_low = _first_threshold_hit(
        context.lows,
        pl.Series(lower_thresholds),
        context.low_blocks,
        horizon_intervals,
        above=False,
    ).to_list()
    maximum_highs = _range_extreme(
        context.high_blocks,
        horizon_intervals,
        maximum=True,
    ).to_list()
    minimum_lows = _range_extreme(
        context.low_blocks,
        horizon_intervals,
        maximum=False,
    ).to_list()
    bars = context.bars
    candidate_times = context.candidate_times
    open_times = context.open_times
    entry_prices = context.entry_prices
    closes = context.closes
    highs = context.highs
    lows = context.lows
    sigmas = context.sigmas
    eligibility_times = context.eligibility_times
    eligibility_states = context.eligibility_states
    eligibility_ids = context.eligibility_ids
    cost_factors = context.cost_factors
    barrier_name = barrier.name
    horizon_minutes = barrier.horizon_minutes

    count = len(candidate_times)
    entry_factor, exit_factor = cost_factors
    columns: dict[str, list[object]] = {
        "candidate_time_us": [],
        "horizon_end_us": [],
        "outcome": [],
        "exit": [],
        "exit_time_us": [],
        "entry_time_us": [],
        "volatility_end_time_us": [],
        "sigma": [],
        "executable_return": [],
        "maximum_favorable_return": [],
        "maximum_adverse_return": [],
        "eligibility_effective_time_us": [],
        "eligibility_evidence_id": [],
    }
    for index, candidate_time in enumerate(candidate_times):
        horizon_end = candidate_time + horizon_minutes * _MINUTE_US
        sigma = sigmas[index]
        outcome = LabelOutcome.UNAVAILABLE
        exit_reason = LabelExit.MISSING_ELIGIBILITY
        exit_time = None
        entry_time = None
        executable_return = None
        maximum_favorable = None
        maximum_adverse = None

        if context.eligibility_required and eligibility_states[index] is None:
            exit_reason = LabelExit.MISSING_ELIGIBILITY
        elif context.eligibility_required and eligibility_states[index] is False:
            exit_reason = LabelExit.INELIGIBLE
        elif sigma is None:
            exit_reason = LabelExit.INSUFFICIENT_VOLATILITY_HISTORY
        elif sigma == 0:
            exit_reason = LabelExit.ZERO_VOLATILITY
        else:
            terminal_index = index + horizon_intervals
            complete = (
                terminal_index < count
                and candidate_times[-1] >= horizon_end
                and candidate_times[terminal_index] == horizon_end
            )
            if not complete:
                exit_reason = LabelExit.HORIZON_INCOMPLETE
            else:
                entry = entry_prices[index]
                maximum_high = maximum_highs[index]
                minimum_low = minimum_lows[index]
                if entry is None or maximum_high is None or minimum_low is None:
                    raise AssertionError("complete horizon has no admitted path values")
                entry_time = open_times[index + 1]
                maximum_favorable = _executable_return_from_factors(
                    entry,
                    maximum_high,
                    entry_factor,
                    exit_factor,
                )
                maximum_adverse = _executable_return_from_factors(
                    entry,
                    minimum_low,
                    entry_factor,
                    exit_factor,
                )
                low_index = first_low[index]
                high_index = first_high[index]
                if low_index is not None and (
                    high_index is None or low_index <= high_index
                ):
                    outcome = LabelOutcome.NEGATIVE
                    exit_reason = LabelExit.LOWER_BARRIER
                    exit_time = candidate_times[low_index]
                    executable_return = _executable_return_from_factors(
                        entry,
                        lows[low_index],
                        entry_factor,
                        exit_factor,
                    )
                elif high_index is not None:
                    outcome = LabelOutcome.POSITIVE
                    exit_reason = LabelExit.UPPER_BARRIER
                    exit_time = candidate_times[high_index]
                    executable_return = _executable_return_from_factors(
                        entry,
                        highs[high_index],
                        entry_factor,
                        exit_factor,
                    )
                else:
                    outcome = LabelOutcome.NEGATIVE
                    exit_reason = LabelExit.EXPIRY
                    exit_time = candidate_times[terminal_index]
                    executable_return = _executable_return_from_factors(
                        entry,
                        closes[terminal_index],
                        entry_factor,
                        exit_factor,
                    )

        values = (
            candidate_time,
            horizon_end,
            outcome.value,
            exit_reason.value,
            exit_time,
            entry_time,
            candidate_time if sigma is not None else None,
            sigma,
            executable_return,
            maximum_favorable,
            maximum_adverse,
            eligibility_times[index],
            eligibility_ids[index],
        )
        for name, value in zip(columns, values, strict=True):
            columns[name].append(value)

    identity = {
        column: [bars[column][0]] * count for column in (*_IDENTITY_COLUMNS, "interval")
    }
    return pl.DataFrame(
        {
            **identity,
            "spec_id": [context.spec_id] * count,
            "numerical_policy_id": [ColumnarNumericalPolicy.EXACT_DECIMAL_V1.value]
            * count,
            "barrier_name": [barrier_name] * count,
            "_barrier_index": [barrier_index] * count,
            **columns,
        }
    )


__all__ = [
    "DEFAULT_COLUMNAR_RESEARCH_LIMITS",
    "ColumnarDependencyError",
    "ColumnarExitCount",
    "ColumnarNumericalPolicy",
    "ColumnarResearchLimits",
    "ColumnarResearchSummary",
    "CoverageConditionedLabels",
    "CoverageConditionedSummary",
    "build_columnar_labels",
    "build_coverage_conditioned_candidate_labels",
    "build_coverage_conditioned_labels",
    "summarize_columnar_labels",
    "summarize_coverage_conditioned_partitions",
    "summarize_research_partitions",
]
