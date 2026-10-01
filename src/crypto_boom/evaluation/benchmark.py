"""Independent, versioned explosive-move benchmark and leakage-safe splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from itertools import pairwise

from crypto_boom import _artifacts
from crypto_boom.market import InstrumentId, KlineEvent

__all__ = (
    "BarrierDefinition",
    "BenchmarkError",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkSpec",
    "ChronologicalFold",
    "CostModel",
    "EligibilitySnapshot",
    "EpisodeLabel",
    "ExplosiveUpEpisode",
    "LabelExit",
    "LabelOutcome",
    "LeakageAudit",
    "PrevalenceRow",
    "SplitModel",
    "SplitPlan",
    "VolatilityModel",
    "benchmark_v1_spec",
    "build_benchmark",
    "chronological_purged_splits",
)

_MINUTE_US = 60_000_000
_BPS = Decimal("10000")


class BenchmarkError(ValueError):
    """Benchmark input or specification violates the frozen label contract."""


class LabelOutcome(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNAVAILABLE = "unavailable"


class LabelExit(StrEnum):
    UPPER_BARRIER = "upper_barrier"
    LOWER_BARRIER = "lower_barrier"
    EXPIRY = "expiry"
    MISSING_ELIGIBILITY = "missing_eligibility"
    INELIGIBLE = "ineligible"
    INSUFFICIENT_VOLATILITY_HISTORY = "insufficient_volatility_history"
    ZERO_VOLATILITY = "zero_volatility"
    NUMERICAL_AMBIGUITY = "numerical_ambiguity"
    HORIZON_INCOMPLETE = "horizon_incomplete"


@dataclass(frozen=True, slots=True)
class CostModel:
    version: str
    entry_bps: Decimal
    exit_bps: Decimal

    def __post_init__(self) -> None:
        if not self.version:
            raise BenchmarkError("cost version must not be empty")
        for name, value in (
            ("entry cost", self.entry_bps),
            ("exit cost", self.exit_bps),
        ):
            if not value.is_finite() or value < 0 or value >= _BPS:
                raise BenchmarkError(f"{name} must be finite and below 10000 bps")

    @property
    def model_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "entry_bps": str(self.entry_bps),
            "exit_bps": str(self.exit_bps),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class VolatilityModel:
    version: str
    window_intervals: int
    minimum_intervals: int

    def __post_init__(self) -> None:
        if not self.version:
            raise BenchmarkError("volatility version must not be empty")
        if self.window_intervals < 2:
            raise BenchmarkError("volatility window must contain at least two returns")
        if not 2 <= self.minimum_intervals <= self.window_intervals:
            raise BenchmarkError("minimum volatility history is outside the window")

    @property
    def model_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "minimum_intervals": self.minimum_intervals,
            "version": self.version,
            "window_intervals": self.window_intervals,
        }


@dataclass(frozen=True, slots=True)
class BarrierDefinition:
    name: str
    horizon_minutes: int
    upper_sigma: Decimal
    lower_sigma: Decimal

    def __post_init__(self) -> None:
        if not self.name:
            raise BenchmarkError("barrier name must not be empty")
        if self.horizon_minutes < 1:
            raise BenchmarkError("barrier horizon must be positive")
        for name, value in (
            ("upper sigma", self.upper_sigma),
            ("lower sigma", self.lower_sigma),
        ):
            if not value.is_finite() or value <= 0:
                raise BenchmarkError(f"{name} must be finite and positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon_minutes": self.horizon_minutes,
            "lower_sigma": str(self.lower_sigma),
            "name": self.name,
            "upper_sigma": str(self.upper_sigma),
        }


@dataclass(frozen=True, slots=True)
class SplitModel:
    version: str
    fold_count: int
    final_holdout_fraction: Decimal

    def __post_init__(self) -> None:
        if not self.version:
            raise BenchmarkError("split version must not be empty")
        if self.fold_count < 1:
            raise BenchmarkError("at least one chronological fold is required")
        if (
            not self.final_holdout_fraction.is_finite()
            or self.final_holdout_fraction <= 0
            or self.final_holdout_fraction >= Decimal("0.5")
        ):
            raise BenchmarkError("final holdout fraction must be in (0, 0.5)")

    @property
    def model_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "final_holdout_fraction": str(self.final_holdout_fraction),
            "fold_count": self.fold_count,
            "version": self.version,
        }


_DEFAULT_METRICS = (
    "episode_recall",
    "alert_precision",
    "detection_delay_us",
    "captured_move_fraction",
    "net_forward_expectancy",
    "maximum_favorable_excursion",
    "maximum_adverse_excursion",
    "time_to_barrier_us",
    "false_alerts_per_utc_day",
    "false_alerts_per_symbol",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    version: str
    interval_us: int
    volatility: VolatilityModel
    costs: CostModel
    barriers: tuple[BarrierDefinition, ...]
    splits: SplitModel
    metrics: tuple[str, ...] = _DEFAULT_METRICS

    def __post_init__(self) -> None:
        if not self.version:
            raise BenchmarkError("benchmark version must not be empty")
        if self.interval_us < 1:
            raise BenchmarkError("benchmark interval must be positive")
        if not self.barriers:
            raise BenchmarkError("at least one barrier definition is required")
        names = tuple(barrier.name for barrier in self.barriers)
        if len(set(names)) != len(names):
            raise BenchmarkError("barrier names must be unique")
        if any(
            barrier.horizon_minutes * _MINUTE_US % self.interval_us
            for barrier in self.barriers
        ):
            raise BenchmarkError("barrier horizons must align with the interval grid")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise BenchmarkError("metric definitions must be non-empty and unique")

    @property
    def spec_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    @property
    def maximum_horizon_us(self) -> int:
        return max(barrier.horizon_minutes for barrier in self.barriers) * _MINUTE_US

    def to_mapping(self) -> dict[str, object]:
        return {
            "barriers": [barrier.to_mapping() for barrier in self.barriers],
            "costs": self.costs.to_mapping(),
            "interval_us": self.interval_us,
            "metrics": list(self.metrics),
            "splits": self.splits.to_mapping(),
            "version": self.version,
            "volatility": self.volatility.to_mapping(),
        }


def benchmark_v1_spec() -> BenchmarkSpec:
    """Return the predeclared, untuned historical benchmark specification."""

    barriers = tuple(
        BarrierDefinition(
            name=f"h{horizon}-u{upper}-d{lower}",
            horizon_minutes=horizon,
            upper_sigma=Decimal(upper),
            lower_sigma=Decimal(lower),
        )
        for horizon in (30, 120, 360)
        for upper, lower in (("2", "1"), ("3", "2"))
    )
    return BenchmarkSpec(
        version="explosive-up-benchmark-v1",
        interval_us=_MINUTE_US,
        volatility=VolatilityModel(
            version="causal-simple-return-rms-v1",
            window_intervals=60,
            minimum_intervals=30,
        ),
        costs=CostModel(
            version="historical-bar-conservative-cost-v1",
            entry_bps=Decimal("15"),
            exit_bps=Decimal("15"),
        ),
        barriers=barriers,
        splits=SplitModel(
            version="walk-forward-purged-v1",
            fold_count=3,
            final_holdout_fraction=Decimal("0.20"),
        ),
    )


@dataclass(frozen=True, slots=True)
class EligibilitySnapshot:
    """Caller-supplied status; this value does not certify its evidence source."""

    instrument: InstrumentId
    effective_time_us: int
    eligible: bool
    evidence_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.effective_time_us < 0:
            raise BenchmarkError("eligibility time must be non-negative")
        if type(self.eligible) is not bool:
            raise BenchmarkError("eligibility state must be boolean")
        if not self.evidence_id:
            raise BenchmarkError("eligibility evidence identity must not be empty")
        if not self.reason:
            raise BenchmarkError("eligibility reason must not be empty")


@dataclass(frozen=True, slots=True)
class EpisodeLabel:
    spec_id: str
    barrier_name: str
    instrument: InstrumentId
    candidate_time_us: int
    horizon_end_us: int
    outcome: LabelOutcome
    exit: LabelExit
    exit_time_us: int | None
    entry_time_us: int | None
    volatility_end_time_us: int | None
    sigma: Decimal | None
    executable_return: Decimal | None
    maximum_favorable_return: Decimal | None
    maximum_adverse_return: Decimal | None
    eligibility_effective_time_us: int | None
    eligibility_evidence_id: str | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "barrier_name": self.barrier_name,
            "candidate_time_us": self.candidate_time_us,
            "eligibility_effective_time_us": self.eligibility_effective_time_us,
            "eligibility_evidence_id": self.eligibility_evidence_id,
            "entry_time_us": self.entry_time_us,
            "executable_return": _decimal_text(self.executable_return),
            "exit": self.exit.value,
            "exit_time_us": self.exit_time_us,
            "horizon_end_us": self.horizon_end_us,
            "instrument": _instrument_mapping(self.instrument),
            "maximum_adverse_return": _decimal_text(self.maximum_adverse_return),
            "maximum_favorable_return": _decimal_text(self.maximum_favorable_return),
            "outcome": self.outcome.value,
            "sigma": _decimal_text(self.sigma),
            "spec_id": self.spec_id,
            "volatility_end_time_us": self.volatility_end_time_us,
        }


@dataclass(frozen=True, slots=True)
class ExplosiveUpEpisode:
    episode_id: str
    barrier_name: str
    instrument: InstrumentId
    first_candidate_time_us: int
    last_candidate_time_us: int
    end_time_us: int
    label_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "barrier_name": self.barrier_name,
            "end_time_us": self.end_time_us,
            "episode_id": self.episode_id,
            "first_candidate_time_us": self.first_candidate_time_us,
            "instrument": _instrument_mapping(self.instrument),
            "label_count": self.label_count,
            "last_candidate_time_us": self.last_candidate_time_us,
        }


@dataclass(frozen=True, slots=True)
class ChronologicalFold:
    index: int
    train_times_us: tuple[int, ...]
    validation_times_us: tuple[int, ...]
    purged_times_us: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "purged_times_us": list(self.purged_times_us),
            "train_times_us": list(self.train_times_us),
            "validation_times_us": list(self.validation_times_us),
        }


@dataclass(frozen=True, slots=True)
class SplitPlan:
    split_model_id: str
    maximum_horizon_us: int
    folds: tuple[ChronologicalFold, ...]
    final_holdout_times_us: tuple[int, ...]
    purged_before_holdout_times_us: tuple[int, ...]

    @property
    def split_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "final_holdout_times_us": list(self.final_holdout_times_us),
            "folds": [fold.to_mapping() for fold in self.folds],
            "maximum_horizon_us": self.maximum_horizon_us,
            "purged_before_holdout_times_us": list(self.purged_before_holdout_times_us),
            "split_model_id": self.split_model_id,
        }


@dataclass(frozen=True, slots=True)
class PrevalenceRow:
    barrier_name: str
    total: int
    available: int
    positive: int
    negative: int
    unavailable: int
    positive_rate: Decimal | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "available": self.available,
            "barrier_name": self.barrier_name,
            "negative": self.negative,
            "positive": self.positive,
            "positive_rate": _decimal_text(self.positive_rate),
            "total": self.total,
            "unavailable": self.unavailable,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """BEN calculation summary, not historical eligibility qualification.

    ``ready`` describes calculation and temporal checks under the supplied
    snapshots. Evidence authority is assessed outside this module.
    """

    spec_id: str
    label_count: int
    episode_count: int
    prevalence: tuple[PrevalenceRow, ...]
    split_id: str
    leakage_audit_id: str
    ready: bool

    @property
    def computation_ready(self) -> bool:
        """Explicit name for the legacy ``ready`` field's limited meaning."""

        return self.ready

    @property
    def report_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "episode_count": self.episode_count,
            "label_count": self.label_count,
            "leakage_audit_id": self.leakage_audit_id,
            "prevalence": [row.to_mapping() for row in self.prevalence],
            "ready": self.ready,
            "spec_id": self.spec_id,
            "split_id": self.split_id,
        }


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    causal_volatility: bool
    next_observation_entry: bool
    point_in_time_eligibility: bool
    purged_chronology: bool
    detector_feature_dependency: bool
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations and all(
            (
                self.causal_volatility,
                self.next_observation_entry,
                self.point_in_time_eligibility,
                self.purged_chronology,
                not self.detector_feature_dependency,
            )
        )

    @property
    def audit_id(self) -> str:
        return _artifacts.content_id(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "causal_volatility": self.causal_volatility,
            "detector_feature_dependency": self.detector_feature_dependency,
            "next_observation_entry": self.next_observation_entry,
            "point_in_time_eligibility": self.point_in_time_eligibility,
            "purged_chronology": self.purged_chronology,
            "violations": list(self.violations),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    spec: BenchmarkSpec
    labels: tuple[EpisodeLabel, ...]
    episodes: tuple[ExplosiveUpEpisode, ...]
    splits: SplitPlan
    leakage_audit: LeakageAudit
    report: BenchmarkReport

    @property
    def result_id(self) -> str:
        return _artifacts.content_id(self._content_mapping())

    def _content_mapping(self) -> dict[str, object]:
        return {
            "episodes": [episode.to_mapping() for episode in self.episodes],
            "labels": [label.to_mapping() for label in self.labels],
            "leakage_audit": self.leakage_audit.to_mapping(),
            "report": self.report.to_mapping(),
            "schema_version": 1,
            "spec": self.spec.to_mapping(),
            "splits": self.splits.to_mapping(),
        }

    def to_bytes(self) -> bytes:
        return _artifacts.canonical_json(
            {"result_id": self.result_id, **self._content_mapping()}
        )


def build_benchmark(
    klines: Sequence[KlineEvent],
    eligibility: Sequence[EligibilitySnapshot],
    spec: BenchmarkSpec,
) -> BenchmarkResult:
    """Build independent labels, episodes, purged splits, and prevalence."""

    bars = tuple(klines)
    if not bars:
        raise BenchmarkError("benchmark requires at least one kline")
    _validate_klines(bars, spec)
    snapshots = tuple(
        sorted(eligibility, key=lambda snapshot: snapshot.effective_time_us)
    )
    if any(snapshot.instrument != bars[0].instrument for snapshot in snapshots):
        raise BenchmarkError("eligibility history contains another instrument")
    if any(
        previous.effective_time_us >= current.effective_time_us
        for previous, current in pairwise(snapshots)
    ):
        raise BenchmarkError("eligibility history is not strictly chronological")

    spec_id = spec.spec_id
    labels = tuple(
        _label_candidate(bars, index, snapshots, barrier, spec, spec_id)
        for index in range(len(bars))
        for barrier in spec.barriers
    )
    candidate_times = tuple(bar.close_time.epoch_microseconds for bar in bars)
    return _assemble_benchmark_result(spec, labels, candidate_times)


def _assemble_benchmark_result(
    spec: BenchmarkSpec,
    labels: tuple[EpisodeLabel, ...],
    candidate_times_us: tuple[int, ...],
) -> BenchmarkResult:
    """Assemble shared benchmark evidence from oracle-equivalent labels."""

    episodes = _group_positive_episodes(labels)
    splits = chronological_purged_splits(
        candidate_times_us,
        maximum_horizon_us=spec.maximum_horizon_us,
        model=spec.splits,
    )
    audit = _audit(labels, splits)
    prevalence = _prevalence(labels, spec.barriers)
    ready = audit.passed and all(row.available > 0 for row in prevalence)
    report = BenchmarkReport(
        spec_id=spec.spec_id,
        label_count=len(labels),
        episode_count=len(episodes),
        prevalence=prevalence,
        split_id=splits.split_id,
        leakage_audit_id=audit.audit_id,
        ready=ready,
    )
    return BenchmarkResult(spec, labels, episodes, splits, audit, report)


def chronological_purged_splits(
    candidate_times_us: Sequence[int],
    *,
    maximum_horizon_us: int,
    model: SplitModel,
) -> SplitPlan:
    """Create growing chronological folds and a never-reused final holdout."""

    times = tuple(candidate_times_us)
    if tuple(sorted(set(times))) != times:
        raise BenchmarkError("candidate times must be unique and chronological")
    if maximum_horizon_us < 1:
        raise BenchmarkError("maximum label horizon must be positive")
    minimum_count = model.fold_count + 2
    if len(times) < minimum_count:
        raise BenchmarkError("not enough candidates for folds and final holdout")

    holdout_count = int(
        (Decimal(len(times)) * model.final_holdout_fraction).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    holdout_start_index = len(times) - max(1, holdout_count)
    holdout = times[holdout_start_index:]
    holdout_start = holdout[0]
    development_source = times[:holdout_start_index]
    development = tuple(
        value
        for value in development_source
        if value + maximum_horizon_us < holdout_start
    )
    purged_before_holdout = tuple(
        value
        for value in development_source
        if value + maximum_horizon_us >= holdout_start
    )
    part_count = model.fold_count + 1
    if len(development) < part_count:
        raise BenchmarkError("purging leaves too few candidates for folds")

    boundaries = tuple(
        index * len(development) // part_count for index in range(part_count + 1)
    )
    folds: list[ChronologicalFold] = []
    for fold_index in range(1, part_count):
        validation = development[boundaries[fold_index] : boundaries[fold_index + 1]]
        if not validation:
            raise BenchmarkError("chronological validation fold is empty")
        validation_start = validation[0]
        train_source = development[: boundaries[fold_index]]
        train = tuple(
            value
            for value in train_source
            if value + maximum_horizon_us < validation_start
        )
        purged = tuple(
            value
            for value in train_source
            if value + maximum_horizon_us >= validation_start
        )
        if not train:
            raise BenchmarkError("purging leaves an empty training fold")
        folds.append(
            ChronologicalFold(
                index=fold_index - 1,
                train_times_us=train,
                validation_times_us=validation,
                purged_times_us=purged,
            )
        )

    return SplitPlan(
        split_model_id=model.model_id,
        maximum_horizon_us=maximum_horizon_us,
        folds=tuple(folds),
        final_holdout_times_us=holdout,
        purged_before_holdout_times_us=purged_before_holdout,
    )


def _validate_klines(
    bars: tuple[KlineEvent, ...],
    spec: BenchmarkSpec,
) -> None:
    instrument = bars[0].instrument
    interval = bars[0].interval
    previous_open: int | None = None
    for bar in bars:
        if bar.instrument != instrument or bar.interval != interval:
            raise BenchmarkError("benchmark input must contain one instrument/interval")
        if not bar.closed or not bar.quality.usable_for_final_transition:
            raise BenchmarkError("benchmark input contains non-final evidence")
        open_time = bar.open_time.epoch_microseconds
        if previous_open is not None and open_time - previous_open != spec.interval_us:
            raise BenchmarkError("benchmark input is not a complete interval grid")
        previous_open = open_time


def _label_candidate(
    bars: tuple[KlineEvent, ...],
    index: int,
    snapshots: tuple[EligibilitySnapshot, ...],
    barrier: BarrierDefinition,
    spec: BenchmarkSpec,
    spec_id: str,
) -> EpisodeLabel:
    candidate = bars[index]
    candidate_time = candidate.close_time.epoch_microseconds
    horizon_end = candidate_time + barrier.horizon_minutes * _MINUTE_US
    snapshot = _eligibility_at(
        snapshots,
        candidate.instrument,
        candidate_time,
    )
    sigma = _causal_sigma(bars, index, spec.volatility)
    eligibility_time = snapshot.effective_time_us if snapshot is not None else None
    eligibility_id = snapshot.evidence_id if snapshot is not None else None

    def label(
        outcome: LabelOutcome,
        exit_reason: LabelExit,
        *,
        exit_time_us: int | None = None,
        entry_time_us: int | None = None,
        executable_return: Decimal | None = None,
        maximum_favorable_return: Decimal | None = None,
        maximum_adverse_return: Decimal | None = None,
    ) -> EpisodeLabel:
        return EpisodeLabel(
            spec_id=spec_id,
            barrier_name=barrier.name,
            instrument=candidate.instrument,
            candidate_time_us=candidate_time,
            horizon_end_us=horizon_end,
            outcome=outcome,
            exit=exit_reason,
            exit_time_us=exit_time_us,
            entry_time_us=entry_time_us,
            volatility_end_time_us=candidate_time if sigma is not None else None,
            sigma=sigma,
            executable_return=executable_return,
            maximum_favorable_return=maximum_favorable_return,
            maximum_adverse_return=maximum_adverse_return,
            eligibility_effective_time_us=eligibility_time,
            eligibility_evidence_id=eligibility_id,
        )

    if snapshot is None:
        return label(LabelOutcome.UNAVAILABLE, LabelExit.MISSING_ELIGIBILITY)
    if not snapshot.eligible:
        return label(LabelOutcome.UNAVAILABLE, LabelExit.INELIGIBLE)
    if sigma is None:
        return label(
            LabelOutcome.UNAVAILABLE,
            LabelExit.INSUFFICIENT_VOLATILITY_HISTORY,
        )
    if sigma == 0:
        return label(LabelOutcome.UNAVAILABLE, LabelExit.ZERO_VOLATILITY)
    if index + 1 >= len(bars) or bars[-1].close_time.epoch_microseconds < horizon_end:
        return label(LabelOutcome.UNAVAILABLE, LabelExit.HORIZON_INCOMPLETE)

    horizon_intervals = barrier.horizon_minutes * _MINUTE_US // spec.interval_us
    path = bars[index + 1 : index + 1 + horizon_intervals]
    if not path or path[-1].close_time.epoch_microseconds != horizon_end:
        return label(LabelOutcome.UNAVAILABLE, LabelExit.HORIZON_INCOMPLETE)

    entry = path[0]
    entry_price = entry.open_price
    entry_time = entry.open_time.epoch_microseconds
    upper = barrier.upper_sigma * sigma
    lower = -(barrier.lower_sigma * sigma)
    favorable = tuple(
        _executable_return(entry_price, bar.high_price, spec.costs) for bar in path
    )
    adverse = tuple(
        _executable_return(entry_price, bar.low_price, spec.costs) for bar in path
    )
    maximum_favorable = max(favorable)
    maximum_adverse = min(adverse)

    for bar, high_return, low_return in zip(path, favorable, adverse, strict=True):
        exit_time = bar.close_time.epoch_microseconds
        if low_return <= lower:
            return label(
                LabelOutcome.NEGATIVE,
                LabelExit.LOWER_BARRIER,
                exit_time_us=exit_time,
                entry_time_us=entry_time,
                executable_return=low_return,
                maximum_favorable_return=maximum_favorable,
                maximum_adverse_return=maximum_adverse,
            )
        if high_return >= upper:
            return label(
                LabelOutcome.POSITIVE,
                LabelExit.UPPER_BARRIER,
                exit_time_us=exit_time,
                entry_time_us=entry_time,
                executable_return=high_return,
                maximum_favorable_return=maximum_favorable,
                maximum_adverse_return=maximum_adverse,
            )

    terminal_return = _executable_return(
        entry_price,
        path[-1].close_price,
        spec.costs,
    )
    return label(
        LabelOutcome.NEGATIVE,
        LabelExit.EXPIRY,
        exit_time_us=path[-1].close_time.epoch_microseconds,
        entry_time_us=entry_time,
        executable_return=terminal_return,
        maximum_favorable_return=maximum_favorable,
        maximum_adverse_return=maximum_adverse,
    )


def _causal_sigma(
    bars: tuple[KlineEvent, ...],
    index: int,
    model: VolatilityModel,
) -> Decimal | None:
    first_return_index = max(1, index - model.window_intervals + 1)
    returns = tuple(
        bars[position].close_price / bars[position - 1].close_price - Decimal(1)
        for position in range(first_return_index, index + 1)
    )
    if len(returns) < model.minimum_intervals:
        return None
    variance = sum((value * value for value in returns), Decimal(0)) / Decimal(
        len(returns)
    )
    return variance.sqrt()


def _eligibility_at(
    snapshots: tuple[EligibilitySnapshot, ...],
    instrument: InstrumentId,
    candidate_time_us: int,
) -> EligibilitySnapshot | None:
    admitted = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.instrument == instrument
        and snapshot.effective_time_us <= candidate_time_us
    )
    return admitted[-1] if admitted else None


def _cost_factors(costs: CostModel) -> tuple[Decimal, Decimal]:
    return (
        Decimal(1) + costs.entry_bps / _BPS,
        Decimal(1) - costs.exit_bps / _BPS,
    )


def _executable_return_from_factors(
    entry_price: Decimal,
    exit_price: Decimal,
    entry_factor: Decimal,
    exit_factor: Decimal,
) -> Decimal:
    return exit_price * exit_factor / (entry_price * entry_factor) - Decimal(1)


def _executable_return(
    entry_price: Decimal,
    exit_price: Decimal,
    costs: CostModel,
) -> Decimal:
    return _executable_return_from_factors(
        entry_price,
        exit_price,
        *_cost_factors(costs),
    )


def _group_positive_episodes(
    labels: tuple[EpisodeLabel, ...],
) -> tuple[ExplosiveUpEpisode, ...]:
    positives = sorted(
        (
            label
            for label in labels
            if label.outcome is LabelOutcome.POSITIVE and label.exit_time_us is not None
        ),
        key=lambda label: (
            label.barrier_name,
            label.instrument.venue.name,
            label.instrument.venue.market,
            label.instrument.symbol,
            label.candidate_time_us,
        ),
    )
    episodes: list[ExplosiveUpEpisode] = []
    group: list[EpisodeLabel] = []
    for label in positives:
        if group and (
            label.barrier_name != group[0].barrier_name
            or label.instrument != group[0].instrument
            or label.candidate_time_us
            > max(member.exit_time_us or member.candidate_time_us for member in group)
        ):
            episodes.append(_episode(group))
            group = []
        group.append(label)
    if group:
        episodes.append(_episode(group))
    return tuple(episodes)


def _episode(group: list[EpisodeLabel]) -> ExplosiveUpEpisode:
    first = group[0]
    end_time = max(label.exit_time_us or label.candidate_time_us for label in group)
    content = {
        "barrier_name": first.barrier_name,
        "end_time_us": end_time,
        "first_candidate_time_us": first.candidate_time_us,
        "instrument": _instrument_mapping(first.instrument),
        "last_candidate_time_us": group[-1].candidate_time_us,
        "label_count": len(group),
    }
    return ExplosiveUpEpisode(
        episode_id=_artifacts.content_id(content),
        barrier_name=first.barrier_name,
        instrument=first.instrument,
        first_candidate_time_us=first.candidate_time_us,
        last_candidate_time_us=group[-1].candidate_time_us,
        end_time_us=end_time,
        label_count=len(group),
    )


def _prevalence(
    labels: tuple[EpisodeLabel, ...],
    barriers: tuple[BarrierDefinition, ...],
) -> tuple[PrevalenceRow, ...]:
    rows: list[PrevalenceRow] = []
    for barrier in barriers:
        selected = tuple(
            label for label in labels if label.barrier_name == barrier.name
        )
        positive = sum(label.outcome is LabelOutcome.POSITIVE for label in selected)
        negative = sum(label.outcome is LabelOutcome.NEGATIVE for label in selected)
        rows.append(
            _prevalence_row_from_counts(
                barrier.name,
                total=len(selected),
                positive=positive,
                negative=negative,
            )
        )
    return tuple(rows)


def _prevalence_row_from_counts(
    barrier_name: str,
    *,
    total: int,
    positive: int,
    negative: int,
) -> PrevalenceRow:
    available = positive + negative
    return PrevalenceRow(
        barrier_name=barrier_name,
        total=total,
        available=available,
        positive=positive,
        negative=negative,
        unavailable=total - available,
        positive_rate=(Decimal(positive) / Decimal(available) if available else None),
    )


def _audit(
    labels: tuple[EpisodeLabel, ...],
    splits: SplitPlan,
) -> LeakageAudit:
    violations: list[str] = []
    available = tuple(
        label for label in labels if label.outcome is not LabelOutcome.UNAVAILABLE
    )
    causal_volatility = all(
        label.volatility_end_time_us is not None
        and label.volatility_end_time_us <= label.candidate_time_us
        for label in available
    )
    if not causal_volatility:
        violations.append("NON_CAUSAL_VOLATILITY")
    next_observation_entry = all(
        label.entry_time_us is not None
        and label.entry_time_us > label.candidate_time_us
        for label in available
    )
    if not next_observation_entry:
        violations.append("ENTRY_NOT_AFTER_CANDIDATE")
    point_in_time_eligibility = all(
        label.eligibility_effective_time_us is not None
        and label.eligibility_effective_time_us <= label.candidate_time_us
        for label in available
    )
    if not point_in_time_eligibility:
        violations.append("FUTURE_ELIGIBILITY")
    purged_chronology = all(
        fold.train_times_us
        and fold.validation_times_us
        and max(fold.train_times_us) + splits.maximum_horizon_us
        < min(fold.validation_times_us)
        for fold in splits.folds
    ) and all(
        value + splits.maximum_horizon_us < splits.final_holdout_times_us[0]
        for fold in splits.folds
        for value in fold.train_times_us + fold.validation_times_us
    )
    if not purged_chronology:
        violations.append("SPLIT_HORIZON_OVERLAP")
    return LeakageAudit(
        causal_volatility=causal_volatility,
        next_observation_entry=next_observation_entry,
        point_in_time_eligibility=point_in_time_eligibility,
        purged_chronology=purged_chronology,
        detector_feature_dependency=False,
        violations=tuple(violations),
    )


def _instrument_mapping(instrument: InstrumentId) -> dict[str, str]:
    return {
        "environment": instrument.environment.value,
        "market": instrument.venue.market,
        "symbol": instrument.symbol,
        "venue": instrument.venue.name,
    }


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)
