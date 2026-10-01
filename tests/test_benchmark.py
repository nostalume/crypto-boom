from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import sys
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import crypto_boom.evaluation.benchmark as benchmark_module
from crypto_boom.evaluation.benchmark import (
    BarrierDefinition,
    BenchmarkError,
    BenchmarkResult,
    BenchmarkSpec,
    CostModel,
    EligibilitySnapshot,
    LabelExit,
    LabelOutcome,
    SplitModel,
    VolatilityModel,
    benchmark_v1_spec,
    build_benchmark,
    chronological_purged_splits,
)
from crypto_boom.market import (
    Environment,
    EpochTimestamp,
    InstrumentId,
    KlineEvent,
    LocalReceipt,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    QualityState,
    SourceDescriptor,
    TimeUnit,
    VenueId,
)

MINUTE_US = 60_000_000
INSTRUMENT = InstrumentId(
    venue=VenueId("binance", "spot"),
    environment=Environment.PRODUCTION,
    symbol="ETHUSDT",
)
QUALITY = ObservationQuality(QualityState.VALID, complete=True)
RUN_ID = UUID("00000000-0000-0000-0000-000000000789")


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        version="test-benchmark-v1",
        interval_us=MINUTE_US,
        volatility=VolatilityModel("test-rms-v1", 3, 2),
        costs=CostModel("zero-cost-v1", Decimal(0), Decimal(0)),
        barriers=(
            BarrierDefinition(
                "h2-u2-d2",
                horizon_minutes=2,
                upper_sigma=Decimal(2),
                lower_sigma=Decimal(2),
            ),
        ),
        splits=SplitModel("test-splits-v1", 2, Decimal("0.20")),
    )


def _bars(
    count: int = 30,
    *,
    highs: dict[int, str] | None = None,
    lows: dict[int, str] | None = None,
) -> tuple[KlineEvent, ...]:
    highs = highs or {}
    lows = lows or {}
    rows = []
    for index in range(count):
        price = Decimal("100") if index % 2 == 0 else Decimal("101")
        high = Decimal(highs.get(index, str(price + Decimal("0.1"))))
        low = Decimal(lows.get(index, str(price - Decimal("0.1"))))
        open_time = index * MINUTE_US
        rows.append(
            KlineEvent(
                instrument=INSTRUMENT,
                raw_symbol="ETHUSDT",
                interval="1m",
                source_event_time=None,
                open_time=EpochTimestamp(open_time, TimeUnit.MICROSECOND),
                close_time=EpochTimestamp(
                    open_time + MINUTE_US - 1,
                    TimeUnit.MICROSECOND,
                ),
                open_price=price,
                high_price=max(high, price),
                low_price=min(low, price),
                close_price=price,
                base_volume=Decimal("10"),
                quote_turnover=price * Decimal("10"),
                taker_buy_base_volume=Decimal("5"),
                taker_buy_quote_turnover=price * Decimal("5"),
                trade_count=10,
                first_trade_id=None,
                last_trade_id=None,
                closed=True,
                provenance=Provenance(
                    source=SourceDescriptor("fixture", "klines", 1),
                    ingestion_run_id=RUN_ID,
                    receipt=LocalReceipt(index + 1, index + 1),
                    payload_digest=PayloadDigest.sha256(f"bar-{index}".encode()),
                    source_revision="fixture-v1",
                    raw_payload_reference=f"bar-{index}",
                ),
                quality=QUALITY,
            )
        )
    return tuple(rows)


def _eligibility(
    *extra: EligibilitySnapshot,
) -> tuple[EligibilitySnapshot, ...]:
    return (
        EligibilitySnapshot(
            instrument=INSTRUMENT,
            effective_time_us=0,
            eligible=True,
            evidence_id="eligibility-0",
            reason="QUALIFIED_FIXTURE",
        ),
        *extra,
    )


def _label_at(result: BenchmarkResult, candidate_time_us: int):
    labels = result.labels
    return next(
        label for label in labels if label.candidate_time_us == candidate_time_us
    )


def _columnar_frame(bars: tuple[KlineEvent, ...]):
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {
            "venue": [bar.instrument.venue.name for bar in bars],
            "market": [bar.instrument.venue.market for bar in bars],
            "environment": [bar.instrument.environment.value for bar in bars],
            "symbol": [bar.instrument.symbol for bar in bars],
            "interval": [bar.interval for bar in bars],
            "open_time": [bar.open_time.epoch_microseconds for bar in bars],
            "close_time": [bar.close_time.epoch_microseconds for bar in bars],
            "open_price": [bar.open_price for bar in bars],
            "high_price": [bar.high_price for bar in bars],
            "low_price": [bar.low_price for bar in bars],
            "close_price": [bar.close_price for bar in bars],
            "quality_state": [bar.quality.state.value for bar in bars],
            "quality_complete": [bar.quality.complete for bar in bars],
        },
        schema_overrides={
            name: pl.Decimal(38, 18)
            for name in (
                "open_price",
                "high_price",
                "low_price",
                "close_price",
            )
        },
    )


def _columnar_eligibility(snapshots: tuple[EligibilitySnapshot, ...]):
    pl = pytest.importorskip("polars")
    return pl.DataFrame(
        {
            "venue": [item.instrument.venue.name for item in snapshots],
            "market": [item.instrument.venue.market for item in snapshots],
            "environment": [item.instrument.environment.value for item in snapshots],
            "symbol": [item.instrument.symbol for item in snapshots],
            "effective_time_us": [item.effective_time_us for item in snapshots],
            "eligible": [item.eligible for item in snapshots],
            "evidence_id": [item.evidence_id for item in snapshots],
            "reason": [item.reason for item in snapshots],
        },
        schema={
            "venue": pl.String,
            "market": pl.String,
            "environment": pl.String,
            "symbol": pl.String,
            "effective_time_us": pl.Int64,
            "eligible": pl.Boolean,
            "evidence_id": pl.String,
            "reason": pl.String,
        },
    )


def test_upper_barrier_uses_next_observation_and_causal_sigma() -> None:
    bars = _bars(highs={4: "110"})
    result = build_benchmark(bars, _eligibility(), _spec())
    candidate = bars[3]
    label = _label_at(result, candidate.close_time.epoch_microseconds)

    assert label.outcome is LabelOutcome.POSITIVE
    assert label.exit is LabelExit.UPPER_BARRIER
    assert label.entry_time_us == bars[4].open_time.epoch_microseconds
    assert label.entry_time_us > label.candidate_time_us
    assert label.volatility_end_time_us == label.candidate_time_us
    assert label.eligibility_evidence_id == "eligibility-0"


def test_same_bar_upper_and_lower_hit_is_conservatively_negative() -> None:
    bars = _bars(highs={4: "110"}, lows={4: "90"})
    result = build_benchmark(bars, _eligibility(), _spec())
    label = _label_at(result, bars[3].close_time.epoch_microseconds)

    assert label.outcome is LabelOutcome.NEGATIVE
    assert label.exit is LabelExit.LOWER_BARRIER


def test_future_eligibility_and_future_prices_do_not_change_candidate_inputs() -> None:
    bars = _bars(highs={4: "110"})
    candidate_time = bars[3].close_time.epoch_microseconds
    future_snapshot = EligibilitySnapshot(
        instrument=INSTRUMENT,
        effective_time_us=bars[10].close_time.epoch_microseconds,
        eligible=False,
        evidence_id="future-halt",
        reason="FUTURE_HALT",
    )
    first = build_benchmark(bars, _eligibility(future_snapshot), _spec())

    changed = list(bars)
    for index in range(10, len(changed)):
        price = Decimal(200 + index)
        changed[index] = replace(
            changed[index],
            open_price=price,
            high_price=price + Decimal(1),
            low_price=price - Decimal(1),
            close_price=price,
            quote_turnover=price * Decimal(10),
            taker_buy_quote_turnover=price * Decimal(5),
        )
    second = build_benchmark(tuple(changed), _eligibility(future_snapshot), _spec())
    first_label = _label_at(first, candidate_time)
    second_label = _label_at(second, candidate_time)

    assert first_label.sigma == second_label.sigma
    assert first_label.eligibility_evidence_id == "eligibility-0"
    assert second_label.eligibility_evidence_id == "eligibility-0"


def test_missing_history_eligibility_and_horizon_remain_unavailable() -> None:
    result = build_benchmark(_bars(), (), _spec())

    assert all(label.outcome is LabelOutcome.UNAVAILABLE for label in result.labels)
    assert {label.exit for label in result.labels} == {LabelExit.MISSING_ELIGIBILITY}
    assert not result.report.ready


@pytest.mark.parametrize("policy_name", ["EXACT_DECIMAL_V1", "FLOAT64_ULP16_V1"])
def test_coverage_conditioned_labels_do_not_claim_historical_eligibility(
    policy_name: str,
) -> None:
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        _to_benchmark_results,
        build_coverage_conditioned_labels,
    )

    bars = _bars(highs={4: "110"})
    result = build_coverage_conditioned_labels(
        _columnar_frame(bars),
        _spec(),
        cohort_id="fixture-coverage-cohort",
        numerical_policy=ColumnarNumericalPolicy[policy_name],
    )
    candidate_time = bars[3].close_time.epoch_microseconds
    candidate = result.labels.filter(
        result.labels["candidate_time_us"] == candidate_time
    ).row(0, named=True)

    assert result.evidence_class == "coverage_conditioned"
    assert result.cohort_id == "fixture-coverage-cohort"
    assert candidate["outcome"] == LabelOutcome.POSITIVE.value
    assert candidate["sample_scope"] == result.evidence_class
    assert result.labels["sample_scope"].unique().to_list() == ["coverage_conditioned"]
    assert "eligibility_evidence_id" not in result.labels.columns
    assert "executable_return" not in result.labels.columns
    with pytest.raises(BenchmarkError, match="cannot become BEN reports"):
        _to_benchmark_results(result.labels, _spec())


@pytest.mark.parametrize("exact", [False, True])
def test_coverage_conditioned_candidate_labels_match_full_label_slice(
    exact: bool,
) -> None:
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        build_coverage_conditioned_candidate_labels,
        build_coverage_conditioned_labels,
    )

    policy = (
        ColumnarNumericalPolicy.EXACT_DECIMAL_V1
        if exact
        else ColumnarNumericalPolicy.FLOAT64_ULP16_V1
    )
    frame = _columnar_frame(_bars(highs={4: "110"}, lows={8: "90"}))
    candidates = (
        frame[[0, 3, 7, len(frame) - 1]]
        .select(
            "venue",
            "market",
            "environment",
            "symbol",
            "interval",
            "close_time",
        )
        .rename({"close_time": "candidate_time_us"})
    )
    full = build_coverage_conditioned_labels(
        frame,
        _spec(),
        cohort_id="fixture-candidate-cohort",
        numerical_policy=policy,
    )

    compact = build_coverage_conditioned_candidate_labels(
        frame,
        candidates,
        _spec(),
        cohort_id="fixture-candidate-cohort",
        numerical_policy=policy,
    )

    expected = full.labels.join(
        candidates,
        on=[
            "venue",
            "market",
            "environment",
            "symbol",
            "interval",
            "candidate_time_us",
        ],
        how="semi",
    )
    assert compact.labels.equals(expected)
    assert len(compact.labels) == len(candidates) * len(_spec().barriers)

    unmatched = candidates.with_columns(
        (candidates["candidate_time_us"] + 1).alias("candidate_time_us")
    )
    with pytest.raises(BenchmarkError, match="match one cohort row"):
        build_coverage_conditioned_candidate_labels(
            frame,
            unmatched,
            _spec(),
            cohort_id="fixture-candidate-cohort",
            numerical_policy=policy,
        )


def test_walk_forward_splits_purge_every_label_horizon() -> None:
    times = tuple(index * MINUTE_US for index in range(60))
    model = SplitModel("split-v1", 3, Decimal("0.20"))

    first = chronological_purged_splits(
        times,
        maximum_horizon_us=2 * MINUTE_US,
        model=model,
    )
    second = chronological_purged_splits(
        times,
        maximum_horizon_us=2 * MINUTE_US,
        model=model,
    )

    assert first == second
    assert first.split_id == second.split_id
    assert len(first.folds) == 3
    holdout_start = first.final_holdout_times_us[0]
    for fold in first.folds:
        assert max(fold.train_times_us) + 2 * MINUTE_US < min(fold.validation_times_us)
        assert max(fold.validation_times_us) + 2 * MINUTE_US < holdout_start


def test_versions_are_immutable_and_content_addressed() -> None:
    spec = benchmark_v1_spec()
    same = benchmark_v1_spec()
    changed = replace(
        spec,
        costs=replace(spec.costs, entry_bps=Decimal("16")),
    )

    assert spec.spec_id == (
        "sha256:56f1bc393b72f19be4ebca1d16a01303afd3bf3ece9a99e71321fb48f9538c91"
    )
    assert spec.spec_id == same.spec_id
    assert spec.spec_id != changed.spec_id
    assert spec.costs.model_id != changed.costs.model_id
    with pytest.raises(FrozenInstanceError):
        spec.__setattr__("version", "mutated")


def test_prevalence_report_and_episode_grouping_are_reproducible() -> None:
    bars = _bars(highs={4: "110", 5: "110", 6: "110"})
    first = build_benchmark(bars, _eligibility(), _spec())
    second = build_benchmark(bars, _eligibility(), _spec())

    assert first.to_bytes() == second.to_bytes()
    assert first.report.report_id == second.report.report_id
    assert first.report.prevalence[0].positive > 0
    assert first.report.prevalence[0].available > 0
    assert first.episodes
    assert first.leakage_audit.passed
    assert first.report.ready
    assert first.report.computation_ready


def test_benchmark_has_no_replay_or_detector_dependency() -> None:
    tree = ast.parse(inspect.getsource(benchmark_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "crypto_boom.replay" not in imported_modules
    assert "crypto_boom.detector" not in imported_modules


def test_declared_costs_change_executable_barrier_result() -> None:
    bars = _bars(highs={4: "102.5"})
    zero_cost = build_benchmark(bars, _eligibility(), _spec())
    costly_spec = replace(
        _spec(),
        costs=CostModel("costly-v1", Decimal("50"), Decimal("50")),
    )
    costly = build_benchmark(bars, _eligibility(), costly_spec)
    candidate_time = bars[3].close_time.epoch_microseconds

    assert _label_at(zero_cost, candidate_time).outcome is LabelOutcome.POSITIVE
    assert _label_at(costly, candidate_time).outcome is LabelOutcome.NEGATIVE


def test_conflicting_eligibility_time_fails_closed() -> None:
    duplicate = EligibilitySnapshot(
        instrument=INSTRUMENT,
        effective_time_us=0,
        eligible=False,
        evidence_id="eligibility-conflict",
        reason="CONFLICT",
    )

    with pytest.raises(BenchmarkError, match="strictly chronological"):
        build_benchmark(_bars(), _eligibility(duplicate), _spec())


def test_incomplete_grid_and_insufficient_split_data_fail_closed() -> None:
    bars = list(_bars())
    bars.pop(5)
    with pytest.raises(BenchmarkError, match="complete interval grid"):
        build_benchmark(tuple(bars), _eligibility(), _spec())

    with pytest.raises(BenchmarkError, match="not enough candidates"):
        chronological_purged_splits(
            (1, 2, 3),
            maximum_horizon_us=1,
            model=SplitModel("split-v1", 2, Decimal("0.20")),
        )


def test_columnar_exact_barrier_boundary_preserves_conservative_tie() -> None:
    from crypto_boom.evaluation.columnar import (
        _to_benchmark_results,
        build_columnar_labels,
    )

    rows = list(_bars())
    for index, text in enumerate(("100", "101", "102.01")):
        price = Decimal(text)
        rows[index] = replace(
            rows[index],
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
        )
    rows[3] = replace(
        rows[3],
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("98"),
        close_price=Decimal("100"),
    )
    bars = tuple(rows)
    candidate = bars[2]
    scalar = build_benchmark(bars, _eligibility(), _spec())
    assert _label_at(scalar, candidate.close_time.epoch_microseconds).sigma == Decimal(
        "0.01"
    )

    result = _to_benchmark_results(
        build_columnar_labels(
            _columnar_frame(bars),
            _columnar_eligibility(_eligibility()),
            _spec(),
        ),
        _spec(),
    )

    assert result == (scalar,)
    assert _label_at(result[0], candidate.close_time.epoch_microseconds).exit is (
        LabelExit.LOWER_BARRIER
    )


@given(
    marks=st.lists(
        st.sampled_from(("none", "upper", "lower", "both")),
        min_size=4,
        max_size=8,
    ),
    cost_bps=st.sampled_from(("0", "15", "50")),
)
@settings(max_examples=10, deadline=None)
def test_columnar_labels_match_scalar_oracle_across_barrier_paths(
    marks: list[str],
    cost_bps: str,
) -> None:
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        _to_benchmark_results,
        build_columnar_labels,
    )

    highs: dict[int, str] = {}
    lows: dict[int, str] = {}
    for offset, mark in enumerate(marks, start=4):
        if mark in {"upper", "both"}:
            highs[offset] = "110"
        if mark in {"lower", "both"}:
            lows[offset] = "90"
    bars = _bars(highs=highs, lows=lows)
    future = EligibilitySnapshot(
        instrument=INSTRUMENT,
        effective_time_us=bars[20].close_time.epoch_microseconds,
        eligible=False,
        evidence_id="future-ineligible",
        reason="PROPERTY_CUTOFF",
    )
    snapshots = _eligibility(future)
    spec = replace(
        _spec(),
        costs=CostModel(
            f"property-cost-{cost_bps}",
            Decimal(cost_bps),
            Decimal(cost_bps),
        ),
    )

    scalar = build_benchmark(bars, snapshots, spec)
    frame = _columnar_frame(bars)
    history = _columnar_eligibility(snapshots)
    columnar = build_columnar_labels(frame, history, spec)
    float64 = build_columnar_labels(
        frame,
        history,
        spec,
        numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
    )

    assert _to_benchmark_results(columnar, spec) == (scalar,)
    discrete = [
        "candidate_time_us",
        "barrier_name",
        "outcome",
        "exit",
        "exit_time_us",
        "entry_time_us",
        "eligibility_evidence_id",
    ]
    assert float64.select(discrete).equals(columnar.select(discrete))


def test_columnar_labels_partition_multiple_symbols_without_crossing() -> None:
    pl = pytest.importorskip("polars")
    from crypto_boom.evaluation.columnar import (
        _to_benchmark_results,
        build_columnar_labels,
    )

    first = _bars(highs={4: "110"})
    other_instrument = InstrumentId(
        venue=INSTRUMENT.venue,
        environment=INSTRUMENT.environment,
        symbol="SOLUSDT",
    )
    second = tuple(
        replace(bar, instrument=other_instrument, raw_symbol="SOLUSDT")
        for bar in _bars(lows={4: "90"})
    )
    other_snapshot = EligibilitySnapshot(
        instrument=other_instrument,
        effective_time_us=0,
        eligible=True,
        evidence_id="eligibility-sol-0",
        reason="QUALIFIED_FIXTURE",
    )
    snapshots = (*_eligibility(), other_snapshot)
    spec = replace(
        _spec(),
        barriers=(
            *_spec().barriers,
            BarrierDefinition(
                "h5-u3-d1",
                horizon_minutes=5,
                upper_sigma=Decimal(3),
                lower_sigma=Decimal(1),
            ),
        ),
    )

    results = _to_benchmark_results(
        build_columnar_labels(
            _columnar_frame((*second, *first)).with_columns(
                pl.col("open_time").cast(pl.Datetime("us", "UTC")),
                pl.col("close_time").cast(pl.Datetime("us", "UTC")),
            ),
            _columnar_eligibility(snapshots),
            spec,
        ),
        spec,
    )
    expected = (
        build_benchmark(first, _eligibility(), spec),
        build_benchmark(second, (other_snapshot,), spec),
    )

    assert results == expected


def test_columnar_labels_refuse_empty_null_and_incomplete_grids() -> None:
    pl = pytest.importorskip("polars")
    from crypto_boom.evaluation.columnar import build_columnar_labels

    with pytest.raises(BenchmarkError, match="at least one kline"):
        build_columnar_labels(
            _columnar_frame(()),
            _columnar_eligibility(()),
            _spec(),
        )

    bars = _bars()
    null_high = _columnar_frame(bars).with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(None)
        .otherwise(pl.col("high_price"))
        .alias("high_price")
    )
    with pytest.raises(BenchmarkError, match="null"):
        build_columnar_labels(null_high, _columnar_eligibility(_eligibility()), _spec())

    wrong_price_type = _columnar_frame(bars).with_columns(
        pl.col("open_price").cast(pl.Decimal(20, 8))
    )
    with pytest.raises(BenchmarkError, match=r"Decimal\(38, 18\)"):
        build_columnar_labels(
            wrong_price_type,
            _columnar_eligibility(_eligibility()),
            _spec(),
        )

    broken = _columnar_frame((*bars[:5], *bars[6:]))
    with pytest.raises(BenchmarkError, match="complete interval grid"):
        build_columnar_labels(broken, _columnar_eligibility(_eligibility()), _spec())

    missing = build_columnar_labels(
        _columnar_frame(bars),
        _columnar_eligibility(()),
        _spec(),
    )
    assert missing["exit"].unique().to_list() == [LabelExit.MISSING_ELIGIBILITY.value]


def test_float64_columnar_policy_is_identified_and_preserves_discrete_labels() -> None:
    pl = pytest.importorskip("polars")
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        _to_benchmark_results,
        build_columnar_labels,
    )

    exact = build_columnar_labels(
        _columnar_frame(_bars(highs={4: "110"}, lows={8: "90"})),
        _columnar_eligibility(_eligibility()),
        _spec(),
    )
    candidate = build_columnar_labels(
        _columnar_frame(_bars(highs={4: "110"}, lows={8: "90"})),
        _columnar_eligibility(_eligibility()),
        _spec(),
        numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
    )

    assert exact["numerical_policy_id"].unique().to_list() == [
        ColumnarNumericalPolicy.EXACT_DECIMAL_V1.value
    ]
    assert candidate["numerical_policy_id"].unique().to_list() == [
        ColumnarNumericalPolicy.FLOAT64_ULP16_V1.value
    ]
    assert candidate.schema["sigma"] == pl.Float64
    discrete = [
        "candidate_time_us",
        "barrier_name",
        "outcome",
        "exit",
        "exit_time_us",
        "entry_time_us",
        "eligibility_evidence_id",
    ]
    assert candidate.select(discrete).equals(exact.select(discrete))
    for column in (
        "sigma",
        "executable_return",
        "maximum_favorable_return",
        "maximum_adverse_return",
    ):
        error = (candidate[column] - exact[column].cast(pl.Float64)).abs()
        maximum_error = error.drop_nulls().max()
        assert isinstance(maximum_error, float)
        assert maximum_error < 1e-12
        assert not candidate[column].drop_nulls().is_nan().any()
        assert candidate[column].drop_nulls().is_finite().all()
    with pytest.raises(BenchmarkError, match="exact policy"):
        _to_benchmark_results(candidate, _spec())


def test_float64_columnar_policy_refuses_near_threshold_decisions() -> None:
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        build_columnar_labels,
    )

    rows = list(_bars())
    for index, text in enumerate(("100", "101", "102.01")):
        price = Decimal(text)
        rows[index] = replace(
            rows[index],
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
        )
    rows[3] = replace(
        rows[3],
        open_price=Decimal("100"),
        high_price=Decimal("102"),
        low_price=Decimal("98"),
        close_price=Decimal("100"),
    )
    candidate_time = rows[2].close_time.epoch_microseconds

    labels = build_columnar_labels(
        _columnar_frame(tuple(rows)),
        _columnar_eligibility(_eligibility()),
        _spec(),
        numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
    )
    label = labels.filter(candidate_time_us=candidate_time).row(0, named=True)

    assert label["outcome"] == LabelOutcome.UNAVAILABLE.value
    assert label["exit"] == "numerical_ambiguity"
    assert label["sigma"] is not None


def test_float64_admission_refuses_nonzero_returns_below_binary64_resolution() -> None:
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        build_columnar_labels,
    )

    rows = []
    for index, row in enumerate(_bars()):
        price = Decimal("10000000000000000000") + index % 2
        rows.append(
            replace(
                row,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
            )
        )
    frame = _columnar_frame(tuple(rows))
    history = _columnar_eligibility(_eligibility())

    exact = build_columnar_labels(frame, history, _spec())
    assert LabelExit.ZERO_VOLATILITY.value not in exact["exit"].to_list()
    with pytest.raises(BenchmarkError, match="binary64 resolution"):
        build_columnar_labels(
            frame,
            history,
            _spec(),
            numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
        )


def test_float64_policy_preserves_discrete_labels_across_decimal128_price_range() -> (
    None
):
    from crypto_boom.evaluation.benchmark import _label_candidate
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        build_columnar_labels,
    )

    scales = tuple(
        Decimal(value)
        for value in (
            "1e-18",
            "1e-12",
            "1e-6",
            "1",
            "1e6",
            "1e12",
            "1e18",
            "9e19",
        )
    )
    rows = []
    for index, row in enumerate(_bars(count=450)):
        price = scales[index % len(scales)]
        rows.append(
            replace(
                row,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
            )
        )
    frame = _columnar_frame(tuple(rows))
    history = _columnar_eligibility(_eligibility())
    spec = benchmark_v1_spec()
    admitted_rows = tuple(rows)
    snapshots = _eligibility()
    exact_labels = {
        (label.candidate_time_us, label.barrier_name): label
        for index in (60, 61, 62, 80, 89)
        for barrier in spec.barriers
        for label in (
            _label_candidate(
                admitted_rows,
                index,
                snapshots,
                barrier,
                spec,
                spec.spec_id,
            ),
        )
    }
    candidate = build_columnar_labels(
        frame,
        history,
        spec,
        numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
    )
    compared = 0
    for row in candidate.iter_rows(named=True):
        key = (row["candidate_time_us"], row["barrier_name"])
        if key not in exact_labels:
            continue
        if row["exit"] == LabelExit.NUMERICAL_AMBIGUITY.value:
            continue
        label = exact_labels[key]
        assert row["outcome"] == label.outcome.value
        assert row["exit"] == label.exit.value
        assert row["exit_time_us"] == label.exit_time_us
        compared += 1
    assert compared == len(exact_labels)
    for column in (
        "sigma",
        "executable_return",
        "maximum_favorable_return",
        "maximum_adverse_return",
    ):
        assert not candidate[column].drop_nulls().is_nan().any()
        assert candidate[column].drop_nulls().is_finite().all()


def test_partitioned_columnar_summary_matches_full_labels_with_spec_halos() -> None:
    from crypto_boom.evaluation.columnar import (
        build_columnar_labels,
        summarize_columnar_labels,
    )

    bars = _bars(count=31, highs={7: "110", 13: "110"}, lows={19: "90"})
    snapshots = _eligibility(
        EligibilitySnapshot(
            instrument=INSTRUMENT,
            effective_time_us=bars[15].close_time.epoch_microseconds,
            eligible=False,
            evidence_id="eligibility-15",
            reason="TEST_CUTOFF",
        )
    )
    spec = replace(
        _spec(),
        barriers=(
            *_spec().barriers,
            BarrierDefinition(
                "h5-u3-d1",
                horizon_minutes=5,
                upper_sigma=Decimal(3),
                lower_sigma=Decimal(1),
            ),
        ),
    )
    frame = _columnar_frame(bars)
    history = _columnar_eligibility(snapshots)

    labels = build_columnar_labels(frame, history, spec)
    expected = {
        tuple(row[:3]): row[3]
        for row in labels.group_by("barrier_name", "outcome", "exit").len().iter_rows()
    }
    summary = summarize_columnar_labels(
        frame,
        history,
        spec,
        partition_candidate_rows=6,
    )

    assert summary.label_count == len(labels)
    assert summary.instrument_count == 1
    assert summary.partition_count == 6
    assert {
        (row.barrier_name, row.outcome.value, row.exit.value): row.count
        for row in summary.exit_counts
    } == expected
    assert sum(row.total for row in summary.prevalence) == len(labels)

    with pytest.raises(BenchmarkError, match="exceed 50000"):
        summarize_columnar_labels(
            frame,
            history,
            spec,
            partition_candidate_rows=50_001,
        )


def test_verified_source_partitions_stream_with_cross_partition_halos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pl = pytest.importorskip("polars")
    import crypto_boom.evaluation.columnar as columnar
    from crypto_boom.evaluation.columnar import (
        ColumnarNumericalPolicy,
        ColumnarResearchLimits,
        build_columnar_labels,
        build_coverage_conditioned_labels,
        summarize_coverage_conditioned_partitions,
        summarize_research_partitions,
    )
    from crypto_boom.storage.source import (
        RESEARCH_STORAGE_VERSION,
        PublishedResearchPartition,
        ResearchPartitionManifest,
    )

    offset = 1_735_776_000_000_000
    bars = _bars(count=31, highs={7: "110", 13: "110"}, lows={19: "90"})
    frame = _columnar_frame(bars).with_columns(
        (pl.col("open_time") + offset).alias("open_time"),
        (pl.col("close_time") + offset).alias("close_time"),
    )
    history = _columnar_eligibility(_eligibility())
    publications = []
    for index, (start, length) in enumerate(((0, 16), (16, 15)), start=1):
        partition = frame.slice(start, length)
        path = tmp_path / f"partition-{index}"
        path.mkdir()
        parquet = path / "klines.parquet"
        partition.write_parquet(parquet)
        digest = f"sha256:{index:064x}"
        manifest = ResearchPartitionManifest(
            schema_version=1,
            storage_version=RESEARCH_STORAGE_VERSION,
            evidence_class="source_only",
            venue="binance",
            market="spot",
            environment="production",
            dataset="research_source_klines",
            symbol="ETHUSDT",
            interval="1m",
            month=f"2025-0{index}",
            source_manifest_id=digest,
            source_revision=digest,
            decoder_version="fixture-v1",
            row_count=length,
            first_open_time_us=partition["open_time"][0],
            last_open_time_us=partition["open_time"][-1],
            parquet_sha256=digest,
            parquet_bytes=parquet.stat().st_size,
        )
        publications.append(PublishedResearchPartition(path, manifest, True))
    lookup = {item.path.resolve(): item for item in publications}
    monkeypatch.setattr(
        columnar,
        "load_published_research_partition",
        lambda path: lookup[path.resolve()],
    )

    labels = build_columnar_labels(
        frame,
        history,
        _spec(),
        numerical_policy=ColumnarNumericalPolicy.FLOAT64_ULP16_V1,
    )
    summary = summarize_research_partitions(
        tuple(item.path for item in reversed(publications)),
        history,
        _spec(),
        limits=ColumnarResearchLimits(
            maximum_instruments=1,
            maximum_source_partitions=2,
            maximum_rows=31,
        ),
        partition_candidate_rows=6,
    )

    assert summary.label_count == len(labels)
    assert summary.partition_count == 6
    assert summary.instrument_count == 1
    assert summary.source_manifest_ids == tuple(
        item.manifest.manifest_id for item in publications
    )
    assert {
        (row.barrier_name, row.outcome.value, row.exit.value): row.count
        for row in summary.exit_counts
    } == {
        tuple(row[:3]): row[3]
        for row in labels.group_by("barrier_name", "outcome", "exit").len().iter_rows()
    }

    for policy in ColumnarNumericalPolicy:
        coverage = summarize_coverage_conditioned_partitions(
            tuple(item.path for item in reversed(publications)),
            _spec(),
            limits=ColumnarResearchLimits(
                maximum_instruments=1,
                maximum_source_partitions=2,
                maximum_rows=31,
            ),
            numerical_policy=policy,
            partition_candidate_rows=6,
        )
        coverage_labels = build_coverage_conditioned_labels(
            frame,
            _spec(),
            cohort_id="fixture-source-cohort",
            numerical_policy=policy,
        ).labels
        assert coverage.evidence_class == "coverage_conditioned_source_only"
        assert not hasattr(coverage, "prevalence")
        assert coverage.source_manifest_ids == summary.source_manifest_ids
        assert coverage.label_count == len(coverage_labels)
        assert coverage.partition_count == summary.partition_count
        assert {
            (row.barrier_name, row.outcome.value, row.exit.value): row.count
            for row in coverage.exit_counts
        } == {
            tuple(row[:3]): row[3]
            for row in coverage_labels.group_by("barrier_name", "outcome", "exit")
            .len()
            .iter_rows()
        }

    boundary_prices = (
        Decimal("10000000000000000000"),
        Decimal("10000000000000000001"),
    )
    for publication, row_index, price in (
        (publications[0], 15, boundary_prices[0]),
        (publications[1], 0, boundary_prices[1]),
    ):
        parquet = publication.path / "klines.parquet"
        partition = pl.read_parquet(parquet).with_row_index("_row")
        partition.with_columns(
            *(
                pl.when(pl.col("_row") == row_index)
                .then(pl.lit(price, dtype=pl.Decimal(38, 18)))
                .otherwise(pl.col(column))
                .alias(column)
                for column in (
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                )
            )
        ).drop("_row").write_parquet(parquet)
    with pytest.raises(BenchmarkError, match="binary64 resolution"):
        summarize_research_partitions(
            tuple(item.path for item in publications),
            history,
            _spec(),
            limits=ColumnarResearchLimits(
                maximum_instruments=1,
                maximum_source_partitions=2,
                maximum_rows=31,
            ),
        )

    reads = 0
    read_parquet = pl.read_parquet

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return read_parquet(*args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", counted_read)
    with pytest.raises(BenchmarkError, match="row limit"):
        summarize_research_partitions(
            tuple(item.path for item in publications),
            history,
            _spec(),
            limits=ColumnarResearchLimits(
                maximum_instruments=1,
                maximum_source_partitions=2,
                maximum_rows=10,
            ),
        )
    assert reads == 0
    defaults = ColumnarResearchLimits()
    assert (
        defaults.maximum_instruments,
        defaults.maximum_source_partitions,
        defaults.maximum_rows,
        defaults.maximum_parquet_bytes,
        defaults.maximum_elapsed_seconds,
    ) == (4, 8, 400_000, 64 * 1_024 * 1_024, 30.0)
    with pytest.raises(ValueError, match="exceeds 500"):
        ColumnarResearchLimits(maximum_instruments=501)


def test_columnar_and_cli_help_import_without_polars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "polars" or name.startswith("polars."):
            raise ImportError("blocked optional research dependency")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop("crypto_boom.evaluation.columnar", None)
    monkeypatch.setattr(builtins, "__import__", blocked)
    importlib.import_module("crypto_boom.evaluation.columnar")
    cli = importlib.import_module("crypto_boom.cli")

    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
