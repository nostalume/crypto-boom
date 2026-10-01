"""Causal minute features shared by research and an eventual online scorer.

No outcome labels, forward shifts, training or transport effects belong here.
Input is one admitted Binance-compatible symbol; times denote completed bars.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_boom.bars import MINUTE_US, admit_bars
from crypto_boom.bars import SOURCE_COLUMNS as SOURCE_COLUMNS

PRICE_FEATURES = (
    "return_1",
    "return_5",
    "return_15",
    "return_60",
    "return_360",
    "volatility_60",
    "volatility_360",
    "drawdown_60",
    "drawdown_360",
    "range_60",
    "close_location",
    "return_previous_5",
)
FLOW_FEATURES = (
    "activity_5",
    "activity_60",
    "buy_share_5",
    "buy_share_60",
    "buy_share_change",
    "trade_activity_5",
    "log_turnover_1440",
)


PATH_FEATURES = (
    "path_efficiency_15",
    "up_impulse_share_15",
    "positive_fraction_15",
    "close_position_60",
    "volatility_compression",
    "scaled_return_5",
)
DYNAMICS_FEATURES = (
    "turnover_change_5",
    "buy_share_previous_5",
    "buy_acceleration_5",
    "price_acceleration_5",
    "flow_price_agreement_15",
    "trade_size_change_5",
)
MARKET_FEATURES = (
    "peer_return_5",
    "relative_return_5",
    "relative_return_60",
    "peer_positive_fraction_15",
    "peer_dispersion_15",
)

SEQUENCE_CHANNELS = (
    "bucket_return",
    "bucket_range",
    "log_turnover",
    "log_trades",
    "buy_share",
    "observed_fraction",
)


def past_sequence(
    source: pl.DataFrame,
    origins: pl.DataFrame,
    *,
    history_minutes: int = 360,
    step_minutes: int = 5,
) -> pl.DataFrame:
    """Past-only, oldest-first nonoverlapping buckets ending at each decision.

    Return/range retain absolute price amplitude; flows are log1p sums, buy share
    is turnover-weighted (neutral 0.5 for zero flow), and observed fraction counts
    minutes with trades AND turnover. Zero activity is retained, never imputed.
    Requires history_minutes + 1 consecutive valid bars (the extra close anchors
    the first return). Rejects missing/invalid history rather than filling it.
    Keys preserve caller order; six fixed-size float32 arrays support batch and
    single-origin inference. No targets, fitting, transport or persistence.
    """
    if (
        type(history_minutes) is not int
        or type(step_minutes) is not int
        or not 1 <= step_minutes <= history_minutes <= 1440
        or history_minutes % step_minutes
    ):
        raise ValueError("invalid sequence sampling policy")
    width = history_minutes // step_minutes
    if len(source) > 2_000_000 or len(origins) * width * 6 > 10_000_000:
        raise ValueError("sequence exceeds source/output budget; split the batch")
    source = admit_bars(source)
    keys = origins.select("symbol", "decision_us")
    if (
        keys["decision_us"].dtype != pl.Int64
        or keys.null_count().row(0) != (0, 0)
        or keys.select(pl.struct("symbol", "decision_us").n_unique()).item()
        != len(keys)
        or (len(keys) and set(keys["symbol"]) != set(source["symbol"]))
    ):
        raise ValueError("invalid sequence origin keys")
    times = source["open_time"].dt.epoch("us").to_numpy() + MINUTE_US
    requested = keys["decision_us"].to_numpy()
    positions = np.searchsorted(times, requested)
    if np.any(positions >= len(times)) or np.any(positions < history_minutes):
        raise ValueError("sequence origin lacks history or source observation")
    valid = (
        (source["quality_complete"] & (source["quality_state"] == "valid"))
        .fill_null(False)
        .to_numpy()
    )
    bad = np.r_[0, np.cumsum(~valid)]
    if (
        not np.array_equal(times[positions], requested)
        or np.any(
            times[positions] - times[positions - history_minutes]
            != history_minutes * MINUTE_US
        )
        or np.any(bad[positions + 1] - bad[positions - history_minutes])
    ):
        raise ValueError("sequence history contains a gap, invalid bar or origin")
    q = pl.col("quote_turnover").rolling_sum(step_minutes)
    buckets = source.select(
        (pl.col("close_price") / pl.col("close_price").shift(step_minutes) - 1).alias(
            "bucket_return"
        ),
        (
            pl.col("high_price").rolling_max(step_minutes)
            / pl.col("low_price").rolling_min(step_minutes)
            - 1
        ).alias("bucket_range"),
        q.log1p().alias("log_turnover"),
        pl.col("trade_count").rolling_sum(step_minutes).log1p().alias("log_trades"),
        pl.when(q > 0)
        .then(pl.col("taker_buy_quote_turnover").rolling_sum(step_minutes) / q)
        .otherwise(0.5)
        .alias("buy_share"),
        ((pl.col("trade_count") > 0) & (pl.col("quote_turnover") > 0))
        .cast(pl.Float64)
        .rolling_mean(step_minutes)
        .alias("observed_fraction"),
    )
    indices = positions[:, None] + np.arange(
        -history_minutes + step_minutes, 1, step_minutes
    )
    arrays = []
    for name in SEQUENCE_CHANNELS:
        values = buckets[name].to_numpy()[indices].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite sequence values")
        arrays.append(pl.Series(name, values, dtype=pl.Array(pl.Float32, width)))
    return keys.with_columns(arrays)


@dataclass(frozen=True)
class SequenceRecipe:
    """Version-1 numeric recipe. Window counts are measured in completed buckets."""

    history_minutes: int
    step_minutes: int
    return_steps: tuple[int, ...]
    volatility_steps: tuple[int, ...]
    activity_reference_steps: int
    buy_steps: int
    observed_steps: int

    def __post_init__(self) -> None:
        if (
            type(self.history_minutes) is not int
            or type(self.step_minutes) is not int
            or not 1 <= self.step_minutes <= self.history_minutes <= 1440
            or self.history_minutes % self.step_minutes
        ):
            raise ValueError("invalid sequence recipe scale")
        width = self.history_minutes // self.step_minutes
        for windows, minimum in ((self.return_steps, 1), (self.volatility_steps, 2)):
            if (
                not isinstance(windows, tuple)
                or tuple(sorted(set(windows))) != windows
                or any(type(w) is not int or not minimum <= w <= width for w in windows)
            ):
                raise ValueError("invalid sequence recipe windows")
        if any(
            type(w) is not int or not 1 <= w <= width
            for w in (self.buy_steps, self.observed_steps)
        ):
            raise ValueError("invalid flow windows")
        if (
            type(self.activity_reference_steps) is not int
            or not 1 <= self.activity_reference_steps < width
        ):
            raise ValueError("activity reference requires preceding buckets")

    @property
    def feature_count(self) -> int:
        return (
            6 * (self.history_minutes // self.step_minutes)
            + len(self.return_steps)
            + len(self.volatility_steps)
            + 3
        )


def sequence_matrix(
    source: pl.DataFrame, origins: pl.DataFrame, recipe: SequenceRecipe
) -> np.ndarray:
    """Shared numerical transformation; model-specific scale is supplied as data."""
    buckets = past_sequence(
        source,
        origins,
        history_minutes=recipe.history_minutes,
        step_minutes=recipe.step_minutes,
    )
    r = buckets["bucket_return"].to_numpy().astype(float)
    q = np.expm1(buckets["log_turnover"].to_numpy().astype(float))
    buy = buckets["buy_share"].to_numpy().astype(float)
    w = recipe.buy_steps
    context = np.column_stack(
        [np.prod(1 + r[:, -n:], axis=1) - 1 for n in recipe.return_steps]
        + [np.std(np.log1p(r[:, -n:]), axis=1, ddof=1) for n in recipe.volatility_steps]
        + [
            np.log(
                (q[:, -1] + 1)
                / (q[:, -recipe.activity_reference_steps - 1 : -1].mean(axis=1) + 1)
            ),
            np.divide(
                (q[:, -w:] * buy[:, -w:]).sum(axis=1),
                q[:, -w:].sum(axis=1),
                out=np.full(len(q), 0.5),
                where=q[:, -w:].sum(axis=1) > 0,
            ),
            buckets["observed_fraction"]
            .to_numpy()[:, -recipe.observed_steps :]
            .mean(axis=1),
        ]
    )
    matrix = np.column_stack(
        [buckets[c].to_numpy() for c in SEQUENCE_CHANNELS] + [context]
    ).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite sequence features")
    return matrix


def iter_feature_segments(
    source: pl.DataFrame, *, rich: bool = False
) -> Iterator[pl.DataFrame]:
    """Yield causal features at every minute, separately across quality gaps."""
    if source["symbol"].n_unique() != 1:
        raise ValueError("forecast feature input must contain exactly one symbol")
    source = source.sort("open_time")
    if source["open_time"].n_unique() != len(source):
        raise ValueError("duplicate symbol-minute in forecast source")
    source = source.filter(
        pl.col("quality_complete") & (pl.col("quality_state") == "valid")
    ).with_columns(
        pl.col("open_time").dt.epoch("us").alias("open_us"),
        pl.col(
            "close_price",
            "high_price",
            "low_price",
            "quote_turnover",
            "taker_buy_quote_turnover",
            "trade_count",
        ).cast(pl.Float64),
    )
    source = source.with_columns(
        (pl.col("open_us").diff() != MINUTE_US)
        .fill_null(True)
        .cum_sum()
        .alias("segment")
    )
    for frame in source.partition_by("segment", maintain_order=True):
        if len(frame) < 1441:
            continue
        c = pl.col("close_price")
        q = pl.col("quote_turnover")
        buy = pl.col("taker_buy_quote_turnover")
        trades = pl.col("trade_count")
        frame = (
            frame.with_columns(
                (pl.col("open_us") + MINUTE_US).alias("decision_us"),
                (c / c.shift(1)).log().alias("log_return"),
                q.rolling_sum(1440).alias("turnover_1440"),
                q.rolling_sum(5).alias("q5"),
                q.rolling_sum(60).alias("q60"),
                buy.rolling_sum(5).alias("b5"),
                buy.rolling_sum(60).alias("b60"),
            )
            .with_columns(
                *[
                    (c / c.shift(w) - 1).alias(f"return_{w}")
                    for w in (1, 5, 15, 60, 360)
                ],
                *[
                    pl.col("log_return").rolling_std(w).alias(f"volatility_{w}")
                    for w in (60, 360)
                ],
                *[(c / c.rolling_max(w) - 1).alias(f"drawdown_{w}") for w in (60, 360)],
                (
                    pl.col("high_price").rolling_max(60)
                    / pl.col("low_price").rolling_min(60)
                    - 1
                ).alias("range_60"),
                pl.when(pl.col("high_price") > pl.col("low_price"))
                .then(
                    (c - pl.col("low_price"))
                    / (pl.col("high_price") - pl.col("low_price"))
                )
                .otherwise(0.5)
                .alias("close_location"),
                (c.shift(5) / c.shift(10) - 1).alias("return_previous_5"),
                ((pl.col("q5") + 1) / (q.shift(5).rolling_mean(60) * 5 + 1))
                .log()
                .alias("activity_5"),
                ((pl.col("q60") + 1) / (q.shift(60).rolling_mean(1380) * 60 + 1))
                .log()
                .alias("activity_60"),
                (
                    (trades.rolling_sum(5) + 1)
                    / (trades.shift(5).rolling_mean(60) * 5 + 1)
                )
                .log()
                .alias("trade_activity_5"),
                *[
                    pl.when(pl.col(f"q{w}") > 0)
                    .then(pl.col(f"b{w}") / pl.col(f"q{w}"))
                    .otherwise(0.5)
                    .alias(f"buy_share_{w}")
                    for w in (5, 60)
                ],
                pl.col("turnover_1440").log1p().alias("log_turnover_1440"),
            )
            .with_columns(
                (pl.col("buy_share_5") - pl.col("buy_share_60")).alias(
                    "buy_share_change"
                ),
            )
        )
        if rich:
            frame = _enrich_path_and_activity(frame)
        yield frame


def _enrich_path_and_activity(frame: pl.DataFrame) -> pl.DataFrame:
    c = pl.col("close_price")
    r = pl.col("log_return")
    q = pl.col("quote_turnover")
    buy = pl.col("taker_buy_quote_turnover")
    trades = pl.col("trade_count")
    frame = frame.with_columns(
        r.abs().rolling_sum(15).alias("_travel15"),
        r.clip(lower_bound=0).rolling_sum(15).alias("_up15"),
        c.rolling_max(60).alias("_max60"),
        c.rolling_min(60).alias("_min60"),
        pl.when(q.rolling_sum(15) > 0)
        .then(buy.rolling_sum(15) / q.rolling_sum(15))
        .otherwise(0.5)
        .alias("_buy15"),
    )
    return frame.with_columns(
        pl.when(pl.col("_travel15") > 0)
        .then(r.rolling_sum(15) / pl.col("_travel15"))
        .otherwise(0.0)
        .alias("path_efficiency_15"),
        pl.when(pl.col("_up15") > 0)
        .then(r.clip(lower_bound=0).rolling_max(15) / pl.col("_up15"))
        .otherwise(0.0)
        .alias("up_impulse_share_15"),
        (r > 0).cast(pl.Float64).rolling_mean(15).alias("positive_fraction_15"),
        pl.when(pl.col("_max60") > pl.col("_min60"))
        .then((c - pl.col("_min60")) / (pl.col("_max60") - pl.col("_min60")))
        .otherwise(0.5)
        .alias("close_position_60"),
        (r.rolling_std(15) / (r.shift(15).rolling_std(60) + 1e-8)).alias(
            "volatility_compression"
        ),
        ((c / c.shift(5)).log() / (pl.col("volatility_60") * (5**0.5) + 1e-6)).alias(
            "scaled_return_5"
        ),
        ((pl.col("q5") + 1) / (pl.col("q5").shift(5) + 1))
        .log()
        .alias("turnover_change_5"),
        pl.col("buy_share_5").shift(5).alias("buy_share_previous_5"),
        (pl.col("buy_share_5") - pl.col("buy_share_5").shift(5)).alias(
            "buy_acceleration_5"
        ),
        (pl.col("return_5") - pl.col("return_previous_5")).alias(
            "price_acceleration_5"
        ),
        ((2 * pl.col("_buy15") - 1) * pl.col("return_15")).alias(
            "flow_price_agreement_15"
        ),
        (
            ((pl.col("q5") + 1) / (trades.rolling_sum(5) + 1))
            / ((q.shift(5).rolling_sum(60) + 1) / (trades.shift(5).rolling_sum(60) + 1))
        )
        .log()
        .alias("trade_size_change_5"),
    )


def add_peer_context(rows: pl.DataFrame, *, minimum_peers: int = 20) -> pl.DataFrame:
    """Leave-one-out same-clock observed-cohort context, never market-wide facts.

    Only input rows admitted from as-of source coverage enter peer membership.
    Insufficient peer sets yield nulls; caller reports coverage/refusals.
    """
    if minimum_peers < 1:
        raise ValueError("minimum peer count must be positive")
    if rows.select(pl.struct("symbol", "decision_us").n_unique()).item() != len(rows):
        raise ValueError("duplicate peer symbol-time")
    totals = rows.group_by("decision_us").agg(
        pl.len().alias("_n"),
        *[pl.col(f"return_{w}").sum().alias(f"_sum{w}") for w in (5, 60)],
        (pl.col("return_15") > 0).sum().alias("_positive15"),
        pl.col("return_15").sum().alias("_sum15"),
        (pl.col("return_15") ** 2).sum().alias("_squares15"),
    )
    joined = rows.join(totals, on="decision_us", how="left")
    n = pl.col("_n").cast(pl.Int64) - 1
    mean15 = (pl.col("_sum15") - pl.col("return_15")) / n
    joined = joined.with_columns(
        n.alias("peer_count"),
        ((pl.col("_sum5") - pl.col("return_5")) / n).alias("peer_return_5"),
        (pl.col("return_5") - (pl.col("_sum5") - pl.col("return_5")) / n).alias(
            "relative_return_5"
        ),
        (pl.col("return_60") - (pl.col("_sum60") - pl.col("return_60")) / n).alias(
            "relative_return_60"
        ),
        (
            (
                pl.col("_positive15").cast(pl.Int64)
                - (pl.col("return_15") > 0).cast(pl.Int64)
            )
            / n
        ).alias("peer_positive_fraction_15"),
        (
            ((pl.col("_squares15") - pl.col("return_15") ** 2) / n - mean15**2)
            .clip(lower_bound=0)
            .sqrt()
        ).alias("peer_dispersion_15"),
    )
    return joined.with_columns(
        *[
            pl.when(pl.col("peer_count") >= minimum_peers)
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
            for name in MARKET_FEATURES
        ]
    ).drop("_n", "_sum5", "_sum60", "_positive15", "_sum15", "_squares15")
