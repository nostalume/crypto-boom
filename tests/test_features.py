"""Reference-group feature invariants used by historical research."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from crypto_boom.bars import MINUTE_US
from crypto_boom.features import (
    MARKET_FEATURES,
    SEQUENCE_CHANNELS,
    add_peer_context,
    past_sequence,
)
from test_forward_prediction import BASE, bars


def test_source_columns_import_remains_compatible():
    from crypto_boom.bars import SOURCE_COLUMNS as canonical
    from crypto_boom.features import SOURCE_COLUMNS

    assert SOURCE_COLUMNS is canonical


def peers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "decision_us": [0, 0, 0],
            "return_5": [0.1, 0.2, -0.1],
            "return_15": [0.1, 0.2, -0.1],
            "return_60": [0.3, 0.4, 0.2],
        }
    )


def test_reference_excludes_self_and_is_not_affected_by_future_rows() -> None:
    rows = peers()
    context = add_peer_context(rows, minimum_peers=2).sort("symbol")
    a = context.filter(pl.col("symbol") == "A")
    assert a["peer_return_5"].item() == pytest.approx(0.05)
    assert a["peer_dispersion_15"].item() == pytest.approx(0.15)
    assert a["peer_positive_fraction_15"].item() == 0.5
    changed = rows.with_columns(
        pl.when(pl.col("symbol") == "A")
        .then(5.0)
        .otherwise(pl.col("return_5"))
        .alias("return_5")
    )
    assert add_peer_context(changed, minimum_peers=2).filter(pl.col("symbol") == "A")[
        "peer_return_5"
    ].item() == pytest.approx(0.05)
    future = changed.with_columns(
        pl.lit(300000000, dtype=pl.Int64).alias("decision_us")
    )
    combined = add_peer_context(pl.concat([rows, future]), minimum_peers=2)
    assert_frame_equal(
        context, combined.filter(pl.col("decision_us") == 0).sort("symbol")
    )


def test_insufficient_peers_remain_unavailable_and_duplicates_are_refused() -> None:
    context = add_peer_context(peers())
    assert context.select(MARKET_FEATURES).null_count().row(0) == (3,) * len(
        MARKET_FEATURES
    )
    with pytest.raises(ValueError, match="duplicate"):
        add_peer_context(pl.concat([peers(), peers().head(1)]))


def test_sequence_arithmetic_order_and_prefix_invariance():
    source = bars(800)
    keys = pl.DataFrame(
        {
            "symbol": ["AAAUSDT"] * 2,
            "decision_us": [BASE + 720 * MINUTE_US, BASE + 361 * MINUTE_US],
        }
    )
    result = past_sequence(source, keys)
    assert_frame_equal(result, past_sequence(source.head(720), keys))
    assert result.select("symbol", "decision_us").equals(keys)
    for row, pos in enumerate((719, 360)):
        for bucket, end in enumerate(range(pos - 355, pos + 1, 5)):
            frame = source.slice(end - 4, 5)
            expected = [
                source["close_price"][end] / source["close_price"][end - 5] - 1,
                frame["high_price"].max() / frame["low_price"].min() - 1,
                np.log1p(frame["quote_turnover"].sum()),
                np.log1p(frame["trade_count"].sum()),
                0.5,
                1.0,
            ]
            for channel, value in zip(SEQUENCE_CHANNELS, expected, strict=True):
                assert result[channel][row][bucket] == pytest.approx(value, abs=1e-6)


def test_sequence_preserves_quiet_history_and_empty_batch():
    source = bars(400).with_columns(
        pl.lit(0.0).alias("quote_turnover"),
        pl.lit(0.0).alias("taker_buy_quote_turnover"),
        pl.lit(0).alias("trade_count"),
    )
    keys = pl.DataFrame(
        {"symbol": ["AAAUSDT"], "decision_us": [BASE + 400 * MINUTE_US]}
    )
    result = past_sequence(source, keys)
    for channel in ("log_turnover", "log_trades", "observed_fraction"):
        assert result[channel][0].to_list() == [0.0] * 72
    assert result["buy_share"][0].to_list() == [0.5] * 72
    assert past_sequence(source, keys.head(0)).schema == result.schema


@pytest.mark.parametrize(
    "defect", ["gap", "quality", "early", "future", "unaligned", "duplicate", "symbol"]
)
def test_sequence_refuses_invalid_history_and_keys(defect):
    source = bars(400)
    keys = pl.DataFrame(
        {"symbol": ["AAAUSDT"], "decision_us": [BASE + 400 * MINUTE_US]}
    )
    if defect == "gap":
        source = source.filter(
            pl.col("open_time").dt.epoch("us") != BASE + 200 * MINUTE_US
        )
    elif defect == "quality":
        source = source.with_columns(
            (pl.col("open_time").dt.epoch("us") != BASE + 200 * MINUTE_US).alias(
                "quality_complete"
            )
        )
    elif defect in ("early", "future", "unaligned"):
        value = {
            "early": BASE + 360 * MINUTE_US,
            "future": BASE + 401 * MINUTE_US,
            "unaligned": BASE + 399 * MINUTE_US + 1,
        }[defect]
        keys = keys.with_columns(pl.lit(value, dtype=pl.Int64).alias("decision_us"))
    elif defect == "duplicate":
        keys = pl.concat([keys, keys])
    else:
        keys = keys.with_columns(pl.lit("OTHER").alias("symbol"))
    with pytest.raises(ValueError):
        past_sequence(source, keys)


def test_sequence_policy_and_resource_bounds():
    source = bars(400)
    keys = pl.DataFrame(
        {"symbol": ["AAAUSDT"], "decision_us": [BASE + 400 * MINUTE_US]}
    )
    for kwargs in (
        {"step_minutes": 0},
        {"history_minutes": 361},
        {"history_minutes": True},
    ):
        with pytest.raises(ValueError, match="policy"):
            past_sequence(source, keys, **kwargs)
    with pytest.raises(ValueError, match="budget"):
        past_sequence(source, pl.concat([keys] * 24000))
