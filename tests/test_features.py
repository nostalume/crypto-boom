"""Reference-group feature invariants used by historical research."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from crypto_boom.features import MARKET_FEATURES, add_peer_context


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
