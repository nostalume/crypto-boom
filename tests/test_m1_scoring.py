"""The maintained M1 row scorer, without touching the reserved confirm months."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

from crypto_boom.research.m1 import FrozenM1Scorer, M1ScoringError


def _bundle() -> dict:
    payload = (
        resources.files("crypto_boom.research").joinpath("m1-v1.json").read_bytes()
    )
    return json.loads(payload)


def test_m1_bundle_scores_only_frozen_symbols_and_cells() -> None:
    scorer = FrozenM1Scorer.from_package()
    bundle = _bundle()
    features = {name: 0.0 for name in bundle["cells"]["primary"]["added_columns"]}
    features.update(
        {
            name: 0.0
            for name in (
                "comove_share",
                "hours_since_completion",
                "log_range24",
                "breadth",
            )
        }
    )
    features["size_quintile"] = 3
    symbol = bundle["symbols"][0]
    result = scorer.score("primary", symbol, 100, features)
    assert result.cell == "primary"
    assert result.symbol == symbol
    assert 0.0 <= result.unvalidated_score <= 1.0
    assert (
        result.bundle_sha256
        == hashlib.sha256(
            resources.files("crypto_boom.research").joinpath("m1-v1.json").read_bytes()
        ).hexdigest()
    )

    altered = _bundle()
    altered["cells"]["primary"]["beta"][0] += 1.0
    altered_scorer = FrozenM1Scorer(json.dumps(altered).encode("utf-8"))
    assert (
        altered_scorer.score("primary", symbol, 100, features).unvalidated_score
        != result.unvalidated_score
    )

    with pytest.raises(M1ScoringError, match="outside frozen M1 code space"):
        scorer.score("primary", "NEWUSDT", 100, features)
    with pytest.raises(M1ScoringError, match="unknown M1 cell"):
        scorer.score("other", symbol, 100, features)
    with pytest.raises(M1ScoringError, match="minute"):
        scorer.score("primary", symbol, -1, features)
    with pytest.raises(M1ScoringError, match="missing M1 feature"):
        scorer.score("primary", symbol, 100, {"size_quintile": 3})
    with pytest.raises(M1ScoringError, match="size_quintile"):
        scorer.score("primary", symbol, 100, {**features, "size_quintile": 9})
