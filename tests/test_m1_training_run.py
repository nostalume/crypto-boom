"""Source-bound research rehearsal and immutable diagnostic receipt."""

from __future__ import annotations

from pathlib import Path

import pytest

from crypto_boom.research.training_run import (
    TrainingRunError,
    run_verified_training,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data/universe-expansion-successor-20260928-v1/design-matrix.parquet"
FROZEN = ROOT / "data/universe-expansion-successor-20260928-v1"


def test_full_fit_requires_explicit_opt_in_and_frozen_outputs_are_forbidden(tmp_path):
    with pytest.raises(TrainingRunError, match="allow_full"):
        run_verified_training(ROOT, "secondary", tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
    with pytest.raises(TrainingRunError, match="frozen study directory"):
        run_verified_training(
            ROOT, "secondary", FROZEN / "new-subdir", rows_per_month=24
        )
    with pytest.raises(TrainingRunError, match="choose one"):
        run_verified_training(
            ROOT, "secondary", tmp_path / "runs", rows_per_month=24, allow_full=True
        )
