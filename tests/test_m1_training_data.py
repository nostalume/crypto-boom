"""The frozen training matrix is admitted by source identity, not path alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from crypto_boom.research.training_data import (
    TrainingDataError,
    load_frozen_successor_training_block,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data/universe-expansion-successor-20260928-v1/design-matrix.parquet"


@pytest.mark.research
def test_training_block_refuses_missing_or_altered_matrix(tmp_path):
    with pytest.raises(TrainingDataError, match="unavailable"):
        load_frozen_successor_training_block(tmp_path, "secondary")
    matrix = (
        tmp_path / "data/universe-expansion-successor-20260928-v1/design-matrix.parquet"
    )
    matrix.parent.mkdir(parents=True)
    matrix.write_bytes(b"not the pinned matrix")
    with pytest.raises(TrainingDataError, match="digest differs"):
        load_frozen_successor_training_block(tmp_path, "secondary")


def test_training_block_refuses_unknown_cell():
    with pytest.raises(TrainingDataError, match="unknown"):
        load_frozen_successor_training_block(ROOT, "future")
