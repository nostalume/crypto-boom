from __future__ import annotations

import os
from pathlib import Path

import pytest

from crypto_boom._artifacts import publication_staging_directory


def test_publication_staging_cleans_or_transfers_one_inherited_child(
    tmp_path: Path,
) -> None:
    parent = tmp_path / ".staging"
    parent.mkdir()

    with publication_staging_directory(parent, prefix="artifact-") as abandoned:
        (abandoned / "payload").write_text("discard", encoding="utf-8")
    assert not abandoned.exists()

    target = tmp_path / "published"
    with publication_staging_directory(parent, prefix="artifact-") as adopted:
        (adopted / "payload").write_text("keep", encoding="utf-8")
        os.replace(adopted, target)
    assert (target / "payload").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("prefix", ("", "../escape", "nested/path"))
def test_publication_staging_rejects_non_component_prefix(
    tmp_path: Path,
    prefix: str,
) -> None:
    with pytest.raises(ValueError, match="one path component"):
        with publication_staging_directory(tmp_path, prefix=prefix):
            pass
