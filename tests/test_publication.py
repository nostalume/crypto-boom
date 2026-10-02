from __future__ import annotations

import os
from pathlib import Path

import pytest

from crypto_boom._artifacts import is_sha256, publication_staging_directory


@pytest.mark.parametrize(
    "value, valid",
    [
        ("sha256:" + "ab01" * 16, True),
        ("ab01" * 16, False),
        ("sha256:" + "AB01" * 16, False),
        ("sha256:" + "g" * 64, False),
        ("sha256:" + "a" * 63, False),
        ("sha256:" + "a" * 65, False),
        ("sha256:sha256:" + "a" * 64, False),
    ],
)
def test_content_identity_requires_prefix_and_exact_lowercase_digest(value, valid):
    assert is_sha256(value) is valid


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


@pytest.mark.parametrize(
    "code, failures", [(5, 1), (32, 2), (33, 3), (5, 4), (None, 1)]
)
def test_directory_adoption_has_bounded_windows_recovery(
    tmp_path, monkeypatch, code, failures
):
    from crypto_boom import _artifacts

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    replace = os.replace
    attempts, sleeps = [], []
    error = PermissionError("injected publication failure")
    if code is not None:
        error.winerror = code

    def rename(a, b):
        attempts.append(1)
        if len(attempts) <= failures:
            raise error
        replace(a, b)

    monkeypatch.setattr(os, "replace", rename)
    monkeypatch.setattr("time.sleep", sleeps.append)
    if code is None or failures == 4:
        with pytest.raises(PermissionError) as caught:
            _artifacts.adopt_directory(source, target, verify_existing=lambda p: None)
        assert caught.value is error
        assert source.exists() and not target.exists()
        assert len(attempts) == (1 if code is None else 4)
    else:
        assert not _artifacts.adopt_directory(
            source, target, verify_existing=lambda p: None
        )
        assert (target / "payload").read_bytes() == b"complete"
        assert len(attempts) == failures + 1
    assert sum(sleeps) <= 0.351


def test_directory_adoption_validates_winner_during_retry(tmp_path, monkeypatch):
    from crypto_boom import _artifacts

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    error = PermissionError("held directory")
    error.winerror = 5
    calls = []

    def rename(a, b):
        calls.append(1)
        raise error

    def sleep(delay):
        target.mkdir()
        (target / "payload").write_bytes(b"winner")

    def verify(path):
        assert (path / "payload").read_bytes() == b"winner"

    monkeypatch.setattr(os, "replace", rename)
    monkeypatch.setattr("time.sleep", sleep)
    assert _artifacts.adopt_directory(source, target, verify_existing=verify)
    assert len(calls) == 1


def test_directory_adoption_cancellation_cleans_staging(tmp_path, monkeypatch):
    from crypto_boom import _artifacts

    error = PermissionError("held directory")
    error.winerror = 5

    def rename(a, b):
        raise error

    def cancel(delay):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", rename)
    monkeypatch.setattr("time.sleep", cancel)
    with pytest.raises(KeyboardInterrupt):
        with publication_staging_directory(tmp_path, prefix="cancel-") as staging:
            _artifacts.adopt_directory(
                staging, tmp_path / "target", verify_existing=lambda p: None
            )
    assert not list(tmp_path.iterdir())
