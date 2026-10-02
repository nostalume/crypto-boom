"""Shared byte and staging mechanisms for immutable local artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from uuid import uuid4

_HASH_CHUNK_BYTES = 64 * 1_024
_MAXIMUM_NAME_ATTEMPTS = 100
_SHA256_PREFIX = "sha256:"


def canonical_json(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def content_id(payload: dict[str, object]) -> str:
    return _SHA256_PREFIX + hashlib.sha256(canonical_json(payload)).hexdigest()


def is_sha256(value: str) -> bool:
    """Recognize canonical, lowercase SHA256 content identities, not raw hashes."""
    digest = value.removeprefix(_SHA256_PREFIX)
    return (
        value.startswith(_SHA256_PREFIX)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return _SHA256_PREFIX + digest.hexdigest(), size


def adopt_directory(
    staging: Path,
    target: Path,
    *,
    verify_existing: Callable[[Path], None],
) -> bool:
    """Adopt atomically; verify race winners; bound transient Windows recovery."""
    delays = (0.05, 0.1, 0.2)
    for attempt in range(len(delays) + 1):
        if target.exists():
            verify_existing(target)
            return True
        try:
            os.replace(staging, target)
        except OSError as exc:
            if target.exists():
                verify_existing(target)
                return True
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == len(
                delays
            ):
                raise
            logging.getLogger(__name__).warning(
                "Publication refused (Windows %s); bounded retry %d/3: %s",
                exc.winerror,
                attempt + 1,
                target,
            )
            time.sleep(delays[attempt])
        else:
            return False
    raise AssertionError("unreachable publication attempt")


@contextmanager
def publication_staging_directory(
    parent: Path,
    *,
    prefix: str,
) -> Iterator[Path]:
    """Create a unique staging child inheriting the publication-root ACL.

    ``tempfile.TemporaryDirectory`` creates a creator-private directory. Moving
    it into a durable target preserves the private ACL on Windows.
    """

    if not prefix or Path(prefix).name != prefix:
        raise ValueError("publication staging prefix must be one path component")
    for _attempt in range(_MAXIMUM_NAME_ATTEMPTS):
        staging = parent / f"{prefix}{uuid4().hex}"
        try:
            staging.mkdir()
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError("could not allocate a publication staging directory")

    try:
        yield staging
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def relative_reference(path: Path, *, base: Path) -> str:
    """Portable locator, not identity; callers retain content hashes separately."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError as exc:
        raise ValueError(
            "portable references require a common filesystem volume"
        ) from exc


def resolve_reference(reference: str, *, base: Path) -> Path:
    """Resolve a v2 relative locator without silently accepting legacy absolutes.

    Parent traversal is intentional for siblings under a shared data root. This is
    not a sandbox: consumers must verify the referenced artifact's content identity.
    """
    if (
        not isinstance(reference, str)
        or not reference
        or reference.startswith("/")
        or "\\" in reference
        or Path(reference).is_absolute()
        or PureWindowsPath(reference).drive
    ):
        raise ValueError("expected a portable relative artifact reference")
    return (base / reference).resolve()
