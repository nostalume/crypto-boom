"""Canonical storage compatibility API.

Implementation ownership lives in :mod:`crypto_boom.storage.canonical`.

The names below are re-exported **lazily**. Importing this package used to
execute :mod:`crypto_boom.storage.canonical` eagerly, and ``canonical`` imports
:mod:`crypto_boom.history.daily` at module scope (``canonical.py:19``), which
drags ``aiohttp`` and the archive-download machinery in behind it. Every caller
of :mod:`crypto_boom.storage` paid that, including ``crypto_boom.cli``, which
reaches :mod:`crypto_boom.storage.source` and calls none of these names.

PEP 562 module ``__getattr__`` defers the work to first access, so ``from
crypto_boom.storage import load_published_canonical_klines`` still works and
``__all__`` still enumerates the compatibility surface, but importing the
package no longer imports the implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crypto_boom.storage.canonical import (
        KLINE_SCHEMA,
        QUARANTINE_SCHEMA,
        STORAGE_VERSION,
        CanonicalizationResult,
        CanonicalKline,
        CanonicalManifest,
        CanonicalQualityReport,
        PublishedCanonicalPartition,
        QuarantinedKline,
        QuarantineReason,
        ReconciliationResult,
        ReconciliationSample,
        SourceKline,
        StorageError,
        StoragePublicationError,
        StorageSchemaError,
        canonicalize_klines,
        load_published_canonical_klines,
        publish_canonical_archive,
        reconcile_kline,
    )

__all__ = (
    "KLINE_SCHEMA",
    "QUARANTINE_SCHEMA",
    "STORAGE_VERSION",
    "CanonicalKline",
    "CanonicalManifest",
    "CanonicalQualityReport",
    "CanonicalizationResult",
    "PublishedCanonicalPartition",
    "QuarantineReason",
    "QuarantinedKline",
    "ReconciliationResult",
    "ReconciliationSample",
    "SourceKline",
    "StorageError",
    "StoragePublicationError",
    "StorageSchemaError",
    "canonicalize_klines",
    "load_published_canonical_klines",
    "publish_canonical_archive",
    "reconcile_kline",
)

_LAZY_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Resolve a re-exported canonical name on first access."""
    if name not in _LAZY_NAMES:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    from crypto_boom.storage import canonical

    value = getattr(canonical, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_NAMES)
