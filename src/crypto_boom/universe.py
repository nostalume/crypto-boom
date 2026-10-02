"""Versioned current-observation instrument pools for bounded research."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import msgspec

from crypto_boom import _artifacts
from crypto_boom.binance_source import BINANCE_SPOT, MetadataCapture
from crypto_boom.market import (
    Environment,
    InstrumentId,
    InstrumentStatus,
    MetadataObservation,
)

POOL_POLICY_VERSION = "binance-spot-altcoin-pool-v1"
MAXIMUM_RESEARCH_INSTRUMENTS = 500
DEFAULT_EXCLUDED_BASE_ASSETS = (
    "BRL",
    "BTC",
    "BUSD",
    "DAI",
    "EUR",
    "FDUSD",
    "TRY",
    "TUSD",
    "USDC",
    "USDP",
)


class PoolError(RuntimeError):
    """A research instrument pool could not be safely produced or loaded."""


class PoolResourceError(PoolError):
    """A requested pool exceeded an explicit research bound."""


class PoolIntegrityError(PoolError):
    """A pool artifact or its retained source evidence is inconsistent."""


class PoolPublicationError(PoolError):
    """A validated pool artifact could not be atomically published."""


@dataclass(frozen=True, slots=True)
class PoolPolicy:
    """Explicit current-universe selection policy; never silently truncates."""

    maximum_instruments: int = MAXIMUM_RESEARCH_INSTRUMENTS
    quote_asset: str = "USDT"
    required_permission: str = "SPOT"
    excluded_base_assets: tuple[str, ...] = DEFAULT_EXCLUDED_BASE_ASSETS
    excluded_symbols: tuple[str, ...] = ()
    version: str = POOL_POLICY_VERSION

    def __post_init__(self) -> None:
        if not 0 < self.maximum_instruments <= MAXIMUM_RESEARCH_INSTRUMENTS:
            raise ValueError("research pool maximum must be between 1 and 500")
        if (
            self.quote_asset != "USDT"
            or self.required_permission != "SPOT"
            or self.version != POOL_POLICY_VERSION
        ):
            raise ValueError("research pool policy identity is unsupported")
        if self.excluded_base_assets != tuple(sorted(set(self.excluded_base_assets))):
            raise ValueError("excluded base assets must be sorted and unique")
        if self.excluded_symbols != tuple(sorted(set(self.excluded_symbols))):
            raise ValueError("excluded symbols must be sorted and unique")
        for symbol in self.excluded_symbols:
            InstrumentId(BINANCE_SPOT, Environment.PRODUCTION, symbol)


DEFAULT_POOL_POLICY = PoolPolicy()


@dataclass(frozen=True, slots=True)
class PoolExclusion:
    """One inspectable reason a current metadata row is outside the pool."""

    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResearchInstrumentPool:
    """Content-addressed current-observation candidate pool."""

    schema_version: int
    policy_version: str
    source_endpoint: str
    source_payload_sha256: str
    observed_at_ns: int
    maximum_instruments: int
    quote_asset: str
    required_permission: str
    excluded_base_assets: tuple[str, ...]
    excluded_symbols: tuple[str, ...]
    symbols: tuple[str, ...]
    exclusions: tuple[PoolExclusion, ...]
    unsupported_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.policy_version != POOL_POLICY_VERSION:
            raise PoolIntegrityError("research pool schema or policy is unsupported")
        if not 0 < self.maximum_instruments <= MAXIMUM_RESEARCH_INSTRUMENTS:
            raise PoolIntegrityError("research pool maximum is invalid")
        if not self.symbols:
            raise PoolIntegrityError("research pool contains no instruments")
        if len(self.symbols) > self.maximum_instruments:
            raise PoolResourceError(
                f"research pool has {len(self.symbols)} instruments; "
                f"limit is {self.maximum_instruments}"
            )
        ordered_fields = (
            self.excluded_base_assets,
            self.excluded_symbols,
            self.symbols,
            self.unsupported_symbols,
        )
        if any(value != tuple(sorted(set(value))) for value in ordered_fields):
            raise PoolIntegrityError(
                "research pool symbol lists must be sorted and unique"
            )
        if (
            tuple(sorted(self.exclusions, key=lambda item: item.symbol))
            != self.exclusions
        ):
            raise PoolIntegrityError("research pool exclusions must use stable order")
        excluded_names = tuple(item.symbol for item in self.exclusions)
        if len(excluded_names) != len(set(excluded_names)):
            raise PoolIntegrityError("research pool exclusions must be unique")
        if set(self.symbols) & set(excluded_names):
            raise PoolIntegrityError("included and excluded pool symbols overlap")
        if self.observed_at_ns <= 0 or not self.source_endpoint:
            raise PoolIntegrityError("research pool source observation is invalid")
        if not _artifacts.is_sha256(self.source_payload_sha256):
            raise PoolIntegrityError("research pool source digest is invalid")
        for symbol in self.symbols:
            try:
                InstrumentId(BINANCE_SPOT, Environment.PRODUCTION, symbol)
            except ValueError as error:
                raise PoolIntegrityError("research pool symbol is invalid") from error

    def _content_mapping(self) -> dict[str, object]:
        return {
            "excluded_base_assets": list(self.excluded_base_assets),
            "excluded_symbols": list(self.excluded_symbols),
            "exclusions": [
                {"reason": item.reason, "symbol": item.symbol}
                for item in self.exclusions
            ],
            "maximum_instruments": self.maximum_instruments,
            "observed_at_ns": self.observed_at_ns,
            "policy_version": self.policy_version,
            "quote_asset": self.quote_asset,
            "required_permission": self.required_permission,
            "schema_version": self.schema_version,
            "source_endpoint": self.source_endpoint,
            "source_payload_sha256": self.source_payload_sha256,
            "symbols": list(self.symbols),
            "unsupported_symbols": list(self.unsupported_symbols),
        }

    @property
    def pool_id(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                _artifacts.canonical_json(self._content_mapping())
            ).hexdigest()
        )

    def to_mapping(self) -> dict[str, object]:
        return {"pool_id": self.pool_id, **self._content_mapping()}

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        return tuple(
            InstrumentId(BINANCE_SPOT, Environment.PRODUCTION, symbol)
            for symbol in self.symbols
        )


_POOL_DECODER = msgspec.json.Decoder(ResearchInstrumentPool)


@dataclass(frozen=True, slots=True)
class PublishedResearchPool:
    """One immutable pool artifact publication."""

    path: Path
    pool: ResearchInstrumentPool
    already_present: bool


def build_research_pool(
    metadata: MetadataCapture,
    *,
    policy: PoolPolicy = DEFAULT_POOL_POLICY,
) -> ResearchInstrumentPool:
    """Select current observed altcoin USDT Spot instruments without truncation."""

    source_digest = "sha256:" + hashlib.sha256(metadata.raw_payload).hexdigest()
    endpoints = {event.provenance.source.endpoint for event in metadata.events}
    receipts = {event.provenance.receipt.wall_time_ns for event in metadata.events}
    payload_digests = {
        event.provenance.payload_digest.value for event in metadata.events
    }
    if len(endpoints) != 1 or len(receipts) != 1 or payload_digests != {source_digest}:
        raise PoolIntegrityError("metadata capture provenance is inconsistent")

    symbols: list[str] = []
    exclusions: list[PoolExclusion] = []
    excluded_bases = set(policy.excluded_base_assets)
    excluded_symbols = set(policy.excluded_symbols)
    for event in sorted(metadata.events, key=lambda item: item.instrument.symbol):
        reason: str | None = None
        if event.observation is not MetadataObservation.OBSERVED:
            reason = "not_observed"
        elif event.status is not InstrumentStatus.TRADING:
            reason = "not_trading"
        elif event.quote_asset != policy.quote_asset:
            reason = "wrong_quote_asset"
        elif policy.required_permission not in event.permissions:
            reason = "missing_permission"
        elif event.base_asset in excluded_bases:
            reason = "excluded_base_asset"
        elif event.instrument.symbol in excluded_symbols:
            reason = "excluded_symbol"

        if reason is None:
            symbols.append(event.instrument.symbol)
        else:
            exclusions.append(PoolExclusion(event.instrument.symbol, reason))

    if len(symbols) > policy.maximum_instruments:
        raise PoolResourceError(
            f"research pool has {len(symbols)} instruments; "
            f"limit is {policy.maximum_instruments}; refusing silent truncation"
        )
    return ResearchInstrumentPool(
        schema_version=1,
        policy_version=policy.version,
        source_endpoint=next(iter(endpoints)),
        source_payload_sha256=source_digest,
        observed_at_ns=next(iter(receipts)),
        maximum_instruments=policy.maximum_instruments,
        quote_asset=policy.quote_asset,
        required_permission=policy.required_permission,
        excluded_base_assets=policy.excluded_base_assets,
        excluded_symbols=policy.excluded_symbols,
        symbols=tuple(symbols),
        exclusions=tuple(exclusions),
        unsupported_symbols=tuple(sorted(set(metadata.unsupported_symbols))),
    )


def publish_research_pool(
    pool: ResearchInstrumentPool,
    *,
    metadata: MetadataCapture,
    output_root: Path,
) -> PublishedResearchPool:
    """Atomically publish a pool with its exact exchangeInfo source payload."""

    actual_source = "sha256:" + hashlib.sha256(metadata.raw_payload).hexdigest()
    if actual_source != pool.source_payload_sha256:
        raise PoolIntegrityError("pool source payload does not match its manifest")
    output_root = output_root.resolve()
    target = (
        output_root
        / "binance"
        / "spot"
        / "research-pools"
        / pool.pool_id.removeprefix("sha256:")
    )
    staging_parent = output_root / ".staging"
    try:
        staging_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PoolPublicationError("pool staging root is unavailable") from error

    with _artifacts.publication_staging_directory(
        staging_parent, prefix="pool-"
    ) as staging:
        _write_bytes(staging / "exchange-info.json", metadata.raw_payload)
        _write_bytes(
            staging / "pool.json", _artifacts.canonical_json(pool.to_mapping())
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if load_published_research_pool(target) != pool:
                    raise PoolPublicationError(
                        "existing pool conflicts with publication"
                    )
                return PublishedResearchPool(target, pool, True)
            os.replace(staging, target)
        except PoolError:
            raise
        except OSError as error:
            if target.exists() and load_published_research_pool(target) == pool:
                return PublishedResearchPool(target, pool, True)
            raise PoolPublicationError("atomic pool publication failed") from error
    return PublishedResearchPool(target, pool, False)


def load_published_research_pool(path: Path) -> ResearchInstrumentPool:
    """Strictly verify and load one content-addressed pool artifact."""

    path = path.resolve()
    pool_payload = _read_bounded(path / "pool.json", maximum=2 * 1024 * 1024)
    source_payload = _read_bounded(
        path / "exchange-info.json",
        maximum=32 * 1024 * 1024,
    )
    try:
        pool = _POOL_DECODER.decode(pool_payload)
        if pool_payload != _artifacts.canonical_json(pool.to_mapping()):
            raise ValueError
    except (msgspec.DecodeError, ValueError) as error:
        raise PoolIntegrityError("pool manifest is invalid") from error
    actual_source = "sha256:" + hashlib.sha256(source_payload).hexdigest()
    if actual_source != pool.source_payload_sha256:
        raise PoolIntegrityError("pool source payload does not match its manifest")
    if path.name != pool.pool_id.removeprefix("sha256:"):
        raise PoolIntegrityError("pool publication path does not match its identity")
    return pool


def _read_bounded(path: Path, *, maximum: int) -> bytes:
    try:
        with path.open("rb") as stream:
            payload = stream.read(maximum + 1)
    except OSError as error:
        raise PoolPublicationError("published pool artifact is unavailable") from error
    if len(payload) > maximum:
        raise PoolResourceError("published pool artifact exceeds its byte limit")
    return payload


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise PoolPublicationError("pool staging write failed") from error
