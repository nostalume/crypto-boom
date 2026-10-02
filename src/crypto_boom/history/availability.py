"""Bounded monthly archive-availability audits for a verified research pool."""

from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from asyncio import sleep
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from time import time_ns

import aiohttp
import msgspec

from crypto_boom import _artifacts
from crypto_boom.universe import ResearchInstrumentPool

OFFICIAL_MONTHLY_ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot/monthly"
OFFICIAL_ARCHIVE_INDEX_URL = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
)
_SPOT_SYMBOL_PREFIX = "data/spot/monthly/klines/"
_S3_XML_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+[*]?([^\r\n]+)")


class AvailabilityError(RuntimeError):
    """A monthly archive availability audit could not be completed safely."""


class AvailabilityResourceError(AvailabilityError):
    """An availability audit exceeded its declared resource envelope."""


class AvailabilityIntegrityError(AvailabilityError):
    """An official checksum response or published audit is inconsistent."""


class AvailabilityPublicationError(AvailabilityError):
    """A completed availability audit could not be atomically published."""


class AvailabilityState(StrEnum):
    """Observed result of one monthly checksum probe."""

    AVAILABLE = "available"
    NOT_FOUND = "not_found"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class HistoricalArchiveSymbolCatalog:
    """Complete observed Spot archive directory listing, not venue eligibility."""

    source_url: str
    prefix: str
    observed_at_ns: int
    page_count: int
    symbols: tuple[str, ...]


async def discover_spot_archive_symbols(
    *,
    base_url: str = OFFICIAL_ARCHIVE_INDEX_URL,
    maximum_pages: int = 8,
    request_timeout_seconds: float = 10.0,
) -> HistoricalArchiveSymbolCatalog:
    """List all historical Spot monthly-kline symbol directories or fail closed.

    A listed directory establishes only archive presence. It does not establish
    a listing date, trading status, or minute-level market eligibility.
    """

    if not 1 <= maximum_pages <= 8 or not 0 < request_timeout_seconds <= 10:
        raise ValueError("archive listing exceeds its page or timeout bound")

    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    symbols: list[str] = []
    token: str | None = None
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        for page_count in range(1, maximum_pages + 1):
            params = {
                "list-type": "2",
                "prefix": _SPOT_SYMBOL_PREFIX,
                "delimiter": "/",
                "max-keys": "1000",
            }
            if token is not None:
                params["continuation-token"] = token
            try:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        raise AvailabilityError(
                            f"archive index returned HTTP {response.status}"
                        )
                    payload = await _read_response(response, maximum=256_000)
            except (TimeoutError, aiohttp.ClientError) as error:
                raise AvailabilityError("archive index request failed") from error

            try:
                root = ET.fromstring(payload)
            except ET.ParseError as error:
                raise AvailabilityIntegrityError(
                    "archive index XML is invalid"
                ) from error
            if root.tag != f"{_S3_XML_NAMESPACE}ListBucketResult":
                raise AvailabilityIntegrityError("archive index root is unexpected")
            if root.findtext(f"{_S3_XML_NAMESPACE}Prefix") != _SPOT_SYMBOL_PREFIX:
                raise AvailabilityIntegrityError("archive index prefix is unexpected")
            for entry in root.findall(
                f"{_S3_XML_NAMESPACE}CommonPrefixes/{_S3_XML_NAMESPACE}Prefix"
            ):
                directory = entry.text or ""
                if not directory.startswith(
                    _SPOT_SYMBOL_PREFIX
                ) or not directory.endswith("/"):
                    raise AvailabilityIntegrityError(
                        "archive symbol directory is invalid"
                    )
                symbol = directory[len(_SPOT_SYMBOL_PREFIX) : -1]
                if not symbol or "/" in symbol:
                    raise AvailabilityIntegrityError(
                        "archive symbol directory is invalid"
                    )
                symbols.append(symbol)

            truncated = root.findtext(f"{_S3_XML_NAMESPACE}IsTruncated")
            if truncated == "false":
                if not symbols or len(symbols) != len(set(symbols)):
                    raise AvailabilityIntegrityError(
                        "archive symbols are empty or repeated"
                    )
                return HistoricalArchiveSymbolCatalog(
                    source_url=base_url,
                    prefix=_SPOT_SYMBOL_PREFIX,
                    observed_at_ns=time_ns(),
                    page_count=page_count,
                    symbols=tuple(sorted(symbols)),
                )
            if truncated != "true":
                raise AvailabilityIntegrityError("archive index pagination is invalid")
            next_token = root.findtext(f"{_S3_XML_NAMESPACE}NextContinuationToken")
            if not next_token or next_token == token:
                raise AvailabilityIntegrityError(
                    "archive index continuation is invalid"
                )
            token = next_token
    raise AvailabilityResourceError("archive listing exceeded page limit")


@dataclass(frozen=True, slots=True)
class AvailabilityLimits:
    """Bounds and conservative scheduling policy for one audit."""

    maximum_probes: int = 12_000
    retry_attempts: int = 4
    retry_base_seconds: float = 1.0
    request_spacing_seconds: float = 0.1
    response_bytes: int = 1_024
    request_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            self.maximum_probes <= 0
            or self.retry_attempts <= 0
            or self.response_bytes <= 0
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError(
                "availability count, byte, and timeout limits must be positive"
            )
        if self.retry_base_seconds < 0 or self.request_spacing_seconds < 0:
            raise ValueError("availability delays cannot be negative")


DEFAULT_AVAILABILITY_LIMITS = AvailabilityLimits()


@dataclass(frozen=True, slots=True)
class MonthlyAvailabilityRequest:
    """Inclusive calendar-month range for one-minute Spot kline archives."""

    start_month: date
    end_month: date
    interval: str = "1m"

    def __post_init__(self) -> None:
        if self.start_month.day != 1 or self.end_month.day != 1:
            raise ValueError("availability months must use the first calendar day")
        if self.end_month < self.start_month:
            raise ValueError("availability end month precedes its start")
        if self.interval != "1m":
            raise ValueError("availability audit supports only one-minute klines")

    @property
    def months(self) -> tuple[str, ...]:
        values: list[str] = []
        current = self.start_month
        while True:
            values.append(current.strftime("%Y-%m"))
            if current == self.end_month:
                break
            current = _next_month(current)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class MonthlyArchiveProbe:
    """One source-observed monthly archive availability result."""

    symbol: str
    month: str
    state: AvailabilityState
    checksum_sha256: str | None
    attempts: int
    http_status: int | None

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise AvailabilityIntegrityError("availability attempts must be positive")
        if self.state is AvailabilityState.AVAILABLE:
            if (
                self.checksum_sha256 is None
                or not _artifacts.is_sha256(self.checksum_sha256)
                or self.http_status != 200
            ):
                raise AvailabilityIntegrityError("available probe identity is invalid")
        elif self.checksum_sha256 is not None:
            raise AvailabilityIntegrityError(
                "unavailable probe cannot carry a checksum"
            )
        if self.state is AvailabilityState.NOT_FOUND and self.http_status != 404:
            raise AvailabilityIntegrityError("not-found probe status is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "checksum_sha256": self.checksum_sha256,
            "http_status": self.http_status,
            "month": self.month,
            "state": self.state.value,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class ArchiveAvailabilityReport:
    """Content-addressed result of one complete bounded availability audit."""

    schema_version: int
    pool_id: str
    base_url: str
    interval: str
    start_month: str
    end_month: str
    probed_at_ns: int
    probes: tuple[MonthlyArchiveProbe, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.pool_id is None
            or not _artifacts.is_sha256(self.pool_id)
        ):
            raise AvailabilityIntegrityError("availability report identity is invalid")
        if self.interval != "1m" or self.probed_at_ns <= 0 or not self.base_url:
            raise AvailabilityIntegrityError("availability report source is invalid")
        if not self.probes:
            raise AvailabilityIntegrityError("availability report contains no probes")
        ordered = tuple(sorted(self.probes, key=lambda item: (item.symbol, item.month)))
        if ordered != self.probes:
            raise AvailabilityIntegrityError(
                "availability probes are not deterministic"
            )
        identities = tuple((item.symbol, item.month) for item in self.probes)
        if len(identities) != len(set(identities)):
            raise AvailabilityIntegrityError("availability probes contain duplicates")

    @property
    def available_count(self) -> int:
        return sum(item.state is AvailabilityState.AVAILABLE for item in self.probes)

    @property
    def not_found_count(self) -> int:
        return sum(item.state is AvailabilityState.NOT_FOUND for item in self.probes)

    @property
    def unresolved_count(self) -> int:
        return sum(item.state is AvailabilityState.UNRESOLVED for item in self.probes)

    @property
    def ready(self) -> bool:
        return self.unresolved_count == 0

    def _content_mapping(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "end_month": self.end_month,
            "interval": self.interval,
            "pool_id": self.pool_id,
            "probed_at_ns": self.probed_at_ns,
            "probes": [item.to_mapping() for item in self.probes],
            "schema_version": self.schema_version,
            "start_month": self.start_month,
        }

    @property
    def report_id(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                _artifacts.canonical_json(self._content_mapping())
            ).hexdigest()
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "available_count": self.available_count,
            "not_found_count": self.not_found_count,
            "ready": self.ready,
            "report_id": self.report_id,
            "unresolved_count": self.unresolved_count,
            **self._content_mapping(),
        }


_AVAILABILITY_DECODER = msgspec.json.Decoder(ArchiveAvailabilityReport)


@dataclass(frozen=True, slots=True)
class PublishedArchiveAvailability:
    """One atomically published monthly availability audit."""

    path: Path
    report: ArchiveAvailabilityReport
    already_present: bool


async def probe_monthly_archive_availability(
    pool: ResearchInstrumentPool,
    request: MonthlyAvailabilityRequest,
    *,
    output_root: Path,
    symbols: tuple[str, ...] = (),
    limits: AvailabilityLimits = DEFAULT_AVAILABILITY_LIMITS,
    base_url: str = OFFICIAL_MONTHLY_ARCHIVE_BASE_URL,
) -> PublishedArchiveAvailability:
    """Probe selected symbol/month checksums and retain absence separately."""

    selected_symbols = _select_symbols(pool, symbols)
    return await _probe_selected_symbols(
        selected_symbols,
        pool.pool_id,
        request,
        output_root=output_root,
        limits=limits,
        base_url=base_url,
    )


async def probe_historical_archive_availability(
    catalog: HistoricalArchiveSymbolCatalog,
    request: MonthlyAvailabilityRequest,
    *,
    symbols: tuple[str, ...],
    output_root: Path,
    limits: AvailabilityLimits = DEFAULT_AVAILABILITY_LIMITS,
    base_url: str = OFFICIAL_MONTHLY_ARCHIVE_BASE_URL,
) -> PublishedArchiveAvailability:
    """Probe a historical directory cohort without asserting current tradability."""
    if (
        not symbols
        or tuple(sorted(set(symbols))) != symbols
        or not set(symbols).issubset(catalog.symbols)
    ):
        raise AvailabilityIntegrityError("invalid historical catalog selection")
    return await _probe_selected_symbols(
        symbols,
        _artifacts.content_id(asdict(catalog)),
        request,
        output_root=output_root,
        limits=limits,
        base_url=base_url,
    )


async def _probe_selected_symbols(
    selected_symbols: tuple[str, ...],
    pool_id: str,
    request: MonthlyAvailabilityRequest,
    *,
    output_root: Path,
    limits: AvailabilityLimits,
    base_url: str,
) -> PublishedArchiveAvailability:
    months = request.months
    probe_count = len(selected_symbols) * len(months)
    if probe_count > limits.maximum_probes:
        raise AvailabilityResourceError(
            f"availability audit has {probe_count} probes; "
            f"limit is {limits.maximum_probes}"
        )

    timeout = aiohttp.ClientTimeout(total=limits.request_timeout_seconds)
    probes: list[MonthlyArchiveProbe] = []
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as session:
        probe_index = 0
        for symbol in selected_symbols:
            for month in months:
                probes.append(
                    await _probe_month(
                        session,
                        symbol=symbol,
                        month=month,
                        interval=request.interval,
                        base_url=base_url,
                        limits=limits,
                    )
                )
                probe_index += 1
                if probe_index < probe_count and limits.request_spacing_seconds > 0:
                    await sleep(limits.request_spacing_seconds)

    report = ArchiveAvailabilityReport(
        schema_version=1,
        pool_id=pool_id,
        base_url=base_url.rstrip("/"),
        interval=request.interval,
        start_month=request.start_month.strftime("%Y-%m"),
        end_month=request.end_month.strftime("%Y-%m"),
        probed_at_ns=time_ns(),
        probes=tuple(probes),
    )
    return _publish_report(report, output_root=output_root)


def _select_symbols(
    pool: ResearchInstrumentPool,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested:
        return pool.symbols
    if len(requested) != len(set(requested)):
        raise AvailabilityIntegrityError(
            "availability symbol filter contains duplicates"
        )
    unknown = sorted(set(requested) - set(pool.symbols))
    if unknown:
        raise AvailabilityIntegrityError(
            "availability symbol filter is not present in research pool: "
            + ", ".join(unknown)
        )
    return tuple(sorted(requested))


async def _probe_month(
    session: aiohttp.ClientSession,
    *,
    symbol: str,
    month: str,
    interval: str,
    base_url: str,
    limits: AvailabilityLimits,
) -> MonthlyArchiveProbe:
    filename = f"{symbol}-{interval}-{month}.zip"
    url = f"{base_url.rstrip('/')}/klines/{symbol}/{interval}/{filename}.CHECKSUM"
    last_status: int | None = None
    for attempt in range(1, limits.retry_attempts + 1):
        retry_after = 0.0
        try:
            async with session.get(url) as response:
                last_status = response.status
                payload = await _read_response(response, maximum=limits.response_bytes)
                if response.status == 200:
                    checksum = _parse_checksum(payload, expected_filename=filename)
                    return MonthlyArchiveProbe(
                        symbol,
                        month,
                        AvailabilityState.AVAILABLE,
                        f"sha256:{checksum}",
                        attempt,
                        200,
                    )
                if response.status == 404:
                    return MonthlyArchiveProbe(
                        symbol,
                        month,
                        AvailabilityState.NOT_FOUND,
                        None,
                        attempt,
                        404,
                    )
                retry_after = _retry_after_seconds(response)
        except AvailabilityError:
            raise
        except (TimeoutError, aiohttp.ClientError):
            last_status = None
        if attempt < limits.retry_attempts:
            delay = max(
                retry_after,
                limits.retry_base_seconds * (2 ** (attempt - 1)),
            )
            if delay > 0:
                await sleep(delay)
    return MonthlyArchiveProbe(
        symbol,
        month,
        AvailabilityState.UNRESOLVED,
        None,
        limits.retry_attempts,
        last_status,
    )


async def _read_response(response: aiohttp.ClientResponse, *, maximum: int) -> bytes:
    if response.content_length is not None and response.content_length > maximum:
        raise AvailabilityResourceError("availability response exceeds byte limit")
    payload = bytearray()
    async for chunk in response.content.iter_chunked(min(maximum, 16_384)):
        payload.extend(chunk)
        if len(payload) > maximum:
            raise AvailabilityResourceError("availability response exceeds byte limit")
    return bytes(payload)


def _parse_checksum(document: bytes, *, expected_filename: str) -> str:
    try:
        text = document.decode("ascii")
    except UnicodeDecodeError as error:
        raise AvailabilityIntegrityError("monthly CHECKSUM is not ASCII") from error
    match = _CHECKSUM_LINE.fullmatch(text.strip())
    if match is None:
        raise AvailabilityIntegrityError("monthly CHECKSUM has an invalid format")
    digest, filename = match.groups()
    if filename != expected_filename:
        raise AvailabilityIntegrityError("monthly CHECKSUM filename is unexpected")
    return digest.lower()


def _retry_after_seconds(response: aiohttp.ClientResponse) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _publish_report(
    report: ArchiveAvailabilityReport,
    *,
    output_root: Path,
) -> PublishedArchiveAvailability:
    output_root = output_root.resolve()
    target = (
        output_root
        / "binance"
        / "spot"
        / "archive-availability"
        / report.pool_id.removeprefix("sha256:")
        / f"{report.start_month}_{report.end_month}"
        / report.report_id.removeprefix("sha256:")
    )
    staging_parent = output_root / ".staging"
    payload = _artifacts.canonical_json(report.to_mapping())
    try:
        staging_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AvailabilityPublicationError(
            "availability staging root is unavailable"
        ) from error
    with _artifacts.publication_staging_directory(
        staging_parent,
        prefix="availability-",
    ) as staging:
        _write_bytes(staging / "availability.json", payload)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                _verify_existing(target, payload)
                return PublishedArchiveAvailability(target, report, True)
            os.replace(staging, target)
        except AvailabilityError:
            raise
        except OSError as error:
            if target.exists():
                _verify_existing(target, payload)
                return PublishedArchiveAvailability(target, report, True)
            raise AvailabilityPublicationError(
                "atomic availability publication failed"
            ) from error
    return PublishedArchiveAvailability(target, report, False)


def load_published_archive_availability(path: Path) -> ArchiveAvailabilityReport:
    """Strictly verify and load one published availability report."""

    path = path.resolve()
    try:
        payload = (path / "availability.json").read_bytes()
    except OSError as error:
        raise AvailabilityPublicationError(
            "published availability report is unavailable"
        ) from error
    if len(payload) > 8 * 1024 * 1024:
        raise AvailabilityResourceError(
            "published availability report exceeds byte limit"
        )
    try:
        report = _AVAILABILITY_DECODER.decode(payload)
        if payload != _artifacts.canonical_json(report.to_mapping()):
            raise ValueError
    except (msgspec.DecodeError, ValueError) as error:
        raise AvailabilityIntegrityError("availability report is invalid") from error
    if path.name != report.report_id.removeprefix("sha256:"):
        raise AvailabilityIntegrityError(
            "availability report path does not match its identity"
        )
    return report


def _verify_existing(target: Path, expected: bytes) -> None:
    try:
        actual = (target / "availability.json").read_bytes()
    except OSError as error:
        raise AvailabilityPublicationError(
            "published availability report is unavailable"
        ) from error
    if actual != expected:
        raise AvailabilityIntegrityError("published availability report conflicts")


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        _artifacts.write_exclusive_bytes(path, payload)
    except OSError as error:
        raise AvailabilityPublicationError(
            "availability staging write failed"
        ) from error


def _next_month(month: date) -> date:
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)
