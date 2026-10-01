"""Shared Binance kline archive errors and wire admission."""

from __future__ import annotations

import re
import zipfile
from datetime import date
from decimal import Decimal
from typing import Literal

from crypto_boom.market import (
    EpochTimestamp,
    EvidenceAdmissionError,
    InstrumentId,
    KlineEvent,
    ObservationQuality,
    PayloadDigest,
    Provenance,
    TimeUnit,
)

DECODER_VERSION = "binance-spot-kline-csv-v1"
_UNSIGNED_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_UNSIGNED_INTEGER = re.compile(r"[0-9]+")
_MICROSECOND_ARCHIVE_START = date(2025, 1, 1)
_MICROSECONDS_PER_MINUTE = 60_000_000


class ArchiveError(RuntimeError):
    """The requested archive could not be safely acquired and published."""


class ArchiveTransportError(ArchiveError):
    """The official archive endpoint did not provide a bounded response."""


class ArchiveIntegrityError(ArchiveError):
    """Downloaded bytes do not satisfy the official checksum/archive contract."""


class ArchiveResourceError(ArchiveError):
    """An acquisition or decode exceeded its admitted resource envelope."""


class ArchiveSchemaError(ArchiveError):
    """Archive rows do not satisfy the declared source or canonical schema."""


class ArchivePublicationError(ArchiveError):
    """Validated staging data could not be atomically published."""


class ArchiveBatchError(ArchiveError):
    """A bounded range stopped at one identified daily partition."""


def admit_single_archive_member(
    archive: zipfile.ZipFile,
    *,
    expected_filename: str,
    maximum_bytes: int,
    subject: Literal["archive", "monthly archive"],
) -> zipfile.ZipInfo:
    """Admit the one expected bounded data member of a Binance archive ZIP."""

    members = [member for member in archive.infolist() if not member.is_dir()]
    if len(members) != 1:
        raise ArchiveIntegrityError(f"{subject} must contain exactly one data member")
    member = members[0]
    if member.filename != expected_filename:
        raise ArchiveIntegrityError(f"{subject} member name is unexpected")
    if member.file_size > maximum_bytes:
        raise ArchiveIntegrityError(f"{subject} member exceeds byte limit")
    return member


def decode_binance_kline_row(
    row: list[str],
    *,
    row_number: int,
    instrument: InstrumentId,
    interval: str,
    member_filename: str,
    timestamp_unit: TimeUnit,
    provenance: Provenance,
    payload_digest: PayloadDigest,
    quality: ObservationQuality,
) -> KlineEvent:
    if len(row) != 12:
        raise ArchiveSchemaError("archive kline row must have exactly 12 columns")

    try:
        event = KlineEvent(
            instrument=instrument,
            raw_symbol=instrument.symbol,
            interval=interval,
            source_event_time=None,
            open_time=EpochTimestamp(
                _parse_integer(row[0], "open time"),
                timestamp_unit,
            ),
            open_price=_parse_decimal(row[1], "open price"),
            high_price=_parse_decimal(row[2], "high price"),
            low_price=_parse_decimal(row[3], "low price"),
            close_price=_parse_decimal(row[4], "close price"),
            base_volume=_parse_decimal(row[5], "base volume"),
            close_time=EpochTimestamp(
                _parse_integer(row[6], "close time"),
                timestamp_unit,
            ),
            quote_turnover=_parse_decimal(row[7], "quote turnover"),
            trade_count=_parse_integer(row[8], "trade count"),
            taker_buy_base_volume=_parse_decimal(
                row[9],
                "taker-buy base volume",
            ),
            taker_buy_quote_turnover=_parse_decimal(
                row[10],
                "taker-buy quote turnover",
            ),
            first_trade_id=None,
            last_trade_id=None,
            closed=True,
            provenance=Provenance(
                source=provenance.source,
                ingestion_run_id=provenance.ingestion_run_id,
                receipt=provenance.receipt,
                payload_digest=payload_digest,
                source_revision=provenance.source_revision,
                raw_payload_reference=f"{member_filename}#row={row_number}",
            ),
            quality=quality,
        )
        _parse_decimal(row[11], "ignored field")
    except EvidenceAdmissionError as error:
        raise ArchiveSchemaError(
            "archive row violates the canonical kline contract"
        ) from error
    return event


def validate_one_minute_kline_time(
    event: KlineEvent,
    *,
    window_start_us: int,
    window_end_us: int,
    previous_open_time_us: int | None,
    window_name: str,
) -> None:
    open_time_us = event.open_time.epoch_microseconds
    close_time_us = event.close_time.epoch_microseconds
    if not window_start_us <= open_time_us < window_end_us:
        raise ArchiveSchemaError(f"archive open time is outside the {window_name}")
    if open_time_us % _MICROSECONDS_PER_MINUTE != 0:
        raise ArchiveSchemaError("archive open time is not minute-aligned")
    raw_minute = 60_000 if event.open_time.unit is TimeUnit.MILLISECOND else 60_000_000
    if event.close_time.raw_value != event.open_time.raw_value + raw_minute - 1:
        raise ArchiveSchemaError("archive close time does not match one-minute grain")
    if close_time_us >= window_end_us:
        raise ArchiveSchemaError(f"archive close time is outside the {window_name}")
    if previous_open_time_us is not None and open_time_us <= previous_open_time_us:
        raise ArchiveSchemaError("archive open times are not strictly increasing")


def archive_timestamp_unit(day: date) -> TimeUnit:
    if day >= _MICROSECOND_ARCHIVE_START:
        return TimeUnit.MICROSECOND
    return TimeUnit.MILLISECOND


def _parse_integer(raw: str, name: str) -> int:
    if _UNSIGNED_INTEGER.fullmatch(raw) is None:
        raise ArchiveSchemaError(f"archive {name} is not an unsigned integer")
    return int(raw)


def _parse_decimal(raw: str, name: str) -> Decimal:
    if _UNSIGNED_DECIMAL.fullmatch(raw) is None:
        raise ArchiveSchemaError(f"archive {name} is not an unsigned decimal")
    return Decimal(raw)
