"""Shared bounded archive transfer; callers own retries and staging cleanup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import aiohttp

from crypto_boom.history.codec import ArchivePublicationError, ArchiveTransportError


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    destination: Path,
    *,
    maximum: int,
    chunk_bytes: int,
    description: str = "archive",
    limit_description: str = "archive response",
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise ArchiveTransportError(
                    f"{description} endpoint returned a non-success status"
                )
            if (
                response.content_length is not None
                and response.content_length > maximum
            ):
                raise ArchiveTransportError(f"{limit_description} exceeds byte limit")
            with destination.open("xb") as stream:
                async for chunk in response.content.iter_chunked(chunk_bytes):
                    size += len(chunk)
                    if size > maximum:
                        raise ArchiveTransportError(
                            f"{limit_description} exceeds byte limit"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except (TimeoutError, aiohttp.ClientError) as error:
        raise ArchiveTransportError(f"{description} request failed") from error
    except OSError as error:
        raise ArchivePublicationError(f"{description} staging write failed") from error
    return digest.hexdigest(), size
