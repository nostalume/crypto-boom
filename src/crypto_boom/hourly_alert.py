"""Read-only hourly Binance Spot volume-shock alert trial.

This is an attention filter, not a return forecast or an order executor.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from threading import Lock
from typing import Any

HOUR_MS = 3_600_000
HISTORY_HOURS = 168
API_BASE = "https://data-api.binance.vision"
GITHUB_API_BASE = "https://api.github.com"
README_START = "<!-- HOURLY_ALERT_STATUS_START -->"
README_END = "<!-- HOURLY_ALERT_STATUS_END -->"
RECEIPT_TAG = "HOURLY_ALERT_RECEIPT_V1"
RECEIPT_KEYS = (
    "bar_close_utc",
    "binance_server_time_utc",
    "scan_started_utc",
    "scan_completed_utc",
    "eligible_symbols",
    "evaluated_symbols",
    "unavailable_history_symbols",
    "signals",
)
MAX_SYMBOLS = 1000
MAX_RESPONSE_BYTES = 2_000_000
MAX_EXCHANGE_INFO_BYTES = 32 * 1024 * 1024
EXCLUDED_BASES = frozenset(
    {
        "BTC",
        "ETH",
        "BNB",
        "USDC",
        "FDUSD",
        "TUSD",
        "USDP",
        "DAI",
        "BUSD",
        "EUR",
        "TRY",
        "BRL",
    }
)


@dataclass(frozen=True)
class HourBar:
    open_ms: int
    close: float
    quote_volume: float
    taker_buy_quote: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    stage: str
    bar_open_ms: int
    quote_volume_usdt: float
    volume_ratio: float
    volume_zscore: float
    ema_premium: float
    net_taker_quote_usdt: float


class UnavailableHistory(Exception):
    """A symbol lacks the required contiguous completed-hour context."""


def _get_json(path: str, params: dict[str, object] | None = None) -> Any:
    url = API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "crypto-boom-alert/0.1"}
    )
    maximum_bytes = (
        MAX_EXCHANGE_INFO_BYTES
        if path == "/api/v3/exchangeInfo"
        else MAX_RESPONSE_BYTES
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read(maximum_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Binance HTTP {exc.code}; scan aborted") from None
    except urllib.error.URLError:
        raise RuntimeError("Binance market-data request failed; scan aborted") from None
    if len(raw) > maximum_bytes:
        raise ValueError("Binance response exceeds the size bound")
    return json.loads(raw)


def eligible_symbols(exchange_info: object) -> tuple[str, ...]:
    if not isinstance(exchange_info, dict) or not isinstance(
        exchange_info.get("symbols"), list
    ):
        raise ValueError("Invalid exchangeInfo response")
    selected: set[str] = set()
    for row in exchange_info["symbols"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid exchangeInfo symbol")
        if (
            row.get("status") != "TRADING"
            or row.get("quoteAsset") != "USDT"
            or row.get("isSpotTradingAllowed") is not True
            or row.get("baseAsset") in EXCLUDED_BASES
        ):
            continue
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{4,30}", symbol):
            # The current source can contain non-ASCII pair IDs. This bounded
            # trial declines them rather than translating ticker identity.
            continue
        selected.add(symbol)
    if not selected or len(selected) > MAX_SYMBOLS:
        raise ValueError("Eligible symbol count is empty or exceeds bound")
    return tuple(sorted(selected))


def parse_klines(payload: object) -> tuple[HourBar, ...]:
    if not isinstance(payload, list) or len(payload) < HISTORY_HOURS + 2:
        return ()
    bars = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 11:
            raise ValueError("Invalid Binance kline row")
        bar = HourBar(int(row[0]), float(row[4]), float(row[7]), float(row[10]))
        if (
            bar.close <= 0
            or bar.quote_volume < 0
            or bar.taker_buy_quote < 0
            or bar.taker_buy_quote > bar.quote_volume + 1e-6
            or not all(
                math.isfinite(value)
                for value in (bar.close, bar.quote_volume, bar.taker_buy_quote)
            )
        ):
            raise ValueError("Invalid kline price or quote volume")
        bars.append(bar)
    if any(b.open_ms - a.open_ms != HOUR_MS for a, b in pairwise(bars)):
        return ()
    return tuple(bars)


def stage(bars: tuple[HourBar, ...], index: int) -> tuple[int, tuple[float, ...]]:
    """Classify one bar against its trailing history.

    Returns the stage (0 = quiet, 1 = watch, 2 = confirmed) and the five metrics the
    rule is built from. This is public because a separate historical replay
    re-runs exactly this decision over recorded history, and a private name is
    not a contract: it would let this module
    change the rule without the replay noticing it had stopped matching.
    """

    history = bars[index - HISTORY_HOURS : index]
    current = bars[index]
    volumes = [bar.quote_volume for bar in history]
    mean = statistics.fmean(volumes)
    if mean <= 0:
        return 0, ()
    deviation = statistics.pstdev(volumes)
    if deviation <= 0:
        return 0, ()
    ratio = current.quote_volume / mean
    zscore = (current.quote_volume - mean) / deviation
    ema = history[0].close
    alpha = 2 / 21
    for bar in history[1:]:
        ema += alpha * (bar.close - ema)
    premium = current.close / ema - 1
    net_taker = 2 * current.taker_buy_quote - current.quote_volume
    metrics = (current.quote_volume, ratio, zscore, premium, net_taker)
    watch = (
        current.quote_volume >= 100_000
        and ratio >= 3
        and zscore >= 3
        and premium >= 0
        and net_taker > 0
    )
    confirmed = (
        watch
        and premium >= 0.05
        and net_taker >= 500_000
        and current.quote_volume >= 500_000
    )
    return (2 if confirmed else 1 if watch else 0), metrics


def evaluate_symbol(
    symbol: str, payload: object, expected_open_ms: int
) -> Signal | None:
    bars = parse_klines(payload)
    if len(bars) < HISTORY_HOURS + 2 or bars[-1].open_ms != expected_open_ms:
        raise UnavailableHistory(symbol)
    current, metrics = stage(bars, len(bars) - 1)
    previous, _ = stage(bars, len(bars) - 2)
    if current <= previous:
        return None
    return Signal(
        symbol=symbol,
        stage="CONFIRMED" if current == 2 else "WATCH",
        bar_open_ms=expected_open_ms,
        quote_volume_usdt=metrics[0],
        volume_ratio=metrics[1],
        volume_zscore=metrics[2],
        ema_premium=metrics[3],
        net_taker_quote_usdt=metrics[4],
    )


class RequestPacer:
    """Bound request start rate across worker threads to eight per second."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + 0.125
        if delay:
            time.sleep(delay)


def scan() -> dict[str, object]:
    started_utc = datetime.now(UTC).isoformat()
    server = _get_json("/api/v3/time")
    if not isinstance(server, dict) or not isinstance(server.get("serverTime"), int):
        raise ValueError("Invalid Binance server time")
    server_ms = server["serverTime"]
    expected_open_ms = server_ms // HOUR_MS * HOUR_MS - HOUR_MS
    if not 0 <= server_ms - (expected_open_ms + HOUR_MS) <= 45 * 60_000:
        raise ValueError("Last completed bar is too old for a timely alert")
    symbols = eligible_symbols(_get_json("/api/v3/exchangeInfo"))
    pacer = RequestPacer()

    def fetch(symbol: str) -> tuple[bool, Signal | None]:
        pacer.wait()
        payload = _get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1h",
                "endTime": expected_open_ms + HOUR_MS - 1,
                "limit": 170,
            },
        )
        try:
            return True, evaluate_symbol(symbol, payload, expected_open_ms)
        except UnavailableHistory:
            return False, None

    signals: list[Signal] = []
    evaluated = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            available, result = future.result()  # A source fault aborts the full scan.
            evaluated += available
            if result is not None:
                signals.append(result)
    signals.sort(
        key=lambda item: (item.stage != "CONFIRMED", -item.volume_ratio, item.symbol)
    )
    return {
        "bar_close_utc": datetime.fromtimestamp(
            (expected_open_ms + HOUR_MS) / 1000, tz=UTC
        ).isoformat(),
        "binance_server_time_utc": datetime.fromtimestamp(
            server_ms / 1000, tz=UTC
        ).isoformat(),
        "scan_started_utc": started_utc,
        "scan_completed_utc": datetime.now(UTC).isoformat(),
        "eligible_symbols": len(symbols),
        "evaluated_symbols": evaluated,
        "unavailable_history_symbols": len(symbols) - evaluated,
        "signals": [asdict(item) for item in signals],
    }


def format_readme_status(report: dict[str, object]) -> str:
    signals = report["signals"]
    assert isinstance(signals, list)
    lines = [
        f"**Last complete scan:** {report['bar_close_utc']} (UTC hour close).  ",
        f"**New conditions:** {len(signals)}; evaluated "
        f"{report['evaluated_symbols']} / eligible {report['eligible_symbols']}; "
        f"unavailable history {report['unavailable_history_symbols']}.",
    ]
    if signals:
        lines.extend(
            [
                "",
                "| Stage | Pair | Turnover (USDT) | Ratio | Z-score | EMA premium | Net taker (USDT) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for signal in signals[:20]:
            lines.append(
                f"| {signal['stage']} | {signal['symbol']} | "
                f"{signal['quote_volume_usdt']:,.0f} | {signal['volume_ratio']:.1f} | "
                f"{signal['volume_zscore']:.1f} | {signal['ema_premium']:+.1%} | "
                f"{signal['net_taker_quote_usdt']:,.0f} |"
            )
        if len(signals) > 20:
            lines.append(f"\n{len(signals) - 20} more matches are in the run log.")
    else:
        lines.append("\nNo new WATCH or CONFIRMED condition in this scan.")
    lines.append(
        "\nAttention filter only; not a return forecast or buy recommendation."
    )
    public_fields = {key: report[key] for key in RECEIPT_KEYS if key in report}
    receipt = json.dumps(
        {"schema_version": 1, **public_fields},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(receipt.encode("utf-8")).decode("ascii")
    lines.append(f"<!-- {RECEIPT_TAG}:{encoded} -->")
    return "\n".join(lines)


def _github_json(method: str, path: str, payload: dict | None = None) -> Any:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GitHub Actions token is not configured")
    request = urllib.request.Request(
        GITHUB_API_BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "crypto-boom-alert/0.1",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"GitHub Contents HTTP {exc.code}; token not logged"
        ) from None
    except urllib.error.URLError:
        raise RuntimeError("GitHub Contents request failed; token not logged") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("GitHub Contents response exceeds the size bound")
    return json.loads(raw)


def replace_readme_status(readme: str, report: dict[str, object]) -> str:
    if readme.count(README_START) != 1 or readme.count(README_END) != 1:
        raise ValueError("README status markers are missing or ambiguous")
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    if start >= end:
        raise ValueError("README status markers are out of order")
    return readme[:start] + "\n" + format_readme_status(report) + "\n" + readme[end:]


def status_commit_message(report: dict[str, object]) -> str:
    signals = report["signals"]
    assert isinstance(signals, list)
    timestamp = str(report["bar_close_utc"]).replace("T", " ")[:16] + " UTC"
    if not signals:
        return f"NO NEW ALTCOIN SIGNALS {timestamp}"
    prefix = f"ALTCOIN ALERT {timestamp}"
    names = [f"{signal['stage']}:{signal['symbol']}" for signal in signals[:5]]
    while names:
        suffix = (
            f" +{len(signals) - len(names)} more" if len(signals) > len(names) else ""
        )
        if len(prefix) + 1 + len(" ".join(names)) + len(suffix) <= 120:
            break
        names.pop()
    suffix = f" +{len(signals) - len(names)} more" if len(signals) > len(names) else ""
    return f"{prefix} {' '.join(names)}{suffix}"


def publish_readme(report: dict[str, object]) -> str:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise RuntimeError("GITHUB_REPOSITORY is not configured")
    branch = os.environ.get("GITHUB_REF_NAME", "")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
        raise RuntimeError("GITHUB_REF_NAME is not a valid branch")
    path = f"/repos/{repository}/contents/README.md"
    current = _github_json("GET", path + "?ref=" + urllib.parse.quote(branch, safe=""))
    if (
        not isinstance(current, dict)
        or not all(isinstance(current.get(key), str) for key in ("sha", "content"))
        or current.get("encoding") != "base64"
    ):
        raise ValueError("Invalid GitHub README response")
    try:
        readme = base64.b64decode(
            current["content"].replace("\n", ""), validate=True
        ).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        raise ValueError("Invalid GitHub README encoding") from None
    updated = replace_readme_status(readme, report)
    if updated == readme:
        return f"https://github.com/{repository}/blob/{branch}/README.md"
    result = _github_json(
        "PUT",
        path,
        {
            "message": status_commit_message(report),
            "content": base64.b64encode(updated.encode("utf-8")).decode("ascii"),
            "sha": current["sha"],
            "branch": branch,
        },
    )
    if not isinstance(result, dict) or not isinstance(result.get("commit"), dict):
        raise ValueError("GitHub did not confirm the README update")
    return f"https://github.com/{repository}/blob/{branch}/README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish-readme",
        action="store_true",
        help="Publish the scan in public README",
    )
    args = parser.parse_args()
    if args.publish_readme and not (
        os.environ.get("GITHUB_TOKEN")
        and os.environ.get("GITHUB_REPOSITORY")
        and os.environ.get("GITHUB_REF_NAME")
    ):
        raise RuntimeError("GitHub Actions token, repository and branch are required")
    report = scan()
    print(json.dumps(report, allow_nan=False, separators=(",", ":")))
    if args.publish_readme:
        url = publish_readme(report)
        print(f"Public status: {url}")
    if sys.platform == "linux":
        import resource

        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Peak scanner process RSS: {peak_mib:.1f} MiB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Alert scan refused: {exc}", file=sys.stderr)
        sys.exit(1)
