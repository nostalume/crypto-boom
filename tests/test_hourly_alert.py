"""Synthetic, network-free tests for the hourly attention filter."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from typing import cast

import pytest

from crypto_boom.research import hourly_alert as alert


def klines(
    *,
    current_quote: float = 600_000,
    current_buy: float = 550_000,
    current_close: float = 106,
    previous_quote: float | None = None,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for index in range(170):
        quote = float(100_000 if index % 2 == 0 else 120_000)
        close = 100.0
        buy = quote / 2
        if index == 168 and previous_quote is not None:
            quote = previous_quote
            buy = quote * 0.9
            close = 101.0
        if index == 169:
            quote, buy, close = current_quote, current_buy, current_close
        rows.append(
            [
                index * alert.HOUR_MS,
                "100",
                "107",
                "99",
                str(close),
                "1",
                (index + 1) * alert.HOUR_MS - 1,
                str(quote),
                1,
                "1",
                str(buy),
                "0",
            ]
        )
    return rows


def test_confirmed_signal_uses_only_completed_history_and_taker_quote() -> None:
    rows = klines()
    signal = alert.evaluate_symbol("TESTUSDT", rows, 169 * alert.HOUR_MS)
    assert signal is not None
    assert signal.stage == "CONFIRMED"
    assert signal.net_taker_quote_usdt == 500_000
    assert signal.ema_premium == pytest.approx(0.06)
    assert signal.volume_ratio > 5
    assert signal.volume_zscore > 3


def test_watch_is_distinct_from_confirmation() -> None:
    signal = alert.evaluate_symbol(
        "TESTUSDT",
        klines(current_quote=360_000, current_buy=220_000, current_close=101),
        169 * alert.HOUR_MS,
    )
    assert signal is not None and signal.stage == "WATCH"


def test_no_signal_without_fresh_completed_and_contiguous_bar() -> None:
    rows = klines()
    with pytest.raises(alert.UnavailableHistory):
        alert.evaluate_symbol("TESTUSDT", rows, 170 * alert.HOUR_MS)
    rows[60][0] = cast(int, rows[60][0]) + 1
    with pytest.raises(alert.UnavailableHistory):
        alert.evaluate_symbol("TESTUSDT", rows, 169 * alert.HOUR_MS)


def test_no_signal_if_hourly_turnover_has_zero_prior_variance() -> None:
    rows = klines()
    for row in rows[:-1]:
        row[7] = "100000"
        row[10] = "50000"
    assert alert.evaluate_symbol("TESTUSDT", rows, 169 * alert.HOUR_MS) is None


def test_no_repeated_watch_when_previous_hour_already_qualified() -> None:
    rows = klines(current_quote=500_000, current_buy=400_000, current_close=101)
    for row in rows[:168]:
        row[7] = "1000" if cast(int, row[0]) % (2 * alert.HOUR_MS) == 0 else "1200"
        row[10] = str(float(row[7]) / 2)
    rows[168][7], rows[168][10], rows[168][4] = "400000", "300000", "101"
    assert alert.evaluate_symbol("TESTUSDT", rows, 169 * alert.HOUR_MS) is None


def test_universe_rejects_currently_ineligible_and_does_not_truncate() -> None:
    def row(symbol: str, base: str, status: str = "TRADING") -> dict:
        return {
            "symbol": symbol,
            "baseAsset": base,
            "quoteAsset": "USDT",
            "status": status,
            "isSpotTradingAllowed": True,
        }

    info = {
        "symbols": [
            row("NEARUSDT", "NEAR"),
            row("BTCUSDT", "BTC"),
            row("OLDUSDT", "OLD", "BREAK"),
        ]
    }
    assert alert.eligible_symbols(info) == ("NEARUSDT",)
    assert alert.eligible_symbols(
        {"symbols": [*info["symbols"], row("PAIR-USDUSDT", "PAIR")]}
    ) == ("NEARUSDT",)
    with pytest.raises(ValueError, match="exceeds bound"):
        alert.eligible_symbols(
            {
                "symbols": [
                    row(f"X{n:04}USDT", f"X{n:04}")
                    for n in range(alert.MAX_SYMBOLS + 1)
                ]
            }
        )


def test_scan_and_readme_status_without_external_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = 169 * alert.HOUR_MS
    calls = []

    def fake_get(path: str, params: dict | None = None) -> object:
        calls.append((path, params))
        if path.endswith("/time"):
            return {"serverTime": expected + alert.HOUR_MS + 7 * 60_000}
        if path.endswith("/exchangeInfo"):
            return {
                "symbols": [
                    {
                        "symbol": "TESTUSDT",
                        "baseAsset": "TEST",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "isSpotTradingAllowed": True,
                    }
                ]
            }
        assert params == {
            "symbol": "TESTUSDT",
            "interval": "1h",
            "endTime": expected + alert.HOUR_MS - 1,
            "limit": 170,
        }
        return klines()

    monkeypatch.setattr(alert, "_get_json", fake_get)
    report = alert.scan()
    assert datetime.fromisoformat(
        str(report["scan_started_utc"])
    ) <= datetime.fromisoformat(str(report["scan_completed_utc"]))
    assert report["eligible_symbols"] == 1
    assert report["evaluated_symbols"] == 1
    assert report["unavailable_history_symbols"] == 0
    signals = report["signals"]
    assert isinstance(signals, list)
    assert len(signals) == 1
    assert isinstance(signals[0], dict)
    assert signals[0]["stage"] == "CONFIRMED"
    assert "TESTUSDT" in alert.format_readme_status(report)
    json.dumps(report, allow_nan=False)
    assert len(calls) == 3


def test_readme_status_replaces_only_the_bounded_region() -> None:
    old = (
        "# Heading\n" + alert.README_START + "\nold\n" + alert.README_END + "\nfooter\n"
    )
    report: dict[str, object] = {
        "bar_close_utc": "2026-09-27T15:00:00+00:00",
        "eligible_symbols": 1,
        "evaluated_symbols": 1,
        "unavailable_history_symbols": 0,
        "signals": [],
        "private_debug": "do-not-publish",
    }
    updated = alert.replace_readme_status(old, report)
    assert updated.startswith("# Heading\n" + alert.README_START)
    assert updated.endswith(alert.README_END + "\nfooter\n")
    assert "No new WATCH or CONFIRMED" in updated
    encoded = re.search(r"<!-- HOURLY_ALERT_RECEIPT_V1:([A-Za-z0-9_=-]+) -->", updated)
    assert encoded is not None
    receipt = json.loads(base64.urlsafe_b64decode(encoded[1]).decode())
    assert receipt == {
        "schema_version": 1,
        **{key: value for key, value in report.items() if key != "private_debug"},
    }
    assert "do-not-publish" not in updated
    with pytest.raises(ValueError, match="markers"):
        alert.replace_readme_status("# No status", report)
    assert (
        alert.status_commit_message(report)
        == "NO NEW ALTCOIN SIGNALS 2026-09-27 15:00 UTC"
    )


def test_commit_subject_names_ranked_matches_and_bounds_length() -> None:
    report: dict[str, object] = {
        "bar_close_utc": "2026-09-27T15:00:00+00:00",
        "signals": [
            {"stage": "CONFIRMED", "symbol": "FIRSTUSDT"},
            *({"stage": "WATCH", "symbol": f"X{index}USDT"} for index in range(6)),
        ],
    }
    title = alert.status_commit_message(report)
    assert title.startswith(
        "ALTCOIN ALERT 2026-09-27 15:00 UTC CONFIRMED:FIRSTUSDT WATCH:X0USDT"
    )
    assert "WATCH:X3USDT +2 more" in title
    assert "X4USDT" not in title
    assert len(title) <= 120


def test_full_ranked_receipt_survives_the_visible_twenty_row_limit() -> None:
    signal = {
        "stage": "WATCH",
        "quote_volume_usdt": 100_000.0,
        "volume_ratio": 3.0,
        "volume_zscore": 3.0,
        "ema_premium": 0.01,
        "net_taker_quote_usdt": 1.0,
    }
    report: dict[str, object] = {
        "bar_close_utc": "2026-09-27T15:00:00+00:00",
        "eligible_symbols": 21,
        "evaluated_symbols": 21,
        "unavailable_history_symbols": 0,
        "signals": [{"symbol": f"X{i}USDT", **signal} for i in range(21)],
    }
    status = alert.format_readme_status(report)
    encoded = re.search(r"<!-- HOURLY_ALERT_RECEIPT_V1:([A-Za-z0-9_=-]+) -->", status)
    assert encoded is not None
    receipt = json.loads(base64.urlsafe_b64decode(encoded[1]))
    assert len(receipt["signals"]) == 21
    assert status.count("| WATCH |") == 20


def test_github_readme_publication_without_external_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    calls: list[tuple[str, str, dict | None]] = []
    original = "# Heading\n" + alert.README_START + "\nold\n" + alert.README_END + "\n"
    report: dict[str, object] = {
        "bar_close_utc": "2026-09-27T15:00:00+00:00",
        "eligible_symbols": 1,
        "evaluated_symbols": 1,
        "unavailable_history_symbols": 0,
        "signals": [],
    }

    def fake_github(method: str, path: str, payload: dict | None = None) -> object:
        calls.append((method, path, payload))
        if method == "GET":
            return {
                "sha": "prior-sha",
                "encoding": "base64",
                "content": base64.b64encode(original.encode()).decode(),
            }
        return {"commit": {"sha": "new-sha"}}

    monkeypatch.setattr(alert, "_github_json", fake_github)
    assert (
        alert.publish_readme(report)
        == "https://github.com/owner/repo/blob/main/README.md"
    )
    assert [call[0] for call in calls] == ["GET", "PUT"]
    assert calls[0][1].endswith("/contents/README.md?ref=main")
    assert calls[1][2] is not None
    assert calls[1][2]["sha"] == "prior-sha"
    assert calls[1][2]["message"] == "NO NEW ALTCOIN SIGNALS 2026-09-27 15:00 UTC"
    published = base64.b64decode(calls[1][2]["content"]).decode()
    assert "No new WATCH or CONFIRMED" in published
    assert published.endswith(alert.README_END + "\n")


def test_github_readme_fails_closed_on_malformed_remote_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    calls: list[str] = []

    def fake_github(method: str, path: str, payload: dict | None = None) -> object:
        calls.append(method)
        return {"sha": "prior-sha", "encoding": "base64", "content": ""}

    monkeypatch.setattr(alert, "_github_json", fake_github)
    with pytest.raises(ValueError, match="markers"):
        alert.publish_readme({"signals": []})
    assert calls == ["GET"]

    def invalid_base64(method: str, path: str, payload: dict | None = None) -> object:
        return {"sha": "prior-sha", "encoding": "base64", "content": "not base64!"}

    monkeypatch.setattr(alert, "_github_json", invalid_base64)
    with pytest.raises(ValueError, match="encoding"):
        alert.publish_readme({"signals": []})
