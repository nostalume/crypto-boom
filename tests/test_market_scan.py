"""Market-wide coverage, time consistency, registry safety and reusable scales."""

import asyncio
import json
from typing import cast

import aiohttp
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from crypto_boom.bars import MINUTE_US
from crypto_boom.features import SequenceRecipe, sequence_matrix
from crypto_boom.market_data import ScanStopped, SpotSnapshotClient
from crypto_boom.market_scan import ScanSettings, scan_market, scan_settings
from crypto_boom.model_runtime import (
    activate_model,
    load_active_model,
    predict_bars,
    publish_model,
)
from crypto_boom.research.hourly import RECIPE, hourly_matrix
from test_forward_prediction import BASE, bars


def bundle(tmp_path, recipe=RECIPE, cadence=60):
    rng = np.random.default_rng(0)
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=2, min_samples_leaf=2
    )
    with threadpool_limits(limits=2):
        model.fit(
            rng.normal(size=(40, recipe.feature_count)), rng.uniform(0.01, 0.1, 40)
        )
    registry = tmp_path / "models"
    record = publish_model(
        registry,
        {"up": model},
        recipe=recipe,
        decision_step_minutes=cadence,
        outputs=[
            {
                "name": "up",
                "label": "Upside",
                "horizon_minutes": 360,
                "unit": "return_fraction",
                "statistic": "quantile",
                "quantile": 0.9,
            }
        ],
        rank_by="up",
        provenance={"kind": "test"},
    )
    activate_model(registry, record["model_id"], trusted=True)
    return record


@pytest.mark.parametrize(
    "recipe,cadence",
    [(RECIPE, 60), (SequenceRecipe(120, 5, (1, 6, 24), (6, 24), 6, 6, 24), 5)],
)
def test_generic_recipe_and_activation(tmp_path, recipe, cadence):
    record = bundle(tmp_path, recipe, cadence)
    models, manifest, loaded = load_active_model(tmp_path / "models")
    assert loaded == recipe
    source = bars(1500)
    values = predict_bars(models, manifest, source, decision_us=BASE + 1500 * MINUTE_US)
    assert np.isfinite(values["up"])
    with pytest.raises(ValueError, match="trust"):
        activate_model(tmp_path / "models", record["model_id"])
    with pytest.raises(ValueError, match="ID"):
        activate_model(tmp_path / "models", "../other", trusted=True)
    weights = tmp_path / "models" / record["model_id"] / "weights.joblib"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_active_model(tmp_path / "models")


def test_extracted_recipe_exact_parity():
    source = bars(1800)
    origins = pl.DataFrame(
        {"symbol": ["AAAUSDT"], "decision_us": [BASE + 1800 * MINUTE_US]}
    )
    np.testing.assert_array_equal(
        sequence_matrix(source, origins, RECIPE), hourly_matrix(source, origins)
    )
    with pytest.raises(ValueError):
        SequenceRecipe(120, 5, (25,), (6,), 6, 6, 24)


@pytest.mark.parametrize("mode", ["complete", "data_error", "rate_stop", "timeout"])
def test_full_scope_common_clock_and_partial_ledger(tmp_path, monkeypatch, mode):
    bundle(tmp_path)
    members = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    decisions = []

    async def universe(self):
        return (
            members,
            {"scope": "fixture Spot USDT", "metadata_refusals": []},
            b"{}",
            (BASE + 1559 * MINUTE_US) // 1000,
        )

    async def fetch(self, symbol, *, decision_us, history_minutes, cache):
        decisions.append(decision_us)
        if symbol == "BBBUSDT" and mode == "timeout":
            raise TimeoutError
        if symbol == "BBBUSDT" and mode == "data_error":
            raise ValueError("missing history")
        if symbol == "BBBUSDT" and mode == "rate_stop":
            raise ScanStopped("HTTP 429")
        return bars(1500).with_columns(pl.lit(symbol).alias("symbol")), {
            "reused": False,
            "source_sha256": "fixture",
        }

    monkeypatch.setattr(SpotSnapshotClient, "universe", universe)
    monkeypatch.setattr(SpotSnapshotClient, "bars", fetch)
    result = asyncio.run(scan_market(ScanSettings(tmp_path, workers=1)))
    assert result["eligible_symbols"] == 3 and len(result["rows"]) == 3
    assert set(decisions) == {BASE + 1500 * MINUTE_US}
    assert (result["status"] == "complete") == (mode == "complete")
    if mode == "rate_stop":
        assert result["rows"][-1]["state"] == "not_attempted"
    if mode == "timeout":
        assert result["rows"][1]["reason"] == "TimeoutError"
    if mode == "data_error":
        assert result["counts"]["success"] == 2
    report = __import__("pathlib").Path(result["report_directory"])
    assert (report / "predictions.csv").is_file() and (report / "report.md").is_file()
    assert (
        json.loads((report / "report.json").read_text(encoding="utf-8"))["counts"]
        == result["counts"]
    )


def test_no_active_model_and_strict_config(tmp_path):
    with pytest.raises(ValueError, match="no active"):
        asyncio.run(scan_market(ScanSettings(tmp_path)))
    with pytest.raises(ValueError, match="does not exist"):
        scan_settings(tmp_path / "missing.toml")
    path = tmp_path / "scan.toml"
    path.write_text('[scan]\ndata_dir="state"\nworkers=2\n')
    assert scan_settings(path).data_dir == tmp_path / "state"
    path.write_text('[scan]\nsymbol="AAAUSDT"\n')
    with pytest.raises(ValueError, match="unknown"):
        scan_settings(path)


def test_snapshot_cache_exact_identity_and_missing_data(tmp_path, monkeypatch):
    calls = []

    async def get(self, path, params, **kwargs):
        calls.append(params)
        rows = [
            [
                params["startTime"] + i * 60000,
                "10",
                "11",
                "9",
                "10",
                "2",
                params["startTime"] + i * 60000 + 59999,
                "20",
                2,
                "1",
                "10",
                "0",
            ]
            for i in range(params["limit"])
        ]
        return json.dumps(rows).encode()

    monkeypatch.setattr(SpotSnapshotClient, "get", get)

    async def execute():
        async with aiohttp.ClientSession() as session:
            client = SpotSnapshotClient(session)
            frame, r = await client.bars(
                "AAAUSDT",
                decision_us=BASE + 1500 * MINUTE_US,
                history_minutes=1440,
                cache=tmp_path,
            )
            assert len(frame) == 1441 and len(calls) == 2
            _, again = await client.bars(
                "AAAUSDT",
                decision_us=BASE + 1500 * MINUTE_US,
                history_minutes=1440,
                cache=tmp_path,
            )
            assert again["reused"] and len(calls) == 2
            target = tmp_path / r["cache_id"] / "bars.parquet"
            target.write_bytes(target.read_bytes() + b"x")
            with pytest.raises(ValueError, match="hash"):
                await client.bars(
                    "AAAUSDT",
                    decision_us=BASE + 1500 * MINUTE_US,
                    history_minutes=1440,
                    cache=tmp_path,
                )

    asyncio.run(execute())


def test_http_rate_stop_does_not_retry():
    class Response:
        status = 429

        def __init__(self):
            self.headers = {"Retry-After": "60"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    async def execute():
        session = Session()
        client = SpotSnapshotClient(cast(aiohttp.ClientSession, session))
        with pytest.raises(ScanStopped, match="429"):
            await client.get("/api/v3/time", {}, weight=1)
        with pytest.raises(ScanStopped):
            await client.get("/api/v3/time", {}, weight=1)
        assert session.calls == 1

    asyncio.run(execute())


@pytest.mark.parametrize("mode", ["normal", "duplicate", "budget"])
def test_observed_universe_and_metadata_refusals(monkeypatch, mode):
    def member(symbol, allowed=True):
        return {
            "symbol": symbol,
            "status": "TRADING",
            "baseAsset": symbol[:-4],
            "quoteAsset": "USDT",
            "permissionSets": [["SPOT"]],
            "isSpotTradingAllowed": allowed,
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.001"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                {"filterType": "NOTIONAL", "minNotional": "5"},
            ],
        }

    members = [member("AAAUSDT"), member("BBBUSDT", False)]
    if mode == "duplicate":
        members.append(member("AAAUSDT"))
    payload = {
        "timezone": "UTC",
        "symbols": members,
        "rateLimits": [
            {
                "rateLimitType": "REQUEST_WEIGHT",
                "interval": "MINUTE",
                "intervalNum": 1,
                "limit": 10 if mode == "budget" else 1200,
            }
        ],
    }

    async def get(self, path, params, **kwargs):
        return json.dumps(
            payload if path.endswith("exchangeInfo") else {"serverTime": 123456789}
        ).encode()

    monkeypatch.setattr(SpotSnapshotClient, "get", get)

    async def execute():
        async with aiohttp.ClientSession() as session:
            client = SpotSnapshotClient(session)
            if mode != "normal":
                with pytest.raises(ValueError):
                    await client.universe()
            else:
                symbols, scope, _, clock = await client.universe()
                assert symbols == ["AAAUSDT"] and not scope["metadata_refusals"]
                assert clock == 123456789

    asyncio.run(execute())
