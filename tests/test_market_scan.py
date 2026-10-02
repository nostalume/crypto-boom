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
from crypto_boom.config import ProjectSettings, project_settings
from crypto_boom.features import SequenceRecipe, sequence_matrix
from crypto_boom.market_data import ScanStopped, SpotSnapshotClient
from crypto_boom.market_scan import scan_market
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
                "label": "\u6700\u5927\u4e0a\u6da8\u7a7a\u95f4 P90",
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
def test_full_scope_common_clock_and_partial_ledger(
    tmp_path, monkeypatch, mode, caplog
):
    caplog.set_level("INFO", logger="crypto_boom.market_scan")
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

    async def pool(session, raw, *, deadline):
        products = [
            {"source": "binance_spot", "base_ticker": s[:-4], "instrument_id": s}
            for s in members
        ]
        products += [
            {
                "source": "okx_perpetual",
                "base_ticker": base,
                "instrument_id": base + "-USDT-SWAP",
            }
            for base in ("AAA", "UNLISTED")
        ]
        return {
            "status": "partial",
            "http_requests": 0,
            "sources": {"binance_spot": {"state": "success"}},
            "products": products,
        }

    monkeypatch.setattr("crypto_boom.market_scan.collect_product_pool", pool)
    monkeypatch.setattr(SpotSnapshotClient, "universe", universe)
    monkeypatch.setattr(SpotSnapshotClient, "bars", fetch)
    result = asyncio.run(scan_market(ProjectSettings(tmp_path, workers=1)))
    assert "Universe acquired: 3 members" in caplog.text
    assert "Product metadata is partial" in caplog.text
    assert "Report published:" in caplog.text
    assert f"Scan {result['status']}:" in caplog.text
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
    prose = (report / "report.md").read_text(encoding="utf-8")
    assert prose.isascii() and "Market-wide prediction report" in prose
    assert result["model_contract"]["outputs"][0]["label"].startswith("\u6700")
    assert result["schema"] == "market-scan-v2"
    assert result["product_metadata_status"] == "partial"
    assert (
        result["rows"][0]["product_candidates"][-1]["mapping_status"]
        == "ticker_match_unverified"
    )
    assert (
        result["product_pool"]["products"][-1]["binance_spot_prediction_state"]
        == "not_covered"
    )
    assert "UNLISTED-USDT-SWAP" in (report / "products.csv").read_text(
        encoding="utf-8-sig"
    )

    assert (
        json.loads((report / "report.json").read_text(encoding="utf-8"))["counts"]
        == result["counts"]
    )


def test_no_active_model_and_strict_config(tmp_path):
    with pytest.raises(ValueError, match="no active"):
        asyncio.run(scan_market(ProjectSettings(tmp_path)))
    with pytest.raises(ValueError, match="does not exist"):
        project_settings(tmp_path / "missing.toml")
    path = tmp_path / "crypto-boom.toml"
    path.write_text('[data]\nroot="state"\n[scan]\nworkers=2\n')
    assert project_settings(path).data_root == tmp_path / "state"
    path.write_text('[scan]\nsymbol="AAAUSDT"\n')
    with pytest.raises(ValueError, match="unknown"):
        project_settings(path)


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


@pytest.mark.parametrize("mutate", [False, True])
def test_optional_input_evidence_preserves_values_and_single_compute(
    tmp_path, monkeypatch, mutate
):
    from crypto_boom import model_runtime

    bundle(tmp_path)
    models, manifest, _ = load_active_model(tmp_path / "models")
    source = bars(1500)
    calls = 0
    original = model_runtime.sequence_matrix

    def matrix(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(model_runtime, "sequence_matrix", matrix)
    if mutate:

        class MutatingEstimator:
            n_features_in_ = RECIPE.feature_count

            def predict(self, values):
                values[0, 0] += 1
                return np.array([0.1])

        models = {"up": MutatingEstimator()}
    normal = predict_bars(models, manifest, source, decision_us=BASE + 1500 * MINUTE_US)
    captured = predict_bars(
        models,
        manifest,
        source,
        decision_us=BASE + 1500 * MINUTE_US,
        include_inputs=True,
    )
    assert calls == 2
    assert captured["values"] == normal
    if mutate:
        assert captured["inputs"]["state"] == "unavailable"
        assert captured["inputs"]["vector"] is None
    else:
        vector = captured["inputs"]["vector"]
        assert vector.dtype == np.float32 and vector.shape == (RECIPE.feature_count,)
        assert not vector.flags.writeable
        np.testing.assert_array_equal(
            vector,
            original(
                source,
                pl.DataFrame(
                    {"symbol": ["AAAUSDT"], "decision_us": [BASE + 1500 * MINUTE_US]}
                ),
                RECIPE,
            )[0],
        )


@pytest.mark.parametrize("mode", ["complete", "write_failure", "budget", "cancel"])
def test_scan_input_publication_is_independent(tmp_path, monkeypatch, mode):
    import pyarrow.parquet as pq

    from crypto_boom import _artifacts, market_scan

    bundle(tmp_path)
    source = bars(1500)

    async def universe(self):
        return (
            ["AAAUSDT", "BBBUSDT"],
            {"scope": "fixture", "metadata_refusals": []},
            b"{}",
            (BASE + 1500 * MINUTE_US) // 1000,
        )

    async def fetch(self, symbol, **kwargs):
        return source.with_columns(pl.lit(symbol).alias("symbol")), {
            "cache_id": "a" * 64,
            "source_sha256": "sha256:" + "b" * 64,
            "reused": True,
        }

    async def pool(*args, **kwargs):
        return {
            "status": "complete",
            "http_requests": 0,
            "sources": {"binance_spot": {"state": "success"}},
            "products": [],
        }

    monkeypatch.setattr(SpotSnapshotClient, "universe", universe)
    monkeypatch.setattr(SpotSnapshotClient, "bars", fetch)
    monkeypatch.setattr(market_scan, "collect_product_pool", pool)
    if mode in {"write_failure", "cancel"}:

        def fail(table, path, **kwargs):
            path.write_bytes(b"partial")
            if mode == "cancel":
                raise KeyboardInterrupt
            raise OSError("disk failure")

        monkeypatch.setattr(pq, "write_table", fail)
    if mode == "budget":
        monkeypatch.setattr(market_scan, "MAX_INPUT_BYTES", RECIPE.feature_count * 4)
    if mode == "cancel":
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(scan_market(ProjectSettings(tmp_path), record_inputs=True))
        assert not list((tmp_path / "reports").iterdir())
        return
    result = asyncio.run(scan_market(ProjectSettings(tmp_path), record_inputs=True))
    assert result["status"] == "complete" and result["counts"] == {"success": 2}
    assert result["schema"] == "market-scan-v3"
    from pathlib import Path

    directory = Path(result["report_directory"])
    evidence = result["input_evidence"]
    assert (
        evidence["state"]
        == {
            "complete": "complete",
            "write_failure": "unavailable",
            "budget": "partial",
        }[mode]
    )
    if mode == "write_failure":
        assert not (directory / "inputs.parquet").exists()
        assert all(
            r["input_evidence"]["reason"] == "publication_failed"
            for r in result["rows"]
        )
    else:
        table = pq.read_table(directory / "inputs.parquet")
        assert len(table) == evidence["rows"] == (1 if mode == "budget" else 2)
        assert (
            evidence["sha256"]
            == _artifacts.file_identity(directory / "inputs.parquet")[0]
        )
        assert table["features"].type.list_size == RECIPE.feature_count
    assert (
        json.loads((directory / "report.json").read_text(encoding="utf-8"))[
            "input_evidence"
        ]
        == evidence
    )
    assert not list((tmp_path / "reports").glob("scan-*"))


def test_scan_cli_explicit_input_recording(tmp_path, monkeypatch, capsys):
    from crypto_boom import cli, market_scan

    config = tmp_path / "crypto-boom.toml"
    config.write_text('[data]\nroot="data"\n')

    async def scan(settings, *, record_inputs=False):
        assert record_inputs is True
        return {
            "status": "complete",
            "model_id": "fixture",
            "eligible_symbols": 0,
            "product_metadata_status": "complete",
            "counts": {},
            "report_directory": str(tmp_path),
            "input_evidence": {"state": "unavailable"},
        }

    monkeypatch.setattr(market_scan, "scan_market", scan)
    assert cli.main(["scan", "--record-inputs", "--config", str(config)]) == 0
    assert (
        json.loads(capsys.readouterr().out)["input_evidence"]["state"] == "unavailable"
    )


@pytest.mark.parametrize("conflict", [False, True])
def test_snapshot_concurrent_publication_verifies_winner(
    tmp_path, monkeypatch, conflict
):
    import os
    import shutil

    from crypto_boom import _artifacts

    async def get(self, path, params, **kwargs):
        return json.dumps(
            [
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
        ).encode()

    def race(source, target):
        shutil.copytree(source, target)
        if conflict:
            (target / "bars.parquet").write_bytes(b"conflicting winner")
        raise FileExistsError("competing publisher won")

    monkeypatch.setattr(SpotSnapshotClient, "get", get)
    monkeypatch.setattr(os, "replace", race)

    async def execute():
        async with aiohttp.ClientSession() as session:
            client = SpotSnapshotClient(session)
            if conflict:
                with pytest.raises(ValueError, match="concurrent source cache"):
                    await client.bars(
                        "AAAUSDT",
                        decision_us=BASE + 10 * MINUTE_US,
                        history_minutes=1,
                        cache=tmp_path,
                    )
            else:
                frame, receipt = await client.bars(
                    "AAAUSDT",
                    decision_us=BASE + 10 * MINUTE_US,
                    history_minutes=1,
                    cache=tmp_path,
                )
                assert receipt["reused"] and len(frame) == 2
                assert (
                    _artifacts.file_identity(
                        tmp_path / receipt["cache_id"] / "bars.parquet"
                    )[0]
                    == receipt["source_sha256"]
                )

    asyncio.run(execute())
    assert not list(tmp_path.glob("snapshot-*"))
    if conflict:
        assert (
            next(tmp_path.glob("*/bars.parquet")).read_bytes() == b"conflicting winner"
        )
