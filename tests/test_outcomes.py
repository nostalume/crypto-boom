"""Offline evidence joins: no market requests or fitted-model dependencies."""

import json

import polars as pl
import pytest

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US
from crypto_boom.research import outcomes
from test_forward_prediction import BASE, bars


@pytest.fixture
def pair(tmp_path):
    snapshots = tmp_path / "snapshots"
    contract: dict = {
        "outputs": [
            {
                "name": "up",
                "statistic": "quantile",
                "unit": "return_fraction",
                "horizon_minutes": 2,
                "quantile": 0.9,
            }
        ]
    }
    reports = []
    for i, n in enumerate([2, 4]):
        spec: dict = {
            "schema": "spot-minute-snapshot-v1",
            "symbol": "AAAUSDT",
            "decision_us": BASE + n * MINUTE_US,
            "history_minutes": n - 1,
        }
        cache_id = _artifacts.content_id(spec).removeprefix("sha256:")
        directory = snapshots / cache_id
        directory.mkdir(parents=True)
        frame = bars(n).with_columns(
            pl.Series("close_price", [10.0, 10.0, 11.0, 10.5][:n]),
            pl.lit(12.0).alias("high_price"),
            pl.lit(9.0).alias("low_price"),
        )
        frame.write_parquet(directory / "bars.parquet")
        digest = _artifacts.file_identity(directory / "bars.parquet")[0]
        (directory / "receipt.json").write_bytes(
            _artifacts.canonical_json({"spec": spec, "sha256": digest})
        )
        doc = {
            "schema": f"market-scan-v{i + 1}",
            "model_contract": contract,
            "model_id": _artifacts.content_id(contract).removeprefix("sha256:"),
            "decision_us": spec["decision_us"],
            "rows": [
                {
                    "symbol": "AAAUSDT",
                    "state": "success",
                    "values": {"up": 0.08},
                    "cache_id": cache_id,
                    "source_sha256": digest,
                },
                {"symbol": "币USDT", "state": "fetch_error"},
            ],
        }
        path = tmp_path / f"input-{i}.json"
        path.write_text(json.dumps(doc))
        reports.append(path)
    return reports, {
        "snapshots": snapshots,
        "as_of_us": BASE + 4 * MINUTE_US,
        "upside_output": "up",
        "output": tmp_path / "result",
    }


def run(pair):
    paths, kwargs = pair
    directory = outcomes.reconcile(*paths, **kwargs)
    return directory, json.loads((directory / "report.json").read_text())


def edit(path, change):
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def test_exact_maturity_loss_coverage_and_reuse(pair):
    directory, report = run(pair)
    assert report["summary"]["counts"] == {"observed": 1, "unscored": 1}
    assert report["summary"]["mean_pinball_loss"] == pytest.approx(0.018)
    assert report["summary"]["empirical_quantile_coverage"] == 0
    assert report["rows"][0]["realized"] == pytest.approx(0.1)
    assert report["rows"][0]["path"]["terminal_2"] == pytest.approx(0.05)
    assert run(pair)[0] == directory
    (directory / "report.md").write_text("corrupt")
    with pytest.raises(ValueError, match="publication differs"):
        run(pair)


@pytest.mark.parametrize(
    "case,state",
    [
        ("pending", "pending"),
        ("missing", "missing_source"),
        ("tamper", "invalid_source"),
        ("duplicate", "conflict"),
        ("anchor", "conflict"),
        ("short", "incomplete_horizon"),
        ("after", "source_after_as_of"),
    ],
)
def test_refusal_states(pair, case, state):
    paths, kwargs = pair
    source = json.loads(paths[1].read_text())["rows"][0]
    barpath = kwargs["snapshots"] / source["cache_id"] / "bars.parquet"
    if case == "pending":
        kwargs["as_of_us"] -= 1
    elif case == "missing":
        barpath.unlink()
    elif case == "tamper":
        barpath.write_bytes(barpath.read_bytes() + b"bad")
    elif case == "duplicate":
        edit(
            paths[0], lambda d: d["rows"].append(d["rows"][0] | {"values": {"up": 0.5}})
        )
    elif case == "anchor":
        frame = pl.read_parquet(barpath).with_columns(
            pl.when(pl.col("open_time").dt.epoch("us") == BASE + MINUTE_US)
            .then(10.1)
            .otherwise(pl.col("close_price"))
            .alias("close_price")
        )
        frame.write_parquet(barpath)
        digest = _artifacts.file_identity(barpath)[0]
        edit(barpath.parent / "receipt.json", lambda d: d.update(sha256=digest))
        edit(paths[1], lambda d: d["rows"][0].update(source_sha256=digest))
    elif case == "short":
        paths[1] = paths[0]
    else:
        edit(paths[1], lambda d: d.update(decision_us=d["decision_us"] + MINUTE_US))
    _, report = run(pair)
    assert report["rows"][0]["state"] == state
    assert report["summary"]["mean_pinball_loss"] is None
    assert "realized" not in report["rows"][0]


def test_identical_duplicates_do_not_change_denominator(pair):
    edit(pair[0][0], lambda d: d["rows"].append(d["rows"][0]))
    assert run(pair)[1]["summary"]["members"] == 2


def test_bad_identity_budget_and_cancellation(pair, monkeypatch):
    edit(pair[0][0], lambda d: d.update(model_id="bad"))
    with pytest.raises(ValueError, match="identity"):
        run(pair)
    assert not pair[1]["output"].exists()
    edit(
        pair[0][0],
        lambda d: d.update(
            model_id=_artifacts.content_id(d["model_contract"]).removeprefix("sha256:")
        ),
    )

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(outcomes.targets, "path_targets", cancel)
    with pytest.raises(KeyboardInterrupt):
        run(pair)
    assert not pair[1]["output"].exists()
    monkeypatch.setattr(outcomes.time, "monotonic", iter([0, 901]).__next__)
    with pytest.raises(TimeoutError, match="900"):
        run(pair)
    assert not pair[1]["output"].exists()


def test_report_byte_limit(pair):
    with pair[0][0].open("wb") as stream:
        stream.truncate(64_000_001)
    with pytest.raises(ValueError, match="64 MB"):
        run(pair)
    assert not pair[1]["output"].exists()


def test_invalid_prediction_is_not_missing_return(pair):
    edit(pair[0][0], lambda d: d["rows"][0].update(values=[]))
    assert run(pair)[1]["rows"][0]["state"] == "invalid_prediction"


def test_duplicate_conflict_stays_conflict(pair):
    def duplicates(doc):
        original = doc["rows"][0]
        doc["rows"].extend([original | {"values": {"up": 0.2}}, original])

    edit(pair[0][0], duplicates)
    assert run(pair)[1]["rows"][0]["state"] == "conflict"


def test_total_source_budget_aborts_publication(pair, monkeypatch):
    monkeypatch.setattr(outcomes, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(RuntimeError, match="512 MB"):
        run(pair)
    assert not pair[1]["output"].exists()


@pytest.fixture
def candidate(pair, monkeypatch, tmp_path):
    from crypto_boom import model_runtime

    _, kwargs = pair
    kwargs.update(
        candidate_id="a" * 64,
        candidate_output="up",
        registry=tmp_path / "models",
        trusted=True,
    )
    identity = json.loads(pair[0][0].read_text())["model_contract"] | {
        "decision_step_minutes": 1
    }
    record = {"model_id": "a" * 64, "identity": identity}
    monkeypatch.setattr(
        model_runtime, "load_model", lambda *args, **kwargs: ({}, record, None)
    )

    def predict(models, record, source, *, decision_us):
        # Future label prices include 11, but candidate can only see two past 10s.
        assert source["close_price"].to_list() == [10.0, 10.0]
        assert source["open_time"].dt.epoch("us").max() < decision_us
        return {"up": 0.2}

    monkeypatch.setattr(model_runtime, "predict_bars", predict)
    return pair, record


def test_frozen_candidate_matched_losses_and_origin_only(candidate):
    pair, _ = candidate
    _, report = run(pair)
    assert report["schema"] == "scan-outcomes-v2"
    comparison = report["comparison"]
    assert comparison["matched"] == 1
    assert comparison["incumbent_matched_loss"] == pytest.approx(0.018)
    assert comparison["candidate_matched_loss"] == pytest.approx(0.01)
    assert comparison["loss_delta_candidate_minus_incumbent"] == pytest.approx(-0.008)
    assert comparison["candidate_states"] == {"not_evaluated": 1, "scored": 1}
    assert not pair[1]["registry"].exists()


def test_candidate_refusal_does_not_hide_observed_outcome(candidate, monkeypatch):
    from crypto_boom import model_runtime

    pair, _ = candidate

    def refuse(*args, **kwargs):
        raise ValueError("insufficient history")

    monkeypatch.setattr(model_runtime, "predict_bars", refuse)
    _, report = run(pair)
    assert report["summary"]["observed"] == 1
    assert report["rows"][0]["candidate"]["state"] == "refused"
    assert report["comparison"]["matched"] == 0
    assert report["comparison"]["gain_vs_incumbent"] is None


def test_candidate_target_mismatch_refuses_before_publication(candidate):
    pair, record = candidate
    record["identity"]["outputs"][0]["quantile"] = 0.5
    with pytest.raises(ValueError, match="not comparable"):
        run(pair)
    assert not pair[1]["output"].exists()


def test_candidate_trust_is_explicit(pair, tmp_path):
    pair[1].update(
        candidate_id="a" * 64, candidate_output="up", registry=tmp_path / "models"
    )
    with pytest.raises(ValueError, match="trust"):
        run(pair)
    assert not pair[1]["output"].exists()


def test_candidate_cadence_and_orphan_options(pair, candidate):
    _, record = candidate
    record["identity"]["decision_step_minutes"] = 1440
    with pytest.raises(ValueError, match="cadence"):
        run(pair)
    pair[1]["candidate_id"] = None
    with pytest.raises(ValueError, match="require a candidate ID"):
        run(pair)
    assert not pair[1]["output"].exists()


def test_scan_v3_optional_evidence_does_not_change_outcome_join(pair):
    for path in pair[0]:
        edit(
            path,
            lambda d: d.update(
                schema="market-scan-v3", input_evidence={"state": "unavailable"}
            ),
        )
    assert run(pair)[1]["summary"]["observed"] == 1
