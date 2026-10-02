"""Temporal selection, equal-origin weighting and outcome binding."""

import json

import pytest

from crypto_boom import _artifacts
from crypto_boom.research import window_diagnostics as temporal
from test_input_drift import edit, snapshot

M = 60_000_000


def selection(root, entries):
    path = root / "selection.json"
    path.write_text(
        json.dumps(
            {
                "schema": "temporal-input-study-v1",
                "kind": "replay",
                "step_minutes": 1,
                "as_of_us": 5 * M,
                "reference_window": {"start_us": M, "end_us": 3 * M},
                "observation_window": {"start_us": 3 * M, "end_us": 6 * M},
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def entry(path, t, outcome=None):
    value = {"decision_us": t * M, "report": f"{path.parent.name}/report.json"}
    if outcome:
        value["outcome"] = f"{outcome.parent.name}/report.json"
    return value


def run(root, manifest):
    directory = temporal.study(manifest, output=root / "out")
    return json.loads((directory / "report.json").read_bytes())


def test_time_weighting_dedup_and_missing_slots(tmp_path):
    a = snapshot(tmp_path, "a", M, ["AAAUSDT"], [[0]])
    b = snapshot(tmp_path, "b", 2 * M, ["AAAUSDT", "BBBUSDT", "CCCUSDT"], [[10]] * 3)
    c = snapshot(tmp_path, "c", 3 * M, ["AAAUSDT"], [[0]])
    manifest = selection(tmp_path, [entry(a, 1), entry(a, 1), entry(b, 2), entry(c, 3)])
    result = run(tmp_path, manifest)
    assert result["duplicate_entries_collapsed"] == 1
    assert result["timeline"][2]["natural"][0]["mean_cdf_distance"] == 0.5
    assert result["timeline"][2]["matched"][0]["reference_times"] == 2
    assert result["summary"]["observation"]["input_states"] == {
        "available": 1,
        "missing_report": 2,
    }
    assert result["summary"]["observation"]["equal_origin_mean_loss"] is None
    assert run(tmp_path, manifest) == result


@pytest.mark.parametrize(
    "case", ["conflict", "time", "contract", "bytes", "work", "overlap"]
)
def test_invalid_studies_refused(tmp_path, monkeypatch, case):
    a = snapshot(tmp_path, "a", M, ["AAAUSDT"], [[0]])
    b = snapshot(tmp_path, "b", 3 * M, ["AAAUSDT"], [[1]])
    manifest = selection(tmp_path, [entry(a, 1), entry(b, 3)])
    if case == "conflict":
        c = snapshot(tmp_path, "c", M, ["AAAUSDT"], [[2]])
        edit(manifest, lambda d: d["entries"].append(entry(c, 1)))
    elif case == "time":
        edit(manifest, lambda d: d["entries"][1].update(decision_us=4 * M))
    elif case == "contract":
        edit(
            b,
            lambda d: d["input_evidence"].update(
                capture_code_sha256="sha256:" + "e" * 64
            ),
        )
    elif case == "bytes":
        monkeypatch.setattr(temporal, "MAX_BYTES", 1)
    elif case == "work":
        monkeypatch.setattr(temporal, "MAX_COMPARISONS", 0)
    else:
        edit(manifest, lambda d: d["observation_window"].update(start_us=2 * M))
    with pytest.raises(ValueError):
        run(tmp_path, manifest)
    assert not (tmp_path / "out").exists()


def test_missing_inputs_and_unmatched_population(tmp_path):
    a = snapshot(tmp_path, "a", M, ["AAAUSDT"], [[0]])
    b = snapshot(tmp_path, "b", 2 * M, ["AAAUSDT"], [[0]])
    c = snapshot(tmp_path, "c", 3 * M, ["BBBUSDT"], [[1]])
    (b.parent / "inputs.parquet").unlink()
    result = run(tmp_path, selection(tmp_path, [entry(a, 1), entry(b, 2), entry(c, 3)]))
    assert result["timeline"][1]["state"] == "missing_inputs"
    assert result["timeline"][2]["matched"] == []
    assert result["timeline"][2]["natural"][0]["reference_times"] == 1


def outcome(root, prediction, change=None):
    def model(doc):
        doc["model_contract"]["outputs"] = [
            {
                "name": "up",
                "statistic": "quantile",
                "unit": "return_fraction",
                "quantile": 0.9,
                "horizon_minutes": 1,
            }
        ]
        doc["model_id"] = _artifacts.content_id(doc["model_contract"]).removeprefix(
            "sha256:"
        )
        doc["input_evidence"]["model_id"] = doc["model_id"]
        doc["rows"][0]["values"] = {"up": 0.1}

    edit(prediction, model)
    doc = json.loads(prediction.read_bytes())
    value = {
        "schema": "scan-outcomes-v1",
        "model_id": doc["model_id"],
        "prediction_report_sha256": _artifacts.file_identity(prediction)[0],
        "decision_us": M,
        "as_of_us": 2 * M,
        "target": {
            "declaration": "maximum_minute_close_rise_including_origin",
            "output": "up",
            "horizon_minutes": 1,
            "quantile": 0.9,
        },
        "rows": [
            {
                "symbol": "AAAUSDT",
                "state": "observed",
                "prediction": 0.1,
                "realized": 0.2,
                "pinball_loss": 0.09,
                "below_quantile": False,
            }
        ],
    }
    if change:
        change(value)
    directory = root / _artifacts.content_id(value).removeprefix("sha256:")
    directory.mkdir()
    path = directory / "report.json"
    path.write_bytes(_artifacts.canonical_json(value))
    return path


def test_realized_loss_and_absent_periods(tmp_path):
    a = snapshot(tmp_path, "a", M, ["AAAUSDT"], [[0]])
    o = outcome(tmp_path, a)
    result = run(tmp_path, selection(tmp_path, [entry(a, 1, o)]))
    assert result["summary"]["reference"]["equal_origin_mean_loss"] == pytest.approx(
        0.09
    )
    assert result["summary"]["reference"]["origins_with_realized_loss"] == 1
    assert result["timeline"][0]["outcome"]["quantile_coverage"] == 0
    assert result["timeline"][2]["comparison_state"] == "no_current_inputs"


@pytest.mark.parametrize("case", ["hash", "future", "immature", "loss", "population"])
def test_invalid_outcomes_rejected(tmp_path, case):
    a = snapshot(tmp_path, "a", M, ["AAAUSDT"], [[0]])

    def change(d):
        if case == "hash":
            d["prediction_report_sha256"] = "sha256:" + "f" * 64
        elif case == "future":
            d["as_of_us"] = 6 * M
        elif case == "immature":
            d["as_of_us"] = M
        elif case == "loss":
            d["rows"][0]["pinball_loss"] = 0
        else:
            d["rows"].append(d["rows"][0])

    o = outcome(tmp_path, a, change)
    with pytest.raises(ValueError):
        run(tmp_path, selection(tmp_path, [entry(a, 1, o)]))
