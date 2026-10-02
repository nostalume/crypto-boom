# Crypto Boom

Research and report on potential large altcoin price moves using causal historical
features and explicitly versioned models. The project supports historical data
acquisition, model research and latest-data market scanning. It does not place orders.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra prediction
uv run --extra prediction crypto-boom --help
```

Copy `crypto-boom.example.toml` to `crypto-boom.toml`. Set `[data].root` to the
shared data directory; relative paths resolve against the configuration file.

## Run predictions

Publish a model using the [research workflow](docs/hourly-workflow.md), then
explicitly activate its ID. Trained prediction bundles are not included; load
only trusted artifacts because joblib can execute code.

```sh
uv run --extra prediction crypto-boom model list --config crypto-boom.toml
uv run --extra prediction crypto-boom model activate --id MODEL_ID --trust-model --config crypto-boom.toml
uv run --extra prediction crypto-boom scan --config crypto-boom.toml
```

Scanning discovers the Binance Spot TRADING USDT universe, fetches completed
history, computes features and predicts with the active model. No symbol list or
model path is required. Binance/OKX spot and perpetual product metadata is reported
separately; OKX does not supply model price inputs, and spot forecasts are not
perpetual-return forecasts.

Progress logs go to **stderr**; structured results go to **stdout**. Reports are
written to `<data.root>/reports/<run-id>/`:

- `report.md`: English summary and ranking preview.
- `predictions.csv` and `report.json`: complete predictions and failure coverage.
- `products.csv`: product coverage and unverified cross-market ticker candidates.

Add `--record-inputs` to save optional inference inputs for later research; see
[the evidence contract](docs/public-interfaces.md#optional-inference-input-evidence).
This does not perform drift detection.

The command prints the report directory. Exit code **0** means complete prediction
coverage, **2** means partial coverage with a report, and **1** means a startup,
universe or publication failure. Check decision time and coverage before use.

## Research and model updates

Training is separate from scanning. Build a historical dataset, fit a candidate,
evaluate on later data, then publish and explicitly activate it. New fits are not
automatically validated or deployed; scanning never retrains the active model.

- [Training, export and publication](docs/hourly-workflow.md)
- [Data access](docs/data-access.md) and [public contracts](docs/public-interfaces.md)
- [Research results and limitations](docs/path-prediction-study.md)

The current research recipe estimates six-hour maximum minute-close upside P90
from hourly context. This is experimental: P90 is not a 90% probability of rising,
expected profit or a trading recommendation.
