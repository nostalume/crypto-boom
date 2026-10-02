# Crypto Boom

Read-only tooling for studying early, explosive altcoin price moves. The
package includes market-history acquisition and coverage checks, replay and
benchmark utilities, live-readiness qualification, and an **unvalidated**
frozen M1 row scorer. It does not place orders or provide trading advice.

## Install and check

The maintained project now lives at the repository root. The former
`.publish-work` checkout has been promoted here; use this root's `src/`, tests,
configuration and Git branch, not the archived pre-migration source tree.

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev --group research
uv run crypto-boom --help
uv run pytest
```

The `research` dependency group supports the historical research suite; the
forward/path workflows can use the smaller `prediction` extra. The base CLI
does not load the optional fitting stack on import.

## Latest-data prediction (experimental)

### Whole-market report

```sh
uv sync --extra prediction
uv run --extra prediction crypto-boom model list
uv run --extra prediction crypto-boom model activate --id MODEL_ID --trust-model
uv run --extra prediction crypto-boom scan
```

Activate a trusted, locally published model once; scanning then needs **neither a
symbol nor a model path**. No trained weights are shipped. First-time preparation
is documented in the [current model research recipe](docs/hourly-workflow.md).
The scan discovers all observed Binance Spot TRADING USDT members, fetches their
latest completed history at one shared decision boundary, validates data, computes
features and writes Markdown, JSON and CSV under `data/reports/`. Failures and
unattempted members remain in the coverage ledger; partial scans exit with code 2.

See [public contracts and configuration](docs/public-interfaces.md) and
[composable data access / shared data root](docs/data-access.md). The selected
model currently uses hourly context to estimate six-hour maximum upside P90;
that is configuration, **not the public API's name or fixed cadence**. It is not a
buy signal or a 90% win probability. Model-specific training/export/replay lives
under `crypto_boom.research`; the [research paper](docs/path-prediction-study.md)
records measured results and failures. Existing research CLIs below remain
available for reproducibility, not as the recommended market-scanning interface.

`crypto-boom-predict` provides `train`, frozen-model `backtest`, and `latest`.
Latest automatically fetches completed minute bars, computes causal features,
and returns Chinese text or JSON. It predicts **from the current origin**, not
from an already-triggered 10% rise. Default outputs are median/P90 estimates of
maximum future minute-close appreciation over 2/6h; the grid is configurable.
Quantiles are not crossing probabilities, terminal returns or trading signals.

Use `uv sync --extra prediction` for the lean fitting/inference dependency set.
No new trained artifact or raw corpus is shipped. Train your own model or deploy
an explicitly trusted artifact with a matching environment. There is no November
calendar gate. Shared data handling, research fitting and prediction runtime have
separate owners; see the [interface and deployment guide](docs/forward-prediction.md)
for commands, canonical inputs, quality metrics, extension and security contracts.

## Reusable path-quality research (experimental)

`crypto-boom-study` adds archive-observed sample pools, checksum-verified causal
feature caches, separately cached multidimensional path labels, grouped feature
selection against three training-only baselines, and latest-data vector forecasts.
It reuses the acquisition interfaces rather than embedding download scripts in
each experiment. See the [target design and public interfaces](docs/path-research.md).
The first deployed vector predicts upside/downside excursion, terminal return
and signed efficiency; retention and threshold-hit outcomes are research labels,
not yet calibrated live probabilities. No arbitrary combined trading score is used.

## Scope

- `crypto_boom.history` and `crypto_boom.storage` acquire and validate bounded
  Binance Spot archive inputs.
- `crypto_boom.evaluation` replays and benchmarks declared decisions.
- `crypto_boom.qualification` evaluates whether a local live-readiness artifact
  meets its source and publication contract.
- `crypto_boom.research.m1.FrozenM1Scorer.from_package()` scores a supplied
  as-of feature row for the frozen symbol/cell code space. Its output is an
  **unvalidated decision score**, not a calibrated probability, live alert, or
  recommendation.
- `crypto_boom.research.training_data` and `training_run` require separately
  held, source-pinned local study inputs. They refuse missing or altered inputs;
  this repository does not provide those inputs.

Use `uv run crypto-boom --help` for CLI commands and options. Operations that
contact a venue or publish local artifacts are explicit, not import-time effects.

## Public-repository boundary

This code-only repository deliberately contains **no** `data/` receipts or
corpora, `.agents/` plans or research notes, generated artifacts, session dumps,
or local credentials. Some historical parity tests require private/local frozen
inputs and are not included here. The packaged `m1-v1.json` is the scorer's
versioned model resource, not a raw corpus or run receipt; its source hashes are
identifiers only and do not make the underlying data available.

The M1 bundle and its synthetic unit checks do not establish forward validation
or profitable performance. Do not treat research output as investment advice.
