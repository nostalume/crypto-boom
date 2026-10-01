# Crypto Boom

Read-only tooling for studying early, explosive altcoin price moves. The
package includes market-history acquisition and coverage checks, replay and
benchmark utilities, live-readiness qualification, and an **unvalidated**
frozen M1 row scorer. It does not place orders or provide trading advice.

## Install and check

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

### Recommended research report: hourly upside space

The current research-selected candidate is a 24-hour hourly-context model of
the next six hours' maximum upside P90—not a buy signal or a 90% win probability.
See the [research paper and negative results](docs/path-prediction-study.md) and
the [end-to-end workflow](docs/hourly-workflow.md) for model preparation.

With your trusted exported model in `data/hourly-upside-v1`:

```sh
uv run --extra prediction crypto-boom-study report --model data/hourly-upside-v1 --symbol STXUSDT --trust-model
```

Writes a readable Markdown report, JSON, and source snapshot to a new timestamped
directory under `data/reports/`. Uses the last completed UTC hour and discloses
its age; failed quality heads are not enabled. `--bars <canonical.parquet>` selects
explicit offline replay instead of `--symbol`. Existing predictors below remain
compatible; the hourly model uses its own versioned format.

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
