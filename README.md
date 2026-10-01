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

The `research` dependency group is needed for fitting and feature experiments;
the base package and CLI do not load those dependencies on import.

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
