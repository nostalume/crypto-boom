# Forward prediction: interfaces, evaluation and deployment

## Boundary contract

| Owner | Interface | Responsibility |
|---|---|---|
| Shared history/storage | `crypto_boom.history`, `storage.source` | Bounded archive acquisition and source-faithful minute parquet |
| Shared bar processing | `bars.admit_bars`, `bars.load_bar_files` | Canonical schema, validation, bounded loading, input hashes |
| Shared current market data | `latest_market.fetch_latest(symbol, minutes=...)` | Closed UTC minute bars and server time, no model or liquidity selection |
| Shared causal features | `features.iter_feature_segments` | Past-only features, separate contiguous quality segments |
| Research model runtime | `research.forward` | `ForecastGrid`, feature eligibility, artifact load/save, `forecast` |
| Research development | `research.forward_training` | `training_rows`, `fit_forward`, `backtest`, shared `evaluate_rows` |
| Research process adapter | `research.predict_cli` | Explicit train/backtest/latest commands, stdout/stderr and text/JSON |

Shared components do not import research modules. Prediction does not import
`forward_training`. The historical `research.acquisition` script remains a
study-specific sampling/orchestration consumer of shared archive interfaces; its
study policy does not belong in generic data acquisition. The older M1 scorer
and its prepared features are not this model's target.

There is no plugin registry, web service or implicit scheduler. To develop a new
model family, reuse shared canonical data/features and provide its own research
runtime and fitting contract. Do not put labels, model thresholds or study dates
in common data processing. To extend this quantile model's output coordinates,
use `ForecastGrid` / `--horizons` / `--quantiles`, then retrain; do not relabel an
old artifact. A changed feature meaning requires a new model schema/artifact.

## Install and commands

Python 3.12; `uv sync --extra prediction` installs this model's requirements,
without TSFEL or interpret-core. The `research` extra/group remains available
for historical research. For installation elsewhere, use `crypto-boom[prediction]`
from the built wheel rather than paths into this checkout.

```sh
uv run crypto-boom-predict train --bars data/SOLUSDT.parquet \
  --start 2025-09-02T00:00:00Z --split 2026-01-01T00:00:00Z \
  --end 2026-03-01T00:00:00Z --horizons 120 360 --quantiles 0.5 0.9 \
  --model data/forward-v1

uv run crypto-boom-predict backtest --bars data/SOLUSDT-later.parquet \
  --start 2026-03-01T00:00:00Z --end 2026-05-01T00:00:00Z \
  --model data/forward-v1 --trust-model

uv run crypto-boom-predict latest --symbol SOLUSDT \
  --model data/forward-v1 --trust-model
uv run crypto-boom-predict latest --symbol SOLUSDT \
  --model data/forward-v1 --trust-model --format json
```

Examples use shell backslash continuation; on PowerShell use a single line or
backticks. `python -m crypto_boom.research.predict_cli` exposes the same commands. Training
and backtest print JSON. Latest defaults to English text. Expected input/I/O
failures print a JSON refusal to stderr and return 2. Use a UTF-8 Windows terminal.

Python consumers can load their own source without any network call:

```python
from pathlib import Path
from crypto_boom.latest_market import fetch_latest
from crypto_boom.research.forward import HISTORY_MINUTES, forecast, load_model

models, metadata = load_model(Path("data/forward-v1"), trusted=True)
bars, server_ms = fetch_latest("SOLUSDT", minutes=HISTORY_MINUTES)
result = forecast(bars, server_ms, models, metadata)
```

Here `trusted=True` has the same security consequence as `--trust-model`.
`forecast` itself has no network, file-writing or fitting effects. Supply the
same canonical bars and an explicit source-clock snapshot for replay/integration.

## Data and target

Input parquet columns: `symbol`, `open_time` (`datetime[us, UTC]`, minute open),
`close_price`, `high_price`, `low_price`, `quote_turnover`,
`taker_buy_quote_turnover`, `trade_count`, `quality_complete` (boolean),
`quality_state` (`valid` for usable rows). `storage.source` outputs contain these
columns. Caller-supplied canonical files must be immutable during a read; hashes
identify the bytes, not source authenticity. Extra columns are not used.

Shared validation sorts but does not deduplicate, forward-fill, invent prices or
repair invalid values. Bad-quality rows and gaps break feature/label continuity.
The model, not shared validation, requires USDT quote turnover >= 1m over the
trailing 24h. Source-specific acquisition validity and model eligibility differ.
The local loader accepts at most 512 files, 2 GB compressed and 10 million rows;
training is in memory, not a streaming service.

An origin is a completed minute close, not a retrospectively selected price
bottom or an already-triggered 10% event. The target for horizon H is
`max(0, max(close[t:t+H+1]) / close[t] - 1)`. It describes maximum **minute-close**
appreciation, not intraminute highs, terminal returns, downside risk or executable
profit. Default coordinates are 120/360 minutes and median/P90. Up to eight
increasing unique integer horizons (1–4320 minutes) and five increasing unique
quantiles in (0,1) are supported. More coordinates incur more independent fits.
Output estimates are made nonnegative and nested across quantile/horizon, and
the evaluator uses exactly that same postprocessing.

Training samples every five minutes using the same 19 causal features as live
scoring. Latest may use any completed minute. Unknown future labels are censored,
never converted into negatives. `--start` is the first origin; preceding bars
provide warm-up. `--end` is the **exclusive observation** cutoff, so any horizon
requiring a later close is censored. Require `start < split < end`; training
origins plus the longest horizon must be strictly before validation starts.
There is no November gate or other reserved calendar interval in this model.

## Reading training quality

The same evaluator serves training-time validation and later frozen backtests.
Backtest refuses origins at/before the model's validation-observation end and
symbols outside the fitted population. It never refits, updates the artifact or
estimates a new baseline from backtest data. Missing training symbols are reported.

- **Pinball loss**: lower is better for the requested quantile. The comparison is
  the constant quantile estimated on training labels only.
- **Relative improvement**: `1 - model_loss / baseline_loss`, not trading return
  or prediction accuracy. Undefined if baseline loss is zero.
- **Observed coverage**: fraction of targets <= forecast; compare with the
  requested quantile, but do not treat a nearby aggregate as proven calibration.
- **By month / by symbol**: whether a pooled gain hides a failing subgroup.
- **Horizon-spaced check**: a fixed UTC subgrid separated by more than the largest
  horizon, reducing within-symbol label overlap. Still not independent trials:
  regimes and symbols remain dependent. No significance claim is produced.
- **Tail support**: 10%/20% are illustrative diagnostic queries, not labels used
  for fitting. Origin counts overlap; distinct symbol-days are not event counts.
- **Censored rows**: feature-eligible origins without every complete horizon.
  Reported candidate rows do not include low-liquidity/warm-up-ineligible minutes;
  do not mistake this for full-market coverage.

This is **prediction backtesting**, not an order/fill/fee/PnL backtest. A positive
baseline comparison does not establish rare explosive-move detection, calibrated
crossing probabilities, robustness to new coins, or profitable execution.

## Artifact and operations

Deploy the installed package plus `model.joblib` and `metadata.json`; no checkout,
private scripts or raw training corpus is required for latest inference. Default
v1 artifacts remain loadable. Exact scikit-learn version, model schema, feature
order, estimator quantiles, temporal metadata and SHA256 must match. Pin the
environment with the lock file; transfer portability does not mean arbitrary
Python/scikit-learn-version compatibility.

**Joblib can execute arbitrary code. Load only artifacts you created or trust.**
Hash verification detects damage, not a malicious author. Artifact admission is
bounded to 100 MB model / 2 MB metadata. Training exclusively creates a new
directory; metadata is the final completion marker. An interrupted directory is
not loadable and is not automatically overwritten. Keep models and corpora under
ignored `data/` or a separately managed location, not in public source commits.

Latest reads one explicitly named ASCII Binance Spot symbol. The shared fetcher
accepts 1–2000 closed minutes; this model requests 1441. At most four requests,
12s timeout each, 2 MB per response, no automatic retry. Gaps, invalid numbers,
missing/unfinished bars, rate/HTTP errors or a minute rollover during acquisition
refuse. Model-level unknown symbols and insufficient liquidity also refuse.
No API key, account operation, order, continuous alert or deployment host is
implied. Source semantics follow the
[Binance Spot REST specification](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data).
