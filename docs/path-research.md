# Sample pools, path quality and batch feature research

Status: experimental interfaces, not a validated trading strategy. The existing
`crypto-boom-predict` remains; `crypto-boom-study` uses a separate model format.
Install `crypto-boom[prediction]`; TSFEL/interpret are not required. There is no
order placement, account connection or implicit network access.

## 1. Sample pools and quality constraints

`sample_pool.acquire_sample_pool` reuses historical archive-directory discovery,
month-availability probes, official archive acquisition and canonical research
partitions. Sampling sorts by `SHA256(seed:symbol)` and fixes original membership;
later missing months or poor performance do not trigger replacement. Directories
include delisted assets, but **directory presence is not reconstructed historical
trading status** and does not eliminate survivorship bias. Historical listing and
delisting evidence is still needed for eligibility.

`selection.json` records directory observation time, seed, exclusions and scope;
`pool.json` records available / not_found / quarantined status per symbol-month.
Unknown network outcomes do not mean absence; unresolved probes prevent a completed
receipt. Quarantine requires explicit months and reasons without modifying sources
or automatically filling gaps. A completed receipt can describe partial coverage
with quality quarantines. Failures preserve verified source files; reruns reuse
identical checksums and partitions. Acquisition is capped at 512 symbol-months
and, by default, 1 GiB compressed downloads per call, not cumulatively across retries.

Feature admission rejects duplicate/misaligned times and invalid prices/volumes.
Missing minutes and bad-quality rows break continuous history, without forward
filling. Each feature cache records missing minutes, zero turnover/trades, large
adjacent price moves, actual coverage and eligible-origin counts. Large moves are
flagged, not removed by outcome. Default origins require at least one million
USDT turnover over the past 1,440 minutes and 1,441 continuous minute bars, sampled
on a UTC five-minute grid. This defines a research population, not executability.

Official source files may be revised. Archive checksums establish byte consistency,
not historical availability at an origin. See the
[Binance public-data documentation](https://github.com/binance/binance-public-data).

## 2. Targets start at the origin, not after a 10% rise

Let C₀ be the last completed minute close, Cₖ the future minute-k close, and
rₖ=Cₖ/C₀−1 for k=0…H. Default horizons are H=120/360 minutes, configurable.
All targets describe **minute-close paths**, not intraminute touches or executable returns.

| Dimension | Definition | Interpretation |
|---|---|---|
| Upside U | max rₖ, including r₀=0 | Maximum close-based appreciation |
| Downside D | −min rₖ | Maximum close-based decline |
| Terminal direction R | r_H | End-of-horizon return, not U |
| Upside retention | max(R,0)/U for U>0 | Fraction of upside retained at expiry; undefined at U=0 |
| Downside retention | max(−R,0)/D for D>0 | Persistence of decline; undefined at D=0 |
| Path efficiency E | R / Σ\|rₖ−rₖ₋₁\| | Direction and tortuosity in −1…1; zero for an unchanged path |
| Occupancy | Fraction of H future closes with rₖ>0 / <0 | Brief pulses versus persistent positions relative to origin |
| Path drawdown | maxₖ(1−Cₖ/maxⱼ≤ₖ Cⱼ) | Peak-to-subsequent-close decline, distinct from origin-relative D |
| Extremum time | First k attaining maximum/minimum rₖ | Origin 0 allowed; ties select first occurrence |
| Amplitude query | First rₖ≥a or rₖ≤−a | Caller-specified a; no default threshold queries or fixed event definition |
| Pre-hit adverse excursion | Largest opposite move before first hit | Smooth arrivals versus arrivals after large adverse moves |

A complete window without a threshold hit has hit=0 and null hit time/pre-hit
adverse excursion. Missing or internally gapped windows have null labels for that
H, not negative outcomes. Undefined retention and incomplete futures are different
conditions; interpret with U/D and complete-window counts rather than zero-filling.
`research.path_targets.path_targets` is independently callable; features never import future labels.

## 3. Compute once, then screen in batches

`feature_batch.build_feature_cache` emits 31 causal features: price 12, flow 7,
path 6, dynamics 6. Identity includes source-byte hashes, code hash, sampling and
liquidity policy. Cache hits verify files and keys without recomputing features.
Separate target caches depend on target specifications and code; changing H or
amplitude queries does not require feature recomputation. Old versions are not automatically deleted.

`research.path_dataset.build_path_dataset` builds batches; `load_path_dataset`
checks cache identities, row counts and one-to-one keys before joining. Consumers
read Parquet or call the interface instead of repeating acquisition/cleaning/features
for every experiment. New v2 manifests store relative references and return
absolute runtime views. Moving the entire reference tree can preserve identity;
copying only a manifest cannot. V1 retains old-path reading; see
[data access](data-access.md) for migration/export constraints.

`research.path_screen.screen_path_features` compares four cumulative groups on
identical rows: price → +flow → +path → +dynamics. This is neither exhaustive
subset search nor causal attribution. Initial responses are U/P90, D/P90, R/P50
and E/P50. Other path-quality dimensions remain research labels, **not all trained
for deployment**. Market cross-sections, order books and funding are not included.

Time is split into train / selection / test. Labels crossing the first two
boundaries are purged by H minutes. Each response selects features on selection,
then evaluates on test without test-driven reselection. Metadata preserves all
candidate attempts. Baselines are training-wide fixed quantiles, per-symbol
training quantiles and training volatility-bin quantiles. Unknown symbols/empty
bins fall back to the global training quantile. Reports include pinball loss,
coverage, reductions against each baseline, per-symbol metrics and metrics on
origins spaced more than H apart within each symbol.

Loss reduction = 1−model loss/baseline loss, not success rate or profit. P90
coverage is the fraction of outcomes at or below the prediction, not a 90% chance
of reaching it. See [scikit-learn's pinball-loss documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#pinball-loss).
Overlapping windows, shared market moves and repeated selection affect effective
evidence. Sparse samples are not necessarily independent. Results establish
neither after-cost returns nor prospective validity; conditional block-interval
limitations appear below.

## 4. CLI and experimental deployment

```sh
crypto-boom-study acquire --start-month 2025-09 --end-month 2026-04 --count 24 --seed forward-path-v1 --output data/pool
crypto-boom-study build --pool data/pool/pool.json --output data/batches --horizons 120 360 --amplitudes 0.05 0.1 0.2
# Replace DATASET.json below with the manifest path printed by build.
crypto-boom-study screen --dataset DATASET.json --train-end 2026-01-01 --selection-end 2026-03-01 --horizon 360 --model data/path-model
crypto-boom-study latest --model data/path-model --symbol BTCUSDT --trust-model
crypto-boom-study latest --model data/path-model --symbol BTCUSDT --trust-model --json
```

Only acquire/latest access the network. Latest reuses `latest_market.fetch_latest`,
automatically computes latest completed-minute features and rejects stale,
incomplete, gapped or low-liquidity inputs. Unseen symbols are marked
`outside_training_symbols`, not represented as validated coverage. Load only
trusted joblib files. Model directories are not overwritten; lack of baseline
advantage does not silently hide a model or switch test periods.

Combined output is a vector: upside + downside + terminal direction + path
efficiency + evidence status. Marginal quantiles are not multiplied into joint
probabilities; upside/downside quantile ratios are not reward/risk ratios.
Arbitrary utility weights do not produce a trading score. A high-quality-rise
probability would require a predefined joint label for appreciation, acceptable
downside, retention and duration, then separate calibration and validation.

## 5. Rolling audit without another model layer

`research.path_screen.rolling_path_audit` reuses the same selection/fitting code
for up to eight UTC-microsecond triples: (training cutoff, selection cutoff/test
start, test cutoff). Training cutoffs must advance and test intervals must not
overlap. Each fold purges H-minute labels crossing its test cutoff. Earlier test
data can become later training history, consistent with rolling evaluation but
not a globally unread blind test.

```sh
crypto-boom-study rolling --dataset DATASET.json --output data/rolling --window 2025-12-01 2026-01-01 2026-02-01 --window 2026-01-01 2026-02-01 2026-03-01
```

Each fold's full report is stored independently. Identity includes dataset, code,
dependency versions, windows and iteration count. Reruns validate cached hashes
without fitting again; later-fold failures preserve completed folds. The audit
does not publish models, select attractive months or modify deployed models.
Report folds separately: averaging percentage reductions is not pooled loss
improvement. Insufficient time/data causes explicit rejection, not shortened windows.

Baseline comparisons use identical valid rows for per-symbol and greater-than-H
spaced origins too. Test diagnostics add paired moving-week intervals relative
to volatility baseline: all symbols from each UTC date enter together, blocks
span seven consecutive calendar days, and 400 fixed-seed resamples yield gain
2.5%/97.5% quantiles. Fewer than 28 observed dates suppress the interval; zero
baseline loss does not produce a fabricated ratio. Within-week dependence is
preserved, but between-week independence is not guaranteed and retraining/selection
uncertainty is excluded. These are **conditional stability diagnostics**, not
significance certification. Intervals crossing zero indicate unstable advantage.

This stage adds no dependency, model family or generic experiment orchestrator.
Historical eligibility evidence, justified coverage expansion and joint-event
calibration remain separate unfinished work.

## 6. Quality first: local audit and no fixed amplitude threshold

```sh
crypto-boom-study audit --corpus data/corpus --start-month 2025-09 --end-month 2026-04 --output data/corpus-quality.json
# Continuous path labels are the default; add --amplitudes only for threshold queries.
crypto-boom-study build --pool data/pool/pool.json --output data/batches --horizons 120 360
```

Public `sample_pool.audit_local_corpus` reuses strict partition loading and
canonical minute validation, bounded to 2,048 partitions, 16 GiB Parquet and
15 minutes. It runs serially without acquisition, fitting or deletion, reporting
validation failures, internal gaps, month-edge coverage, quality flags and zero
turnover/trade counts. CLI refuses existing reports; resource-limit errors do
not publish a complete report.

Directory candidates, sampled symbols, downloaded local partitions and eligible
training origins are different counts. Sampling 24 archive symbols is not a
quality screen of the entire corpus. Local absence is not official archive
absence. Month-edge gaps alone establish neither faults nor listing/delisting
dates. Do not retain only symbols present in every future month.

Quality includes provenance/checksums, temporal continuity, origin-time price
observability and target applicability. Zero trading can be legitimate without
fresh price discovery every minute. Preserve observations and availability
status, and never remove them because later appreciation was poor. Historical
eligibility requires separate evidence.

`PathTargetSpec().amplitudes == ()`, also the CLI default. Continuous U/D/R,
retention and efficiency definitions are unchanged; explicit amplitude queries
remain compatible. Existing caches/models are not rewritten. H is a forecast
horizon and P50/P90 are distribution-query positions, not appreciation-event
thresholds; both must still be explicit. Existing models' one-million-USDT past
24-hour turnover requirement is a **research population/liquidity policy**, not
data truth or a definition of quality. Old population constraints were not silently changed.

Continuous targets have limits: U/D mainly measure excursion, not direction;
retention is unstable near U=0 and needs its denominator; minute closes omit
intraminute touches; absolute amplitudes are not automatically comparable across
symbols. Future work may test pre-origin-volatility normalization, never future
volatility, without untested replacement of targets or weighted trading-score claims.

## 7. Data selection v1: standards, opportunities and explosive paths

**High quality means reliable observations and sufficient origin information,
not favorable future returns.** Cleaning cannot remove all market noise. Poor
measurement, rare-event dilution and insufficient model information are distinct
problems, not all solved by higher liquidity thresholds.

### Fixed minimum standards

1. Explicit provenance/checksums, unique minute keys, consistent units and valid price/trade relationships; reject invalid sources.
2. Only completed minutes through the origin; gaps/bad quality break history without fabricated prices.
3. Current features require 1,441 continuous valid minutes; insufficient history means not ready, not a negative outcome.
4. Explicit origin-price evidence. The conservative v1 observed flag requires both positive trades and turnover in the origin minute; also report age since the last traded minute and traded fraction over 60 minutes. This technical minimum does not establish depth, executability or absence of manipulation.
5. Incomplete/gapped/suspect futures are not no-explosion labels. Label availability and origin admission remain separate. Future missingness may be nonrandom; report its rate rather than deleting hard cases to improve scores.
6. Listing identity and population attributes such as stablecoin/leveraged-token status need provenance, not outcome-based inference. Unverified historical eligibility stays unknown; file checks do not certify it.

`feature_batch.origin_observability(source)` returns a complete per-origin ledger:
valid, price_observed, contiguous_valid_minutes, minutes_since_traded_bar,
observed_fraction_60, feature_history_ready, origin_admissible, selection_reason.
Calculations are past-only, contain no future labels and do not silently remove
inadmissible rows. Age is measured in completed bars, not exact trade timestamps.
Join to cached features/labels on shared keys. These flags are not silently applied
to old models and do not establish a final training population.

### Required controls and coverage

The opportunity pool includes explosive rises, false starts/spike reversals,
declines and ordinary movement. Do not collect only successes or permanently
exclude quiet assets. Conservative admission may miss quiet-to-active transitions;
evaluate events among rejected origins, overall opportunity coverage and admitted
performance together. Refusing to predict is not a correct prediction, and an
attractive admitted-only metric is insufficient.

Future-label-based training sampling does not authorize future-based deployment
admission. Evaluation preserves natural test prevalence. Event weighting and
probability calibration are not implemented by this interface.

### The target remains a path, not simply up or down

Continuous U/D/R, retention, efficiency and arrival behavior describe different
path properties without fixed appreciation thresholds. U P90 is not explosion
probability; reduced excursion loss alone is not explosive-path discovery.
Joint-target predictive value requires independent validation.

Acceptance should include provenance/temporal leakage checks, traceable excluded
populations, event-cluster rather than minute-row support, cross-time/symbol
stability, probability reliability at natural frequencies and overall coverage.
No universal quality score or fixed 10%/20% definition of every explosive move is needed.

## 8. Public continuous-history interface

`features.past_sequence(source, origins, history_minutes=360, step_minutes=5)`
accepts canonical single-symbol minute data and `symbol, decision_us` origin keys.
It returns keys in the same order and six `Array(Float32, 72)` columns:
`bucket_return`, `bucket_range`, `log_turnover`, `log_trades`, `buy_share`,
`observed_fraction`. Arrays run oldest to newest; the final bucket ends at the
origin's completed minute without future data. Defaults cover complete five-minute
buckets rather than sparse price samples.

Return compares each bucket's final close with the previous bucket's final close;
range is bucket high/low minus one. Turnover and trade counts use log1p of bucket
sums; taker-buy share is turnover-weighted. Zero-turnover buckets use neutral 0.5
share while turnover and observed-minute fraction distinguish inactivity. Absolute
price amplitude is not normalized by the window's own maximum move.

```python
from crypto_boom.features import past_sequence

# source: loaded canonical minute DataFrame for one symbol.
# origins: string symbol and Int64 decision_us (completed-minute UTC microseconds).
sequence = past_sequence(source, origins)
latest_sequence = past_sequence(source, origins.tail(1))
```

The interface requires continuous valid history plus one anchor close. Missing,
bad-quality, duplicate or misaligned origins are rejected without fabricated
history or low-volume sample deletion. Each call permits at most two million
source minutes and ten million output values; larger callers must batch.
Parameters must be positive integers with exact divisibility and history at most
1,440 minutes. Output preserves origin order and joins independently cached
features/targets by key.

This shared training/inference transform does not fetch, fit or publish models.
Existing CLIs, model formats and default feature sets remain unchanged; private
sequence models cannot be passed directly to old prediction CLIs. Expansion can
explicitly use `build_feature_cache(..., step_minutes=60, minimum_turnover=0)` to
retain low-activity origins. This chooses an hourly grid, does not establish
minute-level event coverage and does not change old cache/model population policies.

## Recorded-input reference diagnostics

This research workflow uses the optional input evidence emitted by public
`crypto-boom scan --record-inputs`. It does not fetch data, load models, select
training cases, set alarms or change activation. Reference/cohort choices remain
research responsibilities rather than public scan options.

```sh
uv run --extra prediction python -m crypto_boom.research.input_drift --config crypto-boom.toml reference --report REFERENCE_REPORT/report.json --kind observation
uv run --extra prediction python -m crypto_boom.research.input_drift --config crypto-boom.toml compare --reference REFERENCE_DIRECTORY --current LATER_REPORT/report.json
```

Each command prints its directory. Defaults are `<data.root>/runs/input-references`
and `<data.root>/runs/input-diagnostics`; each subcommand accepts `--output`.
References contain a compact admitted scan manifest, an exact input-sidecar copy
and a hash-bound reference manifest. Diagnostics contain English `report.md` and
full `report.json`. Repeating identical publication verifies bytes; conflicts are
refused. Relocate whole reference directories without changing their content-ID
name. No original report or model is overwritten.

Choose `--kind replay` for deliberately reconstructed historical runs; `observation`
is an explicit research declaration, not proof of historical knowability. Neither
kind is a training reference. The first version supports **one snapshot per
reference**, not a multi-period representative population. Do not select a reference
after seeing a desirable comparison and then call the result independent validation.

Input verification binds sidecar hash/size, row/source/time keys, finite values,
unique symbols, exact dtype/width and capture coverage to the report. Missing
original evidence is refused, never reconstructed silently. Partial capture retains
its full universe and prediction-state counts; observed vectors alone determine
input-distribution statistics. Current origin must be strictly later. Model,
recipe, feature/capture code identities, dtype and width must match; a version
change requires a new declared comparison design, not a market-drift claim.

The report separates:

- Universe members added/removed between these snapshots (not proof of listings/delistings).
- Prediction coverage and input-capture coverage.
- Full captured-population feature distributions.
- The same comparison restricted to common captured symbols.

Per-feature descriptive statistics are mean, median, reference standard deviation,
standardized mean change, maximum empirical CDF distance, and the current fraction
outside the reference's observed range. Constant-reference columns have no
standardized score. No common symbols means matched statistics are unavailable
(an empty list), not zero distance. Feature indices identify the pinned vector
order; they are not causal explanations or independent tests. There are no
p-values, universal alarm thresholds, learned preprocessing, profitability claims
or automatic retraining decisions. Realized prediction outcomes remain a separate
[research operation](hourly-workflow.md#reconcile-saved-predictions-with-later-outcomes).

Bounds per snapshot: report <=64 MB / 4,000 raw rows / 2,000 unique members;
sidecar <=20 MB encoded / 32 MB declared decoded / 16 MB numeric payload;
1–16,384 features. These are input/work limits, not a total process-memory guarantee.
Unknown, malformed or oversized inputs abort without a successful partial result.
All selection is explicit; there is no unbounded directory search or retention job.

## Explicit temporal input and outcome study

```sh
uv run --extra prediction python -m crypto_boom.research.window_diagnostics --config crypto-boom.toml --manifest selection.json
```

The command prints a content-addressed directory under
`<data.root>/runs/temporal-diagnostics` (override with `--output`). It contains
English `report.md`, detailed `report.json`, and the original `selection.json`.
This is a research command, not a new public scan mode or monitoring scheduler.

Example selection manifest (timestamps are UTC microseconds; windows are
half-open and aligned to `step_minutes`):

```json
{
  "schema": "temporal-input-study-v1",
  "kind": "observation",
  "step_minutes": 60,
  "as_of_us": 1790906400000000,
  "reference_window": {
    "start_us": 1790863200000000,
    "end_us": 1790870400000000
  },
  "observation_window": {
    "start_us": 1790906400000000,
    "end_us": 1790910000000000
  },
  "entries": [
    {
      "decision_us": 1790866800000000,
      "report": "reports/earlier/report.json",
      "outcome": "outcomes/CONTENT_ID/report.json"
    },
    {
      "decision_us": 1790906400000000,
      "report": "reports/later/report.json"
    }
  ]
}
```

Choose `replay` instead of `observation` for reconstructed historical evidence.
Paths resolve relative to the supplied manifest; no directory discovery occurs.
The copied selection preserves the original bytes, not rebased locators: rerun
with the original manifest, or explicitly rebind paths in a new manifest. Source
hashes in the report identify the evidence actually consumed.

Every expected grid slot appears, including unlisted or missing reports. Missing
sidecars and unavailable capture remain visible; corrupted evidence aborts.
Equivalent repeated origins count once; conflicting evidence at the same origin
is refused rather than resolved by latest-run selection. Models must match
throughout; available vectors must also match recipe, capture/feature code,
dtype and width. Selection should be fixed before inspecting outcomes.

Each observation is compared separately with each available reference origin.
Per-feature CDF distances are then averaged with **equal reference-origin
weight**, not by pooling symbol rows. Full captured populations and common-symbol
subsets remain separate, with pair-specific support and membership changes.
No common symbols means unavailable matched statistics, not zero distance.

An optional outcome must be a content-addressed `scan-outcomes-v1/v2` artifact
from the existing reconciliation command. It must bind the exact prediction
report hash, model, origin and declared target. Its as-of cannot exceed the study
as-of. Observed rows must be mature and cover a unique, complete symbol ledger
including unavailable outcomes; losses and coverage are recomputed and checked.
Period mean losses weight available origins equally. Missing periods or labels
are never zero; changing populations can affect the results. These diagnostics
do not establish a causal relation between feature drift and prediction errors.

Limits: 8 reference slots, 32 observation slots, 64 manifest entries; step 1–1,440
minutes; 64 MB retained reference arrays, 512 MB charged source-file bytes,
200,000 reference/observation/feature comparisons, a checked 900-second deadline,
and 32 MB JSON output. Existing per-snapshot limits still apply. These are bounded
research workloads, not a hard total-memory or preemptive execution guarantee.
There are no p-values, automatic alarms, downloads, training, activation, or
claims that replay observations constitute independent multi-period validation.
