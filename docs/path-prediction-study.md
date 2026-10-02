# From relative-anomaly modes to absolute upside: a retrospective crypto path-prediction study

**Research record: 2026-10-01. Engineering research report, not a peer-reviewed
paper or certification of a trading strategy.**

## Abstract

This project investigates whether price and trading states before a prediction
origin can describe subsequent crypto upside and path quality. Small-sample,
normalized path modes correlated with relative anomalies but were misaligned
with absolute large moves. We introduced separate continuous labels for absolute
upside/downside, terminal return and retention, reused 129 symbols, 985 partitions
and approximately 42.52 million minute records, and compared continuous-history resolutions.

With 24-hour history and a six-hour horizon held fixed, hourly buckets plus
hour-scaled context won validation for gradient-boosted quantile prediction. Test
upside pinball loss fell 6.03% relative to a conditional baseline, and 5.35% on 30
reserved symbols. At a fixed selection budget, absolute-upside-tail precision was
17.19%, versus 15.05% for the conditional baseline. Retention and terminal direction
showed no stable improvement; larger selected upside came with larger downside.
Mainline therefore provides **experimental absolute-upside reports**, not entry
signals, profit probabilities or default automated trading.

## 1. Question and research progression

Original M1 asked whether candidates already up approximately 10% would reach 20%.
The user changed the question to future paths from any valid origin, with 10%/20%
as optional queries rather than mandatory training boundaries. The historical
calendar-based reading restriction was revoked. Causal inputs, temporal splits
and truthful disclosure of previously inspected periods still apply.

1. Build shared acquisition, canonicalization and feature/target caches; separate research fitting from prediction runtime.
2. Compare path quantiles and rolling windows on eight symbols: excursion predictions improve, direction remains weak.
3. Construct eight geometric modes from historical future paths; compare linear and tree classifiers and diagnose target mismatch.
4. Expand to 129 symbols, removing this experiment's fixed minimum-turnover threshold while preserving low activity.
5. Introduce continuous history, absolute-excursion targets and training-only typical cases/hard controls.
6. Test retention and drawdown separately; retain failures instead of repeatedly tuning or changing acceptance criteria.
7. Hold history at 24 hours and compare five-minute/hourly aggregation with corresponding contextual features; select hourly representation.

Implementation, one completed test and demonstrated effectiveness are different
states. Negative findings are retained throughout this report.

## 2. Data and populations

### 2.1 Two distinct experimental pools

The initial eight usable symbols came from a fixed 24-symbol historical-directory
sample out of 728 candidates: 192 symbol-months over September 2025–April 2026,
with 60 available, 131 officially not found and one quarantined for minute-time
granularity. The project did not have only eight symbols in total.

Expansion reused a separate corpus: 129 symbols, 985 partitions, 42,524,250 minute
rows. Audit rejected no partitions and found no internal missing minutes; 20
partitions were not full calendar months and 47 symbol-months were locally absent.
There were 9,905,753 rows with zero turnover and zero trades (23.294%). Completeness
is not predictability or executability. The original corpus used a current trading
list, introducing **survivorship bias**; directory presence is not historical listing eligibility.

Earlier observations do not require survival into later months. Genuine inactivity
is not missingness, prices are not fabricated, and absolute jumps do not trigger
sample deletion. Gaps/bad quality break historical or future windows; incomplete
future windows are not negatives.

### 2.2 Sampling and splits

Hourly evaluation origins yield 705,769 usable feature origins, including 704,995
with complete six-hour futures. Training uses a UTC six-hour grid to reduce
adjacent-label duplication, **not to establish independent event clusters**.

- Training: September–December 2025; original expanded experiment has 43,648 rows and 94 actual training symbols.
- Selection: January–February 2026; 133,414 non-reserved-symbol rows.
- Test: March–April 2026; 186,039 rows across 129 symbols.
- Symbol SHA256 modulo reserves 30 symbols from all fitting. Five additional non-reserved symbols have no training-period samples.
- Labels crossing split boundaries are purged. Test opportunities retain natural frequencies, without future-success selection.

The scale experiment additionally requires full anchored 24-hour history, excluding
89 rows consistently: 43,563 training, 133,413 validation and 186,037 test rows.
Some original logs use “seen” to mean non-reserved, not necessarily present in
training. Expanded diagnostics distinguish actual training symbols, fixed
reserved symbols and symbols lacking training-period history.

## 3. Targets, inputs and evaluation

Let origin close be P₀ and future minute closes over H minutes be Pₜ:

- U=max₀≤t≤H(Pₜ/P₀−1): absolute maximum upside.
- D=−min₀≤t≤H(Pₜ/P₀−1): maximum downside relative to the origin.
- R=P_H/P₀−1: terminal return.
- Retention=max(R,0)/U, defined only for U>0; undefined values are not zero-filled.
- Signed efficiency=R/Σ|Δ(Pₜ/P₀)|; flat paths have value zero.
- Maximum path drawdown=maxₜ[1−Pₜ/maxₛ≤t Pₛ], distinct from D.
- Peak/trough times and fractions of time above/below origin remain separate. Above-origin occupancy is not continuous upward duration.

Relative-anomaly diagnostics use U/(pre-origin minute-return volatility×√H),
leaving zero denominators undefined. This is not yet a formal prediction output
alongside absolute amplitude; timing targets are not trained either.

Inputs use completed history only. Training-period future outcomes may organize
typical examples, but never enter inputs. Control standardization fits only
training data. No fixed 10%/20% training labels are required. Tail evaluation uses
the training U 95th percentile, approximately 5.956% in the expanded experiment:
an evaluation coordinate, not a natural definition of an explosive move.

P90 uses pinball loss; conditional mean retention/drawdown use MSE; modes use
multiclass log-loss and AP. Baselines include constants, symbol priors and past
volatility bins. Expanded/hourly studies primarily compare against training
volatility quartile × trading-activity quartile baselines. Loss reduction is
1−model loss/baseline loss, **not accuracy improvement or profitability**. Paired
seven-day moving blocks with 400 resamples diagnose temporal dependence; they do
not correct for all model selection across the research history.

## 4. Implementations and results

### 4.1 Small-pool path quantiles

Approximately 253,187 origins across eight symbols use 31 causal price, flow,
path and dynamics features. Initial test loss reductions versus training
volatility bins: U P90 7.80%, D P90 12.79%, R P50 0.017%, efficiency P50 about 0.40%.
January–April rolling U gains were 6.90%, 2.27%, 0.99%, 8.49%; D gains were 4.28%,
4.02%, 8.42%, 10.37%. Terminal direction was unstable. ZRO represented about 46.5%
of test samples; rises of at least 20% had only 96 overlapping origins on three
symbol-days. Old forward/path formats and CLIs remain compatible, not retroactively
certified as reliable explosive-move predictors.

### 4.2 Geometric modes and classifiers

On 2,089 training windows spaced more than six hours apart, twelve future
half-hour nodes were normalized by their own maximum amplitude, with relative
amplitude added. RobustScaler+KMeans formed eight exploratory geometries, not
claimed natural market classes. Comparisons used standardized Logistic(C=.1),
31-feature HGB, 43-feature HGB with twelve historical nodes, and inverse-symbol-day
weighted HGB. January calibrated temperature, February selected models, and March–April tested them.

Validation selected 31-feature Logistic with temperature 1.5. Test log-loss was
2.041664, versus constant prior 2.048383 and volatility bins 2.038553: −0.153%
gain against volatility, with block interval approximately [−0.819%, 0.251%].
Strong-mode AP was 0.1423, natural prevalence 0.1042, and volatility-baseline AP
0.1274. Score rank correlation with absolute U was −0.105, versus about +0.169
with relative U, exposing the mismatch between relative anomalies and absolute
large rises. Neither sparse history nodes nor symbol-day weighting won selection;
this does not establish that sequences are useless.

### 4.3 Expanded absolute excursion and continuous history

We compared 31 summaries with 31+432 continuous-history inputs: 72 complete
five-minute buckets across six hours, with six channels. Fixed HGB parameters:
60 iterations, 15 leaves, minimum 50 samples/leaf, learning rate .08, no early
stopping, seed 0. U P90, D P90 and R P50 were fitted separately, six fits total;
validation U loss selected the representation.

| Model | Test U reduction | Test D reduction | Test R reduction |
|---|---:|---:|---:|
| summary31 (validation winner) | 4.385% | 5.704% | 0.001% |
| sequence463 | 4.477% | 6.546% | -0.556% |

Summary31 achieved 3.434% U gain on the 30 reserved symbols; the full-test
seven-day block interval was approximately [3.595%, 5.304%]. Tail AP increased
from the conditional baseline's 0.1183 to 0.1764. Equal-count selection at the
same times raised tail precision from 15.05% to 16.38%, but median terminal
return remained negative. Different populations and sampling policies prevent
attributing old/new result differences solely to expanding the corpus.

### 4.4 Upside quality: retaining failed results

The same 31 features and fixed settings fitted conditional mean retention and
mean maximum path drawdown:

| Target | Validation MSE reduction | Test MSE reduction | Reserved-symbol reduction |
|---|---:|---:|---:|
| Retention (U>0) | -1.30% | -0.56% | -0.85% |
| Maximum path drawdown | -137.53% | -136.97% | -149.08% |

Drawdown test error was about 2.37 times the conditional baseline; deployment was
rejected. Retention reranking reduced downside but also sacrificed positive
terminal amplitude, failing the predeclared joint criteria. More conservative
selection is not better explosive-move discovery. A typical-case library contains
12 training-period retained-rise/spike-and-reversal groups, each with three
nearest hard controls. Some distances are large; this remains exploratory,
not a validated public sampling or weighting strategy.

### 4.5 Time resolution and contextual scale

History (24 hours), horizon (six hours), origins and HGB settings were held fixed:

| Representation | Inputs | Validation U reduction | Test U reduction | Reserved 30 symbols |
|---|---:|---:|---:|---:|
| 288 five-minute buckets×6 + original 31 context | 1759 | 3.71% | 4.03% | 2.68% |
| 24 hourly buckets×6 + original 31 context | 175 | 4.08% | 4.87% | 3.79% |
| **Hourly buckets + hour-scaled context** | **152** | **4.83%** | **6.03%** | **5.35%** |

The six channels are bucket return, high/low range, log1p turnover, log1p trades,
taker-buy share and traded-minute fraction. Eight hourly context features are
1/6/24-hour cumulative returns, 6/24-hour standard deviations of hourly log returns
(ddof=1), log ratio of one-hour turnover to the preceding six-hour mean, six-hour
turnover-weighted buy share and 24-hour observed fraction. Buckets are float32,
context intermediates float64, and final 152 inputs float32; actual recomputation
replaces mere renaming or square-root-of-time conversion.

Hourly context won validation. March/April test U gains were 5.46%/6.47%; loss
improvement over the five-minute representation was about 2.09%, with block
interval [1.51%, 2.47%]. Retention MSE remained 0.81% worse than baseline, so that
head was not deployed. Resolution also changes dimensionality; denoising alone
cannot be credited for the improvement, nor is five-minute behavior proven random.

Post-fit equal-budget diagnostics: hourly tail AP 0.1923 versus summary31 0.1764;
precision 17.19% versus 16.38%. Hourly-selected origins had median upside 2.168%,
downside 2.337% and terminal return −0.523%. **Better absolute-excursion ranking
is not better direction or profitability.**

## 5. Mainline selection and interfaces

The chosen model is the hourly 152-feature HGB's **single U P90 head**, exported
from the validation-winning artifact without refitting and attaching old scores.
Unavailable downside, retention and terminal outputs are null, not zero. Joblib
loads only explicitly trusted local files, with content-hash, sklearn-version and
feature-recipe checks. Hashes establish integrity, not file safety.

The selected model uses UTC hourly training origins. The generic scanner reads
this step from metadata, anchors the whole market at the latest completed hour
at scan start, and fetches the preceding 1,441 minute bars. Hourly cadence is not
a fixed public-interface constraint; replacement models must declare compatible
history, aggregation and decision steps.

The public entry point is `crypto-boom scan`; see [public contracts](public-interfaces.md).
Current-model training, export, publication and single-symbol replay moved to
[research scripts](hourly-workflow.md), replacing public
`crypto-boom-study train-hourly/export-hourly/report`. Existing
acquire/audit/build/screen/rolling workflows remain. Newly trained artifacts are
unevaluated and cannot inherit this report's conclusions; a single training
script is not a replacement for the full scale comparison.

## 6. Threats to validity and unfinished work

1. Current-list selection introduces survivorship bias and misses historical delisted/unavailable assets.
2. Hourly opportunities may miss minute-scale onsets; six-hour training grids do not deduplicate market-event clusters.
3. Quantiles are not success probabilities; regime drift, inactive prices and trading costs remain insufficiently addressed.
4. Repeatedly studied periods are not unread confirmation; symbol reservation does not remove common market shocks.
5. Separate relative-anomaly, peak-time and continuous-persistence outputs are not deployed capabilities.
6. Public typical-case/hard-control selection, event-cluster weights and per-path coverage checks remain unfinished.
7. Evidence does not support retention, terminal direction or trading profitability; endless tuning must not conceal failures.

## 7. Review materials and reproducibility limits

Public implementations include features, feature_batch, sample_pool,
research/path_targets, path_screen, research.hourly and research.hourly_cli;
uv.lock pins dependencies. Existing model formats remain compatible. This report
uses actual project experiments, not external papers as performance evidence;
no literature review was conducted.

The [research evidence index](research-evidence.json) contains machine-readable
summaries and original-report hashes. Originals reside locally under
`data/path-quality-20261001/`, including `expanded-path-v1/`, and are not shipped
with code. The index is not the full raw dataset; external readers without local
files cannot reproduce these results row by row. Key records include source
identities, splits, parameters, failures, predictions, models and hashes.
Research scripts support retraining on legitimately held data, but different data
and populations require separate evaluation.

## 8. Conclusion

The appropriate current deliverable is an experimental absolute-excursion
reporter with explicit boundaries, traceable sources and straightforward operation.
Hourly scale and matching features won the declared comparison, without solving
upside quality. Mainline integration means engineering behavior was checked,
not that trading validity was established. Subsequent research must retain the
unfinished items rather than describe one completed round as achieving the entire objective.
