# Current model research recipe (not the public scan interface)

The public operation is [crypto-boom scan](public-interfaces.md). This page
records the selected `hour_scaled_context_up_360` research model: 24 hours of
continuous history, 60-minute aggregation, 152 features, and P90 of maximum
minute-close upside over the next six hours. Comparisons, failures and limitations
are in the [research report](path-prediction-study.md). Scale and target belong to
model metadata, not the framework name or every future model's configuration.

## Export an evaluated model once, without retraining

These paths are researcher-owned examples, not product defaults:

```sh
uv sync --extra prediction
uv run --extra prediction python -m crypto_boom.research.hourly_cli export-hourly --study PATH_TO_SCALE_STUDY --model data/research-export --trust-model
uv run --extra prediction python -m crypto_boom.research.hourly_cli publish --model data/research-export --trust-model --activate
uv run --extra prediction crypto-boom scan
```

Existing research exports can be published without repeating export. Deployment
loads only activated models from `models/`, independently of research paths.
Publishing defaults to `models` under configured `data.root`; `--registry` is an
explicit research override. Publishing returns a content ID. Omit `--activate`
to select it later with `model activate --id ID --trust-model`.
Models and research data are not distributed through Git. Joblib can execute
code: use only artifacts whose source you trust.

## Training and research replay

```sh
uv run --extra prediction crypto-boom-study build --pool data/pool/pool.json --output data/dataset --step-minutes 60 --minimum-turnover 0 --horizons 360
uv run --extra prediction python -m crypto_boom.research.hourly_cli train-hourly --dataset data/dataset/DATASET_ID.json --train-end 2026-01-01 --model data/new-fit
uv run --extra prediction python -m crypto_boom.research.hourly_cli report --model data/research-export --bars PATH_TO_CANONICAL_BARS --output data/replay --trust-model
```

Replace `DATASET_ID` with the manifest emitted by build. Training uses the fixed
current recipe and a six-hour training grid, purges labels crossing the training
cutoff, retains zero-activity states, and rejects insufficient history. It accepts
100–200,000 candidate origins. Only upside P90 is fitted; new models are marked
`new_fit_not_evaluated`. Training does not automatically reserve the paper's 30
symbols or inherit its scores. Researchers own independent training/test
populations, temporal splits and evaluation. The research script's `--symbol`
online report remains diagnostic, not the recommended deployment operation.

Old `crypto-boom-study train-hourly/export-hourly/report` commands were removed
in favor of the research module above. Acquire/audit/build/screen/rolling remain
research tools. Reusable data and features are shared; each candidate model does
not get another public subcommand. Public scanning needs neither a research
directory nor an experiment-specific path.

Separate relative-anomaly outputs, timing/persistence targets, hard controls,
event weights and survivorship bias remain unresolved research work. This
interface refactor does not change results or establish profitability.

## Update and rollback boundaries

`train-hourly` accepts `--train-end`, not a training-start flag. A rolling training
window needs an explicitly bounded input dataset. The implementation is
`research.hourly.fit_hourly_dataset`; scanning never calls it. Exporting a selected
study model does not refit it.

There is no complete frozen-hourly-candidate evaluation or drift-monitoring
pipeline. `crypto-boom-study screen/rolling` evaluates the summary-feature path
workflow, not this 152-feature candidate. The hourly `report` command is a
single-origin replay, not a batch backtest. Progress logging is not drift detection.

Evaluate a new candidate separately before publication. Publishing and activation
check technical compatibility, not predictive quality; neither evaluates nor
certifies the model. Publish without `--activate`, review the evidence, and then
activate explicitly. Retain the old model ID: rollback uses the same public
`model activate --id OLD_ID --trust-model` command. Do not overwrite old models
or their reports. No automatic retraining, promotion or scheduling is implemented.

## Reconcile saved predictions with later outcomes

This offline research command uses a declared pair of existing scan reports and
verified snapshots under the configured data root. Without candidate options it
loads no model. It never fetches prices, fits a model or changes activation:

```sh
uv run --extra prediction python -m crypto_boom.research.outcomes --predictions EARLIER_REPORT/report.json --outcomes LATER_REPORT/report.json --upside-output upside_p90 --as-of 2026-10-02T02:00:00+00:00 --config crypto-boom.toml
```

Replace report paths and the timezone-aware as-of time. `--upside-output` explicitly
declares an output as maximum future minute-close rise including the origin;
it does not infer target meaning from a label. Only return-fraction quantiles with
1–1,440-minute horizons are supported. The later report supplies price evidence;
its model need not match the earlier model. There is no implicit snapshot search,
partial-window merge or alternate-source fallback. The as-of bound controls event
maturity and source decision time, not proof of historical information availability.

Output is `<data.root>/runs/outcomes/<content-id>/report.json` and `report.md`;
`--output DIRECTORY` overrides the research result root. The command prints the
result directory; English stage logs go to stderr. JSON includes every original
member, source identities, realized path measures, loss and reason states. Identical
duplicate members count once; conflicting evidence is excluded explicitly.

Missing, invalid, incomplete and not-yet-mature outcomes are not zero returns.
Pinball loss and empirical quantile coverage use observed outcomes only; inspect
the full state counts before comparing metrics. One report represents one market
origin, not hundreds of independent temporal trials. No conditional-baseline gain,
feature drift, trading profitability or automatic promotion is established.
The basic command reconciles saved predictions. Optional frozen-candidate
comparison is described below; neither mode is a multi-period backtest.

Resource limits: two reports, each at most 64 MB / 4,000 rows / 2,000 unique members;
snapshots at most 2 MB compressed / 8 MB declared uncompressed, with exact expected
row counts; 512 MB total snapshot reads and a 900-second between-row deadline.
In-flight local IO can exceed that deadline. Global schema/budget failures abort
publication rather than truncate coverage. Report v1/v2/v3 prediction fields are supported; optional input evidence is not
required for realized-outcome reconciliation. Equal
results reuse an existing byte-verified publication; mismatches are refused.
Hashes provide integrity, not source authenticity. No retention/deletion runs.

### Compare one frozen registered candidate

Append these options to the same command:

```sh
--candidate-id CANDIDATE_ID --candidate-output upside_p90 --trust-model
```

The candidate is loaded from `<data.root>/models` with the existing public
`model_runtime.load_model` contract. Trust is explicit because joblib may execute
code. The research script reuses `predict_bars` on each verified original snapshot;
it does not implement another feature recipe, train a model or activate the ID.

Both explicitly declared upside outputs must have equal horizon, quantile, unit
and statistic; the original decision must satisfy candidate cadence. Input recipes
may differ, but the saved history must support the candidate or its row is refused.
Target declarations are research-owned: matching metadata does not prove identical
training labels or an independent training/test split. Verify candidate provenance
and training cutoff separately before interpreting results.

The comparison evaluates only the original report's complete observed-outcome
subset, not candidate opportunity coverage over the entire market. Every original
member remains in the ledger. Candidate refusals do not erase observed incumbent
outcomes. `comparison` reports both losses on the identical matched subset,
candidate quantile coverage, loss delta (candidate minus incumbent) and relative
gain when incumbent loss is nonzero. No matched rows means unavailable metrics,
not zero loss. Do not compare matched candidate loss to unmatched incumbent loss.

Optional comparison writes `scan-outcomes-v2`, including the candidate identity,
provenance, runtime/feature hashes and per-row candidate states. Basic reconciliation
continues to write v1. Existing publications are never overwritten. Only verified
past snapshots enter inference; future prices are label evidence. Replaying the
incumbent as its own candidate is a parity check, not a new performance result.
A single-origin batch cannot establish stable superiority or justify promotion.
