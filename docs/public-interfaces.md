# Public interfaces and whole-market scan contract

## One main workflow

`crypto-boom scan` → load active model → obtain venue universe and clock →
fetch market-wide inputs → validate canonical data → compute causal features →
predict numerical outputs → publish a complete coverage ledger and report.

Install the `prediction` extra, then publish and activate a trusted model (see the
[current research recipe](hourly-workflow.md)). Subsequent runs need only
`uv run --extra prediction crypto-boom scan`: no symbol, hourly designation or model path.

The default prediction universe is **Binance Spot USDT-quoted spot members with
TRADING status at the snapshot**, not the global market. It is not restricted to
training symbols. Unsupported metadata, insufficient history, gaps and request
failures remain in the ledger without scores; they are not false negatives.
Genuine no-trade periods are not automatically bad data; missing minutes must not be fabricated.

## Boundaries and reusable capabilities

| Owner | Interface | Contract |
|---|---|---|
| Historical acquisition | `history`, `storage`; existing archive CLI family | Declared history download and validation, without model logic |
| Latest acquisition | `market_data.SpotSnapshotClient.universe/bars` | Universe snapshot, shared time, request budgets, exact-window caching |
| Source decoding | `binance_source.decode_minute_page` | Exact completed Binance REST minute pages; reject missing, duplicate or unfinished bars |
| Cleaning/admission | `bars.admit_bars` | Source-neutral minute table, sorting, price/trade/time validation; reject rather than guess repairs |
| Shared features | `features.past_sequence/SequenceRecipe/sequence_matrix` | Past-only inputs; recipe-owned history, aggregation and windows; no model-specific name |
| Model runtime | `model_runtime.publish_model/activate_model/load_active_model/predict_bars` | Content addressing, explicit trust, compatibility checks, bound recipe and output semantics |
| Scan/report | `config.ProjectSettings`, `market_scan.scan_market` | All members, common origin, success/failure ledger, atomic publication |
| Research | `research/*` and retained experimental CLIs | Fitting, comparison, selection, export and model-specific replay; not the public deployment contract |

This is not an arbitrary model-plugin system. `numeric-sequence-v1` supports 1–8
numerical outputs and bounded historical sequence recipes (at most 1,440 minutes).
Estimators must match the feature count and predict protocol. Different input
forms require a new explicit protocol and validation, not just replacement
weights. The public runtime does not import research code.

## Configuration and model selection

See [composable data access](data-access.md). Copy `crypto-boom.example.toml` to
`crypto-boom.toml`; `[data].root` selects the single data root. Relative paths are
resolved against that file. Scan settings remain `[scan].workers` and
`timeout_seconds`. Commands accept `--config FILE`; otherwise configuration is
discovered upward. Without a config, `data/` is used only at an identified project
root, not independently in each subdirectory. Old `[scan].data_dir` was removed;
use `[data].root` and one project configuration.

`data.root/models/<content-ID>/` stores immutable manifests and weights;
`active.json` records the local selection:

```sh
uv run --extra prediction crypto-boom model list
uv run --extra prediction crypto-boom model activate --id MODEL_ID --trust-model
uv run --extra prediction crypto-boom scan
```

The model ID identifies configuration and weights, not a research path.
Activation checks trust acknowledgment, hashes and environment; scanning checks
them again. Joblib can execute code. Hashes establish integrity, not source safety:
never activate untrusted models. Weights are not shipped in Git. On a new machine,
transfer a trusted registered model and explicitly reactivate it; no scanner
source change is needed. The [research report](path-prediction-study.md) documents
the current selection and performance rather than hard-coding them into public calls.

## Latest data, caching and budgets

The scanner obtains venue time and selects the latest completed boundary according
to the model's decision step. Every symbol uses that boundary; later minutes are
not mixed in during scanning. Reports give the origin, estimated age and whether
a newer origin may exist by completion. Latest means the shared snapshot at scan
start, not real time at completion. Cache reuse requires identical symbol,
origin, history length and valid hashes; old-origin caches cannot stand in for new data.

Limits: 2,000 prediction members (reject rather than truncate), four workers,
four requests/second, 5,000 market-data requests and a 900-second budget. Up to
three additional product-metadata requests each have a ten-second limit and share
the overall deadline. Market requests have a twelve-second timeout; in-flight
requests can return after the budget. The client uses half the venue's per-minute
weight budget. HTTP 418/429 stops further fetching without retries on other
domains. See the [Binance REST documentation](https://developers.binance.com/en/docs/products/spot/rest-api).
Other processes on the same IP consume capacity too; avoiding rate limits is not
guaranteed. Run only one scanner per data directory. Snapshots can keep growing:
audit data is not automatically deleted; operators own retention and cleanup.

## Reports and exit codes

The public CLI configures standard Python logging at INFO, writing readable English
progress to stderr without altering JSON stdout or report schemas. It respects
logging handlers already installed by an embedding application. Scan logs cover
model loading, universe discovery, metadata degradation, progress every 50 processed
fetch/prediction results, final coverage and successful report publication. Refused
or unattempted members remain in the final ledger; they do not each emit a log line.
Archive ingestion, canonical materialization and model publication/activation also
emit stage messages. Logs do not replace coverage receipts or quality checks.

Python library calls never configure the root logger; applications may configure
standard logging themselves. Pure numerical feature transforms are not instrumented:
this avoids per-row logging and unnecessary feature-code identity/cache changes.
No automatic log files, rotation service or third-party logging dependency is added.

New report headings, explanations and warnings are in English, including the
research prediction CLIs. Existing reports are not rewritten. Native identifiers
and model-contract metadata remain source-faithful; readable reports use the
output ID when a legacy model label is non-ASCII. This changes presentation, not
model identity, prediction values or activation.

Each run creates `data.root/reports/<UTC-time-run-ID>/`:

- `report.md`: readable summary, top-30 ranking preview, scope, freshness and limitations.
- `report.json`: full model contract, every member's status, source hashes, cache identities and predictions.
- `predictions.csv`: all successful and failed members, input sources and product candidates; sortable and not limited to the top 30.
- `products.csv`: separate coverage for four product categories, exact-ticker spot candidates, mapping status and linked spot prediction status.
- `exchange-info.json`: original universe snapshot for provenance review.

Start with `report.md`; use JSON/CSV for complete coverage. The CLI prints the
report directory under the configured data root.
Older reports may predate `products.csv`; they are not rewritten retroactively.

From the project root in PowerShell, with the existing environment and active model:

```powershell
.\.venv\Scripts\crypto-boom.exe model list --config .\crypto-boom.toml
.\.venv\Scripts\crypto-boom.exe scan --config .\crypto-boom.toml
Get-ChildItem .\data\reports -Directory | Sort-Object Name -Descending | Select-Object -First 5
```

Scanning contacts public venue endpoints and writes local snapshots and reports;
it does not train a model or place orders. Open `report.md` inside the directory
printed by the command. Directory order alone does not establish freshness:
check the report's decision time and coverage status.

Success status is `success`; failures distinguish `metadata_error`, `fetch_error`,
`data_or_prediction_error`, `not_completed` and `not_attempted`. Exit code 0 means
all prediction members succeeded; 2 means partial completion with a published
report; 1 means startup, universe or publication failure. An unknown universe
does not produce a false coverage report. Publication is atomic and never overwrites previous reports.

P90 is a conditional quantile, not a 90% probability of rising. Ranking is not
trading advice. Newly listed assets may fall outside training coverage; complete
data does not establish predictive validity. Undeployed downside/quality heads do not mean zero risk.

## Trading products are separate from prediction inputs

The accepted scope includes Binance and OKX CEX spot and perpetual products.
There is no OKX candle, trade or feature adapter: OKX supplies product metadata
only. USDT remains the default scope; perpetuals are distinct from delivery
futures, with no automatic extension to coin-margined contracts, DEX, equity or commodity models.

- Prices, features and models remain Binance Spot-based; spot forecasts are not contract-return forecasts.
- Products preserve venue, spot/perpetual type, native ID, quote/settlement, specifications and snapshot time.
- Identical tickers are candidates, not verified asset identities. Multipliers such as `1000` are not stripped. Non-native mappings require verification; unavailable Binance Spot inputs mean uncovered, not a zero score.
- Public `live/TRADING` status does not establish account permissions, liquidity or executable returns. Funding, basis and liquidation risks are not predicted by the upside-quantile model.

Production `crypto-boom scan` includes the product pool;
the prediction-coverage denominator remains Binance Spot. `market-scan-v2` keeps
spot predictions in `rows` and products separately in `product_pool.products`.
`native` means native Binance Spot correspondence; other exact-ticker matches are
`ticker_match_unverified`. Candidates do not authorize execution, filter the
model population or turn scores into contract-return predictions.
`no_exact_ticker_match` does not prove absence (aliases/multipliers may exist),
and `source_unavailable` does not mean unlisted.

`product_metadata_status` is separate from prediction `status`; the CLI prints
both. Exit codes still follow spot prediction completion. Valid spot forecasts
can be published despite product-source failures; product-pool consumers must
check metadata status. Each source records time, errors or filtering counts,
received-byte hashes and base64 source bytes. Pool snapshot time is not claimed
to equal the model origin. There is no cache fallback, retry or domain switching;
418/429 skips subsequent product requests to that host. This adds no OKX market-data
adapter, account query, order placement or asset-identity certification.

## Optional inference input evidence

```sh
uv run --extra prediction crypto-boom scan --record-inputs --config crypto-boom.toml
```

This opt-in operation records inputs, not drift conclusions. Default scans retain
`market-scan-v2`; opted-in scans write `market-scan-v3` with independent
`input_evidence` status in JSON, Markdown and CLI JSON stdout. Prediction exit codes
remain based on prediction coverage, not optional evidence availability.

`model_runtime.predict_bars(..., include_inputs=False)` retains its existing
prediction-value dictionary. With `include_inputs=True`, it returns an envelope:
`values` contains that same dictionary; `inputs` contains `state`, `reason` and a
read-only NumPy `vector` (or `None`). Features are computed once; the copy preserves
the actual dtype and feature order. Runtime performs no IO. If an estimator mutates
shared input, evidence is unavailable without changing normal numerical execution.
No particular cadence, feature count or prediction target is fixed by this API.

`scan_market(settings, record_inputs=True)` retains at most 16 MB of raw vectors;
this is not a total process-memory limit. Additional vectors are refused with
`input_budget_exceeded` while scores remain. Every successful prediction has a
per-member evidence state. Failed predictions do not imply captured inputs.

When available, `inputs.parquet` contains symbol, decision time, snapshot cache ID,
source SHA256 and a fixed-width typed feature vector. `report.json` binds its hash,
size, dtype, feature count, row count, model/recipe identity and runtime/feature/bar
code hashes. The report and sidecar share the existing atomic publication directory;
no circular report/sidecar hash dependency is introduced. Verify the hash and
metadata before use; hashes establish integrity, not authenticity. Evidence status:

- `complete`: every successful prediction has captured input evidence.
- `partial`: only some successful predictions have evidence.
- `unavailable`: no usable sidecar; consult per-member and publication reasons.

Optional writing failure removes partial sidecar bytes and records unavailability,
while preserving prediction results if the main report can still be written.
Cleanup failure, cancellation or main report failure aborts publication; there is
no guarantee of successful reporting on an unusable filesystem. Files are not
updated after publication. Logs use the existing English stderr logging mechanism.

These are deployment observations, not automatically a training reference or a
representative market sample. Reference-window choice, drift statistics, thresholds
and model-update policy belong to research. No automatic retention, training,
alarm service or activation is added. Readers of old v1/v2 scan reports must treat
absent evidence as unavailable, never reconstruct and label it original capture.
