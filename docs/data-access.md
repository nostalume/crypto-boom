# Composable data capabilities and a shared data root

## Not a universal read(request)

Each dimension owns a distinct responsibility; callers compose them explicitly
rather than passing every data type through a large dispatch function:

| Dimension | Implementation | Ownership |
|---|---|---|
| Source and instrument | `market.InstrumentId(VenueId(...), environment, symbol)` | Explicit venue and native instrument, independent of models |
| Time range | `market.TimeWindow` | UTC microseconds, half-open `[start,end)`; convertible from timezone-aware datetimes |
| Native history source | `storage.source.select_minute_partitions` | Binance Spot minute-archive discovery, source checks, missing months, conflicting-version rejection |
| Typed reading | `bars.read_bar_window` | Bounded minute-Parquet reading and column projection; no network or implicit resampling |
| Time scale | `bars.BarPeriod` / `aggregate_minute_bars` | Actual minute OHLC aggregation; no filling, interpolation or future data |
| Latest source | `market_data.SpotSnapshotClient` | Universe, venue clock, network budgets and completed windows |
| Causal features | `features` | Features from selected inputs; no storage-location or training-target decisions |

Funding, individual trades and order books should have their own source adapters,
readers and transforms, reusing source identity, time windows and project roots.
They must not masquerade as candles. **Their storage/read capabilities are not
implemented in this change.** The historical adapter supports only Binance Spot
production / 1m and explicitly rejects other sources. `TimeWindow` expresses
event time, not evidence that data was available then; historical knowability
requires additional evidence.

## Offline composition example

```python
from datetime import UTC, datetime
from crypto_boom.config import project_settings
from crypto_boom.market import Environment, InstrumentId, TimeWindow, VenueId
from crypto_boom.storage.source import select_minute_partitions
from crypto_boom.bars import (
    SOURCE_COLUMNS,
    BarPeriod,
    read_bar_window,
    aggregate_minute_bars,
)

settings = project_settings()
instrument = InstrumentId(VenueId("binance", "spot"), Environment.PRODUCTION, "BTCUSDT")
window = TimeWindow.from_datetimes(
    datetime(2026, 1, 1, tzinfo=UTC),
    datetime(2026, 1, 2, tzinfo=UTC),
)
selection = select_minute_partitions(
    (settings.data_root / "canonical/archives",),
    instrument,
    window,
)
if selection.missing_months:
    raise ValueError(f"Missing archive months: {selection.missing_months}")
minutes, receipts = read_bar_window(
    [part.path / "klines.parquet" for part in selection.partitions],
    window,
    columns=(*SOURCE_COLUMNS, "open_price"),
)
hours = aggregate_minute_bars(minutes, BarPeriod(60))
```

This requires existing local data and never silently downloads missing inputs.
Multiple instruments are explicit collections composed individually. A live
market-wide universe comes from venue snapshots, not a research sample pool.
Finding a month does not establish minute completeness: admission/aggregation
still rejects gaps, bad quality and incomplete edge buckets. Different manifests
for the same month raise version ambiguity; pinned research can still read its
chosen version. Modification times and mount order do not determine data truth.

An exact minute uses `[t,t+1 minute)`; history before an instant uses an explicit
lookback window. No implicit nearest/asof filling exists. Market-wide latest
resolves to a common completed boundary at scan start. Data period, historical
lookback and future target horizon are distinct; the latter two belong to models/research.

Aggregation aligns to the UTC Unix epoch, accepts integer periods of 1–1,440
minutes, and requires continuous, valid, complete buckets. Open/close use first/last;
high/low use extrema; quote turnover, taker-buy quote amount and trade count are
summed. The result is a Float64 analytical view, not a replacement for native
archive decimal precision. Real `open_price` is required; projections lacking it
must not invent opens from previous closes. Aggregated output is not minute data
and cannot be passed to minute-input models or re-aggregated as though it were.

## One project configuration

Copy `crypto-boom.example.toml` to `crypto-boom.toml`:

```toml
[data]
root = "data"

[scan]
workers = 4
timeout_seconds = 900
```

The data root is relative to the config file. Old `legacy_corpora` configuration
was removed; additional corpora use explicit research `--reuse-corpus`, not
permanent experiment-directory mounts. Config discovery walks upward; explicit
`--config` takes priority, without silently merging configurations. Without
project context, explicit configuration is required to avoid creating unrelated
data roots.

Default locations:

```text
data/
├─ raw/monthly/          # Newly acquired raw archives
├─ canonical/archives/  # Canonical history in its versioned native layout
├─ snapshots/           # Exact completed scan windows and validation records
├─ derived/features/    # Feature caches shared across research runs
├─ derived/targets/     # Target caches shared across research runs
├─ models/              # Registered models and activation records
├─ reports/             # Scan reports
└─ runs/                # Sample selections, dataset manifests and run records
```

Public `crypto-boom scan/model`, research `acquire/build` and current-model
`publish` use project-config defaults. `acquire/build --output` changes experiment
record location only, not the location of new market data or caches. Low-level
functions still allow explicit storage roots; old research callers that omit new
parameters retain their isolation contract. `[scan].data_dir` was removed;
scanning uses `ProjectSettings` without a second settings object. Explicit
`--output` on old archive commands and explicit paths in other model scripts have
not all been rewritten; they are outside the new default workflow. This is not
a claim that every legacy entry point has been migrated.

## Portable caches and manifests

New writes use `feature-cache-v2`, `sample-pool-v2` and `path-dataset-v2`; old v1
remains readable without overwriting evidence. Targets retain their already
path-independent v1 cache format.

- Feature identity includes source-byte hashes/sizes, code and sampling policy, not filename, absolute path or input order. This is **file-byte identity**: re-encoding logically identical Parquet rows may change identity.
- Pool identity includes selection, availability and partition facts; dataset identity includes pool and feature/target identities.
- New manifests store relative references; loaders return absolute-path runtime views without rewriting disk manifests. Moving the whole referenced tree preserves references; copying a manifest alone does not. Reference creation requires the same volume. External mounts do not move automatically, and invalid references do not trigger whole-disk searches.
- After moving raw files separately, explicitly rebind through `feature_source_paths(..., paths=[...])` or research `build_target_cache(..., source_paths=[...])`, verifying byte identity. Dataset builds save current raw bindings for sequence research; changed bindings require a new run directory. Reading cached features does not require online raw files; reusing raw data does require identity checks.

A readable acquisition pool can be exported without moving prices or overwriting history:

```python
from crypto_boom.sample_pool import export_sample_pool

export_sample_pool(
    settings.data_root / "runs/source-pool/pool.json",
    settings.data_root / "runs/portable-pool/pool.json",
)
```

Here `settings` comes from `project_settings()` above. Export verifies the old
pool and partitions, and rejects an existing destination. Old feature caches are
not automatically converted to v2; new builds use new identities while old caches
remain for replay. Reference-aware safe deduplication/cleanup and fine-grained
sharing between latest windows and historical partitions are not implemented.

No universal DataManager, type-plugin registry or database is introduced. The
versioned archive layout already supports current partition lookup. If future
multi-source queries need an index, real consumers should determine its minimum contract.

## Historical evidence and rebinding

Historical reports and caches are not rewritten by relocation. A v1 manifest may
contain retired absolute raw paths: cached tables can remain readable while raw
replay requires explicit byte-verified rebinding. Never guess replacement files
from names or modification times. Moving directories does not refit models or
renew the validity of historical evaluation.

## Source decoder import compatibility

Import the Binance REST minute decoder from
`crypto_boom.binance_source.decode_minute_page`. The former import from
`crypto_boom.bars` is removed without a forwarding alias. Signature, wire checks
and decoded values are unchanged; minute reading, admission and aggregation stay
in `bars`. Research latest fetching and market-wide snapshots retain their own
clock and request policies.

Feature-cache identities include the bytes of `bars.py`. This relocation therefore
changes newly built cache identities, not feature values. Existing verified caches
remain readable and are not removed or rebuilt automatically. Model weights,
activation, archive formats and snapshot/report schemas are unchanged.

### Snapshot publication recovery

Minute snapshots use the shared atomic directory-publication mechanism. If another
writer publishes the same cache key, its receipt and data hash must match the
staged artifact exactly before reuse; conflicting content is refused, not replaced.
Only the losing writer's staging directory is cleaned up.

Windows access-denied, sharing and lock errors (5, 32, 33) receive at most three
additional rename attempts after 50, 100 and 200 milliseconds. Warnings identify
the failed publication and retry. This is a maximum 350-millisecond sleep budget,
not a hard IO deadline; synchronous callers can block during it. A persistent
permission error still propagates, unrelated errors are not retried, and
cancellation is not swallowed. No ACL changes or HTTP retries are performed.
A scan can therefore still finish with explicit partial coverage; successful
publication recovery is not proof that the cause of an earlier failure is known.
