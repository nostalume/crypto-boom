"""Origin-anchored future close-path outcomes, never model input features."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US, admit_bars, load_bar_files
from crypto_boom.feature_batch import feature_source_paths, read_feature_cache


@dataclass(frozen=True)
class PathTargetSpec:
    horizons: tuple[int, ...] = (120, 360)
    amplitudes: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.horizons) <= 8
            or any(type(h) is not int or not 1 <= h <= 4320 for h in self.horizons)
            or tuple(sorted(set(self.horizons))) != self.horizons
        ):
            raise ValueError("invalid path horizons")
        if (
            len(self.amplitudes) > 8
            or any(
                not math.isfinite(a) or not 0 < a <= 5 or round(a, 6) != a
                for a in self.amplitudes
            )
            or tuple(sorted(set(self.amplitudes))) != self.amplitudes
        ):
            raise ValueError(
                "invalid amplitude queries; require sorted positive values at six-decimal precision"
            )


DEFAULT_TARGETS = PathTargetSpec()
PATH_MEASURES = (
    "up",
    "down",
    "terminal",
    "retention_up",
    "retention_down",
    "efficiency",
    "above_fraction",
    "below_fraction",
    "drawdown",
    "peak_minute",
    "trough_minute",
)


def path_targets(
    source: pl.DataFrame, origins: pl.DataFrame, spec: PathTargetSpec = DEFAULT_TARGETS
) -> pl.DataFrame:
    """Bounded vectorized windows; only a complete contiguous horizon is labeled."""
    source = admit_bars(source)
    keys = origins.select("symbol", "decision_us").sort("decision_us")
    if keys.select(pl.struct("symbol", "decision_us").n_unique()).item() != len(
        keys
    ) or (len(keys) and set(keys["symbol"]) != set(source["symbol"])):
        raise ValueError("invalid origin keys")
    valid = (
        source.filter(pl.col("quality_complete") & (pl.col("quality_state") == "valid"))
        .with_columns(pl.col("open_time").dt.epoch("us").alias("open_us"))
        .with_columns(
            (pl.col("open_us").diff() != MINUTE_US)
            .fill_null(True)
            .cum_sum()
            .alias("segment")
        )
    )
    columns: dict[str, np.ndarray] = {}
    for h in spec.horizons:
        for name in PATH_MEASURES:
            columns[f"{name}_{h}"] = np.full(len(keys), np.nan)
        for a in spec.amplitudes:
            token = str(a).replace(".", "p")
            for side in ("up", "down"):
                for measure in ("hit", "first_minute", "adverse_before"):
                    columns[f"{side}_{measure}_{h}_{token}"] = np.full(
                        len(keys), np.nan
                    )
    times = keys["decision_us"].to_numpy()
    seen = np.zeros(len(keys), dtype=bool)
    for frame in valid.partition_by("segment", maintain_order=True):
        decisions = frame["open_us"].to_numpy() + MINUTE_US
        selected = np.flatnonzero((times >= decisions[0]) & (times <= decisions[-1]))
        indices = np.searchsorted(decisions, times[selected])
        if not np.array_equal(decisions[indices], times[selected]):
            raise ValueError("origin is not a valid completed bar")
        seen[selected] = True
        prices = frame["close_price"].to_numpy()
        for h in spec.horizons:
            complete = indices + h < len(prices)
            origins_index, output_index = indices[complete], selected[complete]
            for start in range(0, len(origins_index), 512):
                ix = origins_index[start : start + 512]
                out = output_index[start : start + 512]
                window = prices[ix[:, None] + np.arange(h + 1)]
                r = window / window[:, :1] - 1
                up, down, terminal = r.max(axis=1), -r.min(axis=1), r[:, -1]
                travel = np.abs(np.diff(r, axis=1)).sum(axis=1)
                values = {
                    "up": up,
                    "down": down,
                    "terminal": terminal,
                    "retention_up": np.divide(
                        np.maximum(terminal, 0),
                        up,
                        out=np.full(len(ix), np.nan),
                        where=up > 0,
                    ),
                    "retention_down": np.divide(
                        np.maximum(-terminal, 0),
                        down,
                        out=np.full(len(ix), np.nan),
                        where=down > 0,
                    ),
                    "efficiency": np.divide(
                        terminal, travel, out=np.zeros(len(ix)), where=travel > 0
                    ),
                    "above_fraction": (r[:, 1:] > 0).mean(axis=1),
                    "below_fraction": (r[:, 1:] < 0).mean(axis=1),
                    "drawdown": (
                        1 - window / np.maximum.accumulate(window, axis=1)
                    ).max(axis=1),
                    "peak_minute": r.argmax(axis=1),
                    "trough_minute": r.argmin(axis=1),
                }
                for name, value in values.items():
                    columns[f"{name}_{h}"][out] = value
                for a in spec.amplitudes:
                    token = str(a).replace(".", "p")
                    for side, direction in (("up", 1), ("down", -1)):
                        crossing = direction * r >= a
                        hit = crossing.any(axis=1)
                        first = crossing.argmax(axis=1)
                        adverse = np.maximum.accumulate(
                            np.maximum(-direction * r, 0), axis=1
                        )[np.arange(len(ix)), first]
                        columns[f"{side}_hit_{h}_{token}"][out] = hit
                        columns[f"{side}_first_minute_{h}_{token}"][out] = np.where(
                            hit, first, np.nan
                        )
                        columns[f"{side}_adverse_before_{h}_{token}"][out] = np.where(
                            hit, adverse, np.nan
                        )
    if not seen.all():
        raise ValueError("origin has no valid source observation")
    return keys.with_columns(
        [pl.Series(name, values, nan_to_null=True) for name, values in columns.items()]
    )


def read_target_cache(path: Path) -> tuple[pl.DataFrame, dict]:
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "path-target-cache-v1"
        or _artifacts.content_id(receipt["spec"]) != receipt["cache_id"]
    ):
        raise ValueError("invalid target cache receipt")
    if (
        _artifacts.file_identity(path / "targets.parquet")[0]
        != receipt["parquet_sha256"]
    ):
        raise ValueError("target cache hash mismatch")
    frame = pl.read_parquet(path / "targets.parquet")
    if len(frame) != receipt["rows"] or frame.select(
        pl.struct("symbol", "decision_us").n_unique()
    ).item() != len(frame):
        raise ValueError("invalid cached target keys")
    return frame, receipt


def build_target_cache(
    feature_cache: Path,
    *,
    output_root: Path,
    spec: PathTargetSpec = DEFAULT_TARGETS,
    source_paths: list[Path] | None = None,
) -> tuple[Path, bool]:
    features, receipt = read_feature_cache(feature_cache)
    identity = {
        "feature_cache_id": receipt["cache_id"],
        "target_spec": asdict(spec),
        "target_code": _artifacts.file_identity(Path(__file__))[0],
    }
    cache_id = _artifacts.content_id(identity)
    target = output_root / cache_id.removeprefix("sha256:")
    if target.exists():
        if read_target_cache(target)[1]["cache_id"] != cache_id:
            raise ValueError("target cache location has another identity")
        return target, True
    paths = feature_source_paths(feature_cache, receipt, paths=source_paths)
    source, _ = load_bar_files(paths, start_us=0, end_us=2**63 - 1)
    targets = path_targets(source, features, spec)
    output_root.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output_root, prefix="targets-"
    ) as staging:
        targets.write_parquet(staging / "targets.parquet")
        record: dict[str, object] = {
            "schema": "path-target-cache-v1",
            "cache_id": cache_id,
            "spec": identity,
            "rows": len(targets),
            "parquet_sha256": _artifacts.file_identity(staging / "targets.parquet")[0],
            "complete_by_horizon": {
                str(h): len(targets) - targets[f"up_{h}"].null_count()
                for h in spec.horizons
            },
        }
        _artifacts.write_exclusive_bytes(
            staging / "receipt.json", _artifacts.canonical_json(record)
        )

        def verify(path: Path) -> None:
            if read_target_cache(path)[1]["cache_id"] != cache_id:
                raise ValueError("target cache publication identity conflict")

        _artifacts.adopt_directory(staging, target, verify_existing=verify)
    return target, False
