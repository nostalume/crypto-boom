"""Frozen successor-M1 row scoring, independent of the research runners.

This is **not** a live alert engine. Callers must supply admitted, as-of feature
values and a decision minute. The historical instant/slot construction and the
confirm/stand-down policy are deliberately not imported or reproduced here.
The returned number is an unvalidated decision score, not a calibrated
probability or a recommendation.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from itertools import pairwise
from typing import Any

_BUNDLE_NAME = "m1-v1.json"
_DAY_MINUTES = 1440
_HARMONICS = 3
_ETA_CLIP = 35.0
_SPLINES = ("comove_share", "hours_since_completion", "log_range24", "breadth")


class M1ScoringError(ValueError):
    """A row or bundle cannot be scored under the frozen M1 code space."""


@dataclass(frozen=True, slots=True)
class M1Score:
    cell: str
    symbol: str
    unvalidated_score: float
    bundle_sha256: str


@dataclass(frozen=True, slots=True)
class _Spline:
    name: str
    median: float
    degree: int
    knots: tuple[float, ...]

    def values(self, raw: float) -> tuple[float, ...]:
        value = raw if math.isfinite(raw) else self.median
        knots = self.knots
        degree = self.degree
        count = len(knots) - degree - 1
        value = min(max(value, knots[degree]), knots[count])
        if value == knots[count]:
            return (0.0,) * (count - 1) + (1.0,)
        basis = [float(knots[i] <= value < knots[i + 1]) for i in range(len(knots) - 1)]
        for level in range(1, degree + 1):
            next_basis = []
            for i in range(len(basis) - 1):
                left_width = knots[i + level] - knots[i]
                right_width = knots[i + level + 1] - knots[i + 1]
                left = (value - knots[i]) * basis[i] / left_width if left_width else 0.0
                right = (
                    (knots[i + level + 1] - value) * basis[i + 1] / right_width
                    if right_width
                    else 0.0
                )
                next_basis.append(left + right)
            basis = next_basis
        return tuple(basis[:count])


@dataclass(frozen=True, slots=True)
class _Cell:
    beta: tuple[float, ...]
    frailty: tuple[float, ...]
    splines: tuple[_Spline, ...]
    added_columns: tuple[str, ...]
    added_medians: tuple[float, ...]
    thresholds: tuple[float, ...]
    levels: tuple[float, ...]


def _number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = row[name]
    except KeyError as error:
        raise M1ScoringError(f"missing M1 feature: {name}") from error
    if value is None:
        return math.nan
    if isinstance(value, (str, bytes, bool)):
        raise M1ScoringError(f"M1 feature {name} is not numeric")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise M1ScoringError(f"M1 feature {name} is not numeric") from error


def _interpolate(value: float, x: tuple[float, ...], y: tuple[float, ...]) -> float:
    if value <= x[0]:
        return y[0]
    if value >= x[-1]:
        return y[-1]
    upper = bisect_right(x, value)
    lower = upper - 1
    fraction = (value - x[lower]) / (x[upper] - x[lower])
    return y[lower] + fraction * (y[upper] - y[lower])


class FrozenM1Scorer:
    """Score prepared rows for the 141 frozen symbols and two M1 cells.

    This arithmetic does not establish that a row was available at its minute.
    """

    def __init__(self, payload: bytes) -> None:
        record = json.loads(payload)
        if record.get("schema_version") != 1:
            raise M1ScoringError("unsupported M1 bundle schema")
        symbols = tuple(record["symbols"])
        if len(symbols) != 141 or len(set(symbols)) != len(symbols):
            raise M1ScoringError("M1 bundle has an invalid frozen symbol list")
        if set(record["cells"]) != {"primary", "secondary"}:
            raise M1ScoringError("M1 bundle must contain exactly the two frozen cells")
        cells: dict[str, _Cell] = {}
        for name, item in record["cells"].items():
            splines = tuple(
                _Spline(
                    entry["name"],
                    float(entry["median"]),
                    int(entry["degree"]),
                    tuple(float(value) for value in entry["knots"]),
                )
                for entry in item["splines"]
            )
            added = tuple(item["added_columns"])
            cell = _Cell(
                beta=tuple(float(value) for value in item["beta"]),
                frailty=tuple(float(value) for value in item["frailty"]),
                splines=splines,
                added_columns=added,
                added_medians=tuple(float(item["added_medians"][key]) for key in added),
                thresholds=tuple(float(value) for value in item["layer"]["thresholds"]),
                levels=tuple(float(value) for value in item["layer"]["levels"]),
            )
            width = (
                1
                + sum(len(spline.knots) - spline.degree - 1 for spline in splines)
                + 4
                + 6
                + len(added)
            )
            if (
                tuple(spline.name for spline in splines) != _SPLINES
                or len(cell.beta) != width
                or len(cell.frailty) != len(symbols)
                or len(cell.thresholds) != len(cell.levels)
                or len(cell.thresholds) < 2
                or any(a >= b for a, b in pairwise(cell.thresholds))
            ):
                raise M1ScoringError(f"invalid M1 bundle cell: {name}")
            cells[name] = cell
        self._symbols = {symbol: code for code, symbol in enumerate(symbols)}
        self._cells = cells
        self.bundle_sha256 = hashlib.sha256(payload).hexdigest()
        self.sources: Mapping[str, str] = record["sources"]

    @classmethod
    def from_package(cls) -> FrozenM1Scorer:
        """Load the tracked, package-local bundle; never fetch market data."""
        return cls(
            resources.files("crypto_boom.research").joinpath(_BUNDLE_NAME).read_bytes()
        )

    def score(
        self, cell: str, symbol: str, minute: int, features: Mapping[str, Any]
    ) -> M1Score:
        """Score one prepared feature row; labels and future fields are not read."""
        try:
            model = self._cells[cell]
        except KeyError as error:
            raise M1ScoringError(f"unknown M1 cell: {cell}") from error
        try:
            code = self._symbols[symbol]
        except KeyError as error:
            raise M1ScoringError(
                f"symbol is outside frozen M1 code space: {symbol}"
            ) from error
        if isinstance(minute, bool) or not isinstance(minute, int) or minute < 0:
            raise M1ScoringError("minute must be a non-negative integer")

        quintile = _number(features, "size_quintile")
        if not quintile.is_integer() or not 1 <= quintile <= 5:
            raise M1ScoringError("size_quintile must be an integer from 1 to 5")

        design = [1.0]
        for spline in model.splines:
            design.extend(spline.values(_number(features, spline.name)))
        design.extend(float(quintile == level) for level in range(2, 6))
        angle = 2.0 * math.pi * (minute % _DAY_MINUTES) / _DAY_MINUTES
        for harmonic in range(1, _HARMONICS + 1):
            design.extend((math.sin(harmonic * angle), math.cos(harmonic * angle)))
        for name, median in zip(model.added_columns, model.added_medians, strict=True):
            value = _number(features, name)
            design.append(value if math.isfinite(value) else median)
        eta = sum(
            weight * value for weight, value in zip(model.beta, design, strict=True)
        )
        eta += model.frailty[code]
        raw = 1.0 / (1.0 + math.exp(-min(max(eta, -_ETA_CLIP), _ETA_CLIP)))
        score = _interpolate(raw, model.thresholds, model.levels)
        return M1Score(cell, symbol, score, self.bundle_sha256)
