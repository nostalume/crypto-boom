"""Admission and identity for the minimal run configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class ConfigAdmissionError(ValueError):
    """The supplied configuration cannot enter the admitted domain."""


@dataclass(frozen=True, slots=True)
class AdmittedRunConfig:
    """A validated configuration and its content identity."""

    schema_version: Literal[1]
    config_id: str


def admit_run_config(document: str) -> AdmittedRunConfig:
    """Validate TOML text once and return its admitted identity."""

    try:
        raw = tomllib.loads(document)
    except tomllib.TOMLDecodeError as error:
        raise ConfigAdmissionError("configuration is not valid TOML") from error

    if set(raw) != {"schema_version"}:
        raise ConfigAdmissionError(
            "configuration fields must be exactly: schema_version"
        )

    schema_version = raw["schema_version"]
    if type(schema_version) is not int:
        raise ConfigAdmissionError("schema_version must be an integer")
    if schema_version != 1:
        raise ConfigAdmissionError("schema_version is not supported")

    canonical = json.dumps(
        {"schema_version": schema_version},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return AdmittedRunConfig(
        schema_version=1,
        config_id=f"sha256:{digest}",
    )


@dataclass(frozen=True)
class ProjectSettings:
    """Project location, not the caller's working directory, owns persisted data."""

    data_root: Path
    workers: int = 4
    timeout_seconds: int = 900

    def __post_init__(self) -> None:
        if type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("workers must be 1..4")
        if (
            type(self.timeout_seconds) is not int
            or not 1 <= self.timeout_seconds <= 900
        ):
            raise ValueError("timeout_seconds must be 1..900")


def project_settings(path: Path | None = None) -> ProjectSettings:
    """Read explicit config or discover crypto-boom.toml upwards; no I/O on import.

    Missing project config defaults to the nearest crypto-boom pyproject's data/.
    Only the project data/scan tables are admitted; no legacy scan.data_dir.
    """
    if path is None:
        for parent in (Path.cwd(), *Path.cwd().parents):
            candidate = parent / "crypto-boom.toml"
            if candidate.is_file():
                path = candidate
                break
            if (parent / "scan.toml").is_file():
                raise ValueError(
                    "migrate scan.toml to crypto-boom.toml with [data].root"
                )
            manifest = parent / "pyproject.toml"
            if manifest.is_file():
                name = (
                    tomllib.loads(manifest.read_text(encoding="utf-8"))
                    .get("project", {})
                    .get("name")
                )
                if name != "crypto-boom":
                    raise ValueError(
                        "different project boundary; pass explicit --config"
                    )
                return ProjectSettings((parent / "data").resolve())
        if path is None:
            raise ValueError("no project configuration; pass --config crypto-boom.toml")
    if not path.is_file():
        raise ValueError("explicit project configuration does not exist")
    if path.stat().st_size > 64_000:
        raise ValueError("project configuration exceeds size limit")
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(document) - {"data", "scan"}:
        raise ValueError("unknown project configuration keys")
    data, scan = document.get("data", {}), document.get("scan", {})
    if not isinstance(data, dict) or not isinstance(scan, dict):
        raise ValueError("data and scan must be configuration tables")
    if set(data) - {"root"} or set(scan) - {
        "workers",
        "timeout_seconds",
    }:
        raise ValueError("unknown or conflicting project configuration keys")
    raw_root = data.get("root", "data")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise ValueError("invalid data root")
    root = (path.resolve().parent / raw_root).resolve()
    return ProjectSettings(
        root,
        scan.get("workers", 4),
        scan.get("timeout_seconds", 900),
    )
