"""Admission and identity for the minimal run configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
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
