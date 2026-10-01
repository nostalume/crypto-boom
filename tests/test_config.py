from __future__ import annotations

import pytest

from crypto_boom.config import (
    ConfigAdmissionError,
    admit_run_config,
)


def test_equivalent_documents_have_the_same_config_identity() -> None:
    compact = admit_run_config("schema_version=1")
    spaced = admit_run_config("schema_version = 1\n")

    assert compact == spaced
    assert compact.config_id.startswith("sha256:")
    assert len(compact.config_id) == len("sha256:") + 64


@pytest.mark.parametrize(
    ("document", "expected_message"),
    [
        ("schema_version = ", "configuration is not valid TOML"),
        ("schema_version = true", "schema_version must be an integer"),
        ("schema_version = 2", "schema_version is not supported"),
        ("", "configuration fields must be exactly: schema_version"),
    ],
)
def test_invalid_configuration_is_rejected_without_echoing_input(
    document: str,
    expected_message: str,
) -> None:
    secret_marker = "do-not-echo"

    with pytest.raises(ConfigAdmissionError) as raised:
        admit_run_config(f"{document}\n# {secret_marker}")

    assert str(raised.value) == expected_message
    assert secret_marker not in str(raised.value)
