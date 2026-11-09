"""Tests for :mod:`data_processor.config` and error formatting."""

from __future__ import annotations

import pytest

from data_processor.config import DEFAULT_MAX_FILE_SIZE_MB, get_settings
from data_processor.exceptions import format_size


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_PROCESSOR_MAX_FILE_MB", raising=False)
    settings = get_settings()
    assert settings.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
    assert settings.warnings == ()
    assert "utf-8" in settings.encodings


def test_valid_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PROCESSOR_MAX_FILE_MB", "5")
    settings = get_settings()
    assert settings.max_file_size_bytes == 5 * 1024 * 1024
    assert settings.warnings == ()


def test_invalid_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PROCESSOR_MAX_FILE_MB", "abc")
    settings = get_settings()
    assert settings.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
    assert settings.warnings


def test_non_positive_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PROCESSOR_MAX_FILE_MB", "0")
    settings = get_settings()
    assert settings.warnings


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KB"), (1536, "1.5 KB")],
)
def test_format_size(size: int, expected: str) -> None:
    assert format_size(size) == expected


def test_format_size_negative() -> None:
    with pytest.raises(ValueError):
        format_size(-1)
