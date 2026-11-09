"""Application configuration and immutable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_FILE_SIZE_MB = 50
"""Default maximum CSV size in megabytes."""

SUPPORTED_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "cp1251",
    "cp1252",
    "iso-8859-1",
)
"""Encodings offered in the UI and used for automatic detection."""

DELIMITER_CHOICES: tuple[str, ...] = (",", ";", "\t", "|")
"""Delimiter values offered in the UI."""

PREVIEW_ROWS = 100
"""Number of rows shown in table previews."""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

SAMPLE_DATA_PATH = PROJECT_ROOT / "examples" / "sample_data.csv"
"""Path to the bundled example dataset."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings derived from environment variables.

    Attributes:
        max_file_size_bytes: Hard limit for uploaded files.
        preview_rows: Default number of rows in previews.
        encodings: Encodings considered during automatic detection.
        delimiters: Delimiters considered during automatic detection.
    """

    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
    preview_rows: int = PREVIEW_ROWS
    encodings: tuple[str, ...] = SUPPORTED_ENCODINGS
    delimiters: tuple[str, ...] = DELIMITER_CHOICES
    warnings: tuple[str, ...] = field(default_factory=tuple)


def get_settings() -> Settings:
    """Build settings from the environment.

    The maximum upload size can be configured with
    ``DATA_PROCESSOR_MAX_FILE_MB``.  Invalid values fall back to the default
    instead of crashing the application.

    Returns:
        An immutable :class:`Settings` instance.
    """
    warnings: list[str] = []
    max_mb = DEFAULT_MAX_FILE_SIZE_MB
    raw = os.environ.get("DATA_PROCESSOR_MAX_FILE_MB")
    if raw is not None:
        try:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError
            max_mb = parsed
        except ValueError:
            warnings.append(
                f"DATA_PROCESSOR_MAX_FILE_MB={raw!r} is invalid; "
                f"using {DEFAULT_MAX_FILE_SIZE_MB} MB."
            )
    return Settings(
        max_file_size_bytes=max_mb * 1024 * 1024,
        warnings=tuple(warnings),
    )
