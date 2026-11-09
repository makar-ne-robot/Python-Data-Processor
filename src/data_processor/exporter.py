"""CSV export utilities.

Exports are produced in memory.  String cells that would be interpreted as
formulas by spreadsheet programs are neutralised by prefixing them with an
apostrophe, which prevents CSV Formula Injection.
"""

from __future__ import annotations

import codecs
import io

import pandas as pd

from data_processor.exceptions import ExportError

DANGEROUS_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")
"""Characters that spreadsheet applications may treat as formula prefixes."""


def sanitize_cell(value: object) -> object:
    """Neutralise a spreadsheet formula injection attempt in one cell.

    Non-string values are returned unchanged.  Strings starting with one of the
    dangerous prefixes (optionally after whitespace) receive a leading
    apostrophe so spreadsheet software displays them as text.

    Args:
        value: A single cell value.

    Returns:
        The safe value.
    """
    if not isinstance(value, str) or not value:
        return value
    candidate = value.lstrip(" \t\r\n")
    if candidate.startswith(DANGEROUS_PREFIXES):
        return "'" + value
    return value


def sanitize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``dataframe`` with dangerous string cells neutralised.

    Args:
        dataframe: Frame to sanitise.

    Returns:
        A new frame; the input is not modified.
    """
    working = dataframe.copy(deep=True)
    for column in working.columns:
        series = working[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            working[column] = series.map(sanitize_cell)
    return working


def _validate_delimiter(delimiter: str) -> None:
    if len(delimiter) != 1 or delimiter in {"\n", "\r", '"'}:
        raise ExportError("Delimiter must be a single character other than a quote or newline.")


def _validate_encoding(encoding: str) -> None:
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise ExportError(f"Unknown encoding: {encoding!r}.") from exc


def export_csv(
    dataframe: pd.DataFrame,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
    include_index: bool = False,
    protect_formulas: bool = True,
) -> bytes:
    """Serialise a DataFrame to CSV bytes.

    Args:
        dataframe: Frame to export.
        delimiter: Single-character field separator.
        encoding: Target text encoding.
        include_index: Whether to write the DataFrame index.
        protect_formulas: When ``True``, neutralise dangerous spreadsheet
            formulas before serialisation.

    Returns:
        The encoded CSV document.

    Raises:
        ExportError: If the delimiter/encoding is invalid or the frame cannot
            be represented in the requested encoding.
    """
    _validate_delimiter(delimiter)
    _validate_encoding(encoding)

    working = sanitize_dataframe(dataframe) if protect_formulas else dataframe
    buffer = io.StringIO()
    try:
        working.to_csv(
            buffer,
            sep=delimiter,
            index=include_index,
            lineterminator="\n",
            na_rep="",
        )
        text = buffer.getvalue()
        return text.encode(encoding)
    except UnicodeEncodeError as exc:
        raise ExportError(
            f"The data contains characters that cannot be encoded as {encoding}. "
            "Choose a different encoding (for example utf-8)."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ExportError(f"The data could not be exported: {exc}") from exc
