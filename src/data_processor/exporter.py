"""CSV export utilities.

Exports are produced in memory.  Text values that a spreadsheet program could
interpret as a formula are neutralised by prefixing them with an apostrophe so
the program treats the cell as text.  The same treatment is applied to exported
column headers and, when ``include_index=True``, to the index values and names.

Limitations
-----------
This is a best-effort mitigation, not a guarantee against every CSV or
spreadsheet injection technique:

* Only text values whose first non-whitespace character is ``=``, ``+``, ``-``
  or ``@`` (or a tab/CR) are neutralised.  Other spreadsheet behaviours, such
  as payloads split across several cells, are out of scope.
* The apostrophe becomes part of the exported text and is visible in plain text
  editors; some spreadsheet programs hide it.
* Numeric values (including ordinary negative numbers) are never modified.
* Sanitising column names can, in pathological cases, map two distinct names to
  the same string, producing duplicate CSV headers.
"""

from __future__ import annotations

import codecs
import io
from typing import Any

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


def _is_textual_dtype(dtype: Any) -> bool:
    """Return ``True`` for dtypes that can hold text values."""
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        categories = dtype.categories
        return bool(
            pd.api.types.is_object_dtype(categories) or pd.api.types.is_string_dtype(categories)
        )
    return False


def _sanitize_index(index: pd.Index) -> pd.Index:
    """Return an index whose textual values and names are neutralised.

    Supports a regular index and a :class:`pandas.MultiIndex`.  Non-textual
    values (numbers, dates, ...) are left untouched.
    """
    if isinstance(index, pd.MultiIndex):
        names = [sanitize_cell(name) if isinstance(name, str) else name for name in index.names]
        tuples = [tuple(sanitize_cell(value) for value in values) for values in index]
        return pd.MultiIndex.from_tuples(tuples, names=names)

    sanitized: pd.Index = index
    if _is_textual_dtype(index.dtype):
        sanitized = index.map(sanitize_cell)
    if isinstance(index.name, str):
        sanitized = sanitized.set_names(sanitize_cell(index.name))
    return sanitized


def sanitize_dataframe(dataframe: pd.DataFrame, *, include_index: bool = False) -> pd.DataFrame:
    """Return a copy with dangerous text in cells, headers and index neutralised.

    Cells of object, ``StringDtype`` and textual categorical columns are
    sanitised.  Column headers are always sanitised.  The index values and
    names are sanitised only when ``include_index`` is ``True``, because that is
    when they are written to the CSV.

    Args:
        dataframe: Frame to sanitise.
        include_index: Whether the index is exported and therefore needs
            sanitising.

    Returns:
        A new frame; the input is not modified.
    """
    working = dataframe.copy(deep=True)

    for column in list(working.columns):
        series = working[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.categories
            if _is_textual_dtype(categories.dtype):
                working[column] = series.astype("string").map(sanitize_cell)
        elif _is_textual_dtype(series.dtype):
            working[column] = series.map(sanitize_cell)

    working.columns = _sanitize_index(working.columns)

    if include_index:
        working.index = _sanitize_index(working.index)

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
            formulas in cells, headers and (when ``include_index`` is set) the
            index before serialisation.

    Returns:
        The encoded CSV document.

    Raises:
        ExportError: If the delimiter/encoding is invalid or the frame cannot
            be represented in the requested encoding.
    """
    _validate_delimiter(delimiter)
    _validate_encoding(encoding)

    working = (
        sanitize_dataframe(dataframe, include_index=include_index)
        if protect_formulas
        else dataframe
    )
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
