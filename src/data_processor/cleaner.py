"""Data-cleaning operations.

Every public function takes a DataFrame and returns a
:class:`~data_processor.models.CleaningResult`.  The input frame is never
mutated, which lets the UI keep the original data intact and offer an undo
stack.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from data_processor.exceptions import CleaningError
from data_processor.models import CleaningResult

_FILL_STRATEGIES = ("mean", "median")


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    """Validate that all requested columns exist.

    Args:
        dataframe: Frame to check against.
        columns: Requested column names.

    Returns:
        The validated list of column names.

    Raises:
        CleaningError: If any column is missing.
    """
    resolved: list[str] = []
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise CleaningError(f"Unknown columns: {', '.join(map(str, missing))}")
    resolved.extend(str(column) for column in columns)
    return resolved


def _missing_counts(dataframe: pd.DataFrame, columns: list[str]) -> int:
    if not columns:
        return 0
    return int(dataframe[columns].isna().sum().sum())


def count_missing(dataframe: pd.DataFrame, columns: list[str] | None = None) -> int:
    """Count missing cells, optionally restricted to ``columns``."""
    if columns is None:
        return int(dataframe.isna().sum().sum())
    resolved = _require_columns(dataframe, columns)
    return _missing_counts(dataframe, resolved)


def _validate_threshold(threshold_pct: float) -> float:
    """Validate and normalise a missing-value threshold in percent.

    Args:
        threshold_pct: Threshold in percent (0-100).

    Returns:
        The threshold as a finite float.

    Raises:
        CleaningError: If the value is not a finite number in ``[0, 100]``.
    """
    try:
        value = float(threshold_pct)
    except (TypeError, ValueError) as exc:
        raise CleaningError(
            "Missing threshold must be a number between 0 and 100 percent."
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise CleaningError("Missing threshold must be between 0 and 100 percent.")
    return value


def columns_above_missing_threshold(dataframe: pd.DataFrame, threshold_pct: float) -> list[str]:
    """Return columns whose missing share is greater than ``threshold_pct``.

    Args:
        dataframe: Frame to inspect.
        threshold_pct: Threshold in percent (0-100).

    Returns:
        Names of the offending columns.

    Raises:
        CleaningError: If the threshold is not a finite number in ``[0, 100]``.
    """
    threshold = _validate_threshold(threshold_pct)
    rows = len(dataframe)
    if rows == 0:
        return []
    missing_pct = dataframe.isna().mean() * 100.0
    return [str(name) for name in missing_pct[missing_pct > threshold].index]


def drop_full_duplicates(dataframe: pd.DataFrame) -> CleaningResult:
    """Remove rows that are complete duplicates of an earlier row."""
    before = len(dataframe)
    working = dataframe.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(working)
    return CleaningResult(
        dataframe=working,
        affected_rows=removed,
        affected_cells=0,
        description=f"Removed {removed} full duplicate row(s).",
        details={"removed_rows": removed},
    )


def drop_duplicates_by_columns(dataframe: pd.DataFrame, columns: list[str]) -> CleaningResult:
    """Remove rows whose values duplicate on ``columns``.

    Args:
        dataframe: Frame to clean.
        columns: Columns that define a duplicate.

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If ``columns`` is empty or contains unknown names.
    """
    if not columns:
        raise CleaningError("Select at least one column for duplicate detection.")
    resolved = _require_columns(dataframe, columns)
    before = len(dataframe)
    working = dataframe.drop_duplicates(subset=resolved, keep="first").reset_index(drop=True)
    removed = before - len(working)
    return CleaningResult(
        dataframe=working,
        affected_rows=removed,
        affected_cells=0,
        description=(
            f"Removed {removed} duplicate row(s) based on columns: {', '.join(resolved)}."
        ),
        details={"removed_rows": removed, "columns": resolved},
    )


def drop_rows_with_missing(
    dataframe: pd.DataFrame, columns: list[str] | None = None
) -> CleaningResult:
    """Drop rows that contain at least one missing value.

    Args:
        dataframe: Frame to clean.
        columns: Restrict the check to these columns.  ``None`` checks all.

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If ``columns`` contains unknown names.
    """
    if columns is None:
        subset: list[str] | None = None
    else:
        subset = _require_columns(dataframe, columns)
    before = len(dataframe)
    working = dataframe.dropna(subset=subset, how="any").reset_index(drop=True)
    removed = before - len(working)
    return CleaningResult(
        dataframe=working,
        affected_rows=removed,
        affected_cells=0,
        description=f"Dropped {removed} row(s) containing missing values.",
        details={"removed_rows": removed, "columns": subset},
    )


def drop_columns_above_missing_threshold(
    dataframe: pd.DataFrame, threshold_pct: float
) -> CleaningResult:
    """Drop columns whose missing share exceeds ``threshold_pct``.

    Args:
        dataframe: Frame to clean.
        threshold_pct: Threshold in percent (0-100).

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If the threshold is not a finite number in ``[0, 100]``.
    """
    threshold = _validate_threshold(threshold_pct)
    to_drop = columns_above_missing_threshold(dataframe, threshold)
    working = dataframe.drop(columns=to_drop).reset_index(drop=True)
    return CleaningResult(
        dataframe=working,
        affected_rows=0,
        affected_cells=0,
        description=f"Dropped {len(to_drop)} column(s) above {threshold:g}% missing.",
        details={"dropped_columns": to_drop, "threshold_pct": threshold},
    )


def _coerce_fill_value(series: pd.Series, value: Any) -> Any:
    """Convert a literal fill value to a type compatible with ``series``."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        with_missing = series.isna()
        try:
            numeric = pd.to_numeric(pd.Series([value]), errors="raise").iloc[0]
        except (ValueError, TypeError):
            if with_missing.any():
                raise CleaningError(
                    f"Value {value!r} is not numeric and cannot fill a numeric column."
                ) from None
            return value
        return numeric
    return value


def fill_missing_with_value(
    dataframe: pd.DataFrame, value: Any, columns: list[str] | None = None
) -> CleaningResult:
    """Replace missing values with a fixed value.

    Numeric columns receive a numeric value when ``value`` can be parsed as a
    number; otherwise the literal value is used.

    Args:
        dataframe: Frame to clean.
        value: Replacement value.
        columns: Restrict the replacement to these columns.

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If ``value`` is ``None``, ``columns`` is empty/unknown,
            or the value is incompatible with a numeric column.
    """
    if value is None:
        raise CleaningError("A fill value is required.")

    target_columns = (
        list(map(str, dataframe.columns))
        if columns is None
        else _require_columns(dataframe, columns)
    )
    if not target_columns:
        raise CleaningError("Select at least one column to fill.")

    affected_cells = _missing_counts(dataframe, target_columns)
    working = dataframe.copy(deep=True)
    for column in target_columns:
        if dataframe[column].isna().any():
            fill_value = _coerce_fill_value(dataframe[column], value)
            working[column] = working[column].fillna(fill_value)

    return CleaningResult(
        dataframe=working,
        affected_rows=0,
        affected_cells=affected_cells,
        description=(
            f"Filled {affected_cells} missing cell(s) with {value!r} "
            f"in {len(target_columns)} column(s)."
        ),
        details={"value": value, "columns": target_columns},
    )


def fill_numeric_missing(
    dataframe: pd.DataFrame, strategy: str = "mean", columns: list[str] | None = None
) -> CleaningResult:
    """Fill missing numeric values with the column mean or median.

    Args:
        dataframe: Frame to clean.
        strategy: Either ``"mean"`` or ``"median"``.
        columns: Restrict to these numeric columns.

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If the strategy is unknown or a requested column is not
            numeric.
    """
    if strategy not in _FILL_STRATEGIES:
        raise CleaningError(f"Unsupported strategy {strategy!r}. Use one of {_FILL_STRATEGIES}.")
    numeric = list(dataframe.select_dtypes(include="number").columns)
    if columns is None:
        target_columns = list(map(str, numeric))
    else:
        requested = _require_columns(dataframe, columns)
        non_numeric = [column for column in requested if column not in numeric]
        if non_numeric:
            raise CleaningError("These columns are not numeric: " + ", ".join(non_numeric))
        target_columns = requested

    working = dataframe.copy(deep=True)
    skipped: list[str] = []
    filled_cells = 0
    for column in target_columns:
        series = dataframe[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        statistic = series.mean() if strategy == "mean" else series.median()
        if pd.isna(statistic):
            skipped.append(column)
            continue
        working[column] = working[column].fillna(statistic)
        filled_cells += missing

    return CleaningResult(
        dataframe=working,
        affected_rows=0,
        affected_cells=filled_cells,
        description=(
            f"Filled {filled_cells} numeric missing value(s) using the "
            f"{strategy} in {len(target_columns)} column(s)."
        ),
        details={"strategy": strategy, "columns": target_columns, "skipped": skipped},
    )


def fill_categorical_missing(
    dataframe: pd.DataFrame, columns: list[str] | None = None
) -> CleaningResult:
    """Fill missing categorical values with the most frequent value.

    Args:
        dataframe: Frame to clean.
        columns: Restrict to these non-numeric columns.

    Returns:
        A :class:`CleaningResult`.

    Raises:
        CleaningError: If a requested column is numeric.
    """
    non_numeric = list(dataframe.select_dtypes(exclude="number").columns)
    if columns is None:
        target_columns = list(map(str, non_numeric))
    else:
        requested = _require_columns(dataframe, columns)
        numeric = [column for column in requested if column not in non_numeric]
        if numeric:
            raise CleaningError(
                "These columns are numeric and cannot be filled categorically: "
                + ", ".join(numeric)
            )
        target_columns = requested

    working = dataframe.copy(deep=True)
    skipped: list[str] = []
    filled_cells = 0
    for column in target_columns:
        series = dataframe[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        modes = series.mode(dropna=True)
        if modes.empty:
            skipped.append(column)
            continue
        working[column] = working[column].fillna(modes.iloc[0])
        filled_cells += missing

    return CleaningResult(
        dataframe=working,
        affected_rows=0,
        affected_cells=filled_cells,
        description=(
            f"Filled {filled_cells} categorical missing value(s) with the most "
            f"frequent value in {len(target_columns)} column(s)."
        ),
        details={"columns": target_columns, "skipped": skipped},
    )
