"""Filtering and sorting logic.

Conditions are combined with a logical AND.  The module is intentionally
independent from Streamlit so the exact same code paths are covered by tests.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import pandas as pd

from data_processor.exceptions import FilterError
from data_processor.models import FilterCondition, SortSpec

NUMERIC_OPERATORS: tuple[str, ...] = ("between", ">=", "<=", ">", "<", "==", "!=")
TEXT_OPERATORS: tuple[str, ...] = (
    "contains",
    "not_contains",
    "equals",
    "not_equals",
    "starts_with",
    "ends_with",
    "is_empty",
    "is_not_empty",
)
DATE_OPERATORS: tuple[str, ...] = ("date_between", "before", "after", "on")

_OPERATOR_LABELS: dict[str, str] = {
    "between": "is between",
    ">=": ">=",
    "<=": "<=",
    ">": ">",
    "<": "<",
    "==": "==",
    "!=": "!=",
    "contains": "contains",
    "not_contains": "does not contain",
    "equals": "equals",
    "not_equals": "does not equal",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "is_empty": "is empty",
    "is_not_empty": "is not empty",
    "date_between": "is between (date)",
    "before": "is before",
    "after": "is after",
    "on": "is on",
}


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise FilterError(f"Unknown column: {column!r}")


def _to_float(value: Any, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{label} must be a number, got {value!r}.") from exc


def _to_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a column to datetimes, handling timezone-aware values.

    Mixed timezone offsets cannot be represented by a single tz-aware dtype;
    pandas raises ``ValueError`` in that case, so the values are converted to a
    common UTC timeline instead of crashing.  Timezones are never silently
    discarded from an otherwise uniform tz-aware column.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
        else:
            try:
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            except ValueError:
                try:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
                except (ValueError, TypeError) as exc:
                    raise FilterError(
                        "The date column contains values with incompatible time zones."
                    ) from exc

    if not pd.api.types.is_datetime64_any_dtype(parsed):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(parsed, errors="coerce", utc=True)
        except (ValueError, TypeError) as exc:
            raise FilterError("The date column could not be interpreted as dates.") from exc

    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise FilterError("The date column could not be interpreted as dates.")
    return parsed


def _series_timezone(series: pd.Series) -> Any:
    """Return the timezone of a datetime series, or ``None`` when naive."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz
    return None


def _normalize_timestamp(timestamp: pd.Timestamp, series_tz: Any) -> pd.Timestamp:
    """Make a filter timestamp comparable with a datetime series.

    Naive values (for example from ``st.date_input``) are interpreted in the
    series' timezone.  Aware values are converted to it.  Comparing an aware
    value with a naive column is rejected explicitly instead of silently
    dropping or misinterpreting the timezone.
    """
    if series_tz is None:
        if timestamp.tzinfo is not None:
            raise FilterError(
                "Cannot compare a timezone-aware date with a timezone-naive "
                "column. Use a timezone-naive value or normalize the column."
            )
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(series_tz)
    return timestamp.tz_convert(series_tz)


def _to_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{label} must be a valid date, got {value!r}.") from exc
    if pd.isna(timestamp):
        raise FilterError(f"{label} must be a valid date.")
    return timestamp


def _numeric_mask(dataframe: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    series = pd.to_numeric(dataframe[condition.column], errors="coerce")
    operator = condition.operator
    if operator == "between":
        if condition.second_value is None:
            raise FilterError("A numeric range needs both a lower and an upper bound.")
        lower = _to_float(condition.value, label="Lower bound")
        upper = _to_float(condition.second_value, label="Upper bound")
        if lower > upper:
            lower, upper = upper, lower
        return series.between(lower, upper, inclusive="both").fillna(False)

    if operator not in NUMERIC_OPERATORS:
        raise FilterError(f"Unsupported numeric operator: {operator!r}")
    value = _to_float(condition.value, label="Filter value")
    masks: dict[str, pd.Series] = {
        ">=": series >= value,
        "<=": series <= value,
        ">": series > value,
        "<": series < value,
        "==": series == value,
        # NaN != value evaluates to True in pandas, so explicitly exclude
        # missing values from a "not equal" filter.
        "!=": series.notna() & (series != value),
    }
    return masks[operator].fillna(False).astype(bool)


def _empty_mask(dataframe: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    series = dataframe[condition.column]
    as_text = series.astype("string").str.strip()
    empty = series.isna() | (as_text == "")
    if condition.operator == "is_empty":
        return empty.fillna(True).astype(bool)
    return (~empty).fillna(False).astype(bool)


def _text_mask(dataframe: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    operator = condition.operator
    if operator in ("is_empty", "is_not_empty"):
        return _empty_mask(dataframe, condition)

    if operator not in TEXT_OPERATORS:
        raise FilterError(f"Unsupported text operator: {operator!r}")
    if condition.value is None:
        raise FilterError("A text filter requires a value.")

    series = dataframe[condition.column].astype("string")
    needle = str(condition.value)
    case = condition.case_sensitive

    if operator == "contains":
        mask = series.str.contains(needle, case=case, regex=False, na=False)
    elif operator == "not_contains":
        mask = ~series.str.contains(needle, case=case, regex=False, na=False)
        mask = mask.fillna(True)
    elif operator == "equals":
        mask = series.str.fullmatch(needle, case=case, na=False)
    elif operator == "not_equals":
        mask = ~series.str.fullmatch(needle, case=case, na=False)
        mask = mask.fillna(True)
    elif operator == "starts_with":
        mask = series.str.startswith(needle, na=False)
        if not case:
            mask = series.str.lower().str.startswith(needle.lower(), na=False)
    elif operator == "ends_with":
        mask = series.str.endswith(needle, na=False)
        if not case:
            mask = series.str.lower().str.endswith(needle.lower(), na=False)
    else:  # pragma: no cover - guarded by the operator check above
        raise FilterError(f"Unsupported text operator: {operator!r}")
    return mask.fillna(False).astype(bool)


def _date_mask(dataframe: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    operator = condition.operator
    if operator not in DATE_OPERATORS:
        raise FilterError(f"Unsupported date operator: {operator!r}")
    series = _to_datetime_series(dataframe[condition.column])
    series_tz = _series_timezone(series)
    if operator == "date_between":
        if condition.second_value is None:
            raise FilterError("A date range needs both a start and an end date.")
        start = _normalize_timestamp(_to_timestamp(condition.value, label="Start date"), series_tz)
        end = _normalize_timestamp(
            _to_timestamp(condition.second_value, label="End date"), series_tz
        )
        if start > end:
            start, end = end, start
        return series.between(start, end, inclusive="both").fillna(False)

    timestamp = _normalize_timestamp(_to_timestamp(condition.value, label="Filter date"), series_tz)
    if operator == "before":
        mask = series < timestamp
    elif operator == "after":
        mask = series > timestamp
    else:  # "on"
        mask = series.dt.normalize() == timestamp.normalize()
    return mask.fillna(False).astype(bool)


def apply_filters(dataframe: pd.DataFrame, conditions: list[FilterCondition]) -> pd.DataFrame:
    """Apply all conditions to ``dataframe`` using a logical AND.

    Args:
        dataframe: Frame to filter.
        conditions: Conditions to combine.

    Returns:
        A new filtered frame.  The input is not modified.

    Raises:
        FilterError: If a condition references an unknown column or contains an
            invalid operator/value.
    """
    if not conditions:
        return dataframe.copy(deep=True)

    mask = pd.Series(True, index=dataframe.index)
    for condition in conditions:
        _require_column(dataframe, condition.column)
        if condition.operator in NUMERIC_OPERATORS:
            condition_mask = _numeric_mask(dataframe, condition)
        elif condition.operator in TEXT_OPERATORS:
            condition_mask = _text_mask(dataframe, condition)
        elif condition.operator in DATE_OPERATORS:
            condition_mask = _date_mask(dataframe, condition)
        else:
            raise FilterError(f"Unknown filter operator: {condition.operator!r}")
        mask &= condition_mask.reindex(dataframe.index, fill_value=False)

    return dataframe.loc[mask].copy(deep=True)


def sort_dataframe(dataframe: pd.DataFrame, specs: list[SortSpec]) -> pd.DataFrame:
    """Sort a frame by one or more keys.

    Args:
        dataframe: Frame to sort.
        specs: Sort keys in priority order.

    Returns:
        A new sorted frame.  The input is not modified.

    Raises:
        FilterError: If a column is unknown or ``na_position`` is invalid.
    """
    if not specs:
        return dataframe.copy(deep=True)

    for spec in specs:
        _require_column(dataframe, spec.column)
        if spec.na_position not in ("first", "last"):
            raise FilterError("na_position must be 'first' or 'last'.")

    na_position: Literal["first", "last"] = "first" if specs[0].na_position == "first" else "last"
    return dataframe.sort_values(
        by=[spec.column for spec in specs],
        ascending=[spec.ascending for spec in specs],
        na_position=na_position,
        kind="mergesort",
    ).copy(deep=True)


def describe_condition(condition: FilterCondition) -> str:
    """Return a human-readable description of a filter condition."""
    label = _OPERATOR_LABELS.get(condition.operator, condition.operator)
    column = condition.column
    if condition.operator == "is_empty":
        return f"{column} {label}"
    if condition.operator == "is_not_empty":
        return f"{column} {label}"
    if condition.operator in ("between", "date_between"):
        return f"{column} {label} {condition.value!r} and {condition.second_value!r}"
    if condition.operator in TEXT_OPERATORS:
        case_note = "" if condition.case_sensitive else " (case-insensitive)"
        return f"{column} {label} {condition.value!r}{case_note}"
    return f"{column} {label} {condition.value!r}"


def describe_sort(specs: list[SortSpec]) -> str:
    """Return a human-readable description of the sorting configuration."""
    parts = [f"{spec.column} {'ASC' if spec.ascending else 'DESC'}" for spec in specs]
    return ", ".join(parts)
