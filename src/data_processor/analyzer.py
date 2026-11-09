"""DataFrame analysis utilities.

The functions in this module are read-only with respect to their input frame.
They produce the summaries rendered on the dashboard and embedded in reports.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from data_processor.models import AnalysisResult, ColumnProfile


def missing_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column missing-value summary.

    Args:
        dataframe: Frame to inspect.

    Returns:
        A frame with ``column``, ``dtype``, ``missing``, ``missing_pct`` and
        ``unique`` columns sorted by missing percentage descending.
    """
    rows = len(dataframe)
    records: list[dict[str, Any]] = []
    for name in dataframe.columns:
        series = dataframe[name]
        missing = int(series.isna().sum())
        pct = round(missing / rows * 100.0, 2) if rows else 0.0
        records.append(
            {
                "column": str(name),
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": pct,
                "unique": int(series.nunique(dropna=True)),
            }
        )
    summary = pd.DataFrame(records, columns=["column", "dtype", "missing", "missing_pct", "unique"])
    if not summary.empty:
        summary = summary.sort_values(
            by=["missing_pct", "column"], ascending=[False, True], ignore_index=True
        )
    return summary


def _numeric_statistics(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric = dataframe.select_dtypes(include="number")
    statistics: dict[str, dict[str, float]] = {}
    for name in numeric.columns:
        description = numeric[name].describe()
        statistics[str(name)] = {str(key): float(value) for key, value in description.items()}
    return statistics


def analyze_dataframe(dataframe: pd.DataFrame) -> AnalysisResult:
    """Compute a high-level analysis of a DataFrame.

    The function is safe for empty frames and frames with a single column.

    Args:
        dataframe: Frame to analyze.

    Returns:
        An :class:`AnalysisResult` with row/column counts, duplicate and missing
        statistics, per-column profiles and numeric summary statistics.
    """
    rows = int(dataframe.shape[0])
    columns = int(dataframe.shape[1])
    total_cells = rows * columns
    total_missing = int(dataframe.isna().sum().sum())
    duplicate_rows = int(dataframe.duplicated().sum()) if rows else 0

    profiles: list[ColumnProfile] = []
    for name in dataframe.columns:
        series = dataframe[name]
        missing = int(series.isna().sum())
        profiles.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                non_null=rows - missing,
                missing=missing,
                missing_pct=round(missing / rows * 100.0, 2) if rows else 0.0,
                unique=int(series.nunique(dropna=True)),
            )
        )

    return AnalysisResult(
        rows=rows,
        columns=columns,
        duplicate_rows=duplicate_rows,
        total_missing=total_missing,
        total_cells=total_cells,
        numeric_columns=tuple(
            str(name) for name in dataframe.select_dtypes(include="number").columns
        ),
        column_profiles=tuple(profiles),
        numeric_stats=_numeric_statistics(dataframe),
    )


def duplicate_rows(dataframe: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Return rows that are duplicates according to ``subset``.

    The first occurrence is kept, matching :meth:`pandas.DataFrame.duplicated`.

    Args:
        dataframe: Frame to inspect.
        subset: Columns to consider.  ``None`` means all columns.

    Returns:
        A new frame containing the duplicate rows.
    """
    mask = dataframe.duplicated(subset=subset, keep="first")
    return dataframe.loc[mask].copy()


def numeric_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Return names of numeric columns as strings."""
    return tuple(str(name) for name in dataframe.select_dtypes(include="number").columns)


def categorical_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Return names of non-numeric columns as strings."""
    return tuple(str(name) for name in dataframe.select_dtypes(exclude="number").columns)


def detect_datetime_columns(
    dataframe: pd.DataFrame, *, threshold: float = 0.8, sample_size: int = 100
) -> tuple[str, ...]:
    """Best-effort detection of columns that contain date-like values.

    Only non-numeric columns are considered.  A column is date-like when at
    least ``threshold`` of its sampled non-null values can be parsed as dates.

    Args:
        dataframe: Frame to inspect.
        threshold: Required share of parsable values, between 0 and 1.
        sample_size: Maximum number of non-null values to inspect per column.

    Returns:
        Names of the detected date-like columns.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    detected: list[str] = []
    for name in dataframe.select_dtypes(exclude="number").columns:
        series = dataframe[name].dropna()
        if series.empty:
            continue
        sample = series.head(sample_size).astype(str)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            continue
        ratio = float(parsed.notna().mean())
        if ratio >= threshold:
            detected.append(str(name))
    return tuple(detected)
