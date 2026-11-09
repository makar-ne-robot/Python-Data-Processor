"""Tests for :mod:`data_processor.analyzer`."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processor.analyzer import (
    analyze_dataframe,
    categorical_columns,
    detect_datetime_columns,
    duplicate_rows,
    missing_summary,
    numeric_columns,
)


def test_analyze_counts(mixed_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(mixed_frame)
    assert result.rows == 5
    assert result.columns == 7
    assert result.duplicate_rows == 1
    assert result.total_cells == 35
    assert result.total_missing == int(mixed_frame.isna().sum().sum())


def test_analyze_missing_pct(mixed_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(mixed_frame)
    assert 0 < result.missing_pct < 100
    assert result.missing_pct == pytest.approx(result.total_missing / 35 * 100, abs=0.01)


def test_analyze_column_profiles(mixed_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(mixed_frame)
    profiles = {profile.name: profile for profile in result.column_profiles}
    assert profiles["age"].missing == 2
    assert profiles["age"].non_null == 3
    assert profiles["age"].missing_pct == pytest.approx(40.0)
    assert profiles["id"].unique == 4


def test_analyze_numeric_statistics(mixed_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(mixed_frame)
    assert "amount" in result.numeric_stats
    stats = result.numeric_stats["amount"]
    assert stats["count"] == 4
    assert stats["max"] == 200.0
    assert result.numeric_columns == ("id", "age", "amount")


def test_missing_summary_sorted(mixed_frame: pd.DataFrame) -> None:
    summary = missing_summary(mixed_frame)
    assert list(summary.columns) == [
        "column",
        "dtype",
        "missing",
        "missing_pct",
        "unique",
    ]
    missing_values = summary["missing_pct"].tolist()
    assert missing_values == sorted(missing_values, reverse=True)


def test_duplicate_rows_helper(mixed_frame: pd.DataFrame) -> None:
    duplicates = duplicate_rows(mixed_frame)
    assert len(duplicates) == 1
    assert duplicates.iloc[0]["name"] == "Boris"


def test_duplicate_rows_with_subset(mixed_frame: pd.DataFrame) -> None:
    duplicates = duplicate_rows(mixed_frame, subset=["id"])
    assert len(duplicates) == 1


def test_empty_frame_analyzed(empty_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(empty_frame)
    assert result.rows == 0
    assert result.columns == 0
    assert result.total_missing == 0
    assert result.missing_pct == 0.0
    assert result.column_profiles == ()


def test_empty_with_columns_analyzed(empty_with_columns: pd.DataFrame) -> None:
    result = analyze_dataframe(empty_with_columns)
    assert result.rows == 0
    assert result.columns == 2
    assert all(profile.missing_pct == 0.0 for profile in result.column_profiles)


def test_single_column_analyzed(single_column_frame: pd.DataFrame) -> None:
    result = analyze_dataframe(single_column_frame)
    assert result.columns == 1
    assert result.numeric_columns == ("value",)


def test_column_kind_helpers(mixed_frame: pd.DataFrame) -> None:
    assert set(numeric_columns(mixed_frame)) == {"id", "age", "amount"}
    categorical = categorical_columns(mixed_frame)
    assert "name" in categorical
    assert "id" not in categorical


def test_detect_datetime_columns(mixed_frame: pd.DataFrame) -> None:
    detected = detect_datetime_columns(mixed_frame)
    assert "signup" in detected
    assert "name" not in detected


def test_detect_datetime_columns_invalid_threshold(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        detect_datetime_columns(mixed_frame, threshold=0.0)
