"""Tests for :mod:`data_processor.cleaner`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_processor.cleaner import (
    columns_above_missing_threshold,
    count_missing,
    drop_columns_above_missing_threshold,
    drop_duplicates_by_columns,
    drop_full_duplicates,
    drop_rows_with_missing,
    fill_categorical_missing,
    fill_missing_with_value,
    fill_numeric_missing,
)
from data_processor.exceptions import CleaningError


def test_drop_full_duplicates(mixed_frame: pd.DataFrame) -> None:
    result = drop_full_duplicates(mixed_frame)
    assert result.affected_rows == 1
    assert len(result.dataframe) == 4
    assert "removed" in result.description.lower()


def test_drop_full_duplicates_does_not_mutate(mixed_frame: pd.DataFrame) -> None:
    original = mixed_frame.copy(deep=True)
    drop_full_duplicates(mixed_frame)
    pd.testing.assert_frame_equal(mixed_frame, original)


def test_drop_duplicates_by_columns(mixed_frame: pd.DataFrame) -> None:
    result = drop_duplicates_by_columns(mixed_frame, ["id"])
    assert result.affected_rows == 1
    assert result.details["columns"] == ["id"]


def test_drop_duplicates_by_columns_requires_columns(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        drop_duplicates_by_columns(mixed_frame, [])


def test_drop_duplicates_by_columns_unknown(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        drop_duplicates_by_columns(mixed_frame, ["nope"])


def test_drop_rows_with_missing_all(mixed_frame: pd.DataFrame) -> None:
    expected = len(mixed_frame) - len(mixed_frame.dropna())
    result = drop_rows_with_missing(mixed_frame)
    assert result.affected_rows == expected
    assert not result.dataframe.isna().any().any()


def test_drop_rows_with_missing_subset(mixed_frame: pd.DataFrame) -> None:
    result = drop_rows_with_missing(mixed_frame, ["name"])
    assert result.affected_rows == 1


def test_drop_rows_with_missing_unknown(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        drop_rows_with_missing(mixed_frame, ["ghost"])


def test_columns_above_missing_threshold(mixed_frame: pd.DataFrame) -> None:
    assert columns_above_missing_threshold(mixed_frame, 30) == ["age"]
    assert "amount" in columns_above_missing_threshold(mixed_frame, 10)


def test_columns_above_missing_threshold_invalid(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        columns_above_missing_threshold(mixed_frame, 150)


def test_drop_columns_above_missing_threshold() -> None:
    frame = pd.DataFrame({"keep": [1, 2, 3], "drop": [np.nan, np.nan, 1.0]})
    result = drop_columns_above_missing_threshold(frame, 50)
    assert result.details["dropped_columns"] == ["drop"]
    assert list(result.dataframe.columns) == ["keep"]


def test_fill_missing_with_value_numeric_coercion(mixed_frame: pd.DataFrame) -> None:
    result = fill_missing_with_value(mixed_frame, "0", ["amount"])
    assert result.affected_cells == 1
    assert not result.dataframe["amount"].isna().any()


def test_fill_missing_with_value_text(mixed_frame: pd.DataFrame) -> None:
    result = fill_missing_with_value(mixed_frame, "Unknown", ["name"])
    assert result.affected_cells == 1
    assert result.dataframe["name"].iloc[3] == "Unknown"


def test_fill_missing_with_value_all_columns(mixed_frame: pd.DataFrame) -> None:
    result = fill_missing_with_value(mixed_frame, "0")
    assert not result.dataframe.isna().any().any()


def test_fill_missing_with_value_incompatible(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        fill_missing_with_value(mixed_frame, "not-a-number", ["age"])


def test_fill_numeric_missing_mean() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    result = fill_numeric_missing(frame, "mean")
    assert result.affected_cells == 1
    assert result.dataframe["x"].iloc[1] == pytest.approx(2.0)


def test_fill_numeric_missing_median() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 100.0, np.nan]})
    result = fill_numeric_missing(frame, "median")
    assert result.dataframe["x"].iloc[3] == pytest.approx(2.0)


def test_fill_numeric_missing_invalid_strategy(numeric_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        fill_numeric_missing(numeric_frame, "mode")


def test_fill_numeric_missing_non_numeric(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        fill_numeric_missing(mixed_frame, "mean", ["name"])


def test_fill_numeric_missing_all_nan_skipped() -> None:
    frame = pd.DataFrame({"x": [np.nan, np.nan]})
    result = fill_numeric_missing(frame, "mean", ["x"])
    assert result.affected_cells == 0
    assert result.details["skipped"] == ["x"]


def test_fill_categorical_missing_mode(mixed_frame: pd.DataFrame) -> None:
    result = fill_categorical_missing(mixed_frame, ["city"])
    assert result.affected_cells == 1
    assert result.dataframe["city"].iloc[4] == "Munich"


def test_fill_categorical_missing_numeric_rejected(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError):
        fill_categorical_missing(mixed_frame, ["age"])


def test_count_missing(mixed_frame: pd.DataFrame) -> None:
    assert count_missing(mixed_frame) == int(mixed_frame.isna().sum().sum())
    assert count_missing(mixed_frame, ["name"]) == 1


def test_operations_on_empty_frame(empty_with_columns: pd.DataFrame) -> None:
    assert len(drop_full_duplicates(empty_with_columns).dataframe) == 0
    assert drop_rows_with_missing(empty_with_columns).affected_rows == 0
    assert fill_numeric_missing(empty_with_columns, "mean").affected_cells == 0
    assert fill_categorical_missing(empty_with_columns).affected_cells == 0


# --------------------------------------------------------------------------- #
# Threshold and fill-value validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "threshold", ["abc", None, float("nan"), float("inf"), float("-inf"), -1, 101]
)
def test_missing_threshold_rejects_invalid_values(
    mixed_frame: pd.DataFrame, threshold: object
) -> None:
    with pytest.raises(CleaningError):
        columns_above_missing_threshold(mixed_frame, threshold)


def test_missing_threshold_accepts_boundaries(mixed_frame: pd.DataFrame) -> None:
    assert set(columns_above_missing_threshold(mixed_frame, 0)) == {
        "name",
        "age",
        "city",
        "amount",
    }
    assert columns_above_missing_threshold(mixed_frame, 100) == []


def test_drop_columns_above_missing_threshold_rejects_invalid(
    mixed_frame: pd.DataFrame,
) -> None:
    with pytest.raises(CleaningError):
        drop_columns_above_missing_threshold(mixed_frame, "high")


def test_fill_missing_with_value_requires_value(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(CleaningError, match="fill value"):
        fill_missing_with_value(mixed_frame, None)
