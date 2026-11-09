"""Tests for :mod:`data_processor.filters`."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processor.exceptions import FilterError
from data_processor.filters import (
    apply_filters,
    describe_condition,
    describe_sort,
    sort_dataframe,
)
from data_processor.models import FilterCondition, SortSpec


def test_numeric_between(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("age", "between", 30, 40)])
    assert len(result) == 2
    assert set(result["age"]) == {30.0, 40.0}


def test_numeric_greater_than(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("age", ">", 30)])
    assert result["age"].tolist() == [40.0]


def test_numeric_equality(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("id", "==", 2)])
    assert len(result) == 2


def test_text_contains_case_insensitive(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("name", "contains", "boris")])
    assert len(result) == 2


def test_text_contains_case_sensitive(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(
        mixed_frame, [FilterCondition("name", "contains", "boris", case_sensitive=True)]
    )
    assert result.empty


def test_text_not_contains(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("name", "not_contains", "Boris")])
    assert len(result) == 3


def test_text_starts_and_ends_with(mixed_frame: pd.DataFrame) -> None:
    starts = apply_filters(mixed_frame, [FilterCondition("name", "starts_with", "an")])
    assert starts["name"].tolist() == ["Anna"]
    ends = apply_filters(mixed_frame, [FilterCondition("name", "ends_with", "na")])
    assert set(ends["name"]) == {"Anna", "Dana"}


def test_text_equals_and_not_equals(mixed_frame: pd.DataFrame) -> None:
    equals = apply_filters(mixed_frame, [FilterCondition("name", "equals", "Boris")])
    assert len(equals) == 2
    not_equals = apply_filters(mixed_frame, [FilterCondition("name", "not_equals", "Boris")])
    assert len(not_equals) == 3


def test_text_is_empty_and_not_empty() -> None:
    frame = pd.DataFrame({"text": ["a", "", None, "  "]})
    empty = apply_filters(frame, [FilterCondition("text", "is_empty")])
    assert len(empty) == 3
    not_empty = apply_filters(frame, [FilterCondition("text", "is_not_empty")])
    assert len(not_empty) == 1


def test_date_before(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("signup", "before", "2023-03-01")])
    assert len(result) == 3


def test_date_on(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("signup", "on", "2023-02-01")])
    assert len(result) == 2


def test_date_between(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(
        mixed_frame,
        [FilterCondition("signup", "date_between", "2023-01-15", "2023-03-15")],
    )
    assert len(result) == 3


def test_filters_combined_with_and(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(
        mixed_frame,
        [
            FilterCondition("amount", ">=", 200),
            FilterCondition("city", "contains", "mu"),
        ],
    )
    assert len(result) == 2


def test_empty_conditions_returns_copy(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [])
    pd.testing.assert_frame_equal(result, mixed_frame)
    assert result is not mixed_frame


def test_unknown_column_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(mixed_frame, [FilterCondition("ghost", "==", 1)])


def test_unknown_operator_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(mixed_frame, [FilterCondition("id", "~=", 1)])


def test_numeric_range_requires_second_value(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(mixed_frame, [FilterCondition("age", "between", 30)])


def test_invalid_numeric_value_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(mixed_frame, [FilterCondition("age", ">", "abc")])


def test_invalid_date_value_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        apply_filters(mixed_frame, [FilterCondition("signup", "after", "not-a-date")])


def test_sort_single_column(mixed_frame: pd.DataFrame) -> None:
    result = sort_dataframe(mixed_frame, [SortSpec("age", ascending=True)])
    ages = result["age"].tolist()
    non_null = [age for age in ages if pd.notna(age)]
    assert non_null == sorted(non_null)
    assert pd.isna(ages[-1])


def test_sort_descending(mixed_frame: pd.DataFrame) -> None:
    result = sort_dataframe(mixed_frame, [SortSpec("amount", ascending=False)])
    amounts = [amount for amount in result["amount"].tolist() if pd.notna(amount)]
    assert amounts == sorted(amounts, reverse=True)


def test_sort_multiple_columns(mixed_frame: pd.DataFrame) -> None:
    result = sort_dataframe(
        mixed_frame,
        [SortSpec("tier", ascending=True), SortSpec("id", ascending=False)],
    )
    assert result["tier"].tolist()[0] == "bronze"


def test_sort_unknown_column_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        sort_dataframe(mixed_frame, [SortSpec("ghost")])


def test_sort_invalid_na_position_raises(mixed_frame: pd.DataFrame) -> None:
    with pytest.raises(FilterError):
        sort_dataframe(mixed_frame, [SortSpec("age", na_position="middle")])


def test_sort_empty_specs_returns_copy(mixed_frame: pd.DataFrame) -> None:
    result = sort_dataframe(mixed_frame, [])
    pd.testing.assert_frame_equal(result, mixed_frame)
    assert result is not mixed_frame


def test_sort_does_not_mutate(mixed_frame: pd.DataFrame) -> None:
    original = mixed_frame.copy(deep=True)
    sort_dataframe(mixed_frame, [SortSpec("age", ascending=False)])
    pd.testing.assert_frame_equal(mixed_frame, original)


def test_describe_condition() -> None:
    assert describe_condition(FilterCondition("age", "between", 1, 2)) == ("age is between 1 and 2")
    assert "is empty" in describe_condition(FilterCondition("name", "is_empty"))


def test_describe_sort() -> None:
    assert describe_sort([SortSpec("a", True), SortSpec("b", False)]) == "a ASC, b DESC"


def test_numeric_not_equal(mixed_frame: pd.DataFrame) -> None:
    result = apply_filters(mixed_frame, [FilterCondition("id", "!=", 2)])
    assert len(result) == 3


def test_sort_na_position_first(mixed_frame: pd.DataFrame) -> None:
    result = sort_dataframe(mixed_frame, [SortSpec("age", ascending=True, na_position="first")])
    assert pd.isna(result["age"].iloc[0])


def test_describe_condition_date_and_text() -> None:
    date_text = describe_condition(
        FilterCondition("signup", "date_between", "2023-01-01", "2023-02-01")
    )
    assert "is between (date)" in date_text
    text = describe_condition(FilterCondition("name", "contains", "an"))
    assert "case-insensitive" in text
