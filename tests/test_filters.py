"""Tests for :mod:`data_processor.filters`."""

from __future__ import annotations

import numpy as np
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


# --------------------------------------------------------------------------- #
# Numeric "!=" must not include missing values
# --------------------------------------------------------------------------- #


def _numeric_frame() -> pd.DataFrame:
    return pd.DataFrame({"value": [10.0, 20.0, np.nan, 10.0, np.nan, 30.0]})


def test_numeric_not_equal_regular_values() -> None:
    result = apply_filters(_numeric_frame(), [FilterCondition("value", "!=", 10)])
    assert result["value"].tolist() == [20.0, 30.0]


def test_numeric_not_equal_excludes_nan() -> None:
    result = apply_filters(_numeric_frame(), [FilterCondition("value", "!=", 10)])
    assert not result["value"].isna().any()


def test_numeric_not_equal_excludes_multiple_nan() -> None:
    frame = pd.DataFrame({"value": [1.0, np.nan, 2.0, np.nan, 3.0]})
    result = apply_filters(frame, [FilterCondition("value", "!=", 1)])
    assert result["value"].tolist() == [2.0, 3.0]
    assert result["value"].isna().sum() == 0


def test_numeric_not_equal_empty_series() -> None:
    frame = pd.DataFrame({"value": pd.Series(dtype="float64")})
    result = apply_filters(frame, [FilterCondition("value", "!=", 10)])
    assert result.empty
    assert list(result.columns) == ["value"]


def test_numeric_not_equal_all_missing() -> None:
    frame = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
    result = apply_filters(frame, [FilterCondition("value", "!=", 10)])
    assert result.empty


def test_numeric_not_equal_coerces_non_numeric_values_to_missing() -> None:
    frame = pd.DataFrame({"value": [10, "20", None, "not-a-number"]})
    result = apply_filters(frame, [FilterCondition("value", "!=", 10)])
    assert result["value"].tolist() == ["20"]


@pytest.mark.parametrize(
    ("operator", "value", "second_value", "expected"),
    [
        (">=", 10, None, [10.0, 20.0, 10.0, 30.0]),
        ("<=", 10, None, [10.0, 10.0]),
        (">", 10, None, [20.0, 30.0]),
        ("<", 10, None, []),
        ("==", 10, None, [10.0, 10.0]),
        ("between", 10, 20, [10.0, 20.0, 10.0]),
    ],
)
def test_other_numeric_operators_still_exclude_nan(
    operator: str, value: float, second_value: float | None, expected: list[float]
) -> None:
    result = apply_filters(
        _numeric_frame(), [FilterCondition("value", operator, value, second_value)]
    )
    assert result["value"].tolist() == expected
    assert result["value"].isna().sum() == 0


def test_numeric_not_equal_does_not_mutate_input() -> None:
    frame = _numeric_frame()
    original = frame.copy(deep=True)
    apply_filters(frame, [FilterCondition("value", "!=", 10)])
    pd.testing.assert_frame_equal(frame, original)


# --------------------------------------------------------------------------- #
# Timezone-aware date filtering
# --------------------------------------------------------------------------- #


def _utc_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-02 00:00:00",
                    "2023-01-03 00:00:00",
                    None,
                ],
                utc=True,
            )
        }
    )


def test_date_on_utc_series_with_naive_date() -> None:
    result = apply_filters(_utc_frame(), [FilterCondition("when", "on", "2023-01-02")])
    assert len(result) == 1
    assert result["when"].iloc[0] == pd.Timestamp("2023-01-02", tz="UTC")
    assert isinstance(result["when"].dtype, pd.DatetimeTZDtype)
    assert str(result["when"].dt.tz) == "UTC"


def test_date_before_after_and_between_utc_series() -> None:
    before = apply_filters(_utc_frame(), [FilterCondition("when", "before", "2023-01-02")])
    assert before["when"].tolist() == [pd.Timestamp("2023-01-01", tz="UTC")]

    after = apply_filters(_utc_frame(), [FilterCondition("when", "after", "2023-01-02")])
    assert after["when"].tolist() == [pd.Timestamp("2023-01-03", tz="UTC")]

    between = apply_filters(
        _utc_frame(),
        [FilterCondition("when", "date_between", "2023-01-01", "2023-01-02")],
    )
    assert len(between) == 2


def test_date_filter_with_value_in_another_timezone() -> None:
    # 01:00 in Berlin on Jan 2 is 00:00 UTC on Jan 2.
    value = pd.Timestamp("2023-01-02 01:00", tz="Europe/Berlin")

    on_result = apply_filters(_utc_frame(), [FilterCondition("when", "on", value)])
    assert len(on_result) == 1
    assert on_result["when"].iloc[0] == pd.Timestamp("2023-01-02", tz="UTC")

    after_result = apply_filters(_utc_frame(), [FilterCondition("when", "after", value)])
    assert after_result["when"].tolist() == [pd.Timestamp("2023-01-03", tz="UTC")]


def test_date_filter_berlin_series_keeps_timezone() -> None:
    frame = pd.DataFrame({"when": _utc_frame()["when"].dt.tz_convert("Europe/Berlin")})
    result = apply_filters(frame, [FilterCondition("when", "on", "2023-01-02")])
    assert len(result) == 1
    assert str(result["when"].dt.tz) == "Europe/Berlin"
    assert result["when"].dt.strftime("%Y-%m-%d").iloc[0] == "2023-01-02"


def test_date_filter_excludes_nat_with_timezones() -> None:
    result = apply_filters(_utc_frame(), [FilterCondition("when", "after", "2022-12-31")])
    assert len(result) == 3
    assert result["when"].notna().all()


def test_date_filter_empty_timezone_aware_series() -> None:
    frame = pd.DataFrame({"when": pd.to_datetime([], utc=True)})
    result = apply_filters(frame, [FilterCondition("when", "on", "2023-01-01")])
    assert result.empty
    assert isinstance(result["when"].dtype, pd.DatetimeTZDtype)


def test_date_filter_rejects_aware_value_for_naive_column() -> None:
    frame = pd.DataFrame({"when": pd.to_datetime(["2023-01-01", "2023-01-02"])})
    with pytest.raises(FilterError, match="timezone-aware"):
        apply_filters(
            frame,
            [FilterCondition("when", "after", pd.Timestamp("2023-01-01", tz="UTC"))],
        )


def test_date_filter_mixed_offsets_are_compared_on_utc_instants() -> None:
    frame = pd.DataFrame(
        {
            "when": [
                "2023-01-01T00:00:00+02:00",
                "2023-01-02T00:00:00+00:00",
                None,
            ]
        }
    )
    after = apply_filters(
        frame,
        [FilterCondition("when", "after", "2023-01-01T00:00:00+00:00")],
    )
    assert after["when"].tolist() == ["2023-01-02T00:00:00+00:00"]

    on_result = apply_filters(frame, [FilterCondition("when", "on", "2023-01-02")])
    assert on_result["when"].tolist() == ["2023-01-02T00:00:00+00:00"]


def test_timezone_filter_does_not_mutate_input() -> None:
    frame = _utc_frame()
    original = frame.copy(deep=True)
    apply_filters(frame, [FilterCondition("when", "on", "2023-01-02")])
    pd.testing.assert_frame_equal(frame, original)


def test_date_filter_rejects_nonexistent_local_time() -> None:
    # Some zones skip local midnight on a DST transition; this must surface as
    # a FilterError rather than a raw ValueError.
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(["2018-11-04 03:00", "2018-12-01 12:00"], utc=True).tz_convert(
                "America/Sao_Paulo"
            )
        }
    )
    with pytest.raises(FilterError, match="timezone"):
        apply_filters(frame, [FilterCondition("when", "date_between", "2018-11-04", "2018-11-04")])


# --------------------------------------------------------------------------- #
# Text equals / not_equals must be literal, never regular expressions
# --------------------------------------------------------------------------- #

_REGEX_METACHARACTERS = [".", "[", "]", "*", "+", "?", "\\", "(", ")", "^", "$", "|"]


def _literal_frame() -> pd.DataFrame:
    return pd.DataFrame({"text": ["a.b", "axb", "a[b", "a*b", "A.B", None]})


def test_text_equals_is_literal_not_regex() -> None:
    result = apply_filters(_literal_frame(), [FilterCondition("text", "equals", "a.b")])
    assert set(result["text"]) == {"a.b", "A.B"}


def test_text_equals_case_insensitive() -> None:
    result = apply_filters(
        _literal_frame(), [FilterCondition("text", "equals", "A.B", case_sensitive=False)]
    )
    assert set(result["text"]) == {"a.b", "A.B"}


def test_text_equals_case_sensitive() -> None:
    result = apply_filters(
        _literal_frame(), [FilterCondition("text", "equals", "A.B", case_sensitive=True)]
    )
    assert result["text"].tolist() == ["A.B"]


def test_text_equals_excludes_missing_values() -> None:
    result = apply_filters(_literal_frame(), [FilterCondition("text", "equals", "missing")])
    assert result.empty


@pytest.mark.parametrize("needle", _REGEX_METACHARACTERS)
def test_text_equals_handles_regex_metacharacters(needle: str) -> None:
    frame = pd.DataFrame({"text": [needle, "other", None]})
    result = apply_filters(frame, [FilterCondition("text", "equals", needle)])
    assert result["text"].tolist() == [needle]


def test_text_equals_does_not_match_partial_regex() -> None:
    # The old fullmatch-based implementation treated "." as "any character".
    frame = pd.DataFrame({"text": ["a", ".", "z", None]})
    result = apply_filters(frame, [FilterCondition("text", "equals", ".")])
    assert result["text"].tolist() == ["."]


def test_text_not_equals_is_literal_and_includes_missing_values() -> None:
    result = apply_filters(_literal_frame(), [FilterCondition("text", "not_equals", "a.b")])
    assert set(result["text"].dropna()) == {"axb", "a[b", "a*b"}
    # A missing value is treated as "not equal" by the negative operator.
    assert result["text"].isna().sum() == 1


def test_text_not_equals_case_sensitive() -> None:
    result = apply_filters(
        _literal_frame(), [FilterCondition("text", "not_equals", "A.B", case_sensitive=True)]
    )
    assert "A.B" not in set(result["text"].dropna())
    assert "a.b" in set(result["text"].dropna())


@pytest.mark.parametrize("needle", _REGEX_METACHARACTERS)
def test_text_not_equals_handles_regex_metacharacters(needle: str) -> None:
    frame = pd.DataFrame({"text": [needle, "other"]})
    result = apply_filters(frame, [FilterCondition("text", "not_equals", needle)])
    assert result["text"].tolist() == ["other"]


def test_text_contains_with_metacharacter_is_literal() -> None:
    frame = pd.DataFrame({"text": ["a[b", "axb", None]})
    result = apply_filters(frame, [FilterCondition("text", "contains", "[")])
    assert result["text"].tolist() == ["a[b"]


def test_text_equals_does_not_mutate_input() -> None:
    frame = _literal_frame()
    original = frame.copy(deep=True)
    apply_filters(frame, [FilterCondition("text", "equals", "a.b")])
    pd.testing.assert_frame_equal(frame, original)


# --------------------------------------------------------------------------- #
# date_between must include the entire end day
# --------------------------------------------------------------------------- #


def _datetime_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "when": pd.to_datetime(
                [
                    "2023-01-01 23:59:59",
                    "2023-01-02 00:00:00",
                    "2023-01-02 12:30:00",
                    "2023-01-02 23:59:59",
                    "2023-01-03 00:00:00",
                    None,
                ]
            )
        }
    )


def test_date_between_includes_entire_end_day() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")],
    )
    assert result["when"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2023-01-02 00:00:00",
        "2023-01-02 12:30:00",
        "2023-01-02 23:59:59",
    ]


def test_date_between_includes_start_of_end_day() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")],
    )
    assert pd.Timestamp("2023-01-02 00:00:00") in set(result["when"])


def test_date_between_excludes_values_outside_range() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")],
    )
    assert pd.Timestamp("2023-01-03 00:00:00") not in set(result["when"])
    assert pd.Timestamp("2023-01-01 23:59:59") not in set(result["when"])


def test_date_between_excludes_missing_values() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")],
    )
    assert result["when"].notna().all()


def test_date_between_multi_day_range() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-01", "2023-01-02")],
    )
    assert len(result) == 4


def test_date_between_single_start_day() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-01", "2023-01-01")],
    )
    assert result["when"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == ["2023-01-01 23:59:59"]


def test_date_between_timezone_aware_includes_end_day() -> None:
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(
                [
                    "2023-01-02 00:00:00",
                    "2023-01-02 18:00:00",
                    "2023-01-03 00:00:00",
                ],
                utc=True,
            )
        }
    )
    result = apply_filters(
        frame, [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")]
    )
    assert len(result) == 2


def test_date_between_berlin_series_includes_end_day() -> None:
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(
                [
                    "2023-01-02 00:00:00",
                    "2023-01-02 18:00:00",
                    "2023-01-03 00:00:00",
                ],
                utc=True,
            ).tz_convert("Europe/Berlin")
        }
    )
    result = apply_filters(
        frame, [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")]
    )
    assert len(result) == 2
    assert str(result["when"].dt.tz) == "Europe/Berlin"


def test_date_between_empty_series() -> None:
    frame = pd.DataFrame({"when": pd.to_datetime([])})
    result = apply_filters(
        frame, [FilterCondition("when", "date_between", "2023-01-01", "2023-01-02")]
    )
    assert result.empty


def test_date_between_mixed_offsets_includes_end_day() -> None:
    frame = pd.DataFrame(
        {
            "when": [
                "2023-01-02T10:00:00+02:00",
                "2023-01-02T23:00:00+00:00",
                "2023-01-03T00:00:00+00:00",
            ]
        }
    )
    result = apply_filters(
        frame, [FilterCondition("when", "date_between", "2023-01-02", "2023-01-02")]
    )
    assert result["when"].tolist() == [
        "2023-01-02T10:00:00+02:00",
        "2023-01-02T23:00:00+00:00",
    ]


def test_date_between_handles_extreme_end_timestamp() -> None:
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-01", pd.Timestamp.max)],
    )
    assert len(result) == 5


def test_date_between_handles_midnight_end_near_max() -> None:
    end = pd.Timestamp.max.normalize()
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-01", end)],
    )
    assert len(result) == 5


def test_date_between_respects_explicit_end_time() -> None:
    # An end bound with a time is not extended to the end of the day.
    result = apply_filters(
        _datetime_frame(),
        [FilterCondition("when", "date_between", "2023-01-02 00:00", "2023-01-02 06:00")],
    )
    assert result["when"].dt.strftime("%H:%M:%S").tolist() == ["00:00:00"]


# --------------------------------------------------------------------------- #
# sort_dataframe na_position contract
# --------------------------------------------------------------------------- #


def _na_position_frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 1, 2], "b": [np.nan, 5.0, 1.0]})


def test_sort_multiple_keys_same_na_position_first() -> None:
    result = sort_dataframe(
        _na_position_frame(),
        [SortSpec("a", na_position="first"), SortSpec("b", na_position="first")],
    )
    # Within the a == 1 group, the NaN in the second key comes first.
    assert pd.isna(result["b"].iloc[0])
    assert result["b"].iloc[1] == 5.0


def test_sort_multiple_keys_same_na_position_last() -> None:
    result = sort_dataframe(
        _na_position_frame(),
        [SortSpec("a", na_position="last"), SortSpec("b", na_position="last")],
    )
    # Within the a == 1 group, the NaN in the second key comes last.
    assert result["b"].iloc[0] == 5.0
    assert pd.isna(result["b"].iloc[1])


def test_sort_multiple_keys_conflicting_na_position_raises() -> None:
    with pytest.raises(FilterError, match="na_position"):
        sort_dataframe(
            _na_position_frame(),
            [SortSpec("a", na_position="first"), SortSpec("b", na_position="last")],
        )


def test_sort_mixed_type_column_raises_filter_error() -> None:
    frame = pd.DataFrame({"x": ["a", 1, None]})
    with pytest.raises(FilterError, match="could not be sorted"):
        sort_dataframe(frame, [SortSpec("x")])


# --------------------------------------------------------------------------- #
# Numeric filters must reject NaN and infinities
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"])
def test_numeric_filter_rejects_non_finite_values(value: object) -> None:
    with pytest.raises(FilterError, match="finite"):
        apply_filters(_numeric_frame(), [FilterCondition("value", ">", value)])


def test_numeric_between_rejects_non_finite_bounds() -> None:
    with pytest.raises(FilterError, match="finite"):
        apply_filters(_numeric_frame(), [FilterCondition("value", "between", 0, float("inf"))])
    with pytest.raises(FilterError, match="finite"):
        apply_filters(_numeric_frame(), [FilterCondition("value", "between", float("nan"), 10)])


def test_numeric_filter_handles_infinite_data_values() -> None:
    frame = pd.DataFrame({"value": [1.0, np.inf, -np.inf, np.nan]})
    result = apply_filters(frame, [FilterCondition("value", ">", 0)])
    assert result["value"].tolist() == [1.0, np.inf]
