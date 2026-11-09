"""Filtering and sorting page."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data_processor import DataProcessorError, FilterCondition, SortSpec, apply_filters
from data_processor.analyzer import (
    categorical_columns,
    detect_datetime_columns,
    numeric_columns,
)
from data_processor.filters import describe_condition, describe_sort
from ui import state
from ui.components import empty_state, metrics, page_header, section

_NUMERIC_LABELS: dict[str, str] = {
    "between": "is between (inclusive)",
    ">=": "≥",
    "<=": "≤",
    ">": ">",
    "<": "<",
    "==": "=",
    "!=": "≠",
}
_TEXT_LABELS: dict[str, str] = {
    "contains": "contains",
    "not_contains": "does not contain",
    "equals": "equals",
    "not_equals": "does not equal",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "is_empty": "is empty",
    "is_not_empty": "is not empty",
}
_DATE_LABELS: dict[str, str] = {
    "date_between": "is between (inclusive)",
    "before": "is before",
    "after": "is after",
    "on": "is on",
}


def render() -> None:
    """Render the filter and sort workspace."""
    page_header(
        "Filtering & Sorting",
        "Combine conditions with AND, add sort keys and preview the result immediately.",
    )

    if not state.has_dataset():
        empty_state("Upload a CSV file on the **Upload** page to filter or sort it.")
        return

    working = state.get_working()
    if working is None:
        empty_state("No dataset is loaded.")
        return

    _render_filter_builder(working)
    st.divider()
    _render_active_filters()
    st.divider()
    _render_sort_builder(working)

    st.divider()
    result = state.get_result_dataframe()
    if result is None:
        return
    section("Result preview")
    metrics(
        [
            ("Rows in view", len(result), None),
            ("Total rows", len(working), None),
            ("Filtered out", len(working) - len(result), None),
            ("Active filters", len(state.get_filters()), None),
        ]
    )
    st.dataframe(result.head(200), width="stretch")
    if len(result) > 200:
        st.caption(f"Showing the first 200 of {len(result):,} rows.")


def _render_filter_builder(working: pd.DataFrame) -> None:
    section("Add a filter condition")
    filter_type = st.radio(
        "Condition type",
        ["Numeric", "Text", "Date"],
        horizontal=True,
        key="filter::type",
    )

    condition: FilterCondition | None = None
    if filter_type == "Numeric":
        condition = _numeric_controls(working)
    elif filter_type == "Text":
        condition = _text_controls(working)
    else:
        condition = _date_controls(working)

    if condition is not None:
        add_column, note_column = st.columns([1, 3])
        if add_column.button("Add condition", type="primary", key="filter::add"):
            try:
                apply_filters(working, [condition])
            except DataProcessorError as error:
                state.add_error(str(error))
                st.error(str(error), icon="⚠️")
            else:
                state.add_filter(condition)
                st.toast(f"Added filter: {describe_condition(condition)}", icon="✅")
                st.rerun()
        note_column.caption("All active conditions are combined with **AND**.")


def _numeric_controls(working: pd.DataFrame) -> FilterCondition | None:
    columns = numeric_columns(working)
    if not columns:
        st.info("No numeric columns are available for filtering.")
        return None

    selected = st.selectbox("Numeric column", columns, key="filter::numeric::column")
    operator = st.selectbox(
        "Operator",
        list(_NUMERIC_LABELS),
        format_func=lambda value: _NUMERIC_LABELS[value],
        key="filter::numeric::operator",
    )
    series = pd.to_numeric(working[selected], errors="coerce").dropna()
    default_low = float(series.min()) if not series.empty else 0.0
    default_high = float(series.max()) if not series.empty else 100.0

    if operator == "between":
        left, right = st.columns(2)
        low = left.number_input("Minimum", value=default_low, key="filter::numeric::low")
        high = right.number_input("Maximum", value=default_high, key="filter::numeric::high")
        return FilterCondition(selected, operator, float(low), float(high))

    value = st.number_input("Value", value=default_low, key="filter::numeric::value")
    return FilterCondition(selected, operator, float(value))


def _text_controls(working: pd.DataFrame) -> FilterCondition | None:
    columns = list(categorical_columns(working)) or [str(column) for column in working.columns]
    selected = st.selectbox("Text column", columns, key="filter::text::column")
    operator = st.selectbox(
        "Operator",
        list(_TEXT_LABELS),
        format_func=lambda value: _TEXT_LABELS[value],
        key="filter::text::operator",
    )
    if operator in ("is_empty", "is_not_empty"):
        return FilterCondition(selected, operator)

    value = st.text_input("Search text", key="filter::text::value")
    case_sensitive = st.checkbox("Case sensitive", value=False, key="filter::text::case")
    return FilterCondition(selected, operator, value, case_sensitive=case_sensitive)


def _date_controls(working: pd.DataFrame) -> FilterCondition | None:
    detected = list(detect_datetime_columns(working))
    columns = detected or list(categorical_columns(working))
    if not columns:
        st.info("No columns are available for date filtering.")
        return None

    selected = st.selectbox("Date column", columns, key="filter::date::column")
    operator = st.selectbox(
        "Operator",
        list(_DATE_LABELS),
        format_func=lambda value: _DATE_LABELS[value],
        key="filter::date::operator",
    )
    today = date.today()
    if operator == "date_between":
        left, right = st.columns(2)
        start = left.date_input("Start date", value=today, key="filter::date::start")
        end = right.date_input("End date", value=today, key="filter::date::end")
        return FilterCondition(selected, operator, start, end)

    value = st.date_input("Date", value=today, key="filter::date::value")
    return FilterCondition(selected, operator, value)


def _render_active_filters() -> None:
    section("Active filters")
    conditions = state.get_filters()
    if not conditions:
        st.caption("No filters applied. The full cleaned dataset is shown.")
        return

    for index, condition in enumerate(conditions):
        label_column, remove_column = st.columns([5, 1])
        label_column.markdown(f"- {describe_condition(condition)}")
        if remove_column.button("Remove", key=f"filter::remove::{index}"):
            state.remove_filter(index)
            st.rerun()

    if st.button("Reset all filters", key="filter::reset"):
        state.clear_filters()
        st.rerun()


def _render_sort_builder(working: pd.DataFrame) -> None:
    section("Sorting")
    columns = [str(column) for column in working.columns]
    left, middle, right = st.columns([2, 1, 1])
    selected = left.selectbox("Sort column", columns, key="sort::column")
    direction = middle.selectbox("Direction", ["Ascending", "Descending"], key="sort::dir")
    add_clicked = right.button("Add sort key", key="sort::add", width="stretch")

    if add_clicked:
        if any(spec.column == selected for spec in state.get_sort_specs()):
            st.warning(f"Column {selected!r} is already used as a sort key.")
        else:
            state.add_sort_spec(SortSpec(selected, ascending=direction == "Ascending"))
            st.toast(f"Added sort key: {selected}", icon="✅")
            st.rerun()

    specs = state.get_sort_specs()
    if not specs:
        st.caption("No sorting applied.")
        return

    st.markdown(f"**Sort order:** {describe_sort(specs)}")
    for index, spec in enumerate(specs):
        label_column, remove_column = st.columns([5, 1])
        direction_label = "ascending" if spec.ascending else "descending"
        label_column.markdown(f"- {spec.column} ({direction_label})")
        if remove_column.button("Remove", key=f"sort::remove::{index}"):
            state.remove_sort_spec(index)
            st.rerun()
    if st.button("Clear sorting", key="sort::clear"):
        state.clear_sort_specs()
        st.rerun()
