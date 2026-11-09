"""Data cleaning page: preview and commit transformations safely."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from data_processor import (
    CleaningResult,
    DataProcessorError,
    analyze_dataframe,
    drop_columns_above_missing_threshold,
    drop_duplicates_by_columns,
    drop_full_duplicates,
    drop_rows_with_missing,
    fill_categorical_missing,
    fill_missing_with_value,
    fill_numeric_missing,
    missing_summary,
)
from data_processor.analyzer import categorical_columns, duplicate_rows, numeric_columns
from data_processor.cleaner import columns_above_missing_threshold
from ui import state
from ui.components import empty_state, metrics, page_header, section

_PREVIEW_KEY = "dpp_clean_preview"


def render() -> None:
    """Render the cleaning workspace."""
    page_header(
        "Data Cleaning",
        "Preview every change before applying it. The original file stays intact.",
    )

    if not state.has_dataset():
        empty_state("Upload a CSV file on the **Upload** page before cleaning.")
        return

    working = state.get_working()
    if working is None:
        empty_state("No dataset is loaded.")
        return

    analysis = analyze_dataframe(working)
    metrics(
        [
            ("Rows", analysis.rows, None),
            ("Columns", analysis.columns, None),
            ("Missing cells", analysis.total_missing, None),
            ("Duplicate rows", analysis.duplicate_rows, None),
        ]
    )

    tabs = st.tabs(["Duplicates", "Missing values", "Columns"])
    with tabs[0]:
        _duplicates_tab(working)
    with tabs[1]:
        _missing_tab(working)
    with tabs[2]:
        _columns_tab(working)

    _render_history()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _store_preview(
    op_id: str, result: CleaningResult, *, category: str, name: str, destructive: bool
) -> None:
    st.session_state[_PREVIEW_KEY] = {
        "op": op_id,
        "version": state.data_version(),
        "result": result,
        "category": category,
        "name": name,
        "destructive": destructive,
    }


def _safe_build(builder: Callable[[], CleaningResult]) -> CleaningResult | None:
    try:
        return builder()
    except DataProcessorError as error:
        state.add_error(str(error))
        st.error(str(error), icon="⚠️")
    except Exception as error:  # last-resort guard against unexpected failures
        message = f"Unexpected error: {error}"
        state.add_error(message)
        st.error(message, icon="⚠️")
    return None


def _preview_if_active(op_id: str) -> None:
    preview = st.session_state.get(_PREVIEW_KEY)
    if not preview or preview["op"] != op_id or preview["version"] != state.data_version():
        return

    result: CleaningResult = preview["result"]
    before = state.get_working()
    rows_before = len(before) if before is not None else 0
    rows_after = len(result.dataframe)

    st.info(result.description, icon="🔍")
    metrics(
        [
            ("Rows before", rows_before, None),
            ("Rows after", rows_after, None),
            ("Affected rows", result.affected_rows, None),
            ("Affected cells", result.affected_cells, None),
        ]
    )

    dropped = result.details.get("dropped_columns")
    if dropped:
        st.caption("Columns to drop: " + ", ".join(map(str, dropped)))

    destructive = bool(preview["destructive"])
    confirmed = True
    if destructive:
        confirmed = st.checkbox(
            "I understand this permanently changes the working data",
            key=f"confirm::{op_id}",
        )

    apply_column, cancel_column = st.columns([1, 3])
    if apply_column.button(
        "Apply changes",
        key=f"apply::{op_id}",
        type="primary",
        disabled=not confirmed,
        width="stretch",
    ):
        state.commit_cleaning(result, category=preview["category"], name=preview["name"])
        st.session_state.pop(_PREVIEW_KEY, None)
        st.success(f"Applied: {result.description}", icon="✅")
        st.rerun()
    if cancel_column.button("Discard preview", key=f"cancel::{op_id}"):
        st.session_state.pop(_PREVIEW_KEY, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #


def _duplicates_tab(working: pd.DataFrame) -> None:
    section("Full duplicate rows")
    st.caption("Removes rows whose values are identical across every column.")
    duplicates = duplicate_rows(working)
    if duplicates.empty:
        st.caption("No full duplicate rows were detected.")
    else:
        st.caption(f"{len(duplicates)} duplicate row(s) detected.")
        with st.expander("Show duplicate rows"):
            st.dataframe(duplicates, width="stretch", hide_index=True)
    if st.button("Preview full-duplicate removal", key="preview::dupes::all"):
        result = _safe_build(lambda: drop_full_duplicates(working))
        if result is not None:
            _store_preview(
                "dupes::all",
                result,
                category="duplicates",
                name="Remove full duplicates",
                destructive=True,
            )
    _preview_if_active("dupes::all")

    st.divider()
    section("Duplicates by selected columns")
    columns = [str(column) for column in working.columns]
    selected = st.multiselect(
        "Columns that define a duplicate",
        columns,
        default=columns[: min(3, len(columns))],
        key="dupes::subset",
    )
    if st.button("Preview duplicate removal", key="preview::dupes::subset"):
        result = _safe_build(lambda: drop_duplicates_by_columns(working, selected))
        if result is not None:
            _store_preview(
                "dupes::subset",
                result,
                category="duplicates",
                name="Remove duplicates by columns",
                destructive=True,
            )
    _preview_if_active("dupes::subset")


def _missing_tab(working: pd.DataFrame) -> None:
    all_columns = [str(column) for column in working.columns]
    numeric = list(numeric_columns(working))
    categorical = list(categorical_columns(working))

    section("Drop rows with missing values")
    st.caption("Rows containing at least one missing value in the selected columns are removed.")
    drop_columns = st.multiselect(
        "Check these columns (empty = all columns)",
        all_columns,
        default=[],
        key="missing::drop_rows::cols",
    )
    if st.button("Preview row removal", key="preview::missing::drop_rows"):
        subset = drop_columns or None
        result = _safe_build(lambda: drop_rows_with_missing(working, subset))
        if result is not None:
            _store_preview(
                "missing::drop_rows",
                result,
                category="missing",
                name="Drop rows with missing values",
                destructive=True,
            )
    _preview_if_active("missing::drop_rows")

    st.divider()
    section("Fill missing values with a fixed value")
    fill_columns = st.multiselect(
        "Columns to fill (empty = all columns)",
        all_columns,
        default=[],
        key="missing::fill_value::cols",
    )
    fill_value = st.text_input(
        "Replacement value",
        value="0",
        key="missing::fill_value::value",
        help="Numeric columns receive a numeric value when possible.",
    )
    if st.button("Preview fill", key="preview::missing::fill_value"):
        subset = fill_columns or None
        result = _safe_build(lambda: fill_missing_with_value(working, fill_value, subset))
        if result is not None:
            _store_preview(
                "missing::fill_value",
                result,
                category="missing",
                name="Fill missing with value",
                destructive=False,
            )
    _preview_if_active("missing::fill_value")

    st.divider()
    section("Fill numeric missing values")
    if not numeric:
        st.info("No numeric columns are available.")
    else:
        strategy = st.radio(
            "Strategy",
            ["mean", "median"],
            horizontal=True,
            key="missing::numeric::strategy",
        )
        numeric_selection = st.multiselect(
            "Numeric columns (empty = all)",
            numeric,
            default=[],
            key="missing::numeric::cols",
        )
        if st.button("Preview numeric fill", key="preview::missing::numeric"):
            subset = numeric_selection or None
            result = _safe_build(lambda: fill_numeric_missing(working, strategy, subset))
            if result is not None:
                _store_preview(
                    "missing::numeric",
                    result,
                    category="missing",
                    name=f"Fill numeric missing ({strategy})",
                    destructive=False,
                )
        _preview_if_active("missing::numeric")

    st.divider()
    section("Fill categorical missing values")
    if not categorical:
        st.info("No categorical columns are available.")
    else:
        categorical_selection = st.multiselect(
            "Categorical columns (empty = all)",
            categorical,
            default=[],
            key="missing::categorical::cols",
        )
        if st.button("Preview mode fill", key="preview::missing::categorical"):
            subset = categorical_selection or None
            result = _safe_build(lambda: fill_categorical_missing(working, subset))
            if result is not None:
                _store_preview(
                    "missing::categorical",
                    result,
                    category="missing",
                    name="Fill categorical missing (mode)",
                    destructive=False,
                )
        _preview_if_active("missing::categorical")


def _columns_tab(working: pd.DataFrame) -> None:
    section("Column overview")
    st.dataframe(
        missing_summary(working),
        width="stretch",
        hide_index=True,
        height=min(420, 90 + 35 * max(len(working.columns), 1)),
    )

    st.divider()
    section("Drop columns with many missing values")
    threshold = st.slider(
        "Drop columns with more than this share of missing values (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        key="columns::threshold",
    )
    affected = columns_above_missing_threshold(working, float(threshold))
    if affected:
        st.warning(f"{len(affected)} column(s) would be dropped: " + ", ".join(affected))
    else:
        st.caption("No column exceeds the selected threshold.")

    if st.button("Preview column removal", key="preview::columns::threshold"):
        result = _safe_build(
            lambda: drop_columns_above_missing_threshold(working, float(threshold))
        )
        if result is not None:
            _store_preview(
                "columns::threshold",
                result,
                category="columns",
                name="Drop columns above missing threshold",
                destructive=True,
            )
    _preview_if_active("columns::threshold")


def _render_history() -> None:
    operations = state.get_operations()
    if not operations:
        return
    st.divider()
    section("Applied operations")
    rows = [
        {
            "#": index,
            "operation": record.name,
            "details": record.details,
            "rows": f"{record.rows_before} → {record.rows_after}",
            "affected rows": record.affected_rows,
            "affected cells": record.affected_cells,
        }
        for index, record in enumerate(operations, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
