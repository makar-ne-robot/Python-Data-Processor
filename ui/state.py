"""Streamlit session-state management.

The state layer is the single place that mutates the working dataset.  It
guarantees two invariants:

* the originally loaded DataFrame is never modified; and
* every transformation is recorded in an undo stack and an operation log.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_processor import (
    AnalysisResult,
    CleaningResult,
    DataProcessorError,
    FilterCondition,
    LoadedCSV,
    OperationRecord,
    SortSpec,
    SourceMetadata,
    analyze_dataframe,
    apply_filters,
    sort_dataframe,
)

_ORIGINAL_DF = "dpp_original_df"
_WORKING_DF = "dpp_working_df"
_SOURCE = "dpp_source"
_RAW_BYTES = "dpp_raw_bytes"
_OPERATIONS = "dpp_operations"
_FILTERS = "dpp_filters"
_SORT_SPECS = "dpp_sort_specs"
_WARNINGS = "dpp_warnings"
_ERRORS = "dpp_errors"
_UNDO_STACK = "dpp_undo_stack"
_VERSION = "dpp_data_version"
_INITIALIZED = "dpp_initialized"

_MAX_UNDO = 15


def init_state() -> None:
    """Initialise all session-state keys exactly once."""
    if st.session_state.get(_INITIALIZED):
        return
    st.session_state[_ORIGINAL_DF] = None
    st.session_state[_WORKING_DF] = None
    st.session_state[_SOURCE] = None
    st.session_state[_RAW_BYTES] = None
    st.session_state[_OPERATIONS] = []
    st.session_state[_FILTERS] = []
    st.session_state[_SORT_SPECS] = []
    st.session_state[_WARNINGS] = []
    st.session_state[_ERRORS] = []
    st.session_state[_UNDO_STACK] = []
    st.session_state[_VERSION] = 0
    st.session_state[_INITIALIZED] = True


def has_dataset() -> bool:
    """Return ``True`` when a dataset is loaded."""
    return st.session_state.get(_WORKING_DF) is not None


def load_dataset(loaded: LoadedCSV) -> None:
    """Install a freshly loaded dataset, resetting all derived state."""
    st.session_state[_ORIGINAL_DF] = loaded.dataframe.copy(deep=True)
    st.session_state[_WORKING_DF] = loaded.dataframe.copy(deep=True)
    st.session_state[_SOURCE] = loaded.metadata
    st.session_state[_RAW_BYTES] = loaded.raw_bytes
    st.session_state[_OPERATIONS] = []
    st.session_state[_FILTERS] = []
    st.session_state[_SORT_SPECS] = []
    st.session_state[_WARNINGS] = list(loaded.warnings)
    st.session_state[_ERRORS] = []
    st.session_state[_UNDO_STACK] = []
    _bump_version()


def clear_dataset() -> None:
    """Remove the loaded dataset and every derived value."""
    st.session_state[_ORIGINAL_DF] = None
    st.session_state[_WORKING_DF] = None
    st.session_state[_SOURCE] = None
    st.session_state[_RAW_BYTES] = None
    st.session_state[_OPERATIONS] = []
    st.session_state[_FILTERS] = []
    st.session_state[_SORT_SPECS] = []
    st.session_state[_UNDO_STACK] = []
    _bump_version()


def get_original() -> pd.DataFrame | None:
    """Return the immutable original DataFrame, if any."""
    return st.session_state.get(_ORIGINAL_DF)


def get_working() -> pd.DataFrame | None:
    """Return the current working DataFrame, if any."""
    return st.session_state.get(_WORKING_DF)


def get_source() -> SourceMetadata | None:
    """Return the metadata of the loaded file, if any."""
    return st.session_state.get(_SOURCE)


def get_raw_bytes() -> bytes | None:
    """Return the originally uploaded bytes, if any."""
    return st.session_state.get(_RAW_BYTES)


def get_operations() -> list[OperationRecord]:
    """Return the committed operation log."""
    return list(st.session_state.get(_OPERATIONS, []))


def get_filters() -> list[FilterCondition]:
    """Return active filter conditions."""
    return list(st.session_state.get(_FILTERS, []))


def get_sort_specs() -> list[SortSpec]:
    """Return active sort keys."""
    return list(st.session_state.get(_SORT_SPECS, []))


def get_warnings() -> list[str]:
    """Return collected warnings."""
    return list(st.session_state.get(_WARNINGS, []))


def get_errors() -> list[str]:
    """Return collected errors."""
    return list(st.session_state.get(_ERRORS, []))


def add_warning(message: str) -> None:
    """Record a non-fatal warning (deduplicated)."""
    warnings = st.session_state.setdefault(_WARNINGS, [])
    if message not in warnings:
        warnings.append(message)


def add_error(message: str) -> None:
    """Record a handled error (deduplicated)."""
    errors = st.session_state.setdefault(_ERRORS, [])
    if message not in errors:
        errors.append(message)


def data_version() -> int:
    """Return a counter that changes whenever the working data changes."""
    return int(st.session_state.get(_VERSION, 0))


def analyze_current() -> AnalysisResult | None:
    """Analyze the working DataFrame."""
    working = get_working()
    if working is None:
        return None
    return analyze_dataframe(working)


def get_result_dataframe() -> pd.DataFrame | None:
    """Return the working data with all active filters and sorting applied.

    Conditions that reference columns which no longer exist are ignored, so the
    dashboard and export views never crash after a column is dropped.
    """
    working = get_working()
    if working is None:
        return None

    result = working
    conditions = [condition for condition in get_filters() if condition.column in working.columns]
    specs = [spec for spec in get_sort_specs() if spec.column in working.columns]
    try:
        if conditions:
            result = apply_filters(result, conditions)
        if specs:
            result = sort_dataframe(result, specs)
    except DataProcessorError:
        return working.copy(deep=True)
    return result.copy(deep=True)


def commit_cleaning(result: CleaningResult, *, category: str, name: str) -> OperationRecord:
    """Commit a cleaning result to the working dataset.

    Args:
        result: The outcome produced by a cleaner function.
        category: Report category, e.g. ``"duplicates"``.
        name: Human-readable operation name.

    Returns:
        The operation record that was appended to the log.
    """
    before = get_working()
    if before is None:
        raise DataProcessorError("No dataset is loaded.")

    _push_undo()
    record = OperationRecord(
        category=category,
        name=name,
        details=result.description,
        rows_before=len(before),
        rows_after=len(result.dataframe),
        affected_rows=result.affected_rows,
        affected_cells=result.affected_cells,
        metadata=dict(result.details),
    )
    st.session_state[_WORKING_DF] = result.dataframe.copy(deep=True)
    st.session_state[_OPERATIONS] = [*get_operations(), record]
    _drop_orphaned_views()
    _bump_version()
    return record


def add_filter(condition: FilterCondition) -> None:
    """Append a filter condition."""
    st.session_state[_FILTERS] = [*get_filters(), condition]


def remove_filter(index: int) -> None:
    """Remove the filter at ``index``."""
    conditions = get_filters()
    if 0 <= index < len(conditions):
        del conditions[index]
        st.session_state[_FILTERS] = conditions


def clear_filters() -> None:
    """Remove every active filter."""
    st.session_state[_FILTERS] = []


def add_sort_spec(spec: SortSpec) -> None:
    """Append a sort key."""
    st.session_state[_SORT_SPECS] = [*get_sort_specs(), spec]


def remove_sort_spec(index: int) -> None:
    """Remove the sort key at ``index``."""
    specs = get_sort_specs()
    if 0 <= index < len(specs):
        del specs[index]
        st.session_state[_SORT_SPECS] = specs


def clear_sort_specs() -> None:
    """Remove every sort key."""
    st.session_state[_SORT_SPECS] = []


def can_undo() -> bool:
    """Return ``True`` when an undo snapshot is available."""
    return bool(st.session_state.get(_UNDO_STACK))


def undo_last() -> bool:
    """Restore the previous working dataset.

    Returns:
        ``True`` when a snapshot was restored.
    """
    stack: list[tuple[pd.DataFrame, list[OperationRecord]]] = st.session_state.get(_UNDO_STACK, [])
    if not stack:
        return False
    dataframe, operations = stack.pop()
    st.session_state[_WORKING_DF] = dataframe
    st.session_state[_OPERATIONS] = operations
    st.session_state[_UNDO_STACK] = stack
    _drop_orphaned_views()
    _bump_version()
    return True


def reset_to_original() -> bool:
    """Restore the working dataset from the untouched original.

    Returns:
        ``True`` when a dataset was present.
    """
    original = get_original()
    if original is None:
        return False
    _push_undo()
    st.session_state[_WORKING_DF] = original.copy(deep=True)
    st.session_state[_OPERATIONS] = []
    st.session_state[_FILTERS] = []
    st.session_state[_SORT_SPECS] = []
    _bump_version()
    return True


def operation_summary() -> dict[str, Any]:
    """Return aggregate counters used by the sidebar and export page."""
    operations = get_operations()
    return {
        "operations": len(operations),
        "duplicates_removed": sum(
            record.affected_rows for record in operations if record.category == "duplicates"
        ),
        "missing_handled": sum(
            record.affected_cells for record in operations if record.category == "missing"
        ),
        "rows_removed": sum(record.affected_rows for record in operations),
    }


def _push_undo() -> None:
    working = get_working()
    if working is None:
        return
    stack: list[tuple[pd.DataFrame, list[OperationRecord]]] = st.session_state.setdefault(
        _UNDO_STACK, []
    )
    stack.append((working.copy(deep=True), get_operations()))
    if len(stack) > _MAX_UNDO:
        del stack[0]
    st.session_state[_UNDO_STACK] = stack


def _drop_orphaned_views() -> None:
    working = get_working()
    if working is None:
        return
    columns = set(map(str, working.columns))
    filters = get_filters()
    valid_filters = [condition for condition in filters if condition.column in columns]
    if len(valid_filters) != len(filters):
        st.session_state[_FILTERS] = valid_filters
        add_warning("Filters referencing removed columns were discarded.")
    specs = get_sort_specs()
    valid_specs = [spec for spec in specs if spec.column in columns]
    if len(valid_specs) != len(specs):
        st.session_state[_SORT_SPECS] = valid_specs
        add_warning("Sort keys referencing removed columns were discarded.")


def _bump_version() -> None:
    st.session_state[_VERSION] = int(st.session_state.get(_VERSION, 0)) + 1
