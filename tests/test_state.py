"""Tests for the Streamlit session-state layer, especially Undo snapshots.

Streamlit's real session state is replaced with a plain dict-like object so the
state functions can be exercised without a running server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ui import state

from data_processor.cleaner import (
    drop_columns_above_missing_threshold,
    drop_full_duplicates,
)
from data_processor.models import (
    FilterCondition,
    LoadedCSV,
    SortSpec,
    SourceMetadata,
)


class _FakeSessionState(dict[str, object]):
    """Minimal dict-like replacement for ``st.session_state``."""


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSessionState:
    fake = _FakeSessionState()
    monkeypatch.setattr(state.st, "session_state", fake)
    state.init_state()
    return fake


def _loaded(frame: pd.DataFrame, filename: str = "data.csv") -> LoadedCSV:
    metadata = SourceMetadata(
        filename=filename,
        size_bytes=128,
        encoding="utf-8",
        delimiter=",",
        row_count=len(frame),
        column_count=frame.shape[1],
    )
    return LoadedCSV(dataframe=frame, metadata=metadata, raw_bytes=b"raw", warnings=())


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "city": ["Berlin", "Munich", "Munich", "Paris"],
            "sparse": [1.0, np.nan, np.nan, np.nan],
            "score": [10, 20, 20, 40],
        }
    )


def _set_prepared_exports(session: _FakeSessionState) -> None:
    session[state.EXPORT_CSV_KEY] = {"data": b"csv", "filename": "x.csv"}
    session[state.EXPORT_REPORT_KEY] = {"data": b"report", "filename": "x.txt"}


# --------------------------------------------------------------------------- #
# Undo restores the complete state
# --------------------------------------------------------------------------- #


def test_undo_restores_dataframe_and_operations(session: _FakeSessionState) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))

    result = drop_full_duplicates(state.get_working())
    record = state.commit_cleaning(result, category="duplicates", name="Remove full duplicates")

    assert len(state.get_working()) == 3
    assert state.get_operations() == [record]
    assert state.can_undo()

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), frame)
    assert state.get_operations() == []
    assert not state.can_undo()


def test_undo_restores_filter_and_sort_after_dropped_column(
    session: _FakeSessionState,
) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))

    column_filter = FilterCondition("sparse", ">=", 0)
    column_sort = SortSpec("sparse", ascending=False)
    remaining_sort = SortSpec("score", ascending=True)
    state.add_filter(column_filter)
    state.add_sort_spec(column_sort)
    state.add_sort_spec(remaining_sort)

    result = drop_columns_above_missing_threshold(state.get_working(), 50)
    state.commit_cleaning(result, category="columns", name="Drop sparse column")

    assert "sparse" not in state.get_working().columns
    assert state.get_filters() == []
    assert state.get_sort_specs() == [remaining_sort]
    assert any("removed columns" in warning for warning in state.get_warnings())

    assert state.undo_last()

    assert "sparse" in state.get_working().columns
    assert state.get_filters() == [column_filter]
    assert state.get_sort_specs() == [column_sort, remaining_sort]
    assert state.get_operations() == []
    assert not any("removed columns" in warning for warning in state.get_warnings())


def test_undo_restores_operations_history(session: _FakeSessionState) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))

    first = state.commit_cleaning(
        drop_full_duplicates(state.get_working()),
        category="duplicates",
        name="Remove full duplicates",
    )
    state.add_filter(FilterCondition("score", ">", 0))
    state.add_sort_spec(SortSpec("score", ascending=False))

    after_first = state.get_working().copy(deep=True)
    first_filter = FilterCondition("score", ">", 0)

    state.commit_cleaning(
        drop_columns_above_missing_threshold(state.get_working(), 50),
        category="columns",
        name="Drop sparse column",
    )
    assert len(state.get_operations()) == 2

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), after_first)
    assert state.get_operations() == [first]
    assert state.get_filters() == [first_filter]
    assert state.get_sort_specs() == [SortSpec("score", ascending=False)]

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), frame)
    assert state.get_operations() == []
    assert not state.undo_last()


def test_multiple_sequential_undo(session: _FakeSessionState) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))

    state.commit_cleaning(
        drop_full_duplicates(state.get_working()),
        category="duplicates",
        name="Remove full duplicates",
    )
    after_dedupe = state.get_working().copy(deep=True)
    operations_after_dedupe = state.get_operations()

    state.commit_cleaning(
        drop_columns_above_missing_threshold(state.get_working(), 50),
        category="columns",
        name="Drop sparse column",
    )
    assert "sparse" not in state.get_working().columns

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), after_dedupe)
    assert state.get_operations() == operations_after_dedupe

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), frame)
    assert state.get_operations() == []


def test_undo_snapshots_are_independent(session: _FakeSessionState) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))

    state.commit_cleaning(
        drop_full_duplicates(state.get_working()),
        category="duplicates",
        name="Remove full duplicates",
    )
    after_dedupe = state.get_working().copy(deep=True)

    state.commit_cleaning(
        drop_columns_above_missing_threshold(state.get_working(), 50),
        category="columns",
        name="Drop sparse column",
    )
    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), after_dedupe)

    # Mutating the restored frame must not corrupt the older snapshot.
    state.get_working().loc[0, "score"] = -999
    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), frame)


def test_undo_after_reset_restores_data_view(session: _FakeSessionState) -> None:
    frame = _frame()
    state.load_dataset(_loaded(frame))
    state.commit_cleaning(
        drop_full_duplicates(state.get_working()),
        category="duplicates",
        name="Remove full duplicates",
    )
    state.add_filter(FilterCondition("score", ">", 0))
    state.add_sort_spec(SortSpec("score", ascending=False))

    expected_dataframe = state.get_working().copy(deep=True)
    expected_operations = state.get_operations()
    expected_filters = state.get_filters()
    expected_sorts = state.get_sort_specs()

    assert state.reset_to_original()
    assert state.get_operations() == []
    assert state.get_filters() == []
    assert state.get_sort_specs() == []

    assert state.undo_last()
    pd.testing.assert_frame_equal(state.get_working(), expected_dataframe)
    assert state.get_operations() == expected_operations
    assert state.get_filters() == expected_filters
    assert state.get_sort_specs() == expected_sorts


# --------------------------------------------------------------------------- #
# Export cache invalidation
# --------------------------------------------------------------------------- #


def test_exports_are_cleared_on_dataset_load(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame(), "a.csv"))
    _set_prepared_exports(session)

    state.load_dataset(_loaded(_frame(), "b.csv"))

    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_exports_are_cleared_on_dataset_clear(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))
    _set_prepared_exports(session)

    state.clear_dataset()

    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_exports_are_cleared_on_filter_changes(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))
    condition = FilterCondition("score", ">", 0)

    _set_prepared_exports(session)
    state.add_filter(condition)
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    _set_prepared_exports(session)
    state.remove_filter(0)
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    state.add_filter(condition)
    _set_prepared_exports(session)
    state.clear_filters()
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_exports_are_cleared_on_sort_changes(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))
    spec = SortSpec("score", ascending=False)

    _set_prepared_exports(session)
    state.add_sort_spec(spec)
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    _set_prepared_exports(session)
    state.remove_sort_spec(0)
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    state.add_sort_spec(spec)
    _set_prepared_exports(session)
    state.clear_sort_specs()
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_noop_filter_and_sort_clears_do_not_invalidate(
    session: _FakeSessionState,
) -> None:
    state.load_dataset(_loaded(_frame()))
    _set_prepared_exports(session)

    state.clear_filters()
    state.clear_sort_specs()

    assert state.EXPORT_CSV_KEY in session
    assert state.EXPORT_REPORT_KEY in session


def test_exports_are_cleared_on_cleaning_and_undo(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))

    _set_prepared_exports(session)
    state.commit_cleaning(
        drop_full_duplicates(state.get_working()),
        category="duplicates",
        name="Remove full duplicates",
    )
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    _set_prepared_exports(session)
    assert state.undo_last()
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_exports_are_cleared_on_reset(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))
    _set_prepared_exports(session)

    assert state.reset_to_original()
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_exports_are_cleared_when_messages_change(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))

    _set_prepared_exports(session)
    state.add_warning("First warning")
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session

    _set_prepared_exports(session)
    state.add_warning("First warning")  # duplicate: no state change
    assert state.EXPORT_CSV_KEY in session
    assert state.EXPORT_REPORT_KEY in session

    _set_prepared_exports(session)
    state.add_error("First error")
    assert state.EXPORT_CSV_KEY not in session
    assert state.EXPORT_REPORT_KEY not in session


def test_filter_and_sort_changes_bump_data_version(session: _FakeSessionState) -> None:
    state.load_dataset(_loaded(_frame()))
    version = state.data_version()

    state.add_filter(FilterCondition("score", ">", 0))
    assert state.data_version() == version + 1

    state.add_sort_spec(SortSpec("score", ascending=False))
    assert state.data_version() == version + 2

    state.clear_filters()
    state.clear_sort_specs()
    assert state.data_version() == version + 4
