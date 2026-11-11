"""Tests for export-page signature helpers and stale artefact handling."""

from __future__ import annotations

import pytest
from ui import export, state

from data_processor.models import FilterCondition, SortSpec


class _FakeSessionState(dict[str, object]):
    """Minimal dict-like replacement for ``st.session_state``."""


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSessionState:
    fake = _FakeSessionState()
    monkeypatch.setattr(state.st, "session_state", fake)
    state.init_state()
    return fake


def test_csv_signature_includes_all_inputs() -> None:
    signature = export._csv_signature(
        use_view=True,
        delimiter=";",
        encoding="utf-8",
        include_index=True,
        protect_formulas=False,
    )
    assert signature == (0, True, ";", "utf-8", True, False)


def test_csv_signature_changes_with_each_option() -> None:
    base = export._csv_signature(
        use_view=True,
        delimiter=",",
        encoding="utf-8",
        include_index=False,
        protect_formulas=True,
    )
    variants = [
        export._csv_signature(
            use_view=False,
            delimiter=",",
            encoding="utf-8",
            include_index=False,
            protect_formulas=True,
        ),
        export._csv_signature(
            use_view=True,
            delimiter=";",
            encoding="utf-8",
            include_index=False,
            protect_formulas=True,
        ),
        export._csv_signature(
            use_view=True,
            delimiter=",",
            encoding="cp1251",
            include_index=False,
            protect_formulas=True,
        ),
        export._csv_signature(
            use_view=True,
            delimiter=",",
            encoding="utf-8",
            include_index=True,
            protect_formulas=True,
        ),
        export._csv_signature(
            use_view=True,
            delimiter=",",
            encoding="utf-8",
            include_index=False,
            protect_formulas=False,
        ),
    ]
    assert all(variant != base for variant in variants)
    assert len({base, *variants}) == len(variants) + 1


def test_report_signature_includes_view_and_format() -> None:
    assert export._report_signature(use_view=True, report_format="TXT") == (
        0,
        True,
        "TXT",
    )
    assert export._report_signature(use_view=False, report_format="JSON") == (0, False, "JSON")


def test_signatures_change_after_state_change(session: _FakeSessionState) -> None:
    before = export._csv_signature(
        use_view=True,
        delimiter=",",
        encoding="utf-8",
        include_index=False,
        protect_formulas=True,
    )
    state.add_filter(FilterCondition("score", ">", 0))
    after = export._csv_signature(
        use_view=True,
        delimiter=",",
        encoding="utf-8",
        include_index=False,
        protect_formulas=True,
    )
    assert after != before

    before_report = export._report_signature(use_view=True, report_format="TXT")
    state.add_sort_spec(SortSpec("score", ascending=False))
    after_report = export._report_signature(use_view=True, report_format="TXT")
    assert after_report != before_report


def test_current_export_returns_matching_entry(session: _FakeSessionState) -> None:
    signature = ("signature",)
    session[state.EXPORT_CSV_KEY] = {"data": b"csv", "signature": signature}
    assert export._current_export(state.EXPORT_CSV_KEY, signature) == {
        "data": b"csv",
        "signature": signature,
    }


def test_current_export_discards_stale_entry(session: _FakeSessionState) -> None:
    session[state.EXPORT_REPORT_KEY] = {
        "data": b"report",
        "signature": ("old",),
    }
    assert export._current_export(state.EXPORT_REPORT_KEY, ("new",)) is None
    assert state.EXPORT_REPORT_KEY not in session


def test_current_export_missing_returns_none(session: _FakeSessionState) -> None:
    assert export._current_export(state.EXPORT_CSV_KEY, ("signature",)) is None
