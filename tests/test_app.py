"""End-to-end smoke tests for the Streamlit interface.

These tests drive the real application through Streamlit's ``AppTest`` runner,
so they verify that every page renders and that the main interactive flows
(preview → apply cleaning, export, filtering) work end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from data_processor import load_csv_path
from data_processor.config import SAMPLE_DATA_PATH

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

PAGES = ["Dashboard", "Upload", "Data Cleaning", "Filtering & Sorting", "Export"]


def _fresh_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


def _with_sample_data(app: AppTest) -> AppTest:
    loaded = load_csv_path(SAMPLE_DATA_PATH)
    app.session_state["dpp_original_df"] = loaded.dataframe.copy(deep=True)
    app.session_state["dpp_working_df"] = loaded.dataframe.copy(deep=True)
    app.session_state["dpp_source"] = loaded.metadata
    app.session_state["dpp_raw_bytes"] = loaded.raw_bytes
    return app


def _button(app: AppTest, key: str):
    return next(button for button in app.button if button.key == key)


def _exception_text(app: AppTest) -> str:
    if not app.exception:
        return ""
    return "; ".join(str(getattr(item, "value", item)) for item in app.exception)


def test_app_runs_without_dataset() -> None:
    app = _fresh_app()
    assert not app.exception
    assert any("Upload a CSV file" in info.value for info in app.info)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_dataset(page: str) -> None:
    app = _with_sample_data(_fresh_app())
    app.sidebar.radio[0].set_value(page).run()
    assert _exception_text(app) == ""


def test_directory_page_lists_navigation() -> None:
    app = _fresh_app()
    assert app.sidebar.radio[0].options == PAGES


def test_cleaning_preview_and_apply() -> None:
    app = _with_sample_data(_fresh_app())
    app.session_state["dpp_working_df"] = app.session_state["dpp_working_df"].head(10)
    app.sidebar.radio[0].set_value("Data Cleaning").run()

    _button(app, "preview::dupes::all").click().run()
    assert not app.exception

    next(
        checkbox for checkbox in app.checkbox if checkbox.key == "confirm::dupes::all"
    ).check().run()
    _button(app, "apply::dupes::all").click().run()

    assert _exception_text(app) == ""
    assert len(app.session_state["dpp_operations"]) == 1
    record = app.session_state["dpp_operations"][0]
    assert record.category == "duplicates"


def test_undo_restores_working_data() -> None:
    app = _with_sample_data(_fresh_app())
    app.session_state["dpp_working_df"] = app.session_state["dpp_working_df"].head(10)
    app.sidebar.radio[0].set_value("Data Cleaning").run()
    _button(app, "preview::dupes::all").click().run()
    next(
        checkbox for checkbox in app.checkbox if checkbox.key == "confirm::dupes::all"
    ).check().run()
    _button(app, "apply::dupes::all").click().run()
    assert len(app.session_state["dpp_working_df"]) == 9

    next(button for button in app.button if button.label == "Undo").click().run()
    assert _exception_text(app) == ""
    assert len(app.session_state["dpp_working_df"]) == 10
    assert app.session_state["dpp_operations"] == []


def test_filtering_adds_and_resets_condition() -> None:
    app = _with_sample_data(_fresh_app())
    app.sidebar.radio[0].set_value("Filtering & Sorting").run()

    _button(app, "filter::add").click().run()
    assert _exception_text(app) == ""
    assert len(app.session_state["dpp_filters"]) == 1

    _button(app, "filter::reset").click().run()
    assert app.session_state["dpp_filters"] == []


def test_export_prepares_csv_and_report() -> None:
    app = _with_sample_data(_fresh_app())
    app.sidebar.radio[0].set_value("Export").run()

    _button(app, "export::prepare").click().run()
    assert _exception_text(app) == ""
    assert "dpp_export_csv" in app.session_state
    assert app.session_state["dpp_export_csv"]["data"]

    _button(app, "export::generate_report").click().run()
    assert _exception_text(app) == ""
    assert "dpp_export_report" in app.session_state
    assert b"PROCESSING REPORT" in app.session_state["dpp_export_report"]["data"]
