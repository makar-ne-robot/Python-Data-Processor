"""Python Data Processor — Streamlit entry point.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402
from ui import cleaning, dashboard, export, filtering, state, upload  # noqa: E402
from ui.styles import apply_theme  # noqa: E402

from data_processor import __version__  # noqa: E402
from data_processor.exceptions import format_size  # noqa: E402

PAGES = {
    "Dashboard": dashboard.render,
    "Upload": upload.render,
    "Data Cleaning": cleaning.render,
    "Filtering & Sorting": filtering.render,
    "Export": export.render,
}


def main() -> None:
    """Configure the page and dispatch to the selected page."""
    st.set_page_config(
        page_title="Python Data Processor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    state.init_state()

    _render_sidebar()
    selected = st.session_state.get("navigation", "Dashboard")
    PAGES.get(selected, dashboard.render)()


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 Data Processor")
        st.caption("Load, analyze, clean, filter and export CSV data.")
        st.divider()

        st.radio(
            "Navigation",
            list(PAGES),
            key="navigation",
            label_visibility="collapsed",
        )
        st.divider()

        if state.has_dataset():
            _render_dataset_status()
        else:
            st.caption("No dataset loaded yet.")

        st.divider()
        st.caption(f"Python Data Processor v{__version__}")


def _render_dataset_status() -> None:
    source = state.get_source()
    working = state.get_working()
    summary = state.operation_summary()

    if source is not None:
        st.markdown(f"**{source.filename}**")
        st.caption(
            f"{format_size(source.size_bytes)} · {source.encoding} · delimiter {source.delimiter!r}"
        )
    if working is not None:
        st.caption(f"Current shape: {len(working):,} × {working.shape[1]:,}")

    status_columns = st.columns(2)
    status_columns[0].metric("Operations", summary["operations"])
    status_columns[1].metric("Rows removed", summary["rows_removed"])

    action_columns = st.columns(2)
    undo_clicked = action_columns[0].button(
        "Undo",
        disabled=not state.can_undo(),
        width="stretch",
        help="Revert the last cleaning operation.",
    )
    if undo_clicked and state.undo_last():
        st.toast("Last operation undone.", icon="↩️")
        st.rerun()
    if action_columns[1].button("Reset data", width="stretch"):
        _confirm_reset()

    errors = state.get_errors()
    warnings = state.get_warnings()
    if errors or warnings:
        with st.expander(f"Messages ({len(errors) + len(warnings)})"):
            for message in errors:
                st.error(message, icon="⚠️")
            for message in warnings:
                st.warning(message)


def _confirm_reset() -> None:
    with st.popover("Confirm reset", width="stretch"):
        st.write("Restore the original dataset and clear all operations, filters and sorting.")
        confirmed = st.checkbox("I understand", key="reset::confirm")
        if st.button("Reset to original", type="primary", disabled=not confirmed):
            state.reset_to_original()
            st.toast("Dataset reset to the original.", icon="♻️")
            st.rerun()
        if st.button("Clear dataset", disabled=not confirmed):
            state.clear_dataset()
            st.toast("Dataset cleared.", icon="🗑️")
            st.rerun()


if __name__ == "__main__":
    main()
