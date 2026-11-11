"""Export page: download processed data, the original file and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_processor import DataProcessorError, build_report, export_csv
from data_processor.config import SUPPORTED_ENCODINGS
from ui import state
from ui.components import empty_state, metrics, page_header, section

_DELIMITER_OPTIONS: dict[str, str] = {
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab (\\t)": "\t",
    "Pipe (|)": "|",
}


def _csv_signature(
    *,
    use_view: bool,
    delimiter: str,
    encoding: str,
    include_index: bool,
    protect_formulas: bool,
) -> tuple[object, ...]:
    """Return the identity of the inputs that produced a prepared CSV."""
    return (
        state.data_version(),
        use_view,
        delimiter,
        encoding,
        include_index,
        protect_formulas,
    )


def _report_signature(*, use_view: bool, report_format: str) -> tuple[object, ...]:
    """Return the identity of the inputs that produced a prepared report."""
    return (state.data_version(), use_view, report_format)


def _current_export(key: str, signature: tuple[object, ...]) -> dict[str, Any] | None:
    """Return a prepared export if it matches ``signature``, else discard it."""
    stored = st.session_state.get(key)
    if stored is None:
        return None
    if stored.get("signature") != signature:
        st.session_state.pop(key, None)
        return None
    return stored


def render() -> None:
    """Render export and report generation controls."""
    page_header(
        "Export",
        "Download the processed CSV, keep the original file and generate a report.",
    )

    if not state.has_dataset():
        empty_state("Upload a CSV file on the **Upload** page to export data.")
        return

    source = state.get_source()
    working = state.get_working()
    result = state.get_result_dataframe()
    if source is None or working is None or result is None:
        empty_state("No dataset is loaded.")
        return

    use_view = st.toggle(
        "Apply active filters and sorting to the export",
        value=True,
        help="Disable to export all cleaned rows regardless of the current view.",
        key="export::use_view",
    )
    export_frame = result if use_view else working

    metrics(
        [
            ("Rows to export", len(export_frame), None),
            ("Columns", export_frame.shape[1], None),
            ("Active filters", len(state.get_filters()), None),
            ("Operations", len(state.get_operations()), None),
        ]
    )

    st.divider()
    _render_csv_export(export_frame, source.filename, use_view=use_view)
    st.divider()
    _render_original_download(source.filename)
    st.divider()
    _render_report(export_frame, use_view=use_view)


def _render_csv_export(export_frame: pd.DataFrame, source_filename: str, *, use_view: bool) -> None:
    section("Processed CSV")
    left, middle, right, far = st.columns(4)
    delimiter_label = left.selectbox("Delimiter", list(_DELIMITER_OPTIONS), key="export::delim")
    encoding = middle.selectbox("Encoding", SUPPORTED_ENCODINGS, key="export::encoding")
    include_index = right.toggle("Include index", value=False, key="export::index")
    protect = far.toggle(
        "Protect against CSV injection",
        value=True,
        help="Prefixes cells starting with =, +, - or @ so spreadsheets treat them as text.",
    )
    delimiter = _DELIMITER_OPTIONS[delimiter_label]
    signature = _csv_signature(
        use_view=use_view,
        delimiter=delimiter,
        encoding=encoding,
        include_index=include_index,
        protect_formulas=protect,
    )
    stored = _current_export(state.EXPORT_CSV_KEY, signature)

    st.dataframe(export_frame.head(50), width="stretch")
    if len(export_frame) > 50:
        st.caption(f"Preview shows the first 50 of {len(export_frame):,} rows.")

    if st.button("Prepare CSV", type="primary", key="export::prepare"):
        try:
            payload = export_csv(
                export_frame,
                delimiter=delimiter,
                encoding=encoding,
                include_index=include_index,
                protect_formulas=protect,
            )
        except DataProcessorError as error:
            state.add_error(str(error))
            st.error(str(error), icon="⚠️")
        else:
            filename = _output_name(source_filename, suffix=f"processed.{encoding}.csv")
            st.session_state[state.EXPORT_CSV_KEY] = {
                "data": payload,
                "filename": filename,
                "signature": signature,
            }
            stored = _current_export(state.EXPORT_CSV_KEY, signature)
            st.success(f"Prepared {filename} ({len(payload):,} bytes).", icon="✅")

    if stored:
        st.download_button(
            "Download processed CSV",
            data=stored["data"],
            file_name=stored["filename"],
            mime="text/csv",
            key="export::download_csv",
        )


def _render_original_download(source_filename: str) -> None:
    section("Original file")
    raw = state.get_raw_bytes()
    if not raw:
        st.caption("The original file is not available.")
        return
    st.caption("The uploaded file is kept byte-for-byte and can be downloaded unchanged.")
    st.download_button(
        "Download original file",
        data=raw,
        file_name=source_filename,
        mime="text/csv",
        key="export::download_original",
    )


def _render_report(export_frame: pd.DataFrame, *, use_view: bool) -> None:
    section("Report")
    report_format = st.radio(
        "Report format", ["TXT", "JSON"], horizontal=True, key="export::report_format"
    )
    signature = _report_signature(use_view=use_view, report_format=report_format)
    stored = _current_export(state.EXPORT_REPORT_KEY, signature)

    if st.button("Generate report", key="export::generate_report"):
        source = state.get_source()
        original = state.get_original()
        if source is None or original is None:
            st.error("No source metadata is available.", icon="⚠️")
            return
        try:
            report = build_report(
                source=source,
                original_shape=(len(original), original.shape[1]),
                final_dataframe=export_frame,
                operations=state.get_operations(),
                filters=state.get_filters(),
                sort_specs=state.get_sort_specs(),
                warnings=state.get_warnings(),
                errors=state.get_errors(),
            )
        except DataProcessorError as error:
            state.add_error(str(error))
            st.error(str(error), icon="⚠️")
            return

        if report_format == "TXT":
            text = report.to_text()
            payload = text.encode("utf-8")
            filename = _output_name(source.filename, suffix="report.txt")
            mime = "text/plain"
        else:
            text = report.to_json()
            payload = text.encode("utf-8")
            filename = _output_name(source.filename, suffix="report.json")
            mime = "application/json"
        st.session_state[state.EXPORT_REPORT_KEY] = {
            "data": payload,
            "filename": filename,
            "preview": text,
            "mime": mime,
            "signature": signature,
        }
        stored = _current_export(state.EXPORT_REPORT_KEY, signature)
        st.success(f"Report generated ({filename}).", icon="✅")

    if stored:
        with st.expander("Report preview", expanded=False):
            st.code(
                stored["preview"],
                language="json" if stored["mime"] == "application/json" else "text",
            )
        st.download_button(
            "Download report",
            data=stored["data"],
            file_name=stored["filename"],
            mime=stored["mime"],
            key="export::download_report",
        )


def _output_name(source_filename: str, *, suffix: str) -> str:
    stem = Path(source_filename).stem or "dataset"
    return f"{stem}_{suffix}"
