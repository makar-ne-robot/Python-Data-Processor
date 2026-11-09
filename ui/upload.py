"""Upload page: load a CSV file into the session."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_processor import DataProcessorError, analyze_dataframe, load_csv_bytes, load_csv_path
from data_processor.config import SAMPLE_DATA_PATH, SUPPORTED_ENCODINGS, get_settings
from data_processor.exceptions import format_size
from ui import state
from ui.components import empty_state, metrics, page_header, render_table, section

_ENCODING_OPTIONS = ["Auto-detect", *SUPPORTED_ENCODINGS]
_CUSTOM_DELIMITER = "Custom…"
_DELIMITER_OPTIONS: dict[str, str | None] = {
    "Auto-detect": None,
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab (\\t)": "\t",
    "Pipe (|)": "|",
    _CUSTOM_DELIMITER: "custom",
}


def render() -> None:
    """Render the upload and preview workflow."""
    page_header(
        "Upload",
        "Load a CSV file, inspect its structure and start processing.",
    )

    settings = get_settings()
    for warning in settings.warnings:
        st.warning(warning)

    _render_loader(settings.max_file_size_bytes)

    if state.has_dataset():
        st.divider()
        _render_current_dataset()


def _render_loader(max_file_size: int) -> None:
    with st.container(border=True):
        uploaded = st.file_uploader(
            "CSV file",
            type=["csv", "txt", "tsv"],
            help=(
                f"Maximum size: {format_size(max_file_size)}. "
                "The file is processed in memory and never written to disk."
            ),
        )

        column_encoding, column_delimiter, column_mode = st.columns(3)
        encoding_label = column_encoding.selectbox("Encoding", _ENCODING_OPTIONS)
        delimiter_label = column_delimiter.selectbox("Delimiter", list(_DELIMITER_OPTIONS.keys()))
        strict = column_mode.toggle(
            "Strict mode",
            value=True,
            help="Strict mode rejects malformed rows; disable it to skip them instead.",
        )

        encoding: str | None = None if encoding_label == "Auto-detect" else encoding_label

        delimiter = _DELIMITER_OPTIONS[delimiter_label]
        if delimiter == "custom":
            custom = st.text_input("Custom delimiter (one character)", max_chars=1)
            delimiter = custom or None

        if uploaded is not None:
            st.caption(f"Selected: **{uploaded.name}** — {format_size(uploaded.size)}")

        replace_ok = True
        if state.has_dataset():
            replace_ok = st.checkbox(
                "Replace the currently loaded dataset",
                help="The current working data and operation history will be discarded.",
            )

        column_load, column_sample, column_clear = st.columns(3)
        load_clicked = column_load.button(
            "Load & analyze",
            type="primary",
            disabled=uploaded is None or not replace_ok,
            width="stretch",
        )
        sample_clicked = column_sample.button(
            "Load example dataset",
            disabled=not replace_ok,
            width="stretch",
        )
        clear_clicked = column_clear.button(
            "Clear dataset",
            disabled=not state.has_dataset(),
            width="stretch",
        )

    if load_clicked and uploaded is not None:
        _load_bytes(uploaded.getvalue(), uploaded.name, encoding, delimiter, strict)

    if sample_clicked:
        _load_sample(strict)

    if clear_clicked:
        state.clear_dataset()
        st.success("Dataset cleared.")
        st.rerun()


def _load_bytes(
    raw: bytes,
    filename: str,
    encoding: str | None,
    delimiter: str | None,
    strict: bool,
) -> None:
    try:
        loaded = load_csv_bytes(
            raw,
            filename=filename,
            encoding=encoding,
            delimiter=delimiter,
            strict=strict,
        )
    except DataProcessorError as error:
        state.add_error(str(error))
        st.error(str(error), icon="⚠️")
        return
    except Exception as error:  # last-resort guard for unexpected failures
        message = f"Unexpected error while reading the file: {error}"
        state.add_error(message)
        st.error(message, icon="⚠️")
        return

    state.load_dataset(loaded)
    st.success(
        f"Loaded **{filename}** — {loaded.metadata.row_count:,} rows × "
        f"{loaded.metadata.column_count:,} columns "
        f"(encoding: {loaded.metadata.encoding}, delimiter: {loaded.metadata.delimiter!r}).",
        icon="✅",
    )
    st.rerun()


def _load_sample(strict: bool) -> None:
    if not Path(SAMPLE_DATA_PATH).is_file():
        st.error("The bundled example dataset could not be found.")
        return
    try:
        loaded = load_csv_path(SAMPLE_DATA_PATH, strict=strict)
    except DataProcessorError as error:
        state.add_error(str(error))
        st.error(str(error), icon="⚠️")
        return
    state.load_dataset(loaded)
    st.success(
        f"Loaded example dataset **{loaded.metadata.filename}** "
        f"({loaded.metadata.row_count:,} rows).",
        icon="✅",
    )
    st.rerun()


def _render_current_dataset() -> None:
    source = state.get_source()
    working = state.get_working()
    if source is None or working is None:
        empty_state("No dataset is loaded.")
        return

    section("Loaded dataset")
    metrics(
        [
            ("File", source.filename, None),
            ("Size", format_size(source.size_bytes), None),
            ("Rows", source.row_count, "Rows in the original file"),
            ("Columns", source.column_count, None),
        ]
    )
    metrics(
        [
            ("Encoding", source.encoding, None),
            ("Delimiter", repr(source.delimiter), None),
            ("Current rows", len(working), "After cleaning operations"),
            ("Current columns", working.shape[1], None),
        ]
    )

    for warning in state.get_warnings():
        st.warning(warning)

    section("Preview")
    preview_rows = get_settings().preview_rows
    render_table(working.head(preview_rows))

    section("Columns")
    render_table(
        _column_table(working),
        height=min(420, 90 + 35 * max(len(working.columns), 1)),
    )

    analysis = analyze_dataframe(working)
    if analysis.total_missing:
        st.caption(
            f"Preview shows the first {preview_rows} rows. "
            f"{analysis.total_missing:,} missing value(s) detected in the current data."
        )


def _column_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": [str(column) for column in frame.columns],
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    )
