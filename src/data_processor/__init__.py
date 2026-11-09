"""Python Data Processor — business logic package.

The package is intentionally free of any Streamlit imports so that it can be
reused from scripts, notebooks and tests.
"""

from __future__ import annotations

from data_processor.analyzer import analyze_dataframe, missing_summary
from data_processor.cleaner import (
    drop_columns_above_missing_threshold,
    drop_duplicates_by_columns,
    drop_full_duplicates,
    drop_rows_with_missing,
    fill_categorical_missing,
    fill_missing_with_value,
    fill_numeric_missing,
)
from data_processor.config import Settings, get_settings
from data_processor.exceptions import DataProcessorError
from data_processor.exporter import export_csv, sanitize_cell, sanitize_dataframe
from data_processor.filters import apply_filters, sort_dataframe
from data_processor.loader import load_csv_bytes, load_csv_path
from data_processor.models import (
    AnalysisResult,
    CleaningResult,
    ColumnProfile,
    FilterCondition,
    LoadedCSV,
    OperationRecord,
    SortSpec,
    SourceMetadata,
)
from data_processor.reporter import ProcessingReport, build_report

__all__ = [
    "AnalysisResult",
    "CleaningResult",
    "ColumnProfile",
    "DataProcessorError",
    "FilterCondition",
    "LoadedCSV",
    "OperationRecord",
    "ProcessingReport",
    "Settings",
    "SortSpec",
    "SourceMetadata",
    "analyze_dataframe",
    "apply_filters",
    "build_report",
    "drop_columns_above_missing_threshold",
    "drop_duplicates_by_columns",
    "drop_full_duplicates",
    "drop_rows_with_missing",
    "export_csv",
    "fill_categorical_missing",
    "fill_missing_with_value",
    "fill_numeric_missing",
    "get_settings",
    "load_csv_bytes",
    "load_csv_path",
    "missing_summary",
    "sanitize_cell",
    "sanitize_dataframe",
    "sort_dataframe",
]

__version__ = "1.0.0"
