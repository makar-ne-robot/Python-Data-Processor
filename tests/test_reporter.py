"""Tests for :mod:`data_processor.reporter`."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from data_processor.models import (
    FilterCondition,
    OperationRecord,
    SortSpec,
    SourceMetadata,
)
from data_processor.reporter import ProcessingReport, build_report


def _source() -> SourceMetadata:
    return SourceMetadata(
        filename="input.csv",
        size_bytes=2048,
        encoding="utf-8",
        delimiter=",",
        row_count=10,
        column_count=5,
    )


def _operations() -> list[OperationRecord]:
    return [
        OperationRecord(
            category="duplicates",
            name="Remove full duplicates",
            details="Removed 2 full duplicate row(s).",
            rows_before=10,
            rows_after=8,
            affected_rows=2,
            metadata={"removed_rows": 2},
        ),
        OperationRecord(
            category="missing",
            name="Fill numeric missing (mean)",
            details="Filled 3 numeric missing value(s).",
            rows_before=8,
            rows_after=8,
            affected_cells=3,
            metadata={"strategy": "mean"},
        ),
        OperationRecord(
            category="columns",
            name="Drop columns above missing threshold",
            details="Dropped 2 column(s).",
            rows_before=8,
            rows_after=8,
            metadata={"dropped_columns": ["x", "y"]},
        ),
    ]


def _report(final_frame: pd.DataFrame | None = None) -> ProcessingReport:
    frame = (
        final_frame
        if final_frame is not None
        else pd.DataFrame({"id": [1, 2], "name": ["Anna", "Boris"]})
    )
    return build_report(
        source=_source(),
        original_shape=(10, 5),
        final_dataframe=frame,
        operations=_operations(),
        filters=[FilterCondition("age", "between", 18, 65)],
        sort_specs=[SortSpec("age", ascending=False)],
        warnings=["A sample warning."],
        errors=["A sample error."],
        generated_at=datetime(2024, 5, 1, 12, 30, 0),
    )


def test_report_summary_counts() -> None:
    report = _report()
    assert report.original_rows == 10
    assert report.original_columns == 5
    assert report.final_rows == 2
    assert report.final_columns == 2
    assert report.duplicates_removed == 2
    assert report.missing_values_handled == 3
    assert report.columns_dropped == 2
    assert report.source_filename == "input.csv"
    assert report.generated_at == datetime(2024, 5, 1, 12, 30, 0)


def test_report_statistics() -> None:
    report = _report()
    assert report.final_statistics["rows"] == 2
    assert report.final_statistics["columns"] == 2
    assert "id" in report.final_statistics["numeric_summary"]


def test_report_filters_and_sort() -> None:
    report = _report()
    assert report.filter_descriptions == ("age is between 18 and 65",)
    assert report.sort_description == "age DESC"


def test_report_to_dict_is_json_serializable() -> None:
    report = _report()
    payload = report.to_dict()
    assert payload["summary"]["duplicates_removed"] == 2
    assert payload["source"]["filename"] == "input.csv"
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["generated_at"] == "2024-05-01T12:30:00"
    assert len(decoded["operations"]) == 3


def test_report_to_json_roundtrip() -> None:
    report = _report()
    decoded = json.loads(report.to_json())
    assert decoded["report_title"].startswith("Python Data Processor")
    assert decoded["warnings"] == ["A sample warning."]
    assert decoded["errors"] == ["A sample error."]


def test_report_to_text_contains_sections() -> None:
    text = _report().to_text()
    assert "PYTHON DATA PROCESSOR — PROCESSING REPORT" in text
    assert "input.csv" in text
    assert "Duplicates removed : 2" in text
    assert "Missing handled    : 3" in text
    assert "Columns dropped    : 2" in text
    assert "age is between 18 and 65" in text
    assert "age DESC" in text
    assert "A sample warning." in text
    assert "A sample error." in text


def test_report_handles_empty_final_frame() -> None:
    report = _report(pd.DataFrame({"id": pd.Series(dtype="int64")}))
    assert report.final_rows == 0
    assert report.final_statistics["rows"] == 0
    assert "Duplicates removed : 2" in report.to_text()


def test_report_without_operations() -> None:
    report = build_report(
        source=_source(),
        original_shape=(10, 5),
        final_dataframe=pd.DataFrame({"a": [1]}),
        operations=[],
    )
    assert report.duplicates_removed == 0
    assert report.missing_values_handled == 0
    assert "No transformations were applied." in report.to_text()
    assert "No filters applied." in report.to_text()
