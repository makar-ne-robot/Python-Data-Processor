"""Report generation.

Reports are built from the *actual* operations that were committed, so the
values they contain are always consistent with the data the user exported.
Reports can be rendered as human-readable text or as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from data_processor.analyzer import analyze_dataframe
from data_processor.filters import describe_condition, describe_sort
from data_processor.models import (
    FilterCondition,
    OperationRecord,
    SortSpec,
    SourceMetadata,
)


def _jsonable(value: Any) -> Any:
    """Convert values such as timestamps and numpy scalars to JSON-safe types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return None if (np.isnan(as_float) or np.isinf(as_float)) else as_float
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


@dataclass(eq=False, slots=True)
class ProcessingReport:
    """A complete, renderable processing report."""

    source_filename: str
    source_size_bytes: int
    generated_at: datetime
    encoding: str
    delimiter: str
    original_rows: int
    original_columns: int
    final_rows: int
    final_columns: int
    duplicates_removed: int
    missing_values_handled: int
    columns_dropped: int
    operations: tuple[OperationRecord, ...] = ()
    filter_descriptions: tuple[str, ...] = ()
    sort_description: str = ""
    final_statistics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "report_title": "Python Data Processor — processing report",
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "source": {
                "filename": self.source_filename,
                "size_bytes": self.source_size_bytes,
                "encoding": self.encoding,
                "delimiter": self.delimiter,
            },
            "summary": {
                "original_rows": self.original_rows,
                "original_columns": self.original_columns,
                "final_rows": self.final_rows,
                "final_columns": self.final_columns,
                "rows_removed": self.original_rows - self.final_rows,
                "duplicates_removed": self.duplicates_removed,
                "missing_values_handled": self.missing_values_handled,
                "columns_dropped": self.columns_dropped,
            },
            "operations": [
                {
                    "timestamp": record.timestamp.isoformat(timespec="seconds"),
                    "category": record.category,
                    "name": record.name,
                    "details": record.details,
                    "metadata": _jsonable(record.metadata),
                    "rows_before": record.rows_before,
                    "rows_after": record.rows_after,
                    "affected_rows": record.affected_rows,
                    "affected_cells": record.affected_cells,
                }
                for record in self.operations
            ],
            "filters": list(self.filter_descriptions),
            "sort": self.sort_description,
            "final_statistics": _jsonable(self.final_statistics),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Render the report as a human-readable plain-text document."""
        lines: list[str] = []
        add = lines.append

        add("=" * 68)
        add("PYTHON DATA PROCESSOR — PROCESSING REPORT")
        add("=" * 68)
        add(f"Source file        : {self.source_filename}")
        add(f"Generated at       : {self.generated_at.isoformat(timespec='seconds')}")
        add(f"Encoding           : {self.encoding}")
        add(f"Delimiter          : {self.delimiter!r}")
        add(f"Source size        : {self.source_size_bytes} bytes")
        add("")
        add("-" * 68)
        add("SHAPE")
        add("-" * 68)
        add(f"Original rows      : {self.original_rows}")
        add(f"Original columns   : {self.original_columns}")
        add(f"Final rows         : {self.final_rows}")
        add(f"Final columns      : {self.final_columns}")
        add(f"Rows removed       : {self.original_rows - self.final_rows}")
        add(f"Duplicates removed : {self.duplicates_removed}")
        add(f"Missing handled    : {self.missing_values_handled}")
        add(f"Columns dropped    : {self.columns_dropped}")
        add("")

        add("-" * 68)
        add("OPERATIONS")
        add("-" * 68)
        if self.operations:
            for index, record in enumerate(self.operations, start=1):
                add(f"{index}. [{record.category}] {record.name}")
                add(f"   {record.details}")
                add(
                    "   rows "
                    f"{record.rows_before} -> {record.rows_after}; "
                    f"affected rows={record.affected_rows}, cells={record.affected_cells}"
                )
                add(f"   at {record.timestamp.isoformat(timespec='seconds')}")
        else:
            add("No transformations were applied.")
        add("")

        add("-" * 68)
        add("FILTERS (combined with AND)")
        add("-" * 68)
        if self.filter_descriptions:
            for description in self.filter_descriptions:
                add(f"  - {description}")
        else:
            add("No filters applied.")
        add("")
        add(f"Sorting: {self.sort_description or 'none'}")
        add("")

        add("-" * 68)
        add("FINAL STATISTICS")
        add("-" * 68)
        for key, value in self.final_statistics.items():
            if isinstance(value, dict):
                add(f"{key}:")
                for sub_key, sub_value in value.items():
                    add(f"  {sub_key}: {sub_value}")
            else:
                add(f"{key}: {value}")
        add("")

        add("-" * 68)
        add("WARNINGS")
        add("-" * 68)
        add("\n".join(f"  - {item}" for item in self.warnings) if self.warnings else "  none")
        add("")
        add("-" * 68)
        add("ERRORS")
        add("-" * 68)
        add("\n".join(f"  - {item}" for item in self.errors) if self.errors else "  none")
        add("")
        add("=" * 68)
        return "\n".join(lines)


def _final_statistics(dataframe: pd.DataFrame) -> dict[str, Any]:
    analysis = analyze_dataframe(dataframe)
    numeric_summary: dict[str, dict[str, Any]] = {}
    for column, statistics in analysis.numeric_stats.items():
        numeric_summary[column] = {
            "count": statistics.get("count"),
            "mean": statistics.get("mean"),
            "std": statistics.get("std"),
            "min": statistics.get("min"),
            "max": statistics.get("max"),
        }
    return {
        "rows": analysis.rows,
        "columns": analysis.columns,
        "duplicate_rows": analysis.duplicate_rows,
        "missing_values": analysis.total_missing,
        "missing_pct": analysis.missing_pct,
        "numeric_summary": numeric_summary,
    }


def build_report(
    *,
    source: SourceMetadata,
    original_shape: tuple[int, int],
    final_dataframe: pd.DataFrame,
    operations: list[OperationRecord],
    filters: list[FilterCondition] | None = None,
    sort_specs: list[SortSpec] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    generated_at: datetime | None = None,
) -> ProcessingReport:
    """Assemble a :class:`ProcessingReport` from real processing state.

    Args:
        source: Metadata of the loaded file.
        original_shape: ``(rows, columns)`` of the data as first loaded.
        final_dataframe: The frame that will actually be exported.
        operations: Committed cleaning operations, in order.
        filters: Active filter conditions.
        sort_specs: Active sort keys.
        warnings: Non-fatal warnings collected during the session.
        errors: Errors that were handled and reported to the user.
        generated_at: Timestamp override (defaults to now).

    Returns:
        A populated :class:`ProcessingReport`.
    """
    operations = list(operations)
    filters = list(filters or [])
    sort_specs = list(sort_specs or [])

    duplicates_removed = sum(
        record.affected_rows for record in operations if record.category == "duplicates"
    )
    missing_handled = sum(
        record.affected_cells for record in operations if record.category == "missing"
    )
    columns_dropped = sum(
        len(record.metadata.get("dropped_columns", []) or [])
        for record in operations
        if record.category == "columns"
    )

    return ProcessingReport(
        source_filename=source.filename,
        source_size_bytes=source.size_bytes,
        generated_at=generated_at or datetime.now(),
        encoding=source.encoding,
        delimiter=source.delimiter,
        original_rows=int(original_shape[0]),
        original_columns=int(original_shape[1]),
        final_rows=int(final_dataframe.shape[0]),
        final_columns=int(final_dataframe.shape[1]),
        duplicates_removed=int(duplicates_removed),
        missing_values_handled=int(missing_handled),
        columns_dropped=int(columns_dropped),
        operations=tuple(operations),
        filter_descriptions=tuple(describe_condition(c) for c in filters),
        sort_description=describe_sort(sort_specs) if sort_specs else "",
        final_statistics=_final_statistics(final_dataframe),
        warnings=tuple(warnings or []),
        errors=tuple(errors or []),
    )
