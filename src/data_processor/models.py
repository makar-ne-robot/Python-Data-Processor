"""Typed data containers shared across the application layers.

These dataclasses describe the values that travel between the business-logic
modules and the Streamlit UI.  Keeping them in one place makes the contracts
explicit and keeps ``pandas`` objects wrapped in well named types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(eq=False, slots=True)
class SourceMetadata:
    """Metadata about the file that is currently loaded."""

    filename: str
    size_bytes: int
    encoding: str
    delimiter: str
    row_count: int
    column_count: int


@dataclass(eq=False, slots=True)
class LoadedCSV:
    """Result of reading a CSV file from raw bytes."""

    dataframe: pd.DataFrame
    metadata: SourceMetadata
    raw_bytes: bytes
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Per-column summary produced by the analyzer."""

    name: str
    dtype: str
    non_null: int
    missing: int
    missing_pct: float
    unique: int


@dataclass(eq=False, slots=True)
class AnalysisResult:
    """High level analysis of a :class:`pandas.DataFrame`."""

    rows: int
    columns: int
    duplicate_rows: int
    total_missing: int
    total_cells: int
    numeric_columns: tuple[str, ...]
    column_profiles: tuple[ColumnProfile, ...]
    numeric_stats: dict[str, dict[str, float]]

    @property
    def missing_pct(self) -> float:
        """Percentage of missing cells in the whole frame."""
        if self.total_cells == 0:
            return 0.0
        return round(self.total_missing / self.total_cells * 100.0, 2)


@dataclass(eq=False, slots=True)
class CleaningResult:
    """Outcome of a single cleaning operation.

    A cleaning operation never mutates the input frame.  Instead it returns a
    brand new frame together with counters that feed the final report.
    """

    dataframe: pd.DataFrame
    affected_rows: int
    affected_cells: int
    description: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class OperationRecord:
    """Audit-log entry for one committed transformation."""

    category: str
    name: str
    details: str
    rows_before: int
    rows_after: int
    affected_rows: int = 0
    affected_cells: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class FilterCondition:
    """A single AND-combined filter condition."""

    column: str
    operator: str
    value: Any = None
    second_value: Any = None
    case_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class SortSpec:
    """A single sort key."""

    column: str
    ascending: bool = True
    na_position: str = "last"
