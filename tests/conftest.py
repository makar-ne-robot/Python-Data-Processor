"""Shared pytest fixtures.

All fixtures are self-contained: no test relies on files in the user's home
directory or on the network.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """A small frame with mixed dtypes, missing values and one duplicate row."""
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "name": ["Anna", "Boris", "Boris", None, "Dana"],
            "age": [30.0, np.nan, np.nan, 40.0, 25.0],
            "city": ["Berlin", "Munich", "Munich", "Paris", None],
            "signup": [
                "2023-01-01",
                "2023-02-01",
                "2023-02-01",
                "2023-03-01",
                "2023-04-01",
            ],
            "amount": [100.0, 200.0, 200.0, np.nan, 50.0],
            "tier": ["gold", "silver", "silver", "bronze", "gold"],
        }
    )


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    """A numeric-only frame without missing values."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})


@pytest.fixture
def single_column_frame() -> pd.DataFrame:
    """A frame with exactly one column."""
    return pd.DataFrame({"value": [1, 2, 3]})


@pytest.fixture
def empty_frame() -> pd.DataFrame:
    """A frame with no columns and no rows."""
    return pd.DataFrame()


@pytest.fixture
def empty_with_columns() -> pd.DataFrame:
    """A frame with columns but zero rows."""
    return pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})


@pytest.fixture
def csv_bytes() -> Callable[..., bytes]:
    """Return a factory that encodes a DataFrame as CSV bytes."""

    def _make(
        frame: pd.DataFrame,
        *,
        encoding: str = "utf-8",
        delimiter: str = ",",
        index: bool = False,
        bom: bool = False,
    ) -> bytes:
        buffer = io.StringIO()
        frame.to_csv(buffer, sep=delimiter, index=index, lineterminator="\n")
        payload = buffer.getvalue().encode(encoding)
        if bom:
            payload = b"\xef\xbb\xbf" + payload
        return payload

    return _make
