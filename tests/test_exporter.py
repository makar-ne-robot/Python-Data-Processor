"""Tests for :mod:`data_processor.exporter`."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processor.exceptions import ExportError
from data_processor.exporter import export_csv, sanitize_cell, sanitize_dataframe
from data_processor.loader import load_csv_bytes


def test_export_roundtrip(mixed_frame: pd.DataFrame) -> None:
    payload = export_csv(mixed_frame)
    loaded = load_csv_bytes(payload)
    assert loaded.metadata.row_count == len(mixed_frame)
    assert loaded.metadata.column_count == mixed_frame.shape[1]
    assert list(loaded.dataframe.columns) == list(mixed_frame.columns)


def test_export_delimiter_and_encoding() -> None:
    frame = pd.DataFrame({"city": ["Москва", "Минск"], "value": [1, 2]})
    payload = export_csv(frame, delimiter=";", encoding="cp1251")
    loaded = load_csv_bytes(payload, encoding="cp1251", delimiter=";")
    assert loaded.dataframe["city"].tolist() == ["Москва", "Минск"]


def test_export_include_index() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    payload = export_csv(frame, include_index=True)
    assert payload.startswith(b",a\n0,1\n")


def test_export_without_index() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    payload = export_csv(frame, include_index=False)
    assert payload == b"a\n1\n2\n"


def test_export_protects_formulas() -> None:
    frame = pd.DataFrame({"note": ["=1+1", "+cmd", "-cmd", "@cmd", "safe"]})
    payload = export_csv(frame, protect_formulas=True)
    text = payload.decode("utf-8")
    assert "'=1+1" in text
    assert "'+cmd" in text
    assert "'-cmd" in text
    assert "'@cmd" in text
    assert "safe" in text


def test_export_can_disable_protection() -> None:
    frame = pd.DataFrame({"note": ["=1+1"]})
    payload = export_csv(frame, protect_formulas=False)
    assert b"'=" not in payload


def test_numeric_negative_values_not_sanitized() -> None:
    assert sanitize_cell(-5.0) == -5.0
    assert sanitize_cell(42) == 42


def test_sanitize_cell_prefixes_dangerous_strings() -> None:
    assert sanitize_cell("=SUM(A1)") == "'=SUM(A1)"
    assert sanitize_cell("+1") == "'+1"
    assert sanitize_cell("-1") == "'-1"
    assert sanitize_cell("@user") == "'@user"
    assert sanitize_cell("  =hidden") == "'  =hidden"
    assert sanitize_cell("plain") == "plain"
    assert sanitize_cell("") == ""


def test_sanitize_dataframe_does_not_mutate() -> None:
    frame = pd.DataFrame({"note": ["=1+1", "safe"]})
    sanitized = sanitize_dataframe(frame)
    assert frame["note"].tolist() == ["=1+1", "safe"]
    assert sanitized["note"].tolist() == ["'=1+1", "safe"]


@pytest.mark.parametrize("delimiter", ["", ";;", "\n", "\r", '"'])
def test_invalid_delimiter(delimiter: str) -> None:
    with pytest.raises(ExportError):
        export_csv(pd.DataFrame({"a": [1]}), delimiter=delimiter)


def test_invalid_encoding() -> None:
    with pytest.raises(ExportError):
        export_csv(pd.DataFrame({"a": [1]}), encoding="not-a-codec")


def test_encoding_failure() -> None:
    frame = pd.DataFrame({"city": ["Zürich"]})
    with pytest.raises(ExportError):
        export_csv(frame, encoding="ascii")


def test_export_empty_frame() -> None:
    payload = export_csv(pd.DataFrame())
    assert isinstance(payload, bytes)


def test_export_preserves_column_order() -> None:
    frame = pd.DataFrame({"b": [2], "a": [1]})
    payload = export_csv(frame)
    assert payload.splitlines()[0] == b"b,a"


# --------------------------------------------------------------------------- #
# Header and index protection
# --------------------------------------------------------------------------- #


def test_sanitize_dataframe_protects_column_names() -> None:
    frame = pd.DataFrame({"=cmd": [1], "safe": [2]})
    sanitized = sanitize_dataframe(frame)
    assert list(sanitized.columns) == ["'=cmd", "safe"]
    assert list(frame.columns) == ["=cmd", "safe"]


@pytest.mark.parametrize("name", ["=cmd", "+cmd", "-cmd", "@cmd", "  =cmd", "\t=cmd"])
def test_sanitize_dataframe_protects_dangerous_headers(name: str) -> None:
    frame = pd.DataFrame({name: [1]})
    sanitized = sanitize_dataframe(frame)
    assert list(sanitized.columns) == ["'" + name]


def test_sanitize_dataframe_keeps_non_string_headers() -> None:
    frame = pd.DataFrame({1: [1], "safe": [2]})
    sanitized = sanitize_dataframe(frame)
    assert list(sanitized.columns) == [1, "safe"]


def test_sanitize_dataframe_protects_index_values_and_name() -> None:
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["=x", "safe"], name="=idx"))
    sanitized = sanitize_dataframe(frame, include_index=True)
    assert sanitized.index.tolist() == ["'=x", "safe"]
    assert sanitized.index.name == "'=idx"
    assert frame.index.tolist() == ["=x", "safe"]
    assert frame.index.name == "=idx"


def test_sanitize_dataframe_leaves_index_when_not_included() -> None:
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["=x", "safe"], name="=idx"))
    sanitized = sanitize_dataframe(frame, include_index=False)
    assert sanitized.index.tolist() == ["=x", "safe"]
    assert sanitized.index.name == "=idx"


def test_sanitize_dataframe_keeps_numeric_index_untouched() -> None:
    frame = pd.DataFrame({"v": [1, 2]})
    sanitized = sanitize_dataframe(frame, include_index=True)
    assert sanitized.index.tolist() == [0, 1]


def test_sanitize_dataframe_handles_string_dtype() -> None:
    frame = pd.DataFrame({"note": pd.Series(["=1+1", "safe"], dtype="string")})
    sanitized = sanitize_dataframe(frame)
    assert sanitized["note"].tolist() == ["'=1+1", "safe"]


def test_sanitize_dataframe_handles_categorical() -> None:
    frame = pd.DataFrame({"note": pd.Series(["=1+1", "safe", None], dtype="category")})
    sanitized = sanitize_dataframe(frame)
    assert sanitized["note"].tolist()[:2] == ["'=1+1", "safe"]


def test_sanitize_dataframe_handles_categorical_index() -> None:
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.CategoricalIndex(["=x", "safe"], name="=cat"))
    sanitized = sanitize_dataframe(frame, include_index=True)
    assert sanitized.index.tolist() == ["'=x", "safe"]
    assert sanitized.index.name == "'=cat"


def test_export_protects_header_and_index() -> None:
    frame = pd.DataFrame(
        {"=value": ["=1+1", "safe"]},
        index=pd.Index(["=row", "row2"], name="=id"),
    )
    text = export_csv(frame, include_index=True).decode("utf-8")
    assert "'=value" in text
    assert "'=row" in text
    assert "'=id" in text


def test_export_can_disable_protection_for_header_and_index() -> None:
    frame = pd.DataFrame({"=value": ["=1+1"]}, index=pd.Index(["=row"], name="=id"))
    text = export_csv(frame, include_index=True, protect_formulas=False).decode("utf-8")
    assert "=value" in text
    assert "'=value" not in text
    assert "=row" in text
    assert "'=row" not in text


def test_export_negative_numbers_not_corrupted() -> None:
    frame = pd.DataFrame({"value": [-5.0, 3.0]})
    assert export_csv(frame) == b"value\n-5.0\n3.0\n"


def test_export_multiindex_header_and_index_protected() -> None:
    index = pd.MultiIndex.from_tuples([("=a", 1)], names=["=lvl0", "lvl1"])
    frame = pd.DataFrame({"=col": ["=x"]}, index=index)
    text = export_csv(frame, include_index=True).decode("utf-8")
    assert "'=lvl0" in text
    assert "'=a" in text
    assert "'=col" in text
    assert "'=x" in text


def test_sanitize_dataframe_does_not_mutate_headers_or_index() -> None:
    frame = pd.DataFrame({"=col": ["=1+1"]}, index=pd.Index(["=row"], name="=id"))
    original_columns = list(frame.columns)
    original_index = frame.index.tolist()
    original_name = frame.index.name
    sanitize_dataframe(frame, include_index=True)
    assert list(frame.columns) == original_columns
    assert frame.index.tolist() == original_index
    assert frame.index.name == original_name
