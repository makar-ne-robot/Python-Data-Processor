"""Tests for :mod:`data_processor.loader`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_processor.exceptions import (
    DecodingError,
    EmptyFileError,
    FileTooLargeError,
    ParsingError,
)
from data_processor.loader import (
    detect_delimiter,
    detect_encoding,
    load_csv_bytes,
    load_csv_path,
)


def test_detect_encoding_plain_utf8() -> None:
    assert detect_encoding(b"hello,world") == "utf-8"


def test_detect_encoding_utf8_bom() -> None:
    raw = b"\xef\xbb\xbfid,name\n1,Anna\n"
    assert detect_encoding(raw) == "utf-8-sig"


def test_detect_encoding_cp1251() -> None:
    raw = "имя,город\nАнна,Москва\n".encode("cp1251")
    assert detect_encoding(raw) == "cp1251"


def test_detect_encoding_exhausts_candidates() -> None:
    with pytest.raises(DecodingError):
        detect_encoding(b"\xff\xff\xff", candidates=("utf-8",))


def test_detect_delimiter_comma() -> None:
    assert detect_delimiter("a,b,c\n1,2,3\n") == ","


def test_detect_delimiter_semicolon() -> None:
    assert detect_delimiter("a;b;c\n1;2;3\n") == ";"


def test_detect_delimiter_falls_back_to_comma() -> None:
    assert detect_delimiter("single column no delimiters") == ","


def test_load_basic_utf8(csv_bytes) -> None:
    frame = pd.DataFrame({"id": [1, 2], "name": ["Anna", "Boris"]})
    loaded = load_csv_bytes(csv_bytes(frame), filename="people.csv")
    assert loaded.metadata.filename == "people.csv"
    assert loaded.metadata.encoding == "utf-8"
    assert loaded.metadata.delimiter == ","
    assert loaded.metadata.row_count == 2
    assert loaded.metadata.column_count == 2
    assert list(loaded.dataframe.columns) == ["id", "name"]


def test_load_utf8_bom_strips_bom(csv_bytes) -> None:
    frame = pd.DataFrame({"id": [1], "name": ["Anna"]})
    loaded = load_csv_bytes(csv_bytes(frame, bom=True))
    assert loaded.metadata.encoding == "utf-8-sig"
    assert list(loaded.dataframe.columns) == ["id", "name"]


def test_load_cp1251_roundtrip() -> None:
    text = "id,city\n1,Москва\n2,Минск\n"
    loaded = load_csv_bytes(text.encode("cp1251"))
    assert loaded.metadata.encoding == "cp1251"
    assert loaded.dataframe["city"].tolist() == ["Москва", "Минск"]


def test_load_semicolon_autodetect(csv_bytes) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    loaded = load_csv_bytes(csv_bytes(frame, delimiter=";"))
    assert loaded.metadata.delimiter == ";"
    assert loaded.metadata.column_count == 2


def test_load_tab_autodetect(csv_bytes) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    loaded = load_csv_bytes(csv_bytes(frame, delimiter="\t"))
    assert loaded.metadata.delimiter == "\t"
    assert loaded.metadata.column_count == 2


def test_manual_delimiter_override() -> None:
    raw = b"a;b\n1;2\n"
    loaded = load_csv_bytes(raw, delimiter=",")
    assert loaded.metadata.delimiter == ","
    assert loaded.metadata.column_count == 1


def test_manual_encoding_override() -> None:
    raw = "word\ncafé\n".encode("cp1252")
    loaded = load_csv_bytes(raw, encoding="cp1252")
    assert loaded.metadata.encoding == "cp1252"
    assert loaded.dataframe.iloc[0, 0] == "café"


def test_file_too_large() -> None:
    raw = b"a,b\n" + b"1,2\n" * 100
    with pytest.raises(FileTooLargeError):
        load_csv_bytes(raw, max_bytes=10)


def test_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        load_csv_bytes(b"")


def test_whitespace_only_file() -> None:
    with pytest.raises(EmptyFileError):
        load_csv_bytes(b"   \n\t\n")


def test_binary_content_rejected() -> None:
    with pytest.raises(ParsingError):
        load_csv_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")


def test_malformed_csv_strict_raises() -> None:
    raw = b'a,b\n"x,1\n'
    with pytest.raises(ParsingError):
        load_csv_bytes(raw, strict=True)


def test_malformed_csv_lenient_skips() -> None:
    raw = b'a,b\n"x,1\n'
    loaded = load_csv_bytes(raw, strict=False)
    assert loaded.metadata.column_count == 2
    assert loaded.warnings


def test_unknown_encoding_rejected() -> None:
    with pytest.raises(DecodingError):
        load_csv_bytes(b"a,b\n1,2\n", encoding="definitely-not-a-codec")


def test_header_only_file() -> None:
    loaded = load_csv_bytes(b"a,b\n")
    assert loaded.metadata.row_count == 0
    assert list(loaded.dataframe.columns) == ["a", "b"]


def test_load_csv_path_missing(tmp_path: Path) -> None:
    with pytest.raises(ParsingError):
        load_csv_path(tmp_path / "does-not-exist.csv")


def test_load_csv_path_success(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    loaded = load_csv_path(path)
    assert loaded.metadata.filename == "data.csv"
    assert loaded.metadata.row_count == 1


def test_raw_bytes_preserved(csv_bytes) -> None:
    frame = pd.DataFrame({"a": [1]})
    raw = csv_bytes(frame)
    loaded = load_csv_bytes(raw)
    assert loaded.raw_bytes == raw
