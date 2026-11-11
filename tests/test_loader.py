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
    detect_encoding_details,
    load_csv_bytes,
    load_csv_path,
)


def test_detect_encoding_plain_utf8() -> None:
    assert detect_encoding(b"hello,world") == "utf-8"


def test_detect_encoding_utf8_has_no_warning() -> None:
    detection = detect_encoding_details("id,name\n1,Привет\n".encode())
    assert detection.encoding == "utf-8"
    assert detection.confidence == 1.0
    assert detection.warning is None


def test_detect_encoding_utf8_bom() -> None:
    raw = b"\xef\xbb\xbfid,name\n1,Anna\n"
    detection = detect_encoding_details(raw)
    assert detection.encoding == "utf-8-sig"
    assert detection.warning is None


def test_load_utf16_bom() -> None:
    raw = "a,b\n1,2\n".encode("utf-16")
    detection = detect_encoding_details(raw)
    assert detection.encoding == "utf-16"
    loaded = load_csv_bytes(raw)
    assert list(loaded.dataframe.columns) == ["a", "b"]
    assert loaded.dataframe.iloc[0, 0] == 1


def test_load_utf32_bom() -> None:
    raw = "a,b\n1,2\n".encode("utf-32")
    detection = detect_encoding_details(raw)
    assert detection.encoding == "utf-32"
    loaded = load_csv_bytes(raw)
    assert list(loaded.dataframe.columns) == ["a", "b"]
    assert loaded.dataframe.iloc[0, 0] == 1


def test_detect_encoding_cp1251() -> None:
    raw = "имя,город\nАнна,Москва\n".encode("cp1251")
    assert detect_encoding(raw) == "cp1251"


def test_detect_encoding_cp1251_with_ascii_header() -> None:
    raw = "id,city\n1,Москва\n2,Минск\n".encode("cp1251")
    detection = detect_encoding_details(raw)
    assert detection.encoding == "cp1251"
    assert detection.warning is None


def test_detect_encoding_cp1251_realistic() -> None:
    text = "имя,город,заметка\n1,Москва,Привет мир\n2,Минск,Спасибо\n3,Киев,Добрый день\n"
    detection = detect_encoding_details(text.encode("cp1251"))
    assert detection.encoding == "cp1251"
    assert detection.confidence == 1.0
    assert detection.warning is None


def test_detect_encoding_cp1252_with_real_characters() -> None:
    text = "id,city,note\n1,Zürich,Crème brûlée\n2,München,Škoda\n3,Orléans,Élève\n"
    detection = detect_encoding_details(text.encode("cp1252"))
    assert detection.encoding == "cp1252"
    assert detection.warning is None
    assert "cp1251" in detection.alternatives


def test_load_cp1252_roundtrip() -> None:
    text = "id,city,note\n1,Zürich,Crème brûlée\n2,München,Škoda\n3,Orléans,Élève\n"
    loaded = load_csv_bytes(text.encode("cp1252"))
    assert loaded.metadata.encoding == "cp1252"
    assert loaded.dataframe["city"].tolist() == ["Zürich", "München", "Orléans"]
    assert loaded.dataframe["note"].iloc[0] == "Crème brûlée"


def test_load_cp1252_specific_punctuation() -> None:
    text = "text\n\u201cHello\u201d \u2013 dash \u20ac 5\n"
    loaded = load_csv_bytes(text.encode("cp1252"))
    assert loaded.metadata.encoding == "cp1252"
    assert loaded.dataframe.iloc[0, 0] == "“Hello” – dash € 5"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "id,city\n1,Москва\n2,Минск\n",
            "cp1251",
        ),
        (
            "id,city,note\n1,Zürich,Crème brûlée\n2,München,Škoda\n3,Orléans,Élève\n",
            "cp1252",
        ),
    ],
)
@pytest.mark.parametrize(
    "candidates",
    [
        ("utf-8", "utf-8-sig", "cp1251", "cp1252", "iso-8859-1"),
        ("utf-8", "cp1252", "cp1251", "iso-8859-1"),
    ],
)
def test_detection_is_independent_of_candidate_order(
    text: str, expected: str, candidates: tuple[str, ...]
) -> None:
    encoding = expected
    raw = text.encode(encoding)
    detection = detect_encoding_details(raw, candidates)
    assert detection.encoding == expected


def test_detect_encoding_ambiguous_short_sequence() -> None:
    raw = b"a,b\n1,\xe9\n"
    detection = detect_encoding_details(raw)
    assert detection.encoding in {"cp1251", "cp1252"}
    assert detection.warning is not None
    assert "unreliable" in detection.warning.lower()
    assert detection.alternatives
    assert detection.confidence < 0.5


def test_load_reports_ambiguous_encoding_warning() -> None:
    loaded = load_csv_bytes(b"a,b\n1,\xe9\n")
    assert loaded.metadata.encoding in {"cp1251", "cp1252"}
    assert loaded.warnings
    assert "Encoding detection may be unreliable" in loaded.warnings[0]


def test_low_evidence_encoding_gets_warning() -> None:
    detection = detect_encoding_details("name,price\nMüller,10\nWeiß,20\n".encode("cp1252"))
    assert detection.encoding == "cp1252"
    assert detection.warning is not None


def test_manual_encoding_override_skips_detection_warning() -> None:
    raw = b"a,b\n1,\xe9\n"
    loaded = load_csv_bytes(raw, encoding="cp1252")
    assert loaded.metadata.encoding == "cp1252"
    assert loaded.warnings == ()
    assert loaded.dataframe.iloc[0, 1] == "é"


def test_manual_encoding_override_respects_user_choice() -> None:
    raw = b"a,b\n1,\xe9\n"
    loaded = load_csv_bytes(raw, encoding="cp1251")
    assert loaded.metadata.encoding == "cp1251"
    assert loaded.dataframe.iloc[0, 1] == "й"


def test_manual_iso_8859_1_still_supported() -> None:
    loaded = load_csv_bytes(b"word\ncaf\xe9\n", encoding="iso-8859-1")
    assert loaded.dataframe.iloc[0, 0] == "café"


def test_detect_encoding_rejects_unknown_candidates() -> None:
    with pytest.raises(DecodingError):
        detect_encoding_details(b"a,b\n1,2\n", candidates=("not-a-codec",))


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
