"""CSV loading utilities.

The loader turns raw bytes into a :class:`~data_processor.models.LoadedCSV`.
It performs best-effort encoding and delimiter detection, validates the input
and converts low level ``pandas``/``csv`` errors into the application's own
exception hierarchy.  It never writes uploaded data to disk.
"""

from __future__ import annotations

import codecs
import csv
import io
from pathlib import Path

import pandas as pd

from data_processor.config import Settings, get_settings
from data_processor.exceptions import (
    DecodingError,
    EmptyFileError,
    FileTooLargeError,
    ParsingError,
)
from data_processor.models import LoadedCSV, SourceMetadata

_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
)

_SNIFF_SAMPLE_BYTES = 65536


def detect_encoding(raw: bytes, candidates: tuple[str, ...] | None = None) -> str:
    """Detect the most likely text encoding of ``raw`` bytes.

    Byte-order marks are inspected first.  Otherwise the candidates are tried
    in order and the first one that decodes the whole buffer without errors is
    returned.

    Args:
        raw: Raw file bytes.
        candidates: Encodings to try, in priority order.  Defaults to the
            application's supported encodings.

    Returns:
        The name of the detected codec.

    Raises:
        DecodingError: If no candidate could decode the data.
    """
    candidates = candidates or get_settings().encodings
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return encoding
    for encoding in candidates:
        try:
            raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding
    raise DecodingError(
        "Could not decode the file with any supported encoding "
        f"({', '.join(candidates)}). Try selecting the encoding manually."
    )


def detect_delimiter(text: str, candidates: tuple[str, ...] | None = None) -> str:
    """Detect the field delimiter in decoded CSV text.

    ``csv.Sniffer`` is tried first; if it fails or returns an unexpected value,
    the delimiter that appears most often in the first non-empty line wins.
    Falls back to a comma.

    Args:
        text: Decoded CSV text.
        candidates: Delimiters to consider.

    Returns:
        A single-character delimiter.
    """
    candidates = candidates or get_settings().delimiters
    sample = text[:_SNIFF_SAMPLE_BYTES]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        if dialect.delimiter in candidates:
            return dialect.delimiter
    except csv.Error:
        pass

    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    counts = {candidate: first_line.count(candidate) for candidate in candidates}
    best = max(counts, key=lambda candidate: counts[candidate])
    if counts[best] > 0:
        return best
    return ","


def _validate_delimiter(delimiter: str) -> None:
    if len(delimiter) != 1:
        raise ParsingError(f"Delimiter must be exactly one character, got {delimiter!r}.")


def _validate_encoding(encoding: str) -> None:
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise DecodingError(f"Unknown encoding: {encoding!r}.") from exc


def load_csv_bytes(
    raw: bytes,
    *,
    filename: str = "uploaded.csv",
    encoding: str | None = None,
    delimiter: str | None = None,
    max_bytes: int | None = None,
    strict: bool = True,
    settings: Settings | None = None,
) -> LoadedCSV:
    """Parse raw CSV bytes into a DataFrame.

    Args:
        raw: The file content as bytes.
        filename: Original file name, stored in the metadata.
        encoding: Force a specific encoding; auto-detected when ``None``.
        delimiter: Force a specific delimiter; auto-detected when ``None``.
        max_bytes: Maximum allowed size.  Defaults to the configured limit.
        strict: When ``True`` malformed rows raise :class:`ParsingError`.
            When ``False`` they are skipped and a warning is recorded.
        settings: Optional settings override (useful in tests).

    Returns:
        A :class:`LoadedCSV` containing the parsed frame and metadata.

    Raises:
        FileTooLargeError: If ``raw`` exceeds the size limit.
        EmptyFileError: If the file contains no usable data.
        DecodingError: If the bytes cannot be decoded.
        ParsingError: If the content is not valid CSV.
    """
    settings = settings or get_settings()
    limit = max_bytes if max_bytes is not None else settings.max_file_size_bytes
    warnings: list[str] = []

    if not raw:
        raise EmptyFileError("The uploaded file is empty.")
    if limit is not None and len(raw) > limit:
        raise FileTooLargeError(len(raw), limit)

    chosen_encoding = encoding or detect_encoding(raw, settings.encodings)
    _validate_encoding(chosen_encoding)
    try:
        text = raw.decode(chosen_encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise DecodingError(
            f"Unable to decode the file using {chosen_encoding!r}. Try another encoding."
        ) from exc

    # A manually forced "utf-8" may leave a BOM character in the first header.
    text = text.lstrip("\ufeff")

    if not text.strip():
        raise EmptyFileError("The uploaded file contains no data.")

    non_printable = text.count("\x00")
    if non_printable and non_printable / max(len(text), 1) > 0.01:
        raise ParsingError(
            "The file appears to be binary, not CSV. Check that you uploaded a text file."
        )

    chosen_delimiter = delimiter or detect_delimiter(text, settings.delimiters)
    _validate_delimiter(chosen_delimiter)

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=chosen_delimiter,
            engine="python" if not strict else "c",
            on_bad_lines="error" if strict else "skip",
        )
    except pd.errors.EmptyDataError as exc:
        raise EmptyFileError("The file does not contain any columns.") from exc
    except pd.errors.ParserError as exc:
        raise ParsingError(
            "The file is not valid CSV and could not be parsed. "
            "Check the delimiter, quotes and line endings."
        ) from exc
    except (ValueError, UnicodeError) as exc:
        raise ParsingError(f"The file could not be parsed: {exc}") from exc

    if not strict:
        warnings.append("Lenient parsing was enabled: malformed rows, if any, were skipped.")

    if dataframe.columns.empty:
        raise EmptyFileError("No columns were found in the file.")

    dataframe = dataframe.reset_index(drop=True)

    metadata = SourceMetadata(
        filename=filename,
        size_bytes=len(raw),
        encoding=chosen_encoding,
        delimiter=chosen_delimiter,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
    )
    return LoadedCSV(
        dataframe=dataframe,
        metadata=metadata,
        raw_bytes=raw,
        warnings=tuple(warnings),
    )


def load_csv_path(
    path: str | Path,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    max_bytes: int | None = None,
    strict: bool = True,
    settings: Settings | None = None,
) -> LoadedCSV:
    """Load a CSV file from a filesystem path.

    This is a thin convenience wrapper used by the example-data button and the
    tests.  Missing files surface as :class:`ParsingError` so callers only need
    to handle the application exception hierarchy.

    Args:
        path: Path to the CSV file.
        encoding: Force a specific encoding; auto-detected when ``None``.
        delimiter: Force a specific delimiter; auto-detected when ``None``.
        max_bytes: Maximum allowed size in bytes.
        strict: Whether malformed rows should raise.
        settings: Optional settings override.

    Returns:
        A :class:`LoadedCSV` instance.

    Raises:
        ParsingError: If the path does not exist or is not a file.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ParsingError(f"File not found: {csv_path}")
    raw = csv_path.read_bytes()
    return load_csv_bytes(
        raw,
        filename=csv_path.name,
        encoding=encoding,
        delimiter=delimiter,
        max_bytes=max_bytes,
        strict=strict,
        settings=settings,
    )
