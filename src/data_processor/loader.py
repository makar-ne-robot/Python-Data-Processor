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
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
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
    # UTF-32 LE starts with the UTF-16 LE BOM, so it must be checked first.
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

_SNIFF_SAMPLE_BYTES = 65536
_DETECTION_SAMPLE_BYTES = 65536
_UTF8_CODECS = frozenset({"utf-8", "utf-8-sig"})

# Minimum amount of non-ASCII evidence used to trust a legacy encoding guess.
_MIN_EVIDENCE_BYTES = 8
# Score difference below which two legacy decodings are considered ambiguous.
_AMBIGUITY_MARGIN = 0.35
# Absolute plausibility threshold for a decoded text.
_MIN_PLAUSIBLE_SCORE = 0.5

_SCRIPTS: tuple[str, ...] = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "HEBREW",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CJK",
    "THAI",
    "DEVANAGARI",
)

_CONTROL_CATEGORIES = frozenset({"Cc", "Cf", "Cn", "Co", "Cs"})
_ALLOWED_CONTROL_CHARS = frozenset("\t\n\r")


@dataclass(frozen=True, slots=True)
class EncodingDetection:
    """Outcome of best-effort encoding detection.

    ``confidence`` is a heuristic value between 0 and 1.  When it is low, or
    when ``warning`` is set, the caller should tell the user that the encoding
    was only guessed and that it can be selected manually.
    """

    encoding: str
    confidence: float
    alternatives: tuple[str, ...] = ()
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class _DecodingCandidate:
    encoding: str
    text: str
    score: float
    order: int


@lru_cache(maxsize=4096)
def _script_of(character: str) -> str | None:
    """Return the Unicode script of a character, if it is a letter."""
    name = unicodedata.name(character, "")
    for script in _SCRIPTS:
        if name.startswith(script):
            return script
    return None


def _sample_text(text: str, limit: int = _DETECTION_SAMPLE_BYTES) -> str:
    """Return a representative sample of a long text (head, middle, tail)."""
    if len(text) <= limit:
        return text
    third = limit // 3
    middle_start = max(third, len(text) // 2 - third // 2)
    return text[:third] + text[middle_start : middle_start + third] + text[-third:]


def _sample_bytes(raw: bytes, limit: int = _DETECTION_SAMPLE_BYTES) -> bytes:
    """Return a representative byte sample used for evidence counting."""
    if len(raw) <= limit:
        return raw
    third = limit // 3
    middle_start = max(third, len(raw) // 2 - third // 2)
    return raw[:third] + raw[middle_start : middle_start + third] + raw[-third:]


def _contains_control_characters(text: str) -> bool:
    """Return ``True`` when the text contains non-text control characters."""
    return any(
        unicodedata.category(character) in _CONTROL_CATEGORIES
        and character not in _ALLOWED_CONTROL_CHARS
        for character in text
    )


def _score_text(text: str) -> float:
    """Score how plausible a decoded text looks.

    The score uses the non-ASCII letters as evidence, because ASCII letters are
    identical in every supported legacy encoding.  It rewards a consistent
    dominant Unicode script and penalises mixed-script tokens, tokens made
    almost entirely of accented Latin letters (a symptom of Cyrillic bytes read
    through a Western codec), isolated Cyrillic letters in mostly Latin text,
    and control characters.  A low or close score leads to a warning instead of
    a confident claim.
    """
    sample = _sample_text(text)
    letters = [character for character in sample if character.isalpha()]
    if not letters:
        return 0.0

    non_ascii_letters = [character for character in letters if ord(character) > 127]
    non_ascii_scripts = [
        script for script in map(_script_of, non_ascii_letters) if script is not None
    ]
    if non_ascii_scripts:
        script_counts = Counter(non_ascii_scripts)
        dominant_script, dominant_count = script_counts.most_common(1)[0]
        consistency = dominant_count / len(non_ascii_scripts)
    else:
        dominant_script = None
        consistency = 1.0

    mixed_tokens = 0
    accented_latin_tokens = 0
    token_count = 0
    for token in re.split(r"[\s\W_]+", sample):
        token_letters = [character for character in token if character.isalpha()]
        if not token_letters:
            continue
        token_count += 1
        token_scripts = {
            script
            for script in (_script_of(character) for character in token_letters)
            if script is not None
        }
        if len(token_scripts) > 1:
            mixed_tokens += 1
        non_ascii_latin = sum(
            1
            for character in token_letters
            if _script_of(character) == "LATIN" and ord(character) > 127
        )
        if non_ascii_latin / len(token_letters) >= 0.5:
            accented_latin_tokens += 1

    mixed_ratio = mixed_tokens / token_count if token_count else 0.0
    accented_latin_ratio = accented_latin_tokens / token_count if token_count else 0.0

    ascii_latin_letters = sum(
        1 for character in letters if _script_of(character) == "LATIN" and ord(character) <= 127
    )
    sparse_cyrillic = (
        dominant_script == "CYRILLIC" and len(non_ascii_letters) * 2 < ascii_latin_letters
    )

    control_characters = sum(
        1
        for character in sample
        if unicodedata.category(character) in _CONTROL_CATEGORIES
        and character not in _ALLOWED_CONTROL_CHARS
    )
    control_ratio = control_characters / len(sample) if sample else 0.0

    return (
        3.0 * consistency
        - 4.0 * (1.0 - consistency)
        - 3.0 * mixed_ratio
        - 3.0 * accented_latin_ratio
        - (3.0 if sparse_cyrillic else 0.0)
        - 20.0 * control_ratio
    )


def _canonical_encoding(name: str) -> str | None:
    """Return the canonical codec name, or ``None`` when it is unknown."""
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def detect_encoding_details(
    raw: bytes, candidates: tuple[str, ...] | None = None
) -> EncodingDetection:
    """Detect the most likely text encoding of ``raw`` bytes.

    Byte-order marks are inspected first, then strict UTF-8 is attempted.  For
    legacy single-byte encodings a plausibility score is computed for each
    successful decoding; this avoids choosing an encoding merely because its
    codec accepts arbitrary bytes.  When the evidence is weak, ambiguous or
    implausible, the returned detection carries a warning instead of pretending
    to be certain.

    Args:
        raw: Raw file bytes.
        candidates: Encodings to try, in preference order.  Defaults to the
            application's supported encodings.

    Returns:
        An :class:`EncodingDetection` with the chosen encoding, a confidence
        estimate and an optional warning.

    Raises:
        DecodingError: If no candidate could decode the data.
    """
    candidates = candidates or get_settings().encodings

    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return EncodingDetection(encoding=encoding, confidence=1.0)

    for name in candidates:
        canonical = _canonical_encoding(name)
        if canonical not in _UTF8_CODECS:
            continue
        try:
            raw.decode(canonical)
        except UnicodeDecodeError:
            continue
        return EncodingDetection(encoding=name, confidence=1.0)

    decodings: dict[str, _DecodingCandidate] = {}
    seen_codecs: set[str] = set()
    for order, name in enumerate(candidates):
        canonical = _canonical_encoding(name)
        if canonical is None or canonical in _UTF8_CODECS or canonical in seen_codecs:
            continue
        seen_codecs.add(canonical)
        try:
            text = raw.decode(canonical)
        except UnicodeDecodeError:
            continue
        if text not in decodings:
            decodings[text] = _DecodingCandidate(
                encoding=name,
                text=text,
                score=_score_text(text),
                order=order,
            )

    if not decodings:
        raise DecodingError(
            "Could not decode the file with any supported encoding "
            f"({', '.join(candidates)}). Try selecting the encoding manually."
        )

    records = list(decodings.values())
    clean_records = [record for record in records if not _contains_control_characters(record.text)]
    if clean_records:
        records = clean_records
    records.sort(key=lambda record: (-record.score, record.order))

    best = records[0]
    alternatives = tuple(record.encoding for record in records[1:])
    evidence = sum(1 for byte in _sample_bytes(raw) if byte >= 0x80)

    reasons: list[str] = []
    margin = best.score - records[1].score if len(records) > 1 else 0.0
    if len(records) > 1 and margin < _AMBIGUITY_MARGIN:
        reasons.append(f"ambiguous between {best.encoding!r} and {records[1].encoding!r}")
    if best.score < _MIN_PLAUSIBLE_SCORE:
        reasons.append("the decoded text did not look plausible")
    if evidence < _MIN_EVIDENCE_BYTES:
        reasons.append(f"only {evidence} non-ASCII byte(s) were available")

    warning = None
    if reasons:
        warning = (
            "Encoding detection may be unreliable: "
            + "; ".join(reasons)
            + f". The file was decoded as {best.encoding!r}. "
            "Select the encoding manually if characters look wrong."
        )

    base_confidence = 1.0 if len(records) == 1 else min(1.0, max(0.0, 0.5 + margin / 2.0))
    confidence = round(base_confidence * min(1.0, evidence / _MIN_EVIDENCE_BYTES), 2)

    return EncodingDetection(
        encoding=best.encoding,
        confidence=confidence,
        alternatives=alternatives,
        warning=warning,
    )


def detect_encoding(raw: bytes, candidates: tuple[str, ...] | None = None) -> str:
    """Return the most likely encoding name for ``raw`` bytes.

    This is a convenience wrapper around :func:`detect_encoding_details` for
    callers that only need the codec name.
    """
    return detect_encoding_details(raw, candidates).encoding


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

    if encoding is not None:
        chosen_encoding = encoding
    else:
        detection = detect_encoding_details(raw, settings.encodings)
        chosen_encoding = detection.encoding
        if detection.warning is not None:
            warnings.append(detection.warning)

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
