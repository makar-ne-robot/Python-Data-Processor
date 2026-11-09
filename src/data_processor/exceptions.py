"""Custom exception hierarchy for Python Data Processor.

Every error that is expected to reach the user is represented by a subclass of
:class:`DataProcessorError`.  The UI layer catches this base class, which makes
error handling predictable and keeps raw tracebacks away from the user.
"""

from __future__ import annotations


class DataProcessorError(Exception):
    """Base class for all application-specific errors."""


class FileTooLargeError(DataProcessorError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"File is too large ({format_size(size_bytes)}). "
            f"The maximum allowed size is {format_size(max_bytes)}."
        )


class EmptyFileError(DataProcessorError):
    """Raised when an uploaded file contains no data."""


class DecodingError(DataProcessorError):
    """Raised when the file cannot be decoded with any supported encoding."""


class ParsingError(DataProcessorError):
    """Raised when the decoded text cannot be parsed as tabular data."""


class CleaningError(DataProcessorError):
    """Raised when a cleaning operation receives invalid input."""


class FilterError(DataProcessorError):
    """Raised when a filter or sort request is invalid."""


class ExportError(DataProcessorError):
    """Raised when the processed data cannot be exported."""


class ReportError(DataProcessorError):
    """Raised when a report cannot be generated."""


def format_size(size_bytes: int) -> str:
    """Return a human readable file size using binary units.

    Args:
        size_bytes: Size in bytes.

    Returns:
        A string such as ``"1.5 MB"``.
    """
    if size_bytes < 0:
        raise ValueError("size_bytes must not be negative")
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"
