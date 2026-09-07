from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any, Sequence


NUMERIC_FIELDS = {
    "Altitude",
    "Speed",
    "Heading",
    "Lat",
    "Lon",
    "Age",
    "Range",
    "Bearing",
}
MISSING_TOKENS = {"", "none", "null", "nan"}
SCHEMAS = {
    (
        "ID",
        "Time",
        "Date",
        "Altitude",
        "Speed",
        "Heading",
        "Lat",
        "Lon",
        "Age",
        "Range",
        "Bearing",
        "Tail",
    ): "raw_v1_12",
    (
        "ID",
        "Time",
        "Date",
        "Altitude",
        "Speed",
        "Heading",
        "Lat",
        "Lon",
        "Age",
        "Range",
        "Bearing",
        "Tail",
        "AltisGNSS",
    ): "raw_v2_13_altisgnss",
}


@dataclass(frozen=True)
class DateTimeParseResult:
    timestamp_naive: datetime | None
    status: str
    error: str | None


def _missing(value: Any) -> bool:
    return value is None or str(value).strip().lower() in MISSING_TOKENS


def _literal_parts(value: str) -> list[str] | None:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    quoted_parts = re.findall(r"(?:u|U)?['\"]([^'\"]*)['\"]", text)
    if quoted_parts and len(quoted_parts) == text.count(",") + 1:
        return quoted_parts
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(part) for part in parsed]


def _date_parts(value: str) -> tuple[list[str] | None, str | None]:
    parts = _literal_parts(value)
    if parts is not None:
        if len(parts) != 3:
            return None, "invalid_date_shape"
        return parts, None
    text = value.strip()
    formats = []
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        formats.append(("%Y-%m-%d", text))
    if re.fullmatch(r"\d{2}-\d{2}-\d{2}", text):
        formats.append(("%m-%d-%y", text))
    for fmt, candidate in formats:
        try:
            date = datetime.strptime(candidate, fmt)
        except ValueError:
            return None, "invalid_datetime"
        return [str(date.year), str(date.month), str(date.day)], None
    return None, "invalid_date_format"


def _time_parts(value: str) -> tuple[list[str] | None, str | None]:
    parts = _literal_parts(value)
    if parts is not None:
        if len(parts) != 3:
            return None, "invalid_time_shape"
        return parts, None
    pieces = value.strip().split(":")
    if len(pieces) != 3:
        return None, "invalid_time_format"
    return pieces, None


def parse_raw_datetime(date_value: Any, time_value: Any) -> DateTimeParseResult:
    """Safely parse raw Date/Time values without assigning timezone semantics."""
    if _missing(date_value):
        return DateTimeParseResult(None, "error", "missing_date")
    if _missing(time_value):
        return DateTimeParseResult(None, "error", "missing_time")

    date_parts, date_error = _date_parts(str(date_value))
    if date_error:
        return DateTimeParseResult(None, "error", date_error)
    time_parts, time_error = _time_parts(str(time_value))
    if time_error:
        return DateTimeParseResult(None, "error", time_error)

    assert date_parts is not None and time_parts is not None
    try:
        second_token = time_parts[2].strip()
        if second_token.endswith(("Z", "z")):
            second_token = second_token[:-1]
        second_decimal = Decimal(second_token)
        if not Decimal(0) <= second_decimal < Decimal(60):
            raise ValueError("second out of range")
        total_microseconds = int(
            (second_decimal * Decimal(1_000_000)).quantize(
                Decimal(1), rounding=ROUND_HALF_EVEN
            )
        )
        minute_base = datetime(
            int(date_parts[0]),
            int(date_parts[1]),
            int(date_parts[2]),
            int(time_parts[0]),
            int(time_parts[1]),
        )
        timestamp = minute_base + timedelta(microseconds=total_microseconds)
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        return DateTimeParseResult(None, "error", "invalid_datetime")
    return DateTimeParseResult(timestamp, "ok", None)


def _us_eastern_local_candidate(
    timestamp: datetime,
) -> tuple[datetime, bool, bool]:
    """Return a labelled local-time candidate plus DST ambiguity flags."""
    march_first = datetime(timestamp.year, 3, 1)
    march_sunday = 1 + ((6 - march_first.weekday()) % 7) + 7
    november_first = datetime(timestamp.year, 11, 1)
    november_sunday = 1 + ((6 - november_first.weekday()) % 7)
    spring_gap_start = datetime(timestamp.year, 3, march_sunday, 2)
    spring_gap_end = spring_gap_start + timedelta(hours=1)
    autumn_fold_start = datetime(timestamp.year, 11, november_sunday, 1)
    autumn_fold_end = autumn_fold_start + timedelta(hours=1)
    nonexistent = spring_gap_start <= timestamp < spring_gap_end
    ambiguous = autumn_fold_start <= timestamp < autumn_fold_end
    dst = spring_gap_end <= timestamp < datetime(
        timestamp.year, 11, november_sunday, 2
    )
    offset = -4 if dst or ambiguous else -5
    return timestamp.replace(tzinfo=timezone(timedelta(hours=offset))), ambiguous, nonexistent


def _parse_float(value: Any) -> tuple[float | None, str | None]:
    if _missing(value):
        return None, None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None, "invalid_float"
    if not math.isfinite(parsed):
        return None, "nonfinite_float"
    return parsed, None


def _parse_bool(value: Any) -> tuple[bool | None, str | None]:
    if _missing(value):
        return None, None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True, None
    if normalized in {"false", "0"}:
        return False, None
    return None, "invalid_boolean"


def parse_raw_row(
    header: Sequence[str],
    row: Sequence[str],
    *,
    airport: str,
    source_path: str,
    source_row: int,
    r01_disposition: str,
    archive_member: str | None = None,
) -> dict[str, Any]:
    """Parse one row and preserve failures and provenance explicitly."""
    normalized_header = tuple(value.strip() for value in header)
    if len(row) != len(normalized_header):
        raise ValueError(
            f"column count mismatch: expected {len(normalized_header)}, got {len(row)}"
        )
    try:
        schema_version = SCHEMAS[normalized_header]
    except KeyError as exc:
        raise ValueError(f"unsupported raw schema: {normalized_header!r}") from exc

    raw = dict(zip(normalized_header, row, strict=True))
    result: dict[str, Any] = {}
    field_errors: dict[str, str] = {}
    for field in normalized_header:
        value = raw[field]
        if field in NUMERIC_FIELDS:
            parsed, error = _parse_float(value)
        elif field == "AltisGNSS":
            parsed, error = _parse_bool(value)
        else:
            parsed = None if _missing(value) else str(value).strip()
            error = None
        result[field] = parsed
        if error:
            field_errors[field] = error

    if "AltisGNSS" not in result:
        result["AltisGNSS"] = None
    timestamp = parse_raw_datetime(raw.get("Date"), raw.get("Time"))
    utc_candidate = (
        timestamp.timestamp_naive.replace(tzinfo=timezone.utc)
        if timestamp.timestamp_naive is not None
        else None
    )
    if timestamp.timestamp_naive is not None:
        local_candidate, local_ambiguous, local_nonexistent = (
            _us_eastern_local_candidate(timestamp.timestamp_naive)
        )
    else:
        local_candidate = None
        local_ambiguous = False
        local_nonexistent = False
    result.update(
        {
            "timestamp_naive": timestamp.timestamp_naive,
            "timestamp_utc_candidate": utc_candidate,
            "timestamp_assuming_utc": utc_candidate,
            "timestamp_assuming_airport_local": local_candidate,
            "local_candidate_ambiguous": local_ambiguous,
            "local_candidate_nonexistent": local_nonexistent,
            "timestamp_parse_status": timestamp.status,
            "timestamp_parse_error": timestamp.error,
            "schema_version": schema_version,
            "field_errors": field_errors,
            "airport": airport.upper(),
            "source_path": source_path,
            "archive_member": archive_member,
            "source_row": source_row,
            "r01_disposition": r01_disposition,
        }
    )
    return result
