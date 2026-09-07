from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


STATE_FIELDS = ("Lat", "Lon", "Altitude", "Speed", "Heading")
NON_COMPLETENESS_FIELDS = {
    "airport_id",
    "source_file_id",
    "source_relative_path",
    "source_row_number",
    "source_sequence",
    "source_path",
    "source_row",
    "archive_member",
    "r01_disposition",
    "field_errors",
}


@dataclass(frozen=True)
class DuplicateResolution:
    selected: Mapping[str, Any]
    classification: str
    conflict: bool
    source_rows: tuple[tuple[str, str, str | None, int], ...]


def _age_rank(row: Mapping[str, Any]) -> tuple[int, float]:
    value = row.get("age") if "age" in row else row.get("Age")
    if value is None:
        return (2, math.inf)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (2, math.inf)
    if not math.isfinite(number):
        return (2, math.inf)
    if number >= 0:
        return (0, number)
    return (1, abs(number))


def _completeness(row: Mapping[str, Any]) -> int:
    return sum(
        value is not None and value != ""
        for key, value in row.items()
        if key not in NON_COMPLETENESS_FIELDS
    )


def _integer_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the globally stable duplicate-selection policy key."""
    return (
        *_age_rank(row),
        -_completeness(row),
        str(row.get("airport_id", row.get("airport", ""))),
        str(row.get("source_file_id", "")),
        str(row.get("source_relative_path", "")),
        str(row.get("archive_member") or ""),
        _integer_or_zero(row.get("source_row_number")),
        _integer_or_zero(row.get("source_sequence")),
        str(row.get("source_path", "")),
        _integer_or_zero(row.get("source_row")),
    )


def resolve_duplicate_group(
    rows: Sequence[Mapping[str, Any]],
) -> DuplicateResolution:
    """Resolve an ID/timestamp group without averaging conflicting states."""
    if not rows:
        raise ValueError("duplicate group must contain at least one row")
    states = {tuple(row.get(field) for field in STATE_FIELDS) for row in rows}
    conflict = len(states) > 1
    selected = min(rows, key=selection_key)
    source_rows = tuple(
        sorted(
            (
                (
                    str(row.get("airport", "")),
                    str(row.get("source_path", "")),
                    (
                        str(row["archive_member"])
                        if row.get("archive_member") is not None
                        else None
                    ),
                    int(row.get("source_row", 0)),
                )
                for row in rows
            ),
            key=lambda item: (item[0], item[1], item[2] or "", item[3]),
        )
    )
    return DuplicateResolution(
        selected=selected,
        classification="conflicting_state" if conflict else "identical_state",
        conflict=conflict,
        source_rows=source_rows,
    )
