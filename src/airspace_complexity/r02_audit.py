from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence


EXCLUDED_DISPOSITIONS = {
    "exclude_non_data_file",
    "exclude_source_corrupt_file",
}
STATE_FIELDS = ("Lat", "Lon", "Altitude", "Speed", "Heading")


def _season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _month(row: Mapping[str, Any]) -> int:
    token = str(row.get("date_directory") or "")
    try:
        return int(token[:2])
    except ValueError:
        return 0


def select_stratified_files(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    minimum_per_airport: int = 20,
) -> list[Mapping[str, Any]]:
    """Select a stable round-robin sample across year, season, schema, and disposition."""
    eligible = [
        row
        for row in manifest_rows
        if row.get("extension") == ".csv"
        and row.get("r01_resolved") is True
        and row.get("r01_disposition") not in EXCLUDED_DISPOSITIONS
    ]
    selected: list[Mapping[str, Any]] = []
    for airport in sorted({str(row["airport_id"]) for row in eligible}):
        candidates = sorted(
            (row for row in eligible if row["airport_id"] == airport),
            key=lambda row: (
                str(row.get("year")),
                _season(_month(row)),
                int(row.get("header_column_count") or 0),
                str(row.get("r01_disposition")),
                str(row.get("relative_path")),
            ),
        )
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in candidates:
            key = (
                str(row.get("year")),
                _season(_month(row)),
                int(row.get("header_column_count") or 0),
                str(row.get("r01_disposition")),
            )
            buckets.setdefault(key, []).append(row)
        airport_selected: list[Mapping[str, Any]] = []
        positions = {key: 0 for key in buckets}
        while len(airport_selected) < min(minimum_per_airport, len(candidates)):
            progressed = False
            for key in sorted(buckets):
                position = positions[key]
                if position < len(buckets[key]):
                    airport_selected.append(buckets[key][position])
                    positions[key] += 1
                    progressed = True
                    if len(airport_selected) >= minimum_per_airport:
                        break
            if not progressed:
                break
        selected.extend(airport_selected)
    return sorted(selected, key=lambda row: (row["airport_id"], row["relative_path"]))


class DuplicateAccumulator:
    """Bounded per-file duplicate statistics for ID/timestamp report groups."""

    def __init__(self) -> None:
        self._groups: dict[tuple[Any, Any], dict[str, Any]] = {}

    def add(self, row: Mapping[str, Any]) -> None:
        key = (
            row.get("airport"),
            row.get("ID"),
            row.get("timestamp_naive"),
        )
        state = tuple(row.get(field) for field in STATE_FIELDS)
        age = row.get("Age")
        group = self._groups.setdefault(
            key,
            {"count": 0, "states": set(), "ages": []},
        )
        group["count"] += 1
        group["states"].add(state)
        if isinstance(age, (int, float)) and math.isfinite(age):
            group["ages"].append(float(age))

    def profile(self) -> dict[str, int]:
        counts = Counter(
            {
                "groups": len(self._groups),
                "duplicate_groups": 0,
                "identical_state_groups": 0,
                "conflicting_state_groups": 0,
                "nondecreasing_age_groups": 0,
            }
        )
        for group in self._groups.values():
            if group["count"] < 2:
                continue
            counts["duplicate_groups"] += 1
            if len(group["states"]) == 1:
                counts["identical_state_groups"] += 1
            else:
                counts["conflicting_state_groups"] += 1
            ages = group["ages"]
            if len(ages) >= 2 and all(a <= b for a, b in zip(ages, ages[1:])):
                counts["nondecreasing_age_groups"] += 1
        return dict(counts)


def path_local_date(date_directory: str | None) -> datetime | None:
    if not date_directory:
        return None
    try:
        return datetime.strptime(date_directory, "%m-%d-%y")
    except ValueError:
        return None


def utc_naive_to_us_eastern(utc_timestamp: datetime) -> datetime:
    """Convert naive UTC to US Eastern using post-2007 statutory DST rules."""
    year = utc_timestamp.year
    march_first = datetime(year, 3, 1)
    first_sunday_march = 1 + ((6 - march_first.weekday()) % 7)
    second_sunday_march = first_sunday_march + 7
    dst_start_utc = datetime(year, 3, second_sunday_march, 7)

    november_first = datetime(year, 11, 1)
    first_sunday_november = 1 + ((6 - november_first.weekday()) % 7)
    dst_end_utc = datetime(year, 11, first_sunday_november, 6)
    offset = -4 if dst_start_utc <= utc_timestamp < dst_end_utc else -5
    return utc_timestamp + timedelta(hours=offset)
