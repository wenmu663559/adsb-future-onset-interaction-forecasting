from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from airspace_complexity.duplicate_audit import selection_key


STATE_FIELDS = ("lat", "lon", "altitude", "speed", "heading")
RECORD_TYPES = (
    "id_timestamp",
    "exact_report",
    "full_state",
    "contiguous_state_run",
    "age_repeat",
    "boundary_crossing",
)
DUPLICATE_SUMMARY_SCHEMA_VERSION = "r02_duplicate_summary_v1"
SUMMARY_RECORD_TYPES = (
    "id_timestamp",
    "exact_report",
    "full_state",
    "contiguous_state_run",
    "boundary_crossing",
)
AGE_DIMENSIONS = (
    "airport_id",
    "repeat_count_bin",
    "duration_bin",
    "age_min_bin",
    "age_max_bin",
    "age_trend",
    "state_update_relationship",
)


@dataclass(frozen=True)
class BucketDuplicateProfile:
    """Exact group records from one complete aircraft-hash bucket."""

    id_timestamp: tuple[Mapping[str, Any], ...]
    exact_report: tuple[Mapping[str, Any], ...]
    full_state: tuple[Mapping[str, Any], ...]
    contiguous_state_run: tuple[Mapping[str, Any], ...]
    age_repeat: tuple[Mapping[str, Any], ...]
    boundary_crossing: tuple[Mapping[str, Any], ...]

    def to_records(self) -> list[dict[str, Any]]:
        """Flatten records with stable type tags for concatenation across buckets."""
        records: list[dict[str, Any]] = []
        for record_type in RECORD_TYPES:
            for record in getattr(self, record_type):
                records.append({"record_type": record_type, **record})
        return records


@dataclass(frozen=True)
class BucketDuplicateSummary:
    """Mergeable sufficient statistics from one complete aircraft bucket."""

    duplicate: tuple[Mapping[str, Any], ...]
    age: tuple[Mapping[str, Any], ...]

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in (*self.duplicate, *self.age)]


def stable_bucket(airport_id: str, aircraft_id: str, bucket_count: int) -> int:
    """Assign an airport/aircraft identity to a stable SHA-256 bucket."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    identity = f"{airport_id}\0{aircraft_id}".encode("utf-8")
    prefix = hashlib.sha256(identity).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) % bucket_count


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("report_timestamp_utc must be a datetime or ISO timestamp")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: Any) -> str:
    return _timestamp(value).isoformat()


def _stable_scalar(value: Any) -> tuple[str, str]:
    if value is None:
        return ("none", "")
    return (type(value).__name__, repr(value))


def _state(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in STATE_FIELDS)


def _state_record(state: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(STATE_FIELDS, state))


def _integer_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("airport_id", "")),
        str(row.get("aircraft_id", "")),
        _timestamp(row.get("report_timestamp_utc")),
        _integer_or_zero(row.get("source_sequence")),
        str(row.get("source_file_id", "")),
        str(row.get("source_relative_path", "")),
        str(row.get("archive_member") or ""),
        _integer_or_zero(row.get("source_row_number")),
        tuple(_stable_scalar(value) for value in _state(row)),
    )


def _group_rows(
    rows: Iterable[Mapping[str, Any]],
    key,
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def _finite_ages(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    ages: list[float] = []
    for row in rows:
        value = row.get("age")
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            ages.append(number)
    return ages


def _source_file_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return len({str(row.get("source_file_id", "")) for row in rows})


def _selected_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_airport_id": str(row.get("airport_id", "")),
        "selected_source_file_id": str(row.get("source_file_id", "")),
        "selected_source_relative_path": str(
            row.get("source_relative_path", "")
        ),
        "selected_archive_member": row.get("archive_member"),
        "selected_source_row_number": _integer_or_zero(
            row.get("source_row_number")
        ),
        "selected_source_sequence": _integer_or_zero(row.get("source_sequence")),
    }


def _boundary_record(
    *,
    identity_type: str,
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_file_count = _source_file_count(rows)
    if source_file_count < 2:
        return None
    return {
        "identity_type": identity_type,
        **identity,
        "source_file_count": source_file_count,
        "crossing_groups": 1,
        "crossing_rows": len(rows),
    }


def _repeat_count_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 9:
        return "5-9"
    return "10+"


def _duration_bin(seconds: float) -> str:
    if seconds == 0:
        return "0"
    if seconds <= 1:
        return "(0,1]"
    if seconds <= 5:
        return "(1,5]"
    if seconds <= 30:
        return "(5,30]"
    if seconds <= 60:
        return "(30,60]"
    return ">60"


def _age_bin(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0:
        return "negative"
    if value == 0:
        return "0"
    if value <= 1:
        return "(0,1]"
    if value <= 5:
        return "(1,5]"
    if value <= 30:
        return "(5,30]"
    if value <= 60:
        return "(30,60]"
    return ">60"


def _age_trend(ages: Sequence[float]) -> str:
    if len(ages) < 2:
        return "insufficient"
    if ages[-1] > ages[0]:
        return "increasing"
    if ages[-1] < ages[0]:
        return "decreasing"
    return "unchanged"


def _new_duplicate_summary(airport: str, record_type: str) -> dict[str, Any]:
    return {
        "summary_schema_version": DUPLICATE_SUMMARY_SCHEMA_VERSION,
        "summary_kind": "duplicate",
        "airport_id": airport,
        "record_type": record_type,
        "total_groups": 0,
        "total_rows": 0,
        "duplicate_groups": 0,
        "duplicate_rows": 0,
        "repeat_groups": 0,
        "repeat_rows": 0,
        "excess_rows": 0,
        "identical_state_groups": 0,
        "conflicting_state_groups": 0,
        "file_boundary_crossing_groups": 0,
        "duration_or_span_count": 0,
        "duration_or_span_min_seconds": None,
        "duration_or_span_max_seconds": None,
        "duration_or_span_sum_seconds": 0.0,
    }


def _new_age_summary(key: tuple[str, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "summary_schema_version": DUPLICATE_SUMMARY_SCHEMA_VERSION,
        "summary_kind": "age",
        **dict(zip(AGE_DIMENSIONS, key)),
        "run_count": 0,
        "run_rows": 0,
        "repeat_rows": 0,
    }
    for prefix in (
        "duration",
        "age_delta",
        "duration_minus_age_delta",
    ):
        row[f"{prefix}_count"] = 0
        row[f"{prefix}_min_seconds"] = None
        row[f"{prefix}_max_seconds"] = None
        row[f"{prefix}_sum_seconds"] = 0.0
    return row


def _add_summary_distribution(
    target: dict[str, Any], prefix: str, value: float | None
) -> None:
    if value is None:
        return
    number = float(value)
    if not math.isfinite(number):
        return
    count_key = f"{prefix}_count"
    min_key = f"{prefix}_min_seconds"
    max_key = f"{prefix}_max_seconds"
    sum_key = f"{prefix}_sum_seconds"
    target[count_key] += 1
    target[min_key] = (
        number if target[min_key] is None else min(target[min_key], number)
    )
    target[max_key] = (
        number if target[max_key] is None else max(target[max_key], number)
    )
    target[sum_key] += number


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
) -> BucketDuplicateSummary:
    """Compute exact published statistics without retaining group identities."""
    ordered = sorted((dict(row) for row in rows), key=_row_sort_key)
    airports = sorted({str(row.get("airport_id", "")) for row in ordered})
    duplicate = {
        (airport, record_type): _new_duplicate_summary(airport, record_type)
        for airport in airports
        for record_type in SUMMARY_RECORD_TYPES
    }

    id_timestamp_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            _timestamp(row.get("report_timestamp_utc")),
        ),
    )
    for (airport, _aircraft, _timestamp_value), group in id_timestamp_groups.items():
        count = len(group)
        repeated = count > 1
        states = {_state(row) for row in group}
        source_file_count = _source_file_count(group)
        summary = duplicate[(airport, "id_timestamp")]
        summary["total_groups"] += 1
        summary["total_rows"] += count
        summary["duplicate_groups"] += int(repeated)
        summary["duplicate_rows"] += count if repeated else 0
        summary["excess_rows"] += count - 1 if repeated else 0
        summary["identical_state_groups"] += int(repeated and len(states) == 1)
        summary["conflicting_state_groups"] += int(repeated and len(states) > 1)
        summary["file_boundary_crossing_groups"] += int(source_file_count > 1)
        if source_file_count > 1:
            boundary = duplicate[(airport, "boundary_crossing")]
            boundary["total_groups"] += 1
            boundary["total_rows"] += count

    exact_report_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            _timestamp(row.get("report_timestamp_utc")),
            row.get("lat"),
            row.get("lon"),
            row.get("altitude"),
        ),
    )
    for (airport, *_identity), group in exact_report_groups.items():
        count = len(group)
        repeated = count > 1
        source_file_count = _source_file_count(group)
        summary = duplicate[(airport, "exact_report")]
        summary["total_groups"] += 1
        summary["total_rows"] += count
        summary["duplicate_groups"] += int(repeated)
        summary["duplicate_rows"] += count if repeated else 0
        summary["excess_rows"] += count - 1 if repeated else 0
        summary["file_boundary_crossing_groups"] += int(source_file_count > 1)
        if source_file_count > 1:
            boundary = duplicate[(airport, "boundary_crossing")]
            boundary["total_groups"] += 1
            boundary["total_rows"] += count

    full_state_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            *_state(row),
        ),
    )
    for (airport, *_identity), group in full_state_groups.items():
        timestamps = sorted(
            {_timestamp(row.get("report_timestamp_utc")) for row in group}
        )
        count = len(group)
        repeated = count > 1
        source_file_count = _source_file_count(group)
        span = (timestamps[-1] - timestamps[0]).total_seconds()
        summary = duplicate[(airport, "full_state")]
        summary["total_groups"] += 1
        summary["total_rows"] += count
        summary["repeat_groups"] += int(repeated)
        summary["repeat_rows"] += count if repeated else 0
        summary["excess_rows"] += count - 1 if repeated else 0
        summary["file_boundary_crossing_groups"] += int(source_file_count > 1)
        _add_summary_distribution(summary, "duration_or_span", span)
        if source_file_count > 1:
            boundary = duplicate[(airport, "boundary_crossing")]
            boundary["total_groups"] += 1
            boundary["total_rows"] += count

    age: dict[tuple[str, ...], dict[str, Any]] = {}
    runs = _build_runs(ordered)
    for run_index, run in enumerate(runs):
        first = run[0]
        airport = str(first.get("airport_id", ""))
        aircraft = (airport, str(first.get("aircraft_id", "")))
        timestamps = [_timestamp(row.get("report_timestamp_utc")) for row in run]
        ages = _finite_ages(run)
        source_file_count = _source_file_count(run)
        next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
        if next_run is None or (
            str(next_run[0].get("airport_id", "")),
            str(next_run[0].get("aircraft_id", "")),
        ) != aircraft:
            relationship = "end_of_aircraft"
        elif _timestamp(next_run[0].get("report_timestamp_utc")) == timestamps[-1]:
            relationship = "same_timestamp_update"
        else:
            relationship = "later_timestamp_update"
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        age_min = min(ages) if ages else None
        age_max = max(ages) if ages else None
        age_delta = ages[-1] - ages[0] if len(ages) >= 2 else None

        summary = duplicate[(airport, "contiguous_state_run")]
        summary["total_groups"] += 1
        summary["total_rows"] += len(run)
        summary["repeat_groups"] += int(len(run) > 1)
        summary["repeat_rows"] += len(run) if len(run) > 1 else 0
        summary["file_boundary_crossing_groups"] += int(source_file_count > 1)
        _add_summary_distribution(summary, "duration_or_span", duration)
        if source_file_count > 1:
            boundary = duplicate[(airport, "boundary_crossing")]
            boundary["total_groups"] += 1
            boundary["total_rows"] += len(run)

        age_key = (
            airport,
            _repeat_count_bin(len(run)),
            _duration_bin(duration),
            _age_bin(age_min),
            _age_bin(age_max),
            _age_trend(ages),
            relationship,
        )
        age_summary = age.setdefault(age_key, _new_age_summary(age_key))
        age_summary["run_count"] += 1
        age_summary["run_rows"] += len(run)
        age_summary["repeat_rows"] += len(run) if len(run) > 1 else 0
        _add_summary_distribution(age_summary, "duration", duration)
        _add_summary_distribution(age_summary, "age_delta", age_delta)
        _add_summary_distribution(
            age_summary,
            "duration_minus_age_delta",
            duration - age_delta if age_delta is not None else None,
        )

    duplicate_records = [
        duplicate[(airport, record_type)]
        for airport in airports
        for record_type in SUMMARY_RECORD_TYPES
    ]
    return BucketDuplicateSummary(
        duplicate=tuple(duplicate_records),
        age=tuple(age[key] for key in sorted(age)),
    )


def _build_runs(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    runs: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_aircraft: tuple[str, str] | None = None
    current_state: tuple[Any, ...] | None = None
    for row in rows:
        aircraft = (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
        )
        state = _state(row)
        if current and (aircraft != current_aircraft or state != current_state):
            runs.append(current)
            current = []
        current.append(row)
        current_aircraft = aircraft
        current_state = state
    if current:
        runs.append(current)
    return runs


def analyze_rows(rows: Iterable[Mapping[str, Any]]) -> BucketDuplicateProfile:
    """Sort and exactly profile one externally partitioned aircraft bucket."""
    ordered = sorted((dict(row) for row in rows), key=_row_sort_key)
    id_timestamp_records: list[dict[str, Any]] = []
    exact_report_records: list[dict[str, Any]] = []
    full_state_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    age_repeat_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []

    id_timestamp_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            _timestamp(row.get("report_timestamp_utc")),
        ),
    )
    for (airport_id, aircraft_id, timestamp), group in id_timestamp_groups.items():
        count = len(group)
        duplicate = count > 1
        states = {_state(row) for row in group}
        ages = _finite_ages(group)
        source_file_count = _source_file_count(group)
        selected = min(group, key=selection_key)
        identity = {
            "airport_id": airport_id,
            "aircraft_id": aircraft_id,
            "report_timestamp_utc": timestamp.isoformat(),
        }
        id_timestamp_records.append(
            {
                **identity,
                "row_count": count,
                "total_groups": 1,
                "duplicate_groups": int(duplicate),
                "duplicate_rows": count if duplicate else 0,
                "excess_rows": count - 1 if duplicate else 0,
                "identical_state_groups": int(duplicate and len(states) == 1),
                "conflicting_state_groups": int(duplicate and len(states) > 1),
                "usable_age_monotonic_groups": int(
                    duplicate
                    and len(ages) >= 2
                    and all(left <= right for left, right in zip(ages, ages[1:]))
                ),
                "source_file_count": source_file_count,
                "file_boundary_crossing_groups": int(source_file_count > 1),
                **_selected_provenance(selected),
            }
        )
        boundary = _boundary_record(
            identity_type="id_timestamp", rows=group, identity=identity
        )
        if boundary is not None:
            boundary_records.append(boundary)

    exact_report_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            _timestamp(row.get("report_timestamp_utc")),
            row.get("lat"),
            row.get("lon"),
            row.get("altitude"),
        ),
    )
    for key, group in exact_report_groups.items():
        airport_id, aircraft_id, timestamp, lat, lon, altitude = key
        count = len(group)
        duplicate = count > 1
        source_file_count = _source_file_count(group)
        identity = {
            "airport_id": airport_id,
            "aircraft_id": aircraft_id,
            "report_timestamp_utc": timestamp.isoformat(),
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
        }
        exact_report_records.append(
            {
                **identity,
                "row_count": count,
                "total_groups": 1,
                "duplicate_groups": int(duplicate),
                "duplicate_rows": count if duplicate else 0,
                "excess_rows": count - 1 if duplicate else 0,
                "source_file_count": source_file_count,
                "file_boundary_crossing_groups": int(source_file_count > 1),
            }
        )
        boundary = _boundary_record(
            identity_type="exact_report", rows=group, identity=identity
        )
        if boundary is not None:
            boundary_records.append(boundary)

    full_state_groups = _group_rows(
        ordered,
        lambda row: (
            str(row.get("airport_id", "")),
            str(row.get("aircraft_id", "")),
            *_state(row),
        ),
    )
    for key, group in full_state_groups.items():
        airport_id, aircraft_id, *state = key
        timestamps = sorted(
            {_timestamp(row.get("report_timestamp_utc")) for row in group}
        )
        count = len(group)
        repeat = count > 1
        source_file_count = _source_file_count(group)
        identity = {
            "airport_id": airport_id,
            "aircraft_id": aircraft_id,
            **_state_record(state),
        }
        full_state_records.append(
            {
                **identity,
                "row_count": count,
                "total_groups": 1,
                "repeat_groups": int(repeat),
                "repeat_rows": count if repeat else 0,
                "excess_rows": count - 1 if repeat else 0,
                "distinct_timestamp_count": len(timestamps),
                "first_timestamp_utc": timestamps[0].isoformat(),
                "last_timestamp_utc": timestamps[-1].isoformat(),
                "span_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
                "source_file_count": source_file_count,
                "file_boundary_crossing_groups": int(source_file_count > 1),
            }
        )
        boundary = _boundary_record(
            identity_type="full_state", rows=group, identity=identity
        )
        if boundary is not None:
            boundary_records.append(boundary)

    runs = _build_runs(ordered)
    run_ordinals: dict[tuple[str, str], int] = {}
    for run_index, run in enumerate(runs):
        first = run[0]
        state = _state(first)
        timestamps = [_timestamp(row.get("report_timestamp_utc")) for row in run]
        ages = _finite_ages(run)
        source_file_count = _source_file_count(run)
        aircraft = (
            str(first.get("airport_id", "")),
            str(first.get("aircraft_id", "")),
        )
        next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
        if next_run is None or (
            str(next_run[0].get("airport_id", "")),
            str(next_run[0].get("aircraft_id", "")),
        ) != aircraft:
            state_update_relationship = "end_of_aircraft"
        elif _timestamp(next_run[0].get("report_timestamp_utc")) == timestamps[-1]:
            state_update_relationship = "same_timestamp_update"
        else:
            state_update_relationship = "later_timestamp_update"
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        age_min = min(ages) if ages else None
        age_max = max(ages) if ages else None
        run_identity = {
            "airport_id": aircraft[0],
            "aircraft_id": aircraft[1],
            **_state_record(state),
            "run_ordinal": run_ordinals.get(aircraft, 0),
        }
        run_ordinals[aircraft] = run_identity["run_ordinal"] + 1
        run_record = {
            **run_identity,
            "total_runs": 1,
            "run_rows": len(run),
            "repeat_runs": int(len(run) > 1),
            "repeat_rows": len(run) if len(run) > 1 else 0,
            "distinct_timestamp_count": len(set(timestamps)),
            "first_timestamp_utc": timestamps[0].isoformat(),
            "last_timestamp_utc": timestamps[-1].isoformat(),
            "duration_seconds": duration,
            "age_min": age_min,
            "age_max": age_max,
            "source_file_count": source_file_count,
            "file_boundary_crossing": int(source_file_count > 1),
        }
        run_records.append(run_record)
        age_delta = ages[-1] - ages[0] if len(ages) >= 2 else None
        age_repeat_records.append(
            {
                "airport_id": aircraft[0],
                "repeat_count_bin": _repeat_count_bin(len(run)),
                "duration_bin": _duration_bin(duration),
                "age_min_bin": _age_bin(age_min),
                "age_max_bin": _age_bin(age_max),
                "age_trend": _age_trend(ages),
                "state_update_relationship": state_update_relationship,
                "run_count": 1,
                "run_rows": len(run),
                "repeat_rows": len(run) if len(run) > 1 else 0,
                "duration_seconds": duration,
                "age_delta_seconds": age_delta,
                "duration_minus_age_delta_seconds": (
                    duration - age_delta if age_delta is not None else None
                ),
            }
        )
        boundary = _boundary_record(
            identity_type="contiguous_state_run",
            rows=run,
            identity=run_identity,
        )
        if boundary is not None:
            boundary_records.append(boundary)

    return BucketDuplicateProfile(
        id_timestamp=tuple(id_timestamp_records),
        exact_report=tuple(exact_report_records),
        full_state=tuple(full_state_records),
        contiguous_state_run=tuple(run_records),
        age_repeat=tuple(age_repeat_records),
        boundary_crossing=tuple(boundary_records),
    )
