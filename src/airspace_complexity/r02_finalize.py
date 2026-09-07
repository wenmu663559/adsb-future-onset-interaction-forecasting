"""Deterministically merge independently published R02 airport audits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from airspace_complexity.external_duplicate_audit import (
    DUPLICATE_SUMMARY_SCHEMA_VERSION,
)
from airspace_complexity.r02_resumable import (
    CHECKPOINT_SCHEMA_VERSION,
    AggregateCheckpoint,
    AggregationIdentity,
)


AIRPORTS = ("kagc", "kbtp")
PROFILE_FIELDS = (
    "aircraft_id",
    "report_timestamp_utc",
    "timestamp_assuming_airport_local",
    "local_candidate_ambiguous",
    "local_candidate_nonexistent",
    "lat",
    "lon",
    "altitude",
    "speed",
    "heading",
    "age",
    "range",
    "bearing",
    "tail",
    "altis_gnss",
)
DUPLICATE_RECORD_TYPES = (
    "id_timestamp",
    "exact_report",
    "full_state",
    "contiguous_state_run",
    "boundary_crossing",
)
DUPLICATE_FIELDS = (
    "airport_id",
    "record_type",
    "total_groups",
    "total_rows",
    "duplicate_groups",
    "duplicate_rows",
    "repeat_groups",
    "repeat_rows",
    "excess_rows",
    "identical_state_groups",
    "conflicting_state_groups",
    "file_boundary_crossing_groups",
    "duration_or_span_count",
    "duration_or_span_min_seconds",
    "duration_or_span_max_seconds",
    "duration_or_span_sum_seconds",
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
AGE_FIELDS = (
    *AGE_DIMENSIONS,
    "run_count",
    "run_rows",
    "repeat_rows",
    "duration_count",
    "duration_min_seconds",
    "duration_max_seconds",
    "duration_sum_seconds",
    "age_delta_count",
    "age_delta_min_seconds",
    "age_delta_max_seconds",
    "age_delta_sum_seconds",
    "duration_minus_age_delta_count",
    "duration_minus_age_delta_min_seconds",
    "duration_minus_age_delta_max_seconds",
    "duration_minus_age_delta_sum_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_exact_keys(name: str, mapping: Mapping[str, Any], count: int) -> None:
    expected = {str(index) for index in range(count)}
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual, key=int)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} bucket keys are incomplete: missing={missing}, extra={extra}"
        )


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _add_distribution(
    target: dict[str, Any], prefix: str, value: Any
) -> None:
    number = _finite_float(value)
    if number is None:
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


def _new_duplicate_summary(airport: str, record_type: str) -> dict[str, Any]:
    return {
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
    row: dict[str, Any] = dict(zip(AGE_DIMENSIONS, key))
    row.update(
        {
            "run_count": 0,
            "run_rows": 0,
            "repeat_rows": 0,
            "duration_count": 0,
            "duration_min_seconds": None,
            "duration_max_seconds": None,
            "duration_sum_seconds": 0.0,
            "age_delta_count": 0,
            "age_delta_min_seconds": None,
            "age_delta_max_seconds": None,
            "age_delta_sum_seconds": 0.0,
            "duration_minus_age_delta_count": 0,
            "duration_minus_age_delta_min_seconds": None,
            "duration_minus_age_delta_max_seconds": None,
            "duration_minus_age_delta_sum_seconds": 0.0,
        }
    )
    return row


def _strict_summary_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"compact summary {name} must be a nonnegative integer")
    return value


def _strict_summary_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"compact summary {name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"compact summary {name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"compact summary {name} must be finite")
    return number


def _validate_compact_distribution(
    record: Mapping[str, Any], prefix: str
) -> tuple[int, float | None, float | None, float]:
    count = _strict_summary_count(f"{prefix}_count", record[f"{prefix}_count"])
    total = _strict_summary_number(
        f"{prefix}_sum_seconds", record[f"{prefix}_sum_seconds"]
    )
    minimum = record[f"{prefix}_min_seconds"]
    maximum = record[f"{prefix}_max_seconds"]
    if count == 0:
        if minimum is not None or maximum is not None or total != 0.0:
            raise ValueError(f"compact summary {prefix} distribution is inconsistent")
        return count, None, None, total
    if minimum is None or maximum is None:
        raise ValueError(f"compact summary {prefix} distribution is incomplete")
    low = _strict_summary_number(f"{prefix}_min_seconds", minimum)
    high = _strict_summary_number(f"{prefix}_max_seconds", maximum)
    if low > high:
        raise ValueError(f"compact summary {prefix} distribution is unordered")
    return count, low, high, total


def _merge_compact_distribution(
    target: dict[str, Any], prefix: str, record: Mapping[str, Any]
) -> None:
    count, minimum, maximum, total = _validate_compact_distribution(
        record, prefix
    )
    if count == 0:
        return
    count_key = f"{prefix}_count"
    min_key = f"{prefix}_min_seconds"
    max_key = f"{prefix}_max_seconds"
    sum_key = f"{prefix}_sum_seconds"
    target[count_key] += count
    target[min_key] = (
        minimum
        if target[min_key] is None
        else min(float(target[min_key]), float(minimum))
    )
    target[max_key] = (
        maximum
        if target[max_key] is None
        else max(float(target[max_key]), float(maximum))
    )
    target[sum_key] += total


def _merge_compact_record(
    record: Mapping[str, Any],
    duplicate: dict[tuple[str, str], dict[str, Any]],
    age: dict[tuple[str, ...], dict[str, Any]],
) -> None:
    if record.get("summary_schema_version") != DUPLICATE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unknown compact duplicate summary schema")
    kind = record.get("summary_kind")
    if kind == "duplicate":
        expected = {
            "summary_schema_version",
            "summary_kind",
            *DUPLICATE_FIELDS,
        }
        if set(record) != expected:
            raise ValueError("compact duplicate summary fields mismatch")
        airport = str(record["airport_id"]).lower()
        record_type = str(record["record_type"])
        if record_type not in DUPLICATE_RECORD_TYPES:
            raise ValueError(f"unknown duplicate record type {record_type!r}")
        summary = duplicate[(airport, record_type)]
        for name in (
            "total_groups",
            "total_rows",
            "duplicate_groups",
            "duplicate_rows",
            "repeat_groups",
            "repeat_rows",
            "excess_rows",
            "identical_state_groups",
            "conflicting_state_groups",
            "file_boundary_crossing_groups",
        ):
            summary[name] += _strict_summary_count(name, record[name])
        _merge_compact_distribution(summary, "duration_or_span", record)
        return
    if kind == "age":
        expected = {
            "summary_schema_version",
            "summary_kind",
            *AGE_FIELDS,
        }
        if set(record) != expected:
            raise ValueError("compact age summary fields mismatch")
        key = tuple(str(record[name]) for name in AGE_DIMENSIONS)
        summary = age.setdefault(key, _new_age_summary(key))
        for name in ("run_count", "run_rows", "repeat_rows"):
            summary[name] += _strict_summary_count(name, record[name])
        for prefix in (
            "duration",
            "age_delta",
            "duration_minus_age_delta",
        ):
            _merge_compact_distribution(summary, prefix, record)
        return
    raise ValueError(f"unknown compact summary kind {kind!r}")


def _verify_profiles(
    rows: list[dict[str, Any]], airport_totals: Mapping[str, int]
) -> None:
    required = set(PROFILE_FIELDS)
    grouped: dict[tuple[str, int | None, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        airport = str(row["airport_id"])
        grouped[(airport, row.get("year"), str(row["schema_version"]))].append(
            row
        )
        total = int(row["total"])
        if int(row["missing"]) + int(row["invalid"]) > total:
            raise ValueError(f"profile count overflow for {row['stratum']}")
        if int(row["sample_size"]) > 8192:
            raise ValueError(f"profile sample overflow for {row['stratum']}")
        quantiles = [
            float(row[name])
            for name in ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
            if row.get(name) is not None
        ]
        if quantiles != sorted(quantiles):
            raise ValueError(f"profile quantiles are not monotonic: {row['stratum']}")

    for airport, expected_total in airport_totals.items():
        total_rows = grouped.get((airport, None, "ALL"), [])
        if {str(row["field"]) for row in total_rows} != required:
            raise ValueError(f"airport-total profile coverage failure: {airport}")
        if any(int(row["total"]) != expected_total for row in total_rows):
            raise ValueError(f"airport-total profile row mismatch: {airport}")
        detailed = [
            (key, values)
            for key, values in grouped.items()
            if key[0] == airport and not (key[1] is None and key[2] == "ALL")
        ]
        if not detailed:
            raise ValueError(f"missing detailed profile strata: {airport}")
        for key, values in detailed:
            if {str(row["field"]) for row in values} != required:
                raise ValueError(f"detailed profile coverage failure: {key}")
        for field in required:
            detailed_total = sum(
                int(row["total"])
                for _, values in detailed
                for row in values
                if row["field"] == field
            )
            if detailed_total != expected_total:
                raise ValueError(
                    f"detailed profile conservation failure: {airport}/{field}"
                )


def _verify_timestamp_source(
    timestamp_source: Path,
    expected_sources: Mapping[tuple[str, str], int],
) -> bytes:
    payload = timestamp_source.read_bytes()
    with timestamp_source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed: dict[tuple[str, str], int] = {}
    parse_errors = 0
    for row in rows:
        key = (str(row["airport_id"]).lower(), str(row["relative_path"]))
        if key in observed:
            raise ValueError(f"duplicate timestamp source row: {key}")
        observed[key] = int(row["rows"])
        parse_errors += int(row["timestamp_parse_errors"])
    if observed != dict(expected_sources):
        raise ValueError("timestamp source coverage or row counts mismatch")
    if parse_errors:
        raise ValueError(f"timestamp source contains {parse_errors} parse errors")
    return payload


def _verify_part_artifacts(
    part_manifest: Mapping[str, Any], airport: str
) -> dict[str, Path]:
    if str(part_manifest["airport"]).lower() != airport:
        raise ValueError(f"part airport mismatch: {airport}")
    if int(part_manifest["bucket_count"]) != 256:
        raise ValueError(f"part bucket count mismatch: {airport}")
    invariants = dict(part_manifest["invariants"])
    if not invariants or not all(bool(value) for value in invariants.values()):
        raise ValueError(f"part invariant failure: {airport}")
    artifacts = {
        str(name): Path(path)
        for name, path in dict(part_manifest["artifact_files"]).items()
    }
    expected_hashes = {
        str(name): str(value)
        for name, value in dict(part_manifest["output_hashes"]).items()
    }
    required = {
        "column_profile",
        "duplicate_records",
        "invariants",
        "output_hashes",
    }
    if set(artifacts) != required or set(expected_hashes) != required:
        raise ValueError(f"part artifact set mismatch: {airport}")
    for name, path in artifacts.items():
        actual = _sha256(path)
        if actual != expected_hashes[name]:
            raise ValueError(
                f"part artifact hash mismatch: {airport}/{name}: "
                f"{actual} != {expected_hashes[name]}"
            )
    return artifacts


def _verify_resumption_evidence(
    part_manifest: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
    airport: str,
) -> dict[str, Any] | None:
    raw = part_manifest.get("resumption")
    if raw is None:
        return None
    evidence = dict(raw)
    expected_fields = {
        "schema_version",
        "aggregation_identity",
        "checkpoint_file",
        "checkpoint_sha256",
        "completed_bucket_count",
        "reused_bucket_count",
        "newly_processed_bucket_count",
        "fragment_root",
    }
    if set(evidence) != expected_fields:
        raise ValueError(f"resumption evidence fields mismatch: {airport}")
    if evidence["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"resumption schema mismatch: {airport}")
    identity = AggregationIdentity.from_dict(
        evidence["aggregation_identity"]
    )
    partition_identity = dict(partition_manifest["identity"])
    if dict(identity.partition_identity) != partition_identity:
        raise ValueError(f"resumption partition identity mismatch: {airport}")
    partition_path = (
        Path(partition_manifest["work_dir"]) / "partition_manifest.json"
    )
    if _sha256(partition_path) != identity.partition_manifest_sha256:
        raise ValueError(f"resumption partition hash mismatch: {airport}")
    checkpoint_path = Path(evidence["checkpoint_file"])
    if _sha256(checkpoint_path) != evidence["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {airport}")
    checkpoint = AggregateCheckpoint.from_path(
        checkpoint_path, expected_identity=identity
    )
    bucket_count = int(partition_manifest["bucket_count"])
    expected_keys = {str(index) for index in range(bucket_count)}
    if (
        checkpoint.airport != airport
        or checkpoint.bucket_count != bucket_count
        or set(checkpoint.completed_buckets) != expected_keys
        or int(evidence["completed_bucket_count"]) != bucket_count
    ):
        raise ValueError(f"resumption checkpoint coverage mismatch: {airport}")
    expected_rows = int(partition_manifest["total_rows"])
    if checkpoint.observed_rows != expected_rows:
        raise ValueError(f"resumption checkpoint row mismatch: {airport}")
    nonempty = sum(
        int(int(value) > 0)
        for value in partition_manifest["bucket_row_counts"].values()
    )
    if checkpoint.nonempty_bucket_count != nonempty:
        raise ValueError(f"resumption nonempty bucket mismatch: {airport}")
    reused = int(evidence["reused_bucket_count"])
    processed = int(evidence["newly_processed_bucket_count"])
    if reused < 0 or processed < 0 or reused + processed != bucket_count:
        raise ValueError(f"resumption bucket accounting mismatch: {airport}")
    fragment_root = Path(evidence["fragment_root"])
    if not fragment_root.is_dir():
        raise ValueError(f"resumption fragment root missing: {airport}")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": str(evidence["checkpoint_sha256"]),
        "completed_bucket_count": bucket_count,
        "reused_bucket_count": reused,
        "newly_processed_bucket_count": processed,
        "aggregation_identity": identity.to_dict(),
        "partition_identity_match": True,
        "fragment_root": str(fragment_root),
    }


def _summarize_duplicate_records(
    airport_artifacts: Mapping[str, Mapping[str, Path]],
    airport_totals: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate = {
        (airport, record_type): _new_duplicate_summary(airport, record_type)
        for airport in AIRPORTS
        for record_type in DUPLICATE_RECORD_TYPES
    }
    age: dict[tuple[str, ...], dict[str, Any]] = {}
    for airport in AIRPORTS:
        path = airport_artifacts[airport]["duplicate_records"]
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid duplicate record {path}:{line_number}"
                    ) from exc
                if str(record.get("airport_id", "")).lower() != airport:
                    raise ValueError(
                        f"duplicate record airport mismatch: {path}:{line_number}"
                    )
                if "summary_schema_version" in record:
                    _merge_compact_record(record, duplicate, age)
                    continue
                record_type = str(record.get("record_type"))
                if record_type == "age_repeat":
                    key = tuple(str(record.get(name, "")) for name in AGE_DIMENSIONS)
                    summary = age.setdefault(key, _new_age_summary(key))
                    for name in ("run_count", "run_rows", "repeat_rows"):
                        summary[name] += int(record.get(name, 0))
                    _add_distribution(
                        summary, "duration", record.get("duration_seconds")
                    )
                    _add_distribution(
                        summary, "age_delta", record.get("age_delta_seconds")
                    )
                    _add_distribution(
                        summary,
                        "duration_minus_age_delta",
                        record.get("duration_minus_age_delta_seconds"),
                    )
                    continue
                if record_type not in DUPLICATE_RECORD_TYPES:
                    raise ValueError(
                        f"unknown duplicate record type {record_type!r}"
                    )
                summary = duplicate[(airport, record_type)]
                if record_type in {"id_timestamp", "exact_report", "full_state"}:
                    summary["total_groups"] += int(record.get("total_groups", 0))
                    summary["total_rows"] += int(record.get("row_count", 0))
                    summary["duplicate_groups"] += int(
                        record.get("duplicate_groups", 0)
                    )
                    summary["duplicate_rows"] += int(
                        record.get("duplicate_rows", 0)
                    )
                    summary["repeat_groups"] += int(record.get("repeat_groups", 0))
                    summary["repeat_rows"] += int(record.get("repeat_rows", 0))
                    summary["excess_rows"] += int(record.get("excess_rows", 0))
                    summary["identical_state_groups"] += int(
                        record.get("identical_state_groups", 0)
                    )
                    summary["conflicting_state_groups"] += int(
                        record.get("conflicting_state_groups", 0)
                    )
                    summary["file_boundary_crossing_groups"] += int(
                        record.get("file_boundary_crossing_groups", 0)
                    )
                    _add_distribution(
                        summary,
                        "duration_or_span",
                        record.get("span_seconds"),
                    )
                elif record_type == "contiguous_state_run":
                    summary["total_groups"] += int(record.get("total_runs", 0))
                    summary["total_rows"] += int(record.get("run_rows", 0))
                    summary["repeat_groups"] += int(record.get("repeat_runs", 0))
                    summary["repeat_rows"] += int(record.get("repeat_rows", 0))
                    summary["file_boundary_crossing_groups"] += int(
                        record.get("file_boundary_crossing", 0)
                    )
                    _add_distribution(
                        summary,
                        "duration_or_span",
                        record.get("duration_seconds"),
                    )
                else:
                    summary["total_groups"] += int(record.get("crossing_groups", 0))
                    summary["total_rows"] += int(record.get("crossing_rows", 0))

    for airport, expected_total in airport_totals.items():
        for record_type in (
            "id_timestamp",
            "exact_report",
            "full_state",
            "contiguous_state_run",
        ):
            if duplicate[(airport, record_type)]["total_rows"] != expected_total:
                raise ValueError(
                    f"duplicate row conservation failure: {airport}/{record_type}"
                )
        airport_age = [
            row for key, row in age.items() if key[0] == airport
        ]
        if sum(int(row["run_rows"]) for row in airport_age) != expected_total:
            raise ValueError(f"age/run row conservation failure: {airport}")
        if sum(int(row["run_count"]) for row in airport_age) != duplicate[
            (airport, "contiguous_state_run")
        ]["total_groups"]:
            raise ValueError(f"age/run group conservation failure: {airport}")

    duplicate_rows = [
        duplicate[(airport, record_type)]
        for airport in AIRPORTS
        for record_type in DUPLICATE_RECORD_TYPES
    ]
    age_rows = [age[key] for key in sorted(age)]
    return duplicate_rows, age_rows


def finalize_r02(
    *,
    parts_root: Path,
    work_root: Path,
    output_dir: Path,
    timestamp_source: Path,
    discovery_summary: Path,
    expected_source_count: int = 1911,
    expected_total_rows: int = 62_385_317,
    expected_zero_row_sources: int = 35,
) -> dict[str, Any]:
    """Validate two exact airport audits and publish conservative R02 outputs."""
    started_at = _utc_now()
    parts = Path(parts_root).resolve()
    work = Path(work_root).resolve()
    output = Path(output_dir).resolve()
    discovery_path = Path(discovery_summary).resolve()
    timestamp_path = Path(timestamp_source).resolve()

    part_manifests: dict[str, dict[str, Any]] = {}
    partition_manifests: dict[str, dict[str, Any]] = {}
    airport_artifacts: dict[str, dict[str, Path]] = {}
    airport_resumption: dict[str, dict[str, Any] | None] = {}
    common_identity: dict[str, Any] | None = None
    airport_totals: dict[str, int] = {}
    expected_timestamp_sources: dict[tuple[str, str], int] = {}
    source_count = 0
    zero_row_sources = 0
    global_source_ids: set[str] = set()

    for airport in AIRPORTS:
        part = _load_json(parts / airport / "run_manifest.json")
        partition = _load_json(work / airport / "partition_manifest.json")
        part_identity = dict(part["identity"])
        partition_identity = dict(partition["identity"])
        if part_identity != partition_identity:
            raise ValueError(f"part/partition identity mismatch: {airport}")
        if common_identity is None:
            common_identity = part_identity
        elif part_identity != common_identity:
            raise ValueError(f"airport identity mismatch: {airport}")
        if str(partition["airport"]).lower() != airport:
            raise ValueError(f"partition airport mismatch: {airport}")
        if int(partition["bucket_count"]) != 256:
            raise ValueError(f"partition bucket count mismatch: {airport}")
        for name in ("bucket_files", "bucket_hashes", "bucket_row_counts"):
            _require_exact_keys(
                f"{airport}/{name}", dict(partition[name]), 256
            )
        total = int(partition["total_rows"])
        if sum(int(value) for value in partition["bucket_row_counts"].values()) != total:
            raise ValueError(f"partition bucket conservation failure: {airport}")
        sources = list(partition["sources"])
        if [int(source["source_sequence"]) for source in sources] != list(
            range(len(sources))
        ):
            raise ValueError(f"source sequence failure: {airport}")
        ids = [str(source["source_file_id"]) for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError(f"source ID collision: {airport}")
        overlap = global_source_ids.intersection(ids)
        if overlap:
            raise ValueError(
                f"global source ID collision: {sorted(overlap)}"
            )
        global_source_ids.update(ids)
        if sum(int(source["row_count"]) for source in sources) != total:
            raise ValueError(f"source row conservation failure: {airport}")
        for source in sources:
            key = (airport, str(source["relative_path"]))
            if key in expected_timestamp_sources:
                raise ValueError(f"source path collision: {key}")
            rows = int(source["row_count"])
            expected_timestamp_sources[key] = rows
            zero_row_sources += int(rows == 0)
        source_count += len(sources)
        airport_totals[airport] = total
        airport_artifacts[airport] = _verify_part_artifacts(part, airport)
        airport_resumption[airport] = _verify_resumption_evidence(
            part, partition, airport
        )
        part_manifests[airport] = part
        partition_manifests[airport] = partition

    assert common_identity is not None
    if source_count != expected_source_count:
        raise ValueError(
            f"source count mismatch: {source_count} != {expected_source_count}"
        )
    if sum(airport_totals.values()) != expected_total_rows:
        raise ValueError(
            f"total row mismatch: {sum(airport_totals.values())} "
            f"!= {expected_total_rows}"
        )
    if zero_row_sources != expected_zero_row_sources:
        raise ValueError(
            f"zero-row source mismatch: {zero_row_sources} "
            f"!= {expected_zero_row_sources}"
        )

    discovery = _load_json(discovery_path)
    if int(discovery.get("selected_files", -1)) != 40:
        raise ValueError("discovery audit did not cover exactly 40 files")
    timestamp_payload = _verify_timestamp_source(
        timestamp_path, expected_timestamp_sources
    )

    tables = [
        pq.read_table(airport_artifacts[airport]["column_profile"])
        for airport in AIRPORTS
    ]
    column_table = pa.concat_tables(tables)
    column_rows = column_table.to_pylist()
    _verify_profiles(column_rows, airport_totals)
    duplicate_rows, age_rows = _summarize_duplicate_records(
        airport_artifacts, airport_totals
    )

    output.mkdir(parents=True, exist_ok=True)
    column_path = output / "column_profile.parquet"
    duplicate_path = output / "duplicate_profile.csv"
    age_path = output / "age_profile.csv"
    timestamp_output = output / "timestamp_anomalies.csv"
    _atomic_parquet(column_path, column_table)
    _atomic_csv(duplicate_path, DUPLICATE_FIELDS, duplicate_rows)
    _atomic_csv(age_path, AGE_FIELDS, age_rows)
    _atomic_bytes(timestamp_output, timestamp_payload)

    data_hashes = {
        path.name: _sha256(path)
        for path in (
            age_path,
            column_path,
            duplicate_path,
            timestamp_output,
        )
    }
    all_part_invariants = all(
        all(bool(value) for value in part["invariants"].values())
        for part in part_manifests.values()
    )
    invariants = {
        "computational_checks_passed": True,
        "all_gate_checks_passed": False,
        "airport_identity_match": True,
        "part_partition_identity_match": True,
        "part_output_hashes_verified": True,
        "part_invariants_passed": all_part_invariants,
        "source_count": source_count,
        "expected_source_count": expected_source_count,
        "zero_row_source_count": zero_row_sources,
        "expected_zero_row_source_count": expected_zero_row_sources,
        "total_rows": sum(airport_totals.values()),
        "expected_total_rows": expected_total_rows,
        "bucket_count_per_airport": 256,
        "three_key_row_conservation": True,
        "contiguous_run_row_conservation": True,
        "age_run_row_conservation": True,
        "field_profile_strata_conservation": True,
        "timestamp_source_coverage": True,
        "timestamp_parse_errors": 0,
        "discovery_selected_files": 40,
        "final_independent_review_complete": False,
    }
    manifest = {
        "task": "R02",
        "schema_version": "r02_exact_three_key_gate_v1",
        "identity": common_identity,
        "finalizer": {
            "identity_mode": "content_hash",
            "source_file": str(Path(__file__).resolve()),
            "source_sha256": _sha256(Path(__file__).resolve()),
            "cli_file": str(
                Path(__file__).resolve().parents[2]
                / "scripts/finalize_r02_remediation.py"
            ),
            "cli_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "scripts/finalize_r02_remediation.py"
            ),
        },
        "airports": {
            airport: {
                "partition_manifest": str(
                    work / airport / "partition_manifest.json"
                ),
                "partition_manifest_sha256": _sha256(
                    work / airport / "partition_manifest.json"
                ),
                "part_run_manifest": str(
                    parts / airport / "run_manifest.json"
                ),
                "part_run_manifest_sha256": _sha256(
                    parts / airport / "run_manifest.json"
                ),
                "source_count": len(partition_manifests[airport]["sources"]),
                "zero_row_source_count": sum(
                    int(int(source["row_count"]) == 0)
                    for source in partition_manifests[airport]["sources"]
                ),
                "retained_rows": airport_totals[airport],
                "bucket_count": 256,
                **(
                    {"resumption": airport_resumption[airport]}
                    if airport_resumption[airport] is not None
                    else {}
                ),
            }
            for airport in AIRPORTS
        },
        "discovery": {
            "summary_file": str(discovery_path),
            "summary_sha256": _sha256(discovery_path),
            "selected_files": int(discovery["selected_files"]),
            "sampled_rows": int(discovery["total_rows"]),
        },
        "invariants": invariants,
        "artifact_hashes": data_hashes,
        "hash_manifest_self_exclusion": (
            "output_hashes.json is the terminal hash manifest and cannot "
            "contain its own SHA-256 without recursion"
        ),
        "commands": {
            "aggregate_kagc": (
                "python scripts/run_r02_remediation.py --stage aggregate "
                "--airport kagc --work-dir "
                f'"{work}" --output-dir "{parts / "kagc"}" --bucket-count 256'
            ),
            "aggregate_kbtp": (
                "python scripts/run_r02_remediation.py --stage aggregate "
                "--airport kbtp --work-dir "
                f'"{work}" --output-dir "{parts / "kbtp"}" --bucket-count 256'
            ),
            "finalize": (
                "python scripts/finalize_r02_remediation.py "
                f'--parts-root "{parts}" --work-root "{work}" '
                f'--output-dir "{output}" '
                f'--timestamp-source "{timestamp_path}" '
                f'--discovery-summary "{discovery_path}"'
            ),
            "verify_artifacts": "python scripts/verify_r02_artifacts.py",
            "verify_tests": "python -m pytest -q",
        },
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
    }
    manifest_path = output / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path)

    decision = {
        "task": "R02",
        "decision": "REVISE_R02",
        "r02_gate_passed": False,
        "model_training_authorized": False,
        "passed_checks": [
            "forty_file_discovery_audit",
            "matching_code_config_input_identities",
            "all_1911_authorized_sources_represented",
            "all_35_verified_zero_row_sources_represented",
            "retained_row_conservation",
            "all_256_buckets_per_airport_verified",
            "three_exact_airport_safe_key_profiles",
            "cross_file_boundary_and_repeat_duration_profiles",
            "age_repeat_relationship_profile",
            "field_profile_unique_quantile_and_strata_coverage",
            "final_output_hash_verification",
        ],
        "failed_checks": ["independent_whole_branch_review_pending"],
        "blocking_issues": [
            "Independent whole-branch review has not yet approved the R02 gate."
        ],
        "key_output_files": [
            "outputs/r02/column_profile.parquet",
            "outputs/r02/duplicate_profile.csv",
            "outputs/r02/age_profile.csv",
            "outputs/r02/timestamp_anomalies.csv",
            "outputs/r02/run_manifest.json",
            "outputs/r02/output_hashes.json",
            "reports/REPORT_R02_SCHEMA_AND_TIME.md",
        ],
        "input_manifest_sha256": common_identity["hashes"].get("raw_manifest"),
        "run_manifest_sha256": manifest_hash,
        "reason": (
            "The computational R02 data gate passes with exact identities and "
            "zero unexplained discrepancies; the decision remains REVISE_R02 "
            "until independent whole-branch review is complete."
        ),
    }
    decision_path = output / "decision.json"
    _atomic_json(decision_path, decision)
    final_hashes = {
        **data_hashes,
        manifest_path.name: manifest_hash,
        decision_path.name: _sha256(decision_path),
    }
    _atomic_json(output / "output_hashes.json", final_hashes)
    return decision


def verify_r02_outputs(
    output_dir: Path,
    *,
    expected_source_count: int = 1911,
    expected_total_rows: int = 62_385_317,
    expected_zero_row_sources: int = 35,
) -> dict[str, Any]:
    """Independently verify the published R02 artifact set and conservation."""
    output = Path(output_dir).resolve()
    pre_review_required = {
        "age_profile.csv",
        "column_profile.parquet",
        "decision.json",
        "duplicate_profile.csv",
        "run_manifest.json",
        "timestamp_anomalies.csv",
    }
    hashes = _load_json(output / "output_hashes.json")
    approved_required = {*pre_review_required, "independent_review.json"}
    if set(hashes) not in (pre_review_required, approved_required):
        raise ValueError(
            "output hash manifest set mismatch: "
            f"{set(hashes)} not in "
            f"({pre_review_required}, {approved_required})"
        )
    for name, expected in hashes.items():
        actual = _sha256(output / name)
        if actual != expected:
            raise ValueError(
                f"output hash mismatch for {name}: {actual} != {expected}"
            )

    manifest = _load_json(output / "run_manifest.json")
    decision = _load_json(output / "decision.json")
    fixed_decision_fields = {
        "task",
        "decision",
        "r02_gate_passed",
        "model_training_authorized",
        "passed_checks",
        "failed_checks",
        "blocking_issues",
        "key_output_files",
        "input_manifest_sha256",
        "run_manifest_sha256",
        "reason",
    }
    if set(decision) != fixed_decision_fields:
        raise ValueError("decision schema mismatch")
    if decision["run_manifest_sha256"] != hashes["run_manifest.json"]:
        raise ValueError("decision/run-manifest hash mismatch")
    invariants = dict(manifest["invariants"])
    if (
        invariants["source_count"] != expected_source_count
        or invariants["expected_source_count"] != expected_source_count
        or invariants["total_rows"] != expected_total_rows
        or invariants["expected_total_rows"] != expected_total_rows
        or invariants["zero_row_source_count"] != expected_zero_row_sources
        or invariants["expected_zero_row_source_count"]
        != expected_zero_row_sources
    ):
        raise ValueError("run-manifest count invariant mismatch")
    if invariants["computational_checks_passed"] is not True:
        raise ValueError("run-manifest computational gate failed")
    if decision["decision"] == "REVISE_R02":
        if (
            decision["r02_gate_passed"] is not False
            or decision["model_training_authorized"] is not False
            or invariants["all_gate_checks_passed"] is not False
            or invariants["final_independent_review_complete"] is not False
            or set(hashes) != pre_review_required
        ):
            raise ValueError("pre-review decision is not conservative")
    elif decision["decision"] == "PROCEED_TO_R03":
        if (
            decision["r02_gate_passed"] is not True
            or decision["model_training_authorized"] is not False
            or invariants["all_gate_checks_passed"] is not True
            or invariants["final_independent_review_complete"] is not True
            or set(hashes) != approved_required
            or decision["failed_checks"]
            or decision["blocking_issues"]
        ):
            raise ValueError("approved R02 decision is inconsistent")
        review = _load_json(output / "independent_review.json")
        recorded_review = dict(manifest.get("independent_review", {}))
        if (
            recorded_review.get("record_sha256")
            != hashes["independent_review.json"]
            or recorded_review.get("review_commit") != review["review_commit"]
            or recorded_review.get("reviewed_output_hashes_sha256")
            != review["reviewed_output_hashes_sha256"]
        ):
            raise ValueError("independent review binding mismatch")
    else:
        raise ValueError(f"unknown R02 decision: {decision['decision']}")

    profile = pq.read_table(output / "column_profile.parquet").to_pylist()
    airport_totals = {
        airport: int(manifest["airports"][airport]["retained_rows"])
        for airport in AIRPORTS
    }
    if sum(airport_totals.values()) != expected_total_rows:
        raise ValueError("airport total mismatch")
    _verify_profiles(profile, airport_totals)

    with (output / "duplicate_profile.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        duplicate_rows = list(csv.DictReader(handle))
    by_key = {
        (row["airport_id"], row["record_type"]): row
        for row in duplicate_rows
    }
    for airport, total in airport_totals.items():
        for record_type in (
            "id_timestamp",
            "exact_report",
            "full_state",
            "contiguous_state_run",
        ):
            if int(by_key[(airport, record_type)]["total_rows"]) != total:
                raise ValueError(
                    f"published duplicate conservation failure: "
                    f"{airport}/{record_type}"
                )

    with (output / "age_profile.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        age_rows = list(csv.DictReader(handle))
    for airport, total in airport_totals.items():
        if (
            sum(
                int(row["run_rows"])
                for row in age_rows
                if row["airport_id"] == airport
            )
            != total
        ):
            raise ValueError(f"published age conservation failure: {airport}")

    with (output / "timestamp_anomalies.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        timestamp_rows = list(csv.DictReader(handle))
    if len(timestamp_rows) != expected_source_count:
        raise ValueError("published timestamp source count mismatch")
    if sum(int(row["rows"]) for row in timestamp_rows) != expected_total_rows:
        raise ValueError("published timestamp row conservation failure")
    if sum(int(row["timestamp_parse_errors"]) for row in timestamp_rows):
        raise ValueError("published timestamp parse errors are nonzero")

    return {
        "verified_hash_count": len(hashes),
        "source_count": expected_source_count,
        "total_rows": expected_total_rows,
        "zero_row_sources": expected_zero_row_sources,
        "decision": decision["decision"],
    }


def approve_r02_review(
    output_dir: Path,
    review_record: Path,
    *,
    expected_source_count: int = 1911,
    expected_total_rows: int = 62_385_317,
    expected_zero_row_sources: int = 35,
) -> dict[str, Any]:
    """Bind an independent approval record and authorize only R03."""
    output = Path(output_dir).resolve()
    review_path = Path(review_record).resolve()
    verify_r02_outputs(
        output,
        expected_source_count=expected_source_count,
        expected_total_rows=expected_total_rows,
        expected_zero_row_sources=expected_zero_row_sources,
    )
    decision_path = output / "decision.json"
    manifest_path = output / "run_manifest.json"
    decision = _load_json(decision_path)
    if decision["decision"] != "REVISE_R02":
        raise ValueError("R02 output is not awaiting independent review")

    review = _load_json(review_path)
    required_review_fields = {
        "task",
        "review_decision",
        "reviewed_output_hashes_sha256",
        "review_commit",
        "test_command",
        "test_exit_code",
        "critical_findings",
        "important_findings",
        "reviewed_at_utc",
    }
    if set(review) != required_review_fields:
        raise ValueError("independent review schema mismatch")
    if review["task"] != "R02" or review["review_decision"] != "APPROVE":
        raise ValueError("independent review did not approve R02")
    if (
        int(review["test_exit_code"]) != 0
        or list(review["critical_findings"])
        or list(review["important_findings"])
    ):
        raise ValueError("independent review contains blocking evidence")
    if not str(review["review_commit"]).strip():
        raise ValueError("independent review commit is missing")
    if not str(review["test_command"]).strip():
        raise ValueError("independent review test command is missing")
    reviewed_hash = _sha256(output / "output_hashes.json")
    if review["reviewed_output_hashes_sha256"] != reviewed_hash:
        raise ValueError(
            "reviewed output hash mismatch: "
            f"{review['reviewed_output_hashes_sha256']} != {reviewed_hash}"
        )

    review_output = output / "independent_review.json"
    _atomic_bytes(review_output, review_path.read_bytes())
    review_record_hash = _sha256(review_output)

    manifest = _load_json(manifest_path)
    manifest["invariants"]["all_gate_checks_passed"] = True
    manifest["invariants"]["final_independent_review_complete"] = True
    manifest["independent_review"] = {
        "record_file": str(review_output),
        "record_sha256": review_record_hash,
        "reviewed_output_hashes_sha256": reviewed_hash,
        "review_commit": review["review_commit"],
        "test_command": review["test_command"],
        "test_exit_code": 0,
        "reviewed_at_utc": review["reviewed_at_utc"],
    }
    _atomic_json(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path)

    decision["decision"] = "PROCEED_TO_R03"
    decision["r02_gate_passed"] = True
    decision["model_training_authorized"] = False
    if "independent_whole_branch_review" not in decision["passed_checks"]:
        decision["passed_checks"].append("independent_whole_branch_review")
    decision["failed_checks"] = []
    decision["blocking_issues"] = []
    if (
        "outputs/r02/independent_review.json"
        not in decision["key_output_files"]
    ):
        decision["key_output_files"].append(
            "outputs/r02/independent_review.json"
        )
    decision["run_manifest_sha256"] = manifest_hash
    decision["reason"] = (
        "The exact computational R02 data gate and independent whole-branch "
        "review pass. R03 canonical point construction is authorized; model "
        "training remains prohibited until the later dataset, target, and "
        "sealed-split gates pass."
    )
    _atomic_json(decision_path, decision)

    final_names = (
        "age_profile.csv",
        "column_profile.parquet",
        "decision.json",
        "duplicate_profile.csv",
        "independent_review.json",
        "run_manifest.json",
        "timestamp_anomalies.csv",
    )
    _atomic_json(
        output / "output_hashes.json",
        {name: _sha256(output / name) for name in final_names},
    )
    verify_r02_outputs(
        output,
        expected_source_count=expected_source_count,
        expected_total_rows=expected_total_rows,
        expected_zero_row_sources=expected_zero_row_sources,
    )
    return decision
