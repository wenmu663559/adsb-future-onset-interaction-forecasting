"""Pure capacity arithmetic and bounded observation for R03 Task 8A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from fractions import Fraction
from contextlib import contextmanager
import hashlib
from itertools import groupby
import json
import os
import platform
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Iterable, Mapping, Sequence
import uuid

import pyarrow
import pyarrow.parquet as pq

from airspace_complexity.build_canonical_points import (
    R03RunIdentity,
    build_canonical_points,
    verify_canonical_manifest,
)
from airspace_complexity.canonical_points import (
    BadRow,
    CanonicalPolicies,
    COMPLETENESS_FIELDS,
    CORE_STATE_FIELDS,
    METADATA_FIELDS,
    ParsedRow,
    R03_POLICY_VERSION,
    R03RunParameters,
    age_rank,
    canonical_report_id,
    convert_record,
    membership_schema,
    parsed_rows_schema,
    raw_row_id,
    resolve_group,
    stable_source_order,
    unique_reports_schema,
)
from airspace_complexity.r02_runner import _load_airport_roots
from airspace_complexity.r03_runner import (
    R03SourceRegistry,
    _bad_rows_schema,
    _conflict_schema,
    _iter_mappings,
    enumerate_selected_source,
    verify_source,
)
from airspace_complexity.r03_selection import (
    CapacitySelectionManifest,
    build_capacity_selection,
    capacity_selection_bytes,
    load_and_verify_capacity_selection,
)
from airspace_complexity.r03_capacity_terminal import (
    CleanupFailure,
    TerminalExpectations,
    TerminalJournal,
    TerminalState,
    classify_terminal_shape,
    canonical_terminal_journal_bytes,
    committed_terminal_envelope_sha256,
    derive_terminal_journal_id,
    load_terminal_journal,
    replace_terminal_journal,
)


_AIRPORTS = ("KAGC", "KBTP")
_COUNT_FIELDS = frozenset(
    {"enumerated", "parsed", "bad", "unique", "membership", "conflict"}
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_version",
        "topology",
        "worker_count",
        "worker_peak_rss_bytes",
        "worker_p99_peak_rss_bytes",
        "coordinator_peak_rss_bytes",
        "process_peak_rss_bytes",
        "physical_ram_bytes",
        "wall_time_seconds",
        "rows_per_second",
        "free_bytes_before_run",
        "free_bytes_at_disk_high_water",
        "scratch_staging_high_water_bytes",
        "total_finalized_bytes",
        "finalized_artifacts",
        "finalized_bytes_by_role",
        "finalized_bytes_by_airport",
        "counts_total",
        "counts_by_airport",
        "counts_by_stratum",
        "sample_events",
    }
)
_REQUIRED_EVENTS = frozenset(
    {"periodic", "shard_seal", "merge_complete", "final_promotion"}
)
_RETAINED_EVIDENCE = (
    "capacity_selection_manifest.json",
    "capacity_metrics.json",
    "capacity_verifier_result.json",
    "command_environment.json",
    "hashes.json",
)
_HASHED_EVIDENCE = _RETAINED_EVIDENCE[:-1]
_COMMAND_ENVIRONMENT_FIELDS = frozenset({
    "schema_version", "command_argv", "started_utc", "ended_utc",
    "project_root", "external_root", "capacity_gate_code_sha256",
    "generator_code_hash", "runtime_environment_hash", "python_version",
    "platform", "pyarrow_version", "dependency_lock_sha256",
    "worker_count", "run_parameters",
})
_VERIFIER_EXTRA_FIELDS = frozenset({
    "free_bytes_after_cleanup", "ledger_verification", "metrics_verification",
    "pilot_disk_capacity", "pilot_memory_capacity", "full_disk_capacity",
    "full_memory_capacity", "retained_attempt",
})
_METRICS_FIELDS = frozenset({
    "schema_version", "measurement_status", "measurements", "projection",
    "probe", "performance_review_required", "ledger_verifier_result",
    "semantic_attestation", "build_failure",
})


@dataclass(frozen=True)
class CapacityGateRecord:
    schema_version: str
    capacity_selection_id: str
    build_started: bool
    measurement_status: str
    pilot_capacity_pass: bool
    full_capacity_ready: bool
    performance_review_required: bool
    cleanup_status: str
    decision: str
    is_full_population: bool
    r04_authorized: bool
    model_training_authorized: bool
    paper_main_result_authorized: bool


@dataclass(frozen=True)
class CapacityProbe:
    estimated_peak_bytes: int
    required_free_bytes: int
    available_free_bytes: int
    passed: bool


@dataclass(frozen=True)
class FinalizedArtifact:
    artifact_role: str
    relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class CapacitySample:
    event: str
    monotonic_seconds: Fraction
    scratch_staging_bytes: int
    free_bytes: int
    process_peak_rss_bytes: int


@dataclass(frozen=True)
class CapacityMeasurements:
    schema_version: str
    topology: str
    worker_count: int
    worker_peak_rss_bytes: None
    worker_p99_peak_rss_bytes: None
    coordinator_peak_rss_bytes: None
    process_peak_rss_bytes: int
    physical_ram_bytes: int
    wall_time_seconds: Fraction
    rows_per_second: Fraction
    free_bytes_before_run: int
    free_bytes_at_disk_high_water: int
    scratch_staging_high_water_bytes: int
    total_finalized_bytes: int
    finalized_artifacts: tuple[FinalizedArtifact, ...]
    finalized_bytes_by_role: tuple[tuple[str, int], ...]
    finalized_bytes_by_airport: tuple[tuple[str, int], ...]
    counts_total: tuple[tuple[str, int], ...]
    counts_by_airport: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    counts_by_stratum: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    sample_events: tuple[CapacitySample, ...]

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CapacityMeasurements":
        if not isinstance(record, Mapping) or set(record) != _MEASUREMENT_FIELDS:
            raise ValueError("capacity measurement fields are invalid")
        if record["schema_version"] != "r03_task8a_capacity_metrics_v1":
            raise ValueError("capacity measurement schema version is invalid")
        if record["topology"] != "single_process":
            raise ValueError("capacity measurement topology must be single_process")
        if record["worker_count"] != 1 or isinstance(record["worker_count"], bool):
            raise ValueError("capacity measurement worker_count must equal one")
        for field in (
            "worker_peak_rss_bytes",
            "worker_p99_peak_rss_bytes",
            "coordinator_peak_rss_bytes",
        ):
            if record[field] is not None:
                raise ValueError(f"{field} must be null in single_process topology")

        process_peak = _positive_int(
            "process_peak_rss_bytes", record["process_peak_rss_bytes"]
        )
        physical_ram = _positive_int("physical_ram_bytes", record["physical_ram_bytes"])
        wall_time = _positive_fraction("wall_time_seconds", record["wall_time_seconds"])
        rows_per_second = _positive_fraction("rows_per_second", record["rows_per_second"])
        free_before = _positive_int("free_bytes_before_run", record["free_bytes_before_run"])
        free_at_high_water = _positive_int(
            "free_bytes_at_disk_high_water", record["free_bytes_at_disk_high_water"]
        )
        scratch_high_water = _positive_int(
            "scratch_staging_high_water_bytes",
            record["scratch_staging_high_water_bytes"],
        )
        total_finalized = _positive_int(
            "total_finalized_bytes", record["total_finalized_bytes"]
        )

        raw_artifacts = record["finalized_artifacts"]
        if not isinstance(raw_artifacts, (list, tuple)) or not raw_artifacts:
            raise ValueError("finalized_artifacts must be a nonempty sequence")
        artifacts: list[FinalizedArtifact] = []
        seen_paths: set[str] = set()
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping) or set(raw) != {
                "artifact_role", "relative_path", "size_bytes"
            }:
                raise ValueError("finalized artifact fields are invalid")
            role = raw["artifact_role"]
            relative_path = raw["relative_path"]
            if not isinstance(role, str) or not role:
                raise ValueError("finalized artifact role is invalid")
            normalized = _normalized_relative_path(relative_path)
            if normalized in seen_paths:
                raise ValueError("finalized artifact path is duplicated")
            seen_paths.add(normalized)
            artifacts.append(
                FinalizedArtifact(
                    role,
                    normalized,
                    _positive_int("finalized artifact size", raw["size_bytes"]),
                )
            )
        artifacts.sort(key=lambda artifact: (artifact.artifact_role, artifact.relative_path))
        if sum(artifact.size_bytes for artifact in artifacts) != total_finalized:
            raise ValueError("finalized artifact bytes do not equal total_finalized_bytes")

        by_role = _positive_size_mapping("finalized_bytes_by_role", record["finalized_bytes_by_role"])
        by_airport = _positive_size_mapping(
            "finalized_bytes_by_airport", record["finalized_bytes_by_airport"]
        )
        derived_by_role: dict[str, int] = {}
        derived_by_airport = {"KAGC": 0, "KBTP": 0, "shared": 0}
        for artifact in artifacts:
            derived_by_role[artifact.artifact_role] = (
                derived_by_role.get(artifact.artifact_role, 0) + artifact.size_bytes
            )
            airport = _artifact_airport(artifact.relative_path) or "shared"
            derived_by_airport[airport] += artifact.size_bytes
        if dict(by_role) != derived_by_role:
            raise ValueError("role byte summary does not match finalized artifacts")
        if dict(by_airport) != derived_by_airport:
            raise ValueError("airport byte summary does not match finalized artifacts")

        counts_total = _count_mapping("counts_total", record["counts_total"])
        _validate_conservation("counts_total", dict(counts_total))
        counts_by_airport = _grouped_counts(
            "counts_by_airport", record["counts_by_airport"]
        )
        if {key for key, _ in counts_by_airport} != set(_AIRPORTS):
            raise ValueError("counts_by_airport must contain KAGC and KBTP")
        counts_by_stratum = _grouped_counts(
            "counts_by_stratum", record["counts_by_stratum"]
        )
        _require_group_sum("counts_by_airport", counts_by_airport, dict(counts_total))
        _require_group_sum("counts_by_stratum", counts_by_stratum, dict(counts_total))
        airport_count_maps = {
            airport: dict(counts) for airport, counts in counts_by_airport
        }
        stratum_sums = {
            airport: {field: 0 for field in _COUNT_FIELDS}
            for airport in _AIRPORTS
        }
        for stratum, counts in counts_by_stratum:
            airport = _stratum_airport(stratum)
            for field, count in counts:
                stratum_sums[airport][field] += count
        if stratum_sums != airport_count_maps:
            raise ValueError("stratum counts do not match their airport summaries")
        expected_rate = Fraction(dict(counts_total)["enumerated"], 1) / wall_time
        rate_tolerance = max(Fraction(1, 10**12), abs(expected_rate) / 10**12)
        if abs(rows_per_second - expected_rate) > rate_tolerance:
            raise ValueError("rows_per_second does not match enumerated rows and wall time")
        samples = _sample_events(record["sample_events"])
        if not _REQUIRED_EVENTS.issubset({sample.event for sample in samples}):
            raise ValueError("required explicit capacity samples are absent")
        if max(sample.scratch_staging_bytes for sample in samples) != scratch_high_water:
            raise ValueError("scratch/staging high-water counter is inconsistent")
        high_water_free = min(
            sample.free_bytes
            for sample in samples
            if sample.scratch_staging_bytes == scratch_high_water
        )
        if (
            high_water_free != free_at_high_water
            or free_at_high_water > free_before
            or free_before < min(sample.free_bytes for sample in samples)
        ):
            raise ValueError("free-byte counters are inconsistent")
        if samples[-1].process_peak_rss_bytes != process_peak:
            raise ValueError("process peak RSS counter is inconsistent")

        return cls(
            "r03_task8a_capacity_metrics_v1",
            "single_process",
            1,
            None,
            None,
            None,
            process_peak,
            physical_ram,
            wall_time,
            rows_per_second,
            free_before,
            free_at_high_water,
            scratch_high_water,
            total_finalized,
            tuple(artifacts),
            by_role,
            by_airport,
            counts_total,
            counts_by_airport,
            counts_by_stratum,
            samples,
        )


@dataclass(frozen=True)
class CapacityProjection:
    finalized_sample_bytes_by_airport: tuple[tuple[str, int], ...]
    finalized_bytes_per_row_by_airport: tuple[tuple[str, Fraction], ...]
    scratch_multiplier: Fraction
    pilot_finalized_bytes: int
    pilot_estimated_peak_bytes: int
    pilot_required_free_bytes: int
    full_finalized_bytes: int
    full_estimated_peak_bytes: int
    full_required_free_bytes: int
    projected_pilot_rss_bytes: int
    projected_full_rss_bytes: int
    memory_limit_bytes: int
    pilot_disk_capacity: bool
    pilot_memory_capacity: bool
    pilot_capacity_pass: bool
    full_disk_capacity: bool
    full_memory_capacity: bool
    full_capacity_ready: bool


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def capacity_probe(
    selected_rows: int,
    record_batch_rows: int,
    shard_count: int,
    free_bytes: int,
) -> CapacityProbe:
    """Compute the frozen Task 8A disk probe without filesystem access."""
    selected = _positive_int("selected_rows", selected_rows)
    batch = _positive_int("record_batch_rows", record_batch_rows)
    shards = _positive_int("shard_count", shard_count)
    if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
        raise ValueError("free_bytes must be a non-negative integer")

    peak = selected * 4096 + 2 * batch * 2048 + shards * 65536
    required = (3 * peak + 1) // 2 + 10 * 1024**3
    return CapacityProbe(
        estimated_peak_bytes=peak,
        required_free_bytes=required,
        available_free_bytes=free_bytes,
        passed=free_bytes >= required,
    )


def _positive_fraction(name: str, value: object) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = Fraction(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_fraction(name: str, value: object) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        result = Fraction(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _normalized_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("finalized artifact relative path is invalid")
    candidate = value.replace("\\", "/")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("finalized artifact relative path is invalid")
    return path.as_posix()


def _artifact_airport(relative_path: str) -> str | None:
    matches = []
    for part in PurePosixPath(relative_path).parts:
        value = part.split("=", 1)[1] if part.startswith("airport_id=") else part
        if value in _AIRPORTS:
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def _stratum_airport(stratum: str) -> str:
    matches = [
        airport
        for airport in _AIRPORTS
        if any(stratum.startswith(f"{airport}{separator}") for separator in "/|:")
    ]
    if len(matches) != 1:
        raise ValueError("capacity stratum key does not identify exactly one airport")
    return matches[0]


def _positive_size_mapping(name: str, value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    result: list[tuple[str, int]] = []
    for key, size in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} key is invalid")
        result.append((key, _positive_int(f"{name} value", size)))
    return tuple(sorted(result))


def _count_mapping(name: str, value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or set(value) != _COUNT_FIELDS:
        raise ValueError(f"{name} count fields are invalid")
    result: list[tuple[str, int]] = []
    for key in sorted(_COUNT_FIELDS):
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
        result.append((key, count))
    return tuple(result)


def _validate_conservation(name: str, counts: Mapping[str, int]) -> None:
    if counts["parsed"] + counts["bad"] != counts["enumerated"]:
        raise ValueError(f"{name} parsed plus bad must equal enumerated")
    if counts["membership"] != counts["parsed"]:
        raise ValueError(f"{name} membership must equal parsed")
    if counts["unique"] > counts["parsed"]:
        raise ValueError(f"{name} unique cannot exceed parsed")
    if counts["conflict"] > counts["unique"]:
        raise ValueError(f"{name} conflict cannot exceed unique")


def _grouped_counts(
    name: str, value: object,
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    result: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for group, raw_counts in value.items():
        if not isinstance(group, str) or not group:
            raise ValueError(f"{name} key is invalid")
        counts = _count_mapping(f"{name}.{group}", raw_counts)
        _validate_conservation(f"{name}.{group}", dict(counts))
        result.append((group, counts))
    return tuple(sorted(result))


def _require_group_sum(
    name: str,
    groups: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    total: Mapping[str, int],
) -> None:
    for field in _COUNT_FIELDS:
        if sum(dict(counts)[field] for _, counts in groups) != total[field]:
            raise ValueError(f"{name} does not sum to counts_total")


def _sample_events(value: object) -> tuple[CapacitySample, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sample_events must be a nonempty sequence")
    result: list[CapacitySample] = []
    previous_time: Fraction | None = None
    previous_peak = 0
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "event",
            "monotonic_seconds",
            "scratch_staging_bytes",
            "free_bytes",
            "process_peak_rss_bytes",
        }:
            raise ValueError("capacity sample event fields are invalid")
        event = raw["event"]
        if not isinstance(event, str) or not event:
            raise ValueError("capacity sample event label is invalid")
        monotonic_seconds = _nonnegative_fraction(
            "sample monotonic_seconds", raw["monotonic_seconds"]
        )
        scratch = _nonnegative_int(
            "sample scratch_staging_bytes", raw["scratch_staging_bytes"]
        )
        free = _positive_int("sample free_bytes", raw["free_bytes"])
        peak = _positive_int(
            "sample process_peak_rss_bytes", raw["process_peak_rss_bytes"]
        )
        if previous_time is not None and monotonic_seconds < previous_time:
            raise ValueError("sample monotonic counter regressed")
        if peak < previous_peak:
            raise ValueError("sample process peak RSS counter regressed")
        previous_time = monotonic_seconds
        previous_peak = peak
        result.append(CapacitySample(event, monotonic_seconds, scratch, free, peak))
    return tuple(result)


def _mapping_rows(name: str, value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_AIRPORTS):
        raise ValueError(f"{name} must contain exactly KAGC and KBTP")
    return {airport: _positive_int(f"{name}.{airport}", value[airport]) for airport in _AIRPORTS}


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def _required_free_bytes(peak: int) -> int:
    return (3 * peak + 1) // 2 + 10 * 1024**3


def _hamilton_shared_bytes(shared: int, weights: Mapping[str, int]) -> dict[str, int]:
    total_weight = sum(weights.values())
    allocations = {
        airport: shared * weights[airport] // total_weight for airport in _AIRPORTS
    }
    remaining = shared - sum(allocations.values())
    order = sorted(
        _AIRPORTS,
        key=lambda airport: (-(shared * weights[airport] % total_weight), airport),
    )
    for airport in order[:remaining]:
        allocations[airport] += 1
    return allocations


def project_capacity(
    measurements: CapacityMeasurements,
    population_rows_by_airport: Mapping[str, int],
    pilot_rows_by_airport: Mapping[str, int],
    free_bytes_after_cleanup: int,
) -> CapacityProjection:
    """Project sample measurements using only exact rational arithmetic."""
    if not isinstance(measurements, CapacityMeasurements):
        raise TypeError("measurements must be CapacityMeasurements")
    population_rows = _mapping_rows(
        "population_rows_by_airport", population_rows_by_airport
    )
    pilot_rows = _mapping_rows("pilot_rows_by_airport", pilot_rows_by_airport)
    if isinstance(free_bytes_after_cleanup, bool) or not isinstance(
        free_bytes_after_cleanup, int
    ) or free_bytes_after_cleanup < 0:
        raise ValueError("free_bytes_after_cleanup must be a non-negative integer")
    if any(pilot_rows[airport] > population_rows[airport] for airport in _AIRPORTS):
        raise ValueError("pilot rows cannot exceed population rows")

    sample_counts = {
        airport: dict(counts)["enumerated"]
        for airport, counts in measurements.counts_by_airport
    }
    if any(sample_counts[airport] <= 0 for airport in _AIRPORTS):
        raise ValueError("positive populations require positive airport sample counts")
    direct = {airport: 0 for airport in _AIRPORTS}
    shared = 0
    for artifact in measurements.finalized_artifacts:
        airport = _artifact_airport(artifact.relative_path)
        if airport is not None:
            direct[airport] += artifact.size_bytes
        else:
            shared += artifact.size_bytes
    shared_allocation = _hamilton_shared_bytes(shared, sample_counts)
    finalized_sample = {
        airport: direct[airport] + shared_allocation[airport]
        for airport in _AIRPORTS
    }
    if any(finalized_sample[airport] <= 0 for airport in _AIRPORTS):
        raise ValueError("each airport must have positive finalized sample bytes")
    rates = {
        airport: Fraction(finalized_sample[airport], sample_counts[airport])
        for airport in _AIRPORTS
    }
    scratch_multiplier = max(
        Fraction(1, 1),
        Fraction(
            measurements.scratch_staging_high_water_bytes,
            measurements.total_finalized_bytes,
        ),
    )

    def finalized(rows: Mapping[str, int]) -> int:
        return _ceil_fraction(sum(rates[airport] * rows[airport] for airport in _AIRPORTS))

    def peak(finalized_bytes: int) -> int:
        return _ceil_fraction(finalized_bytes * (1 + scratch_multiplier))

    pilot_finalized = finalized(pilot_rows)
    full_finalized = finalized(population_rows)
    pilot_peak = peak(pilot_finalized)
    full_peak = peak(full_finalized)
    pilot_required = _required_free_bytes(pilot_peak)
    full_required = _required_free_bytes(full_peak)
    projected_pilot_rss = measurements.process_peak_rss_bytes
    projected_full_rss = measurements.process_peak_rss_bytes
    memory_limit = 7 * measurements.physical_ram_bytes // 10
    pilot_disk = free_bytes_after_cleanup >= pilot_required
    pilot_memory = projected_pilot_rss <= memory_limit
    full_disk = free_bytes_after_cleanup >= full_required
    full_memory = projected_full_rss <= memory_limit
    return CapacityProjection(
        tuple((airport, finalized_sample[airport]) for airport in _AIRPORTS),
        tuple((airport, rates[airport]) for airport in _AIRPORTS),
        scratch_multiplier,
        pilot_finalized,
        pilot_peak,
        pilot_required,
        full_finalized,
        full_peak,
        full_required,
        projected_pilot_rss,
        projected_full_rss,
        memory_limit,
        pilot_disk,
        pilot_memory,
        pilot_disk and pilot_memory,
        full_disk,
        full_memory,
        full_disk and full_memory,
    )


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _contained_path(container: Path, candidate: Path) -> Path:
    container_absolute = Path(os.path.abspath(container))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        candidate_absolute.relative_to(container_absolute)
    except ValueError as exc:
        raise ValueError("capacity observer path escapes its destination volume") from exc
    return candidate_absolute


def _reject_reparse_components(container: Path, candidate: Path) -> None:
    """Reject existing ancestors/reparse components without resolving them."""
    candidate.relative_to(container)
    current = Path(candidate.anchor) if candidate.anchor else Path()
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            item_stat = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(item_stat.st_mode) or _is_reparse(item_stat):
            raise ValueError("capacity observer path contains a link or reparse point")


def _path_identity(path: Path) -> tuple[int, int, int]:
    try:
        item_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError("capacity observer root identity is unavailable") from exc
    if stat.S_ISLNK(item_stat.st_mode) or _is_reparse(item_stat):
        raise ValueError("capacity observer root is a link or reparse point")
    return (
        int(item_stat.st_dev),
        int(item_stat.st_ino),
        stat.S_IFMT(item_stat.st_mode),
    )


def _is_reparse(stat_result: os.stat_result) -> bool:
    return bool(
        getattr(stat_result, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _tree_bytes_without_links(root: Path) -> int:
    """Sum regular files below root without following symlinks/reparse points."""
    try:
        root_stat = root.stat(follow_symlinks=False)
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(root_stat.st_mode) or _is_reparse(root_stat):
        raise ValueError("capacity observer root cannot be a link or reparse point")
    if stat.S_ISREG(root_stat.st_mode):
        return int(root_stat.st_size)
    if not stat.S_ISDIR(root_stat.st_mode):
        return 0
    total = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        # The measured builder atomically promotes/removes owned
                        # staging entries while the observer walks them.
                        continue
                    if stat.S_ISLNK(entry_stat.st_mode) or _is_reparse(entry_stat):
                        raise ValueError(
                            "capacity observer encountered a link or reparse point"
                        )
                    if stat.S_ISDIR(entry_stat.st_mode):
                        stack.append(Path(entry.path))
                    elif stat.S_ISREG(entry_stat.st_mode):
                        total += int(entry_stat.st_size)
        except FileNotFoundError:
            # A queued owned directory may have been promoted since it was
            # observed; absence contributes zero to this instantaneous sample.
            continue
    return total


def _os_process_peak_rss_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            if get_process_memory_info(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return 0
        return 0
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, AttributeError, OSError):
        return 0


def _os_physical_ram_bytes() -> int:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return 0
        return 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


class CapacityBuildObserver:
    """Sampler for one single-writer attempt; not hostile-swap resistant."""

    def __init__(
        self,
        attempt_root: Path,
        destination_root: Path,
        *,
        scratch_root: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        measure_tree_bytes: Callable[[Path], int] = _tree_bytes_without_links,
        available_bytes: Callable[[Path], int] | None = None,
        process_peak_rss_bytes: Callable[[], int] = _os_process_peak_rss_bytes,
        physical_ram_bytes: Callable[[], int] = _os_physical_ram_bytes,
        sampler_wait: Callable[[float], bool] | None = None,
    ) -> None:
        self._destination_root = Path(os.path.abspath(destination_root))
        self._attempt_root = _contained_path(self._destination_root, Path(attempt_root))
        _reject_reparse_components(self._destination_root, self._attempt_root)
        self._scratch_root = (
            None
            if scratch_root is None
            else _contained_path(self._attempt_root, Path(scratch_root))
        )
        if self._scratch_root is not None:
            _reject_reparse_components(self._destination_root, self._scratch_root)
        candidates = sorted(
            (self._attempt_root,)
            if self._scratch_root is None
            else (self._attempt_root, self._scratch_root),
            key=lambda path: len(path.parts),
        )
        measurement_roots: list[Path] = []
        for candidate in candidates:
            if any(
                candidate.is_relative_to(ancestor)
                for ancestor in measurement_roots
            ):
                continue
            measurement_roots.append(candidate)
        self._measurement_roots = tuple(measurement_roots)
        self._destination_identity = _path_identity(self._destination_root)
        self._measurement_root_identities = tuple(
            (root, _path_identity(root)) for root in self._measurement_roots
        )
        if any(
            identity[0] != self._destination_identity[0]
            for _, identity in self._measurement_root_identities
        ):
            raise ValueError("capacity observer roots must share one volume")
        self._monotonic = monotonic
        self._measure_tree_bytes = measure_tree_bytes
        self._available_bytes = available_bytes or (
            lambda path: int(shutil.disk_usage(path).free)
        )
        self._read_peak_rss = process_peak_rss_bytes
        self._physical_ram_bytes = _positive_int(
            "physical_ram_bytes", physical_ram_bytes()
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler_wait = sampler_wait or self._stop.wait
        self._samples: list[CapacitySample] = []
        self._closed: tuple[CapacitySample, ...] | None = None
        self._sampler_error: BaseException | None = None
        self._start_monotonic: Fraction | None = None
        self._scratch_high_water = 0
        self._free_at_high_water = 0
        self._free_before = 0
        self._process_peak = 0
        self.sample("periodic", self._attempt_root)
        self._periodic_deadline = self._samples[-1].monotonic_seconds + 1
        self._thread = threading.Thread(
            target=self._sample_periodically,
            name="r03-capacity-sampler",
            daemon=True,
        )
        self._thread.start()

    def _sample_periodically(self) -> None:
        try:
            while not self._stop.is_set():
                now = _nonnegative_fraction("monotonic clock", self._monotonic())
                wait_seconds = max(Fraction(0), self._periodic_deadline - now)
                if self._sampler_wait(float(wait_seconds)) or self._stop.is_set():
                    return
                self.sample("periodic", self._attempt_root)
                self._periodic_deadline += 1
                after_sample = _nonnegative_fraction(
                    "monotonic clock", self._monotonic()
                )
                if after_sample >= self._periodic_deadline:
                    overrun = after_sample - self._periodic_deadline
                    missed = overrun.numerator // overrun.denominator + 1
                    self._periodic_deadline += missed
        except BaseException as exc:
            with self._lock:
                self._sampler_error = exc
            self._stop.set()

    def _measure_bytes(self) -> int:
        return sum(
            _nonnegative_int(
                "scratch/staging bytes", self._measure_tree_bytes(root)
            )
            for root in self._measurement_roots
        )

    def _validate_roots(self) -> None:
        _reject_reparse_components(self._destination_root, self._destination_root)
        if _path_identity(self._destination_root) != self._destination_identity:
            raise ValueError("capacity observer destination root identity changed")
        for root, expected_identity in self._measurement_root_identities:
            _contained_path(self._attempt_root, root)
            _reject_reparse_components(self._destination_root, root)
            if _path_identity(root) != expected_identity:
                raise ValueError("capacity observer measurement root identity changed")

    def sample(self, event: str, root: Path) -> None:
        if not isinstance(event, str) or not event:
            raise ValueError("capacity sample event is invalid")
        contained = _contained_path(self._attempt_root, Path(root))
        _reject_reparse_components(self._destination_root, contained)
        with self._lock:
            if self._closed is not None:
                raise RuntimeError("capacity observer is closed")
            self._validate_roots()
            now = _nonnegative_fraction("monotonic clock", self._monotonic())
            if self._samples and now < self._samples[-1].monotonic_seconds:
                raise ValueError("monotonic clock regressed")
            size = self._measure_bytes()
            free = _positive_int(
                "available destination bytes", self._available_bytes(self._destination_root)
            )
            peak = _positive_int("process peak RSS", self._read_peak_rss())
            if peak < self._process_peak:
                raise ValueError("process peak RSS counter regressed")
            if self._start_monotonic is None:
                self._start_monotonic = now
                self._free_before = free
            if size > self._scratch_high_water:
                self._scratch_high_water = size
                self._free_at_high_water = free
            elif size == self._scratch_high_water:
                self._free_at_high_water = min(self._free_at_high_water or free, free)
            self._process_peak = peak
            self._samples.append(CapacitySample(event, now, size, free, peak))

    def close(self) -> tuple[CapacitySample, ...]:
        if self._closed is not None:
            return self._closed
        self._stop.set()
        self._thread.join()
        if self._sampler_error is not None:
            raise self._sampler_error
        self.sample("periodic", self._attempt_root)
        with self._lock:
            if (
                not self._samples
                or self._scratch_high_water <= 0
                or self._free_before <= 0
                or self._free_at_high_water <= 0
                or self._free_at_high_water > self._free_before
                or self._process_peak <= 0
                or self._physical_ram_bytes <= 0
                or self._start_monotonic is None
                or self._samples[-1].monotonic_seconds <= self._start_monotonic
            ):
                raise ValueError("capacity counters are unavailable or zero")
            self._closed = tuple(self._samples)
            return self._closed

    @property
    def sample_events(self) -> tuple[CapacitySample, ...]:
        with self._lock:
            return tuple(self._samples)

    @property
    def scratch_staging_high_water_bytes(self) -> int:
        return self._scratch_high_water

    @property
    def free_bytes_before_run(self) -> int:
        return self._free_before

    @property
    def free_bytes_at_disk_high_water(self) -> int:
        return self._free_at_high_water

    @property
    def process_peak_rss_bytes(self) -> int:
        return self._process_peak

    @property
    def physical_ram_bytes(self) -> int:
        return self._physical_ram_bytes


def _available_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _fraction_record(value: Fraction) -> dict[str, int]:
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_record(name: str, value: object) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} must use the canonical rational encoding")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError(f"{name} has invalid rational components")
    result = Fraction(numerator, denominator)
    if _fraction_record(result) != dict(value):
        raise ValueError(f"{name} rational encoding is not reduced and canonical")
    return result


def _jsonable(value: object) -> object:
    if isinstance(value, Fraction):
        return _fraction_record(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _pinned_regular_bytes(root: Path, path: Path) -> bytes:
    safe = _contained_path(root, path)
    _reject_reparse_components(root, safe)
    before = os.lstat(safe)
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"capacity evidence is not a regular file: {safe}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(safe, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"capacity evidence is not a regular file: {safe}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = os.lstat(safe)
    if any(
        getattr(before, name, None) != getattr(after, name, None)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    ):
        raise ValueError("capacity evidence changed during independent read")
    return payload


def _seal_bytes(root: Path, path: Path, payload: bytes) -> None:
    """Create one immutable contained file and never replace different bytes."""
    safe = _contained_path(root, path)
    _reject_reparse_components(root, safe)
    if safe.exists():
        if _pinned_regular_bytes(root, safe) != payload:
            raise ValueError(f"existing capacity evidence differs: {safe.name}")
        return
    temporary = safe.with_name(f".{safe.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if _pinned_regular_bytes(root, temporary) != payload:
            raise ValueError("capacity evidence temporary fixity mismatch")
        try:
            os.link(temporary, safe)
        except FileExistsError:
            if _pinned_regular_bytes(root, safe) != payload:
                raise ValueError(f"concurrent capacity evidence differs: {safe.name}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if _pinned_regular_bytes(root, safe) != payload:
        raise ValueError("capacity evidence fixity mismatch")


def _read_json(root: Path, path: Path) -> dict[str, object]:
    try:
        value = json.loads(_pinned_regular_bytes(root, path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"capacity evidence JSON is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"capacity evidence JSON is not an object: {path.name}")
    return value


def _ensure_selection_root(external_root: Path, selection_id: str) -> Path:
    external = Path(os.path.abspath(external_root))
    external.mkdir(parents=True, exist_ok=True)
    _reject_reparse_components(external, external)
    capacity = external / "capacity"
    capacity.mkdir(exist_ok=True)
    _reject_reparse_components(external, capacity)
    selection_root = capacity / selection_id
    selection_root.mkdir(exist_ok=True)
    _reject_reparse_components(external, selection_root)
    if _path_identity(external)[0] != _path_identity(selection_root)[0]:
        raise ValueError("capacity evidence root must stay on the external volume")
    return selection_root


@contextmanager
def _capacity_gate_lock(external_root: Path, selection_id: str):
    """Hold a non-blocking OS lock for the complete selection-root lifecycle."""
    if len(selection_id) != 64 or any(character not in "0123456789abcdef" for character in selection_id):
        raise ValueError("capacity lock selection identifier is invalid")
    external = Path(os.path.abspath(external_root))
    external.mkdir(parents=True, exist_ok=True)
    capacity = external / "capacity"
    capacity.mkdir(exist_ok=True)
    locks = capacity / ".locks"
    locks.mkdir(exist_ok=True)
    _reject_reparse_components(external, locks)
    lock_path = locks / f"{selection_id}.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("concurrent capacity gate already owns selection") from exc
        else:
            import fcntl
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("concurrent capacity gate already owns selection") from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _assert_finalized_root_state(selection_root: Path) -> CapacityGateRecord | None:
    names = {path.name for path in selection_root.iterdir()}
    retained = set(_RETAINED_EVIDENCE)
    if not names:
        return None
    if names in (retained, retained | {"attempts"}):
        return verify_capacity_gate_evidence(selection_root)
    if names in (
        {"capacity_selection_manifest.json"},
        {"capacity_selection_manifest.json", "capacity_metrics.json", "attempts"},
    ):
        return None
    raise ValueError("capacity selection root contains unexpected pre-existing files")


def _new_attempt_path(selection_root: Path) -> Path:
    return selection_root / "attempts" / uuid.uuid4().hex


def _invocation_binding(
    project: Path, external: Path, parameters: R03RunParameters,
    command: Sequence[str],
) -> dict[str, object]:
    environment = _command_environment(
        project, external, parameters, command, "binding", "binding", None, None,
    )
    return {
        name: environment[name] for name in (
            "command_argv", "project_root", "external_root", "run_parameters",
            "capacity_gate_code_sha256", "generator_code_hash",
            "runtime_environment_hash", "dependency_lock_sha256",
        )
    }


def _terminal_owned_attempt(root: Path, expectations: TerminalExpectations) -> tuple[
    str, str, tuple[int, int, int], int
] | None:
    """Authenticate the sole public attempt before generation zero exists."""
    attempts_root = root / "attempts"
    if not attempts_root.exists():
        return None
    _reject_reparse_components(root, attempts_root)
    if not attempts_root.is_dir():
        raise ValueError("terminal attempts root is not a direct directory")
    entries = list(attempts_root.iterdir())
    if len(entries) != 1 or not entries[0].is_dir() or entries[0].is_symlink():
        raise ValueError("terminal prefix must contain exactly one public owned attempt")
    public = entries[0]
    if len(public.name) != 32 or any(char not in "0123456789abcdef" for char in public.name):
        raise ValueError("terminal public attempt name is not owned")
    _reject_reparse_components(root, public)
    identity = _path_identity(public)
    volume = _path_identity(root)[0]
    if identity[0] != volume:
        raise ValueError("terminal public attempt volume differs from selection root")
    invocation = _pinned_regular_bytes(public, public / "invocation.json")
    if hashlib.sha256(invocation).hexdigest() != expectations.invocation_sha256:
        raise ValueError("terminal public attempt invocation differs")
    try:
        parsed = json.loads(invocation.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("terminal public attempt invocation is invalid") from exc
    if _canonical_json_bytes(parsed) != invocation:
        raise ValueError("terminal public attempt invocation is noncanonical")
    return (
        "attempts/" + public.name,
        "attempts/.owned-cleanup-" + public.name,
        identity,
        volume,
    )


def _terminal_metrics_ready(
    root: Path, expectations: TerminalExpectations, *, started_utc: str,
) -> TerminalJournal:
    owned = _terminal_owned_attempt(root, expectations)
    record = TerminalJournal(
        schema_version="r03_task8a_capacity_terminal_journal_v1",
        journal_id="0" * 64,
        generation=0,
        state=TerminalState.METRICS_READY,
        capacity_selection_id=expectations.capacity_selection_id,
        selection_root=expectations.selection_root,
        project_root=expectations.project_root,
        external_root=expectations.external_root,
        invocation_sha256=expectations.invocation_sha256,
        capacity_selection_manifest_sha256=expectations.capacity_selection_manifest_sha256,
        capacity_metrics_sha256=expectations.capacity_metrics_sha256,
        started_utc=started_utc,
        attempt_mode="OWNED" if owned is not None else "NONE",
        attempt_public_relative_path=None if owned is None else owned[0],
        attempt_tombstone_relative_path=None if owned is None else owned[1],
        attempt_path_identity=None if owned is None else owned[2],
        attempt_volume_identity=None if owned is None else owned[3],
        cleanup_status="PENDING",
        retained_attempt_relative_path=None,
        cleanup_failure=None,
        free_bytes_after_cleanup=None,
        decision=None,
        pilot_capacity_pass=None,
        full_capacity_ready=None,
        performance_review_required=None,
        evidence_order=None,
        evidence_payload_sha256=None,
        ended_utc=None,
        committed_envelope_sha256=None,
    )
    return replace(record, journal_id=derive_terminal_journal_id(record))


def prepare_terminal_cleanup(
    selection_root: Path,
    expectations: TerminalExpectations,
    *,
    started_utc: str,
) -> TerminalJournal:
    """Durably bind exact cleanup ownership before a pathname can change."""
    root = Path(os.path.abspath(selection_root))
    current = load_terminal_journal(root, expectations)
    if current is None:
        current = _terminal_metrics_ready(root, expectations, started_utc=started_utc)
        replace_terminal_journal(root, None, current)
    if current.state is TerminalState.METRICS_READY:
        prepared = replace(current, generation=1, state=TerminalState.PREPARED)
        replace_terminal_journal(root, current, prepared)
        return prepared
    if current.state is TerminalState.PREPARED:
        return current
    if current.state in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED}:
        return current
    raise ValueError("terminal cleanup cannot begin after evidence sealing")


def _terminal_expectations_for_metrics(
    selection_root: Path,
    selection: CapacitySelectionManifest,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
) -> TerminalExpectations:
    """Bind generation zero to the exact E1, E2, and invocation bytes."""
    return TerminalExpectations(
        capacity_selection_id=selection.selection_id,
        selection_root=str(selection_root.resolve()),
        project_root=str(project),
        external_root=str(external),
        invocation_sha256=hashlib.sha256(
            _canonical_json_bytes(_invocation_binding(project, external, parameters, command))
        ).hexdigest(),
        capacity_selection_manifest_sha256=hashlib.sha256(
            _pinned_regular_bytes(
                selection_root, selection_root / "capacity_selection_manifest.json"
            )
        ).hexdigest(),
        capacity_metrics_sha256=hashlib.sha256(
            _pinned_regular_bytes(selection_root, selection_root / "capacity_metrics.json")
        ).hexdigest(),
    )


def _terminal_outcome_record(
    selection_root: Path,
    selection: CapacitySelectionManifest,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    *,
    started_utc: str,
    pilot_capacity_pass: bool,
    full_capacity_ready: bool,
    performance_review_required: bool,
) -> CapacityGateRecord:
    """Drive the authenticated cleanup result through the terminal finalizer."""
    expectations = _terminal_expectations_for_metrics(
        selection_root, selection, project, external, parameters, command,
    )
    prepared = prepare_terminal_cleanup(
        selection_root, expectations, started_utc=started_utc,
    )
    outcome = finish_terminal_cleanup(
        selection_root,
        prepared,
        pilot_capacity_pass=pilot_capacity_pass,
        full_capacity_ready=full_capacity_ready,
        performance_review_required=performance_review_required,
    )
    return seal_terminal_evidence(
        selection_root,
        outcome,
        project,
        external,
        parameters,
        command,
    )


def _terminal_delete_owned_attempt(root: Path, journal: TerminalJournal) -> None:
    if journal.attempt_mode != "OWNED":
        return
    assert journal.attempt_public_relative_path is not None
    assert journal.attempt_tombstone_relative_path is not None
    assert journal.attempt_path_identity is not None
    assert journal.attempt_volume_identity is not None
    attempts_root = root / "attempts"
    public = root / journal.attempt_public_relative_path
    tombstone = root / journal.attempt_tombstone_relative_path
    _reject_reparse_components(root, attempts_root)
    if not os.path.lexists(attempts_root):
        # A crash can occur after the exact tombstone and its now-empty parent
        # have been removed but before generation two is durable.  PREPARED is
        # the durable authority for this already-completed destructive step.
        return
    if not attempts_root.is_dir():
        raise ValueError("terminal owned attempt root disappeared")
    children = list(attempts_root.iterdir())
    if not children:
        attempts_root.rmdir()
        return
    allowed = {public.name, tombstone.name}
    if len(children) != 1 or children[0].name not in allowed:
        raise ValueError("terminal owned cleanup has a foreign or ambiguous attempt")
    lexical = children[0]
    _reject_reparse_components(root, lexical)
    if _path_identity(lexical) != journal.attempt_path_identity:
        raise ValueError("terminal owned attempt identity changed")
    if _path_identity(lexical)[0] != journal.attempt_volume_identity:
        raise ValueError("terminal owned attempt volume changed")
    if lexical == public:
        if tombstone.exists() or tombstone.is_symlink():
            raise ValueError("terminal cleanup tombstone already exists")
        os.replace(public, tombstone)
        if public.exists() or public.is_symlink():
            raise ValueError("terminal cleanup public replacement exists")
        lexical = tombstone
        _reject_reparse_components(root, lexical)
        if _path_identity(lexical) != journal.attempt_path_identity:
            raise ValueError("terminal cleanup tombstone identity changed")
    _tree_bytes_without_links(lexical)
    shutil.rmtree(lexical)
    if lexical.exists() or lexical.is_symlink():
        raise ValueError("terminal cleanup tombstone remains")
    if public.exists() or public.is_symlink():
        raise ValueError("terminal cleanup public replacement exists")
    remaining = list(attempts_root.iterdir())
    if remaining:
        raise ValueError("terminal cleanup left a foreign attempt child")
    attempts_root.rmdir()


def _terminal_outcome(
    prepared: TerminalJournal,
    *,
    status: str,
    free_bytes: int,
    pilot_capacity_pass: bool,
    full_capacity_ready: bool,
    performance_review_required: bool,
    failure: CleanupFailure | None = None,
    retained_attempt_relative_path: str | None = None,
) -> TerminalJournal:
    if status == "CLEAN":
        decision = "CAPACITY_PASS_PILOT" if pilot_capacity_pass else "BLOCKED_NEEDS_HUMAN_DECISION"
        return replace(
            prepared, generation=2, state=TerminalState.OUTCOME_CLEAN,
            cleanup_status="CLEAN", retained_attempt_relative_path=None,
            cleanup_failure=None, free_bytes_after_cleanup=free_bytes,
            decision=decision, pilot_capacity_pass=pilot_capacity_pass,
            full_capacity_ready=full_capacity_ready,
            performance_review_required=performance_review_required,
        )
    return replace(
        prepared, generation=2, state=TerminalState.OUTCOME_FAILED,
        cleanup_status="FAILED", retained_attempt_relative_path=retained_attempt_relative_path,
        cleanup_failure=failure, free_bytes_after_cleanup=free_bytes,
        decision="BLOCKED_CLEANUP_REQUIRED", pilot_capacity_pass=False,
        full_capacity_ready=False, performance_review_required=performance_review_required,
    )


def finish_terminal_cleanup(
    selection_root: Path,
    journal: TerminalJournal,
    *,
    pilot_capacity_pass: bool,
    full_capacity_ready: bool,
    performance_review_required: bool,
) -> TerminalJournal:
    """Resume exactly the prepared cleanup, producing one durable outcome."""
    root = Path(os.path.abspath(selection_root))
    expectations = TerminalExpectations(
        capacity_selection_id=journal.capacity_selection_id,
        selection_root=journal.selection_root,
        project_root=journal.project_root,
        external_root=journal.external_root,
        invocation_sha256=journal.invocation_sha256,
        capacity_selection_manifest_sha256=journal.capacity_selection_manifest_sha256,
        capacity_metrics_sha256=journal.capacity_metrics_sha256,
    )
    current = load_terminal_journal(root, expectations)
    if current != journal:
        raise ValueError("terminal cleanup journal is not the authoritative generation")
    if current.state in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED}:
        return current
    if current.state is not TerminalState.PREPARED:
        raise ValueError("terminal cleanup requires PREPARED journal")
    try:
        _terminal_delete_owned_attempt(root, current)
    except (OSError, PermissionError) as exc:
        if current.attempt_mode != "OWNED":
            raise
        shape = classify_terminal_shape(root, current)
        if shape.attempt_kind not in {"PUBLIC", "TOMBSTONE"} or shape.attempt_relative_path is None:
            raise ValueError("terminal cleanup failure does not retain one bound attempt") from exc
        failure = CleanupFailure("remove_attempt", type(exc).__name__)
        outcome = _terminal_outcome(
            current, status="FAILED", free_bytes=_available_bytes(root),
            pilot_capacity_pass=pilot_capacity_pass, full_capacity_ready=full_capacity_ready,
            performance_review_required=performance_review_required, failure=failure,
            retained_attempt_relative_path=shape.attempt_relative_path,
        )
        replace_terminal_journal(root, current, outcome)
        return outcome
    outcome = _terminal_outcome(
        current, status="CLEAN", free_bytes=_available_bytes(root),
        pilot_capacity_pass=pilot_capacity_pass, full_capacity_ready=full_capacity_ready,
        performance_review_required=performance_review_required,
    )
    replace_terminal_journal(root, current, outcome)
    return outcome


def _retained_attempt_path(root: Path, relative_path: object) -> tuple[Path, Path]:
    """Pin the one direct retained-attempt path named in final evidence."""
    if not isinstance(relative_path, str):
        raise ValueError("retained attempt path is invalid")
    parts = relative_path.split("/")
    if len(parts) != 2 or parts[0] != "attempts":
        raise ValueError("retained attempt path is invalid")
    name = parts[1]
    if name.startswith(".owned-cleanup-"):
        identifier = name.removeprefix(".owned-cleanup-")
    else:
        identifier = name
    if (
        not identifier
        or len(identifier) != 32
        or any(character not in "0123456789abcdef" for character in identifier)
        or name not in {identifier, ".owned-cleanup-" + identifier}
    ):
        raise ValueError("retained attempt path is invalid")
    attempts_root = root / "attempts"
    retained = attempts_root / name
    _reject_reparse_components(root, attempts_root)
    _reject_reparse_components(root, retained)
    return attempts_root, retained


def _retained_attempt_evidence(
    root: Path, journal: TerminalJournal,
) -> dict[str, object] | None:
    """Bind a FAILED terminal record to one live no-follow retained attempt."""
    attempts_root = root / "attempts"
    if journal.cleanup_status == "CLEAN":
        if os.path.lexists(attempts_root):
            raise ValueError("clean terminal outcome retains an attempt")
        return None
    if journal.cleanup_status != "FAILED":
        raise ValueError("terminal cleanup outcome is invalid")
    assert journal.retained_attempt_relative_path is not None
    assert journal.attempt_path_identity is not None
    assert journal.attempt_volume_identity is not None
    assert journal.cleanup_failure is not None
    attempts_root, retained = _retained_attempt_path(
        root, journal.retained_attempt_relative_path,
    )
    if not attempts_root.is_dir():
        raise ValueError("retained attempt root is missing")
    children = list(attempts_root.iterdir())
    if len(children) != 1 or children[0].name != retained.name:
        raise ValueError("retained attempt shape is ambiguous")
    if not retained.is_dir() or _path_identity(retained) != journal.attempt_path_identity:
        raise ValueError("retained attempt identity changed")
    if _path_identity(retained)[0] != journal.attempt_volume_identity:
        raise ValueError("retained attempt volume changed")
    _tree_bytes_without_links(retained)
    return {
        "relative_path": journal.retained_attempt_relative_path,
        "path_identity": list(journal.attempt_path_identity),
        "volume_identity": journal.attempt_volume_identity,
        "invocation_sha256": journal.invocation_sha256,
        "cleanup_failure": asdict(journal.cleanup_failure),
    }


def _validate_retained_attempt_evidence(
    root: Path,
    cleanup_status: str,
    value: object,
    *,
    expected_invocation_sha256: str,
) -> None:
    """Independently check the E3 retained-attempt binding after unlink."""
    attempts_root = root / "attempts"
    if cleanup_status == "CLEAN":
        if value is not None or os.path.lexists(attempts_root):
            raise ValueError("clean capacity evidence retains an attempt")
        return
    if cleanup_status != "FAILED" or not isinstance(value, Mapping):
        raise ValueError("failed capacity evidence does not bind a retained attempt")
    if set(value) != {
        "relative_path", "path_identity", "volume_identity", "invocation_sha256",
        "cleanup_failure",
    }:
        raise ValueError("retained attempt evidence fields are invalid")
    attempts_root, retained = _retained_attempt_path(root, value["relative_path"])
    identity = value["path_identity"]
    volume = value["volume_identity"]
    invocation = value["invocation_sha256"]
    failure = value["cleanup_failure"]
    if (
        not isinstance(identity, list)
        or len(identity) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in identity)
        or isinstance(volume, bool)
        or not isinstance(volume, int)
        or volume < 0
        or not isinstance(invocation, str)
        or len(invocation) != 64
        or any(character not in "0123456789abcdef" for character in invocation)
        or invocation != expected_invocation_sha256
        or not isinstance(failure, Mapping)
        or set(failure) != {"stage", "exception_type"}
    ):
        raise ValueError("retained attempt evidence is invalid")
    try:
        normalized_failure = asdict(CleanupFailure(**dict(failure)))
    except (TypeError, ValueError) as exc:
        raise ValueError("retained attempt failure evidence is invalid") from exc
    if not attempts_root.is_dir():
        raise ValueError("failed cleanup does not retain its attempt evidence")
    children = list(attempts_root.iterdir())
    if len(children) != 1 or children[0].name != retained.name or not retained.is_dir():
        raise ValueError("retained attempt shape is ambiguous")
    _reject_reparse_components(root, retained)
    actual_identity = _path_identity(retained)
    if actual_identity != tuple(identity):
        raise ValueError("retained attempt identity changed")
    if actual_identity[0] != volume:
        raise ValueError("retained attempt volume changed")
    _tree_bytes_without_links(retained)
    expected = {
        "relative_path": value["relative_path"],
        "path_identity": list(actual_identity),
        "volume_identity": actual_identity[0],
        "invocation_sha256": expected_invocation_sha256,
        "cleanup_failure": normalized_failure,
    }
    if dict(value) != expected:
        raise ValueError("retained attempt evidence is noncanonical")


def _terminal_verifier_payload(
    root: Path,
    journal: TerminalJournal,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    *,
    ended_utc: str | None = None,
) -> tuple[CapacityGateRecord, bytes, bytes, bytes]:
    """Independently derive the immutable E3--E5 payloads from E1/E2 and outcome."""
    if journal.state not in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED, TerminalState.SEALING, TerminalState.COMMITTED}:
        raise ValueError("terminal evidence requires a durable cleanup outcome")
    selection = CapacitySelectionManifest.from_record(
        _read_json(root, root / "capacity_selection_manifest.json")
    )
    if selection.selection_id != journal.capacity_selection_id:
        raise ValueError("terminal evidence selection does not bind journal")
    expectations = _terminal_expectations_for_metrics(
        root, selection, project, external, parameters, command,
    )
    if any(
        getattr(expectations, field.name) != getattr(journal, field.name)
        for field in fields(TerminalExpectations)
    ):
        raise ValueError("terminal evidence invocation does not bind journal")
    metrics = _read_json(root, root / "capacity_metrics.json")
    status = metrics.get("measurement_status")
    if not isinstance(status, str):
        raise ValueError("terminal evidence metrics status is invalid")
    record = CapacityGateRecord(
        schema_version="r03_task8a_capacity_gate_v1",
        capacity_selection_id=selection.selection_id,
        build_started=status != "NOT_RUN_PROBE_BLOCKED",
        measurement_status=status,
        pilot_capacity_pass=bool(journal.pilot_capacity_pass),
        full_capacity_ready=bool(journal.full_capacity_ready),
        performance_review_required=bool(journal.performance_review_required),
        cleanup_status=journal.cleanup_status,
        decision=str(journal.decision),
        is_full_population=False,
        r04_authorized=False,
        model_training_authorized=False,
        paper_main_result_authorized=False,
    )
    if journal.cleanup_status == "FAILED":
        record = replace(
            record, pilot_capacity_pass=False, full_capacity_ready=False,
            decision="BLOCKED_CLEANUP_REQUIRED",
        )
    if journal.cleanup_status == "CLEAN" and record.pilot_capacity_pass != bool(journal.pilot_capacity_pass):
        raise ValueError("terminal outcome pilot pass is not reproducible")
    if journal.cleanup_status == "CLEAN" and record.full_capacity_ready != bool(journal.full_capacity_ready):
        raise ValueError("terminal outcome full readiness is not reproducible")
    measurement: CapacityMeasurements | None = None
    projection: CapacityProjection | None = None
    if status == "MEASURED":
        raw_measurement = metrics.get("measurements")
        if not isinstance(raw_measurement, Mapping):
            raise ValueError("terminal evidence measured metrics are absent")
        measurement = CapacityMeasurements.from_record(_decode_measurement_record(raw_measurement))
        projection = project_capacity(
            measurement,
            dict(selection.population_rows_by_airport),
            dict(selection.selected_rows_by_airport),
            int(journal.free_bytes_after_cleanup),
        )
    verifier = asdict(record)
    verifier.update({
        "free_bytes_after_cleanup": journal.free_bytes_after_cleanup,
        "ledger_verification": status == "MEASURED",
        "metrics_verification": True,
        "pilot_disk_capacity": False if projection is None else projection.pilot_disk_capacity,
        "pilot_memory_capacity": False if projection is None else projection.pilot_memory_capacity,
        "full_disk_capacity": False if projection is None else projection.full_disk_capacity,
        "full_memory_capacity": False if projection is None else projection.full_memory_capacity,
        "retained_attempt": _retained_attempt_evidence(root, journal),
    })
    environment = _command_environment(
        project, external, parameters, command, journal.started_utc,
        ended_utc or journal.ended_utc or _utc_now(), None, None,
    )
    e3 = _canonical_json_bytes(verifier)
    e4 = _canonical_json_bytes(environment)
    inputs = {
        _RETAINED_EVIDENCE[0]: _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[0]),
        _RETAINED_EVIDENCE[1]: _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[1]),
        _RETAINED_EVIDENCE[2]: e3,
        _RETAINED_EVIDENCE[3]: e4,
    }
    e5 = _canonical_json_bytes({
        "schema_version": "r03_task8a_hashes_v1",
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in inputs.items()},
    })
    return record, e3, e4, e5


def freeze_terminal_evidence(
    selection_root: Path,
    journal: TerminalJournal,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
) -> TerminalJournal:
    """Freeze E1--E5 bytes before an immutable final evidence file is created."""
    root = Path(os.path.abspath(selection_root))
    if journal.state is TerminalState.SEALING:
        return journal
    if journal.state is TerminalState.COMMITTED:
        raise ValueError("terminal evidence is already committed")
    if journal.state not in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED}:
        raise ValueError("terminal evidence requires a generation-two outcome")
    ended_utc = _utc_now()
    _, e3, e4, e5 = _terminal_verifier_payload(
        root, journal, project, external, parameters, command, ended_utc=ended_utc,
    )
    payloads = (
        _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[0]),
        _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[1]),
        e3, e4, e5,
    )
    sealing = replace(
        journal,
        generation=3,
        state=TerminalState.SEALING,
        evidence_order=_RETAINED_EVIDENCE,
        evidence_payload_sha256=tuple(
            (name, hashlib.sha256(payload).hexdigest())
            for name, payload in zip(_RETAINED_EVIDENCE, payloads, strict=True)
        ),
        ended_utc=ended_utc,
    )
    replace_terminal_journal(root, journal, sealing)
    return sealing


def _sealed_payloads(
    root: Path,
    journal: TerminalJournal,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
) -> tuple[CapacityGateRecord, tuple[bytes, ...]]:
    if journal.state not in {TerminalState.SEALING, TerminalState.COMMITTED}:
        raise ValueError("terminal evidence payloads require sealing state")
    record, e3, e4, e5 = _terminal_verifier_payload(
        root, journal, project, external, parameters, command,
    )
    payloads = (
        _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[0]),
        _pinned_regular_bytes(root, root / _RETAINED_EVIDENCE[1]),
        e3, e4, e5,
    )
    assert journal.evidence_payload_sha256 is not None
    digests = tuple((name, hashlib.sha256(payload).hexdigest()) for name, payload in zip(_RETAINED_EVIDENCE, payloads, strict=True))
    if digests != journal.evidence_payload_sha256:
        raise ValueError("terminal evidence payload differs from frozen journal")
    return record, payloads


def _unlink_committed_terminal_journal(root: Path, journal: TerminalJournal) -> None:
    journal_path = root / ".capacity-terminal-journal.json"
    if _pinned_regular_bytes(root, journal_path) != canonical_terminal_journal_bytes(journal):
        raise ValueError("committed terminal journal changed before unlink")
    journal_path.unlink()


def seal_terminal_evidence(
    selection_root: Path,
    journal: TerminalJournal,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
) -> CapacityGateRecord:
    """Seal the unique E3/E4/E5 suffix, verify, commit, reverify, unlink last."""
    root = Path(os.path.abspath(selection_root))
    expectations = _terminal_expectations_for_metrics(
        root,
        CapacitySelectionManifest.from_record(_read_json(root, root / _RETAINED_EVIDENCE[0])),
        project, external, parameters, command,
    )
    current = load_terminal_journal(root, expectations)
    if current != journal:
        raise ValueError("terminal finalizer journal is not authoritative")
    if current.state in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED}:
        current = freeze_terminal_evidence(root, current, project, external, parameters, command)
    classify_terminal_shape(root, current)
    record, payloads = _sealed_payloads(root, current, project, external, parameters, command)
    for name, payload in zip(_RETAINED_EVIDENCE[2:], payloads[2:], strict=True):
        _seal_bytes(root, root / name, payload)
    verified = verify_capacity_gate_evidence(root, allow_terminal_journal=True)
    if verified != record:
        raise ValueError("terminal evidence verification differs from frozen record")
    if current.state is TerminalState.SEALING:
        digest_record = replace(
            current,
            generation=4,
            state=TerminalState.COMMITTED,
            committed_envelope_sha256="0" * 64,
        )
        committed = replace(
            digest_record,
            committed_envelope_sha256=committed_terminal_envelope_sha256(digest_record),
        )
        replace_terminal_journal(root, current, committed)
        current = committed
    if current.state is not TerminalState.COMMITTED:
        raise ValueError("terminal finalizer did not reach committed state")
    if current.committed_envelope_sha256 != committed_terminal_envelope_sha256(current):
        raise ValueError("terminal committed envelope digest is invalid")
    if verify_capacity_gate_evidence(root, allow_terminal_journal=True) != record:
        raise ValueError("terminal committed evidence changed during re-verification")
    _unlink_committed_terminal_journal(root, current)
    return record


def recover_terminal_transaction(
    selection_root: Path,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
) -> CapacityGateRecord:
    """Recover one authenticated terminal journal without rerunning a build."""
    root = Path(os.path.abspath(selection_root))
    journal_path = root / ".capacity-terminal-journal.json"
    if not journal_path.exists() and not os.path.lexists(journal_path):
        # A completed transaction deliberately unlinks the journal last.  Its
        # sole recovery path first rebinds the exact invocation recorded in
        # E4, then independently checks the strict five-file envelope.
        names = {path.name for path in root.iterdir()}
        retained = set(_RETAINED_EVIDENCE)
        if names not in (retained, retained | {"attempts"}):
            raise ValueError("capacity evidence envelope is incomplete or unexpected")
        selection = CapacitySelectionManifest.from_record(
            _read_json(root, root / "capacity_selection_manifest.json")
        )
        expectations = _terminal_expectations_for_metrics(
            root, selection, project, external, parameters, command,
        )
        environment = _read_json(root, root / "command_environment.json")
        expected_binding = _invocation_binding(project, external, parameters, command)
        actual_binding = {
            name: environment.get(name) for name in expected_binding
        }
        actual_invocation_sha256 = hashlib.sha256(
            _canonical_json_bytes(actual_binding)
        ).hexdigest()
        if (
            actual_binding != expected_binding
            or actual_invocation_sha256 != expectations.invocation_sha256
        ):
            raise ValueError("completed capacity evidence invocation does not match recovery")
        return verify_capacity_gate_evidence(root)
    selection = CapacitySelectionManifest.from_record(
        _read_json(root, root / "capacity_selection_manifest.json")
    )
    expectations = _terminal_expectations_for_metrics(root, selection, project, external, parameters, command)
    journal = load_terminal_journal(root, expectations)
    if journal is None:
        return verify_capacity_gate_evidence(root)
    if journal.state is TerminalState.COMMITTED:
        return seal_terminal_evidence(root, journal, project, external, parameters, command)
    if journal.state in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED, TerminalState.SEALING}:
        return seal_terminal_evidence(root, journal, project, external, parameters, command)
    raise ValueError("terminal journal is not ready for evidence recovery")


def _resume_metrics_attempt(
    selection_root: Path,
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    started: str,
    selection: CapacitySelectionManifest,
) -> CapacityGateRecord:
    """Reject pre-journal partial state until journal recovery owns it.

    Task 3 deliberately retires the previous multi-file cleanup recovery.  A
    journal-aware end-to-end recovery/finalizer belongs to Task 4; treating an
    unjournaled E1+E2 prefix as resumable here would silently restore the
    rejected authority model.
    """
    del selection_root, project, external, parameters, command, started, selection
    raise ValueError("unjournaled capacity metrics attempt cannot be resumed")


def _selected_lineage(
    selection: CapacitySelectionManifest,
) -> tuple[dict[str, tuple[str, str]], tuple[str, ...]]:
    raw_to_group: dict[str, tuple[str, str]] = {}
    strata: set[str] = set()
    for source in selection.sources:
        if not source.selected_ordinals:
            continue
        stratum = f"{source.airport_id}|{source.lineage_month}|{source.storage_route}"
        strata.add(stratum)
        for ordinal in source.selected_ordinals:
            identifier = raw_row_id(
                source.source_file_id,
                source.source_content_sha256,
                source.archive_member,
                ordinal,
            )
            if identifier in raw_to_group:
                raise ValueError("capacity selection contains a duplicate raw_row_id")
            raw_to_group[identifier] = (source.airport_id, stratum)
    if len(raw_to_group) != sum(dict(selection.selected_rows_by_airport).values()):
        raise ValueError("capacity selection raw-row map is incomplete")
    return raw_to_group, tuple(sorted(strata))


def _semantic_value(value: object) -> object:
    """Canonicalize scientific row values without using builder artifacts."""
    if isinstance(value, datetime):
        return {"datetime_utc": value.isoformat(timespec="microseconds")}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, tuple):
        return [_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _semantic_value(item) for key, item in value.items()}
    return value


def _semantic_digest(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            _semantic_value(row), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _digest_update(digest: object, row: Mapping[str, object]) -> None:
    payload = json.dumps(
        _semantic_value(row), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))  # type: ignore[attr-defined]
    digest.update(payload)  # type: ignore[attr-defined]


def _oracle_value_key(value: object) -> object:
    return 0.0 if isinstance(value, float) and value == 0.0 else value


def _oracle_group_summary(
    rows: Iterable[ParsedRow], policies: CanonicalPolicies,
) -> tuple[dict[str, object], dict[str, object] | None, int, int, int, str]:
    """Independently summarize one externally sorted group in constant memory."""
    del policies
    winner: ParsedRow | None = None
    first: ParsedRow | None = None
    count = 0
    core_first: dict[str, object] = {}
    metadata_first: dict[str, object] = {}
    core_bits = 0
    metadata_bits = 0
    best_age_count = 0
    best_completeness_count = 0
    for row in rows:
        count += 1
        if first is None:
            first = row
            winner = row
            core_first = {name: _oracle_value_key(getattr(row, name)) for name in CORE_STATE_FIELDS}
            metadata_first = {name: _oracle_value_key(getattr(row, name)) for name in METADATA_FIELDS}
            best_age_count = 1
            best_completeness_count = 1
        else:
            for index, name in enumerate(CORE_STATE_FIELDS):
                if _oracle_value_key(getattr(row, name)) != core_first[name]:
                    core_bits |= 1 << index
            for index, name in enumerate(METADATA_FIELDS):
                if _oracle_value_key(getattr(row, name)) != metadata_first[name]:
                    metadata_bits |= 1 << index
            assert winner is not None
            if age_rank(row.age_s) == age_rank(winner.age_s):
                best_age_count += 1
                completeness = sum(getattr(row, field) is not None for field in COMPLETENESS_FIELDS)
                winner_completeness = sum(
                    getattr(winner, field) is not None for field in COMPLETENESS_FIELDS
                )
                if completeness == winner_completeness:
                    best_completeness_count += 1
    if winner is None or first is None:
        raise ValueError("capacity semantic oracle group is empty")
    if count == 1:
        classification = "singleton"
        reason = "only_member"
    else:
        classification = (
            "core_and_metadata_conflict" if core_bits and metadata_bits else
            "core_state_conflict" if core_bits else
            "metadata_conflict" if metadata_bits else "identical"
        )
        winner_age = age_rank(winner.age_s)
        if best_age_count == 1:
            reason = "min_nonnegative_age" if winner_age[0] == 0 else "closest_zero_negative_age"
        elif best_completeness_count == 1:
            reason = "max_completeness_after_age_tie"
        else:
            reason = "stable_source_order_after_full_tie"
    report_id = canonical_report_id(
        winner.airport_id, winner.aircraft_id, winner.report_timestamp_utc,
    )
    unique = {
        "canonical_report_id": report_id, "selected_raw_row_id": winner.raw_row_id,
        "airport_id": winner.airport_id, "aircraft_id": winner.aircraft_id,
        "report_timestamp_utc": winner.report_timestamp_utc,
        "tail_or_callsign": winner.tail_or_callsign, "lat_deg": winner.lat_deg,
        "lon_deg": winner.lon_deg, "altitude_m": winner.altitude_m,
        "speed_mps": winner.speed_mps, "heading_rad": winner.heading_rad,
        "age_s": winner.age_s, "range_m": winner.range_m,
        "bearing_rad": winner.bearing_rad,
        "altitude_source_gnss": winner.altitude_source_gnss,
        "group_size": count, "is_duplicate": count > 1,
        "duplicate_classification": classification,
        "selection_policy_version": R03_POLICY_VERSION,
        "core_state_difference_bits": core_bits,
        "metadata_difference_bits": metadata_bits,
        "duplicate_conflict_flag": core_bits != 0,
        "metadata_conflict_flag": metadata_bits != 0,
        "any_difference_flag": core_bits != 0 or metadata_bits != 0,
        "parse_quality_bits": winner.parse_quality_bits, "qc_bits": winner.qc_bits,
        "partition_year": winner.partition_year, "partition_month": winner.partition_month,
    }
    conflict = None if not (core_bits or metadata_bits) else {
        "canonical_report_id": report_id, "airport_id": winner.airport_id,
        "partition_year": winner.partition_year, "partition_month": winner.partition_month,
        "duplicate_classification": classification,
        "core_state_difference_bits": core_bits,
        "metadata_difference_bits": metadata_bits, "group_size": count,
        "selected_raw_row_id": winner.raw_row_id,
    }
    return unique, conflict, count, core_bits, metadata_bits, reason


def _oracle_parsed_bytes(row: ParsedRow) -> bytes:
    """Serialize a parsed row by value, independent of its loaded class object."""
    record = asdict(row)
    timestamp = record["report_timestamp_utc"]
    if not isinstance(timestamp, datetime):
        raise TypeError("capacity semantic oracle timestamp is invalid")
    record["report_timestamp_utc"] = timestamp.isoformat(timespec="microseconds")
    return _canonical_json_bytes(record)


def _oracle_parsed_row(payload: bytes) -> ParsedRow:
    """Restore one bounded SQLite payload without relying on pickle identity."""
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise TypeError("capacity semantic oracle row is invalid")
    timestamp = record.get("report_timestamp_utc")
    if not isinstance(timestamp, str):
        raise TypeError("capacity semantic oracle timestamp is invalid")
    record["report_timestamp_utc"] = datetime.fromisoformat(timestamp)
    return ParsedRow(**record)


def _semantic_attestation(
    project: Path,
    selection: CapacitySelectionManifest,
    parameters: R03RunParameters,
    scratch_root: Path,
) -> dict[str, object]:
    """Rebuild selected scientific semantics directly from authenticated source bytes."""
    registry = R03SourceRegistry.from_lineage(project)
    trusted = selection
    selected = {
        item.source_file_id: item.selected_ordinals
        for item in trusted.sources if item.selected_ordinals
    }
    specs = tuple(spec for spec in registry.sources if spec.source_file_id in selected)
    roots = _load_airport_roots(project)
    sources = tuple(verify_source(spec, roots) for spec in specs)
    try:
        identity_record = R03RunIdentity.from_inputs(
            project, sources, parameters, selection_manifest=trusted,
        ).to_dict()
    except RuntimeError as exc:
        if "loaded module origin escapes expected project root" not in str(exc):
            raise
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project / "src")
        child = subprocess.run(
            [
                os.fspath(Path(sys.executable)), "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from airspace_complexity.canonical_points import R03RunParameters; "
                    "from airspace_complexity.r02_runner import _load_airport_roots; "
                    "from airspace_complexity.r03_runner import R03SourceRegistry,verify_source; "
                    "from airspace_complexity.r03_selection import build_capacity_selection; "
                    "from airspace_complexity.build_canonical_points import R03RunIdentity; "
                    "p=Path(sys.argv[1]); params=R03RunParameters.from_mapping(json.loads(sys.argv[2])); "
                    "r=R03SourceRegistry.from_lineage(p); s=build_capacity_selection(r); "
                    "chosen={x.source_file_id for x in s.sources if x.selected_ordinals}; "
                    "sources=tuple(verify_source(x,_load_airport_roots(p)) for x in r.sources if x.source_file_id in chosen); "
                    "print(json.dumps(R03RunIdentity.from_inputs(p,sources,params,selection_manifest=s).to_dict(),sort_keys=True))"
                ),
                str(project), json.dumps(asdict(parameters), sort_keys=True),
            ],
            cwd=project, env=environment, check=False, capture_output=True, text=True,
        )
        if child.returncode != 0:
            raise RuntimeError("capacity independent identity subprocess failed") from exc
        identity_record = json.loads(child.stdout.strip().splitlines()[-1])
    policies = CanonicalPolicies.from_project(project)
    coordinates = {
        airport: (float(values["latitude_deg"]), float(values["longitude_deg"]))
        for airport, values in policies.airports.items()
    }
    raw_to_group, strata = _selected_lineage(trusted)
    zeros = {name: 0 for name in _COUNT_FIELDS}
    by_airport = {airport: dict(zeros) for airport in _AIRPORTS}
    by_stratum = {name: dict(zeros) for name in strata}
    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".semantic-oracle-", dir=scratch) as temporary:
        database_path = Path(temporary) / "oracle.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE parsed(airport TEXT, aircraft TEXT, timestamp TEXT, age_class INTEGER, "
                "year INTEGER, month INTEGER, age_value REAL, completeness INTEGER, source_sequence INTEGER, source_file TEXT, "
                "source_path TEXT, archive_member TEXT, source_row INTEGER, raw_id TEXT, payload BLOB)"
            )
            connection.execute("CREATE TABLE bad(raw_id TEXT PRIMARY KEY, payload BLOB)")
            for source in sources:
                previous: datetime | None = None
                for record in enumerate_selected_source(source, selected[source.spec.source_file_id]):
                    airport, stratum = raw_to_group[record.raw_row_id]
                    by_airport[airport]["enumerated"] += 1
                    by_stratum[stratum]["enumerated"] += 1
                    converted = convert_record(record, policies, coordinates, previous)
                    if isinstance(converted, ParsedRow):
                        previous = converted.report_timestamp_utc
                        rank_class, rank_value = age_rank(converted.age_s)
                        completeness = sum(
                            getattr(converted, field) is not None for field in COMPLETENESS_FIELDS
                        )
                        connection.execute(
                            "INSERT INTO parsed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                converted.airport_id, converted.aircraft_id,
                                converted.report_timestamp_utc.isoformat(timespec="microseconds"),
                                rank_class, converted.partition_year, converted.partition_month,
                                rank_value, completeness, converted.source_sequence,
                                converted.source_file_id, converted.source_relative_path,
                                converted.archive_member or "", converted.source_row_number,
                                converted.raw_row_id,
                                sqlite3.Binary(_oracle_parsed_bytes(converted)),
                            ),
                        )
                        kind = "parsed"
                    elif isinstance(converted, BadRow):
                        connection.execute(
                            "INSERT INTO bad VALUES (?,?)",
                            (
                                converted.raw_row_id,
                                sqlite3.Binary(_canonical_json_bytes(asdict(converted))),
                            ),
                        )
                        kind = "bad"
                    else:
                        raise TypeError("capacity semantic oracle conversion is invalid")
                    by_airport[airport][kind] += 1
                    by_stratum[stratum][kind] += 1
            connection.commit()
            role_digests = {
                role: hashlib.sha256()
                for role in ("parsed", "bad", "unique", "membership", "conflict")
            }
            connection.execute(
                "CREATE TABLE emitted(role TEXT,airport TEXT,year INTEGER,month INTEGER,"
                "aircraft TEXT,timestamp TEXT,report_id TEXT,dedup_rank INTEGER,"
                "raw_id TEXT,payload BLOB)"
            )

            connection.execute(
                "CREATE TABLE groups(airport TEXT, aircraft TEXT, timestamp TEXT, report_id TEXT, "
                "group_size INTEGER, reason TEXT, core_bits INTEGER, metadata_bits INTEGER, "
                "PRIMARY KEY(airport,aircraft,timestamp))"
            )

            def ranked_rows() -> Iterable[ParsedRow]:
                cursor = connection.execute(
                    "SELECT payload FROM parsed ORDER BY airport,aircraft,timestamp,age_class,age_value,"
                    "completeness DESC,source_sequence,source_file,source_path,archive_member,source_row,raw_id"
                )
                return (_oracle_parsed_row(payload) for (payload,) in cursor)

            for natural, rows in groupby(
                ranked_rows(),
                key=lambda row: (row.airport_id, row.aircraft_id, row.report_timestamp_utc),
            ):
                unique, conflict, group_size, core_bits, metadata_bits, reason = (
                    _oracle_group_summary(rows, policies)
                )
                connection.execute(
                    "INSERT INTO emitted VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        "unique", natural[0], unique["partition_year"],
                        unique["partition_month"],
                        natural[1], natural[2].isoformat(timespec="microseconds"),
                        "", 0, "", sqlite3.Binary(json.dumps(
                            _semantic_value(unique), sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False,
                        ).encode("utf-8")),
                    ),
                )
                airport, stratum = raw_to_group[str(unique["selected_raw_row_id"])]
                by_airport[airport]["unique"] += 1
                by_stratum[stratum]["unique"] += 1
                if conflict is not None:
                    connection.execute(
                        "INSERT INTO emitted VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            "conflict", natural[0], conflict["partition_year"],
                            conflict["partition_month"], "", "",
                            str(conflict["canonical_report_id"]), 0, "", sqlite3.Binary(json.dumps(
                            _semantic_value(conflict), sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False,
                        ).encode("utf-8")),
                        ),
                    )
                    by_airport[airport]["conflict"] += 1
                    by_stratum[stratum]["conflict"] += 1
                connection.execute(
                    "INSERT INTO groups VALUES (?,?,?,?,?,?,?,?)",
                    (
                        natural[0], natural[1], natural[2].isoformat(timespec="microseconds"),
                        unique["canonical_report_id"], group_size, reason,
                        core_bits, metadata_bits,
                    ),
                )
            connection.commit()

            # Parsed rows have their own final source-order contract, independent
            # of duplicate winner ranking.
            for (payload,) in connection.execute(
                "SELECT payload FROM parsed ORDER BY airport,year,month,aircraft,timestamp,"
                "source_sequence,source_file,source_path,archive_member,source_row,raw_id"
            ):
                _digest_update(role_digests["parsed"], asdict(_oracle_parsed_row(payload)))

            membership_query = connection.execute(
                "SELECT payload,report_id,group_size,reason,core_bits,metadata_bits,dedup_rank "
                "FROM (SELECT p.payload,g.report_id,g.group_size,g.reason,g.core_bits,g.metadata_bits,"
                "p.raw_id,ROW_NUMBER() OVER (PARTITION BY p.airport,p.aircraft,p.timestamp "
                "ORDER BY p.age_class,p.age_value,p.completeness DESC,p.source_sequence,p.source_file,"
                "p.source_path,p.archive_member,p.source_row,p.raw_id) AS dedup_rank "
                "FROM parsed p JOIN groups g ON p.airport=g.airport AND p.aircraft=g.aircraft "
                "AND p.timestamp=g.timestamp) ORDER BY report_id,dedup_rank,raw_id"
            )
            for payload, report_id, group_size, reason, core_bits, metadata_bits, rank in membership_query:
                row = _oracle_parsed_row(payload)
                age_class, age_value = age_rank(row.age_s)
                membership = {
                    "raw_row_id": row.raw_row_id, "canonical_report_id": report_id,
                    "group_size": group_size, "dedup_rank": rank, "selected": rank == 1,
                    "selection_reason": reason if rank == 1 else "not_selected",
                    "age_rank_class": age_class, "age_rank_value": age_value,
                    "completeness_score": sum(
                        getattr(row, field) is not None for field in COMPLETENESS_FIELDS
                    ),
                    "completeness_denominator": len(COMPLETENESS_FIELDS),
                    "core_state_difference_bits": core_bits,
                    "metadata_difference_bits": metadata_bits,
                    "airport_id": row.airport_id, "source_file_id": row.source_file_id,
                    "source_relative_path": row.source_relative_path,
                    "archive_member": row.archive_member,
                    "source_content_sha256": row.source_content_sha256,
                    "logical_record_ordinal": row.logical_record_ordinal,
                    "source_row_number": row.source_row_number,
                    "source_sequence": row.source_sequence,
                    "partition_year": row.partition_year, "partition_month": row.partition_month,
                }
                connection.execute(
                    "INSERT INTO emitted VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        "membership", row.airport_id, row.partition_year,
                        row.partition_month, "", "", str(report_id), int(rank),
                        row.raw_row_id,
                        sqlite3.Binary(json.dumps(
                            _semantic_value(membership), sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False,
                        ).encode("utf-8")),
                    ),
                )
                airport, stratum = raw_to_group[row.raw_row_id]
                by_airport[airport]["membership"] += 1
                by_stratum[stratum]["membership"] += 1
            for (payload,) in connection.execute("SELECT payload FROM bad ORDER BY raw_id"):
                _digest_update(role_digests["bad"], json.loads(payload))
            role_queries = {
                "unique": (
                    "SELECT payload FROM emitted WHERE role=? "
                    "ORDER BY airport,year,month,aircraft,timestamp"
                ),
                "membership": (
                    "SELECT payload FROM emitted WHERE role=? "
                    "ORDER BY airport,year,month,report_id,dedup_rank,raw_id"
                ),
                "conflict": (
                    "SELECT payload FROM emitted WHERE role=? "
                    "ORDER BY airport,year,month,report_id"
                ),
            }
            for role, query in role_queries.items():
                for (payload,) in connection.execute(query, (role,)):
                    _digest_update(role_digests[role], json.loads(payload))
        finally:
            connection.close()
    total = {name: sum(value[name] for value in by_airport.values()) for name in _COUNT_FIELDS}
    for label, counts in [("counts_total", total), *by_airport.items(), *by_stratum.items()]:
        _validate_conservation(str(label), counts)
    return {
        "schema_version": "r03_task8a_semantic_attestation_v1",
        "identity": identity_record,
        "counts_total": total,
        "counts_by_airport": by_airport,
        "counts_by_stratum": by_stratum,
        "role_digests": {
            role: digest.hexdigest() for role, digest in role_digests.items()
        },
    }


def _artifact_semantic_digests(final_root: Path) -> dict[str, str]:
    role_directories = {
        "parsed": ("parsed_rows", parsed_rows_schema()),
        "bad": ("bad_rows", _bad_rows_schema()),
        "unique": ("unique_reports", unique_reports_schema()),
        "membership": ("report_membership", membership_schema()),
        "conflict": ("duplicate_conflicts", _conflict_schema()),
    }
    result: dict[str, str] = {}
    for name, (directory, schema) in role_directories.items():
        paths = sorted((final_root / directory).rglob("*.parquet"))
        digest = hashlib.sha256()
        for path in paths:
            for row in _iter_mappings(path, schema):
                _digest_update(digest, row)
        result[name] = digest.hexdigest()
    return result


def _validate_artifacts_against_attestation(
    final_root: Path,
    verified: Mapping[str, object],
    attestation: Mapping[str, object],
) -> None:
    if verified.get("identity") != attestation.get("identity"):
        raise ValueError("capacity semantic attestation identity differs from build")
    if _artifact_semantic_digests(final_root) != attestation.get("role_digests"):
        raise ValueError("capacity semantic attestation role digests differ from build")


def _validate_measurement_against_attestation(
    measurement: CapacityMeasurements,
    ledger: object,
    attestation: Mapping[str, object],
) -> None:
    if (
        dict(measurement.counts_total) != attestation.get("counts_total")
        or {name: dict(counts) for name, counts in measurement.counts_by_airport}
        != attestation.get("counts_by_airport")
        or {name: dict(counts) for name, counts in measurement.counts_by_stratum}
        != attestation.get("counts_by_stratum")
    ):
        raise ValueError("capacity measurements differ from semantic attestation")
    identity = attestation.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError("capacity semantic attestation identity is invalid")
    total = attestation["counts_total"]
    if not isinstance(total, Mapping):
        raise ValueError("capacity semantic attestation counts are invalid")
    expected_ledger_counts = {
        "retained_rows": total["enumerated"],
        "parsed_rows": total["parsed"],
        "bad_rows": total["bad"],
        "report_membership": total["membership"],
        "unique_reports": total["unique"],
        "duplicate_excess": int(total["parsed"]) - int(total["unique"]),
    }
    if ledger.get("dataset_id") != identity.get("dataset_id") or ledger.get("counts") != expected_ledger_counts:
        raise ValueError("capacity ledger differs from semantic attestation identity or counts")


def _artifact_identifiers(final_root: Path, role: str, column: str) -> Iterable[str]:
    directory = final_root / role
    paths = sorted(directory.rglob("*.parquet"))
    if not paths:
        raise ValueError(f"capacity artifact role is missing: {role}")
    for path in paths:
        # Read the individual file, not a dataset: artifact paths use Hive-like
        # directory names whose inferred dictionary columns can conflict with
        # the intentionally plain string columns stored in the file.
        parquet = pq.ParquetFile(path)
        try:
            for batch in parquet.iter_batches(batch_size=65_536, columns=[column]):
                values = batch.column(0)
                for index in range(batch.num_rows):
                    yield str(values[index].as_py())
        finally:
            parquet.close()


def _capacity_counts(
    final_root: Path, selection: CapacitySelectionManifest,
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    raw_to_group, strata = _selected_lineage(selection)
    zeros = {name: 0 for name in _COUNT_FIELDS}
    by_airport = {airport: dict(zeros) for airport in _AIRPORTS}
    by_stratum = {name: dict(zeros) for name in strata}
    for identifier, (airport, stratum) in raw_to_group.items():
        by_airport[airport]["enumerated"] += 1
        by_stratum[stratum]["enumerated"] += 1
    role_columns = {
        "parsed": ("parsed_rows", "raw_row_id"),
        "bad": ("bad_rows", "raw_row_id"),
        "membership": ("report_membership", "raw_row_id"),
        "unique": ("unique_reports", "selected_raw_row_id"),
        "conflict": ("duplicate_conflicts", "selected_raw_row_id"),
    }
    for count_name, (role, column) in role_columns.items():
        for identifier in _artifact_identifiers(final_root, role, column):
            try:
                airport, stratum = raw_to_group[identifier]
            except KeyError as exc:
                raise ValueError(f"{role} contains a row outside the selection") from exc
            by_airport[airport][count_name] += 1
            by_stratum[stratum][count_name] += 1
    total = {
        name: sum(counts[name] for counts in by_airport.values())
        for name in _COUNT_FIELDS
    }
    # Reuse the independent measurement validator's conservation rules.
    for label, counts in [("counts_total", total), *by_airport.items(), *by_stratum.items()]:
        _validate_conservation(str(label), counts)
    return total, by_airport, by_stratum


def _finalized_artifacts(final_root: Path) -> list[dict[str, object]]:
    manifest_path = final_root / "canonical_point_manifest.parquet"
    manifest_rows = pq.ParquetFile(manifest_path).read(
        columns=["artifact_role", "relative_path"]
    ).to_pylist()
    records = [
        {
            "artifact_role": str(row["artifact_role"]),
            "relative_path": str(row["relative_path"]),
            "size_bytes": (final_root / str(row["relative_path"])).stat().st_size,
        }
        for row in manifest_rows
    ]
    records.append({
        "artifact_role": "canonical_manifest",
        "relative_path": manifest_path.relative_to(final_root).as_posix(),
        "size_bytes": manifest_path.stat().st_size,
    })
    return sorted(records, key=lambda item: (str(item["artifact_role"]), str(item["relative_path"])))


def _measurement_record(
    observer: CapacityBuildObserver,
    samples: tuple[CapacitySample, ...],
    final_root: Path,
    selection: CapacitySelectionManifest,
) -> dict[str, object]:
    counts_total, by_airport, by_stratum = _capacity_counts(final_root, selection)
    artifacts = _finalized_artifacts(final_root)
    by_role: dict[str, int] = {}
    by_artifact_airport = {"KAGC": 0, "KBTP": 0, "shared": 0}
    for artifact in artifacts:
        role = str(artifact["artifact_role"])
        size = int(artifact["size_bytes"])
        by_role[role] = by_role.get(role, 0) + size
        airport = _artifact_airport(str(artifact["relative_path"])) or "shared"
        by_artifact_airport[airport] += size
    wall_time = samples[-1].monotonic_seconds - samples[0].monotonic_seconds
    if wall_time <= 0:
        raise ValueError("capacity observation wall time is unavailable")
    rows_per_second = Fraction(counts_total["enumerated"], 1) / wall_time
    return {
        "schema_version": "r03_task8a_capacity_metrics_v1",
        "topology": "single_process",
        "worker_count": 1,
        "worker_peak_rss_bytes": None,
        "worker_p99_peak_rss_bytes": None,
        "coordinator_peak_rss_bytes": None,
        "process_peak_rss_bytes": observer.process_peak_rss_bytes,
        "physical_ram_bytes": observer.physical_ram_bytes,
        "wall_time_seconds": wall_time,
        "rows_per_second": rows_per_second,
        "free_bytes_before_run": observer.free_bytes_before_run,
        "free_bytes_at_disk_high_water": observer.free_bytes_at_disk_high_water,
        "scratch_staging_high_water_bytes": observer.scratch_staging_high_water_bytes,
        "total_finalized_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "finalized_artifacts": artifacts,
        "finalized_bytes_by_role": by_role,
        "finalized_bytes_by_airport": by_artifact_airport,
        "counts_total": counts_total,
        "counts_by_airport": by_airport,
        "counts_by_stratum": by_stratum,
        "sample_events": [asdict(sample) for sample in samples],
    }


def _decode_measurement_record(record: Mapping[str, object]) -> dict[str, object]:
    decoded = dict(record)
    decoded["wall_time_seconds"] = _fraction_from_record(
        "wall_time_seconds", decoded["wall_time_seconds"]
    )
    decoded["rows_per_second"] = _fraction_from_record(
        "rows_per_second", decoded["rows_per_second"]
    )
    samples = decoded.get("sample_events")
    if not isinstance(samples, list):
        raise ValueError("capacity sample events are invalid")
    decoded["sample_events"] = [
        {
            **dict(sample),
            "monotonic_seconds": _fraction_from_record(
                "sample monotonic_seconds", sample["monotonic_seconds"]
            ),
        }
        for sample in samples
        if isinstance(sample, Mapping)
    ]
    if len(decoded["sample_events"]) != len(samples):
        raise ValueError("capacity sample event is invalid")
    return decoded


def _static_projection_record(projection: CapacityProjection) -> dict[str, object]:
    excluded = {
        "pilot_disk_capacity", "pilot_memory_capacity", "pilot_capacity_pass",
        "full_disk_capacity", "full_memory_capacity", "full_capacity_ready",
    }
    return {
        field.name: _jsonable(getattr(projection, field.name))
        for field in fields(CapacityProjection)
        if field.name not in excluded
    }


def _command_environment(
    project: Path,
    external: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    started: str,
    ended: str,
    generator_hash: str | None,
    runtime_hash: str | None,
) -> dict[str, object]:
    lock = project / "requirements-lock.txt"
    code_path = Path(__file__).resolve()
    dependency_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    # These are reconstructed from concrete local inputs by the retained
    # verifier. Builder-authored identity strings are deliberately not trusted.
    generator_path = project / "src/airspace_complexity/build_canonical_points.py"
    generator_hash = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    runtime_hash = hashlib.sha256(_canonical_json_bytes({
        "python": sys.version,
        "platform": platform.platform(),
        "pyarrow": pyarrow.__version__,
        "dependency_lock_sha256": dependency_hash,
    })).hexdigest()
    return {
        "schema_version": "r03_task8a_command_environment_v1",
        "command_argv": list(command),
        "started_utc": started,
        "ended_utc": ended,
        "project_root": str(project),
        "external_root": str(external),
        "capacity_gate_code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "generator_code_hash": generator_hash,
        "runtime_environment_hash": runtime_hash,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pyarrow_version": pyarrow.__version__,
        "dependency_lock_sha256": dependency_hash,
        "worker_count": 1,
        "run_parameters": asdict(parameters),
    }


def _validate_command_environment(
    root: Path, environment: Mapping[str, object],
) -> tuple[Path, Path, R03RunParameters]:
    """Bind the retained invocation to concrete, locally reconstructed identities."""
    if set(environment) != _COMMAND_ENVIRONMENT_FIELDS:
        raise ValueError("capacity command environment fields are invalid")
    if environment["schema_version"] != "r03_task8a_command_environment_v1":
        raise ValueError("capacity command environment schema is invalid")
    command = environment["command_argv"]
    if (
        not isinstance(command, list) or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("capacity command environment argv is invalid")
    for field in ("started_utc", "ended_utc"):
        value = environment[field]
        if not isinstance(value, str):
            raise ValueError("capacity command environment timestamp is invalid")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("capacity command environment timestamp is invalid") from exc
    project_value = environment["project_root"]
    external_value = environment["external_root"]
    if not isinstance(project_value, str) or not isinstance(external_value, str):
        raise ValueError("capacity command environment roots are invalid")
    project = Path(project_value).resolve()
    external = Path(external_value).resolve()
    if external != root.parent.parent.resolve() or project != Path(project_value).absolute():
        raise ValueError("capacity command environment roots do not bind evidence")
    parameters_record = environment["run_parameters"]
    if not isinstance(parameters_record, Mapping):
        raise ValueError("capacity command environment parameters are invalid")
    parameters = R03RunParameters.from_mapping(parameters_record)
    if asdict(parameters) != dict(parameters_record):
        raise ValueError("capacity command environment parameters are not canonical")
    if parameters.worker_count != 1 or parameters.max_records_per_airport is not None:
        raise ValueError("capacity command environment parameters are unauthorized")
    lock = project / "requirements-lock.txt"
    generator = project / "src/airspace_complexity/build_canonical_points.py"
    expected_scalars: dict[str, object] = {
        "capacity_gate_code_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "generator_code_hash": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pyarrow_version": pyarrow.__version__,
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "worker_count": 1,
    }
    expected_scalars["runtime_environment_hash"] = hashlib.sha256(
        _canonical_json_bytes({
            "python": expected_scalars["python_version"],
            "platform": expected_scalars["platform"],
            "pyarrow": expected_scalars["pyarrow_version"],
            "dependency_lock_sha256": expected_scalars["dependency_lock_sha256"],
        })
    ).hexdigest()
    if any(environment[name] != value for name, value in expected_scalars.items()):
        raise ValueError("capacity command environment identity hash is invalid")
    return project, external, parameters


def _validate_probe(
    record: object, selection: CapacitySelectionManifest,
    parameters: R03RunParameters,
) -> CapacityProbe:
    fields = {
        "estimated_peak_bytes", "required_free_bytes", "available_free_bytes", "passed"
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise ValueError("capacity probe fields are invalid")
    available = record["available_free_bytes"]
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise ValueError("capacity probe available bytes are invalid")
    expected = capacity_probe(
        sum(dict(selection.selected_rows_by_airport).values()),
        parameters.record_batch_rows,
        parameters.shard_count,
        available,
    )
    if dict(record) != asdict(expected):
        raise ValueError("capacity probe formula is invalid")
    return expected


def _validate_selection_measurement_binding(
    measurement: CapacityMeasurements, selection: CapacitySelectionManifest,
) -> None:
    selected = dict(selection.selected_rows_by_airport)
    airports = {airport: dict(counts) for airport, counts in measurement.counts_by_airport}
    if {airport: counts["enumerated"] for airport, counts in airports.items()} != selected:
        raise ValueError("capacity measurement counts do not bind selected rows")
    expected_strata = {
        f"{item.airport_id}|{item.lineage_month}|{item.storage_route}": item.quota
        for item in selection.strata if item.quota
    }
    actual_strata = {
        name: dict(counts)["enumerated"]
        for name, counts in measurement.counts_by_stratum
    }
    if actual_strata != expected_strata:
        raise ValueError("capacity measurement strata do not bind selection quotas")


def _ledger_verification_record(
    verified: Mapping[str, object], selection: CapacitySelectionManifest,
    parameters: R03RunParameters,
) -> dict[str, object]:
    identity = verified.get("identity")
    counts = verified.get("counts")
    if not isinstance(identity, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("capacity ledger verifier result is incomplete")
    record = {
        "dataset_id": verified.get("dataset_id"),
        "population_kind": identity.get("population_kind"),
        "selection_manifest_sha256": identity.get("selection_manifest_sha256"),
        "run_parameters": identity.get("run_parameters"),
        "counts": dict(counts),
    }
    expected_hash = hashlib.sha256(capacity_selection_bytes(selection)).hexdigest()
    if (
        not isinstance(record["dataset_id"], str)
        or len(record["dataset_id"]) != 64
        or record["population_kind"] != "capacity_sample"
        or record["selection_manifest_sha256"] != expected_hash
        or record["run_parameters"] != _jsonable(asdict(parameters))
        or set(record["counts"]) != {
            "retained_rows", "parsed_rows", "bad_rows", "report_membership",
            "unique_reports", "duplicate_excess",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in record["counts"].values()
        )
    ):
        raise ValueError("capacity ledger verifier result does not bind selection")
    return record


def _validate_ledger_measurement_binding(
    ledger: object, measurement: CapacityMeasurements,
    selection: CapacitySelectionManifest, parameters: R03RunParameters,
) -> None:
    if not isinstance(ledger, Mapping) or set(ledger) != {
        "dataset_id", "population_kind", "selection_manifest_sha256",
        "run_parameters", "counts",
    }:
        raise ValueError("capacity ledger verifier result shape is invalid")
    normalized = _ledger_verification_record(
        {"dataset_id": ledger["dataset_id"], "identity": {
            "population_kind": ledger["population_kind"],
            "selection_manifest_sha256": ledger["selection_manifest_sha256"],
            "run_parameters": ledger["run_parameters"],
        }, "counts": ledger["counts"]},
        selection, parameters,
    )
    if dict(ledger) != normalized:
        raise ValueError("capacity ledger verifier result is noncanonical")
    total = dict(measurement.counts_total)
    counts = normalized["counts"]
    expected = {
        "retained_rows": total["enumerated"],
        "parsed_rows": total["parsed"],
        "bad_rows": total["bad"],
        "report_membership": total["membership"],
        "unique_reports": total["unique"],
        "duplicate_excess": total["parsed"] - total["unique"],
    }
    if counts != expected:
        raise ValueError("capacity ledger verifier counts do not bind measurements")


def _gate_record(document: Mapping[str, object]) -> CapacityGateRecord:
    expected = {field.name for field in fields(CapacityGateRecord)}
    if set(document) != expected | _VERIFIER_EXTRA_FIELDS:
        raise ValueError("capacity verifier result fields are invalid")
    try:
        record = CapacityGateRecord(**{name: document[name] for name in expected})
    except TypeError as exc:
        raise ValueError("capacity verifier result is invalid") from exc
    if record.schema_version != "r03_task8a_capacity_gate_v1":
        raise ValueError("capacity gate schema version is invalid")
    if not isinstance(record.capacity_selection_id, str) or len(record.capacity_selection_id) != 64:
        raise ValueError("capacity selection identifier is invalid")
    for name in (
        "build_started", "pilot_capacity_pass", "full_capacity_ready",
        "performance_review_required", "is_full_population", "r04_authorized",
        "model_training_authorized", "paper_main_result_authorized",
    ):
        if not isinstance(getattr(record, name), bool):
            raise ValueError(f"capacity gate {name} is not boolean")
    if any((record.is_full_population, record.r04_authorized, record.model_training_authorized, record.paper_main_result_authorized)):
        raise ValueError("capacity gate contains a forbidden authorization")
    if record.cleanup_status not in {"CLEAN", "FAILED"}:
        raise ValueError("capacity gate cleanup status is invalid")
    if record.cleanup_status == "FAILED" and (
        record.pilot_capacity_pass or record.full_capacity_ready
    ):
        raise ValueError("failed cleanup cannot authorize capacity")
    expected_decision = (
        "BLOCKED_CLEANUP_REQUIRED" if record.cleanup_status != "CLEAN"
        else "CAPACITY_PASS_PILOT" if record.pilot_capacity_pass
        else "BLOCKED_NEEDS_HUMAN_DECISION"
    )
    if record.decision != expected_decision:
        raise ValueError("capacity gate decision is inconsistent")
    return record


def verify_capacity_gate_evidence(
    selection_root: Path, *, allow_terminal_journal: bool = False,
) -> CapacityGateRecord:
    """Independently re-open and validate the complete five-file envelope."""
    root = Path(os.path.abspath(selection_root))
    _reject_reparse_components(root, root)
    names = {path.name for path in root.iterdir()}
    retained = set(_RETAINED_EVIDENCE)
    permitted = (retained, retained | {"attempts"})
    if allow_terminal_journal:
        permitted = tuple(item | {".capacity-terminal-journal.json"} for item in permitted)
    if names not in permitted:
        raise ValueError("capacity evidence envelope is incomplete or unexpected")
    hashes = _read_json(root, root / "hashes.json")
    if set(hashes) != {"schema_version", "files"} or hashes["schema_version"] != "r03_task8a_hashes_v1":
        raise ValueError("capacity evidence hashes record is invalid")
    claimed = hashes["files"]
    if not isinstance(claimed, Mapping) or set(claimed) != set(_HASHED_EVIDENCE):
        raise ValueError("capacity evidence hashes are incomplete")
    for name in _HASHED_EVIDENCE:
        actual = hashlib.sha256(_pinned_regular_bytes(root, root / name)).hexdigest()
        if claimed[name] != actual:
            raise ValueError(f"capacity evidence hash mismatch: {name}")
    selection = CapacitySelectionManifest.from_record(
        _read_json(root, root / "capacity_selection_manifest.json")
    )
    if capacity_selection_bytes(selection) != _pinned_regular_bytes(
        root, root / "capacity_selection_manifest.json"
    ):
        raise ValueError("capacity selection evidence is not canonical")
    environment = _read_json(root, root / "command_environment.json")
    project, external, parameters = _validate_command_environment(root, environment)
    registry = R03SourceRegistry.from_lineage(project)
    trusted_selection = load_and_verify_capacity_selection(
        root / "capacity_selection_manifest.json", registry
    )
    if selection != trusted_selection or root.name != selection.selection_id:
        raise ValueError("capacity evidence selection does not bind trusted lineage")
    metrics = _read_json(root, root / "capacity_metrics.json")
    if set(metrics) != _METRICS_FIELDS or metrics.get("schema_version") != "r03_task8a_capacity_metrics_evidence_v1":
        raise ValueError("capacity metrics envelope is invalid")
    probe = _validate_probe(metrics["probe"], selection, parameters)
    status = metrics.get("measurement_status")
    if status == "MEASURED":
        if not probe.passed:
            raise ValueError("capacity measured evidence contradicts failed probe")
        raw_measurements = metrics.get("measurements")
        if not isinstance(raw_measurements, Mapping):
            raise ValueError("capacity measurements are absent")
        measurement = CapacityMeasurements.from_record(
            _decode_measurement_record(raw_measurements)
        )
        _validate_selection_measurement_binding(measurement, selection)
        _validate_ledger_measurement_binding(
            metrics["ledger_verifier_result"], measurement, selection, parameters,
        )
        with tempfile.TemporaryDirectory(
            prefix=".retained-semantic-", dir=root
        ) as semantic_scratch:
            expected_attestation = _semantic_attestation(
                project, selection, parameters, Path(semantic_scratch),
            )
        if metrics.get("semantic_attestation") != expected_attestation:
            raise ValueError("capacity semantic attestation is not trusted-source derived")
        _validate_measurement_against_attestation(
            measurement, metrics["ledger_verifier_result"], expected_attestation,
        )
        recomputed = project_capacity(
            measurement,
            dict(selection.population_rows_by_airport),
            dict(selection.selected_rows_by_airport),
            0,
        )
        if metrics.get("projection") != _static_projection_record(recomputed):
            raise ValueError("capacity projection evidence is not reproducible")
        expected_review = measurement.wall_time_seconds > 5 * 3600
    elif status in {"NOT_RUN_PROBE_BLOCKED", "NOT_RUN_BUILD_FAILED"}:
        if metrics.get("measurements") is not None or metrics.get("projection") is not None:
            raise ValueError("not-run capacity evidence fabricates measurements")
        expected_review = False
        if metrics["ledger_verifier_result"] is not None:
            raise ValueError("not-run capacity evidence fabricates ledger verification")
        if metrics["semantic_attestation"] is not None:
            raise ValueError("not-run capacity evidence fabricates semantic attestation")
        failure = metrics["build_failure"]
        if status == "NOT_RUN_PROBE_BLOCKED" and failure is not None:
            raise ValueError("probe-blocked evidence fabricates a build failure")
        if status == "NOT_RUN_BUILD_FAILED" and (
            not isinstance(failure, Mapping)
            or set(failure) != {"stage", "exception_type"}
            or not all(isinstance(failure[name], str) and failure[name] for name in failure)
        ):
            raise ValueError("capacity build failure diagnostics are invalid")
        if status == "NOT_RUN_PROBE_BLOCKED" and probe.passed:
            raise ValueError("capacity probe-blocked status contradicts probe")
        if status == "NOT_RUN_BUILD_FAILED" and not probe.passed:
            raise ValueError("capacity build-failed status contradicts probe")
    else:
        raise ValueError("capacity measurement status is invalid")
    if metrics.get("performance_review_required") is not expected_review:
        raise ValueError("capacity performance review claim is invalid")
    verifier = _read_json(root, root / "capacity_verifier_result.json")
    record = _gate_record(verifier)
    if record.capacity_selection_id != selection.selection_id:
        raise ValueError("capacity gate does not bind the selection")
    if record.measurement_status != status or record.performance_review_required is not expected_review:
        raise ValueError("capacity verifier result does not bind metrics")
    expected_build_started = status != "NOT_RUN_PROBE_BLOCKED"
    if record.build_started is not expected_build_started:
        raise ValueError("capacity build-started claim is invalid")
    command = environment["command_argv"]
    assert isinstance(command, list) and all(isinstance(item, str) for item in command)
    expected_invocation_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            _invocation_binding(project, external, parameters, command)
        )
    ).hexdigest()
    _validate_retained_attempt_evidence(
        root,
        record.cleanup_status,
        verifier["retained_attempt"],
        expected_invocation_sha256=expected_invocation_sha256,
    )
    if status == "MEASURED":
        free_after = verifier.get("free_bytes_after_cleanup")
        if isinstance(free_after, bool) or not isinstance(free_after, int) or free_after < 0:
            raise ValueError("capacity cleanup free bytes are invalid")
        final_projection = project_capacity(
            measurement,
            dict(selection.population_rows_by_airport),
            dict(selection.selected_rows_by_airport),
            free_after,
        )
        inequalities = {
            "pilot_disk_capacity": final_projection.pilot_disk_capacity,
            "pilot_memory_capacity": final_projection.pilot_memory_capacity,
            "full_disk_capacity": final_projection.full_disk_capacity,
            "full_memory_capacity": final_projection.full_memory_capacity,
        }
        if any(verifier.get(name) is not value for name, value in inequalities.items()):
            raise ValueError("capacity final inequalities are not reproducible")
        blocked_cleanup = record.cleanup_status == "FAILED"
        if (
            (not blocked_cleanup and (
                record.pilot_capacity_pass is not final_projection.pilot_capacity_pass
                or record.full_capacity_ready is not final_projection.full_capacity_ready
            ))
            or verifier.get("ledger_verification") is not True
            or verifier.get("metrics_verification") is not True
        ):
            raise ValueError("capacity final projection claims are invalid")
    else:
        expected_ledger = False
        expected_metrics = True
        free_after = verifier.get("free_bytes_after_cleanup")
        valid_free_after = (
            not isinstance(free_after, bool)
            and isinstance(free_after, int)
            and free_after >= 0
        )
        if (
            not valid_free_after
            or any(verifier.get(name) is not False for name in (
                "pilot_disk_capacity", "pilot_memory_capacity",
                "full_disk_capacity", "full_memory_capacity",
            ))
            or verifier.get("ledger_verification") is not expected_ledger
            or verifier.get("metrics_verification") is not expected_metrics
            or record.pilot_capacity_pass or record.full_capacity_ready
        ):
            raise ValueError("capacity ledger or not-run projection claims are invalid")
    return record


def _run_capacity_gate_locked(
    project_root: Path,
    external_root: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    *,
    resume: bool = False,
) -> CapacityGateRecord:
    """Run the one-process R03 pilot and retain only independently verified evidence."""
    project = Path(project_root).resolve()
    external = Path(external_root).resolve()
    if not isinstance(parameters, R03RunParameters):
        raise TypeError("parameters must be R03RunParameters")
    if parameters.worker_count != 1 or parameters.max_records_per_airport is not None:
        raise ValueError("capacity gate requires worker_count=1 and no legacy row cap")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError("capacity command argv is invalid")
    started = _utc_now()
    registry = R03SourceRegistry.from_lineage(project)
    selection = build_capacity_selection(registry)
    selection_root = _ensure_selection_root(external, selection.selection_id)
    partial_names = {path.name for path in selection_root.iterdir()}
    if ".capacity-terminal-journal.json" in partial_names:
        sealed_selection = load_and_verify_capacity_selection(
            selection_root / "capacity_selection_manifest.json", registry
        )
        expectations = _terminal_expectations_for_metrics(
            selection_root, sealed_selection, project, external, parameters, command,
        )
        journal = load_terminal_journal(selection_root, expectations)
        if journal is None:
            raise ValueError("capacity terminal journal disappeared during recovery")
        if journal.state in {TerminalState.OUTCOME_CLEAN, TerminalState.OUTCOME_FAILED,
                             TerminalState.SEALING, TerminalState.COMMITTED}:
            return seal_terminal_evidence(
                selection_root, journal, project, external, parameters, command,
            )
        raise ValueError("capacity terminal journal is not an evidence-ready outcome")
    existing = _assert_finalized_root_state(selection_root)
    if existing is not None:
        environment = _read_json(selection_root, selection_root / "command_environment.json")
        expected_environment = _command_environment(
            project, external, parameters, command,
            str(environment.get("started_utc")), str(environment.get("ended_utc")),
            None, None,
        )
        invocation_fields = {
            "command_argv", "project_root", "external_root", "run_parameters",
            "capacity_gate_code_sha256", "generator_code_hash",
            "runtime_environment_hash", "dependency_lock_sha256",
        }
        if any(environment.get(name) != expected_environment[name] for name in invocation_fields):
            raise ValueError("capacity rerun invocation identity differs")
        return existing
    if partial_names == {
        "capacity_selection_manifest.json", "capacity_metrics.json", "attempts",
    }:
        sealed_selection = load_and_verify_capacity_selection(
            selection_root / "capacity_selection_manifest.json", registry
        )
        return _resume_metrics_attempt(
            selection_root, project, external, parameters, command, started,
            sealed_selection,
        )
    _seal_bytes(
        selection_root,
        selection_root / "capacity_selection_manifest.json",
        capacity_selection_bytes(selection),
    )
    # Independent reconstruction happens from the sealed bytes, never the in-memory object.
    selection = load_and_verify_capacity_selection(
        selection_root / "capacity_selection_manifest.json", registry
    )
    selected_rows = sum(dict(selection.selected_rows_by_airport).values())
    probe = capacity_probe(
        selected_rows,
        parameters.record_batch_rows,
        parameters.shard_count,
        _available_bytes(selection_root),
    )
    probe_record = asdict(probe)
    if not probe.passed:
        metrics = {
            "schema_version": "r03_task8a_capacity_metrics_evidence_v1",
            "measurement_status": "NOT_RUN_PROBE_BLOCKED",
            "measurements": None,
            "projection": None,
            "probe": probe_record,
            "performance_review_required": False,
            "ledger_verifier_result": None,
            "semantic_attestation": None,
            "build_failure": None,
        }
        if (selection_root / "attempts").exists():
            raise ValueError("probe-blocked capacity gate created an attempt directory")
        _seal_bytes(
            selection_root,
            selection_root / "capacity_metrics.json",
            _canonical_json_bytes(metrics),
        )
        if _read_json(selection_root, selection_root / "capacity_metrics.json") != metrics:
            raise ValueError("probe-blocked capacity metrics changed after sealing")
        return _terminal_outcome_record(
            selection_root, selection, project, external, parameters, command,
            started_utc=started,
            pilot_capacity_pass=False,
            full_capacity_ready=False,
            performance_review_required=False,
        )

    attempts_root = selection_root / "attempts"
    attempts_root.mkdir()
    attempt = _new_attempt_path(selection_root)
    if not attempt.resolve(strict=False).is_relative_to(attempts_root.resolve()):
        raise ValueError("capacity attempt path escapes the selection root")
    attempt.mkdir()
    _seal_bytes(
        attempt, attempt / "invocation.json",
        _canonical_json_bytes(_invocation_binding(project, external, parameters, command)),
    )
    build_root = attempt / "build"
    build_root.mkdir()
    if _path_identity(attempt)[0] != _path_identity(selection_root)[0]:
        raise ValueError("capacity attempt is not on the evidence volume")
    observer: CapacityBuildObserver | None = None
    result = None
    measurement_record: dict[str, object] | None = None
    measurement_status = "NOT_RUN_BUILD_FAILED"
    failure: dict[str, str] | None = None
    stage = "observer_start"
    try:
        observer = CapacityBuildObserver(
            attempt, selection_root, scratch_root=build_root,
            available_bytes=_available_bytes,
        )
        stage = "build_canonical_points"
        result = build_canonical_points(
            project,
            build_root,
            parameters,
            resume,
            selection_manifest=selection,
            publication_mode="external_evidence_only",
            observer=observer,
        )
        stage = "observer_close"
        samples = observer.close()
        stage = "verify_canonical_manifest"
        verified = verify_canonical_manifest(result.manifest_path, project)
        if verified.get("dataset_id") != result.identity.dataset_id:
            raise ValueError("capacity build identity changed during independent verification")
        stage = "independent_semantic_attestation"
        semantic_attestation = _semantic_attestation(
            project, selection, parameters, attempt,
        )
        _validate_artifacts_against_attestation(
            result.final_root, verified, semantic_attestation,
        )
        stage = "measurement_record"
        measurement_record = _measurement_record(
            observer=observer,
            samples=samples,
            final_root=result.final_root,
            selection=selection,
        )
        observer = None
    except Exception as exc:
        failure = {"stage": stage, "exception_type": type(exc).__name__}
        if observer is not None:
            try:
                observer.close()
            except Exception:
                pass
        measurement_record = None
    if measurement_record is not None:
        measurement_status = "MEASURED"
        measurement = CapacityMeasurements.from_record(measurement_record)
        static_projection = project_capacity(
            measurement,
            dict(selection.population_rows_by_airport),
            dict(selection.selected_rows_by_airport),
            0,
        )
        performance_review = measurement.wall_time_seconds > 5 * 3600
        metrics = {
            "schema_version": "r03_task8a_capacity_metrics_evidence_v1",
            "measurement_status": measurement_status,
            "measurements": measurement_record,
            "projection": _static_projection_record(static_projection),
            "probe": probe_record,
            "performance_review_required": performance_review,
            "ledger_verifier_result": _ledger_verification_record(
                verified, selection, parameters,
            ),
            "semantic_attestation": semantic_attestation,
            "build_failure": None,
        }
    else:
        performance_review = False
        metrics = {
            "schema_version": "r03_task8a_capacity_metrics_evidence_v1",
            "measurement_status": measurement_status,
            "measurements": None,
            "projection": None,
            "probe": probe_record,
            "performance_review_required": False,
            "ledger_verifier_result": None,
            "semantic_attestation": None,
            "build_failure": failure,
        }
    _seal_bytes(
        selection_root, selection_root / "capacity_metrics.json", _canonical_json_bytes(metrics)
    )
    # Independent parse/validation occurs before destructive cleanup.
    reread_metrics = _read_json(selection_root, selection_root / "capacity_metrics.json")
    if measurement_record is not None:
        CapacityMeasurements.from_record(
            _decode_measurement_record(reread_metrics["measurements"])  # type: ignore[arg-type]
        )
    if measurement_record is not None:
        projection = project_capacity(
            CapacityMeasurements.from_record(measurement_record),
            dict(selection.population_rows_by_airport),
            dict(selection.selected_rows_by_airport),
            # This pre-cleanup measurement is a conservative lower bound: the
            # only permitted terminal action may release, never consume, space.
            _available_bytes(selection_root),
        )
        pilot_pass = projection.pilot_capacity_pass
        full_ready = projection.full_capacity_ready
    else:
        projection = None
        pilot_pass = False
        full_ready = False
    return _terminal_outcome_record(
        selection_root, selection, project, external, parameters, command,
        started_utc=started,
        pilot_capacity_pass=pilot_pass,
        full_capacity_ready=full_ready,
        performance_review_required=performance_review,
    )


def run_capacity_gate(
    project_root: Path,
    external_root: Path,
    parameters: R03RunParameters,
    command: Sequence[str],
    *,
    resume: bool = False,
) -> CapacityGateRecord:
    """Run Gate A while one OS process owns the selection-root lifecycle."""
    project = Path(project_root).resolve()
    external = Path(external_root).resolve()
    if not isinstance(parameters, R03RunParameters):
        raise TypeError("parameters must be R03RunParameters")
    registry = R03SourceRegistry.from_lineage(project)
    selection_id = build_capacity_selection(registry).selection_id
    with _capacity_gate_lock(external, selection_id):
        return _run_capacity_gate_locked(
            project, external, parameters, command, resume=resume,
        )
