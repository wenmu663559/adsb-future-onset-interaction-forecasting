"""Bounded online multi-aircraft complexity pilot helpers."""

from __future__ import annotations

import bisect
import hashlib
import math
import statistics
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from airspace_complexity.canonical_points import (
    CanonicalPolicies,
    ParsedRow,
    convert_record,
    resolve_group,
)
from airspace_complexity.r03_runner import (
    R03SourceRegistry,
    R03SourceSpec,
    enumerate_selected_source,
    enumerate_source,
    verify_source,
)
from airspace_complexity.r03_experiment_contract import empty_pair_aggregate, finite_or_none


_MICROSECONDS = 1_000_000
_EARTH_RADIUS_M = 6_371_008.8


def _coordinate_distance_m(
    left_lat_deg: float, left_lon_deg: float,
    right_lat_deg: float, right_lon_deg: float,
) -> float:
    lat1 = math.radians(left_lat_deg)
    lat2 = math.radians(right_lat_deg)
    dlat = lat2 - lat1
    dlon = math.radians(right_lon_deg - left_lon_deg)
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


@dataclass(frozen=True)
class TerminalVolume:
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    radius_m: float
    floor_offset_m: float
    ceiling_offset_m: float

    def __post_init__(self) -> None:
        for name in (
            "latitude_deg", "longitude_deg", "elevation_m", "radius_m",
            "floor_offset_m", "ceiling_offset_m",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg is outside its domain")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg is outside its domain")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.floor_offset_m >= self.ceiling_offset_m:
            raise ValueError("terminal volume floor must be below its ceiling")


@dataclass(frozen=True)
class PilotReport:
    airport_id: str
    source_file_id: str
    source_day: str
    aircraft_id: str
    timestamp_us: int
    lat_deg: float
    lon_deg: float
    altitude_m: float
    speed_mps: float
    heading_rad: float

    def __post_init__(self) -> None:
        for name in ("airport_id", "source_file_id", "source_day", "aircraft_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if isinstance(self.timestamp_us, bool) or not isinstance(self.timestamp_us, int):
            raise TypeError("timestamp_us must be an integer")
        if self.timestamp_us < 0:
            raise ValueError("timestamp_us must be nonnegative")
        for name in ("lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("lat_deg is outside its domain")
        if not -180.0 <= self.lon_deg <= 180.0:
            raise ValueError("lon_deg is outside its domain")


def filter_terminal_reports(
    reports: Iterable[PilotReport],
    volumes: Mapping[str, TerminalVolume],
) -> tuple[tuple[PilotReport, ...], dict[str, dict[str, int]]]:
    """Apply a common physical terminal-volume rule at every airport."""
    audit = {
        airport: {
            "input_reports": 0,
            "accepted_reports": 0,
            "outside_horizontal": 0,
            "outside_vertical": 0,
        }
        for airport in sorted(volumes)
    }
    accepted: list[PilotReport] = []
    for report in reports:
        if report.airport_id not in volumes:
            raise ValueError(f"missing terminal volume for {report.airport_id}")
        volume = volumes[report.airport_id]
        counts = audit[report.airport_id]
        counts["input_reports"] += 1
        horizontal = _coordinate_distance_m(
            report.lat_deg, report.lon_deg,
            volume.latitude_deg, volume.longitude_deg,
        )
        if horizontal > volume.radius_m:
            counts["outside_horizontal"] += 1
            continue
        lower = volume.elevation_m + volume.floor_offset_m
        upper = volume.elevation_m + volume.ceiling_offset_m
        if not lower <= report.altitude_m <= upper:
            counts["outside_vertical"] += 1
            continue
        counts["accepted_reports"] += 1
        accepted.append(report)
    return tuple(accepted), audit


@dataclass(frozen=True)
class FutureProxyTarget:
    observation_start_us: int
    observation_end_us: int
    min_horizontal_separation_m: float | None
    min_vertical_separation_m: float | None
    proximity_event_count: int
    future_onset_proximity_event_count: int
    cutoff_active_pair_count: int


@dataclass(frozen=True)
class SceneExample:
    scene_id: str
    airport_id: str
    source_file_ids: tuple[str, ...]
    source_day: str
    cutoff_us: int
    history: tuple[tuple[PilotReport, ...], ...]
    targets: Mapping[int, FutureProxyTarget]


@dataclass(frozen=True)
class ExtractionResult:
    reports: tuple[PilotReport, ...]
    opened_source_ids: tuple[str, ...]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class ExperimentRunSpec:
    model: str
    seed: int
    split_sha256: str
    target_sha256: str
    warmup_iterations: int = 10
    timed_iterations: int = 100
    airport_adversarial_weight: float = 0.0
    uses_site_calibration: bool = False
    uses_pair_interactions: bool = False
    target_horizons_seconds: tuple[int, ...] = (120,)
    training_signal: str = "proxy_supervised"
    probe_horizons_seconds: tuple[int, ...] = ()


def experiment_plan(
    *, split_sha256: str, target_sha256: str,
    seeds: tuple[int, ...],
) -> tuple[ExperimentRunSpec, ...]:
    for name, value in (
        ("split_sha256", split_sha256), ("target_sha256", target_sha256)
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be lowercase SHA-256")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("seeds must be distinct integers")
    models = (
        "deepsets", "masked_temporal", "invariant_relation",
        "calibrated_invariant_relation", "calibrated_relation_no_adversary",
        "calibrated_invariant_no_relation",
        "multitask_invariant_relation",
        "airport_conditioned_multitask",
        "masked_relation_ssl",
    )
    adversarial_models = {
        "invariant_relation", "calibrated_invariant_relation",
        "calibrated_invariant_no_relation",
    }
    return tuple(
        ExperimentRunSpec(
            model=model,
            seed=seed,
            split_sha256=split_sha256,
            target_sha256=target_sha256,
            airport_adversarial_weight=(0.1 if model in adversarial_models else 0.0),
            uses_site_calibration=(
                model.startswith("calibrated_")
                or model == "airport_conditioned_multitask"
            ),
            uses_pair_interactions=("relation" in model and "no_relation" not in model),
            target_horizons_seconds=(
                () if model == "masked_relation_ssl" else
                (30, 120, 300) if model in {
                    "multitask_invariant_relation", "airport_conditioned_multitask"
                } else (120,)
            ),
            training_signal=(
                "masked_self_supervision" if model == "masked_relation_ssl"
                else "proxy_supervised"
            ),
            probe_horizons_seconds=(
                (30, 120, 300) if model == "masked_relation_ssl" else ()
            ),
        )
        for model in models
        for seed in seeds
    )


def select_pilot_sources(
    registry: R03SourceRegistry, *, sources_per_airport: int,
    exact_source_ids: tuple[str, ...] | None = None,
) -> tuple[R03SourceSpec, ...]:
    if not isinstance(registry, R03SourceRegistry):
        raise TypeError("registry must be an R03SourceRegistry")
    if sources_per_airport <= 0:
        raise ValueError("sources_per_airport must be positive")
    if exact_source_ids is not None:
        if (
            not isinstance(exact_source_ids, tuple)
            or not exact_source_ids
            or len(set(exact_source_ids)) != len(exact_source_ids)
        ):
            raise ValueError("exact_source_ids must be a nonempty unique tuple")
        by_id = {spec.source_file_id: spec for spec in registry.sources}
        missing = sorted(set(exact_source_ids) - set(by_id))
        if missing:
            raise ValueError(f"unknown exact source ids: {missing}")
        selected = tuple(by_id[source_id] for source_id in exact_source_ids)
        if any(spec.authenticated_row_count <= 0 for spec in selected):
            raise ValueError("exact sources must contain authenticated rows")
        return tuple(
            sorted(selected, key=lambda spec: (spec.airport_id, spec.source_file_id))
        )
    selected: list[R03SourceSpec] = []
    airports = sorted({spec.airport_id for spec in registry.sources})
    for airport in airports:
        candidates = [
            spec for spec in registry.sources
            if spec.airport_id == airport and spec.authenticated_row_count > 0
        ]
        ranked = sorted(
            candidates,
            key=lambda spec: (
                -spec.authenticated_row_count,
                hashlib.sha256(
                    b"r03-online-source-v1\x00"
                    + registry.registry_id.encode("ascii")
                    + b"\x00"
                    + spec.source_file_id.encode("utf-8")
                ).digest(),
                spec.source_file_id,
            ),
        )
        if len(ranked) < sources_per_airport:
            raise ValueError(f"insufficient authenticated sources for {airport}")
        selected.extend(ranked[:sources_per_airport])
    return tuple(sorted(selected, key=lambda spec: (spec.airport_id, spec.source_file_id)))


def extract_contiguous_reports(
    registry: R03SourceRegistry,
    airport_roots: Mapping[str, Path],
    project_root: Path,
    *,
    sources_per_airport: int,
    max_hours_per_source: int,
    exact_source_ids: tuple[str, ...] | None = None,
) -> ExtractionResult:
    if max_hours_per_source <= 0:
        raise ValueError("max_hours_per_source must be positive")
    policies = CanonicalPolicies.from_project(Path(project_root))
    coordinates = {
        airport: (float(values["latitude_deg"]), float(values["longitude_deg"]))
        for airport, values in policies.airports.items()
    }
    selected = select_pilot_sources(
        registry,
        sources_per_airport=sources_per_airport,
        exact_source_ids=exact_source_ids,
    )
    reports: list[PilotReport] = []
    summaries: list[dict[str, object]] = []
    opened: list[str] = []
    interval_us = max_hours_per_source * 3600 * _MICROSECONDS
    for spec in selected:
        verified = verify_source(spec, airport_roots)
        opened.append(spec.source_file_id)
        previous_timestamp = None
        parsed_groups: dict[tuple[str, str, object], list[ParsedRow]] = {}
        bad_count = 0
        enumerated_records = (
            enumerate_selected_source(
                verified,
                tuple(range(1, spec.authenticated_row_count + 1)),
            )
            if spec.r01_disposition == "drop_terminal_record_in_R02"
            else enumerate_source(verified)
        )
        for enumerated in enumerated_records:
            converted = convert_record(
                enumerated, policies, coordinates, previous_timestamp
            )
            if not isinstance(converted, ParsedRow):
                bad_count += 1
                continue
            timestamp = converted.report_timestamp_utc
            if timestamp.tzinfo is not timezone.utc:
                raise ValueError("converted timestamp is not canonical UTC")
            previous_timestamp = timestamp
            timestamp_us = int(timestamp.timestamp() * _MICROSECONDS)
            values = (
                converted.lat_deg,
                converted.lon_deg,
                converted.altitude_m,
                converted.speed_mps,
                converted.heading_rad,
            )
            if any(value is None or not math.isfinite(float(value)) for value in values):
                bad_count += 1
                continue
            parsed_groups.setdefault(
                (converted.airport_id, converted.aircraft_id, timestamp), []
            ).append(converted)
        accepted: list[PilotReport] = []
        parsed_count = sum(len(rows) for rows in parsed_groups.values())
        for _, duplicate_rows in sorted(parsed_groups.items(), key=lambda item: item[0]):
            unique = resolve_group(duplicate_rows, policies).unique_row
            timestamp = unique["report_timestamp_utc"]
            if not hasattr(timestamp, "date"):
                raise TypeError("resolved timestamp is not a datetime")
            accepted.append(PilotReport(
                airport_id=str(unique["airport_id"]),
                source_file_id=spec.source_file_id,
                source_day=timestamp.date().isoformat(),
                aircraft_id=str(unique["aircraft_id"]),
                timestamp_us=int(timestamp.timestamp() * _MICROSECONDS),
                lat_deg=float(unique["lat_deg"]),
                lon_deg=float(unique["lon_deg"]),
                altitude_m=float(unique["altitude_m"]),
                speed_mps=float(unique["speed_mps"]),
                heading_rad=float(unique["heading_rad"]),
            ))
        accepted.sort(key=lambda row: (row.timestamp_us, row.aircraft_id))
        activity: dict[int, tuple[set[str], int]] = {}
        activity_bucket_us = 10 * 60 * _MICROSECONDS
        for report in accepted:
            bucket = report.timestamp_us // activity_bucket_us
            aircraft, count = activity.setdefault(bucket, (set(), 0))
            aircraft.add(report.aircraft_id)
            activity[bucket] = (aircraft, count + 1)
        if activity:
            selected_bucket = min(
                activity,
                key=lambda bucket: (
                    -len(activity[bucket][0]), -activity[bucket][1], bucket,
                ),
            )
            interval_start_us = selected_bucket * activity_bucket_us
            interval_end_us = interval_start_us + interval_us
            accepted = [
                report for report in accepted
                if interval_start_us <= report.timestamp_us <= interval_end_us
            ]
        else:
            interval_start_us = None
            interval_end_us = None
        reports.extend(accepted)
        summaries.append({
            "airport_id": spec.airport_id,
            "source_file_id": spec.source_file_id,
            "source_content_sha256": verified.source_content_sha256,
            "relative_path": spec.relative_path,
            "accepted_report_count": len(accepted),
            "parsed_report_count": parsed_count,
            "duplicate_excess_count": parsed_count - len(accepted),
            "bad_or_incomplete_count": bad_count,
            "interval_start_us": interval_start_us,
            "interval_end_us": interval_end_us,
        })
    reports.sort(
        key=lambda row: (
            row.airport_id, row.source_file_id, row.source_day,
            row.timestamp_us, row.aircraft_id,
        )
    )
    manifest: dict[str, object] = {
        "schema_version": "r03_online_complexity_pilot_selection_v1",
        "decision": "EXPLORATORY_ONLINE_COMPLEXITY_PILOT",
        "registry_id": registry.registry_id,
        "sources_per_airport": sources_per_airport,
        "selection_mode": (
            "exact_frozen_source_ids"
            if exact_source_ids is not None
            else "ranked_sources_per_airport"
        ),
        "max_hours_per_source": max_hours_per_source,
        "selected_source_ids": [spec.source_file_id for spec in selected],
        "sources": summaries,
        "report_count": len(reports),
    }
    return ExtractionResult(
        reports=tuple(reports),
        opened_source_ids=tuple(opened),
        manifest=manifest,
    )


def normalize_scene_tensor(x, split, *, frozen_normalization=None):
    """Normalize observed scene features from train rows or frozen statistics."""

    import numpy as np

    values = np.asarray(x)
    split_values = np.asarray(split)
    if frozen_normalization is None:
        train = split_values == "train"
        if not train.any():
            raise ValueError("training rows are required without frozen normalization")
        observed = values[train, :, :, 6] > 0
        numeric = values[train, :, :, :6][observed]
        mean = numeric.mean(axis=0)
        scale = numeric.std(axis=0)
        scale[scale < 1e-6] = 1.0
    else:
        mean = np.asarray(frozen_normalization[0], dtype=values.dtype)
        scale = np.asarray(frozen_normalization[1], dtype=values.dtype)
        if mean.shape != (6,) or scale.shape != (6,) or np.any(scale <= 0):
            raise ValueError("frozen normalization must contain positive six-vectors")
    mask = values[:, :, :, 6:7]
    values[:, :, :, :6] = ((values[:, :, :, :6] - mean) / scale) * mask
    return values, mean, scale


def _horizontal_distance_m(left: PilotReport, right: PilotReport) -> float:
    return _coordinate_distance_m(
        left.lat_deg, left.lon_deg, right.lat_deg, right.lon_deg
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile values must be nonempty")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def assign_group_splits(groups: Iterable[str]) -> dict[str, str]:
    materialized = sorted(set(groups))
    if not materialized or any(not isinstance(group, str) or not group for group in materialized):
        raise ValueError("split groups must be nonempty strings")
    ranked = sorted(
        materialized,
        key=lambda group: (
            hashlib.sha256(b"r03-online-split-v1\x00" + group.encode("utf-8")).digest(),
            group,
        ),
    )
    train_end = max(1, (len(ranked) * 60) // 100)
    validation_end = max(train_end + (1 if len(ranked) >= 3 else 0), (len(ranked) * 80) // 100)
    validation_end = min(validation_end, len(ranked))
    result: dict[str, str] = {}
    for index, group in enumerate(ranked):
        if index < train_end:
            result[group] = "train"
        elif index < validation_end:
            result[group] = "validation"
        else:
            result[group] = "test"
    return result


def assign_airport_day_splits(groups: Iterable[str]) -> dict[str, str]:
    """Split days independently per airport so every airport is evaluated."""
    materialized = sorted(set(groups))
    if not materialized or any(
        not isinstance(group, str) or "\x00" not in group
        for group in materialized
    ):
        raise ValueError("airport-day groups must use '<airport>\\0<day>'")

    by_airport: dict[str, list[str]] = {}
    for group in materialized:
        airport, day = group.split("\x00", 1)
        if not airport or not day:
            raise ValueError("airport-day groups must use '<airport>\\0<day>'")
        by_airport.setdefault(airport, []).append(group)

    result: dict[str, str] = {}
    for airport, airport_groups in sorted(by_airport.items()):
        if len(airport_groups) < 3:
            raise ValueError(
                f"airport {airport!r} needs at least three independent days "
                "for train/validation/test"
            )
        result.update(assign_group_splits(airport_groups))
    return result


def _latest_step(
    by_aircraft: Mapping[str, tuple[list[int], list[PilotReport]]],
    cutoff_us: int,
    stale_us: int,
) -> tuple[PilotReport, ...]:
    rows: list[PilotReport] = []
    for aircraft, (timestamps, reports) in sorted(by_aircraft.items()):
        index = bisect.bisect_right(timestamps, cutoff_us) - 1
        if index < 0:
            continue
        report = reports[index]
        if cutoff_us - report.timestamp_us <= stale_us:
            rows.append(report)
    return tuple(rows)


def _future_target(
    by_aircraft: Mapping[str, tuple[list[int], list[PilotReport]]],
    cutoff_us: int,
    horizon_seconds: int,
    step_seconds: int,
    stale_seconds: int,
    horizontal_proximity_m: float,
    vertical_proximity_m: float,
) -> FutureProxyTarget:
    end_us = cutoff_us + horizon_seconds * _MICROSECONDS
    horizontal: list[float] = []
    vertical: list[float] = []
    proximity_events = 0
    future_onset_events = 0
    active_pairs: set[tuple[str, str]] = set()
    step_us = step_seconds * _MICROSECONDS
    stale_us = stale_seconds * _MICROSECONDS
    cutoff_rows = _latest_step(by_aircraft, cutoff_us, stale_us)
    cutoff_pairs = {
        tuple(sorted((left.aircraft_id, right.aircraft_id)))
        for left_index, left in enumerate(cutoff_rows)
        for right in cutoff_rows[left_index + 1:]
        if (
            _horizontal_distance_m(left, right) < horizontal_proximity_m
            and abs(left.altitude_m - right.altitude_m) < vertical_proximity_m
        )
    }
    onset_active_pairs = set(cutoff_pairs)
    for sample_us in range(cutoff_us + step_us, end_us + 1, step_us):
        timestamp_rows = tuple(
            row for row in _latest_step(by_aircraft, sample_us, stale_us)
            if row.timestamp_us > cutoff_us
        )
        current_pairs: set[tuple[str, str]] = set()
        for left_index, left in enumerate(timestamp_rows):
            for right in timestamp_rows[left_index + 1:]:
                h_sep = _horizontal_distance_m(left, right)
                v_sep = abs(left.altitude_m - right.altitude_m)
                horizontal.append(h_sep)
                vertical.append(v_sep)
                if h_sep < horizontal_proximity_m and v_sep < vertical_proximity_m:
                    current_pairs.add(tuple(sorted((left.aircraft_id, right.aircraft_id))))
        proximity_events += len(current_pairs - active_pairs)
        future_onset_events += len(current_pairs - onset_active_pairs)
        active_pairs = current_pairs
        onset_active_pairs = current_pairs
    return FutureProxyTarget(
        observation_start_us=cutoff_us + 1,
        observation_end_us=end_us,
        min_horizontal_separation_m=min(horizontal) if horizontal else None,
        min_vertical_separation_m=min(vertical) if vertical else None,
        proximity_event_count=proximity_events,
        future_onset_proximity_event_count=future_onset_events,
        cutoff_active_pair_count=len(cutoff_pairs),
    )


def build_scene_examples(
    reports: Iterable[PilotReport], *, history_seconds: int,
    step_seconds: int, horizons_seconds: tuple[int, ...], stale_seconds: int,
    min_aircraft: int, max_aircraft: int,
    horizontal_proximity_m: float = 5 * 1_852.0,
    vertical_proximity_m: float = 1_000 * 0.3048,
) -> list[SceneExample]:
    if history_seconds <= 0 or step_seconds <= 0 or history_seconds % step_seconds:
        raise ValueError("history_seconds must be a positive multiple of step_seconds")
    if stale_seconds < 0:
        raise ValueError("stale_seconds must be nonnegative")
    if min_aircraft < 2 or max_aircraft < min_aircraft:
        raise ValueError("aircraft bounds are invalid")
    if (
        not math.isfinite(horizontal_proximity_m)
        or not math.isfinite(vertical_proximity_m)
        or horizontal_proximity_m <= 0.0
        or vertical_proximity_m <= 0.0
    ):
        raise ValueError("proximity thresholds must be finite and positive")
    if not horizons_seconds or any(horizon <= 0 for horizon in horizons_seconds):
        raise ValueError("horizons_seconds must be positive")
    materialized = list(reports)
    if not materialized:
        return []
    if any(not isinstance(report, PilotReport) for report in materialized):
        raise TypeError("reports must contain PilotReport values")

    grouped: dict[tuple[str, str], list[PilotReport]] = {}
    for report in materialized:
        grouped.setdefault(
            (report.airport_id, report.source_day), []
        ).append(report)

    step_us = step_seconds * _MICROSECONDS
    stale_us = stale_seconds * _MICROSECONDS
    history_steps = history_seconds // step_seconds
    max_horizon_us = max(horizons_seconds) * _MICROSECONDS
    output: list[SceneExample] = []
    for group_key, group_reports in sorted(grouped.items()):
        group_reports.sort(
            key=lambda row: (row.timestamp_us, row.aircraft_id, row.source_file_id)
        )
        by_aircraft_mutable: dict[str, tuple[list[int], list[PilotReport]]] = {}
        for report in group_reports:
            timestamps, aircraft_reports = by_aircraft_mutable.setdefault(
                report.aircraft_id, ([], [])
            )
            timestamps.append(report.timestamp_us)
            aircraft_reports.append(report)
        first_cutoff = group_reports[0].timestamp_us + (history_steps - 1) * step_us
        last_cutoff = group_reports[-1].timestamp_us - max_horizon_us
        cutoff_us = first_cutoff
        while cutoff_us <= last_cutoff:
            history = tuple(
                _latest_step(
                    by_aircraft_mutable,
                    cutoff_us - (history_steps - 1 - step_index) * step_us,
                    stale_us,
                )
                for step_index in range(history_steps)
            )
            current = history[-1]
            if len(current) >= min_aircraft:
                allowed = set(sorted(row.aircraft_id for row in current)[:max_aircraft])
                limited_history = tuple(
                    tuple(row for row in step if row.aircraft_id in allowed)
                    for step in history
                )
                airport, day = group_key
                sources = tuple(sorted({
                    row.source_file_id for step in limited_history for row in step
                }))
                output.append(SceneExample(
                    scene_id=f"{airport}\x00{day}\x00{cutoff_us}",
                    airport_id=airport,
                    source_file_ids=sources,
                    source_day=day,
                    cutoff_us=cutoff_us,
                    history=limited_history,
                    targets={
                        horizon: _future_target(
                            by_aircraft_mutable, cutoff_us, horizon,
                            step_seconds, stale_seconds,
                            horizontal_proximity_m, vertical_proximity_m,
                        )
                        for horizon in sorted(set(horizons_seconds))
                    },
                ))
            cutoff_us += step_us
    return output


def sparse_neighbor_indices(
    rows: Sequence[PilotReport], *, top_k: int
) -> dict[str, tuple[str, ...]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len({row.aircraft_id for row in rows}) != len(rows):
        raise ValueError("rows must contain at most one report per aircraft")
    result: dict[str, tuple[str, ...]] = {}
    for row in rows:
        ranked = sorted(
            (
                (_horizontal_distance_m(row, candidate), candidate.aircraft_id)
                for candidate in rows if candidate.aircraft_id != row.aircraft_id
            ),
            key=lambda item: (item[0], item[1]),
        )
        result[row.aircraft_id] = tuple(aircraft for _, aircraft in ranked[:top_k])
    return result


def traditional_controls(example: SceneExample) -> dict[str, float]:
    current = example.history[-1]
    if not current:
        raise ValueError("scene has no current aircraft")
    headings = [row.heading_rad for row in current]
    resultant = math.hypot(
        statistics.fmean(math.sin(value) for value in headings),
        statistics.fmean(math.cos(value) for value in headings),
    )
    pair_distances = [
        _horizontal_distance_m(left, right)
        for left_index, left in enumerate(current)
        for right in current[left_index + 1:]
    ]
    return {
        "aircraft_count": float(len(current)),
        "min_pair_distance_m": min(pair_distances, default=0.0),
        "altitude_std_m": statistics.pstdev(row.altitude_m for row in current),
        "speed_std_mps": statistics.pstdev(row.speed_mps for row in current),
        "heading_dispersion": 1.0 - resultant,
    }


def latency_summary(durations_ns: Sequence[int]) -> dict[str, float | int]:
    if not durations_ns or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in durations_ns
    ):
        raise ValueError("durations_ns must contain nonnegative integers")
    milliseconds = [value / 1_000_000.0 for value in durations_ns]
    return {
        "iterations": len(milliseconds),
        "p50_ms": _percentile(milliseconds, 0.50),
        "p95_ms": _percentile(milliseconds, 0.95),
        "max_ms": max(milliseconds),
    }
