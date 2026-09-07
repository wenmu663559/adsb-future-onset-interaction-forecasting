"""Causal baselines for future-onset ADS-B proximity events."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from airspace_complexity.r03_online_complexity import _coordinate_distance_m
from airspace_complexity.r03_task_indicators import (
    extract_pairwise_cpa_features,
    extract_task_indicators,
)


METRES_PER_NAUTICAL_MILE = 1852.0
METRES_PER_FOOT = 0.3048
A_FEATURE_NAMES = (
    "current_aircraft_count",
    "current_altitude_mean_m",
    "current_altitude_std_m",
    "current_speed_mean_mps",
    "current_speed_std_mps",
    "current_altitude_range_m",
    "current_min_pair_distance_m",
    "current_low_altitude_fraction",
)
B_ADDITION_NAMES = (
    "observed_pair_count",
    "current_active_pair_count_5nm_1000ft",
    "closing_pair_count_10nm",
    "mean_pair_horizontal_distance_m",
    "minimum_pair_vertical_separation_m",
)
C_ADDITION_NAMES = tuple(
    name
    for horizon in (30, 120, 300)
    for name in (
        f"cpa_pair_count_{horizon}s",
        f"minimum_projected_horizontal_cpa_m_{horizon}s",
        f"minimum_projected_vertical_separation_m_{horizon}s",
    )
)


def ofat_threshold_settings(
    *, horizontal_nm: Sequence[int | float], vertical_ft: Sequence[int | float],
    primary_horizontal_nm: int | float, primary_vertical_ft: int | float,
) -> tuple[tuple[str, float, float], ...]:
    """Construct a one-factor-at-a-time threshold design around one primary pair."""

    horizontal = tuple(float(value) for value in horizontal_nm)
    vertical = tuple(float(value) for value in vertical_ft)
    primary_h = float(primary_horizontal_nm)
    primary_v = float(primary_vertical_ft)
    if (
        not horizontal or not vertical
        or primary_h not in horizontal or primary_v not in vertical
        or len(set(horizontal)) != len(horizontal)
        or len(set(vertical)) != len(vertical)
        or any(not math.isfinite(value) or value <= 0.0 for value in horizontal + vertical)
    ):
        raise ValueError("threshold settings must be unique positive values containing the primary")
    pairs = (
        {(value, primary_v) for value in horizontal}
        | {(primary_h, value) for value in vertical}
    )
    return tuple(
        (f"h{h:g}nm_v{v:g}ft", h, v)
        for h, v in sorted(pairs)
    )


def _current_reports(scene: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    cutoff_us = int(scene["cutoff_us"])
    current: dict[str, Mapping[str, object]] = {}
    for step in scene["history"]:
        for report in step:
            timestamp_us = int(report["timestamp_us"])
            if timestamp_us > cutoff_us:
                raise ValueError("scene history contains a post-cutoff report")
            aircraft_id = str(report["aircraft_id"])
            previous = current.get(aircraft_id)
            if previous is None or timestamp_us > int(previous["timestamp_us"]):
                current[aircraft_id] = report
    return current


def build_feature_ladder(
    scene: Mapping[str, object], *, airport_lat_deg: float, airport_lon_deg: float,
) -> dict[str, dict[str, float]]:
    """Return nested A=current, B=pairwise, C=CPA causal feature sets."""

    current = _current_reports(scene)
    reports = [current[aircraft_id] for aircraft_id in sorted(current)]
    altitudes = [float(report["altitude_m"]) for report in reports]
    speeds = [float(report["speed_mps"]) for report in reports]
    cosine = math.cos(math.radians(airport_lat_deg))
    positions = [
        (
            (float(report["lat_deg"]) - airport_lat_deg) * 111_320.0,
            (float(report["lon_deg"]) - airport_lon_deg) * 111_320.0 * cosine,
        )
        for report in reports
    ]
    pair_distances = [
        math.dist(left, right)
        for left_index, left in enumerate(positions)
        for right in positions[left_index + 1:]
    ]
    a_values = (
        float(len(reports)),
        statistics.fmean(altitudes),
        statistics.pstdev(altitudes),
        statistics.fmean(speeds),
        statistics.pstdev(speeds),
        max(altitudes) - min(altitudes),
        min(pair_distances, default=100_000.0),
        sum(value < 3_000.0 for value in altitudes) / len(altitudes),
    )
    a = dict(zip(A_FEATURE_NAMES, a_values, strict=True))
    pairwise = extract_pairwise_cpa_features(
        scene,
        airport_lat_deg=airport_lat_deg,
        airport_lon_deg=airport_lon_deg,
        horizons_seconds=(30, 120, 300),
    )
    b = {**a, **{name: float(pairwise[name]) for name in B_ADDITION_NAMES}}
    c = {**b, **{name: float(pairwise[name]) for name in C_ADDITION_NAMES}}
    return {"A": a, "B": b, "C": c}


def causal_baseline_features(
    scene: Mapping[str, object], *, airport_lat_deg: float, airport_lon_deg: float,
    horizontal_nm: float = 5.0, vertical_ft: float = 1000.0,
) -> dict[str, int]:
    """Extract observable persistence, exposure, and CPA baseline features."""

    if horizontal_nm <= 0.0 or vertical_ft <= 0.0:
        raise ValueError("proximity thresholds must be positive")
    current = _current_reports(scene)
    aircraft_ids = sorted(current)
    active_pairs = 0
    for left_index, left_id in enumerate(aircraft_ids):
        left = current[left_id]
        for right_id in aircraft_ids[left_index + 1:]:
            right = current[right_id]
            horizontal = _coordinate_distance_m(
                float(left["lat_deg"]), float(left["lon_deg"]),
                float(right["lat_deg"]), float(right["lon_deg"]),
            )
            vertical = abs(float(left["altitude_m"]) - float(right["altitude_m"]))
            if (
                horizontal < horizontal_nm * METRES_PER_NAUTICAL_MILE
                and vertical < vertical_ft * METRES_PER_FOOT
            ):
                active_pairs += 1

    task_features = extract_task_indicators(
        scene,
        airport_lat_deg=airport_lat_deg,
        airport_lon_deg=airport_lon_deg,
    )
    aircraft_count = len(aircraft_ids)
    return {
        "current_aircraft_count": aircraft_count,
        "current_pair_opportunity": aircraft_count * (aircraft_count - 1) // 2,
        "cutoff_active_pair_count": active_pairs,
        "constant_velocity_cpa_pair_count_120s": int(
            task_features["projected_separation_task_count_120s"]
        ),
    }


def training_prevalence(labels: Sequence[int | bool]) -> float:
    """Return a training-only binary prevalence for a constant risk baseline."""

    values = tuple(labels)
    if not values or any(value not in (0, 1, False, True) for value in values):
        raise ValueError("labels must be a nonempty binary sequence")
    return sum(int(value) for value in values) / len(values)


def exposure_rate_predictions(
    train_counts: Sequence[int | float],
    train_pair_opportunities: Sequence[int | float],
    evaluation_pair_opportunities: Sequence[int | float],
) -> tuple[float, ...]:
    """Fit one training-only event-per-available-pair rate and apply it."""

    counts = tuple(float(value) for value in train_counts)
    train_exposure = tuple(float(value) for value in train_pair_opportunities)
    evaluation_exposure = tuple(float(value) for value in evaluation_pair_opportunities)
    if not counts or len(counts) != len(train_exposure):
        raise ValueError("training counts and pair opportunities must align")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in counts + train_exposure + evaluation_exposure
    ):
        raise ValueError("counts and pair opportunities must be finite nonnegative values")
    total_exposure = sum(train_exposure)
    rate = (
        sum(counts) / total_exposure
        if total_exposure > 0.0
        else sum(counts) / len(counts)
    )
    return tuple(rate * exposure for exposure in evaluation_exposure)


def summarize_reconstruction(
    scene_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize target reconstruction without treating scene rows as independent."""

    if not scene_records:
        raise ValueError("scene records must be nonempty")
    horizon_totals: dict[str, dict[str, int]] = {}
    days: set[str] = set()
    for scene in scene_records:
        days.add(str(scene["source_day"]))
        for horizon, raw_target in dict(scene["targets"]).items():
            target = dict(raw_target)
            legacy = int(target["proximity_event_count"])
            onset = int(target["future_onset_proximity_event_count"])
            cutoff_active = int(target["cutoff_active_pair_count"])
            if min(legacy, onset, cutoff_active) < 0 or onset > legacy:
                raise ValueError("invalid future-onset target accounting")
            totals = horizon_totals.setdefault(str(horizon), {
                "legacy_event_sum": 0,
                "future_onset_sum": 0,
                "legacy_minus_onset_sum": 0,
                "cutoff_active_pair_sum": 0,
                "missing_future_pair_scenes": 0,
            })
            totals["legacy_event_sum"] += legacy
            totals["future_onset_sum"] += onset
            totals["legacy_minus_onset_sum"] += legacy - onset
            totals["cutoff_active_pair_sum"] += cutoff_active
            totals["missing_future_pair_scenes"] += int(
                target["min_horizontal_separation_m"] is None
            )
    return {
        "scene_count": len(scene_records),
        "independent_days": len(days),
        "horizons": dict(sorted(horizon_totals.items(), key=lambda item: int(item[0]))),
    }
