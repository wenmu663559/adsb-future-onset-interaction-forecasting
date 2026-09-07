"""Causal, trajectory-derived proxies for terminal ATC task demand.

These indicators are interpretability references.  They are not controller
workload labels and do not reproduce the full task definitions in Jurinic et
al. (2024), which require flight plans, aircraft type, and planned 4D routes.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence


METRES_PER_NAUTICAL_MILE = 1852.0
METRES_PER_FOOT = 0.3048
TASK_INDICATOR_NAMES = (
    "current_aircraft_count",
    "screening_pair_count",
    "inbound_count_proxy",
    "outbound_count_proxy",
    "mixed_inbound_outbound_pairs",
    "closing_pair_count_10nm",
    "projected_separation_task_count_120s",
    "maneuvering_aircraft_count",
    "heading_dispersion",
)


def _position_m(report: Mapping[str, object], latitude: float, longitude: float):
    cosine = math.cos(math.radians(latitude))
    return (
        (float(report["lon_deg"]) - longitude) * 111_320.0 * cosine,
        (float(report["lat_deg"]) - latitude) * 111_320.0,
        float(report["altitude_m"]),
    )


def _horizontal_distance(left, right):
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _angle_difference(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _scene_snapshots(scene: Mapping[str, object]):
    cutoff_us = int(scene["cutoff_us"])
    reports_by_aircraft: dict[str, list[Mapping[str, object]]] = {}
    for reports in scene["history"]:
        for report in reports:
            if int(report["timestamp_us"]) <= cutoff_us:
                reports_by_aircraft.setdefault(str(report["aircraft_id"]), []).append(report)

    current = {}
    prior_30 = {}
    prior_60 = {}
    for aircraft_id, reports in reports_by_aircraft.items():
        reports.sort(key=lambda report: int(report["timestamp_us"]))
        current[aircraft_id] = reports[-1]
        candidates_30 = [
            report for report in reports
            if 20.0 <= (cutoff_us - int(report["timestamp_us"])) / 1_000_000.0 <= 45.0
        ]
        candidates_60 = [
            report for report in reports
            if 45.0 <= (cutoff_us - int(report["timestamp_us"])) / 1_000_000.0 <= 90.0
        ]
        if candidates_30:
            prior_30[aircraft_id] = min(
                candidates_30,
                key=lambda report: abs(
                    (cutoff_us - int(report["timestamp_us"])) / 1_000_000.0 - 30.0
                ),
            )
        if candidates_60:
            prior_60[aircraft_id] = min(
                candidates_60,
                key=lambda report: abs(
                    (cutoff_us - int(report["timestamp_us"])) / 1_000_000.0 - 60.0
                ),
            )
    return current, prior_30, prior_60


def extract_pairwise_cpa_features(
    scene: Mapping[str, object], *, airport_lat_deg: float, airport_lon_deg: float,
    horizons_seconds: Sequence[int] = (30, 120, 300),
) -> dict[str, float | int]:
    """Extract observed pair geometry and constant-velocity CPA features."""

    horizons = tuple(sorted(set(int(value) for value in horizons_seconds)))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("CPA horizons must be positive")
    current, prior_30, _ = _scene_snapshots(scene)
    aircraft_ids = sorted(current)
    horizontal_distances: list[float] = []
    vertical_distances: list[float] = []
    active_pairs = 0
    closing_pairs = 0
    cpa_counts = {horizon: 0 for horizon in horizons}
    projected_horizontal = {horizon: [] for horizon in horizons}
    projected_vertical = {horizon: [] for horizon in horizons}

    for left_index, left_id in enumerate(aircraft_ids):
        for right_id in aircraft_ids[left_index + 1:]:
            left_now = current[left_id]
            right_now = current[right_id]
            left_position = _position_m(left_now, airport_lat_deg, airport_lon_deg)
            right_position = _position_m(right_now, airport_lat_deg, airport_lon_deg)
            relative_position = tuple(
                left_position[index] - right_position[index] for index in range(3)
            )
            current_horizontal = math.hypot(
                relative_position[0], relative_position[1]
            )
            current_vertical = abs(relative_position[2])
            horizontal_distances.append(current_horizontal)
            vertical_distances.append(current_vertical)
            if (
                current_horizontal < 5.0 * METRES_PER_NAUTICAL_MILE
                and current_vertical < 1000.0 * METRES_PER_FOOT
            ):
                active_pairs += 1

            left_earlier = prior_30.get(left_id)
            right_earlier = prior_30.get(right_id)
            if left_earlier is None or right_earlier is None:
                continue
            left_old = _position_m(left_earlier, airport_lat_deg, airport_lon_deg)
            right_old = _position_m(right_earlier, airport_lat_deg, airport_lon_deg)
            earlier_horizontal = _horizontal_distance(left_old, right_old)
            if (
                current_horizontal < 10.0 * METRES_PER_NAUTICAL_MILE
                and earlier_horizontal > current_horizontal
            ):
                closing_pairs += 1
            left_seconds = max(
                (int(left_now["timestamp_us"]) - int(left_earlier["timestamp_us"]))
                / 1_000_000.0,
                1.0,
            )
            right_seconds = max(
                (int(right_now["timestamp_us"]) - int(right_earlier["timestamp_us"]))
                / 1_000_000.0,
                1.0,
            )
            relative_velocity = tuple(
                (left_position[index] - left_old[index]) / left_seconds
                - (right_position[index] - right_old[index]) / right_seconds
                for index in range(3)
            )
            horizontal_speed_squared = (
                relative_velocity[0] ** 2 + relative_velocity[1] ** 2
            )
            unconstrained_cpa_seconds = (
                0.0
                if horizontal_speed_squared <= 1e-9
                else -(
                    relative_position[0] * relative_velocity[0]
                    + relative_position[1] * relative_velocity[1]
                ) / horizontal_speed_squared
            )
            for horizon in horizons:
                closest_seconds = max(
                    0.0, min(float(horizon), unconstrained_cpa_seconds)
                )
                projected = tuple(
                    relative_position[index]
                    + relative_velocity[index] * closest_seconds
                    for index in range(3)
                )
                horizontal = math.hypot(projected[0], projected[1])
                vertical = abs(projected[2])
                projected_horizontal[horizon].append(horizontal)
                projected_vertical[horizon].append(vertical)
                if (
                    horizontal < 5.0 * METRES_PER_NAUTICAL_MILE
                    and vertical < 1000.0 * METRES_PER_FOOT
                ):
                    cpa_counts[horizon] += 1

    result: dict[str, float | int] = {
        "observed_pair_count": len(horizontal_distances),
        "current_active_pair_count_5nm_1000ft": active_pairs,
        "closing_pair_count_10nm": closing_pairs,
        "mean_pair_horizontal_distance_m": (
            statistics.fmean(horizontal_distances) if horizontal_distances else 100_000.0
        ),
        "minimum_pair_vertical_separation_m": min(
            vertical_distances, default=100_000.0
        ),
    }
    for horizon in horizons:
        result[f"cpa_pair_count_{horizon}s"] = cpa_counts[horizon]
        result[f"minimum_projected_horizontal_cpa_m_{horizon}s"] = min(
            projected_horizontal[horizon], default=100_000.0
        )
        result[f"minimum_projected_vertical_separation_m_{horizon}s"] = min(
            projected_vertical[horizon], default=100_000.0
        )
    return result


def extract_task_indicators(
    scene: Mapping[str, object],
    *,
    airport_lat_deg: float,
    airport_lon_deg: float,
) -> dict[str, float | int]:
    """Extract causal task-demand proxies from one scene's observed history."""

    current, prior_30, prior_60 = _scene_snapshots(scene)
    aircraft_ids = sorted(current)
    aircraft_count = len(aircraft_ids)
    inbound_count = 0
    outbound_count = 0
    maneuvering_count = 0

    for aircraft_id in aircraft_ids:
        now = current[aircraft_id]
        earlier = prior_60.get(aircraft_id) or prior_30.get(aircraft_id)
        if earlier is None:
            continue
        current_position = _position_m(now, airport_lat_deg, airport_lon_deg)
        earlier_position = _position_m(earlier, airport_lat_deg, airport_lon_deg)
        radial_change = math.hypot(*current_position[:2]) - math.hypot(*earlier_position[:2])
        if radial_change <= -0.5 * METRES_PER_NAUTICAL_MILE:
            inbound_count += 1
        elif radial_change >= 0.5 * METRES_PER_NAUTICAL_MILE:
            outbound_count += 1

        if (
            _angle_difference(
                float(now["heading_rad"]), float(earlier["heading_rad"])
            ) >= math.radians(20.0)
            or abs(float(now["speed_mps"]) - float(earlier["speed_mps"]))
            >= 10.0 * 0.514444
            or abs(float(now["altitude_m"]) - float(earlier["altitude_m"]))
            >= 500.0 * METRES_PER_FOOT
        ):
            maneuvering_count += 1

    closing_pairs = 0
    projected_separation_tasks = 0
    for left_index, left_id in enumerate(aircraft_ids):
        for right_id in aircraft_ids[left_index + 1:]:
            left_now = current[left_id]
            right_now = current[right_id]
            left_earlier = prior_30.get(left_id)
            right_earlier = prior_30.get(right_id)
            if left_earlier is None or right_earlier is None:
                continue

            left_position = _position_m(left_now, airport_lat_deg, airport_lon_deg)
            right_position = _position_m(right_now, airport_lat_deg, airport_lon_deg)
            left_old_position = _position_m(
                left_earlier, airport_lat_deg, airport_lon_deg
            )
            right_old_position = _position_m(
                right_earlier, airport_lat_deg, airport_lon_deg
            )
            current_distance = _horizontal_distance(left_position, right_position)
            earlier_distance = _horizontal_distance(
                left_old_position, right_old_position
            )
            if (
                current_distance < 10.0 * METRES_PER_NAUTICAL_MILE
                and earlier_distance > current_distance
            ):
                closing_pairs += 1

            left_seconds = max(
                (int(left_now["timestamp_us"]) - int(left_earlier["timestamp_us"]))
                / 1_000_000.0,
                1.0,
            )
            right_seconds = max(
                (int(right_now["timestamp_us"]) - int(right_earlier["timestamp_us"]))
                / 1_000_000.0,
                1.0,
            )
            relative_position = tuple(
                left_position[i] - right_position[i] for i in range(3)
            )
            relative_velocity = tuple(
                (left_position[i] - left_old_position[i]) / left_seconds
                - (right_position[i] - right_old_position[i]) / right_seconds
                for i in range(3)
            )
            horizontal_speed_squared = (
                relative_velocity[0] ** 2 + relative_velocity[1] ** 2
            )
            if horizontal_speed_squared <= 1e-9:
                closest_seconds = 0.0
            else:
                closest_seconds = -(
                    relative_position[0] * relative_velocity[0]
                    + relative_position[1] * relative_velocity[1]
                ) / horizontal_speed_squared
                closest_seconds = max(0.0, min(120.0, closest_seconds))
            projected = tuple(
                relative_position[i] + relative_velocity[i] * closest_seconds
                for i in range(3)
            )
            if (
                math.hypot(projected[0], projected[1])
                < 5.0 * METRES_PER_NAUTICAL_MILE
                and abs(projected[2]) < 1000.0 * METRES_PER_FOOT
            ):
                projected_separation_tasks += 1

    headings = [float(current[aircraft_id]["heading_rad"]) for aircraft_id in aircraft_ids]
    if headings:
        mean_cosine = sum(math.cos(value) for value in headings) / len(headings)
        mean_sine = sum(math.sin(value) for value in headings) / len(headings)
        heading_dispersion = 1.0 - math.hypot(mean_cosine, mean_sine)
    else:
        heading_dispersion = 0.0

    return {
        "current_aircraft_count": aircraft_count,
        "screening_pair_count": aircraft_count * (aircraft_count - 1) // 2,
        "inbound_count_proxy": inbound_count,
        "outbound_count_proxy": outbound_count,
        "mixed_inbound_outbound_pairs": inbound_count * outbound_count,
        "closing_pair_count_10nm": closing_pairs,
        "projected_separation_task_count_120s": projected_separation_tasks,
        "maneuvering_aircraft_count": maneuvering_count,
        "heading_dispersion": heading_dispersion,
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average_rank = (start + end + 2.0) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator <= 1e-15:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_indicator_relationships(
    records: Sequence[Mapping[str, object]],
    *,
    metric_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Summarize descriptive convergence without treating scenes as independent."""

    target = [math.log1p(float(record["target"])) for record in records]
    result = {}
    for metric_name in metric_names:
        metric = [float(record[metric_name]) for record in records]
        transformed = [math.log1p(value) for value in metric]
        daily_correlations = []
        days = sorted({str(record["day"]) for record in records})
        for day in days:
            indices = [
                index for index, record in enumerate(records)
                if str(record["day"]) == day
            ]
            correlation = _spearman(
                [metric[index] for index in indices],
                [target[index] for index in indices],
            )
            if correlation is not None:
                daily_correlations.append(correlation)

        strata: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            strata[(
                str(record["day"]), int(record["current_aircraft_count"])
            )].append(index)
        metric_residuals = []
        target_residuals = []
        for indices in strata.values():
            metric_mean = statistics.fmean(transformed[index] for index in indices)
            target_mean = statistics.fmean(target[index] for index in indices)
            for index in indices:
                metric_residuals.append(transformed[index] - metric_mean)
                target_residuals.append(target[index] - target_mean)
        residual_correlation = _pearson(metric_residuals, target_residuals)
        scene_correlation = _spearman(metric, target)

        result[metric_name] = {
            "nonzero_scenes": sum(value > 0.0 for value in metric),
            "nonzero_rate": round(sum(value > 0.0 for value in metric) / len(metric), 4),
            "scene_spearman_log_target": (
                round(scene_correlation, 4) if scene_correlation is not None else None
            ),
            "daily_spearman_median": (
                round(statistics.median(daily_correlations), 4)
                if daily_correlations else None
            ),
            "daily_spearman_iqr": (
                [
                    round(_quantile(daily_correlations, 0.25), 4),
                    round(_quantile(daily_correlations, 0.75), 4),
                ] if daily_correlations else None
            ),
            "positive_days": sum(value > 0.0 for value in daily_correlations),
            "evaluable_days": len(daily_correlations),
            "day_and_count_stratified_residual_correlation": (
                round(residual_correlation, 4)
                if residual_correlation is not None else None
            ),
        }
    return result
