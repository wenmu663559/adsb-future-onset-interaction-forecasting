"""Lightweight contracts shared by R03 dataset and training environments."""

from __future__ import annotations

import math
from collections.abc import Sequence


def empty_pair_aggregate(active_nodes: int) -> float | None:
    """Return the neutral relation aggregate when fewer than two nodes exist."""
    if isinstance(active_nodes, bool) or not isinstance(active_nodes, int):
        raise TypeError("active_nodes must be an integer")
    if active_nodes < 0:
        raise ValueError("active_nodes must be nonnegative")
    return 0.0 if active_nodes < 2 else None


def finite_or_none(value: float) -> float | None:
    """Preserve finite metrics while making numerical failure JSON-safe."""
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def summarize_temporal_controls(
    *,
    counts: Sequence[float],
    min_distances_m: Sequence[float],
    altitude_stds_m: Sequence[float],
    speed_stds_mps: Sequence[float],
) -> tuple[float, ...]:
    """Summarize causal scene history without importing training dependencies."""
    sequences = (counts, min_distances_m, altitude_stds_m, speed_stds_mps)
    lengths = {len(values) for values in sequences}
    if lengths != {len(counts)} or not counts:
        raise ValueError("temporal control sequences must be nonempty and equally sized")
    materialized = tuple(tuple(float(value) for value in values) for values in sequences)
    if any(
        not math.isfinite(value) or value < 0.0
        for values in materialized for value in values
    ):
        raise ValueError("temporal control values must be finite and nonnegative")
    count_values, distance_values, altitude_values, speed_values = materialized
    size = len(count_values)

    def mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    count_mean = mean(count_values)
    count_std = math.sqrt(
        math.fsum((value - count_mean) ** 2 for value in count_values) / size
    )
    denominator = max(1, size - 1)
    return (
        count_mean,
        count_std,
        (count_values[-1] - count_values[0]) / denominator,
        mean(distance_values),
        (distance_values[-1] - distance_values[0]) / denominator,
        mean(altitude_values),
        mean(speed_values),
        math.fsum(value > 0.0 for value in count_values) / size,
    )
