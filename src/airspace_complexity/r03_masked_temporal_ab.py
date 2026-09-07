"""Predeclared decision rule for the bounded masked-temporal A/B experiment."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping


def decide_masked_temporal_ab(
    *,
    baseline_mae_by_seed: Mapping[int, float],
    candidate_mae_by_seed: Mapping[int, float],
    day_mae_delta: Mapping[str, float],
    closing_edge_delta_change: float,
    maneuver_representation_delta_change: float,
) -> dict[str, object]:
    """Promote B only when prediction and secondary evidence are both stable."""

    seeds = sorted(set(baseline_mae_by_seed) & set(candidate_mae_by_seed))
    if not seeds or not day_mae_delta:
        raise ValueError("paired seed and day results are required")
    baseline_mean = statistics.fmean(baseline_mae_by_seed[seed] for seed in seeds)
    candidate_mean = statistics.fmean(candidate_mae_by_seed[seed] for seed in seeds)
    relative_improvement = (
        (baseline_mean - candidate_mean) / baseline_mean if baseline_mean else 0.0
    )
    seed_wins = sum(
        candidate_mae_by_seed[seed] < baseline_mae_by_seed[seed] for seed in seeds
    )
    day_nonworse = sum(delta <= 0.0 for delta in day_mae_delta.values())
    required_day_nonworse = math.ceil(0.75 * len(day_mae_delta))
    gates = {
        "mean_test_mae_improves_at_least_2_percent": relative_improvement >= 0.02,
        "at_least_two_of_three_seed_wins": seed_wins >= 2,
        "at_least_three_quarters_test_days_nonworse": (
            day_nonworse >= required_day_nonworse
        ),
        "secondary_task_signal_improves": (
            closing_edge_delta_change > 0.0
            or maneuver_representation_delta_change >= 0.02
        ),
    }
    passed = sum(gates.values())
    return {
        "decision": "PROMOTE_B" if passed == len(gates) else "KEEP_A",
        "passed_gates": passed,
        "required_gates": len(gates),
        "gates": gates,
        "mean_test_mae_relative_improvement": relative_improvement,
        "seed_wins": seed_wins,
        "day_nonworse": day_nonworse,
        "required_day_nonworse": required_day_nonworse,
        "closing_edge_delta_change": closing_edge_delta_change,
        "maneuver_representation_delta_change": (
            maneuver_representation_delta_change
        ),
    }
