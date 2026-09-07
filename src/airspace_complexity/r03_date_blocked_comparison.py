"""Paired uncertainty summaries with independent source days as the unit."""

from __future__ import annotations

from itertools import product
from math import comb
from collections.abc import Mapping

import numpy as np


def exhaustive_day_bootstrap_interval(
    day_deltas: np.ndarray, confidence: float = 0.95
) -> dict[str, float | int]:
    """Enumerate the ordinary bootstrap distribution over independent days."""

    values = np.asarray(day_deltas, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("day_deltas must be a nonempty one-dimensional array")
    if len(values) > 7:
        raise ValueError("exhaustive day bootstrap is limited to at most seven days")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    distribution = np.asarray(
        [values[list(indices)].mean() for indices in product(range(len(values)), repeat=len(values))]
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(distribution, alpha)),
        "upper": float(np.quantile(distribution, 1.0 - alpha)),
        "confidence": confidence,
        "resample_count": int(len(distribution)),
    }


def day_bootstrap_interval(
    day_deltas: np.ndarray,
    confidence: float = 0.95,
    *,
    resamples: int = 10_000,
    random_seed: int = 1701,
) -> dict[str, float | int | str]:
    """Use exact enumeration for small day sets and deterministic sampling otherwise."""

    values = np.asarray(day_deltas, dtype=float)
    if len(values) <= 7:
        result = exhaustive_day_bootstrap_interval(values, confidence)
        return {**result, "method": "exhaustive_day_bootstrap"}
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("day_deltas must be a nonempty one-dimensional array")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    distribution = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(distribution, alpha)),
        "upper": float(np.quantile(distribution, 1.0 - alpha)),
        "confidence": confidence,
        "resample_count": resamples,
        "method": "monte_carlo_day_bootstrap",
    }


def _exact_two_sided_sign_test(negative: int, positive: int) -> float:
    total = negative + positive
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(negative, positive) + 1)) / (2 ** total)
    return float(min(1.0, 2.0 * tail))


def paired_day_summary(
    *, candidate_mae: Mapping[str, float], comparator_mae: Mapping[str, float]
) -> dict[str, object]:
    """Compare paired daily MAEs; negative deltas favor the candidate."""

    if set(candidate_mae) != set(comparator_mae) or not candidate_mae:
        raise ValueError("candidate and comparator must contain the same nonempty day set")
    days = sorted(candidate_mae)
    delta_by_day = {
        day: float(candidate_mae[day] - comparator_mae[day]) for day in days
    }
    deltas = np.asarray([delta_by_day[day] for day in days])
    wins = int((deltas < 0.0).sum())
    ties = int((deltas == 0.0).sum())
    losses = int((deltas > 0.0).sum())
    interval = day_bootstrap_interval(deltas)
    return {
        "delta_definition": "candidate_mae_minus_comparator_mae",
        "day_count": len(days),
        "delta_by_day": delta_by_day,
        "mean_delta": float(deltas.mean()),
        "bootstrap_95_interval": [interval["lower"], interval["upper"]],
        "bootstrap_resample_count": interval["resample_count"],
        "bootstrap_method": interval["method"],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_non_tied_days": wins + losses,
        "exact_two_sided_sign_test_p": _exact_two_sided_sign_test(wins, losses),
    }
