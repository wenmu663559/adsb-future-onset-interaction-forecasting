"""Freeze and apply an interpretable high-interaction operating regime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


REQUIRED_INDICATORS = (
    "current_aircraft_count",
    "closing_pair_count_10nm",
    "projected_separation_task_count_120s",
)


def _higher_quantile(values, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="higher"))


def build_candidate_rules(train_indicators: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    """Build the four predeclared candidates using training rows only."""

    missing = set(REQUIRED_INDICATORS).difference(train_indicators)
    if missing:
        raise ValueError(f"missing training indicators: {sorted(missing)}")
    thresholds = {
        "traffic_q75": _higher_quantile(train_indicators["current_aircraft_count"], 0.75),
        "closing_q75": _higher_quantile(train_indicators["closing_pair_count_10nm"], 0.75),
        "projected_q75": _higher_quantile(
            train_indicators["projected_separation_task_count_120s"], 0.75
        ),
        "traffic_q50": _higher_quantile(train_indicators["current_aircraft_count"], 0.50),
    }

    def condition(indicator: str, threshold: float, group: int = 0):
        return {"indicator": indicator, "operator": ">=", "threshold": threshold, "group": group}

    return [
        {
            "rule_id": "traffic_q75",
            "logic": "all_groups",
            "conditions": [condition("current_aircraft_count", thresholds["traffic_q75"])],
        },
        {
            "rule_id": "closing_q75",
            "logic": "all_groups",
            "conditions": [condition("closing_pair_count_10nm", thresholds["closing_q75"])],
        },
        {
            "rule_id": "projected_q75",
            "logic": "all_groups",
            "conditions": [
                condition(
                    "projected_separation_task_count_120s", thresholds["projected_q75"]
                )
            ],
        },
        {
            "rule_id": "balanced_interaction",
            "logic": "all_groups",
            "conditions": [
                condition("current_aircraft_count", thresholds["traffic_q50"], 0),
                condition("closing_pair_count_10nm", thresholds["closing_q75"], 1),
                condition(
                    "projected_separation_task_count_120s", thresholds["projected_q75"], 1
                ),
            ],
            "group_semantics": "AND across groups; OR within a group",
        },
    ]


def apply_regime_rule(indicators: Mapping[str, np.ndarray], rule: Mapping[str, object]):
    """Apply a rule. Conditions in the same group are OR; groups are AND."""

    conditions = list(rule["conditions"])
    if not conditions:
        raise ValueError("regime rule has no conditions")
    lengths = {len(np.asarray(indicators[str(row["indicator"])])) for row in conditions}
    if len(lengths) != 1:
        raise ValueError("indicator arrays are not aligned")
    groups: dict[int, np.ndarray] = {}
    for row in conditions:
        values = np.asarray(indicators[str(row["indicator"])], dtype=np.float64)
        selected = values >= float(row["threshold"])
        group = int(row.get("group", 0))
        groups[group] = selected if group not in groups else groups[group] | selected
    result = np.ones(lengths.pop(), dtype=bool)
    for selected in groups.values():
        result &= selected
    return result


def select_regime_rule(
    *,
    rules: Sequence[Mapping[str, object]],
    validation_indicators: Mapping[str, np.ndarray],
    y_true,
    days,
    a_predictions: Mapping[int, np.ndarray],
    ridge_prediction,
    min_scenes_per_day: int = 20,
    min_overall_coverage: float = 0.10,
    max_overall_coverage: float = 0.60,
) -> dict[str, object]:
    """Select on validation days by equal-day mean A-minus-Ridge count MAE."""

    y_true = np.asarray(y_true, dtype=np.float64)
    days = np.asarray(days)
    ridge_prediction = np.asarray(ridge_prediction, dtype=np.float64)
    unique_days = sorted(set(days.tolist()))
    ledger = {}
    for rule in rules:
        selected = apply_regime_rule(validation_indicators, rule)
        day_rows = {}
        deltas = []
        eligible = min_overall_coverage <= float(selected.mean()) <= max_overall_coverage
        for day in unique_days:
            mask = selected & (days == day)
            count = int(mask.sum())
            if count < min_scenes_per_day:
                eligible = False
                day_rows[day] = {"scene_count": count, "evaluable": False}
                continue
            ridge_mae = float(np.mean(np.abs(np.expm1(ridge_prediction[mask]) - np.expm1(y_true[mask]))))
            seed_mae = [
                float(np.mean(np.abs(np.expm1(np.asarray(pred)[mask]) - np.expm1(y_true[mask]))))
                for pred in a_predictions.values()
            ]
            a_mae = float(np.mean(seed_mae))
            delta = a_mae - ridge_mae
            deltas.append(delta)
            day_rows[day] = {
                "scene_count": count,
                "evaluable": True,
                "A_seed_mean_count_mae": a_mae,
                "ridge_count_mae": ridge_mae,
                "A_minus_ridge_count_mae": delta,
            }
        ledger[str(rule["rule_id"])] = {
            "eligible": bool(eligible),
            "overall_scene_count": int(selected.sum()),
            "overall_coverage": float(selected.mean()),
            "equal_day_mean_count_mae_delta": float(np.mean(deltas)) if len(deltas) == len(unique_days) else None,
            "daily": day_rows,
        }
    eligible_rows = [
        (float(row["equal_day_mean_count_mae_delta"]), rule_id)
        for rule_id, row in ledger.items()
        if row["eligible"] and row["equal_day_mean_count_mae_delta"] is not None
    ]
    selected_rule_id = min(eligible_rows)[1] if eligible_rows else None
    return {
        "selected_rule_id": selected_rule_id,
        "candidate_ledger": ledger,
        "selection_criterion": "minimum equal-day mean A-minus-Ridge count MAE on original validation days",
        "eligibility": {
            "minimum_scenes_per_validation_day": min_scenes_per_day,
            "minimum_overall_coverage": min_overall_coverage,
            "maximum_overall_coverage": max_overall_coverage,
        },
    }
