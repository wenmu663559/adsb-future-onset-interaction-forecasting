import numpy as np
import pytest

from airspace_complexity.r03_interaction_regime import (
    apply_regime_rule,
    build_candidate_rules,
    select_regime_rule,
)


def test_candidate_thresholds_are_derived_from_training_rows_only():
    train = {
        "current_aircraft_count": np.asarray([3, 4, 5, 6]),
        "closing_pair_count_10nm": np.asarray([0, 1, 4, 8]),
        "projected_separation_task_count_120s": np.asarray([0, 2, 3, 7]),
    }

    rules = {row["rule_id"]: row for row in build_candidate_rules(train)}

    assert rules["traffic_q75"]["conditions"][0]["threshold"] == 6.0
    assert rules["closing_q75"]["conditions"][0]["threshold"] == 8.0
    assert rules["projected_q75"]["conditions"][0]["threshold"] == 7.0
    assert rules["balanced_interaction"]["conditions"][0]["threshold"] == 5.0


def test_combined_rule_uses_and_then_or_logic():
    indicators = {
        "current_aircraft_count": np.asarray([4, 5, 5, 6]),
        "closing_pair_count_10nm": np.asarray([5, 1, 4, 1]),
        "projected_separation_task_count_120s": np.asarray([1, 3, 1, 3]),
    }
    rule = {
        "logic": "all_groups",
        "conditions": [
            {"indicator": "current_aircraft_count", "threshold": 5.0, "group": 0},
            {"indicator": "closing_pair_count_10nm", "threshold": 4.0, "group": 1},
            {"indicator": "projected_separation_task_count_120s", "threshold": 3.0, "group": 1},
        ],
    }

    assert apply_regime_rule(indicators, rule).tolist() == [False, True, True, True]


def test_selection_uses_equal_day_mae_and_rejects_low_coverage_rules():
    indicators = {
        "current_aircraft_count": np.asarray([5, 5, 3, 3, 5, 5, 3, 3]),
        "closing_pair_count_10nm": np.asarray([4, 4, 1, 1, 1, 1, 1, 1]),
        "projected_separation_task_count_120s": np.zeros(8),
    }
    days = np.asarray(["d1"] * 4 + ["d2"] * 4)
    y = np.zeros(8)
    ridge = np.ones(8) * np.log1p(2.0)
    a_predictions = {
        17: np.log1p(np.asarray([1, 1, 4, 4, 1, 1, 4, 4], dtype=float))
    }
    rules = [
        {
            "rule_id": "traffic",
            "logic": "all_groups",
            "conditions": [{"indicator": "current_aircraft_count", "threshold": 5.0, "group": 0}],
        },
        {
            "rule_id": "too_sparse",
            "logic": "all_groups",
            "conditions": [{"indicator": "closing_pair_count_10nm", "threshold": 4.0, "group": 0}],
        },
    ]

    result = select_regime_rule(
        rules=rules,
        validation_indicators=indicators,
        y_true=y,
        days=days,
        a_predictions=a_predictions,
        ridge_prediction=ridge,
        min_scenes_per_day=2,
        min_overall_coverage=0.1,
        max_overall_coverage=0.75,
    )

    assert result["selected_rule_id"] == "traffic"
    assert result["candidate_ledger"]["traffic"]["equal_day_mean_count_mae_delta"] == pytest.approx(-1.0)
    assert result["candidate_ledger"]["too_sparse"]["eligible"] is False
