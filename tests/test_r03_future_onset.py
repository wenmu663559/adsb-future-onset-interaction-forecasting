import math
from pathlib import Path
import subprocess
import sys

import pytest

from airspace_complexity.r03_future_onset import (
    A_FEATURE_NAMES,
    B_ADDITION_NAMES,
    C_ADDITION_NAMES,
    build_feature_ladder,
    causal_baseline_features,
    exposure_rate_predictions,
    ofat_threshold_settings,
    summarize_reconstruction,
    training_prevalence,
)


def _report(aircraft_id, timestamp_us, lon_deg, *, altitude_m=500.0):
    return {
        "aircraft_id": aircraft_id,
        "timestamp_us": timestamp_us,
        "lat_deg": 0.0,
        "lon_deg": lon_deg,
        "altitude_m": altitude_m,
        "speed_mps": 50.0,
        "heading_rad": math.pi / 2.0,
    }


def test_causal_baseline_features_capture_persistence_exposure_and_cpa() -> None:
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [
                _report("left", 70_000_000, -0.020),
                _report("right", 70_000_000, 0.020),
                _report("third", 70_000_000, 0.500),
            ],
            [
                _report("left", cutoff, -0.010),
                _report("right", cutoff, 0.010),
                _report("third", cutoff, 0.500),
            ],
        ],
        "targets": {"120": {"future_onset_proximity_event_count": 99}},
    }

    result = causal_baseline_features(
        scene, airport_lat_deg=0.0, airport_lon_deg=0.0,
    )

    assert result["current_aircraft_count"] == 3
    assert result["current_pair_opportunity"] == 3
    assert result["cutoff_active_pair_count"] == 1
    assert result["constant_velocity_cpa_pair_count_120s"] == 1
    assert 99 not in result.values()


def test_training_prevalence_uses_training_labels_only() -> None:
    assert training_prevalence([0, 1, 1, 0]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="binary"):
        training_prevalence([0, 2])


def test_exposure_rate_prediction_scales_by_pair_opportunity() -> None:
    predictions = exposure_rate_predictions(
        train_counts=[0, 2],
        train_pair_opportunities=[1, 3],
        evaluation_pair_opportunities=[0, 2, 4],
    )

    assert predictions == pytest.approx((0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="nonnegative"):
        exposure_rate_predictions([1], [-1], [1])


def test_reconstruction_summary_separates_persistence_from_future_onset() -> None:
    records = [
        {
            "scene_id": "a",
            "source_day": "2026-01-01",
            "targets": {
                "30": {
                    "proximity_event_count": 3,
                    "future_onset_proximity_event_count": 1,
                    "cutoff_active_pair_count": 2,
                    "min_horizontal_separation_m": 100.0,
                }
            },
        },
        {
            "scene_id": "b",
            "source_day": "2026-01-02",
            "targets": {
                "30": {
                    "proximity_event_count": 0,
                    "future_onset_proximity_event_count": 0,
                    "cutoff_active_pair_count": 0,
                    "min_horizontal_separation_m": None,
                }
            },
        },
    ]

    result = summarize_reconstruction(records)

    assert result["scene_count"] == 2
    assert result["independent_days"] == 2
    assert result["horizons"]["30"] == {
        "legacy_event_sum": 3,
        "future_onset_sum": 1,
        "legacy_minus_onset_sum": 2,
        "cutoff_active_pair_sum": 2,
        "missing_future_pair_scenes": 1,
    }


def test_reconstruction_audit_cli_bootstraps_when_run_as_script() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_r03_future_event_reconstruction.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset-root" in completed.stdout


def test_threshold_sensitivity_is_one_factor_at_a_time_not_cartesian() -> None:
    settings = ofat_threshold_settings(
        horizontal_nm=(3, 5, 7),
        vertical_ft=(500, 1000, 1500),
        primary_horizontal_nm=5,
        primary_vertical_ft=1000,
    )

    assert settings == (
        ("h3nm_v1000ft", 3.0, 1000.0),
        ("h5nm_v500ft", 5.0, 500.0),
        ("h5nm_v1000ft", 5.0, 1000.0),
        ("h5nm_v1500ft", 5.0, 1500.0),
        ("h7nm_v1000ft", 7.0, 1000.0),
    )


def test_feature_ladder_is_nested_and_uses_only_scene_history() -> None:
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [_report("left", 70_000_000, -0.020), _report("right", 70_000_000, 0.020)],
            [_report("left", cutoff, -0.010), _report("right", cutoff, 0.010)],
        ],
        "targets": {"120": {"future_onset_proximity_event_count": 999}},
    }

    first = build_feature_ladder(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)
    scene["targets"]["120"]["future_onset_proximity_event_count"] = 0
    second = build_feature_ladder(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)

    assert first == second
    assert tuple(first["A"]) == A_FEATURE_NAMES
    assert tuple(first["B"]) == A_FEATURE_NAMES + B_ADDITION_NAMES
    assert tuple(first["C"]) == A_FEATURE_NAMES + B_ADDITION_NAMES + C_ADDITION_NAMES
    assert all(math.isfinite(value) for ladder in first.values() for value in ladder.values())
