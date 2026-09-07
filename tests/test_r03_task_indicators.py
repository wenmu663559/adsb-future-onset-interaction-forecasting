import json
import math
from pathlib import Path
import subprocess
import sys

from airspace_complexity.r03_task_indicators import (
    extract_pairwise_cpa_features,
    extract_task_indicators,
    summarize_indicator_relationships,
)


def _report(aircraft_id, timestamp_us, lon_deg, *, heading_deg=90.0, altitude_m=500.0):
    return {
        "aircraft_id": aircraft_id,
        "timestamp_us": timestamp_us,
        "lat_deg": 0.0,
        "lon_deg": lon_deg,
        "altitude_m": altitude_m,
        "speed_mps": 50.0,
        "heading_rad": math.radians(heading_deg),
    }


def test_extract_task_indicators_counts_screening_and_flow_proxies():
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [
                _report("inbound", 40_000_000, 0.020),
                _report("outbound", 40_000_000, 0.010),
                _report("steady", 40_000_000, 0.030),
            ],
            [
                _report("inbound", cutoff, 0.010),
                _report("outbound", cutoff, 0.020),
                _report("steady", cutoff, 0.031),
            ],
        ],
    }

    result = extract_task_indicators(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)

    assert result["current_aircraft_count"] == 3
    assert result["screening_pair_count"] == 3
    assert result["inbound_count_proxy"] == 1
    assert result["outbound_count_proxy"] == 1
    assert result["mixed_inbound_outbound_pairs"] == 1


def test_extract_task_indicators_detects_closing_and_projected_separation_pair():
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [
                _report("left", 70_000_000, -0.020, heading_deg=90.0),
                _report("right", 70_000_000, 0.020, heading_deg=270.0),
            ],
            [
                _report("left", cutoff, -0.010, heading_deg=90.0),
                _report("right", cutoff, 0.010, heading_deg=270.0),
            ],
        ],
    }

    result = extract_task_indicators(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)

    assert result["closing_pair_count_10nm"] == 1
    assert result["projected_separation_task_count_120s"] == 1


def test_extract_task_indicators_does_not_count_diverging_pair_as_projected_task():
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [
                _report("left", 70_000_000, -0.040, heading_deg=270.0),
                _report("right", 70_000_000, 0.040, heading_deg=90.0),
            ],
            [
                _report("left", cutoff, -0.050, heading_deg=270.0),
                _report("right", cutoff, 0.050, heading_deg=90.0),
            ],
        ],
    }

    result = extract_task_indicators(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)

    assert result["closing_pair_count_10nm"] == 0
    assert result["projected_separation_task_count_120s"] == 0


def test_pairwise_cpa_feature_ladder_is_horizon_specific_and_target_free():
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [
                _report("left", 70_000_000, -0.020),
                _report("right", 70_000_000, 0.020),
            ],
            [
                _report("left", cutoff, -0.010),
                _report("right", cutoff, 0.010),
            ],
        ],
        "targets": {"120": {"future_onset_proximity_event_count": 999}},
    }

    first = extract_pairwise_cpa_features(
        scene, airport_lat_deg=0.0, airport_lon_deg=0.0,
        horizons_seconds=(30, 120, 300),
    )
    scene["targets"]["120"]["future_onset_proximity_event_count"] = 0
    second = extract_pairwise_cpa_features(
        scene, airport_lat_deg=0.0, airport_lon_deg=0.0,
        horizons_seconds=(30, 120, 300),
    )

    assert first == second
    assert first["observed_pair_count"] == 1
    assert first["current_active_pair_count_5nm_1000ft"] == 1
    assert first["closing_pair_count_10nm"] == 1
    assert first["cpa_pair_count_30s"] == 1
    assert first["cpa_pair_count_120s"] == 1
    assert first["cpa_pair_count_300s"] == 1
    assert first["minimum_projected_horizontal_cpa_m_120s"] >= 0.0


def test_extract_task_indicators_counts_observed_maneuver_without_future_data():
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [_report("turning", 40_000_000, 0.020, heading_deg=0.0)],
            [_report("turning", cutoff, 0.020, heading_deg=25.0)],
        ],
    }

    result = extract_task_indicators(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)

    assert result["maneuvering_aircraft_count"] == 1
    assert "targets" not in result


def test_summarize_indicator_relationships_uses_day_and_count_strata():
    records = [
        {"day": "2026-01-01", "target": 0.0, "current_aircraft_count": 2, "metric": 0.0},
        {"day": "2026-01-01", "target": 1.0, "current_aircraft_count": 2, "metric": 1.0},
        {"day": "2026-01-02", "target": 0.0, "current_aircraft_count": 2, "metric": 0.0},
        {"day": "2026-01-02", "target": 2.0, "current_aircraft_count": 2, "metric": 2.0},
    ]

    result = summarize_indicator_relationships(records, metric_names=["metric"])

    assert result["metric"] == {
        "nonzero_scenes": 2,
        "nonzero_rate": 0.5,
        "scene_spearman_log_target": 1.0,
        "daily_spearman_median": 1.0,
        "daily_spearman_iqr": [1.0, 1.0],
        "positive_days": 2,
        "evaluable_days": 2,
        "day_and_count_stratified_residual_correlation": 1.0,
    }


def test_summarize_indicator_relationships_marks_constant_metric_not_assessable():
    records = [
        {"day": "2026-01-01", "target": 0.0, "current_aircraft_count": 2, "metric": 1.0},
        {"day": "2026-01-01", "target": 1.0, "current_aircraft_count": 2, "metric": 1.0},
    ]

    result = summarize_indicator_relationships(records, metric_names=["metric"])

    assert result["metric"]["scene_spearman_log_target"] is None
    assert result["metric"]["daily_spearman_median"] is None
    assert result["metric"]["day_and_count_stratified_residual_correlation"] is None


def test_task_indicator_cli_writes_traceable_machine_result(tmp_path):
    from scripts.analyze_r03_task_indicators import main

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    scenes = []
    for index, target in enumerate((0, 1)):
        cutoff = 100_000_000 + index * 10_000_000
        scenes.append({
            "airport_id": "KBTP",
            "source_day": "2026-01-01",
            "split": "test",
            "cutoff_us": cutoff,
            "history": [[{
                **_report("aircraft", cutoff, -79.9510833333),
                "lat_deg": 40.7765833333,
            }]],
            "targets": {"120": {"proximity_event_count": target}},
        })
    scene_path = dataset_root / "pilot_scenes.jsonl"
    scene_path.write_text(
        "".join(json.dumps(scene) + "\n" for scene in scenes), encoding="utf-8"
    )
    output = tmp_path / "result.json"

    assert main(["--dataset-root", str(dataset_root), "--output", str(output)]) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["scene_count"] == 2
    assert result["independent_days"] == 1
    assert result["dataset_sha256"]
    assert "screening_pair_count" in result["metrics"]


def test_task_indicator_cli_bootstraps_src_when_run_as_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_r03_task_indicators.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset-root" in completed.stdout
