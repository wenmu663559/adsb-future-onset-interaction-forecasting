from __future__ import annotations

from scripts.analyze_r03_cpa_coverage import summarize_scene_coverage


def _report(aircraft_id: str, timestamp_us: int, x: float) -> dict[str, object]:
    return {
        "aircraft_id": aircraft_id,
        "timestamp_us": timestamp_us,
        "lat_deg": 0.0,
        "lon_deg": x / 111_320.0,
        "altitude_m": 1000.0,
    }


def test_coverage_separates_missing_prior_and_degenerate_velocity() -> None:
    cutoff = 100_000_000
    scene = {
        "cutoff_us": cutoff,
        "history": [
            [_report("A", 70_000_000, 0.0), _report("B", 70_000_000, 1000.0)],
            [_report("A", cutoff, 100.0), _report("B", cutoff, 1100.0), _report("C", cutoff, 2000.0)],
        ],
    }
    result = summarize_scene_coverage(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)
    assert result["current_aircraft"] == 3
    assert result["current_pairs"] == 3
    assert result["cpa_eligible_pairs"] == 1
    assert result["missing_prior_pairs"] == 2
    assert result["degenerate_relative_horizontal_velocity_pairs"] == 1
    assert result["same_report_used_as_current_and_prior_aircraft"] == 0


def test_coverage_detects_same_stale_report_used_twice() -> None:
    scene = {
        "cutoff_us": 100_000_000,
        "history": [[_report("A", 70_000_000, 0.0), _report("B", 70_000_000, 1000.0)]],
    }
    result = summarize_scene_coverage(scene, airport_lat_deg=0.0, airport_lon_deg=0.0)
    assert result["cpa_eligible_pairs"] == 1
    assert result["same_report_used_as_current_and_prior_aircraft"] == 2
    assert result["degenerate_relative_horizontal_velocity_pairs"] == 1
