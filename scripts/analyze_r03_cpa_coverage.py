"""Describe CPA input coverage on the frozen KBTP/KAGC evaluation scenes.

This is a post-review diagnostic.  It does not fit, select, or score a model.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from airspace_complexity.r03_task_indicators import _position_m, _scene_snapshots
from airspace_complexity.r03_future_onset import build_feature_ladder


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATHS = {
    "KBTP": ROOT / "outputs/experiments/r03_future_onset_confirmation/20260903T075352Z-online-complexity-pilot/pilot_scenes.jsonl",
    "KAGC": ROOT / "outputs/experiments/r03_kagc_external_validation/20260903T080631Z-online-complexity-pilot/pilot_scenes.jsonl",
}
GROUP_CONFIGS = {
    "KBTP": ROOT / "configs/r03_future_onset_confirmation_dates.json",
    "KAGC": ROOT / "configs/r03_kagc_external_validation.json",
}
AIRPORT_CENTERS = {"KBTP": (40.7769, -79.9497), "KAGC": (40.3544, -79.9302)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_scene_coverage(
    scene: Mapping[str, object], *, airport_lat_deg: float, airport_lon_deg: float
) -> dict[str, int]:
    current, prior_30, _ = _scene_snapshots(scene)
    aircraft_ids = sorted(current)
    pairs = list(combinations(aircraft_ids, 2))
    eligible = [(left, right) for left, right in pairs if left in prior_30 and right in prior_30]
    same_report = sum(
        int(current[aircraft_id]["timestamp_us"]) == int(prior_30[aircraft_id]["timestamp_us"])
        for aircraft_id in prior_30
    )
    degenerate = 0
    for left_id, right_id in eligible:
        velocities = []
        for aircraft_id in (left_id, right_id):
            now, earlier = current[aircraft_id], prior_30[aircraft_id]
            now_position = _position_m(now, airport_lat_deg, airport_lon_deg)
            prior_position = _position_m(earlier, airport_lat_deg, airport_lon_deg)
            elapsed = max(
                (int(now["timestamp_us"]) - int(earlier["timestamp_us"])) / 1_000_000.0,
                1.0,
            )
            velocities.append(tuple((now_position[i] - prior_position[i]) / elapsed for i in range(3)))
        relative_xy = (velocities[0][0] - velocities[1][0], velocities[0][1] - velocities[1][1])
        degenerate += relative_xy[0] ** 2 + relative_xy[1] ** 2 <= 1e-9
    return {
        "current_aircraft": len(aircraft_ids),
        "aircraft_with_prior_30": len(prior_30),
        "same_report_used_as_current_and_prior_aircraft": same_report,
        "current_pairs": len(pairs),
        "cpa_eligible_pairs": len(eligible),
        "missing_prior_pairs": len(pairs) - len(eligible),
        "degenerate_relative_horizontal_velocity_pairs": degenerate,
    }


def _group_map(path: Path) -> dict[str, str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["source_file_id"]): str(row["date"]) for row in config["sources"]}


def _summarize(rows: list[dict[str, int]]) -> dict[str, object]:
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    pair_total = totals["current_pairs"]
    eligible_total = totals["cpa_eligible_pairs"]
    scene_fractions = [row["cpa_eligible_pairs"] / row["current_pairs"] for row in rows if row["current_pairs"]]
    return {
        "n_scenes": len(rows),
        "totals": totals,
        "pair_weighted_eligible_fraction": eligible_total / pair_total,
        "scene_equal_eligible_fraction": statistics.fmean(scene_fractions),
        "scenes_with_any_missing_prior_pair": sum(row["missing_prior_pairs"] > 0 for row in rows),
        "scenes_with_any_degenerate_relative_velocity_pair": sum(
            row["degenerate_relative_horizontal_velocity_pairs"] > 0 for row in rows
        ),
        "degenerate_fraction_among_eligible_pairs": (
            totals["degenerate_relative_horizontal_velocity_pairs"] / eligible_total if eligible_total else None
        ),
    }


def run() -> dict[str, object]:
    output: dict[str, object] = {
        "schema_version": "r03_cpa_coverage_v1",
        "analysis_status": "Post-review descriptive diagnostic; frozen scenes; no model fitting, selection, or retuning.",
        "definitions": {
            "eligible_pair": "Both aircraft have a report 20–45 s before cutoff, nearest to 30 s.",
            "same_report": "The latest report is itself 20–45 s old and is therefore selected as both current and prior.",
            "degenerate_pair": "Eligible pair whose estimated relative horizontal speed squared is <=1e-9; CPA time is set to zero.",
        },
        "inputs": {},
        "by_airport": {},
    }
    for airport, path in SCENE_PATHS.items():
        mapping = _group_map(GROUP_CONFIGS[airport])
        center = AIRPORT_CENTERS[airport]
        grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
        all_rows = []
        feature_mismatches = 0
        checked_feature_values = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                scene = json.loads(line)
                groups = {mapping[source] for source in scene["source_file_ids"] if source in mapping}
                if len(groups) != 1:
                    raise ValueError(f"{airport}: scene must map to one source group")
                row = summarize_scene_coverage(scene, airport_lat_deg=center[0], airport_lon_deg=center[1])
                original = build_feature_ladder(scene, airport_lat_deg=center[0], airport_lon_deg=center[1])
                shortened = build_feature_ladder(
                    {**scene, "history": scene["history"][-6:]},
                    airport_lat_deg=center[0], airport_lon_deg=center[1],
                )
                for ladder in ("A", "B", "C"):
                    for name, value in original[ladder].items():
                        checked_feature_values += 1
                        feature_mismatches += value != shortened[ladder][name]
                grouped[next(iter(groups))].append(row)
                all_rows.append(row)
        output["inputs"][airport] = {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        output["by_airport"][airport] = {
            "overall": _summarize(all_rows),
            "by_group": {group: _summarize(rows) for group, rows in sorted(grouped.items())},
            "sixty_second_buffer_equivalence": {
                "comparison": "Original 12 snapshots (t-110..t) versus final 6 snapshots (t-50..t) on identical frozen scenes.",
                "checked_feature_values": checked_feature_values,
                "mismatches": feature_mismatches,
                "interpretation": "The implemented A/B/C ladder uses current reports and approximately 30-s prior reports; this is an input-equivalence check, not a model-performance ablation.",
            },
        }
    return output


def main() -> None:
    report = run()
    path = ROOT / "reports/references/r03_cpa_coverage_2026-09-05.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for airport, values in report["by_airport"].items():
        overall = values["overall"]
        print(airport, overall["n_scenes"], overall["pair_weighted_eligible_fraction"], overall["degenerate_fraction_among_eligible_pairs"])
    print(path)


if __name__ == "__main__":
    main()
