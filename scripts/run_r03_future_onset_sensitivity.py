from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

import numpy as np

from airspace_complexity.r03_future_onset import (
    exposure_rate_predictions,
    ofat_threshold_settings,
    summarize_reconstruction,
)
from airspace_complexity.r03_online_complexity import PilotReport, build_scene_examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_day_mae(targets, predictions, days, selected) -> float:
    values = []
    for day in sorted(set(days[selected])):
        day_rows = selected & (days == day)
        values.append(float(np.mean(np.abs(predictions[day_rows] - targets[day_rows]))))
    return float(np.mean(values))


def run(
    dataset_root: Path, protocol_path: Path, source_config_path: Path, output: Path,
) -> dict[str, object]:
    root = Path(dataset_root)
    report_path = root / "pilot_reports.jsonl"
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    source_config = json.loads(Path(source_config_path).read_text(encoding="utf-8"))
    frozen_splits = dict(source_config["source_day_splits"])
    reports = tuple(
        PilotReport(**json.loads(line)) for line in report_path.open(encoding="utf-8")
    )
    sensitivity = protocol["threshold_sensitivity"]
    primary = protocol["primary_target"]
    settings = ofat_threshold_settings(
        horizontal_nm=sensitivity["horizontal_nm"],
        vertical_ft=sensitivity["vertical_ft"],
        primary_horizontal_nm=primary["horizontal_nm"],
        primary_vertical_ft=primary["vertical_ft"],
    )
    setting_results: dict[str, object] = {}
    for setting_id, horizontal_nm, vertical_ft in settings:
        examples = build_scene_examples(
            reports,
            history_seconds=int(protocol["history_seconds"]),
            step_seconds=int(protocol["sampling_step_seconds"]),
            horizons_seconds=tuple(int(value) for value in protocol["horizons_seconds"]),
            stale_seconds=int(protocol["stale_seconds_primary"]),
            min_aircraft=3,
            max_aircraft=100,
            horizontal_proximity_m=horizontal_nm * 1852.0,
            vertical_proximity_m=vertical_ft * 0.3048,
        )
        split = np.asarray([
            frozen_splits[f"{scene.airport_id}\x00{scene.source_day}"]
            for scene in examples
        ])
        days = np.asarray([scene.source_day for scene in examples])
        pair_opportunity = np.asarray([
            len(scene.history[-1]) * (len(scene.history[-1]) - 1) / 2
            for scene in examples
        ], dtype=np.float64)
        minimal_records = [{
            "source_day": scene.source_day,
            "targets": {
                str(horizon): {
                    "proximity_event_count": target.proximity_event_count,
                    "future_onset_proximity_event_count": target.future_onset_proximity_event_count,
                    "cutoff_active_pair_count": target.cutoff_active_pair_count,
                    "min_horizontal_separation_m": target.min_horizontal_separation_m,
                }
                for horizon, target in scene.targets.items()
            },
        } for scene in examples]
        target_summary = summarize_reconstruction(minimal_records)
        horizon_results: dict[str, object] = {}
        for horizon in protocol["horizons_seconds"]:
            targets = np.asarray([
                scene.targets[int(horizon)].future_onset_proximity_event_count
                for scene in examples
            ], dtype=np.float64)
            train = split == "train"
            exposure_prediction = np.asarray(exposure_rate_predictions(
                targets[train], pair_opportunity[train], pair_opportunity,
            ))
            split_results: dict[str, object] = {}
            for split_name in ("train", "validation", "test"):
                selected = split == split_name
                per_day = {
                    day: {
                        "scene_count": int((selected & (days == day)).sum()),
                        "mean_onset_count": float(targets[selected & (days == day)].mean()),
                        "nonzero_rate": float((targets[selected & (days == day)] > 0).mean()),
                    }
                    for day in sorted(set(days[selected]))
                }
                split_results[split_name] = {
                    "scene_count": int(selected.sum()),
                    "mean_onset_count": float(targets[selected].mean()),
                    "nonzero_rate": float((targets[selected] > 0).mean()),
                    "pair_exposure_equal_day_mae": _equal_day_mae(
                        targets, exposure_prediction, days, selected
                    ),
                    "days": per_day,
                }
            risk_results: dict[str, object] = {}
            for quantile in sensitivity["risk_quantiles"]:
                threshold = max(
                    1, int(np.quantile(targets[train], quantile, method="higher"))
                )
                risk_results[f"q{int(round(100 * quantile))}"] = {
                    "threshold": threshold,
                    "train_prevalence": float((targets[train] >= threshold).mean()),
                    "validation_prevalence": float(
                        (targets[split == "validation"] >= threshold).mean()
                    ),
                    "test_prevalence": float((targets[split == "test"] >= threshold).mean()),
                }
            horizon_results[str(horizon)] = {
                "splits": split_results,
                "risk_quantiles": risk_results,
            }
        setting_results[setting_id] = {
            "horizontal_nm": horizontal_nm,
            "vertical_ft": vertical_ft,
            "target_summary": target_summary,
            "horizons": horizon_results,
        }
    result: dict[str, object] = {
        "schema_version": "r03_future_onset_threshold_sensitivity_v1",
        "decision": "ROBUSTNESS_ONLY_NO_THRESHOLD_SELECTION",
        "selection_from_outcomes": False,
        "design": "one_factor_at_a_time",
        "pilot_reports_sha256": _sha256(report_path),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "settings": setting_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen OFAT future-onset target sensitivity."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json",
    )
    parser.add_argument(
        "--source-config", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_development_sources.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.protocol, args.source_config, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "setting_count": len(result["settings"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
