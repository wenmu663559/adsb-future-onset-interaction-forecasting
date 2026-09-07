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
    causal_baseline_features,
    exposure_rate_predictions,
    training_prevalence,
)


AIRPORT_CENTERS = {
    "KAGC": (40.3544376, -79.9290467),
    "KBTP": (40.7765833333, -79.9510833333),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_metrics(targets, predictions) -> dict[str, float]:
    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.maximum(np.asarray(predictions, dtype=np.float64), 0.0)
    mae = float(np.mean(np.abs(estimate - truth)))
    mean_target = float(np.mean(truth))
    return {
        "count_mae": mae,
        "normalized_mae": mae / max(mean_target, 1e-12),
        "mean_target": mean_target,
    }


def run(dataset_root: Path, protocol_path: Path, output: Path) -> dict[str, object]:
    scene_path = Path(dataset_root) / "pilot_scenes.jsonl"
    rows = [json.loads(line) for line in scene_path.open(encoding="utf-8")]
    if not rows:
        raise ValueError("dataset contains no scenes")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    horizons = tuple(int(value) for value in protocol["horizons_seconds"])
    features = []
    for row in rows:
        airport = str(row["airport_id"])
        center = AIRPORT_CENTERS[airport]
        features.append(causal_baseline_features(
            row, airport_lat_deg=center[0], airport_lon_deg=center[1]
        ))

    splits = np.asarray([str(row["split"]) for row in rows])
    days = np.asarray([str(row["source_day"]) for row in rows])
    train = splits == "train"
    if not train.any():
        raise ValueError("training scenes are required")
    pair_exposure = np.asarray(
        [feature["current_pair_opportunity"] for feature in features],
        dtype=np.float64,
    )
    aircraft_count = np.asarray(
        [feature["current_aircraft_count"] for feature in features],
        dtype=np.float64,
    )
    persistence = np.asarray(
        [feature["cutoff_active_pair_count"] for feature in features],
        dtype=np.float64,
    )
    cpa_120 = np.asarray(
        [feature["constant_velocity_cpa_pair_count_120s"] for feature in features],
        dtype=np.float64,
    )

    horizon_results: dict[str, object] = {}
    for horizon in horizons:
        targets = np.asarray([
            int(row["targets"][str(horizon)]["future_onset_proximity_event_count"])
            for row in rows
        ], dtype=np.float64)
        training_mean = float(targets[train].mean())
        exposure_prediction = np.asarray(exposure_rate_predictions(
            targets[train], pair_exposure[train], pair_exposure,
        ))
        aircraft_count_prediction = np.asarray(exposure_rate_predictions(
            targets[train], aircraft_count[train], aircraft_count,
        ))
        count_predictions = {
            "training_mean": np.full(len(rows), training_mean),
            "cutoff_active_pair_persistence": persistence,
            "current_aircraft_count_rate": aircraft_count_prediction,
            "current_pair_opportunity_rate": exposure_prediction,
        }
        if horizon == 120:
            count_predictions["constant_velocity_CPA_120s"] = cpa_120

        risk_threshold = int(np.quantile(targets[train], 0.75, method="higher"))
        risk_threshold = max(risk_threshold, 1)
        risk_labels = (targets >= risk_threshold).astype(int)
        prevalence = training_prevalence(risk_labels[train].tolist())
        split_results: dict[str, object] = {}
        for split_name in ("validation", "test"):
            selected = splits == split_name
            if not selected.any():
                continue
            per_day: dict[str, object] = {}
            for day in sorted(set(days[selected])):
                day_rows = selected & (days == day)
                per_day[day] = {
                    name: _count_metrics(targets[day_rows], prediction[day_rows])
                    for name, prediction in count_predictions.items()
                }
            split_risk = risk_labels[selected]
            split_results[split_name] = {
                "scene_count": int(selected.sum()),
                "days": per_day,
                "equal_scene_metrics": {
                    name: _count_metrics(targets[selected], prediction[selected])
                    for name, prediction in count_predictions.items()
                },
                "training_prevalence_risk_baseline": {
                    "threshold": risk_threshold,
                    "training_prevalence": prevalence,
                    "observed_prevalence": float(split_risk.mean()),
                    "brier": float(np.mean((prevalence - split_risk) ** 2)),
                },
            }
        horizon_results[str(horizon)] = {
            "training_scene_count": int(train.sum()),
            "training_mean_target": training_mean,
            "splits": split_results,
        }

    result: dict[str, object] = {
        "schema_version": "r03_future_onset_baselines_v1",
        "decision": "DESCRIPTIVE_BASELINE_EVIDENCE_NOT_CONFIRMATION",
        "dataset_scene_sha256": _sha256(scene_path),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "scene_count": len(rows),
        "independent_days": len(set(days)),
        "horizons": horizon_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen causal baselines for future-onset events."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.protocol, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "scene_count": result["scene_count"],
        "independent_days": result["independent_days"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
