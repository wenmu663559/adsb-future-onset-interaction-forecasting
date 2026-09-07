"""Freeze a high-interaction regime using original train/validation dates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
for path in (CHECKOUT_ROOT / "src", SCRIPT_PATH.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from airspace_complexity.r03_frozen_probe import load_invariant_relation_checkpoint
from airspace_complexity.r03_interaction_regime import (
    REQUIRED_INDICATORS,
    build_candidate_rules,
    select_regime_rule,
)
from airspace_complexity.r03_task_indicators import extract_task_indicators
from probe_r03_frozen_task_representation import AIRPORT_CENTERS, _frozen_outputs
from run_r03_online_complexity_pilot import _load_experiment_arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_indicators(scene_path: Path):
    values = {name: [] for name in REQUIRED_INDICATORS}
    days = []
    with scene_path.open(encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            airport = str(scene["airport_id"])
            latitude, longitude = AIRPORT_CENTERS[airport]
            row = extract_task_indicators(
                scene, airport_lat_deg=latitude, airport_lon_deg=longitude
            )
            days.append(str(scene["source_day"]))
            for name in values:
                values[name].append(float(row[name]))
    return {name: np.asarray(rows) for name, rows in values.items()}, np.asarray(days)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--a-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scene_path = args.dataset_root / "pilot_scenes.jsonl"
    x, targets, controls, split, _, _, _ = _load_experiment_arrays(args.dataset_root)
    indicators, days = _load_raw_indicators(scene_path)
    if not (len(x) == len(split) == len(days)):
        raise ValueError("scene, tensor, and split rows are not aligned")

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = split == "train"
    validation = split == "validation"
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        controls[train, :8], targets[train, 1]
    )
    ridge_prediction = ridge.predict(controls[validation, :8])
    a_predictions = {}
    checkpoint_records = []
    for checkpoint_path in args.a_checkpoint:
        model, metadata = load_invariant_relation_checkpoint(checkpoint_path)
        seed = int(metadata["seed"])
        prediction, _ = _frozen_outputs(model, x[validation])
        a_predictions[seed] = prediction
        checkpoint_records.append(
            {
                "seed": seed,
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            }
        )
    if set(a_predictions) != {17, 29, 43}:
        raise ValueError("A checkpoints must contain frozen seeds 17, 29, and 43")

    train_indicators = {name: rows[train] for name, rows in indicators.items()}
    validation_indicators = {
        name: rows[validation] for name, rows in indicators.items()
    }
    rules = build_candidate_rules(train_indicators)
    selection = select_regime_rule(
        rules=rules,
        validation_indicators=validation_indicators,
        y_true=targets[validation, 1],
        days=days[validation],
        a_predictions=a_predictions,
        ridge_prediction=ridge_prediction,
    )
    selected = next(
        (row for row in rules if row["rule_id"] == selection["selected_rule_id"]),
        None,
    )
    result = {
        "schema_version": "r03_interaction_regime_freeze_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "thresholds_from_original_train_rows_and_rule_choice_from_original_validation_days_only",
        "dataset_root": str(args.dataset_root.resolve()),
        "training_scene_sha256": _sha256(scene_path),
        "candidate_rules": rules,
        "selection": selection,
        "selected_rule": selected,
        "checkpoints": sorted(checkpoint_records, key=lambda row: row["seed"]),
        "ridge": "StandardScaler + Ridge(alpha=1.0), fit on original train rows only",
        "target": "log1p future proximity-event count at 120 seconds",
        "confirmation_boundary": "No original test rows or external-date rows were used to build thresholds or choose the rule.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "selected_rule": selected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
