"""Confirm a frozen high-interaction rule on untouched external dates."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
for path in (CHECKOUT_ROOT / "src", SCRIPT_PATH.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from airspace_complexity.r03_date_blocked_comparison import paired_day_summary
from airspace_complexity.r03_frozen_probe import load_invariant_relation_checkpoint
from airspace_complexity.r03_interaction_regime import apply_regime_rule
from probe_r03_frozen_task_representation import _frozen_outputs
from run_r03_external_date_validation import _sha256
from run_r03_online_complexity_pilot import _load_experiment_arrays, _metrics
from select_r03_interaction_regime import _load_raw_indicators


def _git_version():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _regime_metrics(y, ridge_prediction, a_predictions, days, selected):
    a_daily = {}
    ridge_daily = {}
    daily = {}
    for day in sorted(set(days.tolist())):
        mask = selected & (days == day)
        count = int(mask.sum())
        if count < 20:
            daily[day] = {"scene_count": count, "evaluable": False}
            continue
        seed_metrics = {
            str(seed): _metrics(y[mask], prediction[mask])
            for seed, prediction in sorted(a_predictions.items())
        }
        ridge_metrics = _metrics(y[mask], ridge_prediction[mask])
        a_mae = float(np.mean([row["count_mae"] for row in seed_metrics.values()]))
        a_daily[day] = a_mae
        ridge_daily[day] = float(ridge_metrics["count_mae"])
        daily[day] = {
            "scene_count": count,
            "coverage": float(mask.sum() / (days == day).sum()),
            "evaluable": True,
            "A_by_seed": seed_metrics,
            "A_seed_mean_count_mae": a_mae,
            "ridge_controls": ridge_metrics,
        }
    if len(a_daily) != len(set(days.tolist())):
        paired = None
    else:
        paired = paired_day_summary(candidate_mae=a_daily, comparator_mae=ridge_daily)
    return daily, paired


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset-root", type=Path, required=True)
    parser.add_argument("--external-dataset-root", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--a-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    freeze = json.loads(args.frozen_rule.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "r03_interaction_regime_freeze_v1":
        raise ValueError("unsupported frozen-rule schema")
    rule = freeze.get("selected_rule")
    if not rule:
        raise ValueError("frozen rule has no selected regime")
    training_scene_path = args.training_dataset_root / "pilot_scenes.jsonl"
    if _sha256(training_scene_path) != freeze["training_scene_sha256"]:
        raise ValueError("training dataset does not match frozen rule")

    train_x, train_targets, train_controls, train_split, _, mean, scale = (
        _load_experiment_arrays(args.training_dataset_root)
    )
    external_x, external_targets, external_controls, external_split, _, _, _ = (
        _load_experiment_arrays(
            args.external_dataset_root, frozen_normalization=(mean, scale)
        )
    )
    if not np.all(external_split == "external_test"):
        raise ValueError("external dataset contains a non-external split")

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = train_split == "train"
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        train_controls[train, :8], train_targets[train, 1]
    )
    ridge_prediction = ridge.predict(external_controls[:, :8])
    expected_hashes = {int(row["seed"]): row["sha256"] for row in freeze["checkpoints"]}
    a_predictions = {}
    for checkpoint_path in args.a_checkpoint:
        model, metadata = load_invariant_relation_checkpoint(checkpoint_path)
        seed = int(metadata["seed"])
        if _sha256(checkpoint_path) != expected_hashes.get(seed):
            raise ValueError(f"checkpoint does not match frozen rule: seed {seed}")
        prediction, _ = _frozen_outputs(model, external_x)
        a_predictions[seed] = prediction
    if set(a_predictions) != {17, 29, 43}:
        raise ValueError("A checkpoints must contain frozen seeds 17, 29, and 43")

    scene_path = args.external_dataset_root / "pilot_scenes.jsonl"
    indicators, days = _load_raw_indicators(scene_path)
    high = apply_regime_rule(indicators, rule)
    y = external_targets[:, 1]
    high_daily, high_paired = _regime_metrics(
        y, ridge_prediction, a_predictions, days, high
    )
    low_daily, low_paired = _regime_metrics(
        y, ridge_prediction, a_predictions, days, ~high
    )
    high_success = bool(
        high_paired
        and high_paired["wins"] > high_paired["losses"]
        and high_paired["bootstrap_95_interval"][1] < 0.0
    )
    result = {
        "schema_version": "r03_interaction_regime_confirmation_v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "one_frozen_rule_confirmed_once_on_round2_eight_untouched_dates",
        "frozen_rule_path": str(args.frozen_rule.resolve()),
        "frozen_rule_sha256": _sha256(args.frozen_rule),
        "selected_rule": rule,
        "external_dataset_root": str(args.external_dataset_root.resolve()),
        "external_scene_sha256": _sha256(scene_path),
        "external_scene_count": int(len(high)),
        "external_day_count": int(len(set(days.tolist()))),
        "high_interaction": {
            "scene_count": int(high.sum()),
            "coverage": float(high.mean()),
            "daily": high_daily,
            "paired_day_comparison": high_paired,
        },
        "low_interaction_secondary": {
            "scene_count": int((~high).sum()),
            "coverage": float((~high).mean()),
            "daily": low_daily,
            "paired_day_comparison": low_paired,
        },
        "predeclared_success_gate": "high-state A wins more days than it loses and the day-bootstrap 95% interval for A-minus-Ridge count MAE is entirely below zero",
        "high_interaction_gate_passed": high_success,
        "route_decision": (
            "retain conditional A for high-interaction states and Ridge otherwise"
            if high_success
            else "do not deploy the conditional A claim; retain Ridge as the default primary model"
        ),
        "claim_boundary": "Same-airport temporal transport for a trajectory-derived proximity-event proxy; not controller workload or cross-airport generalization.",
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.platform(),
            "numpy": np.__version__,
            "code_version": _git_version(),
            "cuda": "not used",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "coverage": result["high_interaction"]["coverage"],
                "paired": high_paired,
                "gate_passed": high_success,
                "route_decision": result["route_decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
