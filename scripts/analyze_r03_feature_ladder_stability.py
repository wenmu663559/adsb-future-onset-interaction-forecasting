from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_day_comparison(
    baseline_days: dict[str, object], candidate_days: dict[str, object],
    *, family: str, model: str, metric: str,
) -> dict[str, object]:
    common = sorted(set(baseline_days) & set(candidate_days))
    differences = []
    for day in common:
        baseline = float(baseline_days[day][family][model][metric])
        candidate = float(candidate_days[day][family][model][metric])
        differences.append(candidate - baseline)
    return {
        "day_count": len(common),
        "candidate_wins": sum(value < 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "candidate_minus_A_mean": float(np.mean(differences)),
        "candidate_minus_A_median": float(np.median(differences)),
        "differences_by_day": dict(zip(common, differences, strict=True)),
    }


def run(input_path: Path, output_path: Path) -> dict[str, object]:
    result = json.loads(Path(input_path).read_text(encoding="utf-8"))
    selected_count_model = {
        "30": "ridge_log1p",
        "120": "plain_poisson",
        "300": "plain_poisson",
    }
    comparisons: dict[str, object] = {}
    b_pass_horizons = []
    for horizon, model in selected_count_model.items():
        ladders = result["horizons"][horizon]["ladders"]
        horizon_result: dict[str, object] = {}
        for candidate in ("B", "C"):
            split_result: dict[str, object] = {}
            for split in ("validation", "test"):
                baseline = ladders["A"][split]
                contender = ladders[candidate][split]
                count = _paired_day_comparison(
                    baseline["days"], contender["days"],
                    family="count_models", model=model, metric="count_mae",
                )
                probability = _paired_day_comparison(
                    baseline["days"], contender["days"],
                    family="probability_models", model="logistic_ridge", metric="brier",
                )
                split_result[split] = {
                    "count": count,
                    "probability": probability,
                    "equal_day_count_mae_A": baseline["equal_day_count_mae"][model],
                    "equal_day_count_mae_candidate": contender["equal_day_count_mae"][model],
                    "equal_day_brier_A": baseline["equal_day_brier"]["logistic_ridge"],
                    "equal_day_brier_candidate": contender["equal_day_brier"]["logistic_ridge"],
                }
            horizon_result[candidate] = split_result
        b_validation = horizon_result["B"]["validation"]
        if (
            b_validation["equal_day_count_mae_candidate"]
            < b_validation["equal_day_count_mae_A"]
            and b_validation["equal_day_brier_candidate"]
            < b_validation["equal_day_brier_A"]
        ):
            b_pass_horizons.append(int(horizon))
        comparisons[horizon] = {
            "selected_count_model": model,
            "candidates": horizon_result,
        }

    proceed = len(b_pass_horizons) >= 2
    decision = (
        "PROCEED_TO_ONE_SHOT_CONFIRMATION_WITH_PAIRWISE_B"
        if proceed else "STOP_REPRESENTATION_EXPANSION_RETAIN_A"
    )
    analysis: dict[str, object] = {
        "schema_version": "r03_feature_ladder_stability_decision_v1",
        "decision": decision,
        "input_sha256": _sha256(Path(input_path)),
        "decision_basis": "validation equal-day count MAE and logistic Brier; test dates are retrospective stability evidence only",
        "B_pass_horizons": sorted(b_pass_horizons),
        "selected_feature_ladder": "B" if proceed else "A",
        "selected_count_models_by_horizon": selected_count_model,
        "selected_probability_model": "logistic_ridge",
        "representation_expansion": "STOP_AFTER_B",
        "reason_for_not_selecting_C": "CPA additions are modest and not directionally stable at 300 seconds; B is the parsimonious globally shared representation.",
        "comparisons": comparisons,
        "confirmation_constraints": {
            "allow_retuning": False,
            "allow_new_model_family": False,
            "allow_new_feature_selection": False,
            "allow_additional_KBTP_batch": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen A/B/C stability decision."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.input, args.output)
    print(json.dumps({
        "decision": result["decision"],
        "B_pass_horizons": result["B_pass_horizons"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
