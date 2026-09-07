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
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airspace_complexity.r03_future_onset import (
    build_feature_ladder,
    exposure_rate_predictions,
)
from airspace_complexity.r03_probabilistic_multihorizon import (
    binary_log_loss,
    count_metrics,
    expected_calibration_error,
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


def _read_rows(root: Path):
    return [
        json.loads(line)
        for line in (Path(root) / "pilot_scenes.jsonl").open(encoding="utf-8")
    ]


def _matrix(rows, ladder: str):
    features = []
    names = None
    for row in rows:
        center = AIRPORT_CENTERS[str(row["airport_id"])]
        values = build_feature_ladder(
            row, airport_lat_deg=center[0], airport_lon_deg=center[1]
        )[ladder]
        if names is None:
            names = tuple(values)
        if tuple(values) != names:
            raise ValueError("feature names changed across scenes")
        features.append([values[name] for name in names])
    matrix = np.asarray(features, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains nonfinite values")
    return matrix, names


def _fit_count(model_name: str, x, y):
    if model_name == "ridge_log1p":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, np.log1p(y))
        return model, lambda values: np.maximum(np.expm1(model.predict(values)), 0.0)
    if model_name == "plain_poisson":
        model = make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=0.0, max_iter=1000)
        )
        model.fit(x, y)
        return model, lambda values: np.maximum(model.predict(values), 0.0)
    raise ValueError(f"unsupported frozen count model: {model_name}")


def _day_differences(days, truth, candidate, baseline, *, probability=False):
    differences = {}
    candidate_metrics = {}
    baseline_metrics = {}
    for day in sorted(set(days)):
        selected = days == day
        if probability:
            candidate_value = float(np.mean((candidate[selected] - truth[selected]) ** 2))
            baseline_value = float(np.mean((baseline[selected] - truth[selected]) ** 2))
        else:
            candidate_value = float(np.mean(np.abs(candidate[selected] - truth[selected])))
            baseline_value = float(np.mean(np.abs(baseline[selected] - truth[selected])))
        candidate_metrics[day] = candidate_value
        baseline_metrics[day] = baseline_value
        differences[day] = candidate_value - baseline_value
    return {
        "candidate_equal_day_mean": float(np.mean(list(candidate_metrics.values()))),
        "baseline_equal_day_mean": float(np.mean(list(baseline_metrics.values()))),
        "candidate_wins": sum(value < 0.0 for value in differences.values()),
        "ties": sum(value == 0.0 for value in differences.values()),
        "candidate_minus_baseline_by_day": differences,
        "candidate_metric_by_day": candidate_metrics,
        "baseline_metric_by_day": baseline_metrics,
    }


def run(
    development_root: Path, confirmation_root: Path,
    protocol_path: Path, confirmation_protocol_path: Path, output: Path,
) -> dict[str, object]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    frozen = json.loads(Path(confirmation_protocol_path).read_text(encoding="utf-8"))
    expected_hashes = {
        CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json": frozen["development_protocol_sha256"],
        CHECKOUT_ROOT / "configs" / "r03_future_onset_confirmation_dates.json": frozen["confirmation_dates_sha256"],
        CHECKOUT_ROOT / "reports" / "references" / "r03_feature_ladder_stability_decision_2026-09-03.json": frozen["stability_decision_sha256"],
        CHECKOUT_ROOT / "reports" / "references" / "r03_future_onset_feature_ladder_2026-09-03.json": frozen["development_feature_ladder_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency hash mismatch: {path}")
    development_rows = _read_rows(development_root)
    confirmation_rows = _read_rows(confirmation_root)
    train_rows = [row for row in development_rows if row["split"] == "train"]
    date_config = json.loads(
        (CHECKOUT_ROOT / "configs" / "r03_future_onset_confirmation_dates.json")
        .read_text(encoding="utf-8")
    )
    source_to_group = {
        str(item["source_file_id"]): str(item["date"])
        for item in date_config["sources"]
    }
    confirmation_days = []
    for row in confirmation_rows:
        groups = {
            source_to_group[source_id] for source_id in row["source_file_ids"]
            if source_id in source_to_group
        }
        if len(groups) != 1:
            raise ValueError("confirmation scene is not bound to one frozen source group")
        confirmation_days.append(next(iter(groups)))
    confirmation_days = np.asarray(confirmation_days)
    if len(set(confirmation_days)) != 8:
        raise ValueError("confirmation must contain exactly eight independent source groups")

    matrices = {}
    feature_names = {}
    for ladder in ("A", "B"):
        matrices[("train", ladder)], names = _matrix(train_rows, ladder)
        matrices[("confirmation", ladder)], confirmation_names = _matrix(
            confirmation_rows, ladder
        )
        if names != confirmation_names:
            raise ValueError("development and confirmation feature schemas differ")
        feature_names[ladder] = list(names)

    horizon_results: dict[str, object] = {}
    passed_horizons = []
    for horizon in protocol["horizons_seconds"]:
        key = str(horizon)
        train_targets = np.asarray([
            row["targets"][key]["future_onset_proximity_event_count"]
            for row in train_rows
        ], dtype=np.float64)
        confirmation_targets = np.asarray([
            row["targets"][key]["future_onset_proximity_event_count"]
            for row in confirmation_rows
        ], dtype=np.float64)
        selected_model = frozen["selected_count_models_by_horizon"][key]
        count_predictions = {}
        for ladder in ("A", "B"):
            _, predict = _fit_count(
                selected_model, matrices[("train", ladder)], train_targets
            )
            count_predictions[ladder] = predict(
                matrices[("confirmation", ladder)]
            )
        pair_train = matrices[("train", "B")][:, feature_names["B"].index("observed_pair_count")]
        pair_confirmation = matrices[("confirmation", "B")][
            :, feature_names["B"].index("observed_pair_count")
        ]
        pair_baseline = np.asarray(exposure_rate_predictions(
            train_targets, pair_train, pair_confirmation
        ))

        threshold = max(
            1, int(np.quantile(train_targets, 0.75, method="higher"))
        )
        train_risk = (train_targets >= threshold).astype(int)
        confirmation_risk = (confirmation_targets >= threshold).astype(int)
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=1701,
            ),
        )
        logistic.fit(matrices[("train", "B")], train_risk)
        logistic_probability = logistic.predict_proba(
            matrices[("confirmation", "B")]
        )[:, 1]
        prevalence_probability = np.full(
            len(confirmation_rows), float(train_risk.mean())
        )
        versus_exposure = _day_differences(
            confirmation_days, confirmation_targets,
            count_predictions["B"], pair_baseline,
        )
        versus_A = _day_differences(
            confirmation_days, confirmation_targets,
            count_predictions["B"], count_predictions["A"],
        )
        versus_prevalence = _day_differences(
            confirmation_days, confirmation_risk,
            logistic_probability, prevalence_probability, probability=True,
        )
        passed = all((
            versus_exposure["candidate_equal_day_mean"]
            < versus_exposure["baseline_equal_day_mean"],
            versus_exposure["candidate_wins"] >= 5,
            versus_A["candidate_equal_day_mean"]
            < versus_A["baseline_equal_day_mean"],
            versus_A["candidate_wins"] >= 5,
            versus_prevalence["candidate_equal_day_mean"]
            < versus_prevalence["baseline_equal_day_mean"],
            versus_prevalence["candidate_wins"] >= 5,
        ))
        if passed:
            passed_horizons.append(int(horizon))
        horizon_results[key] = {
            "selected_count_model": selected_model,
            "risk_threshold": threshold,
            "passed_all_frozen_gates": passed,
            "B_vs_pair_exposure_count_MAE": versus_exposure,
            "B_vs_A_count_MAE": versus_A,
            "B_logistic_vs_prevalence_Brier": versus_prevalence,
            "B_pooled_count_metrics": count_metrics(
                confirmation_targets, count_predictions["B"]
            ),
            "B_pooled_probability_metrics": {
                "brier": float(np.mean((logistic_probability - confirmation_risk) ** 2)),
                "log_loss": binary_log_loss(confirmation_risk, logistic_probability),
                "ece_10_bins": expected_calibration_error(
                    logistic_probability, confirmation_risk, bins=10
                ),
            },
        }
    route_passed = len(passed_horizons) >= 2
    result: dict[str, object] = {
        "schema_version": "r03_future_onset_confirmation_v1",
        "decision": (
            "CONFIRM_PAIRWISE_B_AND_PROCEED_TO_KAGC"
            if route_passed else "KBTP_CONFIRMATION_GATE_NOT_MET"
        ),
        "route_passed": route_passed,
        "passed_horizons": passed_horizons,
        "development_scene_sha256": _sha256(Path(development_root) / "pilot_scenes.jsonl"),
        "confirmation_scene_sha256": _sha256(Path(confirmation_root) / "pilot_scenes.jsonl"),
        "confirmation_protocol_sha256": _sha256(Path(confirmation_protocol_path)),
        "confirmation_days": sorted(set(confirmation_days)),
        "confirmation_scene_count": len(confirmation_rows),
        "horizons": horizon_results,
        "retuning_performed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one-shot frozen KBTP confirmation.")
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json",
    )
    parser.add_argument(
        "--confirmation-protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_confirmation_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.development_root, args.confirmation_root,
        args.protocol, args.confirmation_protocol, args.output,
    )
    print(json.dumps({
        "decision": result["decision"],
        "passed_horizons": result["passed_horizons"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
