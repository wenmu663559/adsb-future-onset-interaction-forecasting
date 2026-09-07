"""Run the frozen post-review NB2 comparison on corrected future-onset targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression, PoissonRegressor


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
for path in (CHECKOUT_ROOT, CHECKOUT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from airspace_complexity.r03_probabilistic_multihorizon import (
    expected_calibration_error,
    fit_nb2_regression,
    nb2_log_loss,
    nb2_risk_probability,
    predict_nb2_mean,
)
from scripts.run_r03_future_onset_confirmation import _matrix, _read_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset_hashes(root: Path, records: dict[str, dict[str, object]]) -> None:
    for name, record in records.items():
        path = Path(root) / str(record["relative_path"])
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise ValueError(f"dataset hash mismatch: {name}: {path}")


def _frozen_thresholds(
    training_counts: np.ndarray, quantiles: list[float]
) -> dict[str, int]:
    counts = np.asarray(training_counts, dtype=np.float64)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts < 0.0):
        raise ValueError("training counts must be a nonempty nonnegative vector")
    return {
        f"{float(quantile):.2f}": max(
            1, int(np.quantile(counts, float(quantile), method="higher"))
        )
        for quantile in quantiles
    }


def _binary_row_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    label = np.asarray(labels, dtype=np.float64)
    return -(label * np.log(probability) + (1.0 - label) * np.log1p(-probability))


def _poisson_row_nll(y_true: np.ndarray, mean_prediction: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64)
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), 1e-8)
    return -(y * np.log(mean) - mean - gammaln(y + 1.0))


def _group_comparison(
    groups: np.ndarray,
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    metric: str,
) -> dict[str, object]:
    group_values = np.asarray(groups).astype(str)
    y = np.asarray(truth, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if not (group_values.shape == y.shape == candidate_values.shape == reference_values.shape):
        raise ValueError("group comparison inputs must be aligned vectors")

    def losses(prediction: np.ndarray) -> np.ndarray:
        if metric == "mae":
            return np.abs(prediction - y)
        if metric == "brier":
            return (prediction - y) ** 2
        if metric == "log_loss":
            return _binary_row_log_loss(y, prediction)
        if metric == "loss":
            return prediction
        raise ValueError(f"unsupported comparison metric: {metric}")

    candidate_loss = losses(candidate_values)
    reference_loss = losses(reference_values)
    candidate_by_group = {}
    reference_by_group = {}
    differences = {}
    for group in sorted(set(group_values.tolist())):
        selected = group_values == group
        candidate_by_group[group] = float(candidate_loss[selected].mean())
        reference_by_group[group] = float(reference_loss[selected].mean())
        differences[group] = candidate_by_group[group] - reference_by_group[group]
    return {
        "metric": metric,
        "group_count": len(differences),
        "candidate_equal_group_mean": float(np.mean(list(candidate_by_group.values()))),
        "reference_equal_group_mean": float(np.mean(list(reference_by_group.values()))),
        "candidate_minus_reference": float(np.mean(list(differences.values()))),
        "candidate_wins": sum(value < 0.0 for value in differences.values()),
        "reference_wins": sum(value > 0.0 for value in differences.values()),
        "ties": sum(value == 0.0 for value in differences.values()),
        "candidate_by_group": candidate_by_group,
        "reference_by_group": reference_by_group,
        "candidate_minus_reference_by_group": differences,
    }


def _dataset_decision(
    horizon_rows: dict[str, dict[str, object]], *, minimum_horizons: int
) -> dict[str, object]:
    passed = sorted(
        int(horizon)
        for horizon, row in horizon_rows.items()
        if row["count_distribution_pass"] and row["q75_exceedance_pass"]
    )
    return {
        "passed_horizons": passed,
        "passed_horizon_count": len(passed),
        "supports_dataset_specific_distributional_value": len(passed) >= minimum_horizons,
    }


def _source_groups(rows: list[dict[str, object]], config_path: Path) -> np.ndarray:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_to_group = {
        str(item["source_file_id"]): str(item["date"]) for item in config["sources"]
    }
    groups = []
    for row in rows:
        bound = {
            source_to_group[str(source_id)]
            for source_id in row["source_file_ids"]
            if str(source_id) in source_to_group
        }
        if len(bound) != 1:
            raise ValueError("scene is not bound to exactly one frozen source group")
        groups.append(next(iter(bound)))
    return np.asarray(groups)


def _distribution_summary(y: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(y, dtype=np.float64)
    mean = float(counts.mean())
    variance = float(counts.var())
    return {
        "scene_count": len(counts),
        "mean": mean,
        "variance": variance,
        "variance_to_mean_ratio": variance / mean if mean > 0.0 else math.nan,
        "zero_fraction": float(np.mean(counts == 0.0)),
    }


def _fit_models(x_train: np.ndarray, y_train: np.ndarray):
    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    standardized = (x_train - feature_mean) / feature_scale
    poisson_model = PoissonRegressor(alpha=0.0, max_iter=2000).fit(standardized, y_train)
    nb2_model = fit_nb2_regression(x_train, y_train, l2=0.0)
    return feature_mean, feature_scale, poisson_model, nb2_model


def _evaluate_dataset(
    *,
    rows: list[dict[str, object]],
    groups: np.ndarray,
    x_evaluation: np.ndarray,
    feature_names: list[str],
    x_train: np.ndarray,
    y_train_by_horizon: dict[str, np.ndarray],
    thresholds_by_horizon: dict[str, dict[str, int]],
    fitted_by_horizon: dict[str, tuple[np.ndarray, np.ndarray, object, dict[str, object]]],
    quantiles: list[float],
    minimum_horizons: int,
) -> dict[str, object]:
    result: dict[str, object] = {"source_groups": sorted(set(groups.tolist())), "horizons": {}}
    for horizon, y_train in y_train_by_horizon.items():
        y = np.asarray(
            [row["targets"][horizon]["future_onset_proximity_event_count"] for row in rows],
            dtype=np.float64,
        )
        mean, scale, poisson_model, nb2_model = fitted_by_horizon[horizon]
        standardized_eval = (x_evaluation - mean) / scale
        poisson_mean = np.maximum(poisson_model.predict(standardized_eval), 1e-8)
        nb2_mean = predict_nb2_mean(nb2_model, x_evaluation)
        count_comparison = _group_comparison(
            groups,
            np.zeros_like(y),
            nb2_log_loss(y, nb2_mean, float(nb2_model["dispersion"])),
            _poisson_row_nll(y, poisson_mean),
            metric="loss",
        )
        majority = len(set(groups.tolist())) // 2 + 1
        threshold_results = {}
        q75_pass = False
        for quantile in quantiles:
            qkey = f"{float(quantile):.2f}"
            threshold = thresholds_by_horizon[horizon][qkey]
            train_labels = (y_train >= threshold).astype(int)
            labels = (y >= threshold).astype(float)
            standardized_train = (x_train - mean) / scale
            logistic = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=1701,
            ).fit(standardized_train, train_labels)
            logistic_probability = logistic.predict_proba(standardized_eval)[:, 1]
            nb2_probability = nb2_risk_probability(
                nb2_mean,
                dispersion=float(nb2_model["dispersion"]),
                threshold=threshold,
            )
            brier = _group_comparison(
                groups, labels, nb2_probability, logistic_probability, metric="brier"
            )
            log_loss = _group_comparison(
                groups, labels, nb2_probability, logistic_probability, metric="log_loss"
            )
            threshold_pass = (
                brier["candidate_minus_reference"] < 0.0
                and brier["candidate_wins"] >= majority
            ) or (
                log_loss["candidate_minus_reference"] < 0.0
                and log_loss["candidate_wins"] >= majority
            )
            if qkey == "0.75":
                q75_pass = bool(threshold_pass)
            threshold_results[qkey] = {
                "threshold_frozen_from_KBTP_train": threshold,
                "observed_exceedance_fraction": float(labels.mean()),
                "NB2_vs_Logistic_Brier": brier,
                "NB2_vs_Logistic_log_loss": log_loss,
                "NB2_pooled_ECE_10bin": expected_calibration_error(
                    nb2_probability, labels, bins=10
                ),
                "Logistic_pooled_ECE_10bin": expected_calibration_error(
                    logistic_probability, labels, bins=10
                ),
                "threshold_pass": bool(threshold_pass),
            }
        result["horizons"][horizon] = {
            "distribution": _distribution_summary(y),
            "NB2_dispersion_frozen_from_KBTP_train": float(nb2_model["dispersion"]),
            "NB2_vs_Poisson_count_NLL": count_comparison,
            "NB2_vs_Poisson_count_MAE": _group_comparison(
                groups, y, nb2_mean, poisson_mean, metric="mae"
            ),
            "count_distribution_pass": bool(
                count_comparison["candidate_minus_reference"] < 0.0
                and count_comparison["candidate_wins"] >= majority
            ),
            "threshold_sensitivity": threshold_results,
            "q75_exceedance_pass": q75_pass,
            "KBTP_train_to_evaluation_standardized_mean_shift": {
                name: float(value)
                for name, value in zip(
                    feature_names, ((x_evaluation.mean(axis=0) - mean) / scale), strict=True
                )
            },
        }
    result["decision"] = _dataset_decision(
        result["horizons"], minimum_horizons=minimum_horizons
    )
    return result


def run(protocol_path: Path, output: Path) -> dict[str, object]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "r03_nb2_review_protocol_v1":
        raise ValueError("unsupported NB2 review protocol schema")
    _verify_dataset_hashes(CHECKOUT_ROOT, protocol["datasets"])
    paths = {
        name: CHECKOUT_ROOT / str(record["relative_path"])
        for name, record in protocol["datasets"].items()
    }
    development_rows = _read_rows(paths["development"].parent)
    train_rows = [row for row in development_rows if row["split"] == "train"]
    confirmation_rows = _read_rows(paths["confirmation"].parent)
    external_rows = _read_rows(paths["external"].parent)
    if {row["airport_id"] for row in train_rows} != {"KBTP"}:
        raise ValueError("training rows must contain only KBTP")
    if {row["airport_id"] for row in external_rows} != {"KAGC"}:
        raise ValueError("external rows must contain only KAGC")

    x_train, feature_names_tuple = _matrix(train_rows, "B")
    x_confirmation, confirmation_names = _matrix(confirmation_rows, "B")
    x_external, external_names = _matrix(external_rows, "B")
    if feature_names_tuple != confirmation_names or feature_names_tuple != external_names:
        raise ValueError("feature schemas differ across frozen datasets")
    feature_names = list(feature_names_tuple)
    confirmation_groups = _source_groups(
        confirmation_rows, CHECKOUT_ROOT / "configs/r03_future_onset_confirmation_dates.json"
    )
    external_groups = _source_groups(
        external_rows, CHECKOUT_ROOT / "configs/r03_kagc_external_validation.json"
    )

    y_train_by_horizon = {}
    thresholds_by_horizon = {}
    fitted_by_horizon = {}
    model_records = {}
    for horizon_value in protocol["horizons_seconds"]:
        horizon = str(horizon_value)
        y_train = np.asarray(
            [row["targets"][horizon][protocol["target"]] for row in train_rows],
            dtype=np.float64,
        )
        y_train_by_horizon[horizon] = y_train
        thresholds_by_horizon[horizon] = _frozen_thresholds(
            y_train, protocol["threshold_quantiles"]
        )
        fitted = _fit_models(x_train, y_train)
        fitted_by_horizon[horizon] = fitted
        mean, scale, poisson_model, nb2_model = fitted
        model_records[horizon] = {
            "training_distribution": _distribution_summary(y_train),
            "thresholds": thresholds_by_horizon[horizon],
            "NB2": {
                "converged": bool(nb2_model["converged"]),
                "iterations": int(nb2_model["iterations"]),
                "objective": float(nb2_model["objective"]),
                "dispersion": float(nb2_model["dispersion"]),
                "intercept": float(nb2_model["intercept"]),
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        feature_names, nb2_model["coefficients"], strict=True
                    )
                },
            },
            "Poisson": {
                "intercept": float(poisson_model.intercept_),
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(feature_names, poisson_model.coef_, strict=True)
                },
            },
        }

    minimum_horizons = int(protocol["main_text_gate"]["minimum_horizons"])
    evaluations = {
        "KBTP_confirmation": _evaluate_dataset(
            rows=confirmation_rows,
            groups=confirmation_groups,
            x_evaluation=x_confirmation,
            feature_names=feature_names,
            x_train=x_train,
            y_train_by_horizon=y_train_by_horizon,
            thresholds_by_horizon=thresholds_by_horizon,
            fitted_by_horizon=fitted_by_horizon,
            quantiles=protocol["threshold_quantiles"],
            minimum_horizons=minimum_horizons,
        ),
        "KAGC_external": _evaluate_dataset(
            rows=external_rows,
            groups=external_groups,
            x_evaluation=x_external,
            feature_names=feature_names,
            x_train=x_train,
            y_train_by_horizon=y_train_by_horizon,
            thresholds_by_horizon=thresholds_by_horizon,
            fitted_by_horizon=fitted_by_horizon,
            quantiles=protocol["threshold_quantiles"],
            minimum_horizons=minimum_horizons,
        ),
    }
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CHECKOUT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    import scipy
    import sklearn

    result = {
        "schema_version": "r03_nb2_review_supplement_result_v1",
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "dataset_hashes": {
            name: str(record["sha256"]) for name, record in protocol["datasets"].items()
        },
        "training_airport": "KBTP",
        "external_airport": "KAGC",
        "site_tuning_performed": False,
        "target": protocol["target"],
        "feature_ladder": protocol["feature_ladder"],
        "feature_names": feature_names,
        "models": model_records,
        "evaluations": evaluations,
        "interpretation_policy": {
            "fallback": protocol["fallback"],
            "claim_boundary": protocol["claim_boundary"],
        },
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda": "not used",
            "code_version": completed.stdout.strip() if completed.returncode == 0 else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=CHECKOUT_ROOT / "configs/r03_nb2_review_protocol.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHECKOUT_ROOT
        / "reports/references/r03_nb2_review_supplement_2026-09-05.json",
    )
    args = parser.parse_args(argv)
    result = run(args.protocol, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "KBTP_confirmation": result["evaluations"]["KBTP_confirmation"]["decision"],
                "KAGC_external": result["evaluations"]["KAGC_external"]["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
