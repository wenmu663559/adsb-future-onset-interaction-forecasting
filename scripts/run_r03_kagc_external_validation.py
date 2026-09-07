from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT))
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airspace_complexity.r03_future_onset import exposure_rate_predictions
from airspace_complexity.r03_probabilistic_multihorizon import (
    binary_log_loss,
    count_metrics,
    expected_calibration_error,
)
from scripts.run_r03_future_onset_confirmation import (
    _day_differences,
    _fit_count,
    _matrix,
    _read_rows,
    _sha256,
)


def run(
    development_root: Path, kagc_root: Path, protocol_path: Path,
    external_protocol_path: Path, output: Path,
) -> dict[str, object]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    frozen = json.loads(Path(external_protocol_path).read_text(encoding="utf-8"))
    dependency_hashes = {
        CHECKOUT_ROOT / "configs" / "r03_kagc_external_validation.json": frozen["KAGC_date_config_sha256"],
        CHECKOUT_ROOT / "reports" / "references" / "r03_future_onset_locked_evidence_2026-09-03.json": frozen["locked_KBTP_evidence_sha256"],
        CHECKOUT_ROOT / "reports" / "references" / "r03_future_onset_confirmation_2026-09-03.json": frozen["KBTP_confirmation_sha256"],
    }
    for path, expected in dependency_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen external dependency hash mismatch: {path}")

    development_rows = _read_rows(development_root)
    train_rows = [row for row in development_rows if row["split"] == "train"]
    kagc_rows = _read_rows(kagc_root)
    if {row["airport_id"] for row in train_rows} != {"KBTP"}:
        raise ValueError("external models must be trained only on KBTP")
    if {row["airport_id"] for row in kagc_rows} != {"KAGC"}:
        raise ValueError("external rows must contain only KAGC")
    date_config = json.loads(
        (CHECKOUT_ROOT / "configs" / "r03_kagc_external_validation.json")
        .read_text(encoding="utf-8")
    )
    source_to_group = {
        str(item["source_file_id"]): str(item["date"])
        for item in date_config["sources"]
    }
    groups = []
    for row in kagc_rows:
        bound = {
            source_to_group[source_id] for source_id in row["source_file_ids"]
            if source_id in source_to_group
        }
        if len(bound) != 1:
            raise ValueError("KAGC scene is not bound to one frozen source group")
        groups.append(next(iter(bound)))
    groups = np.asarray(groups)
    if len(set(groups)) != 10:
        raise ValueError("KAGC external set must contain ten source groups")

    matrices = {}
    feature_names = {}
    for ladder in ("A", "B"):
        matrices[("train", ladder)], names = _matrix(train_rows, ladder)
        matrices[("KAGC", ladder)], kagc_names = _matrix(kagc_rows, ladder)
        if names != kagc_names:
            raise ValueError("KBTP and KAGC feature schemas differ")
        feature_names[ladder] = list(names)

    horizon_results: dict[str, object] = {}
    passed_horizons = []
    for horizon in protocol["horizons_seconds"]:
        key = str(horizon)
        train_targets = np.asarray([
            row["targets"][key]["future_onset_proximity_event_count"]
            for row in train_rows
        ], dtype=np.float64)
        kagc_targets = np.asarray([
            row["targets"][key]["future_onset_proximity_event_count"]
            for row in kagc_rows
        ], dtype=np.float64)
        model_name = frozen["selected_count_models_by_horizon"][key]
        predictions = {}
        for ladder in ("A", "B"):
            _, predict = _fit_count(
                model_name, matrices[("train", ladder)], train_targets
            )
            predictions[ladder] = predict(matrices[("KAGC", ladder)])
        pair_train = matrices[("train", "B")][
            :, feature_names["B"].index("observed_pair_count")
        ]
        pair_kagc = matrices[("KAGC", "B")][
            :, feature_names["B"].index("observed_pair_count")
        ]
        pair_baseline = np.asarray(exposure_rate_predictions(
            train_targets, pair_train, pair_kagc
        ))
        threshold = max(
            1, int(np.quantile(train_targets, 0.75, method="higher"))
        )
        train_risk = (train_targets >= threshold).astype(int)
        kagc_risk = (kagc_targets >= threshold).astype(int)
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=1701,
            ),
        )
        logistic.fit(matrices[("train", "B")], train_risk)
        logistic_probability = logistic.predict_proba(matrices[("KAGC", "B")])[:, 1]
        prevalence_probability = np.full(len(kagc_rows), float(train_risk.mean()))
        versus_exposure = _day_differences(
            groups, kagc_targets, predictions["B"], pair_baseline
        )
        versus_A = _day_differences(
            groups, kagc_targets, predictions["B"], predictions["A"]
        )
        versus_prevalence = _day_differences(
            groups, kagc_risk, logistic_probability, prevalence_probability,
            probability=True,
        )
        passed = all((
            versus_exposure["candidate_equal_day_mean"]
            < versus_exposure["baseline_equal_day_mean"],
            versus_exposure["candidate_wins"] >= 6,
            versus_prevalence["candidate_equal_day_mean"]
            < versus_prevalence["baseline_equal_day_mean"],
            versus_prevalence["candidate_wins"] >= 6,
        ))
        if passed:
            passed_horizons.append(int(horizon))
        horizon_results[key] = {
            "selected_count_model": model_name,
            "risk_threshold_frozen_from_KBTP": threshold,
            "passed_external_gate": passed,
            "B_vs_KBTP_pair_exposure_count_MAE": versus_exposure,
            "B_vs_A_count_MAE": versus_A,
            "B_logistic_vs_KBTP_prevalence_Brier": versus_prevalence,
            "B_pooled_count_metrics": count_metrics(kagc_targets, predictions["B"]),
            "B_pooled_probability_metrics": {
                "brier": float(np.mean((logistic_probability - kagc_risk) ** 2)),
                "log_loss": binary_log_loss(kagc_risk, logistic_probability),
                "ece_10_bins": expected_calibration_error(
                    logistic_probability, kagc_risk, bins=10
                ),
            },
            "KAGC_target_mean": float(kagc_targets.mean()),
            "KAGC_risk_prevalence": float(kagc_risk.mean()),
            "KBTP_training_target_mean": float(train_targets.mean()),
            "KBTP_training_risk_prevalence": float(train_risk.mean()),
        }
    route_passed = len(passed_horizons) >= 2
    result: dict[str, object] = {
        "schema_version": "r03_kagc_external_validation_v1",
        "decision": (
            "TWO_AIRPORT_TRANSFER_SUPPORTED_WITHOUT_KAGC_TUNING"
            if route_passed else "RETAIN_KBTP_ONLY_PRIMARY_CLAIM"
        ),
        "route_passed": route_passed,
        "passed_horizons": passed_horizons,
        "training_airport": "KBTP",
        "external_airport": "KAGC",
        "site_tuning_performed": False,
        "external_protocol_sha256": _sha256(Path(external_protocol_path)),
        "KAGC_scene_sha256": _sha256(Path(kagc_root) / "pilot_scenes.jsonl"),
        "KAGC_scene_count": len(kagc_rows),
        "independent_source_groups": sorted(set(groups)),
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
        description="Apply the frozen KBTP future-onset scheme to KAGC without tuning."
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--kagc-root", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json",
    )
    parser.add_argument(
        "--external-protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_kagc_external_validation_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.development_root, args.kagc_root,
        args.protocol, args.external_protocol, args.output,
    )
    print(json.dumps({
        "decision": result["decision"],
        "passed_horizons": result["passed_horizons"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
