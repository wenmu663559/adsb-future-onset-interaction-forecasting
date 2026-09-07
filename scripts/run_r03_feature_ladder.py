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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airspace_complexity.r03_future_onset import build_feature_ladder
from airspace_complexity.r03_probabilistic_multihorizon import (
    binary_log_loss,
    calibration_intercept_slope,
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


def _array_sha256(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        materialized = np.ascontiguousarray(array)
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(str(materialized.shape).encode("ascii"))
        digest.update(materialized.tobytes())
    return digest.hexdigest()


def _probability_metrics(labels, probabilities) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    metrics: dict[str, object] = {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": binary_log_loss(y, p),
        "ece_10_bins": expected_calibration_error(p, y, bins=10),
    }
    metrics["calibration"] = (
        calibration_intercept_slope(y, p)
        if len(np.unique(y)) == 2 else None
    )
    return metrics


def _evaluate_by_day(
    *, targets, risk_labels, count_predictions, risk_predictions, days, selected,
) -> dict[str, object]:
    per_day: dict[str, object] = {}
    for day in sorted(set(days[selected])):
        day_rows = selected & (days == day)
        per_day[day] = {
            "scene_count": int(day_rows.sum()),
            "target_mean": float(targets[day_rows].mean()),
            "count_models": {
                name: count_metrics(targets[day_rows], prediction[day_rows])
                for name, prediction in count_predictions.items()
            },
            "probability_models": {
                name: _probability_metrics(risk_labels[day_rows], prediction[day_rows])
                for name, prediction in risk_predictions.items()
            },
        }
    equal_day_count_mae = {
        name: float(np.mean([
            day["count_models"][name]["count_mae"] for day in per_day.values()
        ]))
        for name in count_predictions
    }
    equal_day_brier = {
        name: float(np.mean([
            day["probability_models"][name]["brier"] for day in per_day.values()
        ]))
        for name in risk_predictions
    }
    pooled_count = {
        name: {
            **count_metrics(targets[selected], prediction[selected]),
            "normalized_mae": (
                count_metrics(targets[selected], prediction[selected])["count_mae"]
                / max(float(targets[selected].mean()), 1e-12)
            ),
        }
        for name, prediction in count_predictions.items()
    }
    return {
        "scene_count": int(selected.sum()),
        "day_count": len(per_day),
        "equal_day_count_mae": equal_day_count_mae,
        "equal_day_brier": equal_day_brier,
        "pooled_count_metrics": pooled_count,
        "pooled_probability_metrics": {
            name: _probability_metrics(risk_labels[selected], prediction[selected])
            for name, prediction in risk_predictions.items()
        },
        "days": per_day,
    }


def run(dataset_root: Path, protocol_path: Path, output: Path) -> dict[str, object]:
    root = Path(dataset_root)
    scene_path = root / "pilot_scenes.jsonl"
    rows = [json.loads(line) for line in scene_path.open(encoding="utf-8")]
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    scene_ids = np.asarray([str(row["scene_id"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    days = np.asarray([str(row["source_day"]) for row in rows])
    train = splits == "train"
    if not train.any() or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("feature ladder requires unique scenes and training rows")

    feature_rows = []
    for row in rows:
        center = AIRPORT_CENTERS[str(row["airport_id"])]
        feature_rows.append(build_feature_ladder(
            row, airport_lat_deg=center[0], airport_lon_deg=center[1]
        ))
    feature_names = {
        ladder: tuple(feature_rows[0][ladder]) for ladder in ("A", "B", "C")
    }
    matrices = {
        ladder: np.asarray([
            [features[ladder][name] for name in feature_names[ladder]]
            for features in feature_rows
        ], dtype=np.float64)
        for ladder in ("A", "B", "C")
    }
    if any(
        matrix.shape[0] != len(rows) or not np.isfinite(matrix).all()
        for matrix in matrices.values()
    ):
        raise ValueError("feature matrices are not matched finite scene rows")

    horizon_results: dict[str, object] = {}
    for horizon in protocol["horizons_seconds"]:
        targets = np.asarray([
            int(row["targets"][str(horizon)]["future_onset_proximity_event_count"])
            for row in rows
        ], dtype=np.float64)
        threshold = max(
            1, int(np.quantile(targets[train], 0.75, method="higher"))
        )
        risk_labels = (targets >= threshold).astype(int)
        prevalence = float(risk_labels[train].mean())
        ladder_results: dict[str, object] = {}
        for ladder in ("A", "B", "C"):
            x = matrices[ladder]
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(x[train], np.log1p(targets[train]))
            ridge_prediction = np.maximum(np.expm1(ridge.predict(x)), 0.0)
            hgb = HistGradientBoostingRegressor(
                loss="poisson", max_iter=100, max_leaf_nodes=15,
                learning_rate=0.05, l2_regularization=1.0, random_state=1701,
            )
            hgb.fit(x[train], targets[train])
            hgb_prediction = np.maximum(hgb.predict(x), 0.0)
            poisson = make_pipeline(
                StandardScaler(), PoissonRegressor(alpha=0.0, max_iter=1000)
            )
            poisson.fit(x[train], targets[train])
            poisson_prediction = np.maximum(poisson.predict(x), 0.0)
            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, penalty="l2", solver="lbfgs",
                    max_iter=1000, random_state=1701,
                ),
            )
            logistic.fit(x[train], risk_labels[train])
            logistic_probability = logistic.predict_proba(x)[:, 1]
            count_predictions = {
                "ridge_log1p": ridge_prediction,
                "hist_gradient_boosting_poisson": hgb_prediction,
                "plain_poisson": poisson_prediction,
            }
            risk_predictions = {
                "training_prevalence": np.full(len(rows), prevalence),
                "logistic_ridge": logistic_probability,
            }
            ladder_results[ladder] = {
                "feature_names": list(feature_names[ladder]),
                "feature_count": len(feature_names[ladder]),
                "feature_sha256": _array_sha256(x),
                "validation": _evaluate_by_day(
                    targets=targets, risk_labels=risk_labels,
                    count_predictions=count_predictions,
                    risk_predictions=risk_predictions,
                    days=days, selected=splits == "validation",
                ),
                "test": _evaluate_by_day(
                    targets=targets, risk_labels=risk_labels,
                    count_predictions=count_predictions,
                    risk_predictions=risk_predictions,
                    days=days, selected=splits == "test",
                ),
            }
        horizon_results[str(horizon)] = {
            "risk_threshold": threshold,
            "training_risk_prevalence": prevalence,
            "target_sha256": _array_sha256(targets),
            "ladders": ladder_results,
        }
    result: dict[str, object] = {
        "schema_version": "r03_future_onset_feature_ladder_v1",
        "decision": "DEVELOPMENT_AND_RETROSPECTIVE_EVIDENCE_NOT_CONFIRMATION",
        "dataset_scene_sha256": _sha256(scene_path),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "scene_id_sha256": hashlib.sha256("\n".join(scene_ids).encode("utf-8")).hexdigest(),
        "split_sha256": hashlib.sha256("\n".join(splits).encode("utf-8")).hexdigest(),
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
        description="Run matched A/B/C future-onset feature ladder."
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
