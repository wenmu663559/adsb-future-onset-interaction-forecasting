"""Run the bounded A/B test: invariant relation versus masked temporal relation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
for path in (CHECKOUT_ROOT / "src", SCRIPT_PATH.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from airspace_complexity.r03_frozen_probe import (
    fit_frozen_representation_probes,
    load_invariant_relation_checkpoint,
)
from airspace_complexity.r03_masked_temporal_ab import decide_masked_temporal_ab
from airspace_complexity.r03_masked_temporal_relation import MaskedTemporalRelation
from probe_r03_frozen_task_representation import (
    _add_seed_summary,
    _frozen_outputs,
    _load_indicators,
)
from run_r03_online_complexity_pilot import _load_experiment_arrays, _metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version() -> tuple[str | None, bool | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else None,
        bool(status.stdout.strip()) if status.returncode == 0 else None,
    )


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluate_predictions(y, split, predictions_by_seed):
    metrics = []
    for seed, prediction in sorted(predictions_by_seed.items()):
        for subset_name in ("validation", "test"):
            selected = np.flatnonzero(split == subset_name)
            metrics.append(
                {
                    "seed": seed,
                    "subset": subset_name,
                    **_metrics(y[selected], prediction[selected]),
                }
            )
    return metrics


def _daily_metrics(y, split, days, baseline_predictions, candidate_predictions):
    rows = []
    test_days = sorted(set(days[split == "test"].tolist()))
    for seed in sorted(baseline_predictions):
        for day in test_days:
            selected = np.flatnonzero((split == "test") & (days == day))
            for model_name, predictions in (
                ("A_invariant_relation", baseline_predictions),
                ("B_masked_temporal_relation", candidate_predictions),
            ):
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "day": day,
                        "scene_count": len(selected),
                        **_metrics(y[selected], predictions[seed][selected]),
                    }
                )
    return rows


def _latency(model, steps: int, candidate: bool):
    import torch

    rows = []
    model.eval()
    for aircraft_count in (20, 50, 100):
        synthetic = torch.zeros((1, aircraft_count, steps, 7))
        synthetic[:, :, :, 6] = 1.0
        with torch.inference_mode():
            for _ in range(10):
                model(synthetic)
            timings = []
            for _ in range(100):
                started = time.perf_counter()
                model(synthetic)
                timings.append((time.perf_counter() - started) * 1000.0)
        rows.append(
            {
                "model": (
                    "B_masked_temporal_relation"
                    if candidate
                    else "A_invariant_relation"
                ),
                "aircraft_count": aircraft_count,
                "p50_ms": float(np.percentile(timings, 50)),
                "p95_ms": float(np.percentile(timings, 95)),
            }
        )
    return rows


def _train_candidate(
    *, x, y, split, airports, mean, scale, seed: int, epochs: int,
    output_root: Path, dataset_sha256: str,
):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    model = MaskedTemporalRelation(mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    airport_target = torch.from_numpy((airports == "KBTP").astype(np.int64))
    rng = np.random.default_rng(seed)
    log = []
    best_state = None
    best_validation_mse = float("inf")
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        totals = {"supervised": 0.0, "reconstruction": 0.0, "domain": 0.0}
        observed_batches = 0
        for start in range(0, len(order), 64):
            indices = torch.from_numpy(order[start : start + 64]).long()
            output = model(xt[indices], mask_probability=0.20)
            supervised_loss = nn.functional.mse_loss(
                output["prediction"], yt[indices]
            )
            selected = output["selected"]
            reconstruction_loss = nn.functional.mse_loss(
                output["reconstruction"][selected],
                output["reconstruction_target"][selected],
            )
            domain_loss = nn.functional.cross_entropy(
                model.domain_logits(output["representation"]),
                airport_target[indices],
            )
            loss = supervised_loss + 0.10 * reconstruction_loss + 0.10 * domain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["supervised"] += float(supervised_loss.detach())
            totals["reconstruction"] += float(reconstruction_loss.detach())
            totals["domain"] += float(domain_loss.detach())
            observed_batches += 1
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(xt[validation])["prediction"]
            validation_mse = float(
                nn.functional.mse_loss(validation_prediction, yt[validation])
            )
        log.append(
            {
                "epoch": epoch,
                "train_supervised_mse": totals["supervised"] / observed_batches,
                "train_reconstruction_mse": (
                    totals["reconstruction"] / observed_batches
                ),
                "train_domain_cross_entropy": totals["domain"] / observed_batches,
                "validation_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    checkpoint_path = output_root / f"masked_temporal_relation__seed-{seed}.pt"
    torch.save(
        {
            "model": "masked_temporal_relation",
            "seed": seed,
            "state_dict": best_state,
            "dataset_sha256": dataset_sha256,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_mse,
            "mask_probability": 0.20,
            "reconstruction_weight": 0.10,
            "airport_adversarial_weight": 0.10,
            "target": "log1p(120_second_proximity_event_count)",
        },
        checkpoint_path,
    )
    model.eval()
    predictions = []
    representations = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            output = model(xt[start : start + 128])
            predictions.append(output["prediction"].numpy())
            representations.append(output["representation"].numpy())
    return {
        "model": model,
        "prediction": np.concatenate(predictions),
        "representation": np.concatenate(representations),
        "training_log": log,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_mse,
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    seeds = tuple(args.seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct seeds are required")

    args.output_root.mkdir(parents=True, exist_ok=False)
    scene_path = args.dataset_root / "pilot_scenes.jsonl"
    dataset_sha256 = _sha256(scene_path)
    code_version, code_dirty = _git_version()
    run_id = args.output_root.name
    manifest = {
        "schema_version": "r03_masked_temporal_ab_plan_v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": dataset_sha256,
        "models": ["A_invariant_relation", "B_masked_temporal_relation"],
        "epochs": args.epochs,
        "seeds": list(seeds),
        "mask_probability": 0.20,
        "reconstruction_weight": 0.10,
        "airport_adversarial_weight": 0.10,
        "code_version": code_version,
        "code_dirty_at_start": code_dirty,
        "stop_rule": "PROMOTE_B only if all four preregistered gates pass",
    }
    _json_write(args.output_root / "run_manifest.json", manifest)

    x, targets, _, split, airports, mean, scale = _load_experiment_arrays(
        args.dataset_root
    )
    y = targets[:, 1]
    indicators, days, current_count = _load_indicators(scene_path)

    baseline_predictions = {}
    baseline_representations = {}
    baseline_models = {}
    baseline_checkpoints = []
    for path in args.baseline_checkpoint:
        model, metadata = load_invariant_relation_checkpoint(path)
        seed = int(metadata["seed"])
        if seed not in seeds:
            continue
        prediction, representation = _frozen_outputs(model, x)
        baseline_predictions[seed] = prediction
        baseline_representations[seed] = representation
        baseline_models[seed] = model
        baseline_checkpoints.append(
            {"seed": seed, "path": str(path.resolve()), "sha256": _sha256(path)}
        )
    if set(baseline_predictions) != set(seeds):
        raise ValueError("one baseline checkpoint is required for every seed")

    candidate_predictions = {}
    candidate_representations = {}
    candidate_runs = []
    candidate_models = {}
    for seed in seeds:
        run = _train_candidate(
            x=x, y=y, split=split, airports=airports, mean=mean, scale=scale,
            seed=seed, epochs=args.epochs, output_root=args.output_root,
            dataset_sha256=dataset_sha256,
        )
        candidate_predictions[seed] = run.pop("prediction")
        candidate_representations[seed] = run.pop("representation")
        candidate_models[seed] = run.pop("model")
        candidate_runs.append({"seed": seed, **run})
        _json_write(args.output_root / "partial_training_runs.json", candidate_runs)

    baseline_probe = fit_frozen_representation_probes(
        representations=baseline_representations,
        predictions=baseline_predictions,
        indicators=indicators,
        current_aircraft_count=current_count,
        split=split,
        days=days,
    )
    candidate_probe = fit_frozen_representation_probes(
        representations=candidate_representations,
        predictions=candidate_predictions,
        indicators=indicators,
        current_aircraft_count=current_count,
        split=split,
        days=days,
    )
    _add_seed_summary(baseline_probe)
    _add_seed_summary(candidate_probe)
    baseline_metrics = _evaluate_predictions(y, split, baseline_predictions)
    candidate_metrics = _evaluate_predictions(y, split, candidate_predictions)
    daily = _daily_metrics(
        y, split, days, baseline_predictions, candidate_predictions
    )

    baseline_mae = {
        row["seed"]: row["count_mae"]
        for row in baseline_metrics if row["subset"] == "test"
    }
    candidate_mae = {
        row["seed"]: row["count_mae"]
        for row in candidate_metrics if row["subset"] == "test"
    }
    day_delta = {}
    for day in sorted(set(days[split == "test"].tolist())):
        a = [
            row["count_mae"] for row in daily
            if row["day"] == day and row["model"] == "A_invariant_relation"
        ]
        b = [
            row["count_mae"] for row in daily
            if row["day"] == day and row["model"] == "B_masked_temporal_relation"
        ]
        day_delta[day] = float(np.mean(b) - np.mean(a))
    a_closing = baseline_probe["indicators"]["closing_pair_count_10nm"][
        "summary"
    ]["test"]["edge_only_minus_count_r2"]["mean"]
    b_closing = candidate_probe["indicators"]["closing_pair_count_10nm"][
        "summary"
    ]["test"]["edge_only_minus_count_r2"]["mean"]
    a_maneuver = baseline_probe["indicators"]["maneuvering_aircraft_count"][
        "summary"
    ]["test"]["representation_minus_count_r2"]["mean"]
    b_maneuver = candidate_probe["indicators"]["maneuvering_aircraft_count"][
        "summary"
    ]["test"]["representation_minus_count_r2"]["mean"]
    decision = decide_masked_temporal_ab(
        baseline_mae_by_seed=baseline_mae,
        candidate_mae_by_seed=candidate_mae,
        day_mae_delta=day_delta,
        closing_edge_delta_change=b_closing - a_closing,
        maneuver_representation_delta_change=b_maneuver - a_maneuver,
    )

    latency = _latency(baseline_models[seeds[0]], x.shape[2], False)
    latency.extend(_latency(candidate_models[seeds[0]], x.shape[2], True))
    import scipy
    import sklearn
    import torch

    result = {
        "schema_version": "r03_masked_temporal_ab_results_v1",
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "scene_counts": {
            name: int((split == name).sum())
            for name in ("train", "validation", "test")
        },
        "independent_days": len(set(days.tolist())),
        "baseline_checkpoints": baseline_checkpoints,
        "candidate_runs": candidate_runs,
        "prediction_metrics": {
            "A_invariant_relation": baseline_metrics,
            "B_masked_temporal_relation": candidate_metrics,
        },
        "test_day_metrics": daily,
        "test_day_candidate_minus_baseline_mae": day_delta,
        "task_probes": {
            "A_invariant_relation": baseline_probe,
            "B_masked_temporal_relation": candidate_probe,
        },
        "decision": decision,
        "latency": latency,
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "cuda": "not used",
        },
        "limitations": [
            "The future proximity-event target remains an operational proxy.",
            "Task indicators are interpretability references, not controller workload labels.",
            "B uses one preregistered configuration; no hyperparameter search was performed.",
            "Four held-out days limit precision of day-level generalization estimates.",
        ],
    }
    result_path = args.output_root / "masked_temporal_ab_results.json"
    _json_write(result_path, result)
    print(
        json.dumps(
            {"result": str(result_path), "decision": decision["decision"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
