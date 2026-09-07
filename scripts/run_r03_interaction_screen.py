"""Run the engineering-only R03 multi-aircraft interaction screen."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from airspace_complexity.r03_engineering_smoke import (  # noqa: E402
    build_scene_transitions,
    build_scene_windows,
    constant_velocity_position_delta,
    engineering_decision,
    feature_view,
    interaction_screen_specs,
)
from airspace_complexity.r03_interaction_models import (  # noqa: E402
    DeepSetsRegressor,
    RelationRegressor,
    TokenRegressor,
)


TARGET_FIELDS = ("altitude_m", "speed_mps", "sin_heading", "cos_heading")
FUTURE_TARGET_FIELDS = (
    "lat_deg", "lon_deg", "altitude_m", "speed_mps", "sin_heading", "cos_heading",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _airport_centers(
    blocks: list[dict[str, Any]], fields: list[str],
) -> dict[str, tuple[float, float]]:
    lat_index, lon_index = fields.index("lat_deg"), fields.index("lon_deg")
    by_airport: dict[str, tuple[list[float], list[float]]] = {}
    for block in blocks:
        latitudes, longitudes = by_airport.setdefault(
            str(block["airport_id"]), ([], [])
        )
        for row in block["rows"]:
            if row["values"][lat_index] is not None:
                latitudes.append(float(row["values"][lat_index]))
            if row["values"][lon_index] is not None:
                longitudes.append(float(row["values"][lon_index]))
    return {
        airport: (float(np.mean(latitudes)), float(np.mean(longitudes)))
        for airport, (latitudes, longitudes) in by_airport.items()
    }


def _center_scenes(
    scenes: list[dict[str, Any]], fields: list[str],
    centers: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scene in scenes:
        transformed = dict(scene)
        transformed_rows: list[dict[str, Any]] = []
        for row in scene["rows"]:
            _, values = feature_view(
                fields, list(row["values"]), airport_id=str(scene["airport_id"]),
                mode="airport_centered", airport_centers=centers,
            )
            transformed_rows.append(dict(row, values=values))
        transformed["rows"] = transformed_rows
        output.append(transformed)
    return output


def _normalization(
    scenes: list[dict[str, Any]], width: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([
        [np.nan if value is None else float(value) for value in row["values"]]
        for scene in scenes if scene["split"] == "train"
        for row in scene["rows"]
    ], dtype=np.float64)
    if values.shape[1] != width:
        raise ValueError("feature width changed")
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("training normalization is non-finite")
    scales[scales < 1e-6] = 1.0
    return means, scales


def _arrays(
    scenes: list[dict[str, Any]], fields: list[str], means: np.ndarray,
    scales: np.ndarray, *, max_aircraft: int,
) -> dict[str, tuple[np.ndarray, ...]]:
    target_columns = [fields.index(field) for field in TARGET_FIELDS]
    buckets: dict[str, list[tuple[np.ndarray, ...]]] = {
        "train": [], "validation": [], "test": [],
    }
    for scene in scenes:
        raw = np.array([
            [np.nan if value is None else float(value) for value in row["values"]]
            for row in scene["rows"]
        ], dtype=np.float64)
        observed = np.isfinite(raw)
        normalized = (raw - means) / scales
        normalized[~observed] = 0.0
        for target_index in range(len(scene["rows"])):
            if not observed[target_index, target_columns].all():
                continue
            visible = normalized.copy()
            visible[target_index, target_columns] = 0.0
            target_masks = np.zeros((len(raw), len(target_columns)), dtype=np.float64)
            target_masks[target_index, :] = 1.0
            target_flags = np.zeros((len(raw), 1), dtype=np.float64)
            target_flags[target_index, 0] = 1.0
            node_features = np.concatenate([
                visible, (~observed).astype(np.float64), target_masks, target_flags,
            ], axis=1)
            padded = np.zeros((max_aircraft, node_features.shape[1]), dtype=np.float32)
            padded[:len(raw)] = node_features.astype(np.float32)
            valid = np.zeros(max_aircraft, dtype=bool)
            valid[:len(raw)] = True
            relations = np.zeros((max_aircraft, 3), dtype=np.float32)
            delta = normalized[:, :2] - normalized[target_index, :2]
            relations[:len(raw), :2] = delta.astype(np.float32)
            relations[:len(raw), 2] = np.sqrt((delta ** 2).sum(axis=1)).astype(np.float32)
            target = normalized[target_index, target_columns].astype(np.float32)
            buckets[str(scene["split"])].append((
                padded, valid, np.int64(target_index), relations, target,
            ))
    result: dict[str, tuple[np.ndarray, ...]] = {}
    for split, rows in buckets.items():
        if not rows:
            raise ValueError(f"no interaction examples for {split}")
        result[split] = tuple(np.stack(items) for items in zip(*rows, strict=True))
    return result


def _model(kind: str, node_dim: int, output_dim: int) -> nn.Module:
    kwargs = {"node_dim": node_dim, "hidden_dim": 64, "output_dim": output_dim}
    if kind == "token_mlp":
        return TokenRegressor(**kwargs)
    if kind == "deepsets":
        return DeepSetsRegressor(**kwargs)
    if kind == "relation_gnn":
        return RelationRegressor(**kwargs, relation_dim=3)
    raise ValueError(f"unknown model: {kind}")


def _forward(
    model: nn.Module, kind: str, nodes: torch.Tensor, valid: torch.Tensor,
    target_indices: torch.Tensor, relations: torch.Tensor,
) -> torch.Tensor:
    if kind == "relation_gnn":
        return model(nodes, valid, target_indices, relations)
    return model(nodes, valid, target_indices)


@torch.no_grad()
def _evaluate(
    model: nn.Module, kind: str, arrays: tuple[np.ndarray, ...],
    *, shuffle_relations: bool = False,
) -> float:
    nodes, valid, target_indices, relations, targets = arrays
    model.eval()
    nodes_t = torch.from_numpy(nodes)
    valid_t = torch.from_numpy(valid)
    target_t = torch.from_numpy(target_indices)
    relations_t = torch.from_numpy(relations.copy())
    if shuffle_relations:
        for row in range(len(relations_t)):
            indices = torch.where(valid_t[row])[0].tolist()
            neighbors = [index for index in indices if index != int(target_t[row])]
            if len(neighbors) > 1:
                relations_t[row, neighbors] = relations_t[row, list(reversed(neighbors))]
    predictions = _forward(model, kind, nodes_t, valid_t, target_t, relations_t)
    return float(torch.sqrt(torch.mean(
        (predictions - torch.from_numpy(targets)) ** 2
    )).item())


def _train_one(
    kind: str, seed: int, arrays: dict[str, tuple[np.ndarray, ...]],
    output_dir: Path,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _model(kind, arrays["train"][0].shape[-1], arrays["train"][-1].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train = TensorDataset(*[torch.from_numpy(value) for value in arrays["train"]])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=64, shuffle=True, generator=generator)
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    stale = 0
    for epoch in range(1, 121):
        model.train()
        total_squared = 0.0
        total_values = 0
        for nodes, valid, target_indices, relations, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = _forward(
                model, kind, nodes, valid, target_indices, relations
            )
            loss = torch.mean((predictions - targets) ** 2)
            loss.backward()
            optimizer.step()
            total_squared += float(loss.item()) * targets.numel()
            total_values += targets.numel()
        validation = _evaluate(model, kind, arrays["validation"])
        history.append({
            "epoch": epoch,
            "train_rmse": math.sqrt(total_squared / total_values),
            "validation_rmse": validation,
        })
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output_dir / f"{kind}__seed-{seed}.pt"
    torch.save(best_state, checkpoint)
    test_rmse = _evaluate(model, kind, arrays["test"])
    shuffled = (
        _evaluate(model, kind, arrays["test"], shuffle_relations=True)
        if kind == "relation_gnn" else None
    )
    return {
        "model": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_rmse": best_validation,
        "test_rmse": test_rmse,
        "relation_shuffled_test_rmse": shuffled,
        "relation_shuffle_delta": None if shuffled is None else shuffled - test_rmse,
        "wall_time_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "history": history,
    }


def _summaries(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in ("token_mlp", "deepsets", "relation_gnn"):
        group = [row for row in records if row["model"] == kind]
        validation = [float(row["validation_rmse"]) for row in group]
        test = [float(row["test_rmse"]) for row in group]
        deltas = [float(row["relation_shuffle_delta"]) for row in group if row["relation_shuffle_delta"] is not None]
        rows.append({
            "model": kind,
            "seeds": [int(row["seed"]) for row in group],
            "validation_rmse_mean": float(np.mean(validation)),
            "validation_rmse_sd": float(np.std(validation, ddof=1)),
            "test_rmse_mean": float(np.mean(test)),
            "test_rmse_sd": float(np.std(test, ddof=1)),
            "relation_shuffle_delta_mean": None if not deltas else float(np.mean(deltas)),
        })
    return sorted(rows, key=lambda row: float(row["validation_rmse_mean"]))


def _plot(path: Path, summaries: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in summaries]
    means = [float(row["test_rmse_mean"]) for row in summaries]
    errors = [float(row["test_rmse_sd"]) for row in summaries]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, means, yerr=errors, capsize=5, color=["#5B8FF9", "#61DDAA", "#F6BD16"])
    axis.set_ylabel("Normalized masked-state RMSE (lower is better)")
    axis.set_title("R03 engineering interaction screen (3 seeds)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _future_arrays(
    transitions: list[dict[str, Any]], fields: list[str], feature_means: np.ndarray,
    feature_scales: np.ndarray, *, max_aircraft: int,
    centers: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[np.ndarray, ...]], np.ndarray, np.ndarray, dict[str, list[str]]]:
    target_columns = [fields.index(field) for field in FUTURE_TARGET_FIELDS]

    def cv_delta(values: np.ndarray, airport: str) -> np.ndarray:
        absolute = values.copy()
        absolute[fields.index("lat_deg")] += centers[airport][0]
        absolute[fields.index("lon_deg")] += centers[airport][1]
        latitude_delta, longitude_delta = constant_velocity_position_delta(
            fields, absolute.tolist(), horizon_seconds=30.0,
        )
        result = np.zeros(len(target_columns), dtype=np.float64)
        result[0] = latitude_delta
        result[1] = longitude_delta
        return result

    training_deltas: list[np.ndarray] = []
    for transition in transitions:
        if transition["split"] != "train":
            continue
        current = {str(row["natural_key"][1]): row for row in transition["current_rows"]}
        for future in transition["future_rows"]:
            aircraft = str(future["natural_key"][1])
            before = np.array(current[aircraft]["values"], dtype=np.float64)
            after = np.array(future["values"], dtype=np.float64)
            residual = (
                after[target_columns] - before[target_columns]
                - cv_delta(before, str(transition["airport_id"]))
            )
            if np.isfinite(residual).all():
                training_deltas.append(residual)
    delta_values = np.stack(training_deltas)
    delta_means = delta_values.mean(axis=0)
    delta_scales = delta_values.std(axis=0)
    delta_scales[delta_scales < 1e-8] = 1.0

    rows_by_split: dict[str, list[tuple[np.ndarray, ...]]] = {
        "train": [], "validation": [], "test": [],
    }
    sources: dict[str, list[str]] = {key: [] for key in rows_by_split}
    for transition in transitions:
        split = str(transition["split"])
        current_rows = list(transition["current_rows"])
        current_by_aircraft = {
            str(row["natural_key"][1]): (index, row)
            for index, row in enumerate(current_rows)
        }
        raw = np.array([
            [np.nan if value is None else float(value) for value in row["values"]]
            for row in current_rows
        ], dtype=np.float64)
        observed = np.isfinite(raw)
        normalized = (raw - feature_means) / feature_scales
        normalized[~observed] = 0.0
        usable_future = [
            row for row in transition["future_rows"]
            if np.isfinite(np.array(row["values"], dtype=np.float64)[target_columns]).all()
            and np.isfinite(raw[current_by_aircraft[str(row["natural_key"][1])][0], target_columns]).all()
        ]
        if not usable_future:
            continue
        example_weight = np.float32(1.0 / len(usable_future))
        for future in usable_future:
            aircraft = str(future["natural_key"][1])
            target_index, current_row = current_by_aircraft[aircraft]
            target_flags = np.zeros((len(raw), 1), dtype=np.float64)
            target_flags[target_index, 0] = 1.0
            node_features = np.concatenate([
                normalized, (~observed).astype(np.float64), target_flags,
            ], axis=1)
            padded = np.zeros((max_aircraft, node_features.shape[1]), dtype=np.float32)
            padded[:len(raw)] = node_features.astype(np.float32)
            valid = np.zeros(max_aircraft, dtype=bool)
            valid[:len(raw)] = True
            relations = np.zeros((max_aircraft, 3), dtype=np.float32)
            position_delta = normalized[:, :2] - normalized[target_index, :2]
            relations[:len(raw), :2] = position_delta.astype(np.float32)
            relations[:len(raw), 2] = np.sqrt(
                (position_delta ** 2).sum(axis=1)
            ).astype(np.float32)
            before = np.array(current_row["values"], dtype=np.float64)
            after = np.array(future["values"], dtype=np.float64)
            physical_delta = cv_delta(before, str(transition["airport_id"]))
            target_residual = after[target_columns] - before[target_columns] - physical_delta
            target = ((target_residual - delta_means) / delta_scales).astype(np.float32)
            persistence = ((-physical_delta - delta_means) / delta_scales).astype(np.float32)
            constant_velocity = ((np.zeros(len(target_columns)) - delta_means) / delta_scales).astype(np.float32)
            rows_by_split[split].append((
                padded, valid, np.int64(target_index), relations, target,
                example_weight, persistence, constant_velocity,
            ))
            sources[split].append(str(transition["source_file_id"]))
    arrays = {
        split: tuple(np.stack(items) for items in zip(*rows, strict=True))
        for split, rows in rows_by_split.items()
    }
    return arrays, delta_means, delta_scales, sources


def _future_rmse(predictions: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    per_example = np.mean((predictions - targets) ** 2, axis=1)
    return float(np.sqrt(np.sum(per_example * weights) / np.sum(weights)))


@torch.no_grad()
def _future_predictions(
    model: nn.Module, kind: str, arrays: tuple[np.ndarray, ...],
    *, shuffle_relations: bool = False,
) -> np.ndarray:
    nodes, valid, target_indices, relations = arrays[:4]
    relations = relations.copy()
    if shuffle_relations:
        for row in range(len(relations)):
            neighbors = [
                index for index in np.flatnonzero(valid[row])
                if index != int(target_indices[row])
            ]
            if len(neighbors) > 1:
                relations[row, neighbors] = relations[row, list(reversed(neighbors))]
    model.eval()
    return _forward(
        model, kind, torch.from_numpy(nodes), torch.from_numpy(valid),
        torch.from_numpy(target_indices), torch.from_numpy(relations),
    ).numpy()


def _train_future_one(
    kind: str, seed: int, arrays: dict[str, tuple[np.ndarray, ...]], output_dir: Path,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _model(kind, arrays["train"][0].shape[-1], arrays["train"][4].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_tensors = [torch.from_numpy(value) for value in arrays["train"][:6]]
    nodes, valid, target_indices, relations, targets, weights = train_tensors
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = _forward(model, kind, nodes, valid, target_indices, relations)
        per_example = torch.mean((predictions - targets) ** 2, dim=1)
        loss = torch.sum(per_example * weights) / torch.sum(weights)
        loss.backward()
        optimizer.step()
        validation_predictions = _future_predictions(model, kind, arrays["validation"])
        validation = _future_rmse(
            validation_predictions, arrays["validation"][4], arrays["validation"][5]
        )
        history.append({
            "epoch": epoch, "train_rmse": float(torch.sqrt(loss).item()),
            "validation_rmse": validation,
        })
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    if best_state is None:
        raise RuntimeError("future training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output_dir / f"future__{kind}__seed-{seed}.pt"
    torch.save(best_state, checkpoint)
    test_predictions = _future_predictions(model, kind, arrays["test"])
    test_rmse = _future_rmse(test_predictions, arrays["test"][4], arrays["test"][5])
    shuffled = None
    if kind == "relation_gnn":
        shuffled = _future_rmse(
            _future_predictions(model, kind, arrays["test"], shuffle_relations=True),
            arrays["test"][4], arrays["test"][5],
        )
    return {
        "model": kind, "seed": seed, "best_epoch": best_epoch,
        "validation_rmse": best_validation, "test_rmse": test_rmse,
        "relation_shuffled_test_rmse": shuffled,
        "relation_shuffle_delta": None if shuffled is None else shuffled - test_rmse,
        "wall_time_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint.name, "checkpoint_sha256": _sha256(checkpoint),
        "history": history,
    }


def run_future(dataset_path: Path, output_root: Path) -> Path:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    fields = [str(field) for field in dataset["fields"]]
    blocks = list(dataset["blocks"])
    centers = _airport_centers([b for b in blocks if b["split"] == "train"], fields)
    scenes = _center_scenes(build_scene_windows(
        blocks, window_seconds=30, min_aircraft=3, max_aircraft=12,
    ), fields, centers)
    transitions = build_scene_transitions(scenes)
    feature_means, feature_scales = _normalization(scenes, len(fields))
    arrays, delta_means, delta_scales, sources = _future_arrays(
        transitions, fields, feature_means, feature_scales, max_aircraft=12,
        centers=centers,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-future-interaction-screen")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    records = [
        _train_future_one(str(spec["model"]), int(spec["seed"]), arrays, output_dir)
        for spec in interaction_screen_specs()
    ]
    summaries = _summaries(records)
    baselines = []
    for name, index in (("persistence", 6), ("constant_velocity", 7)):
        baselines.append({
            "model": name,
            "validation_rmse": _future_rmse(
                arrays["validation"][index], arrays["validation"][4], arrays["validation"][5]
            ),
            "test_rmse": _future_rmse(
                arrays["test"][index], arrays["test"][4], arrays["test"][5]
            ),
        })
    payload = {
        "schema_version": "r03_future_interaction_screen_v1",
        "decision": engineering_decision(),
        "dataset": str(dataset_path), "dataset_sha256": _sha256(dataset_path),
        "transition_counts": {
            split: sum(item["split"] == split for item in transitions)
            for split in ("train", "validation", "test")
        },
        "example_counts": {split: int(len(arrays[split][0])) for split in arrays},
        "source_counts": {split: len(set(values)) for split, values in sources.items()},
        "target_fields": list(FUTURE_TARGET_FIELDS),
        "delta_means": delta_means.tolist(), "delta_scales": delta_scales.tolist(),
        "baselines": baselines, "summaries": summaries, "records": records,
        "environment": {
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
        },
    }
    _write_json(output_dir / "summary.json", payload)
    table = [
        {"model": row["model"], "validation_rmse_mean": row["validation_rmse"],
         "validation_rmse_sd": 0.0, "test_rmse_mean": row["test_rmse"],
         "test_rmse_sd": 0.0, "relation_shuffle_delta_mean": None}
        for row in baselines
    ] + summaries
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "model", "seeds", "validation_rmse_mean", "validation_rmse_sd",
            "test_rmse_mean", "test_rmse_sd", "relation_shuffle_delta_mean",
        ])
        writer.writeheader(); writer.writerows(table)
    _plot(output_dir / "future_interaction_comparison.png", table)
    return output_dir


def run(dataset_path: Path, output_root: Path) -> Path:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    fields = [str(field) for field in dataset["fields"]]
    blocks = list(dataset["blocks"])
    centers = _airport_centers(
        [block for block in blocks if block["split"] == "train"], fields
    )
    scenes = build_scene_windows(
        blocks, window_seconds=30, min_aircraft=3, max_aircraft=12
    )
    scenes = _center_scenes(scenes, fields, centers)
    means, scales = _normalization(scenes, len(fields))
    arrays = _arrays(scenes, fields, means, scales, max_aircraft=12)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-interaction-screen")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for spec in interaction_screen_specs():
        record = _train_one(str(spec["model"]), int(spec["seed"]), arrays, output_dir)
        record["spec"] = spec
        records.append(record)
        _write_json(output_dir / f"{spec['model']}__seed-{spec['seed']}.json", record)
    summaries = _summaries(records)
    _write_json(output_dir / "summary.json", {
        "schema_version": "r03_interaction_screen_v1",
        "decision": engineering_decision(),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "scene_counts": {
            split: sum(scene["split"] == split for scene in scenes)
            for split in ("train", "validation", "test")
        },
        "example_counts": {split: int(len(arrays[split][0])) for split in arrays},
        "fields": fields,
        "target_fields": list(TARGET_FIELDS),
        "airport_centers_from_training": centers,
        "normalization_means": means.tolist(),
        "normalization_scales": scales.tolist(),
        "summaries": summaries,
        "records": records,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
        },
    })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    _plot(output_dir / "interaction_comparison.png", summaries)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("static", "future"), default="static")
    args = parser.parse_args()
    runner = run_future if args.task == "future" else run
    output_dir = runner(args.dataset.resolve(), args.output_root.resolve())
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
