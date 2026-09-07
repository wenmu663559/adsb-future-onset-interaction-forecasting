from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

from airspace_complexity.r03_experiment_contract import (
    finite_or_none,
    summarize_temporal_controls,
)

DECISION = "EXPLORATORY_ONLINE_COMPLEXITY_PILOT"
EXPERIMENT_MODELS = (
    "deepsets", "masked_temporal", "invariant_relation",
    "calibrated_invariant_relation", "calibrated_relation_no_adversary",
    "calibrated_invariant_no_relation", "multitask_invariant_relation",
    "airport_conditioned_multitask",
    "masked_relation_ssl",
)
CONTROL_FEATURE_NAMES = (
    "current_aircraft_count", "current_altitude_mean_m", "current_altitude_std_m",
    "current_speed_mean_mps", "current_speed_std_mps", "current_altitude_range_m",
    "current_min_pair_distance_m", "current_low_altitude_fraction",
    "history_aircraft_count_mean", "history_aircraft_count_std",
    "history_aircraft_count_trend_per_step", "history_min_pair_distance_mean_m",
    "history_min_pair_distance_trend_m_per_step", "history_altitude_std_mean_m",
    "history_speed_std_mean_mps", "history_active_step_fraction",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_record(scene: object, split: str) -> dict[str, object]:
    return {
        "scene_id": scene.scene_id,
        "split": split,
        "airport_id": scene.airport_id,
        "source_file_ids": list(scene.source_file_ids),
        "source_day": scene.source_day,
        "cutoff_us": scene.cutoff_us,
        "history": [
            [asdict(report) for report in step]
            for step in scene.history
        ],
        "targets": {
            str(horizon): asdict(target)
            for horizon, target in sorted(scene.targets.items())
        },
    }


def _validated_frozen_splits(
    observed_groups: set[str], frozen_splits: dict[str, str],
) -> dict[str, str]:
    """Require a predeclared split for every and only every observed airport-day."""

    if set(frozen_splits) != observed_groups:
        missing = sorted(observed_groups - set(frozen_splits))
        extra = sorted(set(frozen_splits) - observed_groups)
        raise ValueError(
            "frozen splits must exactly cover observed airport-days; "
            f"missing={missing}; extra={extra}"
        )
    invalid = sorted(
        set(frozen_splits.values()) - {"train", "validation", "test"}
    )
    if invalid:
        raise ValueError(f"invalid frozen split values: {invalid}")
    return dict(sorted(frozen_splits.items()))


def _load_frozen_source_config(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "exact_source_ids" in config:
        source_ids = tuple(config["exact_source_ids"])
        splits = dict(config["source_day_splits"])
    else:
        sources = list(config["sources"])
        source_ids = tuple(str(item["source_file_id"]) for item in sources)
        airport = str(config.get("airport_id") or "KBTP")
        splits = {
            f"{airport}\x00{item['date']}": "test" for item in sources
        }
        splits.update({
            f"{airport}\x00{source_day}": "test"
            for source_day in config.get("derived_utc_dates", [])
        })
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("frozen source config must contain unique source IDs")
    return source_ids, splits


def prepare_pilot(
    project_root: Path,
    output_root: Path,
    *,
    sources_per_airport: int,
    max_hours_per_source: int,
    exact_source_ids: tuple[str, ...] | None = None,
    frozen_source_day_splits: dict[str, str] | None = None,
) -> Path:
    from airspace_complexity.r02_runner import _load_airport_roots
    from airspace_complexity.r03_online_complexity import (
        TerminalVolume,
        assign_airport_day_splits,
        build_scene_examples,
        extract_contiguous_reports,
        filter_terminal_reports,
    )
    from airspace_complexity.r03_runner import R03SourceRegistry

    project = Path(project_root).resolve()
    registry = R03SourceRegistry.from_lineage(project)
    extraction = extract_contiguous_reports(
        registry,
        _load_airport_roots(project),
        project,
        sources_per_airport=sources_per_airport,
        max_hours_per_source=max_hours_per_source,
        exact_source_ids=exact_source_ids,
    )
    terminal_volumes = {
        "KAGC": TerminalVolume(
            latitude_deg=40.3544376, longitude_deg=-79.9290467,
            elevation_m=381.5, radius_m=18_520.0,
            floor_offset_m=-150.0, ceiling_offset_m=1_524.0,
        ),
        "KBTP": TerminalVolume(
            latitude_deg=40.7767, longitude_deg=-79.9512,
            elevation_m=380.4, radius_m=18_520.0,
            floor_offset_m=-150.0, ceiling_offset_m=1_524.0,
        ),
    }
    filtered_reports, terminal_filter_audit = filter_terminal_reports(
        extraction.reports, terminal_volumes
    )
    examples = build_scene_examples(
        filtered_reports,
        history_seconds=120,
        step_seconds=10,
        horizons_seconds=(30, 120, 300),
        stale_seconds=30,
        min_aircraft=3,
        max_aircraft=100,
    )
    group_ids = sorted({
        f"{scene.airport_id}\x00{scene.source_day}"
        for scene in examples
    })
    splits = (
        _validated_frozen_splits(set(group_ids), frozen_source_day_splits)
        if frozen_source_day_splits is not None
        else assign_airport_day_splits(group_ids) if group_ids else {}
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-online-complexity-pilot")
    run_root = Path(output_root).resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    report_path = run_root / "pilot_reports.jsonl"
    with report_path.open("wb") as handle:
        for report in filtered_reports:
            handle.write(_json_bytes(asdict(report)))
    scene_path = run_root / "pilot_scenes.jsonl"
    split_counts = {name: 0 for name in ("train", "validation", "test")}
    airport_scene_counts: dict[str, int] = {}
    with scene_path.open("wb") as handle:
        for scene in examples:
            group_id = f"{scene.airport_id}\x00{scene.source_day}"
            split = splits[group_id]
            split_counts[split] += 1
            airport_scene_counts[scene.airport_id] = (
                airport_scene_counts.get(scene.airport_id, 0) + 1
            )
            handle.write(_json_bytes(_scene_record(scene, split)))

    manifest = dict(extraction.manifest)
    manifest.update({
        "schema_version": "r03_online_complexity_pilot_dataset_v2",
        "decision": DECISION,
        "run_id": run_id,
        "history_seconds": 120,
        "step_seconds": 10,
        "horizons_seconds": [30, 120, 300],
        "stale_seconds": 30,
        "terminal_volume": {
            airport: asdict(volume)
            for airport, volume in sorted(terminal_volumes.items())
        },
        "terminal_filter_audit": terminal_filter_audit,
        "unfiltered_report_count": len(extraction.reports),
        "report_count": len(filtered_reports),
        "proximity_proxy": {
            "sampling_seconds": 10,
            "horizontal_threshold_m": 9_260.0,
            "vertical_threshold_m": 304.8,
            "aggregation": "consecutive_pair_samples_merged_into_episodes",
            "primary_target": "future_onset_proximity_event_count",
            "future_onset_rule": "pairs active at cutoff are excluded until they clear and re-enter",
            "legacy_secondary_target": "proximity_event_count",
        },
        "scene_count": len(examples),
        "scene_counts_by_airport": airport_scene_counts,
        "scene_counts_by_split": split_counts,
        "source_day_splits": splits,
        "artifacts": {
            "pilot_reports.jsonl": _sha256(report_path),
            "pilot_scenes.jsonl": _sha256(scene_path),
        },
    })
    manifest_path = run_root / "pilot_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    hashes = {
        name: _sha256(run_root / name)
        for name in ("pilot_reports.jsonl", "pilot_scenes.jsonl", "pilot_manifest.json")
    }
    (run_root / "artifact_hashes.json").write_bytes(_json_bytes(hashes))
    print(json.dumps({
        "decision": DECISION,
        "run_root": str(run_root),
        "report_count": len(filtered_reports),
        "scene_count": len(examples),
        "scene_counts_by_split": split_counts,
    }, sort_keys=True))
    return run_root


def _load_experiment_arrays(
    dataset_root: Path,
    max_aircraft: int = 32,
    *,
    frozen_normalization=None,
):
    """Load the frozen JSONL scenes into causal padded tensors."""
    import numpy as np

    centers = {
        "KAGC": (40.3544376, -79.9290467),
        "KBTP": (40.7765833333, -79.9510833333),
    }
    rows = [json.loads(line) for line in (dataset_root / "pilot_scenes.jsonl").open(
        encoding="utf-8"
    )]
    if not rows:
        raise ValueError("pilot dataset contains no scenes")
    steps = max(len(row["history"]) for row in rows)
    x = np.zeros((len(rows), max_aircraft, steps, 7), dtype=np.float32)
    targets = np.zeros((len(rows), 3), dtype=np.float32)
    controls = np.zeros((len(rows), len(CONTROL_FEATURE_NAMES)), dtype=np.float32)
    split = []
    airport = []
    for i, row in enumerate(rows):
        split.append(row["split"])
        airport.append(row["airport_id"])
        latest_by_aircraft = {}
        for step_index, reports in enumerate(row["history"]):
            for report in reports:
                latest_by_aircraft[report["aircraft_id"]] = (step_index, report)
        ordered = sorted(
            latest_by_aircraft,
            key=lambda aircraft_id: (
                -latest_by_aircraft[aircraft_id][0], aircraft_id
            ),
        )[:max_aircraft]
        node = {aircraft_id: j for j, aircraft_id in enumerate(ordered)}
        center_lat, center_lon = centers[row["airport_id"]]
        cos_lat = np.cos(np.deg2rad(center_lat))
        for step_index, reports in enumerate(row["history"]):
            offset = steps - len(row["history"]) + step_index
            for report in reports:
                j = node.get(report["aircraft_id"])
                if j is None:
                    continue
                x[i, j, offset] = (
                    (report["lat_deg"] - center_lat) * 111_320.0,
                    (report["lon_deg"] - center_lon) * 111_320.0 * cos_lat,
                    report["altitude_m"], report["speed_mps"],
                    np.sin(report["heading_rad"]), np.cos(report["heading_rad"]), 1.0,
                )
        current = x[i, :, -1]
        present = current[:, 6] > 0
        values = current[present]
        count = float(len(values))
        if len(values):
            xy = values[:, :2]
            if len(xy) > 1:
                distance = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=2))
                distance[distance == 0] = np.inf
                min_distance = float(distance.min())
            else:
                min_distance = 100_000.0
            controls[i, :8] = (
                count, values[:, 2].mean(), values[:, 2].std(),
                values[:, 3].mean(), values[:, 3].std(),
                np.ptp(values[:, 2]), min_distance,
                float((values[:, 2] < 3_000).mean()),
            )
        history_counts = []
        history_min_distances = []
        history_altitude_stds = []
        history_speed_stds = []
        for step_index in range(steps):
            step_values = x[i, :, step_index]
            step_present = step_values[:, 6] > 0
            observed_values = step_values[step_present]
            history_counts.append(float(len(observed_values)))
            if len(observed_values) > 1:
                step_xy = observed_values[:, :2]
                step_distance = np.sqrt(
                    ((step_xy[:, None] - step_xy[None, :]) ** 2).sum(axis=2)
                )
                step_distance[step_distance == 0] = np.inf
                history_min_distances.append(float(step_distance.min()))
            else:
                history_min_distances.append(100_000.0)
            history_altitude_stds.append(
                float(observed_values[:, 2].std()) if len(observed_values) else 0.0
            )
            history_speed_stds.append(
                float(observed_values[:, 3].std()) if len(observed_values) else 0.0
            )
        controls[i, 8:] = summarize_temporal_controls(
            counts=history_counts,
            min_distances_m=history_min_distances,
            altitude_stds_m=history_altitude_stds,
            speed_stds_mps=history_speed_stds,
        )
        targets[i] = tuple(
            np.log1p(row["targets"][str(horizon)]["proximity_event_count"])
            for horizon in (30, 120, 300)
        )
    split = np.asarray(split)
    if frozen_normalization is None and not (split == "test").any():
        raise ValueError("dataset must contain a nonempty test split")
    from airspace_complexity.r03_online_complexity import normalize_scene_tensor

    x, mean, scale = normalize_scene_tensor(
        x, split, frozen_normalization=frozen_normalization
    )
    return x, targets, controls, split, np.asarray(airport), mean, scale


def _metrics(y_true, y_pred) -> dict[str, float | bool | None]:
    import numpy as np

    error = y_pred - y_true
    denominator = float(((y_true - y_true.mean()) ** 2).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        original_true = np.expm1(y_true)
        original_pred = np.maximum(0.0, np.expm1(y_pred))
        count_mae = float(np.abs(original_pred - original_true).mean())
        count_rmse = float(np.sqrt(((original_pred - original_true) ** 2).mean()))
    return {
        "log_mae": float(np.abs(error).mean()),
        "log_rmse": float(np.sqrt((error ** 2).mean())),
        "log_r2": float(1.0 - float((error ** 2).sum()) / denominator) if denominator else 0.0,
        "count_mae": finite_or_none(count_mae),
        "count_rmse": finite_or_none(count_rmse),
        "original_scale_finite": bool(
            finite_or_none(count_mae) is not None and finite_or_none(count_rmse) is not None
        ),
    }


def run_experiment(
    dataset_root: Path, *, epochs: int, seeds: tuple[int, ...],
    models: tuple[str, ...] | None = None,
    result_name: str = "experiment_results.json",
) -> Path:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    from airspace_complexity.r03_frozen_probe import (
        InvariantRelation as SharedInvariantRelation,
        reverse_gradient,
    )

    selected_models = tuple(models or EXPERIMENT_MODELS)
    unknown = sorted(set(selected_models) - set(EXPERIMENT_MODELS))
    if unknown or not selected_models:
        raise ValueError(f"invalid experiment models: {unknown}")
    if Path(result_name).name != result_name or not result_name.endswith(".json"):
        raise ValueError("result_name must be a JSON basename")
    x, targets, controls, split, airports, mean, scale = _load_experiment_arrays(dataset_root)
    y = targets[:, 1]
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    evaluation = validation if len(validation) else test
    results: list[dict[str, object]] = []
    airport_results: list[dict[str, object]] = []
    multitask_results = []

    airport_mean_prediction = np.zeros_like(targets)
    for airport_name in sorted(set(airports.tolist())):
        airport_train = train[airports[train] == airport_name]
        airport_rows = np.flatnonzero(airports == airport_name)
        airport_mean_prediction[airport_rows] = targets[airport_train].mean(axis=0)
    for subset_name, indices in (("validation", evaluation), ("test", test)):
        results.append({
            "model": "airport_mean_baseline", "seed": 1701, "subset": subset_name,
            **_metrics(y[indices], airport_mean_prediction[indices, 1]),
        })
        for airport_name in sorted(set(airports[indices].tolist())):
            airport_indices = indices[airports[indices] == airport_name]
            airport_results.append({
                "model": "airport_mean_baseline", "seed": 1701,
                "subset": subset_name, "airport": airport_name,
                "horizon_seconds": 120,
                **_metrics(
                    y[airport_indices], airport_mean_prediction[airport_indices, 1]
                ),
            })
        for horizon_index, horizon in enumerate((30, 120, 300)):
            multitask_results.append({
                "model": "airport_mean_baseline", "seed": 1701,
                "subset": subset_name, "horizon_seconds": horizon,
                **_metrics(
                    targets[indices, horizon_index],
                    airport_mean_prediction[indices, horizon_index],
                ),
            })
    current_controls = controls[:, :8]
    for name, estimator in (
        ("ridge_controls", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("hist_gradient_boosting_controls", HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.05, random_state=1701
        )),
    ):
        estimator.fit(current_controls[train], y[train])
        for subset_name, indices in (("validation", evaluation), ("test", test)):
            subset_prediction = estimator.predict(current_controls[indices])
            results.append({"model": name, "seed": 1701, "subset": subset_name,
                            **_metrics(y[indices], subset_prediction)})
            for airport_name in sorted(set(airports[indices].tolist())):
                airport_mask = airports[indices] == airport_name
                airport_results.append({
                    "model": name, "seed": 1701, "subset": subset_name,
                    "airport": airport_name, "horizon_seconds": 120,
                    **_metrics(y[indices][airport_mask], subset_prediction[airport_mask]),
                })

    site = (airports == "KBTP").astype(np.float32)[:, None]

    def evaluate_personalized_ridge(model_name, base_controls):
        personalized_controls = np.concatenate(
            (base_controls, site, base_controls * site), axis=1
        )
        best_alpha, best_validation_loss = None, float("inf")
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(personalized_controls[train], targets[train])
            candidate_prediction = candidate.predict(personalized_controls[evaluation])
            candidate_loss = float(
                ((candidate_prediction - targets[evaluation]) ** 2).mean()
            )
            if candidate_loss < best_validation_loss:
                best_alpha, best_validation_loss = alpha, candidate_loss
        personalized_ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        personalized_ridge.fit(personalized_controls[train], targets[train])
        personalized_prediction = personalized_ridge.predict(personalized_controls)
        for subset_name, indices in (("validation", evaluation), ("test", test)):
            results.append({
                "model": model_name, "seed": 1701,
                "subset": subset_name, "selected_alpha": best_alpha,
                **_metrics(y[indices], personalized_prediction[indices, 1]),
            })
            for airport_name in sorted(set(airports[indices].tolist())):
                airport_indices = indices[airports[indices] == airport_name]
                airport_results.append({
                    "model": model_name, "seed": 1701,
                    "subset": subset_name, "airport": airport_name,
                    "horizon_seconds": 120, "selected_alpha": best_alpha,
                    **_metrics(
                        y[airport_indices], personalized_prediction[airport_indices, 1]
                    ),
                })
            for horizon_index, horizon in enumerate((30, 120, 300)):
                multitask_results.append({
                    "model": model_name, "seed": 1701,
                    "subset": subset_name, "horizon_seconds": horizon,
                    "selected_alpha": best_alpha,
                    **_metrics(
                        targets[indices, horizon_index],
                        personalized_prediction[indices, horizon_index],
                    ),
                })

    evaluate_personalized_ridge(
        "airport_calibrated_ridge_controls", current_controls
    )
    evaluate_personalized_ridge(
        "airport_calibrated_temporal_ridge_controls", controls
    )

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, value):
            current = value[:, :, -1]
            mask = current[:, :, 6:7]
            representation = (self.phi(current[:, :, :6]) * mask).sum(1) / mask.sum(1).clamp_min(1)
            return self.head(representation).squeeze(-1), representation

    class MaskedTemporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(7, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, value):
            batch, nodes, steps, features = value.shape
            sequence = value.reshape(batch * nodes, steps, features)
            _, hidden = self.gru(sequence)
            node_rep = hidden[-1].reshape(batch, nodes, 32)
            present = (value[:, :, :, 6].sum(2) > 0).float().unsqueeze(-1)
            representation = (node_rep * present).sum(1) / present.sum(1).clamp_min(1)
            return self.head(representation).squeeze(-1), representation

    class InvariantRelation(SharedInvariantRelation):
        def __init__(self):
            super().__init__(mean, scale)

    class CalibratedInvariantRelation(InvariantRelation):
        """Keep the core representation invariant; calibrate only the proxy head."""
        def __init__(self):
            super().__init__()
            self.site_bias = nn.Embedding(2, 1)
            nn.init.zeros_(self.site_bias.weight)

        def forward(self, value, airport_index):
            prediction, representation = super().forward(value)
            return prediction + self.site_bias(airport_index).squeeze(-1), representation

    class CalibratedInvariantNoRelation(nn.Module):
        """Scene-centred set encoder used to ablate explicit pair interactions."""
        def __init__(self):
            super().__init__()
            self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("feature_scale", torch.tensor(scale, dtype=torch.float32))
            self.node = nn.Sequential(nn.Linear(4, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(65, 48), nn.ReLU(), nn.Linear(48, 1))
            self.domain_head = nn.Sequential(nn.Linear(65, 24), nn.ReLU(), nn.Linear(24, 2))
            self.site_bias = nn.Embedding(2, 1)
            nn.init.zeros_(self.site_bias.weight)

        def forward(self, value, airport_index):
            raw = value[:, :, -1, :6] * self.feature_scale + self.feature_mean
            mask = (value[:, :, -1, 6] > 0).float().unsqueeze(-1)
            denominator = mask.sum(1, keepdim=True).clamp_min(1)
            centre = (raw[:, :, :4] * mask).sum(1, keepdim=True) / denominator
            centred = (raw[:, :, :4] - centre) / raw.new_tensor((10_000, 10_000, 3_000, 100))
            encoded = self.node(centred)
            mean_node = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1)
            max_node = encoded.masked_fill(mask == 0, -1e9).amax(1)
            count = torch.log1p(mask.sum(1)) / 5.0
            representation = torch.cat((mean_node, max_node, count), dim=-1)
            prediction = self.head(representation).squeeze(-1)
            return prediction + self.site_bias(airport_index).squeeze(-1), representation

        def domain_logits(self, representation):
            return self.domain_head(reverse_gradient(representation))

    class MultitaskInvariantRelation(InvariantRelation):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(nn.Linear(65, 48), nn.ReLU(), nn.Linear(48, 3))

    class AirportConditionedMultitask(InvariantRelation):
        """Shared relation encoder with a small zero-initialized airport adapter."""
        def __init__(self):
            super().__init__()
            self.site_adapter = nn.Embedding(2, 130)
            nn.init.zeros_(self.site_adapter.weight)
            self.head = nn.Sequential(nn.Linear(65, 48), nn.ReLU(), nn.Linear(48, 3))

        def forward(self, value, airport_index):
            _, shared = super().forward(value)
            scale_shift = self.site_adapter(airport_index)
            site_scale, site_shift = scale_shift.chunk(2, dim=-1)
            adapted = shared * (1.0 + 0.1 * torch.tanh(site_scale)) + 0.1 * site_shift
            return self.head(adapted), shared

    class MaskedRelationSSL(InvariantRelation):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 6))

        def forward(self, value, mask_probability=0.0, deterministic=False):
            pair, valid, node_mask = self.pair_inputs(value)
            if mask_probability:
                if deterministic:
                    nodes = pair.shape[1]
                    row = torch.arange(nodes, device=value.device)[:, None]
                    column = torch.arange(nodes, device=value.device)[None, :]
                    selected = ((row * nodes + column) % 10 < 3)[None]
                    selected = selected.expand(pair.shape[0], -1, -1) & valid
                else:
                    selected = (torch.rand(valid.shape, device=value.device) < mask_probability) & valid
            else:
                selected = torch.zeros_like(valid)
            masked_pair = pair.masked_fill(selected.unsqueeze(-1), 0.0)
            encoded = self.edge(masked_pair)
            reconstruction = self.decoder(encoded)
            representation = self.aggregate_edges(encoded, valid, node_mask)
            return reconstruction, pair, selected, representation

    device = torch.device("cpu")
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    airport_target = torch.from_numpy((airports == "KBTP").astype(np.int64)).to(device)
    batch_size = 64
    saved_representations = {}
    latency = []
    model_specs = (
        ("deepsets", DeepSets, 0.0, False),
        ("masked_temporal", MaskedTemporal, 0.0, False),
        ("invariant_relation", InvariantRelation, 0.1, False),
        ("calibrated_invariant_relation", CalibratedInvariantRelation, 0.1, True),
        ("calibrated_relation_no_adversary", CalibratedInvariantRelation, 0.0, True),
        ("calibrated_invariant_no_relation", CalibratedInvariantNoRelation, 0.1, True),
        ("multitask_invariant_relation", MultitaskInvariantRelation, 0.0, False),
        ("airport_conditioned_multitask", AirportConditionedMultitask, 0.0, True),
        ("masked_relation_ssl", MaskedRelationSSL, 0.0, False),
    )
    model_specs = tuple(spec for spec in model_specs if spec[0] in selected_models)
    multitask_models = {"multitask_invariant_relation", "airport_conditioned_multitask"}
    ssl_probe_results = []
    for model_name, model_class, adversarial_weight, calibrated in model_specs:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_class().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best_state, best_score = None, float("inf")
            rng = np.random.default_rng(seed)
            for _ in range(epochs):
                model.train()
                order = rng.permutation(train)
                for start in range(0, len(order), batch_size):
                    indices = torch.from_numpy(order[start:start + batch_size]).long()
                    batch = xt[indices].clone()
                    if model_name == "masked_temporal":
                        observed = batch[:, :, :, 6] > 0
                        dropped = (torch.rand(observed.shape) < 0.20) & observed
                        batch[dropped] = 0.0
                    if model_name == "masked_relation_ssl":
                        reconstruction, pair_target, selected, representation = model(
                            batch, mask_probability=0.30
                        )
                        if not selected.any():
                            continue
                        loss = nn.functional.mse_loss(
                            reconstruction[selected], pair_target[selected]
                        )
                        prediction = None
                    elif calibrated:
                        prediction, representation = model(batch, airport_target[indices])
                    else:
                        prediction, representation = model(batch)
                    if model_name == "masked_relation_ssl":
                        pass
                    elif model_name in multitask_models:
                        loss = nn.functional.mse_loss(prediction, target_tensor[indices])
                    else:
                        loss = nn.functional.mse_loss(prediction, yt[indices])
                    if adversarial_weight:
                        loss = loss + adversarial_weight * nn.functional.cross_entropy(
                            model.domain_logits(representation), airport_target[indices]
                        )
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    if model_name == "masked_relation_ssl":
                        reconstructed, pair_target, selected, _ = model(
                            xt[evaluation], mask_probability=0.30, deterministic=True
                        )
                        score = nn.functional.mse_loss(
                            reconstructed[selected], pair_target[selected]
                        ).item()
                        evaluated = None
                    elif calibrated:
                        evaluated = model(xt[evaluation], airport_target[evaluation])[0]
                    else:
                        evaluated = model(xt[evaluation])[0]
                    if model_name == "masked_relation_ssl":
                        pass
                    elif model_name in multitask_models:
                        score = nn.functional.mse_loss(
                            evaluated, target_tensor[evaluation]
                        ).item()
                    else:
                        score = nn.functional.mse_loss(evaluated, yt[evaluation]).item()
                if score < best_score:
                    best_score = score
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(best_state)
            torch.save({
                "model": model_name, "seed": seed, "state_dict": best_state,
                "input_features": 7, "hidden_width": 32,
                "airport_adversarial_weight": adversarial_weight,
                "uses_site_calibration": calibrated,
                "target": (
                    "masked_relative_pair_reconstruction"
                    if model_name == "masked_relation_ssl" else
                    "log1p(30_120_300_second_proximity_event_counts)"
                    if model_name in multitask_models
                    else "log1p(120_second_proximity_event_count)"
                ),
            }, dataset_root / f"{model_name}__seed-{seed}.pt")
            model.eval()
            with torch.no_grad():
                if model_name == "masked_relation_ssl":
                    _, _, _, representation = model(xt)
                    prediction = None
                elif calibrated:
                    prediction, representation = model(xt, airport_target)
                else:
                    prediction, representation = model(xt)
            representation = representation.numpy()
            if model_name == "masked_relation_ssl":
                prediction_by_horizon = []
                for horizon_index, horizon in enumerate((30, 120, 300)):
                    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                        representation[train], targets[train, horizon_index]
                    )
                    horizon_prediction = probe.predict(representation)
                    prediction_by_horizon.append(horizon_prediction)
                    for subset_name, indices in (("validation", evaluation), ("test", test)):
                        ssl_probe_results.append({
                            "model": model_name, "seed": seed, "subset": subset_name,
                            "horizon_seconds": horizon,
                            **_metrics(targets[indices, horizon_index], horizon_prediction[indices]),
                        })
                main_prediction = prediction_by_horizon[1]
            else:
                prediction = prediction.numpy()
                main_prediction = (
                    prediction[:, 1] if model_name in multitask_models
                    else prediction
                )
            saved_representations[(model_name, seed)] = representation
            for subset_name, indices in (("validation", evaluation), ("test", test)):
                results.append({"model": model_name, "seed": seed, "subset": subset_name,
                                **_metrics(y[indices], main_prediction[indices])})
                for airport_name in sorted(set(airports[indices].tolist())):
                    airport_indices = indices[airports[indices] == airport_name]
                    airport_results.append({
                        "model": model_name, "seed": seed, "subset": subset_name,
                        "airport": airport_name, "horizon_seconds": 120,
                        **_metrics(y[airport_indices], main_prediction[airport_indices]),
                    })
                if model_name in multitask_models:
                    for horizon_index, horizon in enumerate((30, 120, 300)):
                        multitask_results.append({
                            "model": model_name, "seed": seed, "subset": subset_name,
                            "horizon_seconds": horizon,
                            **_metrics(
                                targets[indices, horizon_index],
                                prediction[indices, horizon_index],
                            ),
                        })
            if seed == seeds[0]:
                for aircraft_count in (20, 50, 100):
                    synthetic = torch.zeros((1, aircraft_count, x.shape[2], 7))
                    synthetic[:, :, :, 6] = 1.0
                    synthetic_airport = torch.zeros(1, dtype=torch.long)
                    for _ in range(10):
                        if model_name == "masked_relation_ssl": model(synthetic)
                        elif calibrated: model(synthetic, synthetic_airport)
                        else: model(synthetic)
                    timings = []
                    with torch.no_grad():
                        for _ in range(100):
                            started = time.perf_counter()
                            if model_name == "masked_relation_ssl": model(synthetic)
                            elif calibrated: model(synthetic, synthetic_airport)
                            else: model(synthetic)
                            timings.append(
                                (time.perf_counter() - started) * 1000
                            )
                    latency.append({"model": model_name, "aircraft_count": aircraft_count,
                                    "p50_ms": float(np.percentile(timings, 50)),
                                    "p95_ms": float(np.percentile(timings, 95))})

    # A frozen-representation airport probe quantifies location leakage.
    from sklearn.linear_model import LogisticRegression
    airport_probe = []
    labels = (airports == "KBTP").astype(int)
    for (model_name, seed), representation in saved_representations.items():
        if len(np.unique(labels[train])) == 2 and len(test):
            probe = LogisticRegression(max_iter=500).fit(representation[train], labels[train])
            airport_probe.append({"model": model_name, "seed": seed,
                                  "test_accuracy": float(probe.score(representation[test], labels[test]))})
    cross_airport = []
    for source, destination in (("KAGC", "KBTP"), ("KBTP", "KAGC")):
        source_train = train[airports[train] == source]
        destination_test = test[airports[test] == destination]
        if len(source_train) and len(destination_test):
            transfer = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                current_controls[source_train], y[source_train]
            )
            cross_airport.append({"model": "ridge_controls", "train_airport": source,
                                  "test_airport": destination,
                                  **_metrics(y[destination_test], transfer.predict(current_controls[destination_test]))})
    payload = {
        "schema_version": "r03_online_complexity_experiment_v1",
        "dataset_root": str(dataset_root.resolve()),
        "target": "log1p(120_second_proximity_event_count)",
        "epochs": epochs, "seeds": list(seeds),
        "counts": {name: int((split == name).sum()) for name in ("train", "validation", "test")},
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist(), "fit_on": "train_only"},
        "control_features": list(CONTROL_FEATURE_NAMES),
        "dataset_sha256": _sha256(dataset_root / "pilot_scenes.jsonl"),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__},
        "selected_models": list(selected_models),
        "results": results, "multitask_results": multitask_results,
        "airport_results": airport_results,
        "ssl_probe_results": ssl_probe_results,
        "latency": latency, "airport_probe": airport_probe,
        "cross_airport_transfer": cross_airport,
        "limitations": [
            "Proximity events are an operational proxy, not controller-rated complexity.",
            "Masked temporal training uses causal observation dropout; it is not future-data imputation.",
            "This bounded pilot is for method selection, not a final generalization claim.",
        ],
    }
    output = dataset_root / result_name
    output.write_bytes(_json_bytes(payload))
    print(json.dumps({"experiment_results": str(output), "counts": payload["counts"]}, sort_keys=True))
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or run the bounded R03 online-complexity pilot."
    )
    parser.add_argument("--project-root", type=Path, default=CHECKOUT_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=CHECKOUT_ROOT / "outputs" / "experiments" / "r03_online_pilot",
    )
    parser.add_argument("--sources-per-airport", type=int, default=4)
    parser.add_argument("--max-hours-per-source", type=int, default=2)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1701, 1702, 1703))
    parser.add_argument("--models", nargs="+", choices=EXPERIMENT_MODELS)
    parser.add_argument("--result-name", default="experiment_results.json")
    parser.add_argument(
        "--source-config", type=Path,
        help="Frozen JSON containing exact_source_ids and source_day_splits.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run-experiment", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.run_experiment:
        if args.dataset_root is None:
            raise SystemExit("--dataset-root is required with --run-experiment")
        run_experiment(
            args.dataset_root, epochs=args.epochs, seeds=tuple(args.seeds),
            models=tuple(args.models) if args.models else None,
            result_name=args.result_name,
        )
        return 0
    exact_source_ids = None
    frozen_splits = None
    if args.source_config is not None:
        exact_source_ids, frozen_splits = _load_frozen_source_config(
            args.source_config
        )
    prepare_pilot(
        args.project_root,
        args.output_root,
        sources_per_airport=args.sources_per_airport,
        max_hours_per_source=args.max_hours_per_source,
        exact_source_ids=exact_source_ids,
        frozen_source_day_splits=frozen_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
