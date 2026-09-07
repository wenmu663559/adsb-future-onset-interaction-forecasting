"""Small, explicitly non-scientific R03 engineering smoke helpers."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Set
from datetime import datetime, timedelta, timezone
from typing import Any


ENGINEERING_DECISION = "ENGINEERING_SMOKE_ONLY"
_MASK_THRESHOLD_30_PERCENT = (3 * (1 << 64)) // 10
_SCREEN_SEEDS = (1701, 1702, 1703)


def stable_rank(salt: str, identifier: str) -> bytes:
    if not salt or not identifier:
        raise ValueError("salt and identifier must be nonempty")
    return hashlib.sha256(
        salt.encode("utf-8") + b"\x00" + identifier.encode("utf-8")
    ).digest()


def keyed_cell_mask(seed: int, report_id: str, field_name: str) -> bool:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    digest = hashlib.sha256(
        str(seed).encode("ascii")
        + b"\x00"
        + report_id.encode("utf-8")
        + b"\x00"
        + field_name.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") < _MASK_THRESHOLD_30_PERCENT


def utc_microseconds_text(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("UTC microseconds must be an integer")
    timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        microseconds=value
    )
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def feature_view(
    fields: list[str],
    values: list[float | None],
    *,
    airport_id: str,
    mode: str,
    airport_centers: Mapping[str, tuple[float, float]],
) -> tuple[list[str], list[float | None]]:
    if len(fields) != len(values):
        raise ValueError("fields and values must have equal lengths")
    if "lat_deg" not in fields or "lon_deg" not in fields:
        raise ValueError("position fields are required")
    if mode == "absolute":
        return list(fields), list(values)
    if mode == "no_position":
        kept = [index for index, field in enumerate(fields) if field not in {
            "lat_deg", "lon_deg",
        }]
        return [fields[index] for index in kept], [values[index] for index in kept]
    if mode != "airport_centered":
        raise ValueError(f"unknown feature mode: {mode}")
    if airport_id not in airport_centers:
        raise ValueError(f"airport center is unavailable: {airport_id}")
    latitude_index = fields.index("lat_deg")
    longitude_index = fields.index("lon_deg")
    latitude_center, longitude_center = airport_centers[airport_id]
    transformed = list(values)
    if transformed[latitude_index] is not None:
        transformed[latitude_index] = float(transformed[latitude_index]) - latitude_center
    if transformed[longitude_index] is not None:
        transformed[longitude_index] = (
            float(transformed[longitude_index]) - longitude_center
        )
    return list(fields), transformed


def method_screen_specs() -> list[dict[str, object]]:
    methods = (
        ("transformer", "absolute"),
        ("transformer", "no_position"),
        ("transformer", "airport_centered"),
        ("token_mlp", "airport_centered"),
    )
    return [
        {
            "model": model,
            "feature_mode": feature_mode,
            "seed": seed,
            "mask_seed": 1701,
        }
        for model, feature_mode in methods
        for seed in _SCREEN_SEEDS
    ]


def interaction_screen_specs() -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "seed": seed,
            "target_seed": 1701,
            "window_seconds": 30,
            "min_aircraft": 3,
            "max_aircraft": 12,
        }
        for model in ("token_mlp", "deepsets", "relation_gnn")
        for seed in _SCREEN_SEEDS
    ]


def representation_screen_specs() -> list[dict[str, object]]:
    methods = (
        ("transformer", "airport_centered"),
        ("transformer", "airport_centered_track_delta"),
        ("gru", "airport_centered_track_delta"),
        ("token_mlp", "airport_centered_track_delta"),
    )
    return [
        {
            "model": model,
            "feature_mode": feature_mode,
            "seed": seed,
            "mask_seed": 1701,
        }
        for model, feature_mode in methods
        for seed in _SCREEN_SEEDS
    ]


def confirmation_screen_specs() -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "feature_mode": "airport_centered",
            "seed": seed,
            "mask_seed": 1701,
        }
        for model in ("transformer", "token_mlp")
        for seed in _SCREEN_SEEDS
    ]


def temporal_domain_screen_specs() -> list[dict[str, object]]:
    return [
        {
            "model": "transformer",
            "feature_mode": "airport_centered",
            "variant": variant,
            "seed": seed,
            "mask_seed": 1701,
        }
        for variant in ("cell", "span", "span_future", "span_future_domain")
        for seed in _SCREEN_SEEDS
    ]


def mixed_mask_screen_specs() -> list[dict[str, object]]:
    return [
        {
            "model": "transformer",
            "feature_mode": "airport_centered",
            "variant": variant,
            "seed": seed,
            "mask_seed": 1701,
        }
        for variant in ("cell", "span_future", "mixed_future")
        for seed in _SCREEN_SEEDS
    ]


def same_aircraft_future_indices(
    rows: Iterable[Mapping[str, Any]],
) -> list[int | None]:
    materialized = list(rows)
    result: list[int | None] = [None] * len(materialized)
    latest: dict[str, int] = {}
    for index, row in enumerate(materialized):
        natural_key = row.get("natural_key")
        if not isinstance(natural_key, (list, tuple)) or len(natural_key) != 3:
            raise ValueError("row natural key is malformed")
        aircraft = str(natural_key[1])
        if aircraft in latest:
            result[latest[aircraft]] = index
        latest[aircraft] = index
    return result


def track_span_mask_rows(
    rows: Iterable[Mapping[str, Any]], *, seed: int,
) -> list[bool]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    materialized = list(rows)
    by_aircraft: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        natural_key = row.get("natural_key")
        if not isinstance(natural_key, (list, tuple)) or len(natural_key) != 3:
            raise ValueError("row natural key is malformed")
        by_aircraft[str(natural_key[1])].append(index)

    candidates: list[tuple[bytes, list[int]]] = []
    for aircraft, indices in sorted(by_aircraft.items()):
        if len(indices) < 2:
            continue
        spans = [indices[start:start + 4] for start in range(0, len(indices), 4)]
        if len(spans) > 1 and len(spans[-1]) == 1:
            spans[-2].extend(spans.pop())
        for span_number, span in enumerate(spans):
            if len(span) < 2:
                continue
            identifier = f"{seed}\x00{aircraft}\x00{span_number}"
            rank = stable_rank("r03-a3-track-span-v1", identifier)
            candidates.append((rank, span))

    selected = [
        span for rank, span in candidates
        if int.from_bytes(rank[:8], "big") < _MASK_THRESHOLD_30_PERCENT
    ]
    if not selected and candidates:
        selected = [min(candidates, key=lambda item: item[0])[1]]
    result = [False] * len(materialized)
    for span in selected:
        for index in span:
            result[index] = True
    return result


def augment_track_deltas(
    fields: list[str], blocks: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    required = ("lat_deg", "lon_deg", "altitude_m", "speed_mps")
    if any(field not in fields for field in required):
        raise ValueError("track delta fields are unavailable")
    delta_fields = [
        "track_dt_s", "track_dlat_deg", "track_dlon_deg",
        "track_daltitude_m", "track_dspeed_mps",
    ]
    output_blocks: list[dict[str, Any]] = []
    for source_block in blocks:
        block = dict(source_block)
        previous: dict[str, tuple[datetime, list[float | None]]] = {}
        output_rows: list[dict[str, Any]] = []
        for source_row in source_block["rows"]:
            row = dict(source_row)
            aircraft = str(row["natural_key"][1])
            timestamp = datetime.fromisoformat(
                str(row["timestamp"]).replace("Z", "+00:00")
            )
            values = list(row["values"])
            prior = previous.get(aircraft)
            deltas: list[float | None]
            if prior is None:
                deltas = [None] * 5
            else:
                prior_timestamp, prior_values = prior
                elapsed = (timestamp - prior_timestamp).total_seconds()
                if elapsed <= 0:
                    raise ValueError("aircraft timestamps are not increasing")
                deltas = [float(elapsed)]
                for field in required:
                    index = fields.index(field)
                    current_value, prior_value = values[index], prior_values[index]
                    deltas.append(
                        None if current_value is None or prior_value is None
                        else float(current_value) - float(prior_value)
                    )
            previous[aircraft] = (timestamp, values)
            row["values"] = values + deltas
            output_rows.append(row)
        block["rows"] = output_rows
        output_blocks.append(block)
    return list(fields) + delta_fields, output_blocks


def build_scene_windows(
    blocks: Iterable[Mapping[str, Any]],
    *,
    window_seconds: int,
    min_aircraft: int,
    max_aircraft: int,
) -> list[dict[str, Any]]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if min_aircraft < 2 or max_aircraft < min_aircraft:
        raise ValueError("aircraft bounds are invalid")
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = {}
    for block in blocks:
        airport = str(block["airport_id"])
        source = str(block["source_file_id"])
        split = str(block["split"])
        for source_row in block["rows"]:
            row = dict(source_row)
            timestamp = datetime.fromisoformat(
                str(row["timestamp"]).replace("Z", "+00:00")
            )
            bucket = int(timestamp.timestamp()) // window_seconds
            key = (split, airport, source, bucket)
            aircraft = str(row["natural_key"][1])
            existing = grouped.setdefault(key, {}).get(aircraft)
            order = (str(row["timestamp"]), str(row["canonical_report_id"]))
            if existing is None or order > (
                str(existing["timestamp"]), str(existing["canonical_report_id"])
            ):
                grouped[key][aircraft] = row

    scenes: list[dict[str, Any]] = []
    for (split, airport, source, bucket), aircraft_rows in sorted(grouped.items()):
        if len(aircraft_rows) < min_aircraft:
            continue
        scene_id = f"{split}\x00{airport}\x00{source}\x00{bucket}"
        kept_aircraft = sorted(
            aircraft_rows,
            key=lambda aircraft: (stable_rank(scene_id, aircraft), aircraft),
        )[:max_aircraft]
        kept_rows = [aircraft_rows[aircraft] for aircraft in sorted(kept_aircraft)]
        scenes.append({
            "scene_id": scene_id,
            "split": split,
            "airport_id": airport,
            "source_file_id": source,
            "window_start": utc_microseconds_text(
                bucket * window_seconds * 1_000_000
            ),
            "rows": kept_rows,
        })
    return scenes


def build_scene_transitions(
    scenes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for scene in scenes:
        parts = str(scene["scene_id"]).split("\x00")
        if len(parts) != 4:
            raise ValueError("scene id is malformed")
        key = (str(scene["split"]), str(scene["airport_id"]), str(scene["source_file_id"]))
        bucket = int(parts[-1])
        if bucket in grouped[key]:
            raise ValueError("duplicate scene window")
        grouped[key][bucket] = scene
    transitions: list[dict[str, Any]] = []
    for key, sequence in sorted(grouped.items()):
        for bucket, current in sorted(sequence.items()):
            future = sequence.get(bucket + 1)
            if future is None:
                continue
            current_aircraft = {
                str(row["natural_key"][1]) for row in current["rows"]
            }
            future_rows = sorted(
                (
                    dict(row) for row in future["rows"]
                    if str(row["natural_key"][1]) in current_aircraft
                ),
                key=lambda row: str(row["natural_key"][1]),
            )
            if not future_rows:
                continue
            transitions.append({
                "current_scene_id": str(current["scene_id"]),
                "future_scene_id": str(future["scene_id"]),
                "split": key[0],
                "airport_id": key[1],
                "source_file_id": key[2],
                "current_rows": [dict(row) for row in current["rows"]],
                "future_rows": future_rows,
            })
    return transitions


def constant_velocity_position_delta(
    fields: list[str], values: list[float | None], *, horizon_seconds: float,
) -> tuple[float, float]:
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    required = ("lat_deg", "lon_deg", "speed_mps", "sin_heading", "cos_heading")
    if any(field not in fields for field in required):
        raise ValueError("constant-velocity fields are unavailable")
    if any(values[fields.index(field)] is None for field in required):
        raise ValueError("constant-velocity values are unavailable")
    latitude = float(values[fields.index("lat_deg")])
    speed = float(values[fields.index("speed_mps")])
    sin_heading = float(values[fields.index("sin_heading")])
    cos_heading = float(values[fields.index("cos_heading")])
    latitude_delta = speed * cos_heading * horizon_seconds / 111_320.0
    longitude_delta = speed * sin_heading * horizon_seconds / (
        111_320.0 * max(0.1, math.cos(math.radians(latitude)))
    )
    return latitude_delta, longitude_delta


def entity_split_name(identifier: str, *, salt: str) -> str:
    bucket = int.from_bytes(stable_rank(salt, identifier)[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "validation"
    return "test"


def temporal_split_name(
    timestamp: int,
    *,
    train_cutoff: int,
    validation_cutoff: int,
    embargo: int,
) -> str | None:
    if not train_cutoff < validation_cutoff or embargo < 0:
        raise ValueError("temporal split boundaries are invalid")
    if timestamp < train_cutoff - embargo:
        return "train"
    if train_cutoff + embargo < timestamp < validation_cutoff - embargo:
        return "validation"
    if timestamp > validation_cutoff + embargo:
        return "test"
    return None


def summarize_method_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        method_id = f"{record['model']}__{record['feature_mode']}"
        if record.get("variant") is not None:
            method_id += f"__{record['variant']}"
        grouped[method_id].append(record)
    summaries: list[dict[str, object]] = []
    for method_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["seed"]))
        validation = [float(row["best_validation_masked_rmse"]) for row in ordered]
        test = [float(row["test_masked_rmse"]) for row in ordered]
        probe = [float(row["airport_probe_test_accuracy"]) for row in ordered]
        summaries.append({
            "method_id": method_id,
            "model": str(ordered[0]["model"]),
            "feature_mode": str(ordered[0]["feature_mode"]),
            "seeds": [int(row["seed"]) for row in ordered],
            "validation_rmse_mean": statistics.fmean(validation),
            "validation_rmse_sd": statistics.stdev(validation) if len(validation) > 1 else 0.0,
            "test_rmse_mean": statistics.fmean(test),
            "test_rmse_sd": statistics.stdev(test) if len(test) > 1 else 0.0,
            "airport_probe_test_accuracy_mean": statistics.fmean(probe),
            "airport_probe_test_accuracy_sd": (
                statistics.stdev(probe) if len(probe) > 1 else 0.0
            ),
        })
        summary = summaries[-1]
        if all("airport_probe_test_mlp_balanced_accuracy" in row for row in ordered):
            nonlinear_probe = [
                float(row["airport_probe_test_mlp_balanced_accuracy"])
                for row in ordered
            ]
            summary["airport_probe_test_mlp_balanced_accuracy_mean"] = (
                statistics.fmean(nonlinear_probe)
            )
            summary["airport_probe_test_mlp_balanced_accuracy_sd"] = (
                statistics.stdev(nonlinear_probe)
                if len(nonlinear_probe) > 1 else 0.0
            )
        future = [
            float(row["test_future_rmse"]) for row in ordered
            if row.get("test_future_rmse") is not None
        ]
        validation_future = [
            float(row["validation_future_rmse"]) for row in ordered
            if row.get("validation_future_rmse") is not None
        ]
        if validation_future:
            summary["validation_future_rmse_mean"] = statistics.fmean(
                validation_future
            )
            summary["validation_future_rmse_sd"] = (
                statistics.stdev(validation_future)
                if len(validation_future) > 1 else 0.0
            )
        if future:
            summary["test_future_rmse_mean"] = statistics.fmean(future)
            summary["test_future_rmse_sd"] = (
                statistics.stdev(future) if len(future) > 1 else 0.0
            )
        shuffled = [
            float(row["test_shuffled_time_rmse"]) for row in ordered
            if row.get("test_shuffled_time_rmse") is not None
        ]
        if shuffled:
            summary["test_shuffled_time_rmse_mean"] = statistics.fmean(shuffled)
            summary["test_shuffled_time_degradation_mean"] = statistics.fmean([
                shuffled_value / float(row["test_masked_rmse"]) - 1.0
                for shuffled_value, row in zip(shuffled, ordered, strict=True)
            ])
        for metric in (
            "common_cell_validation_rmse", "common_cell_test_rmse",
            "common_span_validation_rmse", "common_span_test_rmse",
        ):
            if all(metric in row for row in ordered):
                values = [float(row[metric]) for row in ordered]
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
    return sorted(
        summaries,
        key=lambda row: (float(row["validation_rmse_mean"]), str(row["method_id"])),
    )


def validate_split_ids(split: Mapping[str, Set[str]]) -> None:
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("split names must be train, validation, and test")
    seen: set[str] = set()
    for name in ("train", "validation", "test"):
        overlap = seen.intersection(split[name])
        if overlap:
            raise ValueError(f"identifier crosses splits: {sorted(overlap)[0]}")
        seen.update(split[name])


def split_sources(
    sources: Iterable[tuple[str, int, str]],
) -> dict[str, set[str]]:
    by_airport: dict[str, list[tuple[int, str]]] = defaultdict(list)
    observed_ids: set[str] = set()
    for airport, sequence, source_id in sources:
        if source_id in observed_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        observed_ids.add(source_id)
        by_airport[airport].append((sequence, source_id))
    result = {"train": set(), "validation": set(), "test": set()}
    for airport, ordered in sorted(by_airport.items()):
        ordered.sort(key=lambda item: (item[0], item[1]))
        count = len(ordered)
        if count < 3:
            raise ValueError(f"airport {airport} has fewer than three sources")
        train_end = max(1, (3 * count) // 5)
        validation_end = max(train_end + 1, (4 * count) // 5)
        validation_end = min(validation_end, count - 1)
        result["train"].update(item[1] for item in ordered[:train_end])
        result["validation"].update(
            item[1] for item in ordered[train_end:validation_end]
        )
        result["test"].update(item[1] for item in ordered[validation_end:])
    validate_split_ids(result)
    return result


def select_ranked_blocks(
    candidates: Iterable[Mapping[str, Any]], *, cap: int, salt: str,
) -> list[dict[str, Any]]:
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("cap must be a positive integer")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        row = dict(candidate)
        required = {"airport_id", "split", "block_id"}
        if not required.issubset(row):
            raise ValueError("block candidate is incomplete")
        grouped[(str(row["airport_id"]), str(row["split"]))].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda row: (
                stable_rank(salt, str(row["block_id"])), str(row["block_id"]),
            ),
        )
        selected.extend(ranked[:cap])
    return selected


def engineering_decision() -> dict[str, object]:
    return {
        "decision": ENGINEERING_DECISION,
        "gate_a_verified": False,
        "scientific_claim_authorized": False,
        "model_reuse_authorized": False,
        "r04_authorized": False,
    }
