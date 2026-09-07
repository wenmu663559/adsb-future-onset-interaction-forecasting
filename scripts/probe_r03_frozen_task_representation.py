"""Probe existing frozen relation checkpoints against task-demand indicators."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from airspace_complexity.r03_frozen_probe import (
    fit_frozen_representation_probes,
    load_invariant_relation_checkpoint,
)
from airspace_complexity.r03_task_indicators import (
    TASK_INDICATOR_NAMES,
    extract_task_indicators,
)
from run_r03_online_complexity_pilot import _load_experiment_arrays


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


def _git_version() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _runtime_environment() -> dict[str, str | None]:
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "operating_system": platform.platform(),
        "code_version": _git_version(),
    }


def _load_indicators(scene_path: Path):
    indicators = {name: [] for name in TASK_INDICATOR_NAMES}
    days = []
    with scene_path.open(encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            airport = str(scene["airport_id"])
            if airport not in AIRPORT_CENTERS:
                raise ValueError(f"unsupported airport: {airport}")
            values = extract_task_indicators(
                scene,
                airport_lat_deg=AIRPORT_CENTERS[airport][0],
                airport_lon_deg=AIRPORT_CENTERS[airport][1],
            )
            days.append(str(scene["source_day"]))
            for name in TASK_INDICATOR_NAMES:
                indicators[name].append(float(values[name]))
    transformed = {
        name: (
            np.asarray(values, dtype=np.float64)
            if name == "heading_dispersion"
            else np.log1p(np.asarray(values, dtype=np.float64))
        )
        for name, values in indicators.items()
    }
    return transformed, np.asarray(days), np.asarray(
        indicators["current_aircraft_count"], dtype=np.float64
    )


def _frozen_outputs(model, x, batch_size: int = 128):
    import torch

    predictions = []
    representations = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            prediction, representation = model(batch)
            predictions.append(prediction.numpy())
            representations.append(representation.numpy())
    return np.concatenate(predictions), np.concatenate(representations)


def _add_seed_summary(result: dict[str, object]) -> None:
    for indicator in result["indicators"].values():
        for subset in ("validation", "test"):
            seed_rows = [
                row[subset]
                for row in indicator["seeds"].values()
                if subset in row
            ]
            if not seed_rows:
                continue
            summary = {}
            for key in (
                "representation_r2",
                "count_r2",
                "representation_minus_count_r2",
                "edge_only_r2",
                "edge_only_minus_count_r2",
                "representation_mae",
                "count_mae",
                "edge_only_mae",
                "representation_spearman",
                "representation_daily_spearman_median",
                "prediction_spearman",
                "prediction_day_count_residual_correlation",
            ):
                values = [float(row[key]) for row in seed_rows if row[key] is not None]
                summary[key] = {
                    "mean": float(np.mean(values)) if values else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                }
            indicator.setdefault("summary", {})[subset] = summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scene_path = args.dataset_root / "pilot_scenes.jsonl"
    x, _, _, split, _, _, _ = _load_experiment_arrays(args.dataset_root)
    indicators, days, current_count = _load_indicators(scene_path)
    if not (len(x) == len(days) == len(split)):
        raise ValueError("scene, tensor, and split rows are not aligned")

    representations = {}
    predictions = {}
    checkpoint_records = []
    for checkpoint_path in args.checkpoint:
        model, metadata = load_invariant_relation_checkpoint(checkpoint_path)
        seed = int(metadata["seed"])
        if seed in representations:
            raise ValueError(f"duplicate seed: {seed}")
        prediction, representation = _frozen_outputs(model, x)
        predictions[seed] = prediction
        representations[seed] = representation
        checkpoint_records.append(
            {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
                "seed": seed,
                "model": metadata["model"],
                "target": metadata.get("target"),
            }
        )

    result = fit_frozen_representation_probes(
        representations=representations,
        predictions=predictions,
        indicators=indicators,
        current_aircraft_count=current_count,
        split=split,
        days=days,
    )
    _add_seed_summary(result)
    result.update(
        {
            "schema_version": "r03_frozen_task_representation_probe_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_sha256": _sha256(scene_path),
            "scene_count": len(days),
            "independent_days": len(set(days.tolist())),
            "checkpoints": checkpoint_records,
            "frozen_network_updates": 0,
            "indicator_transform": (
                "log1p for count-valued indicators; raw heading_dispersion"
            ),
            "comparison": (
                "65-dimensional frozen representation probe versus log1p current-aircraft-count-only Ridge; "
                "all probes fit on train days only"
            ),
            "environment": _runtime_environment(),
            "limitations": [
                "Indicators are trajectory-derived proxies, not controller workload ratings.",
                "The projected-separation indicator is partly convergent with the future proximity-event target.",
                "Linear probe performance demonstrates decodability, not causal use by the prediction head.",
                "Overlapping scenes are dependent; day-level summaries are the safer inferential unit.",
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
