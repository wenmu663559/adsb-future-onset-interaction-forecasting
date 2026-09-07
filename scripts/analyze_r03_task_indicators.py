"""Analyze trajectory-derived terminal task indicators without retraining models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

from airspace_complexity.r03_task_indicators import (
    TASK_INDICATOR_NAMES,
    extract_task_indicators,
    summarize_indicator_relationships,
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


def _git_version() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scene_path = args.dataset_root / "pilot_scenes.jsonl"
    records = []
    with scene_path.open(encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            airport = str(scene["airport_id"])
            try:
                latitude, longitude = AIRPORT_CENTERS[airport]
            except KeyError as error:
                raise ValueError(f"unsupported airport: {airport}") from error
            indicators = extract_task_indicators(
                scene,
                airport_lat_deg=latitude,
                airport_lon_deg=longitude,
            )
            records.append({
                "day": str(scene["source_day"]),
                "split": str(scene["split"]),
                "target": float(scene["targets"]["120"]["proximity_event_count"]),
                **indicators,
            })

    result = {
        "schema_version": "r03_task_indicator_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_sha256": _sha256(scene_path),
        "scene_count": len(records),
        "independent_days": len({record["day"] for record in records}),
        "target": "log1p(120_second_proximity_event_count)",
        "indicator_role": "interpretability and convergent-validity proxies; not workload labels",
        "definitions": {
            "screening_pair_count": "n*(n-1)/2, matching the traffic-screening option in Jurinic et al. (2024)",
            "inbound_count_proxy": "aircraft radial distance decreases by at least 0.5 NM over the nearest 30/60-second history",
            "outbound_count_proxy": "aircraft radial distance increases by at least 0.5 NM over the nearest 30/60-second history",
            "closing_pair_count_10nm": "observed pair distance is decreasing and current distance is below 10 NM",
            "projected_separation_task_count_120s": "constant-velocity closest approach within 120 seconds is below 5 NM horizontally and 1000 ft vertically",
            "maneuvering_aircraft_count": "observed heading change at least 20 degrees, speed change at least 10 kt, or altitude change at least 500 ft",
            "heading_dispersion": "one minus circular heading concentration",
        },
        "metrics": summarize_indicator_relationships(
            records, metric_names=TASK_INDICATOR_NAMES
        ),
        "environment": {
            "python": platform.python_version(),
            "operating_system": platform.platform(),
            "cuda": "not used",
            "deep_learning_framework": "not used",
            "code_version": _git_version(),
        },
        "limitations": [
            "Traffic roles are radial-motion proxies, not flight-plan-derived arrival/departure labels.",
            "Projected tasks use constant velocity rather than planned 4D trajectories.",
            "Scene-level correlations are descriptive because overlapping scenes are dependent.",
            "The future proximity-event target is a proxy, not controller-rated complexity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
