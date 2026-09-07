from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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

from airspace_complexity.r03_future_onset import summarize_reconstruction
from airspace_complexity.r03_online_complexity import PilotReport, build_scene_examples
from scripts.run_r03_online_complexity_pilot import _json_bytes, _scene_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minimal_records(examples) -> list[dict[str, object]]:
    return [
        {
            "scene_id": scene.scene_id,
            "source_day": scene.source_day,
            "targets": {
                str(horizon): asdict(target)
                for horizon, target in sorted(scene.targets.items())
            },
        }
        for scene in examples
    ]


def run(dataset_root: Path, source_config_path: Path, output: Path) -> dict[str, object]:
    root = Path(dataset_root)
    report_path = root / "pilot_reports.jsonl"
    scene_path = root / "pilot_scenes.jsonl"
    source_config = json.loads(Path(source_config_path).read_text(encoding="utf-8"))
    frozen_splits = dict(source_config["source_day_splits"])
    reports = tuple(
        PilotReport(**json.loads(line)) for line in report_path.open(encoding="utf-8")
    )
    staleness_results: dict[str, object] = {}
    primary_examples = None
    for stale_seconds in (15, 30, 60):
        examples = build_scene_examples(
            reports,
            history_seconds=120,
            step_seconds=10,
            horizons_seconds=(30, 120, 300),
            stale_seconds=stale_seconds,
            min_aircraft=3,
            max_aircraft=100,
        )
        summary = summarize_reconstruction(_minimal_records(examples))
        summary["source_days"] = sorted({scene.source_day for scene in examples})
        staleness_results[str(stale_seconds)] = summary
        if stale_seconds == 30:
            primary_examples = examples
    if primary_examples is None:
        raise AssertionError("primary staleness reconstruction missing")

    digest = hashlib.sha256()
    ages_seconds = []
    for scene in primary_examples:
        group = f"{scene.airport_id}\x00{scene.source_day}"
        digest.update(_json_bytes(_scene_record(scene, frozen_splits[group])))
        ages_seconds.extend(
            (scene.cutoff_us - report.timestamp_us) / 1_000_000.0
            for report in scene.history[-1]
        )
    reconstructed_scene_sha256 = digest.hexdigest()
    stored_scene_sha256 = _sha256(scene_path)
    if reconstructed_scene_sha256 != stored_scene_sha256:
        raise ValueError("primary reconstruction does not reproduce stored scene bytes")

    candidates = sorted(
        primary_examples,
        key=lambda scene: (
            hashlib.sha256(scene.scene_id.encode("utf-8")).digest(), scene.scene_id
        ),
    )
    positive = [
        scene for scene in candidates
        if scene.targets[120].future_onset_proximity_event_count > 0
    ][:12]
    zero = [
        scene for scene in candidates
        if scene.targets[120].future_onset_proximity_event_count == 0
    ][:12]
    sample = sorted(positive + zero, key=lambda scene: scene.scene_id)
    sampled_invariants = {
        "sample_size": len(sample),
        "positive_120s": len(positive),
        "zero_120s": len(zero),
        "history_never_after_cutoff": all(
            report.timestamp_us <= scene.cutoff_us
            for scene in sample for step in scene.history for report in step
        ),
        "onset_never_exceeds_legacy": all(
            target.future_onset_proximity_event_count <= target.proximity_event_count
            for scene in sample for target in scene.targets.values()
        ),
        "scene_ids": [scene.scene_id for scene in sample],
    }
    result: dict[str, object] = {
        "schema_version": "r03_future_event_reconstruction_audit_v1",
        "decision": "TARGET_RECONSTRUCTION_AUDIT_NOT_MODEL_SELECTION",
        "dataset_root": str(root.resolve()),
        "pilot_reports_sha256": _sha256(report_path),
        "pilot_scenes_sha256": stored_scene_sha256,
        "primary_reconstruction_sha256": reconstructed_scene_sha256,
        "primary_reconstruction_exact": True,
        "current_report_age_seconds": {
            "count": len(ages_seconds),
            "p50": float(np.quantile(ages_seconds, 0.50)),
            "p95": float(np.quantile(ages_seconds, 0.95)),
            "max": float(max(ages_seconds)),
        },
        "staleness_sensitivity": staleness_results,
        "deterministic_stratified_sample": sampled_invariants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit future-onset reconstruction and ADS-B staleness sensitivity."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-config", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_development_sources.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.dataset_root, args.source_config, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "primary_reconstruction_exact": result["primary_reconstruction_exact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
