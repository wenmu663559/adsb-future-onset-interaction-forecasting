"""Build an R03 canonical-points run and write its evidence boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _project_imports(project_root: Path) -> None:
    root = Path(project_root).resolve()
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--record-batch-rows", type=int)
    parser.add_argument("--max-rows-per-run", type=int)
    parser.add_argument("--max-group-rows-in-memory", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--shard-hash-version")
    parser.add_argument("--final-sort-version")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _arguments().parse_args()
    _project_imports(args.project_root)
    from airspace_complexity.build_canonical_points import (
        build_canonical_points,
        write_r03_run_evidence,
    )
    from airspace_complexity.canonical_points import R03RunParameters

    parameter_names = (
        "shard_count", "record_batch_rows", "max_rows_per_run",
        "max_group_rows_in_memory", "worker_count", "shard_hash_version",
        "final_sort_version",
    )
    parameters = R03RunParameters.from_mapping({
        name: getattr(args, name)
        for name in parameter_names if getattr(args, name) is not None
    })
    result = build_canonical_points(
        args.project_root, args.output_root, parameters, args.resume,
    )
    decision = write_r03_run_evidence(result, args.output_root)
    print(json.dumps({
        "state": result.state,
        "dataset_id": result.identity.dataset_id,
        "decision": decision.decision,
        "r04_authorized": decision.r04_authorized,
        "model_training_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
