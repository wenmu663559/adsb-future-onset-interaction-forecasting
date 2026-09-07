from __future__ import annotations

import argparse
from collections import defaultdict
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

from airspace_complexity.r03_runner import R03SourceRegistry
from scripts.select_r03_future_onset_confirmation_dates import _source_date


MONTH_STRATA = (
    "2021-11", "2021-12", "2022-01", "2022-02", "2022-03",
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
)


def _previously_used_kagc_dates(archive_root: Path) -> set[str]:
    dates: set[str] = set()
    for path in Path(archive_root).glob("*/pilot_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for group in manifest.get("source_day_splits", {}):
            airport, separator, source_day = str(group).partition("\x00")
            if separator and airport == "KAGC":
                dates.add(source_day)
    return dates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze one unused high-volume KAGC source per observed month."
    )
    parser.add_argument("--project-root", type=Path, default=CHECKOUT_ROOT)
    parser.add_argument(
        "--archive-root", type=Path,
        default=CHECKOUT_ROOT / "runs" / "archive",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    excluded = _previously_used_kagc_dates(args.archive_root)
    registry = R03SourceRegistry.from_lineage(args.project_root)
    by_month = defaultdict(list)
    for source in registry.sources:
        if source.airport_id != "KAGC" or source.authenticated_row_count < 50_000:
            continue
        try:
            source_day = _source_date(source.relative_path)
        except ValueError:
            continue
        month = source_day[:7]
        if month in MONTH_STRATA and source_day not in excluded:
            by_month[month].append((source_day, source))
    selected = []
    for month in MONTH_STRATA:
        if not by_month[month]:
            raise ValueError(f"no unused eligible KAGC source for {month}")
        source_day, source = max(
            by_month[month],
            key=lambda item: (
                item[1].authenticated_row_count, item[0], item[1].source_file_id
            ),
        )
        selected.append({
            "stratum": month,
            "date": source_day,
            "authenticated_rows": source.authenticated_row_count,
            "source_file_id": source.source_file_id,
            "relative_path": source.relative_path,
        })
    record = {
        "schema_version": "r03_kagc_external_validation_dates_v1",
        "selection_frozen_at": "2026-09-03",
        "airport_id": "KAGC",
        "selection_rule": "For each of the ten months represented by KAGC data, exclude every KAGC date used in earlier pilots and select the remaining date whose largest authenticated source has the most rows; do not load targets or model outputs.",
        "month_strata": list(MONTH_STRATA),
        "minimum_authenticated_rows": 50_000,
        "max_hours_per_source": 2,
        "excluded_dates": sorted(excluded),
        "excluded_dates_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode("utf-8")
        ).hexdigest(),
        "registry_id": registry.registry_id,
        "sources": selected,
        "allow_site_tuning": False,
        "run_count": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "dates": [item["date"] for item in selected],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
