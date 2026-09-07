from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys


SCRIPT_PATH = Path(__file__).resolve()
CHECKOUT_ROOT = SCRIPT_PATH.parents[1]
if str(CHECKOUT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

from airspace_complexity.r03_runner import R03SourceRegistry


REQUIRED_STRATA = (
    "2020Q3", "2020Q4", "2021H1", "2021H2",
    "2022Q1", "2022Q2", "2022Q3", "2022Q4",
)
DATE_PATTERN = re.compile(r"/(\d{2})-(\d{2})-(\d{2})/")


def _source_date(relative_path: str) -> str:
    match = DATE_PATTERN.search("/" + relative_path.replace("\\", "/") + "/")
    if match is None:
        raise ValueError(f"source path has no date directory: {relative_path}")
    month, day, short_year = (int(value) for value in match.groups())
    return date(2000 + short_year, month, day).isoformat()


def date_stratum(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed.year == 2020 and 7 <= parsed.month <= 9:
        return "2020Q3"
    if parsed.year == 2020 and 10 <= parsed.month <= 12:
        return "2020Q4"
    if parsed.year == 2021:
        return "2021H1" if parsed.month <= 6 else "2021H2"
    if parsed.year == 2022:
        return f"2022Q{(parsed.month - 1) // 3 + 1}"
    raise ValueError(f"date is outside frozen confirmation strata: {value}")


def select_sources(
    sources, *, excluded_dates: set[str], required_strata: tuple[str, ...],
    minimum_authenticated_rows: int,
) -> list[dict[str, object]]:
    best_by_date: dict[str, object] = {}
    for source in sources:
        if str(source.airport_id).upper() != "KBTP":
            continue
        try:
            source_day = _source_date(str(source.relative_path))
            stratum = date_stratum(source_day)
        except ValueError:
            continue
        if stratum not in required_strata or source_day in excluded_dates:
            continue
        if int(source.authenticated_row_count) < minimum_authenticated_rows:
            continue
        previous = best_by_date.get(source_day)
        if previous is None or (
            int(source.authenticated_row_count), str(source.source_file_id)
        ) > (
            int(previous.authenticated_row_count), str(previous.source_file_id)
        ):
            best_by_date[source_day] = source

    output = []
    for stratum in required_strata:
        candidates = [
            (source_day, source)
            for source_day, source in best_by_date.items()
            if date_stratum(source_day) == stratum
        ]
        if not candidates:
            raise ValueError(f"no eligible unused source for {stratum}")
        source_day, source = max(
            candidates,
            key=lambda item: (
                int(item[1].authenticated_row_count), item[0],
                str(item[1].source_file_id),
            ),
        )
        output.append({
            "stratum": stratum,
            "date": source_day,
            "authenticated_rows": int(source.authenticated_row_count),
            "source_file_id": str(source.source_file_id),
            "relative_path": str(source.relative_path),
        })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mechanically freeze unused KBTP future-onset confirmation dates."
    )
    parser.add_argument("--project-root", type=Path, default=CHECKOUT_ROOT)
    parser.add_argument(
        "--protocol", type=Path,
        default=CHECKOUT_ROOT / "configs" / "r03_future_onset_protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    development = protocol["development_dataset"]
    excluded = set(
        development["train_dates"]
        + development["validation_dates"]
        + development["retrospective_test_dates"]
    )
    prior_configs = sorted(
        args.project_root.glob("configs/r03_external_date_selection*.json")
    )
    for path in prior_configs:
        prior = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(item["date"] for item in prior["sources"])
    registry = R03SourceRegistry.from_lineage(args.project_root)
    selected = select_sources(
        registry.sources,
        excluded_dates=excluded,
        required_strata=REQUIRED_STRATA,
        minimum_authenticated_rows=50_000,
    )
    record = {
        "schema_version": "r03_future_onset_confirmation_dates_v1",
        "selection_frozen_at": "2026-09-03",
        "airport_id": "KBTP",
        "selection_role": "single_untouched_future_onset_confirmation",
        "selection_rule": "For each frozen stratum, after excluding every development and three earlier external-round date, select the unused date whose largest authenticated source has the most rows; never load targets or model outputs.",
        "minimum_authenticated_rows": 50_000,
        "max_hours_per_source": 2,
        "required_strata": list(REQUIRED_STRATA),
        "excluded_date_count": len(excluded),
        "excluded_dates_sha256": hashlib.sha256(
            "\n".join(sorted(excluded)).encode("utf-8")
        ).hexdigest(),
        "registry_id": registry.registry_id,
        "sources": selected,
        "allow_retuning": False,
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
