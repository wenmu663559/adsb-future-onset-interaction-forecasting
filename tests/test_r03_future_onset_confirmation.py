from types import SimpleNamespace

from scripts.select_r03_future_onset_confirmation_dates import (
    date_stratum,
    select_sources,
)
from scripts.run_r03_online_complexity_pilot import _load_frozen_source_config


def _source(date: str, rows: int, source_id: str):
    month, day, year = date[5:7], date[8:10], date[2:4]
    return SimpleNamespace(
        airport_id="KBTP",
        relative_path=f"raw/{date[:4]}/{date[:4]}/{month}-{day}-{year}/1.csv",
        authenticated_row_count=rows,
        source_file_id=source_id,
    )


def test_date_stratum_matches_frozen_season_blocks() -> None:
    assert date_stratum("2020-09-30") == "2020Q3"
    assert date_stratum("2020-10-01") == "2020Q4"
    assert date_stratum("2021-06-30") == "2021H1"
    assert date_stratum("2021-07-01") == "2021H2"
    assert date_stratum("2022-01-01") == "2022Q1"
    assert date_stratum("2022-12-31") == "2022Q4"


def test_confirmation_selection_excludes_used_dates_and_uses_largest_source() -> None:
    sources = [
        _source("2020-09-01", 100_000, "used"),
        _source("2020-09-02", 80_000, "small"),
        _source("2020-09-03", 120_000, "largest"),
        _source("2020-09-03", 110_000, "same-day-smaller"),
    ]

    result = select_sources(
        sources,
        excluded_dates={"2020-09-01"},
        required_strata=("2020Q3",),
        minimum_authenticated_rows=50_000,
    )

    assert result == [{
        "stratum": "2020Q3",
        "date": "2020-09-03",
        "authenticated_rows": 120_000,
        "source_file_id": "largest",
        "relative_path": "raw/2020/2020/09-03-20/1.csv",
    }]


def test_confirmation_date_config_loads_as_test_only_source_freeze(tmp_path) -> None:
    path = tmp_path / "confirmation.json"
    path.write_text(
        '{"sources":[{"date":"2026-01-01","source_file_id":"abc"}]}',
        encoding="utf-8",
    )

    source_ids, splits = _load_frozen_source_config(path)

    assert source_ids == ("abc",)
    assert splits == {"KBTP\x002026-01-01": "test"}
