from datetime import datetime

import pytest

from airspace_complexity.raw_parser import parse_raw_datetime


@pytest.mark.parametrize(
    ("date_value", "time_value", "expected"),
    [
        (
            "[u'2021', u'11', u'30']",
            "[u'13', u'17', u'03.074']",
            datetime(2021, 11, 30, 13, 17, 3, 74000),
        ),
        ("2021-11-30", "13:17:03.074", datetime(2021, 11, 30, 13, 17, 3, 74000)),
        ("11-30-21", "13:17:03", datetime(2021, 11, 30, 13, 17, 3)),
        (
            "[u'2020', u'09', u'12']",
            "[u'00', u'27', u'20.135Z']",
            datetime(2020, 9, 12, 0, 27, 20, 135000),
        ),
        (
            "[u'2020', u'11', u'21']",
            "[u'13', u'13', u'43.999999979']",
            datetime(2020, 11, 21, 13, 13, 44),
        ),
    ],
)
def test_parse_raw_datetime_accepts_documented_variants(
    date_value: str, time_value: str, expected: datetime
) -> None:
    result = parse_raw_datetime(date_value, time_value)
    assert result.status == "ok"
    assert result.timestamp_naive == expected
    assert result.error is None


@pytest.mark.parametrize(
    ("date_value", "time_value", "error_code"),
    [
        ("", "13:00:00", "missing_date"),
        ("2021-11-30", "", "missing_time"),
        ("2021-02-29", "13:00:00", "invalid_datetime"),
        ("[u'2021', u'11']", "[u'13', u'00', u'00']", "invalid_date_shape"),
    ],
)
def test_parse_raw_datetime_classifies_failures(
    date_value: str, time_value: str, error_code: str
) -> None:
    result = parse_raw_datetime(date_value, time_value)
    assert result.status == "error"
    assert result.timestamp_naive is None
    assert result.error == error_code


def test_parse_raw_datetime_does_not_execute_input(tmp_path) -> None:
    marker = tmp_path / "executed"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    result = parse_raw_datetime(payload, "[u'13', u'00', u'00']")
    assert result.status == "error"
    assert not marker.exists()
