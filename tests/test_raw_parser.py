from airspace_complexity.raw_parser import parse_raw_row


BASE_HEADER = [
    "ID",
    "Time",
    "Date",
    "Altitude",
    "Speed",
    "Heading",
    "Lat",
    "Lon",
    "Age",
    "Range",
    "Bearing",
    "Tail",
]


def test_parse_raw_row_supports_legacy_12_column_schema() -> None:
    row = [
        "abc123",
        "[u'13', u'17', u'03.074']",
        "[u'2021', u'11', u'30']",
        "2500",
        "115.5",
        "92",
        "40.44",
        "-79.95",
        "0.4",
        "3.2",
        "180",
        "N123AB",
    ]
    parsed = parse_raw_row(
        BASE_HEADER,
        row,
        airport="KAGC",
        source_path="kagc/raw/2021/11-30-21/13.csv",
        source_row=2,
        r01_disposition="retain",
    )
    assert parsed["schema_version"] == "raw_v1_12"
    assert parsed["Altitude"] == 2500.0
    assert parsed["ID"] == "abc123"
    assert parsed["timestamp_parse_status"] == "ok"
    assert parsed["timestamp_utc_candidate"].isoformat().endswith("+00:00")
    assert parsed["timestamp_assuming_airport_local"].utcoffset().total_seconds() in {
        -18000,
        -14400,
    }
    assert parsed["source_row"] == 2
    assert parsed["AltisGNSS"] is None


def test_parse_raw_row_supports_13_column_schema_and_boolean() -> None:
    parsed = parse_raw_row(
        BASE_HEADER + ["AltisGNSS"],
        [
            "abc123",
            "[u'13', u'17', u'03']",
            "[u'2021', u'11', u'30']",
            "2500",
            "115",
            "92",
            "40.44",
            "-79.95",
            "0",
            "3.2",
            "180",
            "",
            "True",
        ],
        airport="KBTP",
        source_path="sample.csv",
        source_row=7,
        r01_disposition="read_verified_archive_member",
    )
    assert parsed["schema_version"] == "raw_v2_13_altisgnss"
    assert parsed["AltisGNSS"] is True
    assert parsed["Tail"] is None


def test_parse_raw_row_preserves_invalid_numeric_as_error_not_imputation() -> None:
    parsed = parse_raw_row(
        BASE_HEADER,
        ["a", "13:00:00", "2021-11-30", "bad"] + [""] * 8,
        airport="KAGC",
        source_path="sample.csv",
        source_row=3,
        r01_disposition="retain",
    )
    assert parsed["Altitude"] is None
    assert parsed["field_errors"] == {"Altitude": "invalid_float"}


def test_parse_raw_row_rejects_column_count_mismatch() -> None:
    try:
        parse_raw_row(
            BASE_HEADER,
            ["too", "short"],
            airport="KAGC",
            source_path="sample.csv",
            source_row=2,
            r01_disposition="retain",
        )
    except ValueError as exc:
        assert "column count" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_raw_row_flags_nonfinite_numeric() -> None:
    parsed = parse_raw_row(
        BASE_HEADER,
        ["a", "13:00:00", "2021-11-30", "inf"] + [""] * 8,
        airport="KAGC",
        source_path="sample.csv",
        source_row=3,
        r01_disposition="retain",
    )
    assert parsed["Altitude"] is None
    assert parsed["field_errors"]["Altitude"] == "nonfinite_float"
