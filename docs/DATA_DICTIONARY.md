# TartanAviation Raw ADS-B Data Dictionary

R02 freezes the following raw-field meanings. Units and intended definitions
are supported by the official TartanAviation data paper and processing code.
Ranges below are observed raw ranges, not validity filters.

| Raw field | Parsed type | Meaning and unit | R02 handling |
|---|---|---|---|
| `ID` | string | ADS-B aircraft identifier | Required; preserve exactly |
| `Time` | structured string | Observation time in UTC | Parse legacy list or formatted time; `Z` is an explicit UTC marker |
| `Date` | structured string | Observation date in UTC | Parse legacy list or formatted date |
| `Altitude` | float | Altitude, feet MSL | Preserve raw; convert with `0.3048 m/ft` in R03 |
| `Speed` | float | Ground speed, knots | Missing allowed and flagged; convert with `0.514444444 m/s per knot` in R03 |
| `Heading` | float | Track heading, degrees | Missing allowed and flagged; convert to radians in R03 |
| `Lat` | float | WGS84 latitude, decimal degrees | Preserve raw; physical/spatial QC in R03+ |
| `Lon` | float | WGS84 longitude, decimal degrees | Preserve raw; physical/spatial QC in R03+ |
| `Age` | float | Seconds since the receiver's last observation of the report | Freshness/cache metadata; not event time |
| `Range` | float | Great-circle distance from airport centre, kilometres | Preserve and independently validate in R03; convert to metres |
| `Bearing` | float | Bearing from airport centre, signed degrees from north | Convert to radians in R03 |
| `Tail` | nullable string | Aircraft registration or callsign-like label | Preserve; never use as a required identifier |
| `AltisGNSS` | nullable boolean | Whether altitude source is GNSS | Absent in the 12-column schema; invalid tokens are flagged, not imputed |

## Schema versions

- `raw_v1_12`: the 12 fields through `Tail`.
- `raw_v2_13_altisgnss`: the same fields plus `AltisGNSS`.

KAGC contains the 13-column schema. KBTP contains both versions. Field names,
not column position alone, define the semantic mapping.

## Full-corpus parse evidence

- R01-authorized files: 1,911, including 35 verified zero-length archive
  members represented explicitly as zero-row sources.
- Retained raw records: 62,385,317.
- Timestamp parse success: 62,385,317/62,385,317.
- KAGC schema rows: 26,846,786 in `raw_v2_13_altisgnss`.
- KBTP schema rows: 4,323,904 in `raw_v1_12` and 31,214,627 in
  `raw_v2_13_altisgnss`.

## Quality contract

R03 must retain a quality flag rather than silently removing or imputing:

- missing optional value;
- invalid boolean/numeric token;
- negative `Age`;
- physical-range anomaly;
- duplicate state conflict;
- source terminal record dropped under R01;
- source archive recovery or zero-row source.

Raw extrema are not acceptance ranges. For example, extreme altitude, speed,
range, and coordinates remain evidence for R03 physical validation.
