# R02 Timezone Decision

## Decision

`Date` + `Time` represent **UTC observation timestamps**.

R03 must write `report_timestamp_utc` as timezone-aware UTC. Airport-local time
may be derived with the historical `America/New_York` rules but must never
replace the UTC authority.

## Evidence

1. The raw file
   `kbtp/raw/2020/2020/09-11-20/50.csv` contains seconds such as
   `20.135Z`; `Z` is an explicit UTC designator.
2. Across all 62,385,317 retained records, converting the raw timestamp from
   UTC to US Eastern gives a local calendar date matching the source directory
   date in 62,385,317/62,385,317 cases.
3. Treating the raw date directly as local matches only:
   - KAGC: 23,337,747/26,846,786 (86.93%);
   - KBTP: 32,546,138/35,538,531 (91.58%).
   The failures are concentrated around UTC/local midnight, as expected.
4. The official processing code uses UTC terminology and aligns these
   timestamps with time-indexed auxiliary observations.

## Parsing rules

- Accept legacy three-part list strings, ISO-like formatted values, optional
  fractional seconds, and explicit trailing `Z`.
- Use decimal microsecond rounding; values such as `43.999999979` carry to the
  next whole second.
- A naive parsed timestamp is an intermediate representation only. It receives
  UTC semantics after parsing.
- DST affects only derived local time. For these 2020–2022 data, US Eastern
  uses UTC−5 in standard time and UTC−4 in daylight time.

## Validation result

- Timestamp parse failures after the documented parser revision: 0.
- Per-file timestamp evidence rows: 1,911/1,911.
- Stage-blocking timezone ambiguity: none.
