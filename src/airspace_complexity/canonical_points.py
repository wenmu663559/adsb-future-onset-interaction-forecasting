"""Frozen R03 canonical identifiers, policies, and Arrow schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import pyarrow as pa
import yaml

from .raw_parser import SCHEMAS, parse_raw_row

if TYPE_CHECKING:
    from .r03_runner import EnumeratedRecord


R03_POLICY_VERSION = "r03_canonical_points_v1"
COMPLETENESS_FIELDS = (
    "lat_deg",
    "lon_deg",
    "altitude_m",
    "speed_mps",
    "heading_rad",
    "age_s",
    "range_m",
    "bearing_rad",
    "tail_or_callsign",
)
CORE_STATE_FIELDS = (
    "lat_deg",
    "lon_deg",
    "altitude_m",
    "speed_mps",
    "heading_rad",
)
METADATA_FIELDS = (
    "age_s",
    "range_m",
    "bearing_rad",
    "tail_or_callsign",
    "altitude_source_gnss",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_SCHEMA_VERSION = "r03_qc_policy_v1"
_POLICY_KEYS = {"schema_version", "coordinate_source", "airports", "geometry"}
_COORDINATE_SOURCE = {
    "authority": "FAA_NASR_APT_BASE",
    "effective_date": "2026-07-09",
    "source_url": "https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/2026-07-09/",
}
_AIRPORTS = {
    "KAGC": {"latitude_deg": 40.3544376, "longitude_deg": -79.9290467},
    "KBTP": {"latitude_deg": 40.7765833333, "longitude_deg": -79.9510833333},
}
_GEOMETRY = {
    "earth_model": "mean_spherical_earth",
    "earth_radius_m": 6371008.8,
    "bearing_convention": "initial_true_clockwise_from_north",
    "range_absolute_tolerance_m": 500.0,
    "range_relative_tolerance": 0.05,
    "bearing_absolute_tolerance_deg": 5.0,
}
_PARAMETER_DEFAULTS: dict[str, object] = {
    "shard_count": 256,
    "record_batch_rows": 65_536,
    "max_rows_per_run": 250_000,
    "max_group_rows_in_memory": 50_000,
    "worker_count": 1,
    "shard_hash_version": "sha256_canonical_key_v1",
    "final_sort_version": "r03_final_sort_v1",
    "max_records_per_airport": None,
}

_PARSE_QUALITY_VERSION = "r03_parse_quality_v1"
_QC_REGISTRY_VERSION = "r03_advisory_qc_v1"


@dataclass(frozen=True)
class QualityRule:
    """One immutable entry in a versioned row-quality bit registry."""

    registry_version: str
    code: str
    bit: int
    domain: str
    severity: str
    condition: str
    affected_fields: tuple[str, ...]
    action: str


@dataclass(frozen=True)
class ParsedRow:
    raw_row_id: str
    airport_id: str
    source_file_id: str
    source_relative_path: str
    archive_member: str | None
    source_content_sha256: str
    logical_record_ordinal: int
    source_row_number: int
    source_sequence: int
    aircraft_id: str
    tail_or_callsign: str | None
    report_timestamp_utc: datetime
    lat_deg: float | None
    lon_deg: float | None
    altitude_m: float | None
    speed_mps: float | None
    heading_rad: float | None
    age_s: float | None
    range_m: float | None
    bearing_rad: float | None
    altitude_source_gnss: bool | None
    raw_schema_version: str
    parse_quality_bits: int
    qc_bits: int
    raw_altitude_ft: float | None
    raw_speed_knots: float | None
    raw_heading_deg: float | None
    raw_lat_deg: float | None
    raw_lon_deg: float | None
    raw_age_s: float | None
    raw_range_km: float | None
    raw_bearing_deg: float | None
    raw_tail: str | None
    raw_altisgnss: bool | None
    partition_year: int
    partition_month: int


@dataclass(frozen=True)
class BadRow:
    raw_row_id: str
    airport_id: str
    source_file_id: str
    source_relative_path: str
    archive_member: str | None
    source_content_sha256: str
    logical_record_ordinal: int
    source_row_number: int
    source_sequence: int
    raw_tokens: tuple[str, ...]
    raw_payload: str
    redaction_status: str
    parser_status: str
    reason_code: str
    parse_quality_bits: int


@dataclass(frozen=True)
class ResolvedGroup:
    """The lossless, deterministic result for one natural-key group."""

    unique_row: Mapping[str, object]
    membership_rows: tuple[Mapping[str, object], ...]
    conflict_row: Mapping[str, object] | None


def _natural_key(row: ParsedRow) -> tuple[str, str, datetime]:
    return (row.airport_id, row.aircraft_id, row.report_timestamp_utc)


def shard_for_key(
    natural_key: tuple[str, str, datetime],
    shard_count: int,
    shard_hash_version: str = "sha256_canonical_key_v1",
) -> int:
    """Return the frozen content-addressed shard for a canonical natural key."""

    if shard_hash_version != "sha256_canonical_key_v1":
        raise ValueError(f"unsupported shard_hash_version: {shard_hash_version!r}")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if not isinstance(natural_key, tuple) or len(natural_key) != 3:
        raise ValueError("natural_key must be (airport_id, aircraft_id, timestamp)")
    airport_id, aircraft_id, timestamp = natural_key
    _require_text(airport_id, "airport_id")
    _require_text(aircraft_id, "aircraft_id")
    payload = canonical_json({
        "version": shard_hash_version,
        "airport_id": airport_id,
        "aircraft_id": aircraft_id,
        "timestamp_utc_microsecond_z": _format_utc_timestamp(timestamp),
    })
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % shard_count


def age_rank(age_s: float | None) -> tuple[int, float]:
    """Return the frozen age class and finite/positive-infinity rank value."""

    if age_s is None or isinstance(age_s, bool) or not isinstance(age_s, (int, float)):
        if age_s is None:
            return (2, math.inf)
        raise TypeError("age_s must be a float or None")
    value = float(age_s)
    if not math.isfinite(value):
        return (2, math.inf)
    if value >= 0.0:
        return (0, 0.0 if value == 0.0 else value)
    return (1, abs(value))


def stable_source_order(row: ParsedRow) -> tuple[object, ...]:
    """Return the machine-independent frozen source order."""

    return (
        row.airport_id,
        row.source_sequence,
        row.source_file_id,
        row.source_relative_path,
        (row.archive_member or "").replace("\\", "/"),
        row.source_row_number,
        row.raw_row_id,
    )


def _completeness(row: ParsedRow) -> int:
    return sum(getattr(row, field) is not None for field in COMPLETENESS_FIELDS)


def _canonical_scalar(value: object) -> object:
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _difference_bits(rows: Sequence[ParsedRow], fields: Sequence[str]) -> int:
    bits = 0
    for index, field in enumerate(fields):
        first = _canonical_scalar(getattr(rows[0], field))
        if any(_canonical_scalar(getattr(row, field)) != first for row in rows[1:]):
            bits |= 1 << index
    return bits


def _winner_reason(ordered: Sequence[ParsedRow]) -> str:
    if len(ordered) == 1:
        return "only_member"
    winner = ordered[0]
    winner_age = age_rank(winner.age_s)
    same_age = [row for row in ordered if age_rank(row.age_s) == winner_age]
    if len(same_age) == 1:
        return "min_nonnegative_age" if winner_age[0] == 0 else "closest_zero_negative_age"
    best_completeness = _completeness(winner)
    same_completeness = [row for row in same_age if _completeness(row) == best_completeness]
    if len(same_completeness) == 1:
        return "max_completeness_after_age_tie"
    return "stable_source_order_after_full_tie"


def resolve_group(
    rows: Sequence[ParsedRow], policies: CanonicalPolicies
) -> ResolvedGroup:
    """Resolve one nonempty natural-key group without discarding any member."""

    if not isinstance(policies, CanonicalPolicies):
        raise TypeError("policies must be CanonicalPolicies")
    if not rows:
        raise ValueError("duplicate group must be nonempty")
    if not all(isinstance(row, ParsedRow) for row in rows):
        raise TypeError("duplicate group members must be ParsedRow values")
    natural_key = _natural_key(rows[0])
    if any(_natural_key(row) != natural_key for row in rows[1:]):
        raise ValueError("all duplicate group rows must share one natural key")
    if len({row.raw_row_id for row in rows}) != len(rows):
        raise ValueError("raw_row_id values must be unique within a group")

    ordered = sorted(
        rows,
        key=lambda row: (*age_rank(row.age_s), -_completeness(row), stable_source_order(row)),
    )
    winner = ordered[0]
    report_id = canonical_report_id(*natural_key)
    core_bits = _difference_bits(ordered, CORE_STATE_FIELDS)
    metadata_bits = _difference_bits(ordered, METADATA_FIELDS)
    if len(ordered) == 1:
        classification = "singleton"
    elif core_bits and metadata_bits:
        classification = "core_and_metadata_conflict"
    elif core_bits:
        classification = "core_state_conflict"
    elif metadata_bits:
        classification = "metadata_conflict"
    else:
        classification = "identical"
    reason = _winner_reason(ordered)
    group_size = len(ordered)

    membership: list[Mapping[str, object]] = []
    for rank, row in enumerate(ordered, start=1):
        age_class, age_value = age_rank(row.age_s)
        membership.append({
            "raw_row_id": row.raw_row_id,
            "canonical_report_id": report_id,
            "group_size": group_size,
            "dedup_rank": rank,
            "selected": rank == 1,
            "selection_reason": reason if rank == 1 else "not_selected",
            "age_rank_class": age_class,
            "age_rank_value": age_value,
            "completeness_score": _completeness(row),
            "completeness_denominator": len(COMPLETENESS_FIELDS),
            "core_state_difference_bits": core_bits,
            "metadata_difference_bits": metadata_bits,
            "airport_id": row.airport_id,
            "source_file_id": row.source_file_id,
            "source_relative_path": row.source_relative_path,
            "archive_member": row.archive_member,
            "source_content_sha256": row.source_content_sha256,
            "logical_record_ordinal": row.logical_record_ordinal,
            "source_row_number": row.source_row_number,
            "source_sequence": row.source_sequence,
            "partition_year": row.partition_year,
            "partition_month": row.partition_month,
        })

    unique = {
        "canonical_report_id": report_id,
        "selected_raw_row_id": winner.raw_row_id,
        "airport_id": winner.airport_id,
        "aircraft_id": winner.aircraft_id,
        "report_timestamp_utc": winner.report_timestamp_utc,
        "tail_or_callsign": winner.tail_or_callsign,
        "lat_deg": winner.lat_deg,
        "lon_deg": winner.lon_deg,
        "altitude_m": winner.altitude_m,
        "speed_mps": winner.speed_mps,
        "heading_rad": winner.heading_rad,
        "age_s": winner.age_s,
        "range_m": winner.range_m,
        "bearing_rad": winner.bearing_rad,
        "altitude_source_gnss": winner.altitude_source_gnss,
        "group_size": group_size,
        "is_duplicate": group_size > 1,
        "duplicate_classification": classification,
        "selection_policy_version": R03_POLICY_VERSION,
        "core_state_difference_bits": core_bits,
        "metadata_difference_bits": metadata_bits,
        "duplicate_conflict_flag": core_bits != 0,
        "metadata_conflict_flag": metadata_bits != 0,
        "any_difference_flag": core_bits != 0 or metadata_bits != 0,
        "parse_quality_bits": winner.parse_quality_bits,
        "qc_bits": winner.qc_bits,
        "partition_year": winner.partition_year,
        "partition_month": winner.partition_month,
    }
    conflict = None
    if core_bits or metadata_bits:
        conflict = {
            "canonical_report_id": report_id,
            "airport_id": winner.airport_id,
            "partition_year": winner.partition_year,
            "partition_month": winner.partition_month,
            "duplicate_classification": classification,
            "core_state_difference_bits": core_bits,
            "metadata_difference_bits": metadata_bits,
            "group_size": group_size,
            "selected_raw_row_id": winner.raw_row_id,
        }
    return ResolvedGroup(unique, tuple(membership), conflict)


def _quality_rule(
    registry_version: str,
    code: str,
    bit_index: int,
    domain: str,
    severity: str,
    condition: str,
    affected_fields: tuple[str, ...],
    action: str,
) -> QualityRule:
    return QualityRule(
        registry_version=registry_version,
        code=code,
        bit=1 << bit_index,
        domain=domain,
        severity=severity,
        condition=condition,
        affected_fields=affected_fields,
        action=action,
    )


_PARSE_RULE_SPECS = (
    ("invalid_boolean_token", "parser", "warning", "a non-missing boolean token cannot be parsed", ("AltisGNSS",), "retain_and_flag"),
    ("invalid_numeric_token", "parser", "warning", "a non-missing numeric token cannot be parsed", tuple(sorted(("Altitude", "Speed", "Heading", "Lat", "Lon", "Age", "Range", "Bearing"))), "retain_and_flag"),
    ("invalid_or_missing_aircraft_id", "identity", "error", "aircraft ID is missing or is not a nonempty UTF-8 scalar string", ("ID",), "route_to_bad_rows"),
    ("invalid_or_missing_utc_timestamp", "identity", "error", "Date or Time is missing or cannot form a valid UTC timestamp", ("Date", "Time"), "route_to_bad_rows"),
    ("nonfinite_numeric_token", "parser", "warning", "a numeric token is NaN or infinite", tuple(sorted(("Altitude", "Speed", "Heading", "Lat", "Lon", "Age", "Range", "Bearing"))), "retain_and_flag"),
    ("row_parser_failure", "parser", "error", "a supported-schema row cannot be parsed", tuple(), "route_to_bad_rows"),
    ("unsupported_schema", "parser", "error", "the normalized header is not a frozen raw schema", tuple(), "route_to_bad_rows"),
)

_QC_RULE_SPECS = (
    ("bearing_geometry_inconsistent", "geometry", "warning", "reported bearing differs from the airport-derived initial bearing beyond tolerance", ("Lat", "Lon", "Bearing")),
    ("bearing_geometry_skipped", "geometry", "info", "bearing consistency inputs or airport coordinates are unavailable, or bearing is undefined at zero range", ("Lat", "Lon", "Bearing")),
    ("heading_out_of_domain", "kinematics", "warning", "raw heading is outside the inclusive [0, 360] degree domain", ("Heading",)),
    ("latitude_out_of_domain", "position", "warning", "latitude is outside the inclusive [-90, 90] degree domain", ("Lat",)),
    ("longitude_out_of_domain", "position", "warning", "longitude is outside the inclusive [-180, 180] degree domain", ("Lon",)),
    ("missing_age", "freshness", "warning", "Age is missing or has an invalid non-numeric token", ("Age",)),
    ("missing_observation_fields", "completeness", "warning", "one or more observation fields are null", ("Altitude", "Speed", "Heading", "Lat", "Lon", "Range", "Bearing")),
    ("negative_age", "freshness", "warning", "Age is negative", ("Age",)),
    ("negative_range", "geometry", "warning", "reported range is negative", ("Range",)),
    ("negative_speed", "kinematics", "warning", "reported speed is negative", ("Speed",)),
    ("nonfinite_age", "freshness", "warning", "Age token is nonfinite", ("Age",)),
    ("range_geometry_inconsistent", "geometry", "warning", "reported range differs from airport-derived range beyond tolerance", ("Lat", "Lon", "Range")),
    ("range_geometry_skipped", "geometry", "info", "range consistency inputs or airport coordinates are unavailable", ("Lat", "Lon", "Range")),
    ("timestamp_decrease", "ordering", "warning", "UTC timestamp decreases relative to the previous parsed-valid row in the source", ("Date", "Time")),
)


def _build_registry(
    version: str,
    specs: tuple[tuple[object, ...], ...],
    *,
    action: str | None = None,
) -> Mapping[str, QualityRule]:
    records: dict[str, QualityRule] = {}
    for bit_index, spec in enumerate(sorted(specs, key=lambda item: str(item[0]))):
        if action is None:
            code, domain, severity, condition, affected_fields, rule_action = spec
        else:
            code, domain, severity, condition, affected_fields = spec
            rule_action = action
        records[str(code)] = _quality_rule(
            version,
            str(code),
            bit_index,
            str(domain),
            str(severity),
            str(condition),
            tuple(affected_fields),  # type: ignore[arg-type]
            str(rule_action),
        )
    return MappingProxyType(records)


_PARSE_QUALITY_REGISTRY = _build_registry(_PARSE_QUALITY_VERSION, _PARSE_RULE_SPECS)
_QC_REGISTRY = _build_registry(
    _QC_REGISTRY_VERSION, _QC_RULE_SPECS, action="retain_and_flag"
)


def parse_quality_registry() -> Mapping[str, QualityRule]:
    """Return the immutable, code-sorted parse-quality registry."""
    return _PARSE_QUALITY_REGISTRY


def qc_registry() -> Mapping[str, QualityRule]:
    """Return the immutable, code-sorted advisory-QC registry."""
    return _QC_REGISTRY


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    return value


def _canonical_value(value: object) -> object:
    if value is None:
        return {"$r03_type": "null"}
    if isinstance(value, str):
        return _require_text(value, "string value")
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON mapping keys must be strings")
            _require_text(key, "mapping key")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise ValueError(f"canonical JSON does not support {type(value).__name__}")


def canonical_json(record: Mapping[str, object]) -> bytes:
    """Serialize a supported mapping deterministically as canonical UTF-8 JSON."""
    if not isinstance(record, Mapping):
        raise ValueError("canonical JSON input must be a mapping")
    normalized = _canonical_value(record)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_canonical(record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def raw_row_id(
    source_file_id: str,
    source_content_sha256: str,
    archive_member: str | None,
    logical_record_ordinal: int,
) -> str:
    """Return the immutable identity of one logical record before parsing."""
    source_file_id = _require_text(source_file_id, "source_file_id")
    source_content_sha256 = _require_text(
        source_content_sha256, "source_content_sha256"
    )
    if not _SHA256_RE.fullmatch(source_content_sha256):
        raise ValueError("source_content_sha256 must be a 64-character lowercase hex hash")
    if archive_member is not None:
        archive_member = _require_text(archive_member, "archive_member")
    if isinstance(logical_record_ordinal, bool) or not isinstance(
        logical_record_ordinal, int
    ):
        raise ValueError("logical_record_ordinal must be an integer")
    if logical_record_ordinal < 1:
        raise ValueError("logical_record_ordinal must be one-based")
    return _sha256_canonical(
        {
            "version": R03_POLICY_VERSION,
            "source_file_id": source_file_id,
            "source_content_sha256": source_content_sha256,
            "archive_member": archive_member,
            "logical_record_ordinal": logical_record_ordinal,
        }
    )


def _format_utc_timestamp(timestamp: datetime) -> str:
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is not timezone.utc:
        raise ValueError("report_timestamp_utc must be UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{timestamp.microsecond:06d}Z"


def canonical_report_id(
    airport_id: str, aircraft_id: str, report_timestamp_utc: datetime
) -> str:
    """Return the stable report identity used for R03 duplicate grouping."""
    return _sha256_canonical(
        {
            "version": R03_POLICY_VERSION,
            "airport_id": _require_text(airport_id, "airport_id"),
            "aircraft_id": _require_text(aircraft_id, "aircraft_id"),
            "report_timestamp_utc": _format_utc_timestamp(report_timestamp_utc),
        }
    )


@dataclass(frozen=True)
class R03RunParameters:
    shard_count: int
    record_batch_rows: int
    max_rows_per_run: int
    max_group_rows_in_memory: int
    worker_count: int
    shard_hash_version: str
    final_sort_version: str
    max_records_per_airport: int | None

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "R03RunParameters":
        if not isinstance(record, Mapping):
            raise ValueError("R03 run parameters must be a mapping")
        unknown = set(record) - set(_PARAMETER_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown R03 run parameter fields: {sorted(unknown)}")
        values = {**_PARAMETER_DEFAULTS, **record}
        for name in (
            "shard_count",
            "record_batch_rows",
            "max_rows_per_run",
            "max_group_rows_in_memory",
            "worker_count",
        ):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        maximum = values["max_records_per_airport"]
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0
        ):
            raise ValueError("max_records_per_airport must be null or a positive integer")
        for name in ("shard_hash_version", "final_sort_version"):
            value = values[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
            _require_text(value, name)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CanonicalPolicies:
    policy_version: str
    schema_version: str
    coordinate_source: dict[str, object]
    airports: dict[str, dict[str, float]]
    geometry: dict[str, object]

    @classmethod
    def from_project(cls, project_root: Path) -> "CanonicalPolicies":
        policy_path = project_root / "configs" / "data" / "r03_qc_policy.yaml"
        if not policy_path.is_file():
            raise FileNotFoundError(f"R03 QC policy not found: {policy_path}")
        data = yaml.load(
            policy_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
        if not isinstance(data, dict):
            raise ValueError("R03 QC policy must be a mapping")
        if set(data) != _POLICY_KEYS:
            unknown = set(data) - _POLICY_KEYS
            missing = _POLICY_KEYS - set(data)
            raise ValueError(
                f"invalid R03 QC policy keys; unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        if data["schema_version"] != _POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported R03 QC policy version")
        _validate_exact_mapping("coordinate_source", data["coordinate_source"], _COORDINATE_SOURCE)
        _validate_airports(data["airports"])
        _validate_exact_mapping("geometry", data["geometry"], _GEOMETRY)
        return cls(
            policy_version=R03_POLICY_VERSION,
            schema_version=data["schema_version"],
            coordinate_source=data["coordinate_source"],
            airports=data["airports"],
            geometry=data["geometry"],
        )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _validate_exact_mapping(
    name: str, value: object, expected: Mapping[str, object]
) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"R03 QC policy {name} has invalid keys")
    for key, expected_value in expected.items():
        actual = value[key]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(f"R03 QC policy {name}.{key} has an invalid value")


def _validate_airports(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(_AIRPORTS):
        raise ValueError("R03 QC policy airports has invalid keys")
    for airport_id, expected_coordinates in _AIRPORTS.items():
        _validate_exact_mapping(
            f"airports.{airport_id}", value[airport_id], expected_coordinates
        )


def _safe_token(token: object) -> tuple[str, bool]:
    text = str(token)
    safe = text.encode("utf-8", errors="replace").decode("utf-8")
    return safe, safe != text


def _bad_row(
    record: "EnumeratedRecord", reason_code: str, parser_status: str
) -> BadRow:
    safe_header: list[str] = []
    safe_row: list[str] = []
    replaced = False
    for token in record.header:
        safe, changed = _safe_token(token)
        safe_header.append(safe)
        replaced |= changed
    for token in record.row:
        safe, changed = _safe_token(token)
        safe_row.append(safe)
        replaced |= changed
    source = record.source
    spec = source.spec
    return BadRow(
        raw_row_id=record.raw_row_id,
        airport_id=spec.airport_id,
        source_file_id=spec.source_file_id,
        source_relative_path=spec.relative_path,
        archive_member=spec.archive_member,
        source_content_sha256=source.source_content_sha256,
        logical_record_ordinal=record.logical_record_ordinal,
        source_row_number=record.source_row_number,
        source_sequence=spec.source_sequence,
        raw_tokens=tuple(safe_row),
        raw_payload=json.dumps(
            {"header": safe_header, "row": safe_row},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        redaction_status="utf8_replacement_applied" if replaced else "not_required",
        parser_status=parser_status,
        reason_code=reason_code,
        parse_quality_bits=_PARSE_QUALITY_REGISTRY[reason_code].bit,
    )


def _valid_aircraft_id(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _quality_bits(field_errors: Mapping[str, str]) -> int:
    bits = 0
    for error in field_errors.values():
        if error == "invalid_float":
            bits |= _PARSE_QUALITY_REGISTRY["invalid_numeric_token"].bit
        elif error == "nonfinite_float":
            bits |= _PARSE_QUALITY_REGISTRY["nonfinite_numeric_token"].bit
        elif error == "invalid_boolean":
            bits |= _PARSE_QUALITY_REGISTRY["invalid_boolean_token"].bit
    return bits


def _flag(bits: int, code: str) -> int:
    return bits | _QC_REGISTRY[code].bit


def _geometry_from_airport(
    airport_lat_deg: float,
    airport_lon_deg: float,
    target_lat_deg: float,
    target_lon_deg: float,
    earth_radius_m: float,
) -> tuple[float, float]:
    airport_lat = airport_lat_deg * math.pi / 180.0
    target_lat = target_lat_deg * math.pi / 180.0
    delta_lat = target_lat - airport_lat
    delta_lon = (target_lon_deg - airport_lon_deg) * math.pi / 180.0
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(airport_lat)
        * math.cos(target_lat)
        * math.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * math.atan2(
        math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine))
    )
    east = math.sin(delta_lon) * math.cos(target_lat)
    north = (
        math.cos(airport_lat) * math.sin(target_lat)
        - math.sin(airport_lat) * math.cos(target_lat) * math.cos(delta_lon)
    )
    bearing_deg = math.degrees(math.atan2(east, north)) % 360.0
    return earth_radius_m * central_angle, bearing_deg


def partition_key(
    timestamp_utc: datetime, airport_id: str
) -> tuple[str, int, int]:
    """Return the frozen airport/UTC-year/UTC-month partition key."""
    if not isinstance(timestamp_utc, datetime) or timestamp_utc.tzinfo is not timezone.utc:
        raise ValueError("timestamp_utc must be timezone.utc")
    airport_id = _require_text(airport_id, "airport_id")
    if not airport_id:
        raise ValueError("airport_id must be non-empty")
    return airport_id, timestamp_utc.year, timestamp_utc.month


def convert_record(
    record: "EnumeratedRecord",
    policies: CanonicalPolicies,
    airport_coordinates: Mapping[str, tuple[float, float]],
    previous_timestamp_utc: datetime | None,
) -> ParsedRow | BadRow:
    """Convert one pre-identified row without filtering advisory-QC failures."""
    if previous_timestamp_utc is not None and (
        not isinstance(previous_timestamp_utc, datetime)
        or previous_timestamp_utc.tzinfo is not timezone.utc
    ):
        raise ValueError("previous_timestamp_utc must be timezone.utc or None")
    normalized_header = tuple(value.strip() for value in record.header)
    if normalized_header not in SCHEMAS:
        return _bad_row(record, "unsupported_schema", "unsupported_schema")
    try:
        parsed = parse_raw_row(
            record.header,
            record.row,
            airport=record.source.spec.airport_id,
            source_path=record.source.spec.relative_path,
            source_row=record.source_row_number,
            r01_disposition=record.source.spec.r01_disposition,
            archive_member=record.source.spec.archive_member,
        )
    except Exception as exc:
        return _bad_row(record, "row_parser_failure", type(exc).__name__)

    aircraft_id = parsed.get("ID")
    if aircraft_id is None or aircraft_id == "":
        return _bad_row(
            record,
            "invalid_or_missing_aircraft_id",
            "missing_aircraft_id",
        )
    if not _valid_aircraft_id(aircraft_id):
        return _bad_row(
            record,
            "invalid_or_missing_aircraft_id",
            "invalid_aircraft_id",
        )
    timestamp = parsed.get("timestamp_utc_candidate")
    if timestamp is None:
        parser_status = str(
            parsed.get("timestamp_parse_error") or "invalid_utc_timestamp"
        )
        return _bad_row(
            record,
            "invalid_or_missing_utc_timestamp",
            parser_status,
        )
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is not timezone.utc:
        return _bad_row(
            record,
            "invalid_or_missing_utc_timestamp",
            "non_utc_timestamp",
        )

    raw_altitude = parsed.get("Altitude")
    raw_speed = parsed.get("Speed")
    raw_heading = parsed.get("Heading")
    raw_lat = parsed.get("Lat")
    raw_lon = parsed.get("Lon")
    raw_age = parsed.get("Age")
    raw_range = parsed.get("Range")
    raw_bearing = parsed.get("Bearing")
    field_errors = parsed.get("field_errors", {})
    parse_bits = _quality_bits(field_errors)
    qc_bits = 0

    if raw_lat is not None and not -90.0 <= raw_lat <= 90.0:
        qc_bits = _flag(qc_bits, "latitude_out_of_domain")
    if raw_lon is not None and not -180.0 <= raw_lon <= 180.0:
        qc_bits = _flag(qc_bits, "longitude_out_of_domain")
    if raw_speed is not None and raw_speed < 0.0:
        qc_bits = _flag(qc_bits, "negative_speed")
    if raw_range is not None and raw_range < 0.0:
        qc_bits = _flag(qc_bits, "negative_range")
    if raw_heading is not None and not 0.0 <= raw_heading <= 360.0:
        qc_bits = _flag(qc_bits, "heading_out_of_domain")
    age_error = field_errors.get("Age")
    if raw_age is None:
        qc_bits = _flag(
            qc_bits,
            "nonfinite_age" if age_error == "nonfinite_float" else "missing_age",
        )
    elif raw_age < 0.0:
        qc_bits = _flag(qc_bits, "negative_age")
    observation_values = (
        raw_altitude, raw_speed, raw_heading, raw_lat, raw_lon, raw_range, raw_bearing
    )
    if any(value is None for value in observation_values):
        qc_bits = _flag(qc_bits, "missing_observation_fields")
    if previous_timestamp_utc is not None and timestamp < previous_timestamp_utc:
        qc_bits = _flag(qc_bits, "timestamp_decrease")

    coordinates = airport_coordinates.get(record.source.spec.airport_id)
    position_available = (
        raw_lat is not None
        and raw_lon is not None
        and -90.0 <= raw_lat <= 90.0
        and -180.0 <= raw_lon <= 180.0
        and coordinates is not None
        and len(coordinates) == 2
        and all(math.isfinite(value) for value in coordinates)
        and -90.0 <= coordinates[0] <= 90.0
        and -180.0 <= coordinates[1] <= 180.0
    )
    if position_available:
        assert coordinates is not None and raw_lat is not None and raw_lon is not None
        expected_range, expected_bearing = _geometry_from_airport(
            coordinates[0],
            coordinates[1],
            raw_lat,
            raw_lon,
            float(policies.geometry["earth_radius_m"]),
        )
        if raw_range is None:
            qc_bits = _flag(qc_bits, "range_geometry_skipped")
        else:
            reported_range = raw_range * 1000.0
            tolerance = max(
                float(policies.geometry["range_absolute_tolerance_m"]),
                float(policies.geometry["range_relative_tolerance"]) * expected_range,
            )
            if abs(reported_range - expected_range) > tolerance:
                qc_bits = _flag(qc_bits, "range_geometry_inconsistent")
        if raw_bearing is None or expected_range == 0.0:
            qc_bits = _flag(qc_bits, "bearing_geometry_skipped")
        else:
            circular_error = abs((raw_bearing - expected_bearing + 180.0) % 360.0 - 180.0)
            if circular_error > float(policies.geometry["bearing_absolute_tolerance_deg"]):
                qc_bits = _flag(qc_bits, "bearing_geometry_inconsistent")
    else:
        qc_bits = _flag(qc_bits, "range_geometry_skipped")
        qc_bits = _flag(qc_bits, "bearing_geometry_skipped")

    airport_id, year, month = partition_key(timestamp, record.source.spec.airport_id)
    return ParsedRow(
        raw_row_id=record.raw_row_id,
        airport_id=airport_id,
        source_file_id=record.source.spec.source_file_id,
        source_relative_path=record.source.spec.relative_path,
        archive_member=record.source.spec.archive_member,
        source_content_sha256=record.source.source_content_sha256,
        logical_record_ordinal=record.logical_record_ordinal,
        source_row_number=record.source_row_number,
        source_sequence=record.source.spec.source_sequence,
        aircraft_id=aircraft_id,
        tail_or_callsign=parsed.get("Tail"),
        report_timestamp_utc=timestamp,
        lat_deg=raw_lat,
        lon_deg=raw_lon,
        altitude_m=None if raw_altitude is None else raw_altitude * 0.3048,
        speed_mps=None if raw_speed is None else raw_speed * 0.5144444444444445,
        heading_rad=None if raw_heading is None else raw_heading * math.pi / 180.0,
        age_s=raw_age,
        range_m=None if raw_range is None else raw_range * 1000.0,
        bearing_rad=None if raw_bearing is None else raw_bearing * math.pi / 180.0,
        altitude_source_gnss=parsed.get("AltisGNSS"),
        raw_schema_version=str(parsed["schema_version"]),
        parse_quality_bits=parse_bits,
        qc_bits=qc_bits,
        raw_altitude_ft=raw_altitude,
        raw_speed_knots=raw_speed,
        raw_heading_deg=raw_heading,
        raw_lat_deg=raw_lat,
        raw_lon_deg=raw_lon,
        raw_age_s=raw_age,
        raw_range_km=raw_range,
        raw_bearing_deg=raw_bearing,
        raw_tail=parsed.get("Tail"),
        raw_altisgnss=parsed.get("AltisGNSS"),
        partition_year=year,
        partition_month=month,
    )


def _field(name: str, type_: pa.DataType, *, nullable: bool = False) -> pa.Field:
    return pa.field(name, type_, nullable=nullable)


def parsed_rows_schema() -> pa.Schema:
    return pa.schema([
        _field("raw_row_id", pa.string()), _field("airport_id", pa.string()),
        _field("source_file_id", pa.string()), _field("source_relative_path", pa.string()),
        _field("archive_member", pa.string(), nullable=True), _field("source_content_sha256", pa.string()),
        _field("logical_record_ordinal", pa.int64()), _field("source_row_number", pa.int64()),
        _field("source_sequence", pa.int64()), _field("aircraft_id", pa.string()),
        _field("tail_or_callsign", pa.string(), nullable=True), _field("report_timestamp_utc", pa.timestamp("us", tz="UTC")),
        _field("lat_deg", pa.float64(), nullable=True), _field("lon_deg", pa.float64(), nullable=True),
        _field("altitude_m", pa.float64(), nullable=True), _field("speed_mps", pa.float64(), nullable=True),
        _field("heading_rad", pa.float64(), nullable=True), _field("age_s", pa.float64(), nullable=True),
        _field("range_m", pa.float64(), nullable=True), _field("bearing_rad", pa.float64(), nullable=True),
        _field("altitude_source_gnss", pa.bool_(), nullable=True), _field("raw_schema_version", pa.string()),
        _field("parse_quality_bits", pa.uint64()), _field("qc_bits", pa.uint64()),
        _field("raw_altitude_ft", pa.float64(), nullable=True), _field("raw_speed_knots", pa.float64(), nullable=True),
        _field("raw_heading_deg", pa.float64(), nullable=True), _field("raw_lat_deg", pa.float64(), nullable=True),
        _field("raw_lon_deg", pa.float64(), nullable=True), _field("raw_age_s", pa.float64(), nullable=True),
        _field("raw_range_km", pa.float64(), nullable=True), _field("raw_bearing_deg", pa.float64(), nullable=True),
        _field("raw_tail", pa.string(), nullable=True), _field("raw_altisgnss", pa.bool_(), nullable=True),
        _field("partition_year", pa.int32()), _field("partition_month", pa.int8()),
    ])


def unique_reports_schema() -> pa.Schema:
    return pa.schema([
        _field("canonical_report_id", pa.string()), _field("selected_raw_row_id", pa.string()),
        _field("airport_id", pa.string()), _field("aircraft_id", pa.string()),
        _field("report_timestamp_utc", pa.timestamp("us", tz="UTC")), _field("tail_or_callsign", pa.string(), nullable=True),
        _field("lat_deg", pa.float64(), nullable=True), _field("lon_deg", pa.float64(), nullable=True),
        _field("altitude_m", pa.float64(), nullable=True), _field("speed_mps", pa.float64(), nullable=True),
        _field("heading_rad", pa.float64(), nullable=True), _field("age_s", pa.float64(), nullable=True),
        _field("range_m", pa.float64(), nullable=True), _field("bearing_rad", pa.float64(), nullable=True),
        _field("altitude_source_gnss", pa.bool_(), nullable=True), _field("group_size", pa.int64()),
        _field("is_duplicate", pa.bool_()), _field("duplicate_classification", pa.string()),
        _field("selection_policy_version", pa.string()), _field("core_state_difference_bits", pa.uint8()),
        _field("metadata_difference_bits", pa.uint8()), _field("duplicate_conflict_flag", pa.bool_()),
        _field("metadata_conflict_flag", pa.bool_()), _field("any_difference_flag", pa.bool_()),
        _field("parse_quality_bits", pa.uint64()), _field("qc_bits", pa.uint64()),
        _field("partition_year", pa.int32()), _field("partition_month", pa.int8()),
    ])


def membership_schema() -> pa.Schema:
    return pa.schema([
        _field("raw_row_id", pa.string()), _field("canonical_report_id", pa.string()),
        _field("group_size", pa.int64()), _field("dedup_rank", pa.int64()), _field("selected", pa.bool_()),
        _field("selection_reason", pa.string()), _field("age_rank_class", pa.int8()),
        _field("age_rank_value", pa.float64()), _field("completeness_score", pa.int8()),
        _field("completeness_denominator", pa.int8()), _field("core_state_difference_bits", pa.uint8()),
        _field("metadata_difference_bits", pa.uint8()), _field("airport_id", pa.string()),
        _field("source_file_id", pa.string()), _field("source_relative_path", pa.string()),
        _field("archive_member", pa.string(), nullable=True), _field("source_content_sha256", pa.string()),
        _field("logical_record_ordinal", pa.int64()), _field("source_row_number", pa.int64()),
        _field("source_sequence", pa.int64()), _field("partition_year", pa.int32()), _field("partition_month", pa.int8()),
    ])
