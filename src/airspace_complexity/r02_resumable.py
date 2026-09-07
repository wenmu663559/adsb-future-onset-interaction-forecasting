"""Fail-closed, bucket-checkpointed aggregation for exact R02 audits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from airspace_complexity import external_duplicate_audit, profile_stats, r02_runner
from airspace_complexity.external_duplicate_audit import summarize_rows
from airspace_complexity.profile_stats import FieldProfile
from airspace_complexity.r02_runner import AuditRunManifest, PartitionManifest


AGGREGATION_SCHEMA_VERSION = "r02_resumable_aggregate_v4"
FRAGMENT_SCHEMA_VERSION = "r02_bucket_fragment_v4"
CHECKPOINT_SCHEMA_VERSION = "r02_resumable_checkpoint_v1"
PROFILE_STATE_SCHEMA_VERSION = "r02_profile_state_v1"
_ATOMIC_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _module_path(module: object) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise ValueError(f"module has no source path: {module!r}")
    return Path(value).resolve()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_fields(
    name: str, record: Mapping[str, Any], expected: set[str]
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(record) != expected:
        raise ValueError(f"{name} fields mismatch")


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt, delay in enumerate((0.0, *_ATOMIC_REPLACE_RETRY_DELAYS)):
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(_ATOMIC_REPLACE_RETRY_DELAYS):
                raise


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class AggregationIdentity:
    """Immutable partition and implementation identity for fragment reuse."""

    schema_version: str
    partition_manifest_sha256: str
    partition_identity: Mapping[str, Any]
    implementation_hashes: Mapping[str, str]

    @classmethod
    def from_partition(cls, partition_path: Path) -> "AggregationIdentity":
        path = Path(partition_path).resolve()
        partition = PartitionManifest.from_path(path)
        implementation_paths = {
            "r02_resumable": Path(__file__).resolve(),
            "profile_stats": _module_path(profile_stats),
            "external_duplicate_audit": _module_path(external_duplicate_audit),
            "r02_runner": _module_path(r02_runner),
        }
        return cls(
            schema_version=AGGREGATION_SCHEMA_VERSION,
            partition_manifest_sha256=_sha256_file(path),
            partition_identity=partition.identity.to_dict(),
            implementation_hashes={
                name: _sha256_file(source)
                for name, source in sorted(implementation_paths.items())
            },
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "AggregationIdentity":
        expected = {
            "schema_version",
            "partition_manifest_sha256",
            "partition_identity",
            "implementation_hashes",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise ValueError("aggregation identity fields mismatch")
        if record["schema_version"] != AGGREGATION_SCHEMA_VERSION:
            raise ValueError("unsupported aggregation identity schema")
        return cls(
            schema_version=str(record["schema_version"]),
            partition_manifest_sha256=str(
                record["partition_manifest_sha256"]
            ),
            partition_identity=dict(record["partition_identity"]),
            implementation_hashes={
                str(name): str(value)
                for name, value in dict(
                    record["implementation_hashes"]
                ).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "partition_manifest_sha256": self.partition_manifest_sha256,
            "partition_identity": dict(self.partition_identity),
            "implementation_hashes": dict(
                sorted(self.implementation_hashes.items())
            ),
        }


@dataclass(frozen=True)
class BucketFragmentManifest:
    schema_version: str
    aggregation_identity_sha256: str
    airport: str
    bucket_key: str
    input_bucket_sha256: str
    input_row_count: int
    valid_timestamp_rows: int
    artifact_hashes: Mapping[str, str]
    completed_at_utc: str

    @classmethod
    def from_dict(
        cls, record: Mapping[str, Any]
    ) -> "BucketFragmentManifest":
        _require_exact_fields(
            "fragment manifest",
            record,
            {
                "schema_version",
                "aggregation_identity_sha256",
                "airport",
                "bucket_key",
                "input_bucket_sha256",
                "input_row_count",
                "valid_timestamp_rows",
                "artifact_hashes",
                "completed_at_utc",
            },
        )
        if record["schema_version"] != FRAGMENT_SCHEMA_VERSION:
            raise ValueError("unsupported fragment manifest schema")
        airport = str(record["airport"])
        if airport not in {"kagc", "kbtp"}:
            raise ValueError("invalid fragment airport")
        bucket_key = str(record["bucket_key"])
        if not bucket_key.isdigit() or str(int(bucket_key)) != bucket_key:
            raise ValueError("invalid fragment bucket key")
        input_rows = _strict_int(
            "input_row_count", record["input_row_count"]
        )
        valid_rows = _strict_int(
            "valid_timestamp_rows", record["valid_timestamp_rows"]
        )
        if valid_rows > input_rows:
            raise ValueError("valid timestamp rows exceed input rows")
        hashes = dict(record["artifact_hashes"])
        expected_artifacts = {
            "duplicate_records.jsonl",
            "profile_state.json",
        }
        if set(hashes) != expected_artifacts:
            raise ValueError("fragment artifact set mismatch")
        return cls(
            schema_version=str(record["schema_version"]),
            aggregation_identity_sha256=str(
                record["aggregation_identity_sha256"]
            ),
            airport=airport,
            bucket_key=bucket_key,
            input_bucket_sha256=str(record["input_bucket_sha256"]),
            input_row_count=input_rows,
            valid_timestamp_rows=valid_rows,
            artifact_hashes={str(name): str(value) for name, value in hashes.items()},
            completed_at_utc=str(record["completed_at_utc"]),
        )

    @classmethod
    def from_path(cls, path: Path) -> "BucketFragmentManifest":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "aggregation_identity_sha256": self.aggregation_identity_sha256,
            "airport": self.airport,
            "bucket_key": self.bucket_key,
            "input_bucket_sha256": self.input_bucket_sha256,
            "input_row_count": self.input_row_count,
            "valid_timestamp_rows": self.valid_timestamp_rows,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "completed_at_utc": self.completed_at_utc,
        }


@dataclass(frozen=True)
class AggregateCheckpoint:
    schema_version: str
    identity: AggregationIdentity
    airport: str
    bucket_count: int
    completed_buckets: Mapping[str, str]
    observed_rows: int
    nonempty_bucket_count: int
    started_at_utc: str
    updated_at_utc: str

    @classmethod
    def from_dict(
        cls,
        record: Mapping[str, Any],
        *,
        expected_identity: AggregationIdentity | None = None,
    ) -> "AggregateCheckpoint":
        _require_exact_fields(
            "checkpoint",
            record,
            {
                "schema_version",
                "identity",
                "airport",
                "bucket_count",
                "completed_buckets",
                "observed_rows",
                "nonempty_bucket_count",
                "started_at_utc",
                "updated_at_utc",
            },
        )
        if record["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        identity = AggregationIdentity.from_dict(record["identity"])
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("checkpoint identity mismatch")
        airport = str(record["airport"])
        if airport not in {"kagc", "kbtp"}:
            raise ValueError("invalid checkpoint airport")
        bucket_count = _strict_int(
            "bucket_count", record["bucket_count"], minimum=1
        )
        completed = {
            str(key): str(value)
            for key, value in dict(record["completed_buckets"]).items()
        }
        for key, value in completed.items():
            if (
                not key.isdigit()
                or str(int(key)) != key
                or int(key) >= bucket_count
            ):
                raise ValueError("invalid completed bucket key")
            if value != "EMPTY" and len(value) != 64:
                raise ValueError("invalid completed fragment hash")
        observed_rows = _strict_int(
            "observed_rows", record["observed_rows"]
        )
        nonempty = _strict_int(
            "nonempty_bucket_count", record["nonempty_bucket_count"]
        )
        if nonempty > len(completed):
            raise ValueError("nonempty bucket count exceeds completed buckets")
        return cls(
            schema_version=str(record["schema_version"]),
            identity=identity,
            airport=airport,
            bucket_count=bucket_count,
            completed_buckets=dict(
                sorted(completed.items(), key=lambda item: int(item[0]))
            ),
            observed_rows=observed_rows,
            nonempty_bucket_count=nonempty,
            started_at_utc=str(record["started_at_utc"]),
            updated_at_utc=str(record["updated_at_utc"]),
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_identity: AggregationIdentity | None = None,
    ) -> "AggregateCheckpoint":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")),
            expected_identity=expected_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "airport": self.airport,
            "bucket_count": self.bucket_count,
            "completed_buckets": dict(self.completed_buckets),
            "observed_rows": self.observed_rows,
            "nonempty_bucket_count": self.nonempty_bucket_count,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


def validate_fragment(
    fragment_dir: Path,
    identity: AggregationIdentity,
    bucket_key: str,
    expected_hash: str,
    expected_rows: int,
) -> BucketFragmentManifest:
    """Verify that one completed fragment is safe to reuse."""
    bucket = str(bucket_key)
    fragment = Path(fragment_dir).resolve()
    try:
        if fragment.name != f"bucket-{int(bucket):03d}":
            raise ValueError("fragment directory name mismatch")
        expected_files = {
            "duplicate_records.jsonl",
            "profile_state.json",
            "fragment_manifest.json",
        }
        actual_files = {
            path.name for path in fragment.iterdir() if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("fragment file set mismatch")
        manifest = BucketFragmentManifest.from_path(
            fragment / "fragment_manifest.json"
        )
        if manifest.aggregation_identity_sha256 != _canonical_sha256(
            identity.to_dict()
        ):
            raise ValueError("aggregation identity mismatch")
        if manifest.bucket_key != bucket:
            raise ValueError("fragment bucket key mismatch")
        if manifest.input_bucket_sha256 != expected_hash:
            raise ValueError("fragment input hash mismatch")
        if manifest.input_row_count != expected_rows:
            raise ValueError("fragment input row count mismatch")
        for name, expected in manifest.artifact_hashes.items():
            if _sha256_file(fragment / name) != expected:
                raise ValueError(f"fragment artifact hash mismatch: {name}")
        with (fragment / "duplicate_records.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"duplicate record {line_number} is not an object"
                    )
                if value.get("summary_schema_version") != (
                    external_duplicate_audit.DUPLICATE_SUMMARY_SCHEMA_VERSION
                ):
                    raise ValueError(
                        f"duplicate record {line_number} summary schema mismatch"
                    )
                if value.get("summary_kind") not in {"duplicate", "age"}:
                    raise ValueError(
                        f"duplicate record {line_number} summary kind mismatch"
                    )
        profile_state = json.loads(
            (fragment / "profile_state.json").read_text(encoding="utf-8")
        )
        _require_exact_fields(
            "profile state",
            profile_state,
            {"schema_version", "profiles"},
        )
        if profile_state["schema_version"] != PROFILE_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported profile state schema")
        if not isinstance(profile_state["profiles"], list):
            raise TypeError("profile state profiles must be a list")
        for profile in profile_state["profiles"]:
            _require_exact_fields(
                "profile state record",
                profile,
                {"airport_id", "year", "schema_version", "field", "state"},
            )
            FieldProfile.from_state(profile["state"])
        return manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"bucket {bucket} fragment validation failed: {exc}") from exc


def _profile_sort_key(
    key: tuple[str, int | None, str, str]
) -> tuple[str, int, str, str]:
    airport, year, schema, field = key
    return airport, -1 if year is None else year, schema, field


def _fragment_profiles(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int | None, str, str], FieldProfile]:
    profiles: dict[tuple[str, int | None, str, str], FieldProfile] = {}
    for row in rows:
        airport = str(row["airport_id"])
        year = int(row["year"]) if row.get("year") is not None else None
        schema_version = str(row["schema_version"])
        identity = (
            str(row["source_file_id"]),
            int(row["source_row_number"]),
        )
        for audit_field, raw_field in r02_runner._PROFILE_FIELDS.items():
            error = r02_runner._profile_error(
                row, audit_field, raw_field
            )
            value = row.get(audit_field)
            profile_key = (airport, year, schema_version, audit_field)
            profile = profiles.setdefault(profile_key, FieldProfile())
            profile.add((*identity, audit_field), value, error)
    return profiles


def _with_airport_total_profiles(
    profiles: Mapping[
        tuple[str, int | None, str, str], FieldProfile
    ],
) -> dict[tuple[str, int | None, str, str], FieldProfile]:
    combined = dict(profiles)
    totals: dict[tuple[str, int | None, str, str], FieldProfile] = {}
    for (airport, year, schema, field), profile in profiles.items():
        if year is None and schema == "ALL":
            raise ValueError("fragment profile state contains airport total")
        total_key = (airport, None, "ALL", field)
        totals.setdefault(total_key, FieldProfile()).merge(profile)
    combined.update(totals)
    return combined


def _table_to_rows(table: pa.Table) -> list[dict[str, Any]]:
    """Convert audit rows without PyArrow's slow timezone scalar path."""
    timestamp_name = "report_timestamp_utc"
    if timestamp_name not in table.column_names:
        raise ValueError("audit table is missing report_timestamp_utc")
    timestamp_micros = table[timestamp_name].cast(pa.int64()).to_pylist()
    rows = table.drop([timestamp_name]).to_pylist()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for row, micros in zip(rows, timestamp_micros):
        row[timestamp_name] = (
            None
            if micros is None
            else epoch + timedelta(microseconds=int(micros))
        )
    return rows


def process_bucket(
    partition: PartitionManifest,
    bucket_key: str,
    identity: AggregationIdentity,
    fragment_root: Path,
) -> BucketFragmentManifest:
    """Compute and atomically publish one nonempty bucket fragment."""
    key = str(bucket_key)
    if key not in partition.bucket_files:
        raise ValueError(f"unknown bucket key: {key}")
    expected_hash = partition.bucket_hashes[key]
    expected_rows = int(partition.bucket_row_counts[key])
    if expected_rows <= 0:
        raise ValueError(f"bucket {key} is empty")
    bucket_path = Path(partition.bucket_files[key])
    if _sha256_file(bucket_path) != expected_hash:
        raise ValueError(f"bucket {key} input hash mismatch")
    actual_rows = pq.ParquetFile(bucket_path).metadata.num_rows
    if actual_rows != expected_rows:
        raise ValueError(f"bucket {key} input row count mismatch")

    root = Path(fragment_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"bucket-{int(key):03d}"
    if final.exists():
        return validate_fragment(
            final, identity, key, expected_hash, expected_rows
        )
    temporary = root / f".bucket-{int(key):03d}-{uuid.uuid4().hex[:8]}.tmp"
    temporary.mkdir()
    duplicate_path = temporary / "duplicate_records.jsonl"
    profile_path = temporary / "profile_state.json"
    rows = _table_to_rows(pq.read_table(bucket_path))
    valid_rows = [
        row for row in rows if row.get("report_timestamp_utc") is not None
    ]
    with duplicate_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in summarize_rows(valid_rows).to_records():
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    profiles = _fragment_profiles(rows)
    _atomic_json(
        profile_path,
        {
            "schema_version": PROFILE_STATE_SCHEMA_VERSION,
            "profiles": [
                {
                    "airport_id": profile_key[0],
                    "year": profile_key[1],
                    "schema_version": profile_key[2],
                    "field": profile_key[3],
                    "state": profiles[profile_key].to_state(),
                }
                for profile_key in sorted(profiles, key=_profile_sort_key)
            ],
        },
    )
    manifest = BucketFragmentManifest(
        schema_version=FRAGMENT_SCHEMA_VERSION,
        aggregation_identity_sha256=_canonical_sha256(identity.to_dict()),
        airport=partition.airport,
        bucket_key=key,
        input_bucket_sha256=expected_hash,
        input_row_count=expected_rows,
        valid_timestamp_rows=len(valid_rows),
        artifact_hashes={
            duplicate_path.name: _sha256_file(duplicate_path),
            profile_path.name: _sha256_file(profile_path),
        },
        completed_at_utc=_utc_now(),
    )
    _atomic_json(temporary / "fragment_manifest.json", manifest.to_dict())
    _replace_with_retry(temporary, final)
    return validate_fragment(
        final, identity, key, expected_hash, expected_rows
    )


def _after_bucket_checkpoint(bucket_key: str) -> None:
    """Test seam invoked only after a bucket checkpoint is durable."""


def _new_checkpoint(
    partition: PartitionManifest, identity: AggregationIdentity
) -> AggregateCheckpoint:
    now = _utc_now()
    return AggregateCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        identity=identity,
        airport=partition.airport,
        bucket_count=partition.bucket_count,
        completed_buckets={},
        observed_rows=0,
        nonempty_bucket_count=0,
        started_at_utc=now,
        updated_at_utc=now,
    )


def _updated_checkpoint(
    checkpoint: AggregateCheckpoint,
    completed: Mapping[str, str],
    partition: PartitionManifest,
) -> AggregateCheckpoint:
    keys = set(completed)
    return AggregateCheckpoint(
        schema_version=checkpoint.schema_version,
        identity=checkpoint.identity,
        airport=checkpoint.airport,
        bucket_count=checkpoint.bucket_count,
        completed_buckets=dict(
            sorted(completed.items(), key=lambda item: int(item[0]))
        ),
        observed_rows=sum(
            int(partition.bucket_row_counts[key]) for key in keys
        ),
        nonempty_bucket_count=sum(
            int(int(partition.bucket_row_counts[key]) > 0) for key in keys
        ),
        started_at_utc=checkpoint.started_at_utc,
        updated_at_utc=_utc_now(),
    )


def _verify_partition_inputs(partition: PartitionManifest) -> None:
    expected = {str(index) for index in range(partition.bucket_count)}
    for name, mapping in (
        ("bucket_files", partition.bucket_files),
        ("bucket_hashes", partition.bucket_hashes),
        ("bucket_row_counts", partition.bucket_row_counts),
    ):
        if set(mapping) != expected:
            raise ValueError(f"partition {name} keys mismatch")
    observed = 0
    for key in sorted(expected, key=int):
        path = Path(partition.bucket_files[key])
        if _sha256_file(path) != partition.bucket_hashes[key]:
            raise ValueError(f"bucket {key} input hash mismatch")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != int(partition.bucket_row_counts[key]):
            raise ValueError(f"bucket {key} input row count mismatch")
        observed += rows
    if observed != partition.total_rows:
        raise ValueError("partition row conservation failure")


def resume_fragments(
    partition: PartitionManifest,
    identity: AggregationIdentity,
    namespace: Path,
) -> tuple[AggregateCheckpoint, int, int]:
    """Verify existing progress and process every unfinished bucket."""
    _verify_partition_inputs(partition)
    root = Path(namespace).resolve()
    fragments = root / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "aggregate_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = AggregateCheckpoint.from_path(
            checkpoint_path, expected_identity=identity
        )
        if (
            checkpoint.airport != partition.airport
            or checkpoint.bucket_count != partition.bucket_count
        ):
            raise ValueError("checkpoint partition contract mismatch")
    else:
        checkpoint = _new_checkpoint(partition, identity)

    completed = dict(checkpoint.completed_buckets)
    for key, manifest_hash in list(completed.items()):
        rows = int(partition.bucket_row_counts[key])
        if rows == 0:
            if manifest_hash != "EMPTY":
                raise ValueError(f"bucket {key} empty marker mismatch")
            continue
        fragment = fragments / f"bucket-{int(key):03d}"
        validate_fragment(
            fragment,
            identity,
            key,
            partition.bucket_hashes[key],
            rows,
        )
        if _sha256_file(fragment / "fragment_manifest.json") != manifest_hash:
            raise ValueError(f"bucket {key} fragment manifest hash mismatch")

    for fragment in sorted(fragments.glob("bucket-*")):
        if not fragment.is_dir():
            continue
        suffix = fragment.name.removeprefix("bucket-")
        if not suffix.isdigit():
            raise ValueError(f"unknown completed fragment directory: {fragment}")
        key = str(int(suffix))
        if key in completed:
            continue
        if key not in partition.bucket_files:
            raise ValueError(f"orphan fragment bucket out of range: {key}")
        rows = int(partition.bucket_row_counts[key])
        if rows == 0:
            raise ValueError(f"empty bucket {key} has an orphan fragment")
        validate_fragment(
            fragment,
            identity,
            key,
            partition.bucket_hashes[key],
            rows,
        )
        completed[key] = _sha256_file(fragment / "fragment_manifest.json")

    reused = len(completed)
    checkpoint = _updated_checkpoint(checkpoint, completed, partition)
    if completed or checkpoint_path.exists():
        _atomic_json(checkpoint_path, checkpoint.to_dict())

    processed = 0
    for key in sorted(partition.bucket_files, key=int):
        if key in completed:
            continue
        rows = int(partition.bucket_row_counts[key])
        if rows == 0:
            completed[key] = "EMPTY"
        else:
            fragment = process_bucket(
                partition, key, identity, fragments
            )
            completed[key] = _sha256_file(
                fragments
                / f"bucket-{int(key):03d}"
                / "fragment_manifest.json"
            )
            if fragment.input_row_count != rows:
                raise AssertionError("validated fragment row count changed")
        processed += 1
        checkpoint = _updated_checkpoint(checkpoint, completed, partition)
        _atomic_json(checkpoint_path, checkpoint.to_dict())
        _after_bucket_checkpoint(key)
    return checkpoint, reused, processed


def _load_fragment_profile_state(
    path: Path,
) -> dict[tuple[str, int | None, str, str], FieldProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_exact_fields(
        "profile state", payload, {"schema_version", "profiles"}
    )
    if payload["schema_version"] != PROFILE_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported profile state schema")
    profiles: dict[tuple[str, int | None, str, str], FieldProfile] = {}
    for record in payload["profiles"]:
        _require_exact_fields(
            "profile state record",
            record,
            {"airport_id", "year", "schema_version", "field", "state"},
        )
        key = (
            str(record["airport_id"]),
            None if record["year"] is None else int(record["year"]),
            str(record["schema_version"]),
            str(record["field"]),
        )
        if key in profiles:
            raise ValueError(f"duplicate profile state key: {key}")
        profiles[key] = FieldProfile.from_state(record["state"])
    return profiles


def _verify_existing_publication(
    output: Path,
    identity: AggregationIdentity,
) -> AuditRunManifest:
    pointer = output / "run_manifest.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    run = AuditRunManifest.from_dict(payload)
    resumption = dict(payload.get("resumption", {}))
    if resumption.get("aggregation_identity") != identity.to_dict():
        raise ValueError("published aggregation identity mismatch")
    if not run.invariants or not all(bool(value) for value in run.invariants.values()):
        raise ValueError("published aggregate invariant failure")
    required = {
        "column_profile",
        "duplicate_records",
        "invariants",
        "output_hashes",
    }
    if set(run.artifact_files) != required or set(run.output_hashes) != required:
        raise ValueError("published aggregate artifact set mismatch")
    for name, path_text in run.artifact_files.items():
        if _sha256_file(Path(path_text)) != run.output_hashes[name]:
            raise ValueError(f"published aggregate hash mismatch: {name}")
    return run


def aggregate_buckets_resumable(
    partition_manifest_path: Path,
    output_dir: Path,
) -> AuditRunManifest:
    """Resume bucket fragments and atomically publish one airport audit."""
    started_at = _utc_now()
    partition_path = Path(partition_manifest_path).resolve()
    partition = PartitionManifest.from_path(partition_path)
    identity = AggregationIdentity.from_partition(partition_path)
    output = Path(output_dir).resolve()
    r02_runner._validate_output_dir(
        Path(partition.project_root).resolve(), output
    )
    if (output / "run_manifest.json").exists():
        return _verify_existing_publication(output, identity)

    namespace = output / "resumable-aggregate-v4"
    checkpoint, reused, processed = resume_fragments(
        partition, identity, namespace
    )
    expected_keys = {str(index) for index in range(partition.bucket_count)}
    if set(checkpoint.completed_buckets) != expected_keys:
        raise RuntimeError("resumable checkpoint is incomplete")

    generations = output / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    staging = output / f".generation-{generation_id}.tmp"
    final_generation = generations / generation_id
    staging.mkdir()
    duplicate_staged = staging / "duplicate_records.jsonl"
    profiles: dict[tuple[str, int | None, str, str], FieldProfile] = {}
    with duplicate_staged.open(
        "wb"
    ) as destination:
        for key in sorted(expected_keys, key=int):
            if int(partition.bucket_row_counts[key]) == 0:
                continue
            fragment = (
                namespace
                / "fragments"
                / f"bucket-{int(key):03d}"
            )
            validate_fragment(
                fragment,
                identity,
                key,
                partition.bucket_hashes[key],
                int(partition.bucket_row_counts[key]),
            )
            with (fragment / "duplicate_records.jsonl").open("rb") as source:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            for profile_key, incoming in _load_fragment_profile_state(
                fragment / "profile_state.json"
            ).items():
                current = profiles.get(profile_key)
                if current is None:
                    profiles[profile_key] = incoming
                else:
                    current.merge(incoming)
        destination.flush()
        os.fsync(destination.fileno())

    profiles = _with_airport_total_profiles(profiles)
    source_sequences = [
        int(source["source_sequence"]) for source in partition.sources
    ]
    source_ids = [str(source["source_file_id"]) for source in partition.sources]
    actual_bucket_counts = {
        key: pq.ParquetFile(path).metadata.num_rows
        for key, path in partition.bucket_files.items()
    }
    invariants = {
        "all_bucket_hashes_verified": True,
        "all_sources_completed": (
            source_sequences == list(range(len(source_sequences)))
            and len(source_ids) == len(set(source_ids))
            and sum(int(source["row_count"]) for source in partition.sources)
            == partition.total_rows
        ),
        "analyzed_nonempty_bucket_count_matches": (
            checkpoint.nonempty_bucket_count
            == sum(int(value > 0) for value in actual_bucket_counts.values())
        ),
        "bucket_row_counts_match": (
            actual_bucket_counts
            == {
                str(key): int(value)
                for key, value in partition.bucket_row_counts.items()
            }
        ),
        "partition_row_conservation": (
            partition.total_rows
            == sum(partition.bucket_row_counts.values())
            == checkpoint.observed_rows
        ),
    }
    if not all(invariants.values()):
        failed = ", ".join(name for name, value in invariants.items() if not value)
        raise RuntimeError(f"aggregate invariant failure: {failed}")

    column_staged = staging / "column_profile.parquet"
    invariants_staged = staging / "invariants.json"
    hashes_staged = staging / "output_hashes.json"
    r02_runner._atomic_parquet(
        column_staged,
        pa.Table.from_pylist(r02_runner._profile_records(profiles)),
    )
    _atomic_json(invariants_staged, invariants)
    staged_artifacts = {
        "column_profile": column_staged,
        "duplicate_records": duplicate_staged,
        "invariants": invariants_staged,
    }
    output_hashes = {
        name: _sha256_file(path) for name, path in staged_artifacts.items()
    }
    _atomic_json(hashes_staged, output_hashes)
    staged_artifacts["output_hashes"] = hashes_staged
    output_hashes["output_hashes"] = _sha256_file(hashes_staged)
    artifact_files = {
        name: str(final_generation / path.name)
        for name, path in staged_artifacts.items()
    }
    commands = {
        "aggregate": (
            "python scripts/run_r02_resumable_aggregate.py "
            f"--airport {partition.airport} "
            f'--work-dir "{Path(partition.work_dir).parent}" '
            f'--output-dir "{output}" '
            f"--bucket-count {partition.bucket_count}"
        ),
        "verify_tests": (
            "python -m pytest tests/test_r02_resumable.py "
            "tests/test_r02_runner.py -q"
        ),
    }
    run = AuditRunManifest(
        identity=partition.identity,
        airport=partition.airport,
        bucket_count=partition.bucket_count,
        schema_version=partition.schema_version,
        output_dir=str(output),
        artifact_files=artifact_files,
        output_hashes=output_hashes,
        invariants=invariants,
        commands=commands,
        analyzed_bucket_count=checkpoint.nonempty_bucket_count,
        started_at_utc=started_at,
        ended_at_utc=_utc_now(),
    )
    checkpoint_path = namespace / "aggregate_checkpoint.json"
    pointer_payload = {
        **run.to_dict(),
        "resumption": {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "aggregation_identity": identity.to_dict(),
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "completed_bucket_count": len(checkpoint.completed_buckets),
            "reused_bucket_count": reused,
            "newly_processed_bucket_count": processed,
            "fragment_root": str(namespace / "fragments"),
        },
    }
    _replace_with_retry(staging, final_generation)
    _atomic_json(output / "run_manifest.json", pointer_payload)
    return run
