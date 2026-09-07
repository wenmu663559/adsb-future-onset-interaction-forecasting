"""Resumable, hash-guarded partition and aggregation for the R02 audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, TextIO

import pyarrow
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import zipfile_deflate64  # noqa: F401  # Registers ZIP compression method 9.

from airspace_complexity.external_duplicate_audit import analyze_rows, stable_bucket
from airspace_complexity.profile_stats import FieldProfile
from airspace_complexity.r02_audit import EXCLUDED_DISPOSITIONS
from airspace_complexity.raw_parser import parse_raw_row


PARTITION_SCHEMA_VERSION = "r02_partition_v1"
ROW_GROUP_ROWS = 4096
_CSV_FIELD_SIZE_LIMIT = 64 * 1024 * 1024
_ATOMIC_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_IDENTITY_FILES = {
    "raw_manifest": Path("data/manifests/raw_file_manifest.parquet"),
    "archive_comparison": Path("outputs/r01/archive_comparison.csv"),
    "raw_schema": Path("configs/data/raw_schema.yaml"),
    "parser_source": Path("src/airspace_complexity/raw_parser.py"),
    "runner_source": Path("src/airspace_complexity/r02_runner.py"),
    "requirements_lock": Path("requirements-lock.txt"),
}
_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("airport_id", pa.string(), nullable=False),
        pa.field("aircraft_id", pa.string(), nullable=False),
        pa.field("report_timestamp_utc", pa.timestamp("us", tz="+00:00")),
        pa.field("timestamp_assuming_airport_local", pa.string()),
        pa.field("local_candidate_ambiguous", pa.bool_(), nullable=False),
        pa.field("local_candidate_nonexistent", pa.bool_(), nullable=False),
        pa.field("timestamp_parse_status", pa.string(), nullable=False),
        pa.field("timestamp_parse_error", pa.string()),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("altitude", pa.float64()),
        pa.field("speed", pa.float64()),
        pa.field("heading", pa.float64()),
        pa.field("age", pa.float64()),
        pa.field("range", pa.float64()),
        pa.field("bearing", pa.float64()),
        pa.field("tail", pa.string()),
        pa.field("altis_gnss", pa.bool_()),
        pa.field("source_file_id", pa.string(), nullable=False),
        pa.field("source_relative_path", pa.string(), nullable=False),
        pa.field("archive_member", pa.string()),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("source_sequence", pa.int64(), nullable=False),
        pa.field("r01_disposition", pa.string(), nullable=False),
        pa.field("year", pa.int32()),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("field_errors_json", pa.string(), nullable=False),
    ]
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True)
class RunIdentity:
    """Code, input and environment identity that guards checkpoint reuse."""

    git_commit: str
    git_dirty: bool
    hashes: Mapping[str, str]
    versions: Mapping[str, str]

    @classmethod
    def from_project(cls, project_root: Path) -> "RunIdentity":
        root = Path(project_root).resolve()
        hashes = {
            name: _sha256_file(root / relative)
            for name, relative in _IDENTITY_FILES.items()
        }
        return cls(
            git_commit=_git_output(root, "rev-parse", "HEAD"),
            git_dirty=bool(
                _git_output(root, "status", "--porcelain", "--untracked-files=normal")
            ),
            hashes=hashes,
            versions={
                "python": sys.version,
                "pyarrow": pyarrow.__version__,
                "pyyaml": yaml.__version__,
                "zipfile_deflate64": _distribution_version("zipfile-deflate64"),
            },
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RunIdentity":
        return cls(
            git_commit=str(record["git_commit"]),
            git_dirty=bool(record["git_dirty"]),
            hashes={
                str(key): str(value)
                for key, value in dict(record["hashes"]).items()
            },
            versions={
                str(key): str(value)
                for key, value in dict(record["versions"]).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "hashes": dict(sorted(self.hashes.items())),
            "versions": dict(sorted(self.versions.items())),
        }


@dataclass(frozen=True)
class PartitionManifest:
    """Finalized partition files and the identity that authorizes their reuse."""

    identity: RunIdentity
    airport: str
    bucket_count: int
    schema_version: str
    work_dir: str
    sources: tuple[Mapping[str, Any], ...]
    total_rows: int
    bucket_files: Mapping[str, str]
    bucket_hashes: Mapping[str, str]
    bucket_row_counts: Mapping[str, int]
    started_at_utc: str
    ended_at_utc: str
    project_root: str

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PartitionManifest":
        if not record.get("project_root"):
            raise ValueError(
                "legacy partition manifest missing project_root; "
                "regenerate partition"
            )
        return cls(
            identity=RunIdentity.from_dict(record["identity"]),
            airport=str(record["airport"]),
            bucket_count=int(record["bucket_count"]),
            schema_version=str(record["schema_version"]),
            work_dir=str(record["work_dir"]),
            sources=tuple(dict(item) for item in record["sources"]),
            total_rows=int(record["total_rows"]),
            bucket_files={
                str(key): str(value)
                for key, value in dict(record["bucket_files"]).items()
            },
            bucket_hashes={
                str(key): str(value)
                for key, value in dict(record["bucket_hashes"]).items()
            },
            bucket_row_counts={
                str(key): int(value)
                for key, value in dict(record["bucket_row_counts"]).items()
            },
            started_at_utc=str(record["started_at_utc"]),
            ended_at_utc=str(record["ended_at_utc"]),
            project_root=str(record["project_root"]),
        )

    @classmethod
    def from_path(cls, path: Path) -> "PartitionManifest":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "airport": self.airport,
            "bucket_count": self.bucket_count,
            "schema_version": self.schema_version,
            "work_dir": self.work_dir,
            "sources": [dict(item) for item in self.sources],
            "total_rows": self.total_rows,
            "bucket_files": dict(sorted(self.bucket_files.items())),
            "bucket_hashes": dict(sorted(self.bucket_hashes.items())),
            "bucket_row_counts": dict(sorted(self.bucket_row_counts.items())),
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "project_root": self.project_root,
        }


@dataclass(frozen=True)
class AuditRunManifest:
    """Atomic aggregate artifacts and all facts needed to reproduce them."""

    identity: RunIdentity
    airport: str
    bucket_count: int
    schema_version: str
    output_dir: str
    artifact_files: Mapping[str, str]
    output_hashes: Mapping[str, str]
    invariants: Mapping[str, bool]
    commands: Mapping[str, str]
    analyzed_bucket_count: int
    started_at_utc: str
    ended_at_utc: str

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "AuditRunManifest":
        return cls(
            identity=RunIdentity.from_dict(record["identity"]),
            airport=str(record["airport"]),
            bucket_count=int(record["bucket_count"]),
            schema_version=str(record["schema_version"]),
            output_dir=str(record["output_dir"]),
            artifact_files={
                str(key): str(value)
                for key, value in dict(record["artifact_files"]).items()
            },
            output_hashes={
                str(key): str(value)
                for key, value in dict(record["output_hashes"]).items()
            },
            invariants={
                str(key): bool(value)
                for key, value in dict(record["invariants"]).items()
            },
            commands={
                str(key): str(value)
                for key, value in dict(record["commands"]).items()
            },
            analyzed_bucket_count=int(record["analyzed_bucket_count"]),
            started_at_utc=str(record["started_at_utc"]),
            ended_at_utc=str(record["ended_at_utc"]),
        )

    @classmethod
    def from_path(cls, path: Path) -> "AuditRunManifest":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "airport": self.airport,
            "bucket_count": self.bucket_count,
            "schema_version": self.schema_version,
            "output_dir": self.output_dir,
            "artifact_files": dict(sorted(self.artifact_files.items())),
            "output_hashes": dict(sorted(self.output_hashes.items())),
            "invariants": dict(sorted(self.invariants.items())),
            "commands": dict(sorted(self.commands.items())),
            "analyzed_bucket_count": self.analyzed_bucket_count,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
        }


def source_file_id(airport_id: str, relative_path: str) -> str:
    """Return the globally stable source identity required by R02."""
    payload = f"{airport_id}\0{relative_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calendar_date(value: Any) -> datetime:
    try:
        return datetime.strptime(str(value), "%m-%d-%y")
    except (TypeError, ValueError):
        return datetime.max


def _natural_filename(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def ordered_sources(
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Order sources by airport, calendar directory, natural filename and path."""
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("airport_id", "")).casefold(),
            _calendar_date(row.get("date_directory")),
            _natural_filename(row.get("filename")),
            str(row.get("relative_path", "")).casefold(),
            str(row.get("relative_path", "")),
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_with_retry(source: Path, destination: Path) -> None:
    for delay in (*_ATOMIC_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_airport_roots(project_root: Path) -> dict[str, Path]:
    config = yaml.safe_load(
        (project_root / "configs/local_paths.yaml").read_text(encoding="utf-8")
    )
    return {
        str(key).lower(): Path(value)
        for key, value in config["raw_airport_dirs"].items()
    }


def _load_recovery(
    project_root: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    path = project_root / "outputs/r01/archive_comparison.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (
                str(row["airport_id"]).lower(),
                str(row["extracted_relative_path"]),
            ): {str(key): str(value) for key, value in row.items()}
            for row in rows
            if row.get("status")
            in {"content_mismatch_recoverable", "missing_extracted_recoverable"}
            and row.get("extracted_relative_path")
        }


@contextmanager
def open_source_binary(
    row: Mapping[str, Any],
    airport_roots: Mapping[str, Path],
    recovery: Mapping[tuple[str, str], Mapping[str, str]],
) -> Iterator[tuple[BinaryIO, str | None]]:
    """Open exact bytes for an R01-authorized source materialization."""
    airport = str(row["airport_id"]).lower()
    relative_path = str(row["relative_path"])
    if row["r01_disposition"] == "read_verified_archive_member":
        try:
            comparison = recovery[(airport, relative_path)]
        except KeyError as exc:
            raise ValueError(
                f"missing verified archive routing for {airport}:{relative_path}"
            ) from exc
        archive_path = airport_roots[airport] / comparison["archive_relative_path"]
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(comparison["member_path"], "r") as binary:
                yield binary, comparison["member_path"]
        return
    path = airport_roots[airport] / relative_path
    with path.open("rb") as binary:
        yield binary, None


@contextmanager
def open_source(
    row: Mapping[str, Any],
    airport_roots: Mapping[str, Path],
    recovery: Mapping[tuple[str, str], Mapping[str, str]],
) -> Iterator[tuple[TextIO, str | None]]:
    """Decode an R01-authorized source with the established R02 policy."""
    with open_source_binary(row, airport_roots, recovery) as (binary, member):
        with TextIOWrapper(
            binary, encoding="utf-8-sig", errors="replace", newline=""
        ) as text:
            yield text, member


def iter_logical_records(
    handle: TextIO, disposition: str
) -> Iterator[tuple[list[str], list[str], int]]:
    """Yield tolerant logical rows, dropping only the final terminal record."""
    csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
    reader = csv.reader(handle, strict=False)
    try:
        header = next(reader)
    except StopIteration:
        return
    pending: tuple[list[str], int] | None = None
    for source_row, row in enumerate(reader, start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if pending is not None:
            yield header, pending[0], pending[1]
        pending = (row, source_row)
    if pending is not None and disposition != "drop_terminal_record_in_R02":
        yield header, pending[0], pending[1]


def _eligible_sources(project_root: Path, airport: str) -> list[Mapping[str, Any]]:
    rows = pq.read_table(
        project_root / "data/manifests/raw_file_manifest.parquet"
    ).to_pylist()
    return ordered_sources(
        row
        for row in rows
        if str(row.get("airport_id", "")).lower() == airport
        and row.get("extension") == ".csv"
        and row.get("r01_resolved") is True
        and row.get("r01_disposition") not in EXCLUDED_DISPOSITIONS
    )


def _audit_row(
    parsed: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    airport: str,
    archive_member: str | None,
    source_sequence: int,
) -> dict[str, Any]:
    local_candidate = parsed.get("timestamp_assuming_airport_local")
    try:
        year = int(str(source.get("year")))
    except (TypeError, ValueError):
        year = None
    return {
        "airport_id": airport,
        "aircraft_id": str(parsed.get("ID") or ""),
        "report_timestamp_utc": parsed.get("timestamp_utc_candidate"),
        "timestamp_assuming_airport_local": (
            local_candidate.isoformat() if local_candidate is not None else None
        ),
        "local_candidate_ambiguous": bool(
            parsed.get("local_candidate_ambiguous")
        ),
        "local_candidate_nonexistent": bool(
            parsed.get("local_candidate_nonexistent")
        ),
        "timestamp_parse_status": str(parsed["timestamp_parse_status"]),
        "timestamp_parse_error": parsed.get("timestamp_parse_error"),
        "lat": parsed.get("Lat"),
        "lon": parsed.get("Lon"),
        "altitude": parsed.get("Altitude"),
        "speed": parsed.get("Speed"),
        "heading": parsed.get("Heading"),
        "age": parsed.get("Age"),
        "range": parsed.get("Range"),
        "bearing": parsed.get("Bearing"),
        "tail": parsed.get("Tail"),
        "altis_gnss": parsed.get("AltisGNSS"),
        "source_file_id": source_file_id(airport, str(source["relative_path"])),
        "source_relative_path": str(source["relative_path"]),
        "archive_member": archive_member,
        "source_row_number": int(parsed["source_row"]),
        "source_sequence": source_sequence,
        "r01_disposition": str(source["r01_disposition"]),
        "year": year,
        "schema_version": str(parsed["schema_version"]),
        "field_errors_json": json.dumps(
            parsed.get("field_errors") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _partition_one_source(
    source: Mapping[str, Any],
    *,
    airport: str,
    source_sequence: int,
    bucket_count: int,
    airport_roots: Mapping[str, Path],
    recovery: Mapping[tuple[str, str], Mapping[str, str]],
    fragments_root: Path,
) -> dict[str, Any]:
    source_name = f"{source_sequence:08d}"
    final_directory = fragments_root / source_name
    temporary_directory = fragments_root / f"{source_name}.tmp"
    if final_directory.exists():
        shutil.rmtree(final_directory)
    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)
    temporary_directory.mkdir(parents=True)
    buffers: dict[int, list[dict[str, Any]]] = {}
    writers: dict[int, pq.ParquetWriter] = {}
    bucket_counts: dict[str, int] = {}
    row_count = 0

    def flush(bucket: int) -> None:
        rows = buffers.get(bucket)
        if not rows:
            return
        writer = writers.get(bucket)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_directory / f"{bucket:04d}.parquet",
                _AUDIT_SCHEMA,
                compression="zstd",
            )
            writers[bucket] = writer
        writer.write_table(
            pa.Table.from_pylist(rows, schema=_AUDIT_SCHEMA),
            row_group_size=ROW_GROUP_ROWS,
        )
        rows.clear()

    try:
        with open_source(source, airport_roots, recovery) as (
            handle,
            archive_member,
        ):
            for header, raw_row, source_row in iter_logical_records(
                handle, str(source["r01_disposition"])
            ):
                parsed = parse_raw_row(
                    header,
                    raw_row,
                    airport=airport,
                    source_path=str(source["relative_path"]),
                    source_row=source_row,
                    r01_disposition=str(source["r01_disposition"]),
                    archive_member=archive_member,
                )
                row = _audit_row(
                    parsed,
                    source,
                    airport=airport,
                    archive_member=archive_member,
                    source_sequence=source_sequence,
                )
                bucket = stable_bucket(
                    airport, str(row["aircraft_id"]), bucket_count
                )
                buffers.setdefault(bucket, []).append(row)
                key = str(bucket)
                bucket_counts[key] = bucket_counts.get(key, 0) + 1
                row_count += 1
                if len(buffers[bucket]) >= ROW_GROUP_ROWS:
                    flush(bucket)
        for bucket in list(buffers):
            flush(bucket)
    finally:
        for writer in writers.values():
            writer.close()
    _replace_with_retry(temporary_directory, final_directory)
    fragment_hashes = {
        str(int(path.stem)): _sha256_file(path)
        for path in sorted(final_directory.glob("*.parquet"))
    }
    return {
        "source_sequence": source_sequence,
        "source_file_id": source_file_id(airport, str(source["relative_path"])),
        "relative_path": str(source["relative_path"]),
        "r01_disposition": str(source["r01_disposition"]),
        "row_count": row_count,
        "bucket_row_counts": dict(sorted(bucket_counts.items())),
        "fragment_hashes": fragment_hashes,
    }


def _after_source_completed(_source: Mapping[str, Any]) -> None:
    """Test seam invoked only after the source checkpoint is durable."""


def _guard_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    identity: RunIdentity,
    airport: str,
    bucket_count: int,
) -> None:
    expected = {
        "identity": identity.to_dict(),
        "airport": airport,
        "bucket_count": bucket_count,
        "schema_version": PARTITION_SCHEMA_VERSION,
    }
    actual = {key: checkpoint.get(key) for key in expected}
    if actual != expected:
        raise ValueError("checkpoint identity mismatch; refusing reuse")


def _verify_completed_fragments(
    run_dir: Path,
    completed_sources: Iterable[Mapping[str, Any]],
    bucket_count: int,
) -> None:
    for source in completed_sources:
        sequence = int(source["source_sequence"])
        source_directory = run_dir / "source_fragments" / f"{sequence:08d}"
        if not source_directory.is_dir():
            raise ValueError(
                f"checkpoint fragment directory missing for source {sequence}"
            )
        expected = {
            str(key): str(value)
            for key, value in dict(source.get("fragment_hashes") or {}).items()
        }
        actual_paths = {
            str(int(path.stem)): path
            for path in source_directory.glob("*.parquet")
        }
        if set(actual_paths) != set(expected):
            raise ValueError(
                f"checkpoint fragment set mismatch for source {sequence}"
            )
        for key, path in actual_paths.items():
            if _sha256_file(path) != expected[key]:
                raise ValueError(
                    f"checkpoint fragment hash mismatch for source {sequence}, "
                    f"bucket {key}"
                )
        actual_counts = {
            key: pq.ParquetFile(path).metadata.num_rows
            for key, path in actual_paths.items()
        }
        expected_counts = {
            str(key): int(value)
            for key, value in dict(
                source.get("bucket_row_counts") or {}
            ).items()
        }
        if (
            set(expected_counts)
            - {str(index) for index in range(bucket_count)}
            or actual_counts != expected_counts
            or sum(actual_counts.values()) != int(source["row_count"])
        ):
            raise ValueError(
                f"checkpoint fragment row count mismatch for source {sequence}"
            )


def _verify_completed_source_metadata(
    completed_sources: list[Mapping[str, Any]],
    ordered: list[Mapping[str, Any]],
    airport: str,
) -> None:
    sequences = [int(source["source_sequence"]) for source in completed_sources]
    if sequences != list(range(len(completed_sources))):
        raise ValueError("checkpoint source metadata mismatch: sequence")
    if len(completed_sources) > len(ordered):
        raise ValueError("checkpoint source metadata mismatch: source count")
    for sequence, (completed, current) in enumerate(
        zip(completed_sources, ordered)
    ):
        relative_path = str(current["relative_path"])
        expected = {
            "source_sequence": sequence,
            "source_file_id": source_file_id(airport, relative_path),
            "relative_path": relative_path,
            "r01_disposition": str(current["r01_disposition"]),
        }
        actual = {key: completed.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"checkpoint source metadata mismatch for source {sequence}"
            )


def _expected_bucket_keys(bucket_count: int) -> set[str]:
    return {str(index) for index in range(bucket_count)}


def _require_bucket_manifest_keys(manifest: PartitionManifest) -> None:
    expected = _expected_bucket_keys(manifest.bucket_count)
    if (
        set(manifest.bucket_files) != expected
        or set(manifest.bucket_hashes) != expected
        or set(manifest.bucket_row_counts) != expected
    ):
        raise ValueError("bucket manifest key mismatch")


def _verify_finalized_buckets(manifest: PartitionManifest) -> None:
    _require_bucket_manifest_keys(manifest)
    for key, path_text in manifest.bucket_files.items():
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != manifest.bucket_hashes[key]:
            raise ValueError(f"finalized bucket hash mismatch for bucket {key}")


def _validate_artifact_dir(
    project_root: Path,
    artifact_dir: Path,
    *,
    label: str,
    representative_paths: Iterable[Path],
) -> None:
    try:
        relative = artifact_dir.relative_to(project_root)
    except ValueError:
        return
    if not relative.parts:
        raise ValueError(f"{label} inside project must be ignored")
    directory_result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--no-index",
            "--",
            relative.as_posix(),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if directory_result.returncode != 0:
        raise ValueError(f"{label} inside project must be ignored")
    candidates = [
        (relative / representative).as_posix()
        for representative in representative_paths
    ]
    for candidate in candidates:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", candidate],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"{label} inside project must be ignored")


def _validate_work_dir(project_root: Path, work_dir: Path) -> None:
    _validate_artifact_dir(
        project_root,
        work_dir,
        label="work_dir",
        representative_paths=(
            Path("kagc/checkpoint.json"),
            Path("kagc/checkpoint.json.tmp"),
            Path("kagc/partition_manifest.json"),
            Path("kagc/partition_manifest.json.tmp"),
            Path("kagc/source_fragments/00000000/0000.parquet"),
            Path("kagc/source_fragments/00000000.tmp/0000.parquet"),
            Path("kagc/buckets/bucket-0000.parquet"),
            Path("kagc/buckets/bucket-0000.parquet.tmp"),
            Path("kbtp/checkpoint.json"),
            Path("kbtp/checkpoint.json.tmp"),
            Path("kbtp/partition_manifest.json"),
            Path("kbtp/partition_manifest.json.tmp"),
            Path("kbtp/source_fragments/00000000/0000.parquet"),
            Path("kbtp/source_fragments/00000000.tmp/0000.parquet"),
            Path("kbtp/buckets/bucket-0000.parquet"),
            Path("kbtp/buckets/bucket-0000.parquet.tmp"),
        ),
    )


def _validate_output_dir(project_root: Path, output_dir: Path) -> None:
    _validate_artifact_dir(
        project_root,
        output_dir,
        label="output_dir",
        representative_paths=(
            Path("run_manifest.json"),
            Path("run_manifest.json.tmp"),
            Path(".generation-generation-probe.tmp/column_profile.parquet"),
            Path(".generation-generation-probe.tmp/column_profile.parquet.tmp"),
            Path(".generation-generation-probe.tmp/duplicate_records.jsonl"),
            Path(".generation-generation-probe.tmp/invariants.json"),
            Path(".generation-generation-probe.tmp/invariants.json.tmp"),
            Path(".generation-generation-probe.tmp/output_hashes.json"),
            Path(".generation-generation-probe.tmp/output_hashes.json.tmp"),
            Path("generations/generation-probe/column_profile.parquet"),
            Path("generations/generation-probe/duplicate_records.jsonl"),
            Path("generations/generation-probe/invariants.json"),
            Path("generations/generation-probe/output_hashes.json"),
        ),
    )


def _finalize_buckets(
    *,
    run_dir: Path,
    sources: Iterable[Mapping[str, Any]],
    bucket_count: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    fragments_root = run_dir / "source_fragments"
    buckets_dir = run_dir / "buckets"
    buckets_dir.mkdir(parents=True, exist_ok=True)
    bucket_files: dict[str, str] = {}
    bucket_hashes: dict[str, str] = {}
    bucket_counts: dict[str, int] = {}
    source_sequences = [int(source["source_sequence"]) for source in sources]
    for bucket in range(bucket_count):
        key = str(bucket)
        final_path = buckets_dir / f"bucket-{bucket:04d}.parquet"
        temporary = final_path.with_name(f"{final_path.name}.tmp")
        writer: pq.ParquetWriter | None = None
        row_count = 0
        try:
            for sequence in source_sequences:
                fragment = (
                    fragments_root
                    / f"{sequence:08d}"
                    / f"{bucket:04d}.parquet"
                )
                if not fragment.exists():
                    continue
                parquet = pq.ParquetFile(fragment)
                row_count += parquet.metadata.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, _AUDIT_SCHEMA, compression="zstd"
                    )
                for batch in parquet.iter_batches(batch_size=ROW_GROUP_ROWS):
                    writer.write_table(pa.Table.from_batches([batch]))
            if writer is None:
                pq.write_table(
                    pa.Table.from_pylist([], schema=_AUDIT_SCHEMA),
                    temporary,
                    compression="zstd",
                )
            else:
                writer.close()
                writer = None
        finally:
            if writer is not None:
                writer.close()
        _replace_with_retry(temporary, final_path)
        bucket_files[key] = str(final_path.resolve())
        bucket_hashes[key] = _sha256_file(final_path)
        bucket_counts[key] = row_count
    return bucket_files, bucket_hashes, bucket_counts


def partition_sources(
    project_root: Path,
    airport: str,
    work_dir: Path,
    bucket_count: int = 256,
    resume: bool = True,
) -> PartitionManifest:
    """Stream one airport into stable, atomic Parquet audit buckets."""
    root = Path(project_root).resolve()
    normalized_airport = airport.lower()
    if normalized_airport not in {"kagc", "kbtp"}:
        raise ValueError("airport must be kagc or kbtp")
    if isinstance(bucket_count, bool) or not isinstance(bucket_count, int):
        raise TypeError("bucket_count must be an integer")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    resolved_work_dir = Path(work_dir).resolve()
    _validate_work_dir(root, resolved_work_dir)
    identity = RunIdentity.from_project(root)
    run_dir = resolved_work_dir / normalized_airport
    if not resume and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    manifest_path = run_dir / "partition_manifest.json"
    if resume and manifest_path.exists():
        existing = PartitionManifest.from_path(manifest_path)
        checkpoint_shape = {
            "identity": existing.identity.to_dict(),
            "airport": existing.airport,
            "bucket_count": existing.bucket_count,
            "schema_version": existing.schema_version,
        }
        _guard_checkpoint(
            checkpoint_shape,
            identity=identity,
            airport=normalized_airport,
            bucket_count=bucket_count,
        )
        _verify_finalized_buckets(existing)
        return existing

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not resume:
            raise AssertionError("fresh run retained an old checkpoint")
        _guard_checkpoint(
            checkpoint,
            identity=identity,
            airport=normalized_airport,
            bucket_count=bucket_count,
        )
    else:
        checkpoint = {
            "identity": identity.to_dict(),
            "airport": normalized_airport,
            "bucket_count": bucket_count,
            "schema_version": PARTITION_SCHEMA_VERSION,
            "started_at_utc": _utc_now(),
            "completed_sources": [],
        }
        _atomic_json(checkpoint_path, checkpoint)

    sources = _eligible_sources(root, normalized_airport)
    completed_sources = [
        dict(item) for item in checkpoint["completed_sources"]
    ]
    _verify_completed_source_metadata(
        completed_sources, sources, normalized_airport
    )
    _verify_completed_fragments(run_dir, completed_sources, bucket_count)
    completed = {
        str(item["source_file_id"]): dict(item)
        for item in completed_sources
    }
    airport_roots = _load_airport_roots(root)
    if normalized_airport not in airport_roots:
        raise ValueError(f"missing raw root for airport {normalized_airport}")
    recovery = _load_recovery(root)
    fragments_root = run_dir / "source_fragments"
    fragments_root.mkdir(parents=True, exist_ok=True)
    for sequence, source in enumerate(sources):
        identity_key = source_file_id(
            normalized_airport, str(source["relative_path"])
        )
        if identity_key in completed:
            continue
        source_record = _partition_one_source(
            source,
            airport=normalized_airport,
            source_sequence=sequence,
            bucket_count=bucket_count,
            airport_roots=airport_roots,
            recovery=recovery,
            fragments_root=fragments_root,
        )
        completed[identity_key] = source_record
        checkpoint["completed_sources"] = sorted(
            completed.values(), key=lambda item: int(item["source_sequence"])
        )
        _atomic_json(checkpoint_path, checkpoint)
        _after_source_completed(source_record)

    completed_sources = tuple(
        sorted(completed.values(), key=lambda item: int(item["source_sequence"]))
    )
    bucket_files, bucket_hashes, bucket_row_counts = _finalize_buckets(
        run_dir=run_dir,
        sources=completed_sources,
        bucket_count=bucket_count,
    )
    total_rows = sum(int(source["row_count"]) for source in completed_sources)
    if total_rows != sum(bucket_row_counts.values()):
        raise RuntimeError("partition row conservation failed")
    manifest = PartitionManifest(
        identity=identity,
        airport=normalized_airport,
        bucket_count=bucket_count,
        schema_version=PARTITION_SCHEMA_VERSION,
        work_dir=str(run_dir),
        sources=completed_sources,
        total_rows=total_rows,
        bucket_files=bucket_files,
        bucket_hashes=bucket_hashes,
        bucket_row_counts=bucket_row_counts,
        started_at_utc=str(checkpoint["started_at_utc"]),
        ended_at_utc=_utc_now(),
        project_root=str(root),
    )
    _atomic_json(manifest_path, manifest.to_dict())
    return manifest


_PROFILE_FIELDS = {
    "aircraft_id": "ID",
    "report_timestamp_utc": "timestamp_utc_candidate",
    "timestamp_assuming_airport_local": "timestamp_assuming_airport_local",
    "local_candidate_ambiguous": "local_candidate_ambiguous",
    "local_candidate_nonexistent": "local_candidate_nonexistent",
    "lat": "Lat",
    "lon": "Lon",
    "altitude": "Altitude",
    "speed": "Speed",
    "heading": "Heading",
    "age": "Age",
    "range": "Range",
    "bearing": "Bearing",
    "tail": "Tail",
    "altis_gnss": "AltisGNSS",
}


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    _replace_with_retry(temporary, path)


def _profile_error(
    row: Mapping[str, Any],
    audit_field: str,
    raw_field: str,
) -> str | None:
    if audit_field == "report_timestamp_utc":
        value = row.get("timestamp_parse_error")
        return str(value) if value else None
    errors = json.loads(str(row.get("field_errors_json") or "{}"))
    value = errors.get(raw_field)
    return str(value) if value else None


def _profile_records(
    profiles: Mapping[tuple[str, int | None, str, str], FieldProfile],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (airport, year, schema_version, field), profile in sorted(
        profiles.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        stratum = f"{airport}|{year if year is not None else 'ALL'}|{schema_version}|{field}"
        records.append(
            {
                "airport_id": airport,
                "year": year,
                "schema_version": schema_version,
                "field": field,
                **profile.to_record(stratum),
            }
        )
    return records


def _commands(
    partition_manifest: PartitionManifest,
    output_dir: Path,
) -> dict[str, str]:
    work_parent = Path(partition_manifest.work_dir).parent
    base = (
        "python scripts/run_r02_remediation.py "
        f"--airport {partition_manifest.airport} "
        f'--work-dir "{work_parent}" '
        f'--output-dir "{output_dir}" '
        f"--bucket-count {partition_manifest.bucket_count}"
    )
    return {
        "partition": f"{base} --stage partition",
        "aggregate": f"{base} --stage aggregate",
        "all": f"{base} --stage all",
        "verify_hashes": (
            "python scripts/run_r02_remediation.py "
            f"--stage aggregate --airport {partition_manifest.airport} "
            f'--work-dir "{work_parent}" --output-dir "{output_dir}" '
            f"--bucket-count {partition_manifest.bucket_count}"
        ),
        "verify_tests": "python -m pytest tests/test_r02_runner.py -q",
    }


def aggregate_buckets(
    partition_manifest: PartitionManifest,
    output_dir: Path,
) -> AuditRunManifest:
    """Verify and exactly aggregate one finalized airport partition."""
    started_at = _utc_now()
    output = Path(output_dir).resolve()
    _validate_output_dir(
        Path(partition_manifest.project_root).resolve(),
        output,
    )
    _require_bucket_manifest_keys(partition_manifest)
    for key in sorted(
        partition_manifest.bucket_files, key=lambda value: int(value)
    ):
        path = Path(partition_manifest.bucket_files[key])
        actual = _sha256_file(path)
        expected = partition_manifest.bucket_hashes.get(key)
        if actual != expected:
            raise ValueError(
                f"bucket hash mismatch for bucket {key}: {actual} != {expected}"
            )
    actual_bucket_counts = {
        key: pq.ParquetFile(path_text).metadata.num_rows
        for key, path_text in partition_manifest.bucket_files.items()
    }

    profiles: dict[tuple[str, int | None, str, str], FieldProfile] = {}
    analyzed_bucket_count = 0
    observed_rows = 0
    nonempty_bucket_count = 0
    output.mkdir(parents=True, exist_ok=True)
    generations = output / "generations"
    generations.mkdir(exist_ok=True)
    generation_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    staging = output / f".generation-{generation_id}.tmp"
    final_generation = generations / generation_id
    staging.mkdir()
    duplicate_staged = staging / "duplicate_records.jsonl"
    try:
        with duplicate_staged.open(
            "w", encoding="utf-8", newline="\n"
        ) as duplicate_handle:
            for key in sorted(
                partition_manifest.bucket_files,
                key=lambda value: int(value),
            ):
                if actual_bucket_counts[key] == 0:
                    continue
                nonempty_bucket_count += 1
                table = pq.read_table(partition_manifest.bucket_files[key])
                rows = table.to_pylist()
                observed_rows += len(rows)
                valid_timestamp_rows = [
                    row
                    for row in rows
                    if row.get("report_timestamp_utc") is not None
                ]
                for record in analyze_rows(
                    valid_timestamp_rows
                ).to_records():
                    duplicate_handle.write(
                        json.dumps(
                            record,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    duplicate_handle.write("\n")
                duplicate_handle.flush()
                analyzed_bucket_count += 1
                for row in rows:
                    airport = str(row["airport_id"])
                    year = (
                        int(row["year"])
                        if row.get("year") is not None
                        else None
                    )
                    schema_version = str(row["schema_version"])
                    identity = (
                        str(row["source_file_id"]),
                        int(row["source_row_number"]),
                    )
                    for audit_field, raw_field in _PROFILE_FIELDS.items():
                        error = _profile_error(
                            row, audit_field, raw_field
                        )
                        value = row.get(audit_field)
                        for key_tuple in (
                            (
                                airport,
                                year,
                                schema_version,
                                audit_field,
                            ),
                            (airport, None, "ALL", audit_field),
                        ):
                            profile = profiles.setdefault(
                                key_tuple, FieldProfile()
                            )
                            profile.add(
                                (*identity, audit_field), value, error
                            )

        source_sequences = [
            int(source["source_sequence"])
            for source in partition_manifest.sources
        ]
        source_ids = [
            str(source["source_file_id"])
            for source in partition_manifest.sources
        ]
        invariants = {
            "all_bucket_hashes_verified": True,
            "all_sources_completed": (
                source_sequences == list(range(len(source_sequences)))
                and len(source_ids) == len(set(source_ids))
                and sum(
                    int(source["row_count"])
                    for source in partition_manifest.sources
                )
                == partition_manifest.total_rows
            ),
            "analyzed_nonempty_bucket_count_matches": (
                analyzed_bucket_count == nonempty_bucket_count
            ),
            "bucket_row_counts_match": (
                actual_bucket_counts
                == {
                    str(key): int(value)
                    for key, value in (
                        partition_manifest.bucket_row_counts.items()
                    )
                }
            ),
            "partition_row_conservation": (
                partition_manifest.total_rows
                == sum(partition_manifest.bucket_row_counts.values())
                == observed_rows
            ),
        }
        if not all(invariants.values()):
            failed = ", ".join(
                key for key, value in invariants.items() if not value
            )
            raise RuntimeError(f"aggregate invariant failure: {failed}")

        column_staged = staging / "column_profile.parquet"
        invariants_staged = staging / "invariants.json"
        hashes_staged = staging / "output_hashes.json"
        _atomic_parquet(
            column_staged,
            pa.Table.from_pylist(_profile_records(profiles)),
        )
        _atomic_json(invariants_staged, invariants)
        staged_artifacts = {
            "column_profile": column_staged,
            "duplicate_records": duplicate_staged,
            "invariants": invariants_staged,
        }
        output_hashes = {
            name: _sha256_file(path)
            for name, path in staged_artifacts.items()
        }
        _atomic_json(hashes_staged, output_hashes)
        staged_artifacts["output_hashes"] = hashes_staged
        output_hashes["output_hashes"] = _sha256_file(hashes_staged)
        artifact_files = {
            name: str(final_generation / path.name)
            for name, path in staged_artifacts.items()
        }
        run = AuditRunManifest(
            identity=partition_manifest.identity,
            airport=partition_manifest.airport,
            bucket_count=partition_manifest.bucket_count,
            schema_version=partition_manifest.schema_version,
            output_dir=str(output),
            artifact_files=artifact_files,
            output_hashes=output_hashes,
            invariants=invariants,
            commands=_commands(partition_manifest, output),
            analyzed_bucket_count=analyzed_bucket_count,
            started_at_utc=started_at,
            ended_at_utc=_utc_now(),
        )
        _replace_with_retry(staging, final_generation)
        _atomic_json(output / "run_manifest.json", run.to_dict())
        return run
    finally:
        if staging.exists():
            shutil.rmtree(staging)
