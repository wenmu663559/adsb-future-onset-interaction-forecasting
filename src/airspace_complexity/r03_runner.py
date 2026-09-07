"""Authoritative R01/R02 source registry and pre-parse R03 enumeration."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import heapq
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from airspace_complexity.canonical_points import (
    CanonicalPolicies,
    ParsedRow,
    age_rank,
    canonical_json,
    canonical_report_id,
    membership_schema,
    parsed_rows_schema,
    raw_row_id,
    resolve_group,
    shard_for_key,
    stable_source_order,
    unique_reports_schema,
)
from airspace_complexity.r02_runner import iter_logical_records, open_source_binary


_REGISTRY_SCHEMA_VERSION = "r03_source_registry_v1"
_SEQUENCE_SCHEMA_VERSION = "r03_source_sequence_v1"
_ENUMERATION_SCHEMA_VERSION = "r03_source_enumeration_v1"
_AIRPORT_ORDER = ("KAGC", "KBTP")
_RECOVERY_STATUSES = {
    "content_mismatch_recoverable",
    "missing_extracted_recoverable",
}
_EXCLUDED_DISPOSITIONS = {
    "exclude_non_data_file",
    "exclude_source_corrupt_file",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CRC32_RE = re.compile(r"[0-9a-f]{8}")
_CSV_FIELD_SIZE_LIMIT = 64 * 1024 * 1024
_SOURCE_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024
_SHARD_HASH_VERSION = "sha256_canonical_key_v1"
_FINAL_SORT_VERSION = "r03_final_sort_v1"
_SHARD_SORT_CONTRACT = "(natural_key,age_rank_class,age_rank_value,-completeness_score,stable_source_order)"
_FINAL_SORT_CONTRACTS = {
    "parsed_rows": "(natural_key,stable_source_order)",
    "unique_reports": "natural_key",
    "report_membership": "(canonical_report_id,dedup_rank,raw_row_id)",
    "duplicate_conflicts": "canonical_report_id",
}


class BlockedSourceError(RuntimeError):
    """Raised when lineage or source bytes cannot be trusted exactly."""


@dataclass(frozen=True)
class R03SourceSpec:
    airport_id: str
    source_file_id: str
    relative_path: str
    r01_disposition: str
    source_sequence: int
    extracted_sha256: str | None
    archive_relative_path: str | None
    archive_sha256: str | None
    archive_member: str | None
    member_crc32: str | None
    member_size_bytes: int | None
    authenticated_row_count: int


@dataclass(frozen=True)
class VerifiedR03Source:
    spec: R03SourceSpec
    source_content_sha256: str
    resolved_path: Path
    materialization_kind: str


@dataclass(frozen=True)
class EnumeratedRecord:
    source: VerifiedR03Source
    logical_record_ordinal: int
    source_row_number: int
    header: tuple[str, ...]
    row: tuple[str, ...]
    raw_row_id: str


@dataclass(frozen=True)
class SourceEnumerationResult:
    registry_id: str
    r02_partition_manifest_sha256: str
    source: VerifiedR03Source
    header_sha256: str
    logical_record_count: int
    first_raw_row_id: str | None
    last_raw_row_id: str | None
    first_source_row_number: int | None
    last_source_row_number: int | None
    status: str


@dataclass(frozen=True)
class ShardRun:
    schema_version: str
    artifact_role: str
    shard_hash_version: str
    shard_count: int
    shard_id: int
    run_index: int
    relative_path: str
    row_count: int
    schema_fingerprint: str
    sha256: str
    sort_contract: str
    min_natural_key: str
    max_natural_key: str
    finalized: bool
    path: Path


@dataclass(frozen=True)
class ArtifactCounts:
    parsed_rows: int = 0
    unique_reports: int = 0
    report_membership: int = 0
    duplicate_conflicts: int = 0
    duplicate_excess: int = 0


@dataclass(frozen=True)
class ShardArtifactSet:
    shard_id: int
    parsed_rows: Path
    unique_reports: Path
    report_membership: Path
    duplicate_conflicts: Path
    counts: ArtifactCounts


@dataclass(frozen=True)
class FinalPartition:
    artifact_role: str
    airport_id: str
    partition_year: int
    partition_month: int
    path: Path
    relative_path: str
    row_count: int
    schema_fingerprint: str
    sha256: str
    sort_contract: str
    finalized: bool = True


@dataclass(frozen=True)
class ConservationResult:
    passed: bool
    counts: Mapping[str, int]
    invariants: Mapping[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class R03SourceRegistry:
    schema_version: str
    registry_id: str
    sources: tuple[R03SourceSpec, ...]
    raw_manifest_sha256: str
    archive_comparison_sha256: str
    r02_run_manifest_sha256: str
    r02_decision_sha256: str
    partition_manifest_sha256_by_airport: tuple[tuple[str, str], ...]
    source_sequence_manifest_sha256: str

    @classmethod
    def from_lineage(cls, project_root: Path) -> "R03SourceRegistry":
        root = Path(project_root).resolve()
        raw_path = root / "data/manifests/raw_file_manifest.parquet"
        comparison_path = root / "outputs/r01/archive_comparison.csv"
        run_path = root / "outputs/r02/run_manifest.json"
        decision_path = root / "outputs/r02/decision.json"
        try:
            raw_hash = _sha256_file(raw_path)
            comparison_hash = _sha256_file(comparison_path)
            run_hash = _sha256_file(run_path)
            decision_hash = _sha256_file(decision_path)
            run = _read_json(run_path)
            decision = _read_json(decision_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise BlockedSourceError(f"R03 lineage cannot be loaded: {exc}") from exc

        _require(
            decision.get("decision") == "PROCEED_TO_R03"
            and decision.get("r02_gate_passed") is True,
            "R02 decision does not authorize R03",
        )
        _require_hash_match(
            decision.get("input_manifest_sha256"), raw_hash, "decision raw manifest"
        )
        _require_hash_match(
            decision.get("run_manifest_sha256"), run_hash, "decision run manifest"
        )
        _require(
            dict(run.get("invariants") or {}).get("all_gate_checks_passed") is True,
            "R02 run manifest gate is not complete",
        )
        run_identity_hashes = dict(
            dict(run.get("identity") or {}).get("hashes") or {}
        )
        _require_hash_match(
            run_identity_hashes.get("raw_manifest"), raw_hash, "R02 raw manifest"
        )
        _require_hash_match(
            run_identity_hashes.get("archive_comparison"),
            comparison_hash,
            "R02 archive comparison",
        )

        try:
            raw_manifest = pq.ParquetFile(raw_path)
            comparison_rows = _read_csv(comparison_path)
        except (OSError, ValueError) as exc:
            raise BlockedSourceError(f"R01 lineage cannot be read: {exc}") from exc

        raw_by_locator: dict[tuple[str, str], Mapping[str, object]] = {}
        try:
            for batch in raw_manifest.iter_batches(batch_size=65_536):
                for index in range(batch.num_rows):
                    row = _row_from_batch(batch, index)
                    airport = str(row.get("airport_id") or "").upper()
                    relative = _normalize_relative_path(
                        row.get("relative_path"), "raw-manifest relative_path"
                    )
                    locator = (airport, relative)
                    _require(locator not in raw_by_locator, f"duplicate raw locator {locator}")
                    raw_by_locator[locator] = row
        except (OSError, ValueError) as exc:
            raise BlockedSourceError(f"R01 raw manifest cannot be streamed: {exc}") from exc

        recovery_by_locator: dict[
            tuple[str, str], Mapping[str, str]
        ] = {}
        for row in comparison_rows:
            if row.get("status") not in _RECOVERY_STATUSES:
                continue
            airport = str(row.get("airport_id") or "").upper()
            extracted = _normalize_relative_path(
                row.get("extracted_relative_path"),
                "archive comparison extracted_relative_path",
            )
            locator = (airport, extracted)
            _require(
                locator not in recovery_by_locator,
                f"duplicate archive recovery locator {locator}",
            )
            recovery_by_locator[locator] = row

        airport_entries = {
            str(key).upper(): value
            for key, value in dict(run.get("airports") or {}).items()
        }
        _require(
            set(airport_entries) == set(_AIRPORT_ORDER),
            "R02 run manifest must name exactly KAGC and KBTP",
        )

        specs: list[R03SourceSpec] = []
        partition_hashes: list[tuple[str, str]] = []
        seen_partition_locators: set[tuple[str, str]] = set()
        for airport in _AIRPORT_ORDER:
            entry = dict(airport_entries[airport])
            partition_path = _resolve_lineage_path(
                root, entry.get("partition_manifest"), "partition manifest"
            )
            try:
                partition_hash = _sha256_file(partition_path)
                partition = _read_json(partition_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise BlockedSourceError(
                    f"{airport} partition manifest cannot be loaded: {exc}"
                ) from exc
            _require_hash_match(
                entry.get("partition_manifest_sha256"),
                partition_hash,
                f"{airport} partition manifest",
            )
            partition_hashes.append((airport, partition_hash))
            _require(
                str(partition.get("airport") or "").upper() == airport,
                f"{airport} partition airport mismatch",
            )
            partition_identity_hashes = dict(
                dict(partition.get("identity") or {}).get("hashes") or {}
            )
            _require_hash_match(
                partition_identity_hashes.get("raw_manifest"),
                raw_hash,
                f"{airport} partition raw manifest",
            )
            _require_hash_match(
                partition_identity_hashes.get("archive_comparison"),
                comparison_hash,
                f"{airport} partition archive comparison",
            )
            partition_sources = list(partition.get("sources") or [])
            _require(
                entry.get("source_count") == len(partition_sources),
                f"{airport} partition source count mismatch",
            )
            sequences = [item.get("source_sequence") for item in partition_sources]
            _require(
                sequences == list(range(len(partition_sources))),
                f"{airport} source_sequence is not contiguous zero-based order",
            )
            for source in partition_sources:
                relative = _normalize_relative_path(
                    source.get("relative_path"), "partition relative_path"
                )
                locator = (airport, relative)
                _require(
                    locator not in seen_partition_locators,
                    f"duplicate source locator {locator}",
                )
                seen_partition_locators.add(locator)
                raw = raw_by_locator.get(locator)
                _require(raw is not None, f"partition source missing from R01 {locator}")
                disposition = str(source.get("r01_disposition") or "")
                _require(
                    raw.get("extension") == ".csv"
                    and raw.get("r01_resolved") is True
                    and raw.get("r01_disposition") not in _EXCLUDED_DISPOSITIONS,
                    f"partition source is not R01-authorized {locator}",
                )
                _require(
                    disposition == str(raw.get("r01_disposition") or ""),
                    f"R01 disposition mismatch for {locator}",
                )
                extracted_hash = _require_sha256(
                    raw.get("sha256"), f"extracted SHA-256 for {locator}"
                )
                archive_relative: str | None = None
                archive_hash: str | None = None
                archive_member: str | None = None
                member_crc: str | None = None
                member_size: int | None = None
                if disposition == "read_verified_archive_member":
                    recovery = recovery_by_locator.get(locator)
                    _require(
                        recovery is not None,
                        f"missing authorized archive recovery routing for {locator}",
                    )
                    archive_relative = _normalize_relative_path(
                        recovery.get("archive_relative_path"),
                        "archive_relative_path",
                    )
                    archive_member = _normalize_relative_path(
                        recovery.get("member_path"), "archive member"
                    )
                    archive_row = raw_by_locator.get(
                        (airport, archive_relative)
                    )
                    _require(
                        archive_row is not None
                        and archive_row.get("extension") == ".zip"
                        and archive_row.get("r01_resolved") is True,
                        f"archive routing is not bound to R01 {locator}",
                    )
                    archive_hash = _require_sha256(
                        archive_row.get("sha256"), f"archive SHA-256 for {locator}"
                    )
                    member_crc = str(recovery.get("member_crc32") or "").lower()
                    _require(
                        _CRC32_RE.fullmatch(member_crc) is not None,
                        f"invalid archive CRC32 for {locator}",
                    )
                    member_size = _require_nonnegative_int(
                        recovery.get("member_size_bytes"),
                        f"archive member size for {locator}",
                    )
                specs.append(
                    R03SourceSpec(
                        airport_id=airport,
                        source_file_id=_require_text(
                            source.get("source_file_id"),
                            f"source_file_id for {locator}",
                        ),
                        relative_path=relative,
                        r01_disposition=disposition,
                        source_sequence=int(source["source_sequence"]),
                        extracted_sha256=extracted_hash,
                        archive_relative_path=archive_relative,
                        archive_sha256=archive_hash,
                        archive_member=archive_member,
                        member_crc32=member_crc,
                        member_size_bytes=member_size,
                        authenticated_row_count=_require_nonnegative_int(
                            source.get("row_count"), f"R02 row_count for {locator}"
                        ),
                    )
                )

        _require(
            dict(run.get("invariants") or {}).get("source_count") == len(specs),
            "R02 invariant source count mismatch",
        )
        _reject_duplicate_materialized_locators(specs)

        sequence_records = [
            {
                "airport_id": spec.airport_id,
                "source_file_id": spec.source_file_id,
                "relative_path": spec.relative_path,
                "r01_disposition": spec.r01_disposition,
                "source_sequence": spec.source_sequence,
                "authenticated_row_count": spec.authenticated_row_count,
            }
            for spec in specs
        ]
        sequence_hash = _sha256_canonical(
            {
                "schema_version": _SEQUENCE_SCHEMA_VERSION,
                "sources": sequence_records,
            }
        )
        partition_hash_tuple = tuple(partition_hashes)
        registry_record = {
            "schema_version": _REGISTRY_SCHEMA_VERSION,
            "raw_manifest_sha256": raw_hash,
            "archive_comparison_sha256": comparison_hash,
            "r02_run_manifest_sha256": run_hash,
            "r02_decision_sha256": decision_hash,
            "partition_manifest_sha256_by_airport": partition_hash_tuple,
            "source_sequence_manifest_sha256": sequence_hash,
        }
        return cls(
            schema_version=_REGISTRY_SCHEMA_VERSION,
            registry_id=_sha256_canonical(registry_record),
            sources=tuple(specs),
            raw_manifest_sha256=raw_hash,
            archive_comparison_sha256=comparison_hash,
            r02_run_manifest_sha256=run_hash,
            r02_decision_sha256=decision_hash,
            partition_manifest_sha256_by_airport=partition_hash_tuple,
            source_sequence_manifest_sha256=sequence_hash,
        )


def verify_source(
    spec: R03SourceSpec, airport_roots: Mapping[str, Path]
) -> VerifiedR03Source:
    roots = {
        str(key).upper(): Path(value).resolve()
        for key, value in airport_roots.items()
    }
    root = roots.get(spec.airport_id.upper())
    if root is None:
        raise BlockedSourceError(f"missing raw root for {spec.airport_id}")
    expected_content_hash = _require_sha256(
        spec.extracted_sha256, "extracted source SHA-256"
    )
    try:
        if spec.r01_disposition != "read_verified_archive_member":
            _require(
                all(
                    value is None
                    for value in (
                        spec.archive_relative_path,
                        spec.archive_sha256,
                        spec.archive_member,
                        spec.member_crc32,
                        spec.member_size_bytes,
                    )
                ),
                "extracted source unexpectedly contains archive routing",
            )
            relative = _normalize_relative_path(spec.relative_path, "relative_path")
            resolved = _safe_join(root, relative, "extracted source")
            actual_hash = _sha256_file(resolved)
            _require_hash_match(expected_content_hash, actual_hash, "source SHA-256")
            return VerifiedR03Source(
                spec=spec,
                source_content_sha256=actual_hash,
                resolved_path=resolved,
                materialization_kind="extracted_file",
            )

        archive_relative = _normalize_relative_path(
            spec.archive_relative_path, "archive_relative_path"
        )
        archive_member = _normalize_relative_path(spec.archive_member, "archive member")
        _require(
            archive_relative == spec.archive_relative_path
            and archive_member == spec.archive_member,
            "archive paths must already be normalized with forward slashes",
        )
        archive_hash = _require_sha256(spec.archive_sha256, "archive SHA-256")
        member_crc = str(spec.member_crc32 or "").lower()
        _require(
            _CRC32_RE.fullmatch(member_crc) is not None,
            "invalid archive member CRC32",
        )
        member_size = _require_nonnegative_int(
            spec.member_size_bytes, "archive member size"
        )
        resolved = _safe_join(root, archive_relative, "archive source")
        _require_hash_match(archive_hash, _sha256_file(resolved), "archive SHA-256")
        with zipfile.ZipFile(resolved) as archive:
            members: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                normalized = _normalize_relative_path(info.filename, "ZIP member")
                _require(
                    normalized not in members,
                    f"duplicate normalized archive member {normalized}",
                )
                members[normalized] = info
            info = members.get(archive_member)
            _require(info is not None, f"declared archive member missing: {archive_member}")
            _require(
                info.file_size == member_size,
                f"archive member size mismatch for {archive_member}",
            )
            _require(
                f"{info.CRC:08x}" == member_crc,
                f"archive member CRC32 mismatch for {archive_member}",
            )
            with archive.open(info, "r") as handle:
                member_hash, actual_size = _sha256_stream(handle)
            _require(
                actual_size == member_size,
                f"archive member byte count mismatch for {archive_member}",
            )
            _require_hash_match(
                expected_content_hash, member_hash, "archive member content SHA-256"
            )
        return VerifiedR03Source(
            spec=spec,
            source_content_sha256=member_hash,
            resolved_path=resolved,
            materialization_kind="archive_member",
        )
    except BlockedSourceError:
        raise
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise BlockedSourceError(f"source verification failed: {exc}") from exc


def enumerate_source(source: VerifiedR03Source) -> Iterator[EnumeratedRecord]:
    try:
        with _authenticated_content_spool(source) as content:
            csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
            with _text_view(content) as boundary_handle:
                for _ in csv.reader(boundary_handle, strict=True):
                    pass
            with _text_view(content) as handle:
                for ordinal, (header, raw_row, source_row) in enumerate(
                    iter_logical_records(
                        handle, source.spec.r01_disposition
                    ),
                    start=1,
                ):
                    yield EnumeratedRecord(
                        source=source,
                        logical_record_ordinal=ordinal,
                        source_row_number=source_row,
                        header=tuple(header),
                        row=tuple(raw_row),
                        raw_row_id=raw_row_id(
                            source.spec.source_file_id,
                            source.source_content_sha256,
                            source.spec.archive_member,
                            ordinal,
                        ),
                    )
    except BlockedSourceError:
        raise
    except (csv.Error, OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise BlockedSourceError(
            f"source logical-record boundaries cannot be enumerated: {exc}"
        ) from exc


def enumerate_selected_source(
    source: VerifiedR03Source,
    selected_ordinals: tuple[int, ...],
) -> Iterator[EnumeratedRecord]:
    """Enumerate only authenticated selected ordinals and stop at their maximum."""

    if (
        not isinstance(selected_ordinals, tuple)
        or not selected_ordinals
        or any(
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal <= 0
            for ordinal in selected_ordinals
        )
        or any(
            left >= right
            for left, right in zip(selected_ordinals, selected_ordinals[1:])
        )
        or selected_ordinals[-1] > source.spec.authenticated_row_count
    ):
        raise ValueError(
            "selected ordinals must be a nonempty strictly increasing tuple "
            "within authenticated R02 row_count"
        )
    selected = frozenset(selected_ordinals)
    maximum = selected_ordinals[-1]
    observed = 0
    try:
        with _authenticated_content_spool(source) as content:
            csv.field_size_limit(_CSV_FIELD_SIZE_LIMIT)
            with _text_view(content) as handle:
                for ordinal, (header, raw_row, source_row) in enumerate(
                    iter_logical_records(handle, source.spec.r01_disposition),
                    start=1,
                ):
                    observed = ordinal
                    if ordinal in selected:
                        yield EnumeratedRecord(
                            source=source,
                            logical_record_ordinal=ordinal,
                            source_row_number=source_row,
                            header=tuple(header),
                            row=tuple(raw_row),
                            raw_row_id=raw_row_id(
                                source.spec.source_file_id,
                                source.source_content_sha256,
                                source.spec.archive_member,
                                ordinal,
                            ),
                        )
                    if ordinal == maximum:
                        break
        if observed < maximum:
            raise BlockedSourceError(
                "selected source count does not match authenticated R02 row_count"
            )
    except BlockedSourceError:
        raise
    except (csv.Error, OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise BlockedSourceError(
            f"source logical-record boundaries cannot be enumerated: {exc}"
        ) from exc


def enumeration_manifest(
    registry: R03SourceRegistry,
    results: Iterable[SourceEnumerationResult],
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None,
) -> list[dict[str, object]]:
    completed = list(results)
    expected_sources = (
        registry.sources
        if selected_ordinals is None
        else tuple(
            source
            for source in registry.sources
            if source.source_file_id in selected_ordinals
        )
    )
    if selected_ordinals is not None:
        _require(
            set(selected_ordinals) == {source.source_file_id for source in expected_sources}
            and all(selected_ordinals.values()),
            "selected enumeration source mapping is invalid",
        )
    _require(
        len(completed) == len(expected_sources),
        "enumeration result population is incomplete",
    )
    partition_hashes = dict(registry.partition_manifest_sha256_by_airport)
    rows: list[dict[str, object]] = []
    for expected, result in zip(expected_sources, completed):
        _require(
            result.source.spec == expected,
            "enumeration results are duplicate, missing, or out of registry order",
        )
        _require(
            result.registry_id == registry.registry_id,
            "enumeration result belongs to a foreign registry",
        )
        _require(
            result.r02_partition_manifest_sha256
            == partition_hashes[expected.airport_id],
            "enumeration result partition hash mismatch",
        )
        _require(result.status == "complete", "enumeration result is not complete")
        _require_sha256(result.header_sha256, "header SHA-256")
        _require_sha256(
            result.source.source_content_sha256, "source content SHA-256"
        )
        _require(
            result.source.source_content_sha256 == expected.extracted_sha256,
            "enumeration result source content hash mismatch",
        )
        count = _require_nonnegative_int(
            result.logical_record_count, "logical record count"
        )
        if selected_ordinals is not None:
            _require(
                count == len(selected_ordinals[expected.source_file_id]),
                "selected enumeration logical record count mismatch",
            )
        boundaries = (
            result.first_raw_row_id,
            result.last_raw_row_id,
            result.first_source_row_number,
            result.last_source_row_number,
        )
        if count == 0:
            _require(
                all(value is None for value in boundaries),
                "zero-row enumeration must have null boundaries",
            )
        else:
            _require(
                all(value is not None for value in boundaries),
                "nonempty enumeration must have complete boundaries",
            )
            _require_sha256(result.first_raw_row_id, "first raw_row_id")
            _require_sha256(result.last_raw_row_id, "last raw_row_id")
            first_row = _require_nonnegative_int(
                result.first_source_row_number, "first source row number"
            )
            last_row = _require_nonnegative_int(
                result.last_source_row_number, "last source row number"
            )
            _require(
                first_row >= 2 and last_row >= first_row,
                "enumeration source-row boundaries are invalid",
            )
        rows.append(
            {
                "schema_version": _ENUMERATION_SCHEMA_VERSION,
                "registry_id": registry.registry_id,
                "raw_manifest_sha256": registry.raw_manifest_sha256,
                "archive_comparison_sha256": registry.archive_comparison_sha256,
                "r02_run_manifest_sha256": registry.r02_run_manifest_sha256,
                "r02_decision_sha256": registry.r02_decision_sha256,
                "r02_partition_manifest_sha256": result.r02_partition_manifest_sha256,
                "source_sequence_manifest_sha256": registry.source_sequence_manifest_sha256,
                "airport_id": expected.airport_id,
                "source_file_id": expected.source_file_id,
                "relative_path": expected.relative_path,
                "r01_disposition": expected.r01_disposition,
                "source_sequence": expected.source_sequence,
                "archive_relative_path": expected.archive_relative_path,
                "archive_member": expected.archive_member,
                "source_content_sha256": result.source.source_content_sha256,
                "header_sha256": result.header_sha256,
                "logical_record_count": count,
                "first_raw_row_id": result.first_raw_row_id,
                "last_raw_row_id": result.last_raw_row_id,
                "first_source_row_number": result.first_source_row_number,
                "last_source_row_number": result.last_source_row_number,
                "status": "complete",
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)[0]


def _sha256_stream(handle: object) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _copy_authenticated_bytes(
    source: BinaryIO, destination: BinaryIO
) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    size = 0
    crc32 = 0
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
        crc32 = zlib.crc32(chunk, crc32)
    destination.seek(0)
    return digest.hexdigest(), size, f"{crc32 & 0xFFFFFFFF:08x}"


def _sha256_canonical(record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockedSourceError(message)


def _require_text(value: object, label: str) -> str:
    text = str(value) if value is not None else ""
    _require(bool(text), f"{label} must be nonempty")
    return text


def _require_sha256(value: object, label: str) -> str:
    text = _require_text(value, label)
    _require(_SHA256_RE.fullmatch(text) is not None, f"{label} is invalid")
    return text


def _require_hash_match(expected: object, actual: str, label: str) -> None:
    expected_hash = _require_sha256(expected, f"expected {label} hash")
    _require(expected_hash == actual, f"{label} SHA-256 mismatch")


def _require_nonnegative_int(value: object, label: str) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BlockedSourceError(f"{label} must be an integer") from exc
    _require(parsed >= 0, f"{label} must be nonnegative")
    return parsed


def _normalize_relative_path(value: object, label: str) -> str:
    text = _require_text(value, label).replace("\\", "/")
    path = PurePosixPath(text)
    _require(
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not a safe relative path",
    )
    return path.as_posix()


_CAPACITY_LINEAGE_RE = re.compile(
    r"raw/(?P<year>[0-9]{4})/(?P=year)/"
    r"(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12][0-9]|3[01])-"
    r"(?P<short_year>[0-9]{2})/(?P<file>[1-9][0-9]*)\.csv"
)


def parse_lineage_month(relative_path: str) -> str:
    normalized = _normalize_relative_path(relative_path, "capacity lineage path")
    match = _CAPACITY_LINEAGE_RE.fullmatch(normalized)
    if match is None:
        raise BlockedSourceError("capacity lineage path does not match frozen grammar")
    year = int(match.group("year"))
    if int(match.group("short_year")) != year % 100:
        raise BlockedSourceError("capacity lineage path year components disagree")
    try:
        datetime(year, int(match.group("month")), int(match.group("day")))
    except ValueError as exc:
        raise BlockedSourceError("capacity lineage path has an invalid calendar date") from exc
    return f"{year:04d}-{int(match.group('month')):02d}"


def capacity_storage_route(spec: R03SourceSpec) -> str:
    return "archive_recovery" if spec.r01_disposition == "read_verified_archive_member" else "direct"


def capacity_stratum_key(spec: R03SourceSpec) -> tuple[str, str, str]:
    return spec.airport_id, parse_lineage_month(spec.relative_path), capacity_storage_route(spec)


def _resolve_lineage_path(root: Path, value: object, label: str) -> Path:
    text = _require_text(value, label)
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved.exists() or not path.is_absolute():
        return resolved

    relocation_path = root / "configs" / "local_artifact_relocations.json"
    if not relocation_path.is_file():
        return resolved
    try:
        relocation_config = _read_json(relocation_path)
        _require(
            relocation_config.get("schema_version")
            == "local_artifact_relocations_v1",
            "unsupported local artifact relocation schema",
        )
        relocations = list(relocation_config.get("relocations") or [])
        for relocation in relocations:
            item = dict(relocation)
            old_root = Path(_require_text(item.get("old_root"), "old_root"))
            new_root = Path(_require_text(item.get("new_root"), "new_root"))
            try:
                suffix = path.relative_to(old_root)
            except ValueError:
                continue
            candidate = (new_root / suffix).resolve()
            if candidate.exists():
                return candidate
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        raise BlockedSourceError(
            f"local artifact relocations cannot be loaded: {exc}"
        ) from exc
    return resolved


def _safe_join(root: Path, relative: str, label: str) -> Path:
    path = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BlockedSourceError(f"{label} escapes its airport root") from exc
    return path


def _reject_duplicate_materialized_locators(specs: Iterable[R03SourceSpec]) -> None:
    seen: set[tuple[str, str, str | None]] = set()
    for spec in specs:
        locator = (
            spec.airport_id,
            spec.archive_relative_path or spec.relative_path,
            spec.archive_member,
        )
        _require(locator not in seen, f"duplicate source locator {locator}")
        seen.add(locator)


@contextmanager
def _authenticated_content_spool(
    source: VerifiedR03Source,
) -> Iterator[BinaryIO]:
    spec = source.spec
    row, airport_roots, recovery = _source_opening_arguments(source)
    with tempfile.SpooledTemporaryFile(
        max_size=_SOURCE_SPOOL_MEMORY_LIMIT, mode="w+b"
    ) as content:
        with open_source_binary(row, airport_roots, recovery) as (
            origin,
            opened_member,
        ):
            content_hash, actual_size, actual_crc = _copy_authenticated_bytes(
                origin, content
            )
        if source.materialization_kind == "extracted_file":
            _require(
                spec.r01_disposition != "read_verified_archive_member",
                "extracted materialization contradicts archive disposition",
            )
            _require(
                opened_member is None,
                "extracted source unexpectedly opened an archive member",
            )
        elif source.materialization_kind == "archive_member":
            _require(
                spec.r01_disposition == "read_verified_archive_member",
                "archive materialization contradicts source disposition",
            )
            member = _normalize_relative_path(spec.archive_member, "archive member")
            _require(
                opened_member is not None
                and _normalize_relative_path(opened_member, "opened archive member")
                == member,
                "opened archive member does not match verified routing",
            )
            member_size = _require_nonnegative_int(
                spec.member_size_bytes, "archive member size"
            )
            member_crc = str(spec.member_crc32 or "").lower()
            _require(
                _CRC32_RE.fullmatch(member_crc) is not None,
                "invalid archive member CRC32",
            )
            _require(
                actual_size == member_size,
                f"archive member byte count mismatch for {member}",
            )
            _require(
                actual_crc == member_crc,
                f"archive member computed CRC32 mismatch for {member}",
            )
        else:
            raise BlockedSourceError(
                f"unsupported materialization kind {source.materialization_kind}"
            )
        _require_hash_match(
            source.source_content_sha256,
            content_hash,
            "enumerated source content",
        )
        _require_hash_match(
            spec.extracted_sha256,
            content_hash,
            "registry source content",
        )
        content.seek(0)
        yield content


def _source_opening_arguments(
    source: VerifiedR03Source,
) -> tuple[
    dict[str, object],
    dict[str, Path],
    dict[tuple[str, str], dict[str, str]],
]:
    spec = source.spec
    materialized_relative = (
        spec.archive_relative_path
        if source.materialization_kind == "archive_member"
        else spec.relative_path
    )
    normalized = _normalize_relative_path(
        materialized_relative, "materialized relative path"
    )
    airport_root = source.resolved_path.resolve()
    for _ in PurePosixPath(normalized).parts:
        airport_root = airport_root.parent
    row: dict[str, object] = {
        "airport_id": spec.airport_id.lower(),
        "relative_path": spec.relative_path,
        "r01_disposition": spec.r01_disposition,
    }
    recovery: dict[tuple[str, str], dict[str, str]] = {}
    if spec.r01_disposition == "read_verified_archive_member":
        recovery[(spec.airport_id.lower(), spec.relative_path)] = {
            "archive_relative_path": str(spec.archive_relative_path),
            "member_path": str(spec.archive_member),
        }
    return row, {spec.airport_id.lower(): airport_root}, recovery


@contextmanager
def _text_view(content: BinaryIO) -> Iterator[TextIO]:
    content.seek(0)
    handle = TextIOWrapper(
        content, encoding="utf-8-sig", errors="replace", newline=""
    )
    try:
        yield handle
    finally:
        handle.detach()


def _conflict_schema() -> pa.Schema:
    return pa.schema([
        pa.field("canonical_report_id", pa.string(), nullable=False),
        pa.field("airport_id", pa.string(), nullable=False),
        pa.field("partition_year", pa.int32(), nullable=False),
        pa.field("partition_month", pa.int8(), nullable=False),
        pa.field("duplicate_classification", pa.string(), nullable=False),
        pa.field("core_state_difference_bits", pa.uint8(), nullable=False),
        pa.field("metadata_difference_bits", pa.uint8(), nullable=False),
        pa.field("group_size", pa.int64(), nullable=False),
        pa.field("selected_raw_row_id", pa.string(), nullable=False),
    ])


def _bad_rows_schema() -> pa.Schema:
    return pa.schema([
        pa.field("raw_row_id", pa.string(), nullable=False),
        pa.field("airport_id", pa.string(), nullable=False),
        pa.field("source_file_id", pa.string(), nullable=False),
        pa.field("source_relative_path", pa.string(), nullable=False),
        pa.field("archive_member", pa.string(), nullable=True),
        pa.field("source_content_sha256", pa.string(), nullable=False),
        pa.field("logical_record_ordinal", pa.int64(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("source_sequence", pa.int64(), nullable=False),
        pa.field("raw_tokens", pa.list_(pa.string()), nullable=False),
        pa.field("raw_payload", pa.string(), nullable=False),
        pa.field("redaction_status", pa.string(), nullable=False),
        pa.field("parser_status", pa.string(), nullable=False),
        pa.field("reason_code", pa.string(), nullable=False),
        pa.field("parse_quality_bits", pa.uint64(), nullable=False),
    ])


def _schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _timestamp_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError("natural-key timestamp must be UTC")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _natural_key_text(row: ParsedRow) -> str:
    return canonical_json({
        "airport_id": row.airport_id,
        "aircraft_id": row.aircraft_id,
        "report_timestamp_utc": _timestamp_z(row.report_timestamp_utc),
    }).decode("utf-8")


def _validate_partition_components(
    airport_id: object, partition_year: object, partition_month: object,
) -> tuple[str, int, int]:
    if airport_id not in _AIRPORT_ORDER:
        raise ValueError(f"airport_id is not frozen for R03: {airport_id!r}")
    if isinstance(partition_year, bool) or not isinstance(partition_year, int) or not 1 <= partition_year <= 9999:
        raise ValueError("partition_year must be in 1..9999")
    if isinstance(partition_month, bool) or not isinstance(partition_month, int) or not 1 <= partition_month <= 12:
        raise ValueError("partition_month must be in 1..12")
    return str(airport_id), partition_year, partition_month


def _validate_parsed_partition(row: ParsedRow) -> tuple[str, int, int]:
    airport, year, month = _validate_partition_components(
        row.airport_id, row.partition_year, row.partition_month
    )
    _timestamp_z(row.report_timestamp_utc)
    if year != row.report_timestamp_utc.year or month != row.report_timestamp_utc.month:
        raise ValueError("partition year/month must match the UTC report timestamp")
    return airport, year, month


def _contained_target(root: Path, relative: Path) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"output target escapes output_root: {relative.as_posix()}") from exc
    return target


def _row_from_batch(batch: pa.RecordBatch, index: int) -> dict[str, object]:
    row: dict[str, object] = {}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for column, name in enumerate(batch.schema.names):
        scalar = batch.column(column)[index]
        field_type = batch.schema.field(column).type
        if pa.types.is_timestamp(field_type) and field_type.tz == "UTC" and scalar.is_valid:
            row[name] = epoch + timedelta(microseconds=int(scalar.value))
        else:
            row[name] = scalar.as_py()
    return row


def _parsed_from_mapping(row: Mapping[str, object]) -> ParsedRow:
    return ParsedRow(**{field: row[field] for field in ParsedRow.__dataclass_fields__})


def _parsed_rank_key(row: ParsedRow) -> tuple[object, ...]:
    completeness = sum(
        getattr(row, field) is not None
        for field in ("lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad", "age_s", "range_m", "bearing_rad", "tail_or_callsign")
    )
    return (
        row.airport_id, row.aircraft_id, row.report_timestamp_utc,
        *age_rank(row.age_s), -completeness, stable_source_order(row),
    )


def _iter_parsed(path: Path) -> Iterator[ParsedRow]:
    parquet = pq.ParquetFile(path)
    try:
        if not parquet.schema_arrow.equals(parsed_rows_schema(), check_metadata=False):
            raise ValueError(f"parsed shard run schema mismatch: {path}")
        for batch in parquet.iter_batches(batch_size=65_536):
            for index in range(batch.num_rows):
                yield _parsed_from_mapping(_row_from_batch(batch, index))
    finally:
        parquet.close()


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def spill_parsed_batches(
    batches: Iterable[pa.RecordBatch], scratch_root: Path, shard_count: int,
    max_rows_per_run: int, shard_hash_version: str = _SHARD_HASH_VERSION,
) -> tuple[ShardRun, ...]:
    """Write bounded, sorted and hashed shard runs without whole-shard materialization."""

    if shard_hash_version != _SHARD_HASH_VERSION:
        raise ValueError(f"unsupported shard_hash_version: {shard_hash_version!r}")
    if isinstance(max_rows_per_run, bool) or not isinstance(max_rows_per_run, int) or max_rows_per_run <= 0:
        raise ValueError("max_rows_per_run must be a positive integer")
    shard_for_key(("_", "_", datetime(1970, 1, 1, tzinfo=timezone.utc)), shard_count, shard_hash_version)
    root = Path(scratch_root)
    root.mkdir(parents=True, exist_ok=True)
    run_indices = [0] * shard_count
    descriptors: list[ShardRun] = []
    schema = parsed_rows_schema()
    for source_batch in batches:
        if not isinstance(source_batch, pa.RecordBatch):
            raise TypeError("batches must contain pyarrow.RecordBatch values")
        if not source_batch.schema.equals(schema, check_metadata=False):
            raise ValueError("parsed batch schema mismatch")
        for offset in range(0, source_batch.num_rows, max_rows_per_run):
            batch = source_batch.slice(offset, min(max_rows_per_run, source_batch.num_rows - offset))
            buckets: dict[int, list[int]] = {}
            row_cache: dict[int, ParsedRow] = {}
            for index in range(batch.num_rows):
                parsed = _parsed_from_mapping(_row_from_batch(batch, index))
                _validate_parsed_partition(parsed)
                row_cache[index] = parsed
                shard_id = shard_for_key(
                    (parsed.airport_id, parsed.aircraft_id, parsed.report_timestamp_utc),
                    shard_count, shard_hash_version,
                )
                buckets.setdefault(shard_id, []).append(index)
            for shard_id in sorted(buckets):
                indices = sorted(buckets[shard_id], key=lambda item: _parsed_rank_key(row_cache[item]))
                table = pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64()))
                run_index = run_indices[shard_id]
                run_indices[shard_id] += 1
                relative = Path("shard_runs") / f"shard-{shard_id:05d}" / f"run-{run_index:06d}.parquet"
                path = root / relative
                _write_parquet_atomic(path, table)
                ordered_rows = [row_cache[index] for index in indices]
                descriptors.append(ShardRun(
                    schema_version="r03_shard_run_v1", artifact_role="parsed_rows_shard_run",
                    shard_hash_version=shard_hash_version, shard_count=shard_count,
                    shard_id=shard_id, run_index=run_index, relative_path=relative.as_posix(),
                    row_count=len(indices), schema_fingerprint=_schema_fingerprint(schema),
                    sha256=_sha256_file(path), sort_contract=_SHARD_SORT_CONTRACT,
                    min_natural_key=_natural_key_text(ordered_rows[0]),
                    max_natural_key=_natural_key_text(ordered_rows[-1]), finalized=True, path=path,
                ))
    return tuple(sorted(descriptors, key=lambda run: (run.shard_id, run.run_index)))


class _BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, buffer_rows: int = 1024):
        self.path, self.schema, self.buffer_rows = path, schema, buffer_rows
        self.rows: list[Mapping[str, object]] = []
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(path.name + ".tmp")
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self.closed = False

    def append(self, row: Mapping[str, object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows)
            self.rows.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            try:
                self.writer.close()
            except Exception:
                pass
            self.closed = True
        self.temporary.unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)


class GroupArtifactWriters:
    """Atomic, schema-checked writers for the four shard-level Task 4 artifacts."""

    def __init__(
        self, output_root: Path, shard_id: int = 0, *, shard_count: int = 256,
        shard_hash_version: str = _SHARD_HASH_VERSION,
        final_sort_version: str = _FINAL_SORT_VERSION,
    ) -> None:
        if shard_hash_version != _SHARD_HASH_VERSION:
            raise ValueError(f"unsupported shard_hash_version: {shard_hash_version!r}")
        if final_sort_version != _FINAL_SORT_VERSION:
            raise ValueError(f"unsupported final_sort_version: {final_sort_version!r}")
        if (
            isinstance(shard_count, bool) or not isinstance(shard_count, int)
            or shard_count <= 0 or isinstance(shard_id, bool)
            or not isinstance(shard_id, int) or not 0 <= shard_id < shard_count
        ):
            raise ValueError("shard_id must be inside shard_count")
        self.output_root, self.shard_id, self.shard_count = Path(output_root), shard_id, shard_count
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.final_base = _contained_target(self.output_root, Path(f"shard-{shard_id:05d}"))
        if self.final_base.exists():
            raise FileExistsError(f"finalized shard already exists: {self.final_base}")
        base = Path(tempfile.mkdtemp(
            prefix=f".shard-{shard_id:05d}.staging-", dir=self.output_root
        ))
        self.staging_base = base
        self._parsed = _BufferedParquetWriter(base / "parsed_rows.parquet", parsed_rows_schema())
        self._unique = _BufferedParquetWriter(base / "unique_reports.parquet", unique_reports_schema())
        self._membership = _BufferedParquetWriter(base / "report_membership.parquet", membership_schema())
        self._conflicts = _BufferedParquetWriter(base / "duplicate_conflicts.parquet", _conflict_schema())
        self.artifact_set: ShardArtifactSet | None = None

    def write_parsed(self, row: ParsedRow) -> None:
        self._parsed.append(asdict(row))

    def write_resolved(self, resolved: object) -> None:
        self._unique.append(resolved.unique_row)
        for member in resolved.membership_rows:
            self._membership.append(member)
        if resolved.conflict_row is not None:
            self._conflicts.append(resolved.conflict_row)

    def write_unique(self, row: Mapping[str, object]) -> None:
        self._unique.append(row)

    def write_membership(self, row: Mapping[str, object]) -> None:
        self._membership.append(row)

    def write_conflict(self, row: Mapping[str, object]) -> None:
        self._conflicts.append(row)

    def finalize(self, counts: ArtifactCounts) -> ShardArtifactSet:
        try:
            for writer in (self._parsed, self._unique, self._membership, self._conflicts):
                writer.close()
            actual = (self._parsed.count, self._unique.count, self._membership.count, self._conflicts.count)
            declared = (counts.parsed_rows, counts.unique_reports, counts.report_membership, counts.duplicate_conflicts)
            if actual != declared:
                raise ValueError(f"artifact counts mismatch: actual={actual}, declared={declared}")
            staging_paths = (self._parsed.path, self._unique.path, self._membership.path, self._conflicts.path)
            schemas = (parsed_rows_schema(), unique_reports_schema(), membership_schema(), _conflict_schema())
            for path, schema in zip(staging_paths, schemas):
                parquet = pq.ParquetFile(path)
                try:
                    if not parquet.schema_arrow.equals(schema, check_metadata=False):
                        raise ValueError(f"staged shard artifact schema mismatch: {path}")
                finally:
                    parquet.close()
                _sha256_file(path)
            if self.final_base.exists():
                raise FileExistsError(f"finalized shard already exists: {self.final_base}")
            final_paths = tuple(self.final_base / path.name for path in staging_paths)
            artifact_set = ShardArtifactSet(self.shard_id, *final_paths, counts)
            os.replace(self.staging_base, self.final_base)
            self.artifact_set = artifact_set
            return self.artifact_set
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        for writer in (self._parsed, self._unique, self._membership, self._conflicts):
            writer.abort()
        if self.staging_base.is_dir():
            shutil.rmtree(self.staging_base)


class _GroupAccumulator:
    """Bound one duplicate group and spill overflow to a non-checkpoint Parquet file."""

    def __init__(self, root: Path, limit: int):
        self.root, self.limit = root, limit
        self.rows: list[ParsedRow] = []
        self.path: Path | None = None
        self.writer: pq.ParquetWriter | None = None

    def append(self, row: ParsedRow) -> None:
        if len(self.rows) >= self.limit and self.writer is None:
            self.root.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(suffix=".parquet", dir=self.root, delete=False)
            handle.close()
            self.path = Path(handle.name)
            self.writer = pq.ParquetWriter(self.path, parsed_rows_schema(), compression="zstd")
            self._flush()
        self.rows.append(row)
        if self.writer is not None and len(self.rows) >= self.limit:
            self._flush()

    def _flush(self) -> None:
        assert self.writer is not None
        self.writer.write_table(pa.Table.from_pylist([asdict(row) for row in self.rows], schema=parsed_rows_schema()))
        self.rows.clear()

    def finish(self) -> tuple[list[ParsedRow] | None, Path | None]:
        if self.writer is None:
            return self.rows, None
        if self.rows:
            self._flush()
        self.writer.close()
        return None, self.path


def _iter_group(rows: list[ParsedRow] | None, path: Path | None) -> Iterator[ParsedRow]:
    if rows is not None:
        yield from rows
    else:
        assert path is not None
        yield from _iter_parsed(path)


def _large_group_rows(
    rows: list[ParsedRow] | None, path: Path | None, writers: GroupArtifactWriters,
    policies: CanonicalPolicies,
) -> tuple[int, bool]:
    iterator = _iter_group(rows, path)
    winner = next(iterator)
    winner_rank = age_rank(winner.age_s)
    winner_complete = sum(getattr(winner, field) is not None for field in (
        "lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad", "age_s", "range_m", "bearing_rad", "tail_or_callsign"
    ))
    group_size = 1
    same_age = 1
    same_complete = 1
    core_fields = ("lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad")
    metadata_fields = ("age_s", "range_m", "bearing_rad", "tail_or_callsign", "altitude_source_gnss")
    first_core = [getattr(winner, field) for field in core_fields]
    first_metadata = [getattr(winner, field) for field in metadata_fields]
    core_bits = metadata_bits = 0
    for row in iterator:
        group_size += 1
        complete = sum(getattr(row, field) is not None for field in (
            "lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad", "age_s", "range_m", "bearing_rad", "tail_or_callsign"
        ))
        if age_rank(row.age_s) == winner_rank:
            same_age += 1
            if complete == winner_complete:
                same_complete += 1
        for index, field in enumerate(core_fields):
            if getattr(row, field) != first_core[index]:
                core_bits |= 1 << index
        for index, field in enumerate(metadata_fields):
            if getattr(row, field) != first_metadata[index]:
                metadata_bits |= 1 << index
    if group_size == 1:
        reason = "only_member"
    elif same_age == 1:
        reason = "min_nonnegative_age" if winner_rank[0] == 0 else "closest_zero_negative_age"
    elif same_complete == 1:
        reason = "max_completeness_after_age_tie"
    else:
        reason = "stable_source_order_after_full_tie"
    if group_size == 1:
        classification = "singleton"
    elif core_bits and metadata_bits:
        classification = "core_and_metadata_conflict"
    elif core_bits:
        classification = "core_state_conflict"
    elif metadata_bits:
        classification = "metadata_conflict"
    else:
        classification = "identical"
    base = resolve_group([winner], policies)
    unique = dict(base.unique_row)
    unique.update({
        "group_size": group_size, "is_duplicate": group_size > 1,
        "duplicate_classification": classification,
        "core_state_difference_bits": core_bits, "metadata_difference_bits": metadata_bits,
        "duplicate_conflict_flag": core_bits != 0,
        "metadata_conflict_flag": metadata_bits != 0,
        "any_difference_flag": bool(core_bits or metadata_bits),
    })
    writers.write_unique(unique)
    for rank, row in enumerate(_iter_group(rows, path), start=1):
        member = dict(resolve_group([row], policies).membership_rows[0])
        member.update({
            "group_size": group_size, "dedup_rank": rank, "selected": rank == 1,
            "selection_reason": reason if rank == 1 else "not_selected",
            "core_state_difference_bits": core_bits,
            "metadata_difference_bits": metadata_bits,
        })
        writers.write_membership(member)
    if core_bits or metadata_bits:
        conflict = {
            "canonical_report_id": unique["canonical_report_id"], "airport_id": winner.airport_id,
            "partition_year": winner.partition_year, "partition_month": winner.partition_month,
            "duplicate_classification": classification,
            "core_state_difference_bits": core_bits, "metadata_difference_bits": metadata_bits,
            "group_size": group_size, "selected_raw_row_id": winner.raw_row_id,
        }
        writers.write_conflict(conflict)
    return group_size, bool(core_bits or metadata_bits)


def _validate_shard_runs(
    runs: Sequence[ShardRun], writers: GroupArtifactWriters,
) -> tuple[Path, ...]:
    identities: set[tuple[int, int]] = set()
    paths: set[Path] = set()
    relative_paths: set[str] = set()
    authenticated: list[Path] = []
    expected_schema = parsed_rows_schema()
    expected_fingerprint = _schema_fingerprint(expected_schema)
    for run in runs:
        if not isinstance(run, ShardRun):
            raise TypeError("resolve_sorted_shard requires ShardRun descriptors")
        if run.finalized is not True:
            raise ValueError("shard run descriptor is not finalized")
        if run.schema_version != "r03_shard_run_v1":
            raise ValueError("shard run schema_version mismatch")
        if run.artifact_role != "parsed_rows_shard_run":
            raise ValueError("shard run artifact_role mismatch")
        if run.shard_hash_version != _SHARD_HASH_VERSION:
            raise ValueError("shard run shard_hash_version mismatch")
        if (
            isinstance(run.shard_count, bool) or not isinstance(run.shard_count, int)
            or run.shard_count <= 0 or run.shard_count != writers.shard_count
        ):
            raise ValueError("shard run shard_count mismatch")
        if (
            isinstance(run.shard_id, bool) or not isinstance(run.shard_id, int)
            or run.shard_id < 0 or run.shard_id != writers.shard_id
        ):
            raise ValueError("shard run shard_id mismatch")
        if isinstance(run.run_index, bool) or not isinstance(run.run_index, int) or run.run_index < 0:
            raise ValueError("shard run run_index must be nonnegative")
        identity = (run.shard_id, run.run_index)
        if identity in identities:
            raise ValueError("duplicate shard run identity")
        identities.add(identity)
        if (
            not run.relative_path or "\\" in run.relative_path
            or PurePosixPath(run.relative_path).is_absolute()
            or ".." in PurePosixPath(run.relative_path).parts
            or PurePosixPath(run.relative_path).as_posix() != run.relative_path
        ):
            raise ValueError("invalid shard run relative_path")
        if run.relative_path in relative_paths:
            raise ValueError("duplicate shard run relative_path")
        relative_paths.add(run.relative_path)
        path = Path(run.path).resolve()
        relative_parts = PurePosixPath(run.relative_path).parts
        if len(path.parts) < len(relative_parts) or tuple(path.parts[-len(relative_parts):]) != relative_parts:
            raise ValueError("shard run path does not match relative_path")
        if path in paths:
            raise ValueError("duplicate shard run path")
        paths.add(path)
        if not path.is_file():
            raise ValueError(f"shard run is missing: {path}")
        if isinstance(run.row_count, bool) or not isinstance(run.row_count, int) or run.row_count <= 0:
            raise ValueError("completed shard run must be nonempty")
        if run.schema_fingerprint != expected_fingerprint:
            raise ValueError("shard run schema fingerprint mismatch")
        if run.sort_contract != _SHARD_SORT_CONTRACT:
            raise ValueError("shard run sort contract mismatch")
        if _SHA256_RE.fullmatch(run.sha256) is None or _sha256_file(path) != run.sha256:
            raise ValueError("shard run SHA-256 mismatch")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != run.row_count:
            raise ValueError("shard run row_count mismatch")
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError("shard run Arrow schema mismatch")
        first: ParsedRow | None = None
        last: ParsedRow | None = None
        previous: tuple[object, ...] | None = None
        observed = 0
        for row in _iter_parsed(path):
            _validate_parsed_partition(row)
            key = _parsed_rank_key(row)
            if previous is not None and key < previous:
                raise ValueError(f"parsed shard run is not sorted: {path}")
            if shard_for_key(
                (row.airport_id, row.aircraft_id, row.report_timestamp_utc),
                writers.shard_count,
            ) != writers.shard_id:
                raise ValueError(f"parsed row does not belong to writer shard {writers.shard_id}")
            first = row if first is None else first
            last = row
            previous = key
            observed += 1
        if observed != run.row_count or first is None or last is None:
            raise ValueError("shard run observed row count mismatch")
        if run.min_natural_key != _natural_key_text(first) or run.max_natural_key != _natural_key_text(last):
            raise ValueError("shard run natural-key bounds mismatch")
        authenticated.append(path)
    return tuple(authenticated)


def resolve_sorted_shard(
    runs: Sequence[ShardRun], writers: GroupArtifactWriters,
    max_group_rows_in_memory: int, *, policies: CanonicalPolicies | None = None,
) -> ArtifactCounts:
    """Authenticate, k-way merge, and resolve one shard of sorted runs."""

    try:
        return _resolve_sorted_shard_authenticated(
            _validate_shard_runs(runs, writers), writers, max_group_rows_in_memory,
            policies,
        )
    except Exception:
        writers.abort()
        group_temporary = writers.output_root / ".group-tmp"
        if group_temporary.is_dir():
            shutil.rmtree(group_temporary, ignore_errors=True)
        raise


def _resolve_sorted_shard_authenticated(
    run_paths: Sequence[Path], writers: GroupArtifactWriters,
    max_group_rows_in_memory: int, policies: CanonicalPolicies | None = None,
) -> ArtifactCounts:

    if isinstance(max_group_rows_in_memory, bool) or not isinstance(max_group_rows_in_memory, int) or max_group_rows_in_memory <= 0:
        raise ValueError("max_group_rows_in_memory must be a positive integer")
    def validated(path: Path) -> Iterator[ParsedRow]:
        previous: tuple[object, ...] | None = None
        for parsed in _iter_parsed(path):
            key = _parsed_rank_key(parsed)
            if previous is not None and key < previous:
                raise ValueError(f"parsed shard run is not sorted: {path}")
            if shard_for_key(
                (parsed.airport_id, parsed.aircraft_id, parsed.report_timestamp_utc),
                writers.shard_count,
            ) != writers.shard_id:
                raise ValueError(f"parsed row does not belong to writer shard {writers.shard_id}")
            previous = key
            yield parsed

    streams = [validated(path) for path in run_paths]
    heap: list[tuple[tuple[object, ...], int, ParsedRow]] = []
    for index, stream in enumerate(streams):
        try:
            row = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (_parsed_rank_key(row), index, row))
    policies = policies or CanonicalPolicies.from_project(Path(__file__).resolve().parents[2])
    current_key: tuple[str, str, datetime] | None = None
    accumulator: _GroupAccumulator | None = None
    parsed_count = unique_count = membership_count = conflicts = duplicate_excess = 0

    def finish_group(group: _GroupAccumulator | None) -> None:
        nonlocal unique_count, membership_count, conflicts, duplicate_excess
        if group is None:
            return
        in_memory, spill_path = group.finish()
        if in_memory is not None:
            resolved = resolve_group(in_memory, policies)
            writers.write_resolved(resolved)
            size = len(in_memory)
            conflict = resolved.conflict_row is not None
        else:
            size, conflict = _large_group_rows(None, spill_path, writers, policies)
        unique_count += 1
        membership_count += size
        duplicate_excess += size - 1
        conflicts += int(conflict)
        if spill_path is not None:
            spill_path.unlink(missing_ok=True)

    while heap:
        _, stream_index, row = heapq.heappop(heap)
        key = (row.airport_id, row.aircraft_id, row.report_timestamp_utc)
        if current_key != key:
            finish_group(accumulator)
            accumulator = _GroupAccumulator(writers.output_root / ".group-tmp", max_group_rows_in_memory)
            current_key = key
        assert accumulator is not None
        accumulator.append(row)
        writers.write_parsed(row)
        parsed_count += 1
        try:
            following = next(streams[stream_index])
        except StopIteration:
            pass
        else:
            heapq.heappush(heap, (_parsed_rank_key(following), stream_index, following))
    finish_group(accumulator)
    counts = ArtifactCounts(parsed_count, unique_count, membership_count, conflicts, duplicate_excess)
    writers.finalize(counts)
    group_temporary = writers.output_root / ".group-tmp"
    try:
        group_temporary.rmdir()
    except OSError:
        pass
    return counts


def _iter_mappings(path: Path, schema: pa.Schema) -> Iterator[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    try:
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ValueError(f"artifact schema mismatch: {path}")
        for batch in parquet.iter_batches(batch_size=65_536):
            for index in range(batch.num_rows):
                yield _row_from_batch(batch, index)
    finally:
        parquet.close()


def _final_sort_key(role: str, row: Mapping[str, object]) -> tuple[object, ...]:
    if role == "parsed_rows":
        parsed = _parsed_from_mapping(row)
        return (
            parsed.airport_id, parsed.aircraft_id, parsed.report_timestamp_utc,
            stable_source_order(parsed),
        )
    if role == "unique_reports":
        return (row["airport_id"], row["aircraft_id"], row["report_timestamp_utc"])
    if role == "report_membership":
        return (row["canonical_report_id"], row["dedup_rank"], row["raw_row_id"])
    if role == "duplicate_conflicts":
        return (row["canonical_report_id"],)
    raise ValueError(f"unknown artifact role: {role}")


def _validate_external_report_partitions(
    shard_outputs: Sequence[ShardArtifactSet],
) -> None:
    """Join M/C partition claims to U on disk before final-output creation."""

    handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    handle.close()
    path = Path(handle.name)
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE unique_reports(
                report_id TEXT NOT NULL, airport TEXT NOT NULL,
                year INTEGER NOT NULL, month INTEGER NOT NULL
            );
            CREATE TABLE report_refs(
                role TEXT NOT NULL, report_id TEXT NOT NULL, airport TEXT NOT NULL,
                year INTEGER NOT NULL, month INTEGER NOT NULL
            );
        """)
        for shard in shard_outputs:
            connection.executemany(
                "INSERT INTO unique_reports VALUES (?,?,?,?)",
                (
                    (report_id, airport, year, month)
                    for report_id, airport, year, month in _scan_columns(
                        [shard.unique_reports],
                        ("canonical_report_id", "airport_id", "partition_year", "partition_month"),
                    )
                ),
            )
            for role, artifact in (
                ("membership", shard.report_membership),
                ("conflict", shard.duplicate_conflicts),
            ):
                connection.executemany(
                    "INSERT INTO report_refs VALUES (?,?,?,?,?)",
                    (
                        (role, report_id, airport, year, month)
                        for report_id, airport, year, month in _scan_columns(
                            [artifact],
                            ("canonical_report_id", "airport_id", "partition_year", "partition_month"),
                        )
                    ),
                )
        connection.executescript("""
            CREATE INDEX unique_report_partition ON unique_reports(report_id);
            CREATE INDEX referenced_report_partition ON report_refs(report_id);
        """)
        duplicate_unique = connection.execute(
            "SELECT COUNT(*) FROM (SELECT report_id FROM unique_reports GROUP BY report_id HAVING COUNT(*)<>1)"
        ).fetchone()[0]
        bad_reference = connection.execute("""
            SELECT COUNT(*)
            FROM report_refs r
            LEFT JOIN unique_reports u ON r.report_id=u.report_id
            WHERE u.report_id IS NULL OR r.airport<>u.airport OR r.year<>u.year OR r.month<>u.month
        """).fetchone()[0]
        if duplicate_unique or bad_reference:
            raise ValueError("membership/conflict report reference partition mismatch")
    finally:
        connection.close()
        path.unlink(missing_ok=True)


def merge_final_partitions(
    shard_outputs: Sequence[ShardArtifactSet], output_root: Path,
    max_rows_per_run: int, final_sort_version: str = _FINAL_SORT_VERSION,
) -> tuple[FinalPartition, ...]:
    """Externally merge shard contributions into deterministic logical partitions."""

    if final_sort_version != _FINAL_SORT_VERSION:
        raise ValueError(f"unsupported final_sort_version: {final_sort_version!r}")
    if isinstance(max_rows_per_run, bool) or not isinstance(max_rows_per_run, int) or max_rows_per_run <= 0:
        raise ValueError("max_rows_per_run must be a positive integer")
    shard_ids: set[int] = set()
    artifact_paths: set[Path] = set()
    for shard in shard_outputs:
        if not isinstance(shard, ShardArtifactSet):
            raise TypeError("shard_outputs must contain ShardArtifactSet values")
        if isinstance(shard.shard_id, bool) or not isinstance(shard.shard_id, int) or shard.shard_id < 0:
            raise ValueError("shard output shard_id must be nonnegative")
        if shard.shard_id in shard_ids:
            raise ValueError("duplicate shard_id in final merge")
        shard_ids.add(shard.shard_id)
        for path in (
            shard.parsed_rows, shard.unique_reports,
            shard.report_membership, shard.duplicate_conflicts,
        ):
            resolved = Path(path).resolve()
            if resolved in artifact_paths:
                raise ValueError("duplicate artifact path in final merge")
            artifact_paths.add(resolved)
            if not resolved.is_file():
                raise ValueError(f"missing shard artifact: {resolved}")
    _validate_external_report_partitions(shard_outputs)
    root = Path(output_root)
    temporary_root = _contained_target(root, Path(".merge-tmp"))
    roles = {
        "parsed_rows": (parsed_rows_schema(), lambda item: item.parsed_rows),
        "unique_reports": (unique_reports_schema(), lambda item: item.unique_reports),
        "report_membership": (membership_schema(), lambda item: item.report_membership),
        "duplicate_conflicts": (_conflict_schema(), lambda item: item.duplicate_conflicts),
    }
    # Preflight every partition component and target before creating any merge run.
    for role, (schema, path_for) in roles.items():
        for shard in sorted(shard_outputs, key=lambda value: value.shard_id):
            for row in _iter_mappings(path_for(shard), schema):
                airport, year, month = _validate_partition_components(
                    row["airport_id"], row["partition_year"], row["partition_month"]
                )
                if role in {"parsed_rows", "unique_reports"}:
                    timestamp = row["report_timestamp_utc"]
                    if not isinstance(timestamp, datetime):
                        raise ValueError("report timestamp must be a UTC datetime")
                    _timestamp_z(timestamp)
                    if year != timestamp.year or month != timestamp.month:
                        raise ValueError("partition year/month must match the UTC report timestamp")
                _contained_target(
                    root,
                    Path(role) / f"airport_id={airport}" / f"year={year:04d}"
                    / f"month={month:02d}" / "part-00000.parquet",
                )
    run_paths: dict[tuple[str, str, int, int], list[Path]] = {}
    run_number: dict[tuple[str, str, int, int], int] = {}
    for role, (schema, path_for) in roles.items():
        for shard in sorted(shard_outputs, key=lambda value: value.shard_id):
            iterator = _iter_mappings(path_for(shard), schema)
            while True:
                chunk: list[dict[str, object]] = []
                try:
                    for _ in range(max_rows_per_run):
                        chunk.append(next(iterator))
                except StopIteration:
                    pass
                if not chunk:
                    break
                partitions: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
                for row in chunk:
                    airport, year, month = _validate_partition_components(
                        row["airport_id"], row["partition_year"], row["partition_month"]
                    )
                    key = (role, airport, year, month)
                    partitions.setdefault(key, []).append(row)
                for key, rows in partitions.items():
                    rows.sort(key=lambda value: _final_sort_key(role, value))
                    index = run_number.get(key, 0)
                    run_number[key] = index + 1
                    run = _contained_target(
                        root,
                        Path(".merge-tmp") / role / key[1]
                        / f"{key[2]:04d}-{key[3]:02d}" / f"run-{index:06d}.parquet",
                    )
                    _write_parquet_atomic(run, pa.Table.from_pylist(rows, schema=schema))
                    run_paths.setdefault(key, []).append(run)
                if len(chunk) < max_rows_per_run:
                    break

    finals: list[FinalPartition] = []
    for key in sorted(run_paths):
        role, airport, year, month = key
        schema = roles[role][0]
        streams = [_iter_mappings(path, schema) for path in run_paths[key]]
        heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
        for index, stream in enumerate(streams):
            try:
                row = next(stream)
            except StopIteration:
                continue
            heapq.heappush(heap, (_final_sort_key(role, row), index, row))
        relative = Path(role) / f"airport_id={airport}" / f"year={year:04d}" / f"month={month:02d}" / "part-00000.parquet"
        target = _contained_target(root, relative)
        writer = _BufferedParquetWriter(target, schema, min(max_rows_per_run, 65_536))
        count = 0
        while heap:
            _, index, row = heapq.heappop(heap)
            writer.append(row)
            count += 1
            try:
                following = next(streams[index])
            except StopIteration:
                continue
            heapq.heappush(heap, (_final_sort_key(role, following), index, following))
        writer.close()
        finals.append(FinalPartition(
            artifact_role=role, airport_id=airport, partition_year=year,
            partition_month=month, path=target, relative_path=relative.as_posix(),
            row_count=count, schema_fingerprint=_schema_fingerprint(schema),
            sha256=_sha256_file(target), sort_contract=_FINAL_SORT_CONTRACTS[role],
        ))
    if temporary_root.is_dir():
        shutil.rmtree(temporary_root)
    return tuple(finals)


_CONSERVATION_INVARIANTS = (
    "retained_ids_unique", "parsed_ids_unique", "bad_ids_unique",
    "parsed_bad_ids_disjoint", "retained_ids_equal_parsed_union_bad",
    "membership_ids_unique", "membership_ids_equal_parsed_ids",
    "membership_reports_reference_unique", "unique_natural_keys_unique",
    "one_selected_membership_per_unique_report",
    "membership_group_size_matches_unique",
    "membership_count_equals_sum_group_size",
    "parsed_count_equals_unique_plus_duplicate_excess",
)


def _artifact_paths(root: Path, role: str) -> list[Path]:
    directory = root / role
    if directory.is_dir():
        return sorted(directory.rglob("*.parquet"), key=lambda path: path.as_posix())
    direct = root / f"{role}.parquet"
    if direct.is_file():
        return [direct]
    return sorted(
        (
            path for path in root.rglob(f"{role}.parquet")
            if ".merge-tmp" not in path.parts
            and not any(part.startswith(".shard-") for part in path.parts)
        ),
        key=lambda path: path.as_posix(),
    )


def stream_final_artifact_rows(
    output_root: Path, role: str, columns: Sequence[str],
) -> Iterator[dict[str, object]]:
    """Yield final-ledger rows batchwise for reporting and external verification.

    This deliberately exposes no table-returning convenience API: callers that
    need an audit must retain only their aggregate state, not an entire ledger.
    """
    paths = _artifact_paths(Path(output_root), role)
    if not paths:
        raise ValueError(f"missing final artifact role: {role}")
    for path in paths:
        parquet = pq.ParquetFile(path)
        try:
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path} is missing requested columns {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=65_536, columns=list(columns)):
                for index in range(batch.num_rows):
                    yield _row_from_batch(batch, index)
        finally:
            parquet.close()


def _scan_columns(paths: Sequence[Path], columns: Sequence[str]) -> Iterator[tuple[object, ...]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        try:
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path} is missing conservation columns {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=65_536, columns=list(columns)):
                for index in range(batch.num_rows):
                    row = _row_from_batch(batch, index)
                    yield tuple(row[column] for column in columns)
        finally:
            parquet.close()


def verify_layered_conservation(
    output_root: Path, sources: Sequence[VerifiedR03Source], scratch_root: Path,
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None,
) -> ConservationResult:
    """Independently verify the exact R/P/B/U/M ledgers through a disk-backed join."""

    if selected_ordinals is not None and set(selected_ordinals) != {
        source.spec.source_file_id for source in sources
    }:
        raise ValueError("selected conservation mapping does not match sources")
    root = Path(output_root)
    required_schemas = {
        "parsed_rows": parsed_rows_schema(),
        "bad_rows": _bad_rows_schema(),
        "unique_reports": unique_reports_schema(),
        "report_membership": membership_schema(),
    }
    artifact_paths = {role: _artifact_paths(root, role) for role in required_schemas}
    conflict_paths = _artifact_paths(root, "duplicate_conflicts")
    try:
        artifacts_valid = all(artifact_paths[role] for role in required_schemas)
        if artifacts_valid:
            for role, schema in required_schemas.items():
                for path in artifact_paths[role]:
                    parquet = pq.ParquetFile(path)
                    try:
                        if not parquet.schema_arrow.equals(schema, check_metadata=False):
                            artifacts_valid = False
                            break
                    finally:
                        parquet.close()
        if artifacts_valid:
            for path in conflict_paths:
                parquet = pq.ParquetFile(path)
                try:
                    if not parquet.schema_arrow.equals(_conflict_schema(), check_metadata=False):
                        artifacts_valid = False
                        break
                finally:
                    parquet.close()
    except (OSError, ValueError, pa.ArrowException):
        artifacts_valid = False
    if not artifacts_valid:
        invariants = {name: False for name in _CONSERVATION_INVARIANTS}
        return ConservationResult(
            False,
            {
                "retained_rows": 0, "parsed_rows": 0, "bad_rows": 0,
                "report_membership": 0, "unique_reports": 0,
                "duplicate_excess": 0,
            },
            invariants,
            _CONSERVATION_INVARIANTS,
        )

    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    database = tempfile.NamedTemporaryFile(suffix=".sqlite3", dir=scratch, delete=False)
    database.close()
    db_path = Path(database.name)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE retained(raw_id TEXT NOT NULL);
            CREATE TABLE parsed(raw_id TEXT NOT NULL, report_id TEXT NOT NULL);
            CREATE TABLE bad(raw_id TEXT NOT NULL);
            CREATE TABLE membership(raw_id TEXT NOT NULL, report_id TEXT NOT NULL,
                selected INTEGER NOT NULL, dedup_rank INTEGER NOT NULL, group_size INTEGER NOT NULL,
                airport TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL);
            CREATE TABLE unique_reports(report_id TEXT NOT NULL, selected_raw_id TEXT NOT NULL,
                airport TEXT NOT NULL, aircraft TEXT NOT NULL, timestamp TEXT NOT NULL,
                group_size INTEGER NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL);
            CREATE TABLE conflicts(report_id TEXT NOT NULL, airport TEXT NOT NULL,
                year INTEGER NOT NULL, month INTEGER NOT NULL);
        """)
        def retained_rows() -> Iterator[tuple[str]]:
            for source in sources:
                records = (
                    enumerate_source(source)
                    if selected_ordinals is None
                    else enumerate_selected_source(
                        source, selected_ordinals[source.spec.source_file_id]
                    )
                )
                for record in records:
                    yield (record.raw_row_id,)

        connection.executemany("INSERT INTO retained VALUES (?)", retained_rows())
        connection.executemany(
            "INSERT INTO parsed VALUES (?,?)",
            (
                (raw_id, canonical_report_id(airport, aircraft, timestamp))
                for raw_id, airport, aircraft, timestamp in _scan_columns(
                    artifact_paths["parsed_rows"],
                    ("raw_row_id", "airport_id", "aircraft_id", "report_timestamp_utc"),
                )
            ),
        )
        connection.executemany(
            "INSERT INTO bad VALUES (?)",
            _scan_columns(artifact_paths["bad_rows"], ("raw_row_id",)),
        )
        connection.executemany(
            "INSERT INTO membership VALUES (?,?,?,?,?,?,?,?)",
            (
                (raw_id, report_id, int(bool(selected)), dedup_rank, group_size, airport, year, month)
                for raw_id, report_id, selected, dedup_rank, group_size, airport, year, month in _scan_columns(
                    artifact_paths["report_membership"],
                    (
                        "raw_row_id", "canonical_report_id", "selected", "dedup_rank",
                        "group_size", "airport_id", "partition_year", "partition_month",
                    ),
                )
            ),
        )
        connection.executemany(
            "INSERT INTO unique_reports VALUES (?,?,?,?,?,?,?,?)",
            (
                (report_id, selected_raw_id, airport, aircraft, _timestamp_z(timestamp), group_size, year, month)
                for report_id, selected_raw_id, airport, aircraft, timestamp, group_size, year, month in _scan_columns(
                    artifact_paths["unique_reports"],
                    (
                        "canonical_report_id", "selected_raw_row_id", "airport_id", "aircraft_id",
                        "report_timestamp_utc", "group_size", "partition_year", "partition_month",
                    ),
                )
            ),
        )
        connection.executemany(
            "INSERT INTO conflicts VALUES (?,?,?,?)",
            _scan_columns(
                conflict_paths,
                ("canonical_report_id", "airport_id", "partition_year", "partition_month"),
            ),
        )
        connection.executescript("""
            CREATE INDEX retained_id ON retained(raw_id);
            CREATE INDEX parsed_id ON parsed(raw_id);
            CREATE INDEX bad_id ON bad(raw_id);
            CREATE INDEX membership_raw ON membership(raw_id);
            CREATE INDEX membership_report ON membership(report_id);
            CREATE INDEX unique_report ON unique_reports(report_id);
            CREATE INDEX unique_natural ON unique_reports(airport,aircraft,timestamp);
            CREATE INDEX conflict_report ON conflicts(report_id);
        """)
        count = lambda table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        distinct = lambda table, column: int(connection.execute(f"SELECT COUNT(DISTINCT {column}) FROM {table}").fetchone()[0])
        counts = {
            "retained_rows": count("retained"), "parsed_rows": count("parsed"),
            "bad_rows": count("bad"), "report_membership": count("membership"),
            "unique_reports": count("unique_reports"),
        }
        duplicate_excess = int(connection.execute(
            "SELECT COALESCE(SUM(group_size - 1),0) FROM unique_reports"
        ).fetchone()[0])
        counts["duplicate_excess"] = duplicate_excess
        scalar = lambda query: int(connection.execute(query).fetchone()[0])
        invariants = {
            "retained_ids_unique": counts["retained_rows"] == distinct("retained", "raw_id"),
            "parsed_ids_unique": counts["parsed_rows"] == distinct("parsed", "raw_id"),
            "bad_ids_unique": counts["bad_rows"] == distinct("bad", "raw_id"),
            "parsed_bad_ids_disjoint": scalar("SELECT COUNT(*) FROM parsed JOIN bad USING(raw_id)") == 0,
            "retained_ids_equal_parsed_union_bad": (
                scalar("SELECT COUNT(*) FROM (SELECT raw_id FROM retained EXCEPT SELECT raw_id FROM (SELECT raw_id FROM parsed UNION SELECT raw_id FROM bad))") == 0
                and scalar("SELECT COUNT(*) FROM (SELECT raw_id FROM (SELECT raw_id FROM parsed UNION SELECT raw_id FROM bad) EXCEPT SELECT raw_id FROM retained)") == 0
                and counts["retained_rows"] == counts["parsed_rows"] + counts["bad_rows"]
            ),
            "membership_ids_unique": counts["report_membership"] == distinct("membership", "raw_id"),
            "membership_ids_equal_parsed_ids": (
                scalar("SELECT COUNT(*) FROM (SELECT raw_id FROM parsed EXCEPT SELECT raw_id FROM membership)") == 0
                and scalar("SELECT COUNT(*) FROM (SELECT raw_id FROM membership EXCEPT SELECT raw_id FROM parsed)") == 0
            ),
            "membership_reports_reference_unique": (
                count("unique_reports") == distinct("unique_reports", "report_id")
                and scalar("SELECT COUNT(*) FROM membership m LEFT JOIN unique_reports u ON m.report_id=u.report_id WHERE u.report_id IS NULL") == 0
                and scalar("SELECT COUNT(*) FROM membership m LEFT JOIN parsed p ON m.raw_id=p.raw_id WHERE p.raw_id IS NULL OR m.report_id<>p.report_id") == 0
                and scalar("SELECT COUNT(*) FROM membership m LEFT JOIN unique_reports u ON m.report_id=u.report_id WHERE u.report_id IS NULL OR m.airport<>u.airport OR m.year<>u.year OR m.month<>u.month") == 0
            ),
            "unique_natural_keys_unique": count("unique_reports") == scalar(
                "SELECT COUNT(*) FROM (SELECT airport,aircraft,timestamp FROM unique_reports GROUP BY airport,aircraft,timestamp)"
            ) and scalar(
                "SELECT COUNT(*) FROM unique_reports WHERE year<>CAST(substr(timestamp,1,4) AS INTEGER) OR month<>CAST(substr(timestamp,6,2) AS INTEGER)"
            ) == 0,
            "one_selected_membership_per_unique_report": scalar(
                "SELECT COUNT(*) FROM unique_reports u LEFT JOIN (SELECT report_id,SUM(selected) AS n FROM membership GROUP BY report_id) m ON u.report_id=m.report_id WHERE COALESCE(m.n,0) <> 1"
            ) == 0 and scalar(
                "SELECT COUNT(*) FROM membership WHERE selected <> (dedup_rank = 1)"
            ) == 0 and scalar(
                "SELECT COUNT(*) FROM unique_reports u LEFT JOIN membership m ON u.report_id=m.report_id AND m.selected=1 WHERE m.raw_id IS NULL OR u.selected_raw_id<>m.raw_id"
            ) == 0,
            "membership_group_size_matches_unique": scalar(
                "SELECT COUNT(*) FROM unique_reports u LEFT JOIN (SELECT report_id,COUNT(*) AS n,MIN(group_size) AS lo,MAX(group_size) AS hi FROM membership GROUP BY report_id) m ON u.report_id=m.report_id WHERE COALESCE(m.n,-1)<>u.group_size OR m.lo<>u.group_size OR m.hi<>u.group_size"
            ) == 0,
            "membership_count_equals_sum_group_size": counts["report_membership"] == scalar(
                "SELECT COALESCE(SUM(group_size),0) FROM unique_reports"
            ),
            "parsed_count_equals_unique_plus_duplicate_excess": counts["parsed_rows"] == counts["unique_reports"] + duplicate_excess,
        }
        conflict_references_invalid = scalar(
            "SELECT COUNT(*) FROM conflicts c LEFT JOIN unique_reports u ON c.report_id=u.report_id WHERE u.report_id IS NULL OR c.airport<>u.airport OR c.year<>u.year OR c.month<>u.month"
        ) != 0
        if count("conflicts") and count("unique_reports") != distinct("unique_reports", "report_id"):
            conflict_references_invalid = True
        if conflict_references_invalid:
            invariants = {name: False for name in _CONSERVATION_INVARIANTS}
        failures = tuple(name for name in _CONSERVATION_INVARIANTS if not invariants[name])
        return ConservationResult(not failures, counts, invariants, failures)
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)


# Task 5 owns publication in a focused module, while this lazy compatibility
# boundary keeps the R03 runner as the public entry point established by Tasks
# 1--4 without an import-time cycle.
_TASK5_EXPORTS = {
    "R03RunIdentity", "R03RunResult", "RunCheckpoint", "build_canonical_points",
    "verify_canonical_manifest", "verify_current_pointer",
}


def __getattr__(name: str) -> object:
    if name not in _TASK5_EXPORTS:
        raise AttributeError(name)
    from airspace_complexity import build_canonical_points as task5
    return getattr(task5, name)
