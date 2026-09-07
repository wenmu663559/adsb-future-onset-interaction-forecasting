"""Pure deterministic R03 capacity-selection manifest construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from airspace_complexity.canonical_points import canonical_json
from airspace_complexity.r03_runner import R03SourceRegistry, R03SourceSpec, capacity_stratum_key


CAPACITY_SCHEMA_VERSION = "r03_task8_lite_capacity_selection_v1"
CAPACITY_SELECTION_SALT = "r03-task8-lite-capacity-v1"
CAPACITY_CAP_PER_AIRPORT = 250_000


def _key_bytes(key: tuple[str, ...]) -> tuple[bytes, ...]:
    return tuple(part.encode("utf-8") for part in key)


@dataclass(frozen=True)
class _RankedSource:
    source_file_id: str
    source_sequence: int
    authenticated_row_count: int


@dataclass(frozen=True)
class _AllocatedSource:
    source_file_id: str
    source_sequence: int
    authenticated_row_count: int
    quota: int


def _source_rank(source: _RankedSource) -> tuple[bytes, int]:
    return (
        hashlib.sha256(
            CAPACITY_SELECTION_SALT.encode() + b"\x00" + source.source_file_id.encode("utf-8")
        ).digest(),
        source.source_sequence,
    )


def _rank_ordered_sources(row_counts: Sequence[int]) -> tuple[_RankedSource, ...]:
    """Small pure-fixture adapter used to exercise source-prefix allocation."""
    return tuple(
        _RankedSource(f"source{sequence}", sequence, count)
        for sequence, count in enumerate(row_counts)
    )


def _allocate_hamilton(
    total: int,
    populations: Mapping[tuple[str, ...], int],
    capacities: Mapping[tuple[str, ...], int],
    order: Sequence[tuple[str, ...]],
) -> dict[tuple[str, ...], int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("Hamilton total exceeds capacity")
    if set(populations) != set(capacities) or not set(order).issubset(populations):
        raise ValueError("Hamilton keys disagree")
    if any(populations[key] < 0 or capacities[key] < 0 for key in order):
        raise ValueError("Hamilton inputs must be nonnegative")
    if total == 0:
        return {key: 0 for key in order}
    active = [key for key in order if populations[key] > 0 and capacities[key] > 0]
    if not active:
        raise ValueError("Hamilton has no positive population")
    allocation = {key: 0 for key in order}
    remaining = total
    while remaining:
        budget = remaining
        denominator = sum(populations[key] for key in active)
        if denominator <= 0:
            raise ValueError("Hamilton exhausted capacity")
        floors = {key: min(capacities[key] - allocation[key], (budget * populations[key]) // denominator) for key in active}
        floor_total = sum(floors.values())
        for key, amount in floors.items():
            allocation[key] += amount
        remaining -= floor_total
        if remaining == 0:
            break
        def compare_remainders(left: tuple[str, ...], right: tuple[str, ...]) -> int:
            left_remainder = budget * populations[left] % denominator
            right_remainder = budget * populations[right] % denominator
            # Express the comparison as cross multiplication so the
            # deterministic integer rule remains explicit if denominators
            # later become stratum-specific.
            comparison = right_remainder * denominator - left_remainder * denominator
            if comparison:
                return comparison
            return (_key_bytes(left) > _key_bytes(right)) - (_key_bytes(left) < _key_bytes(right))

        ranked = sorted(
            (key for key in active if allocation[key] < capacities[key]),
            key=cmp_to_key(compare_remainders),
        )
        for key in ranked:
            if remaining == 0:
                break
            allocation[key] += 1
            remaining -= 1
        active = [key for key in active if allocation[key] < capacities[key]]
    return allocation


def _allocate_strata(
    total: int, populations: Mapping[tuple[str, str, str], int]
) -> dict[tuple[str, str, str], int]:
    order = tuple(sorted(populations, key=_key_bytes))
    positive = tuple(key for key in order if populations[key] > 0)
    if total < len(positive) or total > sum(populations.values()):
        raise ValueError("stratum total is infeasible")
    result = {key: 0 for key in order}
    for key in positive:
        result[key] = 1
    remaining = total - len(positive)
    if remaining:
        residual_capacities = {key: populations[key] - 1 for key in positive}
        positive_populations = {key: populations[key] for key in positive}
        extra = _allocate_hamilton(remaining, positive_populations, residual_capacities, positive)
        for key, amount in extra.items():
            result[key] += amount
    return result


def _allocate_capacity_sources(
    sources: Sequence[_RankedSource], stratum_quota: int
) -> tuple[_AllocatedSource, ...]:
    if stratum_quota < 0:
        raise ValueError("source quota is invalid")
    ranked = tuple(sorted(sources, key=_source_rank))
    prefix: list[_RankedSource] = []
    covered = 0
    for source in ranked:
        if source.authenticated_row_count < 0:
            raise ValueError("source count is invalid")
        prefix.append(source)
        covered += source.authenticated_row_count
        if covered >= stratum_quota:
            break
    if covered < stratum_quota:
        raise ValueError("source prefix cannot cover quota")
    keys = tuple((item.source_file_id, str(item.source_sequence)) for item in prefix)
    populations = {key: item.authenticated_row_count for key, item in zip(keys, prefix)}
    quotas = _allocate_hamilton(stratum_quota, populations, populations, keys)
    return tuple(
        _AllocatedSource(item.source_file_id, item.source_sequence, item.authenticated_row_count, quotas[key])
        for key, item in zip(keys, prefix)
    )


def _systematic_midpoints(population: int, quota: int) -> tuple[int, ...]:
    if population <= 0 or quota <= 0 or quota > population:
        raise ValueError("midpoint population/quota is invalid")
    values = tuple(((2 * j + 1) * population) // (2 * quota) + 1 for j in range(quota))
    if len(set(values)) != quota or values[-1] > population:
        raise ValueError("midpoint selection is not distinct and bounded")
    return values


@dataclass(frozen=True)
class SelectedSource:
    airport_id: str
    source_file_id: str
    source_sequence: int
    relative_path: str
    r01_disposition: str
    source_content_sha256: str
    archive_member: str | None
    authenticated_row_count: int
    lineage_month: str
    storage_route: str
    selected_ordinals: tuple[int, ...]

    @property
    def selected_count(self) -> int:
        return len(self.selected_ordinals)


@dataclass(frozen=True)
class SelectionStratum:
    airport_id: str
    lineage_month: str
    storage_route: str
    population_rows: int
    quota: int
    selected_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class CapacitySelectionManifest:
    schema_version: str
    selection_salt: str
    population_kind: str
    cap_per_airport: int
    parent_lineage: Mapping[str, object]
    population_rows_by_airport: tuple[tuple[str, int], ...]
    selected_rows_by_airport: tuple[tuple[str, int], ...]
    strata: tuple[SelectionStratum, ...]
    sources: tuple[SelectedSource, ...]

    @property
    def selection_id(self) -> str:
        return hashlib.sha256(capacity_selection_bytes(self)).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_salt": self.selection_salt,
            "population_kind": self.population_kind,
            "cap_per_airport": self.cap_per_airport,
            "parent_lineage": dict(self.parent_lineage),
            "population_rows_by_airport": [list(item) for item in self.population_rows_by_airport],
            "selected_rows_by_airport": [list(item) for item in self.selected_rows_by_airport],
            "strata": [{"airport_id": item.airport_id, "lineage_month": item.lineage_month, "storage_route": item.storage_route, "population_rows": item.population_rows, "quota": item.quota, "selected_source_ids": list(item.selected_source_ids)} for item in self.strata],
            "sources": [{"airport_id": item.airport_id, "source_file_id": item.source_file_id, "source_sequence": item.source_sequence, "relative_path": item.relative_path, "r01_disposition": item.r01_disposition, "source_content_sha256": item.source_content_sha256, "archive_member": item.archive_member, "authenticated_row_count": item.authenticated_row_count, "lineage_month": item.lineage_month, "storage_route": item.storage_route, "selected_count": item.selected_count, "selected_ordinals": list(item.selected_ordinals)} for item in self.sources],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CapacitySelectionManifest":
        top_fields = {
            "schema_version", "selection_salt", "population_kind", "cap_per_airport",
            "parent_lineage", "population_rows_by_airport", "selected_rows_by_airport",
            "strata", "sources",
        }
        source_fields = {
            "airport_id", "source_file_id", "source_sequence", "relative_path",
            "r01_disposition", "source_content_sha256", "archive_member",
            "authenticated_row_count", "lineage_month", "storage_route",
            "selected_count", "selected_ordinals",
        }
        stratum_fields = {
            "airport_id", "lineage_month", "storage_route", "population_rows",
            "quota", "selected_source_ids",
        }
        lineage_fields = {
            "registry_id", "raw_manifest_sha256", "archive_comparison_sha256",
            "r02_run_manifest_sha256", "r02_decision_sha256",
            "partition_manifest_sha256_by_airport", "source_sequence_manifest_sha256",
        }

        def text(value: object) -> str:
            if not isinstance(value, str):
                raise ValueError("selection manifest text is invalid")
            return value

        def integer(value: object) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("selection manifest integer is invalid")
            return value

        def pairs(value: object) -> tuple[tuple[str, int], ...]:
            if not isinstance(value, list):
                raise ValueError("selection manifest pairs are invalid")
            result = []
            for item in value:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("selection manifest pair is invalid")
                result.append((text(item[0]), integer(item[1])))
            return tuple(result)

        def archive_member(value: object) -> str | None:
            if value == {"$r03_type": "null"}:
                return None
            return text(value)

        def text_list(value: object) -> tuple[str, ...]:
            if not isinstance(value, list):
                raise ValueError("selection manifest text list is invalid")
            return tuple(text(item) for item in value)

        def integer_list(value: object) -> tuple[int, ...]:
            if not isinstance(value, list):
                raise ValueError("selection manifest integer list is invalid")
            return tuple(integer(item) for item in value)

        try:
            if not isinstance(record, Mapping) or set(record) != top_fields:
                raise ValueError("selection manifest fields are invalid")
            lineage = record["parent_lineage"]
            if not isinstance(lineage, Mapping) or set(lineage) != lineage_fields:
                raise ValueError("selection manifest lineage is invalid")
            if not isinstance(lineage["partition_manifest_sha256_by_airport"], list):
                raise ValueError("selection manifest partition hashes are invalid")
            for item in lineage["partition_manifest_sha256_by_airport"]:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("selection manifest partition hash is invalid")
                text(item[0]); text(item[1])
            for field in lineage_fields - {"partition_manifest_sha256_by_airport"}:
                text(lineage[field])
            if not isinstance(record["strata"], list) or not isinstance(record["sources"], list):
                raise ValueError("selection manifest collections are invalid")
            strata = tuple(SelectionStratum(
                text(item["airport_id"]), text(item["lineage_month"]), text(item["storage_route"]), integer(item["population_rows"]), integer(item["quota"]), text_list(item["selected_source_ids"])
            ) for item in record["strata"] if isinstance(item, Mapping) and set(item) == stratum_fields)
            if len(strata) != len(record["strata"]):
                raise ValueError("selection manifest stratum is invalid")
            sources = tuple(SelectedSource(
                text(item["airport_id"]), text(item["source_file_id"]), integer(item["source_sequence"]), text(item["relative_path"]), text(item["r01_disposition"]), text(item["source_content_sha256"]), archive_member(item["archive_member"]), integer(item["authenticated_row_count"]), text(item["lineage_month"]), text(item["storage_route"]), integer_list(item["selected_ordinals"])
            ) for item in record["sources"] if isinstance(item, Mapping) and set(item) == source_fields)
            if len(sources) != len(record["sources"]):
                raise ValueError("selection manifest source is invalid")
            manifest = cls(text(record["schema_version"]), text(record["selection_salt"]), text(record["population_kind"]), integer(record["cap_per_airport"]), dict(lineage), pairs(record["population_rows_by_airport"]), pairs(record["selected_rows_by_airport"]), strata, sources)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("selection manifest has invalid shape") from exc
        if any(integer(item["selected_count"]) != source.selected_count for item, source in zip(record["sources"], sources)):
            raise ValueError("selection manifest selected_count mismatch")
        return manifest


def _parent_lineage(registry: R03SourceRegistry) -> dict[str, object]:
    return {
        "registry_id": registry.registry_id,
        "raw_manifest_sha256": registry.raw_manifest_sha256,
        "archive_comparison_sha256": registry.archive_comparison_sha256,
        "r02_run_manifest_sha256": registry.r02_run_manifest_sha256,
        "r02_decision_sha256": registry.r02_decision_sha256,
        "partition_manifest_sha256_by_airport": [list(item) for item in registry.partition_manifest_sha256_by_airport],
        "source_sequence_manifest_sha256": registry.source_sequence_manifest_sha256,
    }


def build_capacity_selection(registry: R03SourceRegistry) -> CapacitySelectionManifest:
    specs = tuple(sorted(registry.sources, key=lambda item: (item.airport_id, item.source_sequence, item.source_file_id)))
    source_strata = {id(spec): capacity_stratum_key(spec) for spec in specs}
    populations = {key: 0 for key in sorted(set(source_strata.values()), key=_key_bytes)}
    for spec in specs:
        populations[source_strata[id(spec)]] += spec.authenticated_row_count
    airports = tuple(sorted({spec.airport_id for spec in specs}, key=lambda item: item.encode("utf-8")))
    population_rows = tuple((airport, sum(spec.authenticated_row_count for spec in specs if spec.airport_id == airport)) for airport in airports)
    selected_rows = tuple((airport, min(CAPACITY_CAP_PER_AIRPORT, population)) for airport, population in population_rows)
    quotas: dict[tuple[str, str, str], int] = {}
    for airport, total in selected_rows:
        airport_populations = {key: value for key, value in populations.items() if key[0] == airport}
        quotas.update(_allocate_strata(total, airport_populations))
    source_quotas: dict[int, int] = {id(spec): 0 for spec in specs}
    strata: list[SelectionStratum] = []
    for key in sorted(populations, key=_key_bytes):
        members = tuple(spec for spec in specs if source_strata[id(spec)] == key)
        allocation = _allocate_capacity_sources(tuple(_RankedSource(spec.source_file_id, spec.source_sequence, spec.authenticated_row_count) for spec in members), quotas[key]) if quotas[key] else ()
        quota_by_id = {item.source_file_id: item.quota for item in allocation}
        for spec in members:
            source_quotas[id(spec)] = quota_by_id.get(spec.source_file_id, 0)
        strata.append(SelectionStratum(*key, populations[key], quotas[key], tuple(item.source_file_id for item in allocation if item.quota)))
    selected_sources = tuple(SelectedSource(spec.airport_id, spec.source_file_id, spec.source_sequence, spec.relative_path, spec.r01_disposition, str(spec.extracted_sha256), spec.archive_member, spec.authenticated_row_count, *source_strata[id(spec)][1:], _systematic_midpoints(spec.authenticated_row_count, source_quotas[id(spec)]) if source_quotas[id(spec)] else ()) for spec in specs)
    return CapacitySelectionManifest(CAPACITY_SCHEMA_VERSION, CAPACITY_SELECTION_SALT, "capacity_sample", CAPACITY_CAP_PER_AIRPORT, _parent_lineage(registry), population_rows, selected_rows, tuple(strata), selected_sources)


def capacity_selection_bytes(manifest: CapacitySelectionManifest) -> bytes:
    return canonical_json(manifest.to_record())


def load_and_verify_capacity_selection(path: Path, registry: R03SourceRegistry) -> CapacitySelectionManifest:
    raw = Path(path).read_bytes()
    try:
        manifest = CapacitySelectionManifest.from_record(json.loads(raw))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("selection manifest cannot be loaded") from exc
    if raw != capacity_selection_bytes(manifest):
        raise ValueError("selection manifest bytes are not canonical")
    expected = build_capacity_selection(registry)
    if manifest != expected or raw != capacity_selection_bytes(expected):
        raise ValueError("selection manifest does not bind to registry")
    return manifest
