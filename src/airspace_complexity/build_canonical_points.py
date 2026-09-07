"""Identity and publication boundary for the R03 canonical point ledgers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import ast
import builtins
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import heapq
import io
from itertools import zip_longest
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import sys
import tempfile
from types import MappingProxyType
import uuid
from typing import Iterable, Iterator, Literal, Mapping, Protocol, Sequence
import zipfile

import pyarrow
import pyarrow.parquet as pq
import yaml

from airspace_complexity.canonical_points import (
    BadRow,
    CanonicalPolicies,
    ParsedRow,
    R03RunParameters,
    canonical_json,
    canonical_report_id,
    convert_record,
    membership_schema,
    parsed_rows_schema,
    unique_reports_schema,
)
from airspace_complexity.r03_runner import (
    ArtifactCounts,
    FinalPartition,
    GroupArtifactWriters,
    R03SourceRegistry,
    ShardArtifactSet,
    ShardRun,
    SourceEnumerationResult,
    VerifiedR03Source,
    _bad_rows_schema,
    _artifact_paths,
    _conflict_schema,
    _final_sort_key,
    _iter_mappings,
    _parsed_rank_key,
    _schema_fingerprint,
    _write_parquet_atomic,
    enumerate_selected_source,
    enumerate_source,
    enumeration_manifest,
    merge_final_partitions,
    resolve_sorted_shard,
    spill_parsed_batches,
    stream_final_artifact_rows,
    verify_layered_conservation,
    verify_source,
)
from airspace_complexity.r02_runner import _load_airport_roots
from airspace_complexity.r03_selection import (
    CapacitySelectionManifest,
    build_capacity_selection,
    capacity_selection_bytes,
    load_and_verify_capacity_selection,
)


_IDENTITY_SCHEMA_VERSION = "r03_dataset_identity_v2"
_BUILD_SCHEMA_VERSION = "r03_build_identity_v1"
_CHECKPOINT_SCHEMA_VERSION = "r03_checkpoint_compatibility_v1"


class R03BuildObserver(Protocol):
    """Narrow event sink; the builder does not import capacity orchestration."""

    def sample(self, event: str, root: Path) -> None:
        raise NotImplementedError



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_canonical(record: object) -> str:
    # ``canonical_json`` deliberately accepts a top-level mapping only; the
    # wrapper preserves that canonical encoding for sequence payloads too.
    payload = record if isinstance(record, Mapping) else {"value": record}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _module_origin(module_name: str) -> Path:
    module = sys.modules.get(module_name)
    origin = getattr(module, "__file__", None) if module else None
    if not origin:
        raise RuntimeError(f"cannot locate module source for {module_name}")
    path = Path(origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"loaded module origin is not a regular file: {module_name}")
    spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not spec_origin or Path(spec_origin).resolve() != path:
        raise RuntimeError(f"module origin metadata is incoherent: {module_name}")
    executed: set[Path] = set()
    def add_code(candidate: object) -> None:
        code = getattr(inspect.unwrap(candidate), "__code__", None)
        filename = getattr(code, "co_filename", None)
        if isinstance(filename, str) and not filename.startswith("<"):
            executed.add(Path(filename).resolve())
    for value in vars(module).values():
        if inspect.isfunction(value) and value.__module__ == module_name:
            add_code(value)
        elif inspect.isclass(value) and value.__module__ == module_name:
            for member_name, member in vars(value).items():
                # Dataclass-generated dunder methods legitimately originate in
                # stdlib helpers.  They are metadata, not executed R03 logic.
                if member_name.startswith("__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member) and member.__module__ == module_name:
                    add_code(member)
    if not executed or executed != {path}:
        raise RuntimeError(f"executed code disagrees with module origin: {module_name}")
    return path


def _loaded_project_origins(
    project_root: Path, module_names: Mapping[str, str],
) -> dict[str, Path]:
    """Return authenticated origins of the modules actually imported to run R03.

    Identity must never be based on a convenient source-tree copy: an operator
    could execute one implementation while labelling the output with another.
    Only loaded regular files contained by the trusted project root are valid.
    """
    root = Path(project_root).resolve()
    origins: dict[str, Path] = {}
    for logical_name, module_name in sorted(module_names.items()):
        origin = _module_origin(module_name)
        try:
            origin.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"loaded module origin escapes expected project root: {module_name}"
            ) from exc
        if origin.relative_to(root).as_posix() != logical_name:
            raise RuntimeError(
                f"loaded module origin does not match its frozen project path: {module_name}"
            )
        origins[logical_name] = origin
    return origins


def _bundle_hash(
    project_root: Path,
    entries: Mapping[str, str],
) -> str:
    """Hash the bytes of authenticated loaded modules under stable names."""
    return _sha256_canonical(
        {
            logical_name: _sha256_file(origin)
            for logical_name, origin in _loaded_project_origins(project_root, entries).items()
        }
    )


def _symbol_bundle_dependency_closure(
    project_root: Path,
    module_name: str,
    logical_name: str,
    symbols: Sequence[str],
) -> dict[str, object]:
    """Return an auditable global dependency closure of executed symbols.

    ``r03_runner`` intentionally contains both scientific decisions and file
    production machinery.  Hashing its complete file in either identity would
    either make a writer refactor change the dataset identity or, worse, omit a
    scientific change.  The explicit symbol lists at the call site are the
    reviewable boundary between the two contracts.
    """
    def assigned_names(node: ast.AST) -> tuple[str, ...]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.arg):
                names.add(child.arg)
            elif isinstance(child, ast.alias):
                names.add(child.asname or child.name.split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
        return tuple(names)

    builtin_names = frozenset(dir(builtins))

    def global_references(node: ast.AST) -> tuple[str, ...]:
        local = set(assigned_names(node))
        names: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            if child.id not in local and child.id not in builtin_names:
                names.add(child.id)
        return tuple(sorted(names))

    @dataclass(frozen=True)
    class _ModuleContext:
        name: str
        origin: Path
        logical_path: str
        module: object
        tree: ast.Module
        source_lines: tuple[str, ...]
        definitions: Mapping[str, ast.AST]
        imports: Mapping[str, tuple[ast.AST, ast.alias]]

    root = Path(project_root).resolve()
    contexts: dict[str, _ModuleContext] = {}
    active_imports: set[tuple[str, str]] = set()

    def is_project_module(name: str) -> bool:
        return name == "airspace_complexity" or name.startswith("airspace_complexity.")

    def context_for(name: str) -> _ModuleContext:
        cached = contexts.get(name)
        if cached is not None:
            return cached
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        origin = _module_origin(name)
        try:
            logical_path = origin.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"loaded module origin escapes expected project root: {name}"
            ) from exc
        try:
            source_text = origin.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=str(origin))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"cannot parse authenticated module source: {name}") from exc
        definitions: dict[str, ast.AST] = {}
        imports: dict[str, tuple[ast.AST, ast.alias]] = {}
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[item.name] = item
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                for assigned in assigned_names(item):
                    definitions[assigned] = item
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                for alias in item.names:
                    bound = alias.asname or (
                        alias.name if isinstance(item, ast.ImportFrom)
                        else alias.name.split(".")[0]
                    )
                    imports[bound] = (item, alias)
        context = _ModuleContext(
            name=name,
            origin=origin,
            logical_path=logical_path,
            module=module,
            tree=tree,
            source_lines=tuple(source_text.splitlines(keepends=True)),
            definitions=definitions,
            imports=imports,
        )
        contexts[name] = context
        return context

    def node_source(context: _ModuleContext, node: ast.AST) -> str:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not isinstance(start, int) or not isinstance(end, int):
            raise RuntimeError(
                f"identity symbol has no stable source extent: {context.name}"
            )
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            decorator_lines = [
                getattr(decorator, "lineno", start) for decorator in decorators
            ]
            start = min([start, *decorator_lines])
        return "".join(context.source_lines[start - 1:end])

    def symbol_node(context: _ModuleContext, symbol: str) -> ast.AST:
        body: Sequence[ast.stmt] = context.tree.body
        found: ast.AST | None = None
        for index, component in enumerate(symbol.split(".")):
            if index == 0:
                found = context.definitions.get(component)
            else:
                found = next(
                    (
                        item for item in body
                        if isinstance(item, (
                            ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
                        )) and item.name == component
                    ),
                    None,
                )
            if found is None:
                raise RuntimeError(
                    f"authenticated module is missing frozen identity symbol "
                    f"{context.name}.{symbol}"
                )
            body = getattr(found, "body", ())
        return found

    def require_bound_object_origin(
        context: _ModuleContext, original_symbol: str, value: object,
    ) -> None:
        """Reject a runtime binding that does not execute the authenticated file."""
        if inspect.isfunction(value):
            filename = getattr(inspect.unwrap(value).__code__, "co_filename", None)
            if not isinstance(filename, str) or Path(filename).resolve() != context.origin:
                raise RuntimeError(
                    f"imported binding origin disagrees with authenticated source: "
                    f"{context.name}.{original_symbol}"
                )
        elif inspect.isclass(value):
            for member_name, member in vars(value).items():
                if member_name.startswith("__"):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member) and member.__module__ == context.name:
                    filename = getattr(member.__code__, "co_filename", None)
                    if not isinstance(filename, str) or Path(filename).resolve() != context.origin:
                        raise RuntimeError(
                            f"imported binding origin disagrees with authenticated source: "
                            f"{context.name}.{original_symbol}"
                        )

    def imported_module_name(context: _ModuleContext, node: ast.AST) -> str:
        if isinstance(node, ast.Import):
            raise RuntimeError("module imports must be handled without a member lookup")
        assert isinstance(node, ast.ImportFrom)
        if node.module is None and not node.level:
            raise RuntimeError(f"invalid import binding in {context.name}")
        if node.level:
            package = context.name.rpartition(".")[0]
            relative = "." * node.level + (node.module or "")
            return importlib.util.resolve_name(relative, package)
        assert node.module is not None
        return node.module

    def build_closure(
        context: _ModuleContext, roots: Sequence[str], *, nested: bool,
    ) -> dict[str, object]:
        definitions: dict[str, str] = {}
        imports: dict[str, str] = {}
        unclassified: set[str] = set()
        visited: set[str] = set()
        special_globals = {
            "__file__": _sha256_canonical(
                {"executed_module_origin": context.logical_path}
            ),
        }

        def imported_binding_digest(
            bound_name: str, binding: tuple[ast.AST, ast.alias],
        ) -> tuple[str, str]:
            node, alias = binding
            if isinstance(node, ast.Import):
                # ``import package.member`` binds ``package`` unless an
                # explicit alias is present.  Authenticate the object Python
                # actually placed in the consuming module, not the leaf used
                # only to ensure an attribute was loaded.
                source_module = alias.name if alias.asname else alias.name.split(".")[0]
                imported_symbol = "<module>"
            else:
                if alias.name == "*":
                    raise RuntimeError(
                        f"star import is not an auditable identity dependency: {context.name}"
                    )
                source_module = imported_module_name(context, node)
                imported_symbol = alias.name
            try:
                imported_module = importlib.import_module(source_module)
                expected_binding = (
                    imported_module if isinstance(node, ast.Import)
                    else getattr(imported_module, imported_symbol)
                )
            except (AttributeError, ImportError) as exc:
                raise RuntimeError(
                    f"authenticated imported binding is missing: "
                    f"{source_module}.{imported_symbol}"
                ) from exc
            try:
                consuming_binding = getattr(context.module, bound_name)
            except AttributeError as exc:
                raise RuntimeError(
                    f"consuming import binding is missing: "
                    f"{context.name}.{bound_name}"
                ) from exc
            if consuming_binding is not expected_binding:
                raise RuntimeError(
                    f"consuming import binding does not match authenticated "
                    f"source: {context.name}.{bound_name}"
                )
            if is_project_module(source_module):
                target = context_for(source_module)
                if isinstance(node, ast.Import):
                    target_closure = {
                        "module_sha256": _sha256_file(target.origin),
                    }
                else:
                    try:
                        value = getattr(target.module, imported_symbol)
                    except AttributeError as exc:
                        raise RuntimeError(
                            f"authenticated imported binding is missing: "
                            f"{source_module}.{imported_symbol}"
                        ) from exc
                    require_bound_object_origin(target, imported_symbol, value)
                    cycle_key = (source_module, imported_symbol)
                    if cycle_key in active_imports:
                        raise RuntimeError(
                            f"cyclic imported identity dependency: "
                            f"{source_module}.{imported_symbol}"
                        )
                    active_imports.add(cycle_key)
                    try:
                        target_closure = build_closure(
                            target, (imported_symbol,), nested=True,
                        )
                    finally:
                        active_imports.remove(cycle_key)
                target_record: Mapping[str, object] = {
                    "source_path": target.logical_path,
                    "closure": target_closure,
                }
            else:
                # Third-party/stdlib code is authenticated by the frozen
                # runtime environment record.  The binding still records the
                # exact module, exported member, and local name without
                # making harmless import formatting identity-bearing.
                target_record = {"runtime_module": source_module}
            binding_key = (
                f"{bound_name}<-{source_module}.{imported_symbol}"
            )
            return binding_key, _sha256_canonical({
                "source_module": source_module,
                "original_symbol": imported_symbol,
                "bound_name": bound_name,
                "target": target_record,
            })

        def visit(name: str, node: ast.AST) -> None:
            if name in visited:
                return
            visited.add(name)
            definitions[name] = hashlib.sha256(
                node_source(context, node).encode("utf-8")
            ).hexdigest()
            for reference in global_references(node):
                dependency = context.definitions.get(reference)
                if dependency is not None:
                    visit(reference, dependency)
                    continue
                binding = context.imports.get(reference)
                if binding is not None:
                    binding_key, binding_digest = imported_binding_digest(
                        reference, binding,
                    )
                    imports[binding_key] = binding_digest
                    continue
                if reference in special_globals:
                    definitions[reference] = special_globals[reference]
                    continue
                unclassified.add(reference)

        for symbol in roots:
            value: object = context.module
            for component in symbol.split("."):
                value = getattr(value, component, None)
                if value is None:
                    raise RuntimeError(
                        f"authenticated module is missing frozen identity symbol "
                        f"{context.name}.{symbol}"
                    )
            if nested:
                require_bound_object_origin(context, symbol, value)
            visit(symbol, symbol_node(context, symbol))
        return {
            "definitions": dict(sorted(definitions.items())),
            "imports": dict(sorted(imports.items())),
            "unclassified": tuple(sorted(unclassified)),
        }

    top_context = context_for(module_name)
    return build_closure(top_context, symbols, nested=False)


def _symbol_bundle_hash(
    project_root: Path,
    module_name: str,
    logical_name: str,
    symbols: Sequence[str],
) -> str:
    """Hash an authenticated symbol bundle and its explicit global closure."""
    closure = _symbol_bundle_dependency_closure(
        project_root, module_name, logical_name, symbols
    )
    return _closure_hash(logical_name, closure)


def _closure_hash(logical_name: str, closure: Mapping[str, object]) -> str:
    """Hash a checked dependency closure without losing its audit records."""
    unclassified = closure["unclassified"]
    if unclassified:
        raise RuntimeError(
            f"unclassified behavior-bearing globals in {logical_name}: {unclassified}"
        )
    return _sha256_canonical({
        "definitions": dict(closure["definitions"]),
        "imports": dict(closure["imports"]),
    })


def _identity_dependency_records(
    logical_name: str, closure: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    """Expose closure records in frozen identity using only logical paths."""
    unclassified = closure.get("unclassified")
    if unclassified:
        raise RuntimeError(
            f"unclassified behavior-bearing globals in {logical_name}: {unclassified}"
        )
    definitions = dict(closure["definitions"])
    imports = dict(closure["imports"])
    records = [
        *( (f"{logical_name}::{name}", str(value))
           for name, value in definitions.items() ),
        *( (f"{logical_name}::import:{name}", str(value))
           for name, value in imports.items() ),
    ]
    return tuple(sorted(records, key=lambda item: item[0]))


# These are intentionally data/selection/order semantics, not mere convenience
# lists: changes here are dataset changes.  Keep them narrow enough that a
# parquet/checkpoint writer change remains operational.
_R02_SEMANTIC_SYMBOLS = (
    "open_source_binary",
    "iter_logical_records",
)
_CANONICAL_SEMANTIC_SYMBOLS = (
    # Conversion, schemas, duplicate selection, and frozen output ordering
    # define the dataset.  Deliberately omit ``shard_for_key``: its only R03
    # role is operational partition routing and therefore belongs in the
    # generator closure reached from ``spill_parsed_batches``.
    "BadRow",
    "CanonicalPolicies",
    "ParsedRow",
    "age_rank",
    "canonical_json",
    "canonical_report_id",
    "convert_record",
    "membership_schema",
    "parsed_rows_schema",
    "resolve_group",
    "stable_source_order",
    "unique_reports_schema",
)
_CANONICAL_GENERATOR_SYMBOLS = (
    # This class owns validation and defaults for both semantic and
    # operational knobs.  Dataset identity receives only the already-resolved
    # scientific values below; implementation/default provenance belongs to
    # the build that resolved them.
    "R03RunParameters",
)
_R03_SEMANTIC_SYMBOLS = (
    "R03SourceRegistry.from_lineage",
    "verify_source",
    "enumerate_source",
    "enumerate_selected_source",
    "enumeration_manifest",
    "_sha256_file",
    "_normalize_relative_path",
    "_safe_join",
    "_final_sort_key",
)

# These are the operational implementations that can change bytes, shard/run
# layout, checkpoint mechanics, or publication behavior without changing the
# logical point dataset.
_R03_GENERATOR_SYMBOLS = (
    "_write_parquet_atomic",
    "spill_parsed_batches",
    "_BufferedParquetWriter",
    "GroupArtifactWriters",
    "_GroupAccumulator",
    "_large_group_rows",
    "resolve_sorted_shard",
    "_resolve_sorted_shard_authenticated",
    "merge_final_partitions",
)

_SELECTION_SEMANTIC_SYMBOLS = (
    "CapacitySelectionManifest.from_record",
    "build_capacity_selection",
    "capacity_selection_bytes",
    "load_and_verify_capacity_selection",
)

# These public entry points cover identity construction, production, restart,
# publication, and independent verification.  Their recursive builder-local
# dependency closure authenticates every imported object that Task 5 can
# execute; the builder's own definitions remain generator-only provenance.
_BUILDER_EXECUTION_SYMBOLS = (
    "build_canonical_points",
    "verify_canonical_manifest",
    "verify_current_pointer",
)


def _schema_hash() -> str:
    schemas = {
        "bad_rows": _bad_rows_schema(),
        "duplicate_conflicts": _conflict_schema(),
        "parsed_rows": parsed_rows_schema(),
        "report_membership": membership_schema(),
        "unique_reports": unique_reports_schema(),
    }
    return _sha256_canonical(
        {
            name: hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
            for name, schema in sorted(schemas.items())
        }
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _verified_selection_inputs(
    registry: R03SourceRegistry,
    selection_manifest: CapacitySelectionManifest,
) -> tuple[
    tuple[object, ...], Mapping[str, tuple[int, ...]], bytes, str,
]:
    """Rebuild a selection from trusted lineage and freeze its execution inputs."""

    if not isinstance(selection_manifest, CapacitySelectionManifest):
        raise TypeError("selection_manifest must be CapacitySelectionManifest")
    trusted = build_capacity_selection(registry)
    supplied_bytes = capacity_selection_bytes(selection_manifest)
    trusted_bytes = capacity_selection_bytes(trusted)
    if selection_manifest != trusted or supplied_bytes != trusted_bytes:
        raise ValueError("selection manifest is not reproducible from trusted lineage")
    ordinals = MappingProxyType(
        {
            source.source_file_id: source.selected_ordinals
            for source in trusted.sources
            if source.selected_ordinals
        }
    )
    selected_specs = tuple(
        spec for spec in registry.sources if spec.source_file_id in ordinals
    )
    if tuple(spec.source_file_id for spec in selected_specs) != tuple(ordinals):
        raise ValueError("selection manifest source order disagrees with trusted registry")
    return selected_specs, ordinals, trusted_bytes, hashlib.sha256(trusted_bytes).hexdigest()


@dataclass(frozen=True)
class R03RunIdentity:
    """Frozen scientific, operational, and checkpoint-reuse identities."""

    schema_version: str
    dataset_id: str
    build_id: str
    checkpoint_compatibility_id: str
    population_kind: Literal["full_population", "capacity_sample"]
    selection_manifest_sha256: str | None
    registry_id: str
    raw_manifest_hash: str
    archive_comparison_hash: str
    r01_disposition_hash: str
    r02_run_manifest_hash: str
    r02_decision_hash: str
    r02_partition_manifest_hash: str
    source_sequence_manifest_hash: str
    source_fingerprint_hash: str
    parser_hash: str
    canonical_schema_hash: str
    unit_policy_hash: str
    qc_policy_hash: str
    dedup_policy_hash: str
    semantic_code_hash: str
    semantic_code_dependencies: tuple[tuple[str, str], ...]
    generator_code_hash: str
    generator_code_dependencies: tuple[tuple[str, str], ...]
    runtime_environment_hash: str
    run_parameters: R03RunParameters

    @classmethod
    def from_inputs(
        cls,
        project_root: Path,
        sources: Sequence[VerifiedR03Source],
        parameters: R03RunParameters,
        *,
        selection_manifest: CapacitySelectionManifest | None = None,
    ) -> "R03RunIdentity":
        root = Path(project_root).resolve()
        # Authenticate the complete consuming binding graph before any of its
        # imported objects can influence validation or identity construction.
        builder_execution_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.build_canonical_points",
            "src/airspace_complexity/build_canonical_points.py",
            _BUILDER_EXECUTION_SYMBOLS,
        )
        if not isinstance(parameters, R03RunParameters):
            raise TypeError("parameters must be R03RunParameters")
        registry = R03SourceRegistry.from_lineage(root)
        if selection_manifest is None:
            population_kind: Literal["full_population", "capacity_sample"] = "full_population"
            selection_manifest_sha256 = None
            expected = tuple(registry.sources)
        else:
            if parameters.worker_count != 1:
                raise ValueError("capacity selection requires worker_count == 1")
            if parameters.max_records_per_airport is not None:
                raise ValueError("capacity selection cannot use max_records_per_airport")
            population_kind = "capacity_sample"
            expected, _, _, selection_manifest_sha256 = _verified_selection_inputs(
                registry, selection_manifest
            )
        if tuple(source.spec for source in sources) != expected:
            raise ValueError("sources must exactly match the authorized population order")
        if any(source.source_content_sha256 != source.spec.extracted_sha256 for source in sources):
            raise ValueError("verified source content hash does not match registry")

        source_records = [
            {
                "airport_id": source.spec.airport_id,
                "source_file_id": source.spec.source_file_id,
                "relative_path": source.spec.relative_path,
                "r01_disposition": source.spec.r01_disposition,
                "source_sequence": source.spec.source_sequence,
                "source_content_sha256": source.source_content_sha256,
                "archive_relative_path": source.spec.archive_relative_path,
                "archive_sha256": source.spec.archive_sha256,
                "archive_member": source.spec.archive_member,
                "member_crc32": source.spec.member_crc32,
                "member_size_bytes": source.spec.member_size_bytes,
                "authenticated_row_count": source.spec.authenticated_row_count,
            }
            for source in sources
        ]
        source_fingerprint_hash = _sha256_canonical(source_records)
        r01_disposition_hash = _sha256_canonical(
            [
                {
                    "airport_id": item["airport_id"],
                    "source_file_id": item["source_file_id"],
                    "r01_disposition": item["r01_disposition"],
                    "source_sequence": item["source_sequence"],
                }
                for item in source_records
            ]
        )
        parser_hash = _bundle_hash(
            root,
            {"src/airspace_complexity/raw_parser.py": "airspace_complexity.raw_parser"},
        )
        canonical_semantic_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.canonical_points",
            "src/airspace_complexity/canonical_points.py",
            _CANONICAL_SEMANTIC_SYMBOLS,
        )
        canonical_generator_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.canonical_points",
            "src/airspace_complexity/canonical_points.py",
            _CANONICAL_GENERATOR_SYMBOLS,
        )
        r02_semantic_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.r02_runner",
            "src/airspace_complexity/r02_runner.py",
            _R02_SEMANTIC_SYMBOLS,
        )
        r03_semantic_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.r03_runner",
            "src/airspace_complexity/r03_runner.py",
            _R03_SEMANTIC_SYMBOLS,
        )
        selection_semantic_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.r03_selection",
            "src/airspace_complexity/r03_selection.py",
            _SELECTION_SEMANTIC_SYMBOLS,
        )
        builder_code_hash = _bundle_hash(
            root,
            {"src/airspace_complexity/build_canonical_points.py": "airspace_complexity.build_canonical_points"},
        )
        r03_generator_closure = _symbol_bundle_dependency_closure(
            root,
            "airspace_complexity.r03_runner",
            "src/airspace_complexity/r03_runner.py",
            _R03_GENERATOR_SYMBOLS,
        )
        semantic_code_dependencies = tuple(sorted((
            ("src/airspace_complexity/raw_parser.py::<whole_module>", parser_hash),
            *_identity_dependency_records(
                "src/airspace_complexity/canonical_points.py",
                canonical_semantic_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/r02_runner.py", r02_semantic_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/r03_runner.py", r03_semantic_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/r03_selection.py",
                selection_semantic_closure,
            ),
        ), key=lambda item: item[0]))
        generator_code_dependencies = tuple(sorted((
            ("src/airspace_complexity/build_canonical_points.py::<whole_module>", builder_code_hash),
            *_identity_dependency_records(
                "src/airspace_complexity/build_canonical_points.py",
                builder_execution_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/canonical_points.py",
                canonical_generator_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/r03_runner.py", r03_generator_closure,
            ),
            *_identity_dependency_records(
                "src/airspace_complexity/r03_selection.py",
                selection_semantic_closure,
            ),
        ), key=lambda item: item[0]))
        semantic_code_hash = _sha256_canonical(
            {
                "canonical_points": _closure_hash(
                    "src/airspace_complexity/canonical_points.py",
                    canonical_semantic_closure,
                ),
                "r02_source_reading": _closure_hash(
                    "src/airspace_complexity/r02_runner.py", r02_semantic_closure,
                ),
                "r03_source_and_order": _closure_hash(
                    "src/airspace_complexity/r03_runner.py", r03_semantic_closure,
                ),
                "capacity_selection": _closure_hash(
                    "src/airspace_complexity/r03_selection.py",
                    selection_semantic_closure,
                ),
            }
        )
        generator_code_hash = _sha256_canonical(
            {
                # The builder owns checkpoint/manifest/publication mechanics.
                "build_canonical_points": builder_code_hash,
                "builder_execution_dependencies": _closure_hash(
                    "src/airspace_complexity/build_canonical_points.py",
                    builder_execution_closure,
                ),
                "canonical_parameter_resolution": _closure_hash(
                    "src/airspace_complexity/canonical_points.py",
                    canonical_generator_closure,
                ),
                "r03_writer_and_shards": _closure_hash(
                    "src/airspace_complexity/r03_runner.py", r03_generator_closure,
                ),
                "capacity_selection_binding": _closure_hash(
                    "src/airspace_complexity/r03_selection.py",
                    selection_semantic_closure,
                ),
            }
        )
        policy_file = root / "configs/data/r03_qc_policy.yaml"
        if not policy_file.is_file():
            raise RuntimeError("R03 policy is missing from the expected project root")
        policy_hash = _sha256_file(policy_file)
        canonical_schema_hash = _schema_hash()
        unit_policy_hash = _sha256_canonical({"unit_conversion": semantic_code_hash})
        qc_policy_hash = _sha256_canonical({"policy": policy_hash, "semantic": semantic_code_hash})
        dedup_policy_hash = _sha256_canonical({"deduplication": semantic_code_hash})
        loaded_origins = _loaded_project_origins(
            root,
            {
                "src/airspace_complexity/raw_parser.py": "airspace_complexity.raw_parser",
                "src/airspace_complexity/canonical_points.py": "airspace_complexity.canonical_points",
                "src/airspace_complexity/r02_runner.py": "airspace_complexity.r02_runner",
                "src/airspace_complexity/r03_runner.py": "airspace_complexity.r03_runner",
                "src/airspace_complexity/r03_selection.py": "airspace_complexity.r03_selection",
                "src/airspace_complexity/build_canonical_points.py": "airspace_complexity.build_canonical_points",
            },
        )
        dependency_lock = root / "requirements-lock.txt"
        if not dependency_lock.is_file():
            raise RuntimeError("dependency lock is missing from the expected project root")
        runtime_environment_hash = _sha256_canonical(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "pyarrow": pyarrow.__version__,
                "pyyaml": yaml.__version__,
                "zipfile_deflate64": _distribution_version("zipfile-deflate64"),
                "dependency_lock_sha256": _sha256_file(dependency_lock),
                "parquet_writer": "pyarrow.parquet.zstd",
                "resolved_project_module_origins": {
                    name: origin.relative_to(root).as_posix()
                    for name, origin in sorted(loaded_origins.items())
                },
                "cuda": "not_applicable",
                "framework": "not_applicable",
                "seed": "not_applicable",
            }
        )
        parameter_record = asdict(parameters)
        dataset_payload = {
            "schema_version": _IDENTITY_SCHEMA_VERSION,
            "population_kind": population_kind,
            "selection_manifest_sha256": selection_manifest_sha256,
            "registry_id": registry.registry_id,
            "raw_manifest_hash": registry.raw_manifest_sha256,
            "archive_comparison_hash": registry.archive_comparison_sha256,
            "r02_run_manifest_hash": registry.r02_run_manifest_sha256,
            "r02_decision_hash": registry.r02_decision_sha256,
            "r02_partition_hashes": registry.partition_manifest_sha256_by_airport,
            "source_sequence_hash": registry.source_sequence_manifest_sha256,
            "r01_disposition_hash": r01_disposition_hash,
            "source_fingerprint_hash": source_fingerprint_hash,
            "parser_hash": parser_hash,
            "semantic_code_hash": semantic_code_hash,
            "canonical_schema_hash": canonical_schema_hash,
            "unit_policy_hash": unit_policy_hash,
            "qc_policy_hash": qc_policy_hash,
            "dedup_policy_hash": dedup_policy_hash,
            "final_sort_version": parameters.final_sort_version,
            "max_records_per_airport": parameters.max_records_per_airport,
        }
        dataset_id = _sha256_canonical(dataset_payload)
        build_payload = {
            "schema_version": _BUILD_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "generator_code_hash": generator_code_hash,
            "runtime_environment_hash": runtime_environment_hash,
            "shard_count": parameters.shard_count,
            "shard_hash_version": parameters.shard_hash_version,
            "record_batch_rows": parameters.record_batch_rows,
            "max_rows_per_run": parameters.max_rows_per_run,
            "max_group_rows_in_memory": parameters.max_group_rows_in_memory,
            "worker_count": parameters.worker_count,
        }
        build_id = _sha256_canonical(build_payload)
        return cls(
            schema_version=_IDENTITY_SCHEMA_VERSION,
            dataset_id=dataset_id,
            build_id=build_id,
            checkpoint_compatibility_id=_sha256_canonical(
                {"schema_version": _CHECKPOINT_SCHEMA_VERSION, "build_id": build_id}
            ),
            population_kind=population_kind,
            selection_manifest_sha256=selection_manifest_sha256,
            registry_id=registry.registry_id,
            raw_manifest_hash=registry.raw_manifest_sha256,
            archive_comparison_hash=registry.archive_comparison_sha256,
            r01_disposition_hash=r01_disposition_hash,
            r02_run_manifest_hash=registry.r02_run_manifest_sha256,
            r02_decision_hash=registry.r02_decision_sha256,
            r02_partition_manifest_hash=_sha256_canonical(
                dict(registry.partition_manifest_sha256_by_airport)
            ),
            source_sequence_manifest_hash=registry.source_sequence_manifest_sha256,
            source_fingerprint_hash=source_fingerprint_hash,
            parser_hash=parser_hash,
            canonical_schema_hash=canonical_schema_hash,
            unit_policy_hash=unit_policy_hash,
            qc_policy_hash=qc_policy_hash,
            dedup_policy_hash=dedup_policy_hash,
            semantic_code_hash=semantic_code_hash,
            semantic_code_dependencies=semantic_code_dependencies,
            generator_code_hash=generator_code_hash,
            generator_code_dependencies=generator_code_dependencies,
            runtime_environment_hash=runtime_environment_hash,
            run_parameters=parameters,
        )

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["run_parameters"] = asdict(self.run_parameters)
        record["semantic_code_dependencies"] = [
            [name, digest] for name, digest in self.semantic_code_dependencies
        ]
        record["generator_code_dependencies"] = [
            [name, digest] for name, digest in self.generator_code_dependencies
        ]
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> "R03RunIdentity":
        values = dict(record)
        expected_fields = {field.name for field in __import__("dataclasses").fields(cls)}
        if set(values) != expected_fields:
            raise ValueError("R03 identity fields are invalid")
        if values.get("population_kind") not in {"full_population", "capacity_sample"}:
            raise ValueError("R03 identity population_kind is invalid")
        selection_hash = values.get("selection_manifest_sha256")
        if values["population_kind"] == "full_population":
            if selection_hash is not None:
                raise ValueError("full population identity cannot bind a selection")
        elif (
            not isinstance(selection_hash, str)
            or len(selection_hash) != 64
            or any(character not in "0123456789abcdef" for character in selection_hash)
        ):
            raise ValueError("capacity identity selection hash is invalid")
        values["run_parameters"] = R03RunParameters.from_mapping(
            dict(values["run_parameters"])  # type: ignore[arg-type]
        )
        run_parameters = values["run_parameters"]
        if values["population_kind"] == "capacity_sample":
            if run_parameters.worker_count != 1:  # type: ignore[union-attr]
                raise ValueError("capacity identity requires worker_count == 1")
            if run_parameters.max_records_per_airport is not None:  # type: ignore[union-attr]
                raise ValueError(
                    "capacity identity requires max_records_per_airport to be null"
                )
        for field in ("semantic_code_dependencies", "generator_code_dependencies"):
            raw = values.get(field)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{field} must be a sequence")
            pairs: list[tuple[str, str]] = []
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(f"{field} entries must be string pairs")
                name, digest = item
                if not isinstance(name, str) or not isinstance(digest, str):
                    raise ValueError(f"{field} entries must be string pairs")
                pairs.append((name, digest))
            if tuple(sorted(pairs)) != tuple(pairs):
                raise ValueError(f"{field} entries must be sorted")
            values[field] = tuple(pairs)
        return cls(**values)  # type: ignore[arg-type]


_CHECKPOINT_FORMAT = "r03_run_checkpoint_v1"
_MANIFEST_FORMAT = "r03_canonical_manifest_v2"
_POINTER_FORMAT = "r03_current_pointer_v1"
_SEALED_SHARD_FORMAT = "r03_sealed_shard_v2"
_MANIFEST_ROLES = (
    "source_enumeration",
    "bad_rows",
    "parsed_rows",
    "unique_reports",
    "report_membership",
    "duplicate_conflicts",
)
_FINAL_SCHEMAS = {
    "bad_rows": _bad_rows_schema(),
    "parsed_rows": parsed_rows_schema(),
    "unique_reports": unique_reports_schema(),
    "report_membership": membership_schema(),
    "duplicate_conflicts": _conflict_schema(),
}
_SORT_CONTRACTS = {
    "source_enumeration": "(airport_id,source_sequence,source_file_id)",
    "bad_rows": "raw_row_id",
    "parsed_rows": "(natural_key,stable_source_order)",
    "unique_reports": "natural_key",
    "report_membership": "(canonical_report_id,dedup_rank,raw_row_id)",
    "duplicate_conflicts": "canonical_report_id",
}
_SEALED_SORT_CONTRACTS = {
    "bad_rows": "raw_row_id",
    "parsed_rows": "(natural_key,age_rank_class,age_rank_value,-completeness_score,stable_source_order)",
    "unique_reports": "natural_key",
    "report_membership": "matches_parsed_resolution_rank",
    "duplicate_conflicts": "subset_of_unique_reports_natural_key",
}


def _termination_seam(point: str) -> None:
    """Subprocess-only fault seam used to test process-crash publication order."""
    if os.environ.get("AIRSPACE_R03_TERMINATE_AT") == point:
        os._exit(97)


def _enumeration_schema() -> pyarrow.Schema:
    return pyarrow.schema(
        [
            pyarrow.field("schema_version", pyarrow.string(), nullable=False),
            pyarrow.field("registry_id", pyarrow.string(), nullable=False),
            pyarrow.field("raw_manifest_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("archive_comparison_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("r02_run_manifest_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("r02_decision_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("r02_partition_manifest_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("source_sequence_manifest_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("airport_id", pyarrow.string(), nullable=False),
            pyarrow.field("source_file_id", pyarrow.string(), nullable=False),
            pyarrow.field("relative_path", pyarrow.string(), nullable=False),
            pyarrow.field("r01_disposition", pyarrow.string(), nullable=False),
            pyarrow.field("source_sequence", pyarrow.int64(), nullable=False),
            pyarrow.field("archive_relative_path", pyarrow.string(), nullable=True),
            pyarrow.field("archive_member", pyarrow.string(), nullable=True),
            pyarrow.field("source_content_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("header_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("logical_record_count", pyarrow.int64(), nullable=False),
            pyarrow.field("first_raw_row_id", pyarrow.string(), nullable=True),
            pyarrow.field("last_raw_row_id", pyarrow.string(), nullable=True),
            pyarrow.field("first_source_row_number", pyarrow.int64(), nullable=True),
            pyarrow.field("last_source_row_number", pyarrow.int64(), nullable=True),
            pyarrow.field("status", pyarrow.string(), nullable=False),
        ]
    )


def _manifest_schema() -> pyarrow.Schema:
    return pyarrow.schema(
        [
            pyarrow.field("schema_version", pyarrow.string(), nullable=False),
            pyarrow.field("project_root", pyarrow.string(), nullable=False),
            pyarrow.field("identity_json", pyarrow.string(), nullable=False),
            pyarrow.field("dataset_id", pyarrow.string(), nullable=False),
            pyarrow.field("build_id", pyarrow.string(), nullable=False),
            pyarrow.field("logical_content_root", pyarrow.string(), nullable=False),
            pyarrow.field("population_kind", pyarrow.string(), nullable=False),
            pyarrow.field("selection_manifest_sha256", pyarrow.string(), nullable=True),
            pyarrow.field("max_records_per_airport", pyarrow.int64(), nullable=True),
            pyarrow.field("is_full_population", pyarrow.bool_(), nullable=False),
            pyarrow.field("r04_authorized", pyarrow.bool_(), nullable=False),
            pyarrow.field("model_training_authorized", pyarrow.bool_(), nullable=False),
            pyarrow.field("artifact_role", pyarrow.string(), nullable=False),
            pyarrow.field("relative_path", pyarrow.string(), nullable=False),
            pyarrow.field("row_count", pyarrow.int64(), nullable=False),
            pyarrow.field("schema_fingerprint", pyarrow.string(), nullable=False),
            pyarrow.field("physical_sha256", pyarrow.string(), nullable=False),
            pyarrow.field("logical_lower_bound", pyarrow.string(), nullable=True),
            pyarrow.field("logical_upper_bound", pyarrow.string(), nullable=True),
            pyarrow.field("sort_contract", pyarrow.string(), nullable=False),
            pyarrow.field("parent_hashes_json", pyarrow.string(), nullable=False),
            pyarrow.field("finalized", pyarrow.bool_(), nullable=False),
        ]
    )


@dataclass(frozen=True)
class RunCheckpoint:
    """The deliberately small, independently authenticated resume boundary."""

    schema_version: str
    attempt_id: str
    identity: R03RunIdentity
    generation: int
    state: str
    terminal_error: str | None
    canonical_payload_hash: str
    sealed_shards: tuple[dict[str, object], ...]

    @classmethod
    def load(cls, path: Path) -> "RunCheckpoint":
        record = _read_json_object(path)
        required = {
            "schema_version", "attempt_id", "identity", "generation", "state",
            "terminal_error", "canonical_payload_hash", "sealed_shards",
        }
        if set(record) != required:
            raise ValueError("checkpoint has unexpected fields")
        if record["schema_version"] != _CHECKPOINT_FORMAT:
            raise ValueError("checkpoint schema version mismatch")
        if not isinstance(record["attempt_id"], str) or not record["attempt_id"]:
            raise ValueError("checkpoint attempt ID is invalid")
        if isinstance(record["generation"], bool) or not isinstance(record["generation"], int) or record["generation"] < 0:
            raise ValueError("checkpoint generation is invalid")
        if not isinstance(record["state"], str) or record["state"] not in {
            "NEW", "PREFLIGHTED", "ENUMERATED", "SHARDING", "MERGING",
            "VERIFYING", "FINALIZED", "PUBLISHED", "FAILED",
        }:
            raise ValueError("checkpoint state is invalid")
        if record["terminal_error"] is not None and not isinstance(record["terminal_error"], str):
            raise ValueError("checkpoint terminal error is invalid")
        entries = record["sealed_shards"]
        if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
            raise ValueError("checkpoint sealed shards are invalid")
        expected_generation = {
            "NEW": 0,
            "PREFLIGHTED": 1,
            "ENUMERATED": 2,
            "SHARDING": 3 + len(entries),
            "MERGING": 4 + len(entries),
            "VERIFYING": 5 + len(entries),
            "FINALIZED": 6 + len(entries),
            "PUBLISHED": 7 + len(entries),
        }
        if (
            record["state"] in expected_generation
            and record["generation"] != expected_generation[str(record["state"])]
        ):
            raise ValueError("checkpoint generation does not match its sealed-shard state")
        if record["state"] == "FAILED" and record["generation"] < 1:
            raise ValueError("failed checkpoint generation is invalid")
        payload = {
            "schema_version": record["schema_version"], "attempt_id": record["attempt_id"],
            "identity": record["identity"], "generation": record["generation"],
            "state": record["state"], "terminal_error": record["terminal_error"],
            "sealed_shards": entries,
        }
        if record["canonical_payload_hash"] != _sha256_canonical(payload):
            raise ValueError("checkpoint canonical payload hash mismatch")
        return cls(
            schema_version=str(record["schema_version"]), attempt_id=str(record["attempt_id"]),
            identity=R03RunIdentity.from_dict(dict(record["identity"])),
            generation=int(record["generation"]), state=str(record["state"]),
            terminal_error=record["terminal_error"] if isinstance(record["terminal_error"], str) else None,
            canonical_payload_hash=str(record["canonical_payload_hash"]),
            sealed_shards=tuple(dict(item) for item in entries),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version, "attempt_id": self.attempt_id,
            "identity": self.identity.to_dict(), "generation": self.generation,
            "state": self.state, "terminal_error": self.terminal_error,
            "sealed_shards": [dict(item) for item in self.sealed_shards],
        }
        return {**payload, "canonical_payload_hash": _sha256_canonical(payload)}

    def next(self, state: str, sealed_shards: Iterable[dict[str, object]] | None = None, terminal_error: str | None = None) -> "RunCheckpoint":
        return replace(
            self, generation=self.generation + 1, state=state,
            terminal_error=terminal_error,
            sealed_shards=tuple(self.sealed_shards if sealed_shards is None else sealed_shards),
            canonical_payload_hash="",
        )


@dataclass(frozen=True)
class R03RunResult:
    identity: R03RunIdentity
    state: str
    output_root: Path
    final_root: Path
    manifest_path: Path | None
    logical_content_root: str | None
    is_full_population: bool
    r04_authorized: bool
    model_training_authorized: bool
    preflight: Mapping[str, object] | None = None


@dataclass(frozen=True)
class R03Decision:
    """The intentionally narrow R03-to-R04 authorization record."""

    decision: str
    dataset_id: str
    checks: Mapping[str, bool]
    is_full_population: bool
    r04_authorized: bool
    model_training_authorized: bool = False

    @classmethod
    def from_checks(
        cls,
        dataset_id: str,
        checks: Mapping[str, bool],
        *,
        is_full_population: bool = True,
        blocked: bool = False,
    ) -> "R03Decision":
        if not isinstance(dataset_id, str) or len(dataset_id) != 64:
            raise ValueError("R03 decision requires a dataset SHA-256 identifier")
        normalized: dict[str, bool] = {}
        for name, passed in sorted(checks.items()):
            if not isinstance(name, str) or not name or not isinstance(passed, bool):
                raise ValueError("R03 decision checks must be named booleans")
            normalized[name] = passed
        proceed = bool(normalized) and all(normalized.values()) and is_full_population and not blocked
        return cls(
            "BLOCKED_NEEDS_HUMAN_DECISION" if blocked else ("PROCEED_TO_R04" if proceed else "REVISE_R03"),
            dataset_id,
            normalized,
            is_full_population,
            proceed,
            False,
        )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, record: Mapping[str, object]) -> None:
    """Replace a JSON file only after its contents have reached the OS cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.name == "checkpoint.json":
            _termination_seam("checkpoint-replace-pre")
            # Compatibility alias retained for the original Task 5 test seam.
            _termination_seam("checkpoint-replace")
        if path.name == "current.json":
            _termination_seam("pointer-replace-pre")
            _termination_seam("pointer-replace")
        os.replace(temporary, path)
        if path.name == "checkpoint.json":
            _termination_seam("checkpoint-replace-post")
        if path.name == "current.json":
            _termination_seam("pointer-replace-post")
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Atomically write exact immutable parent-record bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    """Best effort only: Windows does not expose portable directory fsync."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_no_reparse_escape(root: Path, target: Path) -> Path:
    lexical_root = Path(root)
    if not lexical_root.is_absolute():
        lexical_root = Path.cwd() / lexical_root
    lexical_target = Path(target)
    if not lexical_target.is_absolute():
        lexical_target = lexical_root / lexical_target
    _reject_existing_reparse_components(lexical_root)
    try:
        lexical_relative = lexical_target.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("output path escapes the output root") from exc
    # Inspect the supplied lexical path *before* resolving it.  Resolving a
    # contained alias first turns ``root/alias/file`` into ``root/real/file``
    # and silently erases the forbidden reparse traversal.
    current = lexical_root
    for part in lexical_relative.parts:
        if part in {".", ".."}:
            raise ValueError("output path escapes the output root")
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError("cannot safely inspect output path component") from exc
        attributes = getattr(status, "st_file_attributes", 0)
        if stat.S_ISLNK(status.st_mode) or attributes & 0x400:
            raise ValueError("output paths may not traverse symlinks or reparse points")

    resolved_root = lexical_root.resolve()
    normalized_target = lexical_target.resolve()
    try:
        normalized_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("output path escapes the output root") from exc
    return normalized_target


def _verification_scratch_root(final_root: Path) -> Path:
    """Allocate verifier-private work beside, never inside, an immutable root."""
    final = Path(final_root)
    output = final.parent
    _reject_existing_reparse_components(output)
    _assert_no_reparse_escape(output, output)
    _assert_no_reparse_escape(output, final)
    _ensure_output_volume(output, final)
    scratch = Path(tempfile.mkdtemp(prefix=".r03-verify-", dir=output))
    # ``mkdtemp`` creates the directory atomically.  Re-check the lexical path
    # before use so a reparse swap cannot turn it into a traversal.
    scratch = _assert_no_reparse_escape(output, scratch)
    _termination_seam("verify-scratch-created")
    return scratch


def _reject_existing_reparse_components(path: Path) -> None:
    current = Path(path.anchor) if path.anchor else Path.cwd().anchor
    for part in path.parts[1:] if path.anchor else path.parts:
        if part in {".", ".."}:
            raise ValueError("output root may not contain dot traversal")
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ValueError("cannot safely inspect output root component") from exc
        if stat.S_ISLNK(status.st_mode) or getattr(status, "st_file_attributes", 0) & 0x400:
            raise ValueError("output root may not traverse symlinks or reparse points")


def _ensure_output_volume(output_root: Path, target: Path) -> None:
    output_drive = os.path.splitdrive(str(Path(output_root).resolve()))[0].casefold()
    target_drive = os.path.splitdrive(str(Path(target).resolve()))[0].casefold()
    if (
        not output_drive
        or output_drive.startswith("\\\\")
        or output_drive != target_drive
    ):
        raise ValueError("staging and final roots must be on the same local volume")


def _acquire_lock(output_root: Path, identity: R03RunIdentity, attempt_id: str) -> Path:
    # One output root has one mutable publication pointer, regardless of the
    # requested build ID.  A build-scoped lock would permit competing writers.
    lock = _assert_no_reparse_escape(output_root, output_root / ".r03-writer.lock")
    record = {
        "schema_version": "r03_writer_lock_v1", "hostname": socket.gethostname(),
        "pid": os.getpid(), "build_id": identity.build_id, "attempt_id": attempt_id,
    }
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_json_object(lock)
        if existing.get("hostname") != socket.gethostname():
            raise RuntimeError("another R03 builder owns the output root")
        pid = existing.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("R03 writer lock cannot be safely reclaimed")
        if pid == os.getpid():
            raise RuntimeError("another R03 builder owns the output root")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink(missing_ok=True)
            return _acquire_lock(output_root, identity, attempt_id)
        except OSError as exc:
            # Windows reports ERROR_INVALID_PARAMETER (87), rather than
            # ESRCH, for a PID which has ceased to exist.
            if os.name == "nt" and getattr(exc, "winerror", None) == 87:
                lock.unlink(missing_ok=True)
                return _acquire_lock(output_root, identity, attempt_id)
            raise RuntimeError("R03 writer lock cannot be safely reclaimed") from exc
        except PermissionError as exc:
            raise RuntimeError("R03 writer lock cannot be safely reclaimed") from exc
        raise RuntimeError("another R03 builder owns the output root")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    return lock


def _release_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def _artifact_schema(role: str) -> pyarrow.Schema:
    if role == "source_enumeration":
        return _enumeration_schema()
    return _FINAL_SCHEMAS[role]


def _normal_value(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("logical timestamp is naive")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, list):
        return [_normal_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normal_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normal_value(item) for key, item in sorted(value.items())}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("logical row contains a forbidden nonfinite value")
        return 0.0 if value == 0.0 else value
    return value


def _canonical_row_bytes(row: Mapping[str, object], role: str) -> bytes:
    logical = dict(row)
    age_rank = logical.get("age_rank_value")
    if isinstance(age_rank, float) and math.isinf(age_rank) and age_rank > 0:
        if (
            role != "report_membership"
            or type(logical.get("age_rank_class")) is not int
            or logical["age_rank_class"] != 2
        ):
            raise ValueError("logical row contains a forbidden nonfinite value")
        logical["age_rank_value"] = {"$r03_float": "positive_infinity"}
    return json.dumps(
        _normal_value(logical), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _row_key(role: str, row: Mapping[str, object]) -> tuple[object, ...]:
    if role == "bad_rows":
        return (str(row["raw_row_id"]),)
    if role == "source_enumeration":
        return (str(row["airport_id"]), int(row["source_sequence"]), str(row["source_file_id"]))
    return _final_sort_key(role, row)


def _logical_bounds(path: Path, role: str, schema: pyarrow.Schema) -> tuple[int, str | None, str | None]:
    count = 0
    previous: tuple[object, ...] | None = None
    lower: str | None = None
    upper: str | None = None
    for row in _iter_mappings(path, schema):
        key = _row_key(role, row)
        if previous is not None and key < previous:
            raise ValueError(f"artifact is not sorted: {path}")
        encoded = _canonical_row_bytes(row, role).decode("utf-8")
        lower = encoded if lower is None else lower
        upper = encoded
        previous = key
        count += 1
    return count, lower, upper


def _sealed_logical_bounds(
    shard_root: Path, role: str,
) -> tuple[int, str | None, str | None]:
    """Validate the actual pre-merge emission order of one sealed shard role."""
    path = shard_root / f"{role}.parquet"
    schema = _artifact_schema(role)
    if role in {"bad_rows", "unique_reports"}:
        return _logical_bounds(path, role, schema)
    count = 0
    lower: str | None = None
    upper: str | None = None

    def observe(row: Mapping[str, object]) -> None:
        nonlocal count, lower, upper
        encoded = _canonical_row_bytes(row, role).decode("utf-8")
        lower = encoded if lower is None else lower
        upper = encoded
        count += 1

    if role == "parsed_rows":
        previous: tuple[object, ...] | None = None
        for row in _iter_mappings(path, schema):
            parsed = ParsedRow(
                **{field: row[field] for field in ParsedRow.__dataclass_fields__}
            )
            key = _parsed_rank_key(parsed)
            if previous is not None and key < previous:
                raise ValueError(f"sealed parsed shard is not resolution-rank sorted: {path}")
            observe(row)
            previous = key
        return count, lower, upper
    if role == "report_membership":
        parsed_rows = _iter_mappings(
            shard_root / "parsed_rows.parquet", parsed_rows_schema()
        )
        membership_rows = _iter_mappings(path, schema)
        missing = object()
        previous_report: str | None = None
        expected_rank = 0
        for parsed, member in zip_longest(
            parsed_rows, membership_rows, fillvalue=missing
        ):
            if parsed is missing or member is missing:
                raise ValueError("sealed membership does not match parsed-row cardinality")
            assert isinstance(parsed, dict) and isinstance(member, dict)
            report_id = canonical_report_id(
                str(parsed["airport_id"]),
                str(parsed["aircraft_id"]),
                parsed["report_timestamp_utc"],  # type: ignore[arg-type]
            )
            expected_rank = expected_rank + 1 if report_id == previous_report else 1
            if (
                member["raw_row_id"] != parsed["raw_row_id"]
                or member["canonical_report_id"] != report_id
                or member["dedup_rank"] != expected_rank
                or member["airport_id"] != parsed["airport_id"]
                or member["partition_year"] != parsed["partition_year"]
                or member["partition_month"] != parsed["partition_month"]
            ):
                raise ValueError("sealed membership is not in parsed resolution order")
            observe(member)
            previous_report = report_id
        return count, lower, upper
    if role == "duplicate_conflicts":
        unique_rows = iter(_iter_mappings(
            shard_root / "unique_reports.parquet", unique_reports_schema()
        ))
        candidate = next(unique_rows, None)
        for conflict in _iter_mappings(path, schema):
            while (
                candidate is not None
                and candidate["canonical_report_id"] != conflict["canonical_report_id"]
            ):
                candidate = next(unique_rows, None)
            if candidate is None or (
                conflict["airport_id"] != candidate["airport_id"]
                or conflict["partition_year"] != candidate["partition_year"]
                or conflict["partition_month"] != candidate["partition_month"]
            ):
                raise ValueError("sealed conflicts are not a natural-order unique subset")
            observe(conflict)
            candidate = next(unique_rows, None)
        return count, lower, upper
    raise ValueError(f"unknown sealed shard artifact role: {role}")


def _logical_content_root(artifacts: Sequence[dict[str, object]], root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"r03_logical_content_v2\0")
    for artifact in sorted(artifacts, key=lambda value: (str(value["artifact_role"]), str(value["relative_path"]))):
        role = str(artifact["artifact_role"])
        path = _assert_no_reparse_escape(root, root / str(artifact["relative_path"]))
        schema = _artifact_schema(role)
        digest.update(role.encode("utf-8") + b"\0")
        digest.update(schema.serialize().to_pybytes())
        for row in _iter_mappings(path, schema):
            digest.update(_canonical_row_bytes(row, role) + b"\n")
    return digest.hexdigest()


def _write_empty(path: Path, schema: pyarrow.Schema) -> None:
    _write_parquet_atomic(path, pyarrow.Table.from_pylist([], schema=schema))


def _write_bad_shards(
    bad_writers: Mapping[int, pq.ParquetWriter],
    bad_buffers: Mapping[int, list[dict[str, object]]],
) -> None:
    for shard_id, writer in bad_writers.items():
        rows = bad_buffers[shard_id]
        if rows:
            writer.write_table(pyarrow.Table.from_pylist(rows, schema=_bad_rows_schema()))
            rows.clear()


def _new_bad_writers(root: Path, shard_count: int) -> tuple[dict[int, pq.ParquetWriter], dict[int, list[dict[str, object]]], dict[int, Path]]:
    writers: dict[int, pq.ParquetWriter] = {}
    buffers: dict[int, list[dict[str, object]]] = {}
    paths: dict[int, Path] = {}
    for shard_id in range(shard_count):
        path = root / "bad-runs" / f"shard-{shard_id:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[shard_id] = pq.ParquetWriter(
            path, _bad_rows_schema(), compression="zstd"
        )
        buffers[shard_id] = []
        paths[shard_id] = path
    return writers, buffers, paths


def _close_bad_writers(
    writers: Mapping[int, pq.ParquetWriter],
    buffers: Mapping[int, list[dict[str, object]]],
) -> None:
    try:
        _write_bad_shards(writers, buffers)
    finally:
        for writer in writers.values():
            writer.close()


def _sealed_artifact(path: Path, role: str) -> dict[str, object]:
    schema = _artifact_schema(role)
    count, lower, upper = _sealed_logical_bounds(path.parent, role)
    return {
        "role": role, "relative_path": path.name, "row_count": count,
        "schema_fingerprint": _schema_fingerprint(schema), "sha256": _sha256_file(path),
        "logical_lower_bound": lower, "logical_upper_bound": upper,
        "sort_contract": _SEALED_SORT_CONTRACTS[role], "finalized": True,
    }


def _write_sealed_shard_manifest(
    shard_root: Path, shard_id: int, identity: R03RunIdentity,
) -> dict[str, object]:
    artifacts = [
        _sealed_artifact(shard_root / "bad_rows.parquet", "bad_rows"),
        _sealed_artifact(shard_root / "parsed_rows.parquet", "parsed_rows"),
        _sealed_artifact(shard_root / "unique_reports.parquet", "unique_reports"),
        _sealed_artifact(shard_root / "report_membership.parquet", "report_membership"),
        _sealed_artifact(shard_root / "duplicate_conflicts.parquet", "duplicate_conflicts"),
    ]
    payload: dict[str, object] = {
        "schema_version": _SEALED_SHARD_FORMAT, "shard_id": shard_id,
        "identity": identity.to_dict(), "finalized": True,
        "artifacts": artifacts,
    }
    payload["canonical_payload_hash"] = _sha256_canonical(payload)
    path = shard_root / "shard_manifest.json"
    _atomic_json(path, payload)
    return {
        "shard_id": shard_id,
        "relative_path": (Path("sealed") / shard_root.name / path.name).as_posix(),
        "sha256": _sha256_file(path),
        "finalized": True,
    }


def _verify_sealed_shard(
    staging: Path, entry: Mapping[str, object], identity: R03RunIdentity,
) -> ShardArtifactSet:
    if set(entry) != {"shard_id", "relative_path", "sha256", "finalized"}:
        raise ValueError("checkpoint shard entry has unexpected fields")
    shard_id = entry.get("shard_id")
    if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
        raise ValueError("checkpoint shard ID is invalid")
    relative = entry.get("relative_path")
    if not isinstance(relative, str) or not relative.startswith("sealed/") or ".." in Path(relative).parts:
        raise ValueError("checkpoint shard path is invalid")
    manifest_path = _assert_no_reparse_escape(staging, staging / relative)
    if not manifest_path.is_file() or _sha256_file(manifest_path) != entry.get("sha256"):
        raise ValueError("checkpoint-referenced shard manifest hash mismatch")
    manifest = _read_json_object(manifest_path)
    expected = {"schema_version", "shard_id", "identity", "finalized", "artifacts", "canonical_payload_hash"}
    if set(manifest) != expected or manifest.get("schema_version") != _SEALED_SHARD_FORMAT:
        raise ValueError("sealed shard manifest shape mismatch")
    payload = {key: value for key, value in manifest.items() if key != "canonical_payload_hash"}
    if manifest.get("canonical_payload_hash") != _sha256_canonical(payload):
        raise ValueError("sealed shard manifest payload hash mismatch")
    if manifest.get("identity") != identity.to_dict() or manifest.get("shard_id") != shard_id or manifest.get("finalized") is not True:
        raise ValueError("sealed shard identity mismatch")
    listed = manifest.get("artifacts")
    if not isinstance(listed, list) or len(listed) != 5:
        raise ValueError("sealed shard artifact set is incomplete")
    artifact_paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for item in listed:
        if not isinstance(item, dict) or set(item) != {
            "role", "relative_path", "row_count", "schema_fingerprint", "sha256",
            "logical_lower_bound", "logical_upper_bound", "sort_contract", "finalized",
        }:
            raise ValueError("sealed shard artifact shape mismatch")
        role = item.get("role")
        if role not in _FINAL_SCHEMAS or role in artifact_paths:
            raise ValueError("sealed shard artifact role mismatch")
        path = _assert_no_reparse_escape(manifest_path.parent, manifest_path.parent / str(item["relative_path"]))
        if not path.is_file() or _sha256_file(path) != item.get("sha256"):
            raise ValueError("sealed shard artifact SHA-256 mismatch")
        if item.get("schema_fingerprint") != _schema_fingerprint(_artifact_schema(str(role))):
            raise ValueError("sealed shard artifact schema fingerprint mismatch")
        count, lower, upper = _sealed_logical_bounds(manifest_path.parent, str(role))
        if count != item.get("row_count") or lower != item.get("logical_lower_bound") or upper != item.get("logical_upper_bound"):
            raise ValueError("sealed shard artifact bounds/count mismatch")
        if item.get("sort_contract") != _SEALED_SORT_CONTRACTS[role] or item.get("finalized") is not True:
            raise ValueError("sealed shard artifact sort/finalization mismatch")
        artifact_paths[str(role)] = path
        counts[str(role)] = count
    return ShardArtifactSet(
        shard_id=shard_id, parsed_rows=artifact_paths["parsed_rows"],
        unique_reports=artifact_paths["unique_reports"],
        report_membership=artifact_paths["report_membership"],
        duplicate_conflicts=artifact_paths["duplicate_conflicts"],
        counts=ArtifactCounts(
            parsed_rows=counts["parsed_rows"], unique_reports=counts["unique_reports"],
            report_membership=counts["report_membership"], duplicate_conflicts=counts["duplicate_conflicts"],
            duplicate_excess=counts["parsed_rows"] - counts["unique_reports"],
        ),
    )


def _remove_uncheckpointed_shards(staging: Path, referenced: set[int]) -> None:
    sealed = staging / "sealed"
    if not sealed.is_dir():
        return
    for path in sealed.iterdir():
        if not path.is_dir() or not path.name.startswith("shard-"):
            continue
        try:
            shard_id = int(path.name.split("-", 1)[1])
        except ValueError:
            shutil.rmtree(path, ignore_errors=True)
            continue
        if shard_id not in referenced:
            shutil.rmtree(path, ignore_errors=True)


def _merge_bad_rows(shard_outputs: Sequence[ShardArtifactSet], root: Path, max_rows: int) -> Path:
    """External-sort bad rows by raw ID without materializing the corpus."""
    run_root = root / ".bad-merge-runs"
    runs: list[Path] = []
    run_index = 0
    for shard in sorted(shard_outputs, key=lambda value: value.shard_id):
        source = shard.parsed_rows.parent / "bad_rows.parquet"
        iterator = _iter_mappings(source, _bad_rows_schema())
        while True:
            rows: list[dict[str, object]] = []
            try:
                for _ in range(max_rows):
                    rows.append(next(iterator))
            except StopIteration:
                pass
            if not rows:
                break
            rows.sort(key=lambda row: _row_key("bad_rows", row))
            run = run_root / f"run-{run_index:06d}.parquet"
            run_index += 1
            _write_parquet_atomic(run, pyarrow.Table.from_pylist(rows, schema=_bad_rows_schema()))
            runs.append(run)
            if len(rows) < max_rows:
                break
    target = root / "bad_rows" / "part-00000.parquet"
    if not runs:
        _write_empty(target, _bad_rows_schema())
        return target
    streams = [_iter_mappings(path, _bad_rows_schema()) for path in runs]
    heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
    for index, stream in enumerate(streams):
        try:
            row = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (_row_key("bad_rows", row), index, row))
    temporary = target.with_name(target.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, _bad_rows_schema(), compression="zstd")
    buffer: list[dict[str, object]] = []
    try:
        while heap:
            _, index, row = heapq.heappop(heap)
            buffer.append(row)
            if len(buffer) >= min(max_rows, 65_536):
                writer.write_table(pyarrow.Table.from_pylist(buffer, schema=_bad_rows_schema()))
                buffer.clear()
            try:
                following = next(streams[index])
            except StopIteration:
                continue
            heapq.heappush(heap, (_row_key("bad_rows", following), index, following))
        if buffer:
            writer.write_table(pyarrow.Table.from_pylist(buffer, schema=_bad_rows_schema()))
    finally:
        writer.close()
    os.replace(temporary, target)
    shutil.rmtree(run_root, ignore_errors=True)
    return target


def _sort_bad_file(path: Path, max_rows: int) -> None:
    """External-sort one sealed-shard bad ledger before it becomes reusable."""
    run_root = path.parent / ".bad-sort-runs"
    runs: list[Path] = []
    iterator = _iter_mappings(path, _bad_rows_schema())
    index = 0
    while True:
        rows: list[dict[str, object]] = []
        try:
            for _ in range(max_rows):
                rows.append(next(iterator))
        except StopIteration:
            pass
        if not rows:
            break
        rows.sort(key=lambda row: _row_key("bad_rows", row))
        run = run_root / f"run-{index:06d}.parquet"
        index += 1
        _write_parquet_atomic(run, pyarrow.Table.from_pylist(rows, schema=_bad_rows_schema()))
        runs.append(run)
        if len(rows) < max_rows:
            break
    temporary = path.with_name(path.name + ".sorted.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, _bad_rows_schema(), compression="zstd")
    streams = [_iter_mappings(run, _bad_rows_schema()) for run in runs]
    heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
    for stream_index, stream in enumerate(streams):
        try:
            row = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (_row_key("bad_rows", row), stream_index, row))
    rows: list[dict[str, object]] = []
    try:
        while heap:
            _, stream_index, row = heapq.heappop(heap)
            rows.append(row)
            if len(rows) >= min(max_rows, 65_536):
                writer.write_table(pyarrow.Table.from_pylist(rows, schema=_bad_rows_schema()))
                rows.clear()
            try:
                next_row = next(streams[stream_index])
            except StopIteration:
                continue
            heapq.heappush(heap, (_row_key("bad_rows", next_row), stream_index, next_row))
        if rows:
            writer.write_table(pyarrow.Table.from_pylist(rows, schema=_bad_rows_schema()))
    finally:
        writer.close()
    os.replace(temporary, path)
    shutil.rmtree(run_root, ignore_errors=True)


def _checkpoint_entry_set(
    checkpoint: RunCheckpoint, staging: Path, identity: R03RunIdentity, shard_count: int,
) -> dict[int, ShardArtifactSet]:
    if checkpoint.identity != identity or checkpoint.identity.checkpoint_compatibility_id != identity.checkpoint_compatibility_id:
        raise ValueError("checkpoint identity is incompatible with this build")
    reused: dict[int, ShardArtifactSet] = {}
    for entry in checkpoint.sealed_shards:
        artifact = _verify_sealed_shard(staging, entry, identity)
        if artifact.shard_id >= shard_count or artifact.shard_id in reused:
            raise ValueError("checkpoint has invalid or duplicate sealed shard")
        reused[artifact.shard_id] = artifact
    _remove_uncheckpointed_shards(staging, set(reused))
    return reused


def _write_checkpoint(path: Path, checkpoint: RunCheckpoint) -> RunCheckpoint:
    _atomic_json(path, checkpoint.to_dict())
    return RunCheckpoint.load(path)


def _final_artifact_record(
    root: Path, path: Path, role: str, identity: R03RunIdentity, logical_root: str,
    parent_hashes: Mapping[str, str], project_root: Path,
) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    count, lower, upper = _logical_bounds(path, role, _artifact_schema(role))
    return {
        "schema_version": _MANIFEST_FORMAT, "project_root": str(project_root.resolve()),
        "identity_json": json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":")),
        "dataset_id": identity.dataset_id, "build_id": identity.build_id,
        "logical_content_root": logical_root,
        "population_kind": identity.population_kind,
        "selection_manifest_sha256": identity.selection_manifest_sha256,
        "max_records_per_airport": identity.run_parameters.max_records_per_airport,
        "is_full_population": identity.population_kind == "full_population",
        "r04_authorized": False, "model_training_authorized": False,
        "artifact_role": role,
        "relative_path": relative, "row_count": count,
        "schema_fingerprint": _schema_fingerprint(_artifact_schema(role)),
        "physical_sha256": _sha256_file(path), "logical_lower_bound": lower,
        "logical_upper_bound": upper, "sort_contract": _SORT_CONTRACTS[role],
        "parent_hashes_json": json.dumps(dict(sorted(parent_hashes.items())), sort_keys=True, separators=(",", ":")),
        "finalized": True,
    }


def _ensure_empty_final_roles(root: Path, partitions: Sequence[FinalPartition]) -> list[tuple[str, Path]]:
    outputs = [(partition.artifact_role, partition.path) for partition in partitions]
    have = {role for role, _ in outputs}
    for role, schema in _FINAL_SCHEMAS.items():
        if role not in have:
            path = root / role / "part-00000.parquet"
            _write_empty(path, schema)
            outputs.append((role, path))
    return outputs


def _discard_prepublication_intermediates(staging: Path) -> None:
    """Do not promote checkpoints, runs, spills, or sealed work products as data."""
    for name in ("runs", "bad-runs", "sealed", "checkpoint.json"):
        path = _assert_no_reparse_escape(staging, staging / name)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for temporary in staging.rglob("*.tmp"):
        temporary.unlink(missing_ok=True)


def _verify_final_artifact_record(root: Path, record: Mapping[str, object]) -> None:
    required = {
        "schema_version", "project_root", "identity_json", "dataset_id", "build_id", "logical_content_root",
        "population_kind", "selection_manifest_sha256",
        "max_records_per_airport", "is_full_population", "r04_authorized", "model_training_authorized",
        "artifact_role", "relative_path", "row_count", "schema_fingerprint", "physical_sha256",
        "logical_lower_bound", "logical_upper_bound", "sort_contract", "parent_hashes_json", "finalized",
    }
    if set(record) != required or record.get("schema_version") != _MANIFEST_FORMAT or record.get("finalized") is not True:
        raise ValueError("canonical manifest row shape mismatch")
    role = record.get("artifact_role")
    if role not in _MANIFEST_ROLES:
        raise ValueError("canonical manifest has an unknown role")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative or ".." in Path(relative).parts or Path(relative).is_absolute():
        raise ValueError("canonical manifest path is unsafe")
    path = _assert_no_reparse_escape(root, root / relative)
    if not path.is_file() or _sha256_file(path) != record.get("physical_sha256"):
        raise ValueError("canonical manifest physical fixity mismatch")
    schema = _artifact_schema(str(role))
    file = pq.ParquetFile(path)
    try:
        if not file.schema_arrow.equals(schema, check_metadata=False):
            raise ValueError("canonical manifest artifact schema mismatch")
    finally:
        file.close()
    if record.get("schema_fingerprint") != _schema_fingerprint(schema):
        raise ValueError("canonical manifest schema fingerprint mismatch")
    count, lower, upper = _logical_bounds(path, str(role), schema)
    if count != record.get("row_count") or lower != record.get("logical_lower_bound") or upper != record.get("logical_upper_bound"):
        raise ValueError("canonical manifest logical bounds/count mismatch")
    if record.get("sort_contract") != _SORT_CONTRACTS[role]:
        raise ValueError("canonical manifest sort contract mismatch")
    try:
        parents = json.loads(str(record.get("parent_hashes_json")))
    except json.JSONDecodeError as exc:
        raise ValueError("canonical manifest parent hashes are invalid") from exc
    if not isinstance(parents, dict) or not all(isinstance(key, str) and isinstance(value, str) and len(value) == 64 for key, value in parents.items()):
        raise ValueError("canonical manifest parent hashes are invalid")


_INDEPENDENT_RAW_SCHEMAS = {
    (
        "ID", "Time", "Date", "Altitude", "Speed", "Heading", "Lat", "Lon",
        "Age", "Range", "Bearing", "Tail",
    ): "raw_v1_12",
    (
        "ID", "Time", "Date", "Altitude", "Speed", "Heading", "Lat", "Lon",
        "Age", "Range", "Bearing", "Tail", "AltisGNSS",
    ): "raw_v2_13_altisgnss",
}
_INDEPENDENT_PARSE_BITS = {
    "invalid_boolean_token": 1,
    "invalid_numeric_token": 2,
    "invalid_or_missing_aircraft_id": 4,
    "invalid_or_missing_utc_timestamp": 8,
    "nonfinite_numeric_token": 16,
    "row_parser_failure": 32,
    "unsupported_schema": 64,
}
_INDEPENDENT_QC_BITS = {
    "bearing_geometry_inconsistent": 1,
    "bearing_geometry_skipped": 2,
    "heading_out_of_domain": 4,
    "latitude_out_of_domain": 8,
    "longitude_out_of_domain": 16,
    "missing_age": 32,
    "missing_observation_fields": 64,
    "negative_age": 128,
    "negative_range": 256,
    "negative_speed": 512,
    "nonfinite_age": 1024,
    "range_geometry_inconsistent": 2048,
    "range_geometry_skipped": 4096,
    "timestamp_decrease": 8192,
}
_INDEPENDENT_COMPLETENESS = (
    "lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad", "age_s",
    "range_m", "bearing_rad", "tail_or_callsign",
)
_INDEPENDENT_CORE = ("lat_deg", "lon_deg", "altitude_m", "speed_mps", "heading_rad")
_INDEPENDENT_METADATA = (
    "age_s", "range_m", "bearing_rad", "tail_or_callsign", "altitude_source_gnss",
)


def _independent_normal(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("semantic verifier received a naive timestamp")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, tuple):
        return [_independent_normal(item) for item in value]
    if isinstance(value, list):
        return [_independent_normal(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _independent_normal(item) for key, item in sorted(value.items())}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic verifier received a forbidden nonfinite value")
        return 0.0 if value == 0.0 else value
    return value


def _independent_row_json(row: Mapping[str, object], role: str) -> str:
    logical = dict(row)
    age_rank = logical.get("age_rank_value")
    if isinstance(age_rank, float) and math.isinf(age_rank) and age_rank > 0:
        if (
            role != "report_membership"
            or type(logical.get("age_rank_class")) is not int
            or logical["age_rank_class"] != 2
        ):
            raise ValueError("semantic verifier received a forbidden nonfinite value")
        logical["age_rank_value"] = {"$r03_float": "positive_infinity"}
    return json.dumps(
        _independent_normal(logical), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _independent_identity_value(value: object) -> object:
    if value is None:
        return {"$r03_type": "null"}
    if isinstance(value, Mapping):
        return {str(key): _independent_identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_independent_identity_value(item) for item in value]
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _independent_sha256_record(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(_independent_identity_value(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _independent_text(value: object) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    return str(value).strip()


def _independent_float(value: object) -> tuple[float | None, str | None]:
    text = _independent_text(value)
    if text is None:
        return None, None
    try:
        parsed = float(text)
    except ValueError:
        return None, "invalid_float"
    if not math.isfinite(parsed):
        return None, "nonfinite_float"
    return parsed, None


def _independent_bool(value: object) -> tuple[bool | None, str | None]:
    text = _independent_text(value)
    if text is None:
        return None, None
    if text.lower() in {"true", "1"}:
        return True, None
    if text.lower() in {"false", "0"}:
        return False, None
    return None, "invalid_boolean"


def _independent_literal_parts(value: str) -> list[str] | None:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    quoted = __import__("re").findall(r"(?:u|U)?['\"]([^'\"]*)['\"]", text)
    if quoted and len(quoted) == text.count(",") + 1:
        return quoted
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(part) for part in parsed]


def _independent_datetime(date_value: object, time_value: object) -> tuple[datetime | None, str | None]:
    date_text = _independent_text(date_value)
    time_text = _independent_text(time_value)
    if date_text is None:
        return None, "missing_date"
    if time_text is None:
        return None, "missing_time"
    date = _independent_literal_parts(date_text)
    if date is None:
        if __import__("re").fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            date = date_text.split("-")
        elif __import__("re").fullmatch(r"\d{2}-\d{2}-\d{2}", date_text):
            try:
                parsed = datetime.strptime(date_text, "%m-%d-%y")
            except ValueError:
                return None, "invalid_datetime"
            date = [str(parsed.year), str(parsed.month), str(parsed.day)]
        else:
            return None, "invalid_date_format"
    if len(date) != 3:
        return None, "invalid_date_shape"
    time = _independent_literal_parts(time_text)
    if time is None:
        time = time_text.split(":")
        if len(time) != 3:
            return None, "invalid_time_format"
    if len(time) != 3:
        return None, "invalid_time_shape"
    try:
        seconds = time[2].strip().rstrip("Zz")
        decimal = Decimal(seconds)
        if not Decimal(0) <= decimal < Decimal(60):
            raise ValueError("second out of range")
        microseconds = int((decimal * Decimal(1_000_000)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        return datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), tzinfo=timezone.utc) + timedelta(microseconds=microseconds), None
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        return None, "invalid_datetime"


def _independent_bad(
    source: VerifiedR03Source, ordinal: int, source_row: int, header: Sequence[str],
    row: Sequence[str], raw_id: str, reason: str, status: str,
) -> dict[str, object]:
    safe_header = [str(token).encode("utf-8", errors="replace").decode("utf-8") for token in header]
    safe_row = [str(token).encode("utf-8", errors="replace").decode("utf-8") for token in row]
    replaced = safe_header != list(header) or safe_row != list(row)
    spec = source.spec
    return {
        "raw_row_id": raw_id, "airport_id": spec.airport_id,
        "source_file_id": spec.source_file_id, "source_relative_path": spec.relative_path,
        "archive_member": spec.archive_member, "source_content_sha256": source.source_content_sha256,
        "logical_record_ordinal": ordinal, "source_row_number": source_row,
        "source_sequence": spec.source_sequence, "raw_tokens": safe_row,
        "raw_payload": json.dumps({"header": safe_header, "row": safe_row}, ensure_ascii=True, separators=(",", ":")),
        "redaction_status": "utf8_replacement_applied" if replaced else "not_required",
        "parser_status": status, "reason_code": reason,
        "parse_quality_bits": _INDEPENDENT_PARSE_BITS[reason],
    }


def _independent_geometry(
    airport_lat: float, airport_lon: float, target_lat: float, target_lon: float, radius: float,
) -> tuple[float, float]:
    latitude = airport_lat * math.pi / 180.0
    target = target_lat * math.pi / 180.0
    delta_lat = target - latitude
    delta_lon = (target_lon - airport_lon) * math.pi / 180.0
    haversine = math.sin(delta_lat / 2.0) ** 2 + math.cos(latitude) * math.cos(target) * math.sin(delta_lon / 2.0) ** 2
    angle = 2.0 * math.atan2(math.sqrt(haversine), math.sqrt(max(0.0, 1.0 - haversine)))
    east = math.sin(delta_lon) * math.cos(target)
    north = math.cos(latitude) * math.sin(target) - math.sin(latitude) * math.cos(target) * math.cos(delta_lon)
    return radius * angle, math.degrees(math.atan2(east, north)) % 360.0


def _independent_convert(
    source: VerifiedR03Source, ordinal: int, source_row: int, header: Sequence[str], row: Sequence[str],
    raw_id: str, policies: CanonicalPolicies, previous: datetime | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    normalized = tuple(value.strip() for value in header)
    if normalized not in _INDEPENDENT_RAW_SCHEMAS:
        return None, _independent_bad(source, ordinal, source_row, header, row, raw_id, "unsupported_schema", "unsupported_schema")
    if len(row) != len(normalized):
        return None, _independent_bad(source, ordinal, source_row, header, row, raw_id, "row_parser_failure", "ValueError")
    raw = dict(zip(normalized, row, strict=True))
    values: dict[str, object] = {}
    errors: dict[str, str] = {}
    for field in normalized:
        if field in {"Altitude", "Speed", "Heading", "Lat", "Lon", "Age", "Range", "Bearing"}:
            parsed, error = _independent_float(raw[field])
        elif field == "AltisGNSS":
            parsed, error = _independent_bool(raw[field])
        else:
            parsed, error = _independent_text(raw[field]), None
        values[field] = parsed
        if error is not None:
            errors[field] = error
    values.setdefault("AltisGNSS", None)
    timestamp, timestamp_error = _independent_datetime(raw.get("Date"), raw.get("Time"))
    if values.get("ID") is None:
        return None, _independent_bad(source, ordinal, source_row, header, row, raw_id, "invalid_or_missing_aircraft_id", "missing_aircraft_id")
    aircraft = values["ID"]
    if not isinstance(aircraft, str) or not aircraft:
        return None, _independent_bad(source, ordinal, source_row, header, row, raw_id, "invalid_or_missing_aircraft_id", "invalid_aircraft_id")
    if timestamp is None:
        return None, _independent_bad(source, ordinal, source_row, header, row, raw_id, "invalid_or_missing_utc_timestamp", timestamp_error or "invalid_utc_timestamp")
    def number(name: str) -> float | None:
        value = values.get(name)
        return float(value) if isinstance(value, (int, float)) else None
    altitude, speed, heading = number("Altitude"), number("Speed"), number("Heading")
    latitude, longitude, age = number("Lat"), number("Lon"), number("Age")
    range_km, bearing = number("Range"), number("Bearing")
    parse_bits = 0
    for error in errors.values():
        if error == "invalid_float":
            parse_bits |= _INDEPENDENT_PARSE_BITS["invalid_numeric_token"]
        elif error == "nonfinite_float":
            parse_bits |= _INDEPENDENT_PARSE_BITS["nonfinite_numeric_token"]
        elif error == "invalid_boolean":
            parse_bits |= _INDEPENDENT_PARSE_BITS["invalid_boolean_token"]
    qc = 0
    def flag(code: str) -> None:
        nonlocal qc
        qc |= _INDEPENDENT_QC_BITS[code]
    if latitude is not None and not -90.0 <= latitude <= 90.0: flag("latitude_out_of_domain")
    if longitude is not None and not -180.0 <= longitude <= 180.0: flag("longitude_out_of_domain")
    if speed is not None and speed < 0.0: flag("negative_speed")
    if range_km is not None and range_km < 0.0: flag("negative_range")
    if heading is not None and not 0.0 <= heading <= 360.0: flag("heading_out_of_domain")
    if age is None: flag("nonfinite_age" if errors.get("Age") == "nonfinite_float" else "missing_age")
    elif age < 0.0: flag("negative_age")
    if any(value is None for value in (altitude, speed, heading, latitude, longitude, range_km, bearing)):
        flag("missing_observation_fields")
    if previous is not None and timestamp < previous: flag("timestamp_decrease")
    airport_values = policies.airports.get(source.spec.airport_id)
    position = (
        latitude is not None and longitude is not None and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0 and airport_values is not None
    )
    if position:
        expected_range, expected_bearing = _independent_geometry(
            float(airport_values["latitude_deg"]), float(airport_values["longitude_deg"]), latitude, longitude,
            float(policies.geometry["earth_radius_m"]),
        )
        if range_km is None:
            flag("range_geometry_skipped")
        else:
            tolerance = max(float(policies.geometry["range_absolute_tolerance_m"]), float(policies.geometry["range_relative_tolerance"]) * expected_range)
            if abs(range_km * 1000.0 - expected_range) > tolerance: flag("range_geometry_inconsistent")
        if bearing is None or expected_range == 0.0:
            flag("bearing_geometry_skipped")
        else:
            circular = abs((bearing - expected_bearing + 180.0) % 360.0 - 180.0)
            if circular > float(policies.geometry["bearing_absolute_tolerance_deg"]): flag("bearing_geometry_inconsistent")
    else:
        flag("range_geometry_skipped")
        flag("bearing_geometry_skipped")
    return {
        "raw_row_id": raw_id, "airport_id": source.spec.airport_id,
        "source_file_id": source.spec.source_file_id, "source_relative_path": source.spec.relative_path,
        "archive_member": source.spec.archive_member, "source_content_sha256": source.source_content_sha256,
        "logical_record_ordinal": ordinal, "source_row_number": source_row, "source_sequence": source.spec.source_sequence,
        "aircraft_id": aircraft, "tail_or_callsign": values.get("Tail"),
        "report_timestamp_utc": timestamp, "lat_deg": latitude, "lon_deg": longitude,
        "altitude_m": None if altitude is None else altitude * 0.3048,
        "speed_mps": None if speed is None else speed * 0.5144444444444445,
        "heading_rad": None if heading is None else heading * math.pi / 180.0,
        "age_s": age, "range_m": None if range_km is None else range_km * 1000.0,
        "bearing_rad": None if bearing is None else bearing * math.pi / 180.0,
        "altitude_source_gnss": values.get("AltisGNSS"), "raw_schema_version": _INDEPENDENT_RAW_SCHEMAS[normalized],
        "parse_quality_bits": parse_bits, "qc_bits": qc, "raw_altitude_ft": altitude,
        "raw_speed_knots": speed, "raw_heading_deg": heading, "raw_lat_deg": latitude,
        "raw_lon_deg": longitude, "raw_age_s": age, "raw_range_km": range_km,
        "raw_bearing_deg": bearing, "raw_tail": values.get("Tail"),
        "raw_altisgnss": values.get("AltisGNSS"), "partition_year": timestamp.year,
        "partition_month": timestamp.month,
    }, None


def _independent_source_records(source: VerifiedR03Source) -> Iterator[tuple[int, int, tuple[str, ...], tuple[str, ...], str]]:
    def iterate(binary: object) -> Iterator[tuple[int, int, tuple[str, ...], tuple[str, ...], str]]:
        with __import__("io").TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as handle:
            csv.field_size_limit(64 * 1024 * 1024)
            reader = csv.reader(handle, strict=False)
            try:
                header = tuple(next(reader))
            except StopIteration:
                return
            pending: tuple[tuple[str, ...], int] | None = None
            ordinal = 0
            for source_row, row in enumerate(reader, start=2):
                row_tuple = tuple(row)
                if not row_tuple or all(not value.strip() for value in row_tuple):
                    continue
                if pending is not None:
                    ordinal += 1
                    raw_id = _independent_sha256_record({
                        "version": "r03_canonical_points_v1", "source_file_id": source.spec.source_file_id,
                        "source_content_sha256": source.source_content_sha256,
                        "archive_member": source.spec.archive_member, "logical_record_ordinal": ordinal,
                    })
                    yield ordinal, pending[1], header, pending[0], raw_id
                pending = (row_tuple, source_row)
            if pending is not None and source.spec.r01_disposition != "drop_terminal_record_in_R02":
                ordinal += 1
                raw_id = _independent_sha256_record({
                    "version": "r03_canonical_points_v1", "source_file_id": source.spec.source_file_id,
                    "source_content_sha256": source.source_content_sha256,
                    "archive_member": source.spec.archive_member, "logical_record_ordinal": ordinal,
                })
                yield ordinal, pending[1], header, pending[0], raw_id
    if source.materialization_kind == "extracted_file":
        if _sha256_file(source.resolved_path) != source.source_content_sha256:
            raise ValueError("semantic verifier source content fixity mismatch")
        with source.resolved_path.open("rb") as binary:
            yield from iterate(binary)
        return
    if source.materialization_kind != "archive_member" or not source.spec.archive_member:
        raise ValueError("semantic verifier source materialization is invalid")
    with zipfile.ZipFile(source.resolved_path) as archive:
        with archive.open(source.spec.archive_member, "r") as binary:
            digest = hashlib.sha256()
            while chunk := binary.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != source.source_content_sha256:
            raise ValueError("semantic verifier archive member fixity mismatch")
        with archive.open(source.spec.archive_member, "r") as binary:
            yield from iterate(binary)


def _independent_selected_source_records(
    source: VerifiedR03Source,
    selected_ordinals: tuple[int, ...],
) -> Iterator[tuple[int, int, tuple[str, ...], tuple[str, ...], str]]:
    """Independent ordinal filter used only by the semantic verification oracle."""

    if (
        not selected_ordinals
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in selected_ordinals
        )
        or any(left >= right for left, right in zip(selected_ordinals, selected_ordinals[1:]))
        or selected_ordinals[-1] > source.spec.authenticated_row_count
    ):
        raise ValueError("semantic verifier selected ordinals are invalid")
    wanted = frozenset(selected_ordinals)
    maximum = selected_ordinals[-1]
    observed = 0
    for record in _independent_source_records(source):
        observed = record[0]
        if observed in wanted:
            yield record
        if observed == maximum:
            break
    if observed < maximum:
        raise ValueError(
            "selected source count does not match authenticated R02 row_count"
        )


def _independent_age_rank(age: object) -> tuple[int, float]:
    if age is None:
        return 2, math.inf
    value = float(age)
    if not math.isfinite(value):
        return 2, math.inf
    return (0, 0.0 if value == 0.0 else value) if value >= 0.0 else (1, abs(value))


def _independent_report_id(airport: str, aircraft: str, timestamp: str) -> str:
    return _independent_sha256_record({
        "version": "r03_canonical_points_v1", "airport_id": airport,
        "aircraft_id": aircraft, "report_timestamp_utc": timestamp,
    })


def _independent_artifact_paths(root: Path, role: str) -> list[Path]:
    directory = _assert_no_reparse_escape(root, root / role)
    if not directory.is_dir():
        raise ValueError(f"semantic {role} directory is missing")
    paths = sorted(path for path in directory.rglob("*.parquet") if path.is_file())
    if not paths:
        raise ValueError(f"semantic {role} artifacts are missing")
    return [_assert_no_reparse_escape(root, path) for path in paths]


def _independent_parquet_rows(paths: Sequence[Path], schema: pyarrow.Schema) -> Iterator[dict[str, object]]:
    for path in paths:
        file = pq.ParquetFile(path)
        try:
            if not file.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(f"semantic artifact schema mismatch: {path}")
            for batch in file.iter_batches(batch_size=65_536):
                epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
                for index in range(batch.num_rows):
                    row: dict[str, object] = {}
                    for column, name in enumerate(batch.schema.names):
                        scalar = batch.column(column)[index]
                        type_ = batch.schema.field(column).type
                        if pyarrow.types.is_timestamp(type_) and type_.tz == "UTC" and scalar.is_valid:
                            row[name] = epoch + timedelta(microseconds=int(scalar.value))
                        else:
                            row[name] = scalar.as_py()
                    yield row
        finally:
            file.close()


def _independent_compare_role(
    final_root: Path, role: str, schema: pyarrow.Schema, expected: Iterator[str],
) -> None:
    actual = (
        _independent_row_json(row, role)
        for row in _independent_parquet_rows(
            _independent_artifact_paths(final_root, role), schema
        )
    )
    for observed, trusted in zip_longest(actual, expected):
        if observed != trusted:
            raise ValueError(f"semantic {role} ledger mismatch")


def _verify_recomputed_semantics(
    final_root: Path,
    project_root: Path,
    sources: Sequence[VerifiedR03Source],
    parameters: R03RunParameters,
    scratch: Path,
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None,
) -> None:
    """Independent bounded P/B/U/M/C oracle over authenticated source bytes.

    This deliberately does not call build conversion, sharding, group resolver,
    final merge, source enumeration, or producer parquet helpers.  SQLite is
    used only as an external bounded sort/group store; all semantic rows are
    parsed and resolved by the local oracle before ordered comparison.
    """
    scratch.mkdir(parents=True, exist_ok=False)
    database = scratch / "semantic-oracle.sqlite3"
    connection = __import__("sqlite3").connect(database)
    try:
        connection.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE parsed (
              raw_id TEXT PRIMARY KEY, row_json TEXT NOT NULL, airport TEXT NOT NULL,
              year INTEGER NOT NULL, month INTEGER NOT NULL, aircraft TEXT NOT NULL,
              timestamp TEXT NOT NULL, age_class INTEGER NOT NULL,
              age_value REAL NOT NULL, completeness INTEGER NOT NULL, source_sequence INTEGER NOT NULL,
              source_file TEXT NOT NULL, source_relative TEXT NOT NULL, archive_member TEXT NOT NULL,
              source_row INTEGER NOT NULL
            );
            CREATE TABLE bad (raw_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE unique_rows (report_id TEXT PRIMARY KEY, airport TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL, aircraft TEXT NOT NULL, timestamp TEXT NOT NULL, row_json TEXT NOT NULL);
            CREATE TABLE membership_rows (airport TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL, report_id TEXT NOT NULL, rank INTEGER NOT NULL, raw_id TEXT NOT NULL, row_json TEXT NOT NULL);
            CREATE TABLE conflict_rows (airport TEXT NOT NULL, year INTEGER NOT NULL, month INTEGER NOT NULL, report_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
        """)
        policies = CanonicalPolicies.from_project(project_root)
        previous: dict[tuple[str, int], datetime | None] = {}
        for source in sources:
            airport = source.spec.airport_id
            records = (
                _independent_source_records(source)
                if selected_ordinals is None
                else _independent_selected_source_records(
                    source, selected_ordinals[source.spec.source_file_id]
                )
            )
            for ordinal, source_row, header, raw, raw_id in records:
                key = (airport, source.spec.source_sequence)
                parsed, bad = _independent_convert(source, ordinal, source_row, header, raw, raw_id, policies, previous.get(key))
                if parsed is None:
                    assert bad is not None
                    connection.execute(
                        "INSERT INTO bad VALUES (?, ?)",
                        (raw_id, _independent_row_json(bad, "bad_rows")),
                    )
                    continue
                timestamp = parsed["report_timestamp_utc"]
                assert isinstance(timestamp, datetime)
                previous[key] = timestamp
                age_class, age_value = _independent_age_rank(parsed["age_s"])
                completeness = sum(parsed[field] is not None for field in _INDEPENDENT_COMPLETENESS)
                connection.execute(
                    "INSERT INTO parsed VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        raw_id, _independent_row_json(parsed, "parsed_rows"), parsed["airport_id"],
                        parsed["partition_year"], parsed["partition_month"], parsed["aircraft_id"],
                        _independent_normal(timestamp), age_class, age_value, completeness,
                        parsed["source_sequence"], parsed["source_file_id"], parsed["source_relative_path"],
                        str(parsed["archive_member"] or "").replace("\\", "/"), parsed["source_row_number"],
                    ),
                )
        group_query = (
            "SELECT raw_id, row_json FROM parsed WHERE airport=? AND aircraft=? AND timestamp=? "
            "ORDER BY age_class, age_value, completeness DESC, airport, source_sequence, "
            "source_file, source_relative, archive_member, source_row, raw_id"
        )
        for airport, aircraft, timestamp in connection.execute(
            "SELECT airport, aircraft, timestamp FROM parsed GROUP BY airport, aircraft, timestamp ORDER BY airport, aircraft, timestamp"
        ):
            group_key = (airport, aircraft, timestamp)
            winner_record = connection.execute(group_query, group_key).fetchone()
            if winner_record is None:
                raise ValueError("semantic oracle group unexpectedly empty")
            winner_id = str(winner_record[0])
            winner = json.loads(str(winner_record[1]))
            winner_age = _independent_age_rank(winner["age_s"])
            winner_complete = sum(
                winner[field] is not None for field in _INDEPENDENT_COMPLETENESS
            )
            size = 0
            same_age = 0
            same_complete = 0
            core_bits = 0
            metadata_bits = 0
            # The rows stay in SQLite.  This cursor decodes one record at a
            # time, so even a single natural-key group larger than the producer
            # spill limit has O(1) Python working memory.
            for raw_id, row_json in connection.execute(group_query, group_key):
                row = json.loads(str(row_json))
                size += 1
                age_rank = _independent_age_rank(row["age_s"])
                completeness = sum(
                    row[field] is not None for field in _INDEPENDENT_COMPLETENESS
                )
                if age_rank == winner_age:
                    same_age += 1
                    if completeness == winner_complete:
                        same_complete += 1
                for index, field in enumerate(_INDEPENDENT_CORE):
                    if row[field] != winner[field]:
                        core_bits |= 1 << index
                for index, field in enumerate(_INDEPENDENT_METADATA):
                    if row[field] != winner[field]:
                        metadata_bits |= 1 << index
            if size == 1:
                reason, classification = "only_member", "singleton"
            else:
                reason = (
                    "min_nonnegative_age" if winner_age[0] == 0 else "closest_zero_negative_age"
                ) if same_age == 1 else (
                    "max_completeness_after_age_tie" if same_complete == 1 else "stable_source_order_after_full_tie"
                )
                classification = "core_and_metadata_conflict" if core_bits and metadata_bits else (
                    "core_state_conflict" if core_bits else ("metadata_conflict" if metadata_bits else "identical")
                )
            report_id = _independent_report_id(str(airport), str(aircraft), str(timestamp))
            unique = {
                "canonical_report_id": report_id, "selected_raw_row_id": winner_id,
                "airport_id": winner["airport_id"], "aircraft_id": winner["aircraft_id"],
                "report_timestamp_utc": timestamp, "tail_or_callsign": winner["tail_or_callsign"],
                "lat_deg": winner["lat_deg"], "lon_deg": winner["lon_deg"], "altitude_m": winner["altitude_m"],
                "speed_mps": winner["speed_mps"], "heading_rad": winner["heading_rad"], "age_s": winner["age_s"],
                "range_m": winner["range_m"], "bearing_rad": winner["bearing_rad"],
                "altitude_source_gnss": winner["altitude_source_gnss"], "group_size": size,
                "is_duplicate": size > 1, "duplicate_classification": classification,
                "selection_policy_version": "r03_canonical_points_v1", "core_state_difference_bits": core_bits,
                "metadata_difference_bits": metadata_bits, "duplicate_conflict_flag": core_bits != 0,
                "metadata_conflict_flag": metadata_bits != 0, "any_difference_flag": bool(core_bits or metadata_bits),
                "parse_quality_bits": winner["parse_quality_bits"], "qc_bits": winner["qc_bits"],
                "partition_year": winner["partition_year"], "partition_month": winner["partition_month"],
            }
            connection.execute(
                "INSERT INTO unique_rows VALUES (?,?,?,?,?,?,?)",
                (
                    report_id, airport, winner["partition_year"], winner["partition_month"],
                    aircraft, timestamp, _independent_row_json(unique, "unique_reports"),
                ),
            )
            # A second disk-backed pass emits membership ranks in the same
            # authoritative order without retaining the prior group.
            for rank, (raw_id, row_json) in enumerate(
                connection.execute(group_query, group_key), start=1
            ):
                raw_id = str(raw_id)
                row = json.loads(str(row_json))
                age_class, age_value = _independent_age_rank(row["age_s"])
                membership = {
                    "raw_row_id": raw_id, "canonical_report_id": report_id, "group_size": size,
                    "dedup_rank": rank, "selected": rank == 1,
                    "selection_reason": reason if rank == 1 else "not_selected",
                    "age_rank_class": age_class, "age_rank_value": age_value,
                    "completeness_score": sum(row[field] is not None for field in _INDEPENDENT_COMPLETENESS),
                    "completeness_denominator": len(_INDEPENDENT_COMPLETENESS),
                    "core_state_difference_bits": core_bits, "metadata_difference_bits": metadata_bits,
                    "airport_id": row["airport_id"], "source_file_id": row["source_file_id"],
                    "source_relative_path": row["source_relative_path"], "archive_member": row["archive_member"],
                    "source_content_sha256": row["source_content_sha256"],
                    "logical_record_ordinal": row["logical_record_ordinal"], "source_row_number": row["source_row_number"],
                    "source_sequence": row["source_sequence"], "partition_year": row["partition_year"],
                    "partition_month": row["partition_month"],
                }
                connection.execute(
                    "INSERT INTO membership_rows VALUES (?,?,?,?,?,?,?)",
                    (
                        row["airport_id"], row["partition_year"], row["partition_month"],
                        report_id, rank, raw_id,
                        _independent_row_json(membership, "report_membership"),
                    ),
                )
            if core_bits or metadata_bits:
                conflict = {
                    "canonical_report_id": report_id, "airport_id": winner["airport_id"],
                    "partition_year": winner["partition_year"], "partition_month": winner["partition_month"],
                    "duplicate_classification": classification, "core_state_difference_bits": core_bits,
                    "metadata_difference_bits": metadata_bits, "group_size": size, "selected_raw_row_id": winner_id,
                }
                connection.execute(
                    "INSERT INTO conflict_rows VALUES (?,?,?,?,?)",
                    (
                        winner["airport_id"], winner["partition_year"], winner["partition_month"],
                        report_id, _independent_row_json(conflict, "duplicate_conflicts"),
                    ),
                )
        connection.commit()
        checks = (
            ("bad_rows", _bad_rows_schema(), "SELECT row_json FROM bad ORDER BY raw_id"),
            ("parsed_rows", parsed_rows_schema(), "SELECT row_json FROM parsed ORDER BY airport, year, month, aircraft, timestamp, source_sequence, source_file, source_relative, archive_member, source_row, raw_id"),
            ("unique_reports", unique_reports_schema(), "SELECT row_json FROM unique_rows ORDER BY airport, year, month, aircraft, timestamp"),
            ("report_membership", membership_schema(), "SELECT row_json FROM membership_rows ORDER BY airport, year, month, report_id, rank, raw_id"),
            ("duplicate_conflicts", _conflict_schema(), "SELECT row_json FROM conflict_rows ORDER BY airport, year, month, report_id"),
        )
        for role, schema, query in checks:
            _independent_compare_role(final_root, role, schema, (str(row[0]) for row in connection.execute(query)))
    finally:
        connection.close()


def verify_canonical_manifest(manifest_path: Path, expected_project_root: Path) -> dict[str, object]:
    """Verify data, schema, ordering, lineage, and conservation without trust in the manifest."""
    project_root = Path(expected_project_root).resolve()
    lexical_path = Path(manifest_path)
    lexical_root = lexical_path.parent
    _assert_no_reparse_escape(lexical_root, lexical_path)
    path = lexical_path.resolve()
    final_root = path.parent
    if path.name != "canonical_point_manifest.parquet" or not path.is_file():
        raise ValueError("canonical manifest path is invalid")
    file = pq.ParquetFile(path)
    try:
        if not file.schema_arrow.equals(_manifest_schema(), check_metadata=False):
            raise ValueError("canonical manifest schema mismatch")
        rows = list(_iter_mappings(path, _manifest_schema()))
    finally:
        file.close()
    if not rows:
        raise ValueError("canonical manifest must name mandatory artifacts")
    shared = {
        "schema_version": rows[0].get("schema_version"), "project_root": rows[0].get("project_root"),
        "identity_json": rows[0].get("identity_json"),
        "dataset_id": rows[0].get("dataset_id"), "build_id": rows[0].get("build_id"),
        "logical_content_root": rows[0].get("logical_content_root"),
        "population_kind": rows[0].get("population_kind"),
        "selection_manifest_sha256": rows[0].get("selection_manifest_sha256"),
        "max_records_per_airport": rows[0].get("max_records_per_airport"),
        "is_full_population": rows[0].get("is_full_population"),
        "r04_authorized": rows[0].get("r04_authorized"),
        "model_training_authorized": rows[0].get("model_training_authorized"),
    }
    if (
        shared["project_root"] != str(project_root)
        or not all(isinstance(shared[key], str) and shared[key] for key in ("schema_version", "project_root", "identity_json", "dataset_id", "build_id", "logical_content_root"))
        or not isinstance(shared["is_full_population"], bool)
        or shared["r04_authorized"] is not False
        or shared["model_training_authorized"] is not False
        or shared["max_records_per_airport"] is not None
        or shared["population_kind"] not in {"full_population", "capacity_sample"}
        or shared["is_full_population"]
        != (shared["population_kind"] == "full_population")
        or (
            shared["population_kind"] == "full_population"
            and shared["selection_manifest_sha256"] is not None
        )
        or (
            shared["population_kind"] == "capacity_sample"
            and (
                not isinstance(shared["selection_manifest_sha256"], str)
                or len(str(shared["selection_manifest_sha256"])) != 64
            )
        )
    ):
        raise ValueError("canonical manifest trusted lineage root mismatch")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if any(row.get(key) != value for key, value in shared.items()):
            raise ValueError("canonical manifest has mixed identity values")
        identity = (str(row["artifact_role"]), str(row["relative_path"]))
        if identity in seen:
            raise ValueError("canonical manifest names a duplicate artifact")
        seen.add(identity)
        _verify_final_artifact_record(final_root, row)
    if {role for role, _ in seen} != set(_MANIFEST_ROLES):
        raise ValueError("canonical manifest mandatory roles are incomplete")
    actual_logical_root = _logical_content_root(rows, final_root)
    if actual_logical_root != shared["logical_content_root"]:
        raise ValueError("canonical manifest logical-content root mismatch")

    registry = R03SourceRegistry.from_lineage(project_root)
    try:
        declared_identity = R03RunIdentity.from_dict(
            json.loads(str(shared["identity_json"]))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("canonical manifest identity payload is invalid") from exc
    selection_manifest: CapacitySelectionManifest | None = None
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None
    if shared["population_kind"] == "capacity_sample":
        selection_path = _assert_no_reparse_escape(
            final_root, final_root / "capacity_selection_manifest.json"
        )
        try:
            selection_manifest = load_and_verify_capacity_selection(
                selection_path, registry
            )
        except (OSError, ValueError) as exc:
            raise ValueError("canonical capacity selection manifest is invalid") from exc
        selected_specs, selected_ordinals, selection_bytes, selection_hash = (
            _verified_selection_inputs(registry, selection_manifest)
        )
        if (
            selection_hash != shared["selection_manifest_sha256"]
            or selection_bytes != selection_path.read_bytes()
        ):
            raise ValueError("canonical selection manifest fixity mismatch")
    else:
        selected_specs = registry.sources
    airport_roots = _load_airport_roots(project_root)
    sources = tuple(verify_source(spec, airport_roots) for spec in selected_specs)
    expected_identity = R03RunIdentity.from_inputs(
        project_root,
        sources,
        declared_identity.run_parameters,
        selection_manifest=selection_manifest,
    )
    if declared_identity != expected_identity:
        raise ValueError("canonical manifest identity is not trusted-lineage derived")
    if (
        shared["dataset_id"] != expected_identity.dataset_id
        or shared["build_id"] != expected_identity.build_id
    ):
        raise ValueError("canonical manifest dataset/build identity mismatch")
    parents = json.loads(str(rows[0]["parent_hashes_json"]))
    expected_parents = {
        "raw_manifest": registry.raw_manifest_sha256,
        "archive_comparison": registry.archive_comparison_sha256,
        "r02_run_manifest": registry.r02_run_manifest_sha256,
        "r02_decision": registry.r02_decision_sha256,
        "source_sequence": registry.source_sequence_manifest_sha256,
    }
    if selection_manifest is not None:
        expected_parents["capacity_selection"] = str(
            shared["selection_manifest_sha256"]
        )
    if set(parents) != set(expected_parents) | {"source_enumeration"} or any(
        parents.get(key) != value for key, value in expected_parents.items()
    ):
        raise ValueError("canonical manifest parent lineage mismatch")
    if any(
        json.loads(str(row["parent_hashes_json"])) != parents
        for row in rows
    ):
        raise ValueError("canonical manifest has mixed parent lineage")
    enumeration_record = next(row for row in rows if row["artifact_role"] == "source_enumeration")
    if parents.get("source_enumeration") != enumeration_record.get("physical_sha256"):
        raise ValueError("canonical manifest source enumeration parent mismatch")
    enumeration_path = _assert_no_reparse_escape(final_root, final_root / str(enumeration_record["relative_path"]))
    actual_enumeration = list(_iter_mappings(enumeration_path, _enumeration_schema()))
    if actual_enumeration != _recompute_enumeration_results(
        registry, sources, selected_ordinals
    ):
        raise ValueError("canonical manifest source enumeration mismatch")
    scratch = _verification_scratch_root(final_root)
    try:
        _verify_recomputed_semantics(
            final_root, project_root, sources, expected_identity.run_parameters,
            scratch / "semantic", selected_ordinals,
        )
        conservation = verify_layered_conservation(
            final_root, sources, scratch, selected_ordinals,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not conservation.passed:
        raise ValueError(f"canonical manifest conservation failed: {conservation.failures}")
    return {
        "schema_version": str(shared["schema_version"]), "project_root": str(shared["project_root"]),
        "dataset_id": str(shared["dataset_id"]), "build_id": str(shared["build_id"]),
        "logical_content_root": str(shared["logical_content_root"]),
        "canonical_manifest_sha256": _sha256_file(path), "manifest_path": str(path),
        "identity": expected_identity.to_dict(),
        "counts": dict(conservation.counts),
    }


def _pointer_record(
    project_root: Path, final_root: Path, verified: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": _POINTER_FORMAT, "project_root": str(Path(project_root).resolve()),
        "dataset_id": verified["dataset_id"], "build_id": verified["build_id"],
        "final_directory": final_root.name,
        "manifest_relative_path": (Path(final_root.name) / "canonical_point_manifest.parquet").as_posix(),
        "canonical_manifest_sha256": verified["canonical_manifest_sha256"],
        "logical_content_root": verified["logical_content_root"], "finalized": True,
    }


def verify_current_pointer(output_root: Path, expected_project_root: Path) -> dict[str, object]:
    lexical_output = Path(output_root).absolute()
    _reject_existing_reparse_components(lexical_output)
    output = lexical_output.resolve()
    pointer_path = _assert_no_reparse_escape(output, output / "current.json")
    pointer = _read_json_object(pointer_path)
    required = {
        "schema_version", "project_root", "dataset_id", "build_id", "final_directory",
        "manifest_relative_path", "canonical_manifest_sha256", "logical_content_root", "finalized",
    }
    if set(pointer) != required or pointer.get("schema_version") != _POINTER_FORMAT or pointer.get("finalized") is not True:
        raise ValueError("current pointer shape mismatch")
    root = Path(expected_project_root).resolve()
    if pointer.get("project_root") != str(root):
        raise ValueError("current pointer trusted project root mismatch")
    final_directory = pointer.get("final_directory")
    manifest_relative = pointer.get("manifest_relative_path")
    if not isinstance(final_directory, str) or not isinstance(manifest_relative, str):
        raise ValueError("current pointer path is invalid")
    final_root = _assert_no_reparse_escape(output, output / final_directory)
    manifest = _assert_no_reparse_escape(output, output / manifest_relative)
    if manifest.parent != final_root or final_root.name != f"r03-{pointer.get('dataset_id')}":
        raise ValueError("current pointer containment or identity mismatch")
    if not manifest.is_file() or _sha256_file(manifest) != pointer.get("canonical_manifest_sha256"):
        raise ValueError("current pointer manifest fixity mismatch")
    verified = verify_canonical_manifest(manifest, root)
    for key in ("dataset_id", "build_id", "logical_content_root"):
        if pointer.get(key) != verified.get(key):
            raise ValueError("current pointer identity mismatch")
    return verified


def _build_shard_runs(
    sources: Sequence[VerifiedR03Source], root: Path, staging: Path,
    parameters: R03RunParameters,
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[int, list[ShardRun]], dict[int, Path], list[SourceEnumerationResult]]:
    """Convert the authenticated input stream while keeping only one record batch resident."""
    policies = CanonicalPolicies.from_project(root)
    coordinates = {
        airport: (float(values["latitude_deg"]), float(values["longitude_deg"]))
        for airport, values in policies.airports.items()
    }
    runs: dict[int, list[ShardRun]] = {index: [] for index in range(parameters.shard_count)}
    next_run_index = [0] * parameters.shard_count
    bad_writers, bad_buffers, bad_paths = _new_bad_writers(staging, parameters.shard_count)
    parsed_buffer: list[ParsedRow] = []
    source_results: list[SourceEnumerationResult] = []
    batch_index = 0

    def flush_parsed() -> None:
        nonlocal batch_index
        if not parsed_buffer:
            return
        batches = pyarrow.Table.from_pylist(
            [asdict(row) for row in parsed_buffer], schema=parsed_rows_schema()
        ).to_batches()
        batch_root = staging / "runs" / f"batch-{batch_index:08d}"
        batch_index += 1
        for run in spill_parsed_batches(
            batches, batch_root, parameters.shard_count, parameters.max_rows_per_run,
            parameters.shard_hash_version,
        ):
            shifted = replace(
                # ``spill_parsed_batches`` starts its index at zero for every
                # input batch.  Allocate the published descriptor index from
                # the per-shard global counter instead of adding its local
                # index after incrementing, which could repeat an index when
                # a multi-run batch is followed by another batch.
                run, run_index=next_run_index[run.shard_id],
                relative_path=(Path("batch-%08d" % (batch_index - 1)) / run.relative_path).as_posix(),
            )
            next_run_index[run.shard_id] += 1
            runs[run.shard_id].append(shifted)
        parsed_buffer.clear()

    try:
        for source in sources:
            count = 0
            header_hash: str | None = None
            first_id: str | None = None
            last_id: str | None = None
            first_number: int | None = None
            last_number: int | None = None
            previous_timestamp: datetime | None = None
            records = (
                enumerate_source(source)
                if selected_ordinals is None
                else enumerate_selected_source(
                    source, selected_ordinals[source.spec.source_file_id]
                )
            )
            for record in records:
                count += 1
                header_hash = header_hash or hashlib.sha256(
                    canonical_json({"header": list(record.header)})
                ).hexdigest()
                first_id = record.raw_row_id if first_id is None else first_id
                last_id = record.raw_row_id
                first_number = record.source_row_number if first_number is None else first_number
                last_number = record.source_row_number
                converted = convert_record(record, policies, coordinates, previous_timestamp)
                if isinstance(converted, ParsedRow):
                    previous_timestamp = converted.report_timestamp_utc
                    parsed_buffer.append(converted)
                    if len(parsed_buffer) >= parameters.record_batch_rows:
                        flush_parsed()
                else:
                    shard_id = int.from_bytes(hashlib.sha256(record.raw_row_id.encode("ascii")).digest(), "big") % parameters.shard_count
                    bad_buffers[shard_id].append(asdict(converted))
                    if len(bad_buffers[shard_id]) >= parameters.record_batch_rows:
                        _write_bad_shards(bad_writers, bad_buffers)
            source_results.append(
                SourceEnumerationResult(
                    registry_id="", r02_partition_manifest_sha256="", source=source,
                    header_sha256=header_hash or hashlib.sha256(canonical_json({"header": []})).hexdigest(),
                    logical_record_count=count, first_raw_row_id=first_id, last_raw_row_id=last_id,
                    first_source_row_number=first_number, last_source_row_number=last_number,
                    status="complete",
                )
            )
        flush_parsed()
    finally:
        _close_bad_writers(bad_writers, bad_buffers)
    return runs, bad_paths, source_results


def _bind_source_results(registry: R03SourceRegistry, results: Sequence[SourceEnumerationResult]) -> list[SourceEnumerationResult]:
    partitions = dict(registry.partition_manifest_sha256_by_airport)
    return [
        replace(
            result, registry_id=registry.registry_id,
            r02_partition_manifest_sha256=partitions[result.source.spec.airport_id],
        )
        for result in results
    ]


def _recompute_enumeration_results(
    registry: R03SourceRegistry, sources: Sequence[VerifiedR03Source],
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None,
) -> list[dict[str, object]]:
    results: list[SourceEnumerationResult] = []
    for source in sources:
        count = 0
        header_hash: str | None = None
        first_id: str | None = None
        last_id: str | None = None
        first_number: int | None = None
        last_number: int | None = None
        records = (
            enumerate_source(source)
            if selected_ordinals is None
            else enumerate_selected_source(
                source, selected_ordinals[source.spec.source_file_id]
            )
        )
        for record in records:
            count += 1
            header_hash = header_hash or hashlib.sha256(
                canonical_json({"header": list(record.header)})
            ).hexdigest()
            first_id = record.raw_row_id if first_id is None else first_id
            last_id = record.raw_row_id
            first_number = record.source_row_number if first_number is None else first_number
            last_number = record.source_row_number
        results.append(SourceEnumerationResult(
            registry_id="", r02_partition_manifest_sha256="", source=source,
            header_sha256=header_hash or hashlib.sha256(canonical_json({"header": []})).hexdigest(),
            logical_record_count=count, first_raw_row_id=first_id, last_raw_row_id=last_id,
            first_source_row_number=first_number, last_source_row_number=last_number,
            status="complete",
        ))
    return enumeration_manifest(
        registry,
        _bind_source_results(registry, results),
        selected_ordinals,
    )


def _seal_missing_shards(
    staging: Path, identity: R03RunIdentity, runs: Mapping[int, Sequence[ShardRun]],
    bad_paths: Mapping[int, Path], parameters: R03RunParameters,
    checkpoint: RunCheckpoint, checkpoint_path: Path, reused: dict[int, ShardArtifactSet],
    observer: R03BuildObserver | None,
) -> tuple[RunCheckpoint, dict[int, ShardArtifactSet]]:
    entries = list(checkpoint.sealed_shards)
    for shard_id in range(parameters.shard_count):
        if shard_id in reused:
            continue
        shard_root = staging / "sealed"
        writer = GroupArtifactWriters(
            shard_root, shard_id, shard_count=parameters.shard_count,
            shard_hash_version=parameters.shard_hash_version,
            final_sort_version=parameters.final_sort_version,
        )
        resolve_sorted_shard(runs[shard_id], writer, parameters.max_group_rows_in_memory)
        artifact = writer.artifact_set
        if artifact is None:
            raise RuntimeError("sealed shard writer did not produce an artifact set")
        bad_target = artifact.parsed_rows.parent / "bad_rows.parquet"
        os.replace(bad_paths[shard_id], bad_target)
        _sort_bad_file(bad_target, parameters.max_rows_per_run)
        entry = _write_sealed_shard_manifest(artifact.parsed_rows.parent, shard_id, identity)
        # A sealed directory becomes reusable only after this atomic checkpoint update.
        _termination_seam("after-shard-sealing")
        entries.append(entry)
        checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("SHARDING", entries))
        if observer is not None:
            observer.sample("shard_seal", staging)
        reused[shard_id] = _verify_sealed_shard(staging, entry, identity)
    return checkpoint, reused


def _available_disk_bytes(path: Path) -> int:
    """Return the currently available capacity of the publication volume."""
    return int(shutil.disk_usage(path).free)


def _rss_bytes() -> int:
    """Read this process's working set without a third-party monitoring dependency."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
            )
            if ok:
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return 0
        return 0
    try:
        import resource

        # Linux reports KiB while macOS reports bytes.
        resident = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return resident if sys.platform == "darwin" else resident * 1024
    except (ImportError, AttributeError, OSError):
        return 0


def _preflight_capacity(
    sources: Sequence[VerifiedR03Source], parameters: R03RunParameters, output_root: Path,
) -> dict[str, object]:
    """Enumerate once and reject an infeasible local run before artifact creation."""
    source_records = 0
    for source in sources:
        source_records += sum(1 for _ in enumerate_source(source))
    batch_bytes = parameters.record_batch_rows * 2_048
    group_bytes = parameters.max_group_rows_in_memory * 1_024
    estimated_working_set = max(1, batch_bytes + group_bytes)
    estimated_disk = max(
        1,
        source_records * 4_096 + batch_bytes * 2 + parameters.shard_count * 65_536,
    )
    available = _available_disk_bytes(output_root)
    rss = _rss_bytes()
    memory_limit = max(64 * 1024 * 1024, estimated_working_set * 4)
    disk_ok = available >= estimated_disk
    memory_ok = rss <= memory_limit
    return {
        "schema_version": "r03_preflight_capacity_v1",
        "enumerated_source_records": source_records,
        "record_batch_rows": parameters.record_batch_rows,
        "max_group_rows_in_memory": parameters.max_group_rows_in_memory,
        "estimated_batch_bytes": batch_bytes,
        "estimated_working_set_bytes": estimated_working_set,
        "estimated_disk_bytes": estimated_disk,
        "available_disk_bytes": available,
        "rss_bytes": rss,
        "memory_limit_bytes": memory_limit,
        "disk_capacity": disk_ok,
        "memory_capacity": memory_ok,
    }


def build_canonical_points(
    project_root: Path, output_root: Path, parameters: R03RunParameters, resume: bool,
    preflight_only: bool = False,
    *,
    selection_manifest: CapacitySelectionManifest | None = None,
    publication_mode: Literal["project_record", "external_evidence_only"] = "project_record",
    observer: R03BuildObserver | None = None,
) -> R03RunResult:
    """Build and publish canonical R03 ledgers with checkpoint-referenced reuse only.

    This supports tested local-NTFS single-writer process-crash consistency.  It
    intentionally does not claim power-loss, network-share, or universal OS
    crash durability.
    """
    root = Path(project_root).resolve()
    raw_output = Path(output_root).absolute()
    _reject_existing_reparse_components(raw_output)
    output = raw_output.resolve()
    if not isinstance(parameters, R03RunParameters):
        raise TypeError("parameters must be R03RunParameters")
    if parameters.worker_count != 1 and (
        selection_manifest is not None or observer is not None
    ):
        raise ValueError("capacity measurement requires worker_count == 1")
    if parameters.max_records_per_airport is not None:
        raise ValueError(
            "max_records_per_airport prefix sampling is retired; "
            "use a verified selection manifest"
        )
    if publication_mode not in {"project_record", "external_evidence_only"}:
        raise ValueError("publication_mode is invalid")
    if selection_manifest is None and publication_mode == "external_evidence_only":
        raise ValueError("external_evidence_only requires a capacity selection manifest")
    if selection_manifest is not None and publication_mode != "external_evidence_only":
        raise ValueError("capacity selection builds must use external_evidence_only")
    output.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_escape(output, output)
    registry = R03SourceRegistry.from_lineage(root)
    selection_bytes: bytes | None = None
    selected_ordinals: Mapping[str, tuple[int, ...]] | None = None
    if selection_manifest is None:
        selected_specs = registry.sources
    else:
        selected_specs, selected_ordinals, selection_bytes, _ = _verified_selection_inputs(
            registry, selection_manifest
        )
    airport_roots = _load_airport_roots(root)
    sources = tuple(verify_source(spec, airport_roots) for spec in selected_specs)
    identity = R03RunIdentity.from_inputs(
        root, sources, parameters, selection_manifest=selection_manifest
    )
    final_root = _assert_no_reparse_escape(output, output / f"r03-{identity.dataset_id}")
    staging = _assert_no_reparse_escape(output, output / f".r03-build-{identity.build_id}.tmp")
    _ensure_output_volume(output, staging)
    _ensure_output_volume(output, final_root)
    full_population = identity.population_kind == "full_population"
    preflight = (
        _preflight_capacity(sources, parameters, output)
        if full_population
        else None
    )
    if preflight is not None and (
        not preflight["disk_capacity"] or not preflight["memory_capacity"]
    ):
        return R03RunResult(
            identity, "BLOCKED_NEEDS_HUMAN_DECISION", output, final_root, None,
            None, full_population, False, False, preflight,
        )
    if preflight_only:
        if not full_population:
            raise ValueError("capacity selection does not use R03 full-registry preflight")
        return R03RunResult(
            identity, "PREFLIGHTED", output, final_root, None, None,
            full_population, False, False, preflight,
        )

    attempt_id = uuid.uuid4().hex
    lock = _acquire_lock(output, identity, attempt_id)
    try:
        existing_verified: dict[str, object] | None = None
        existing_identity: R03RunIdentity | None = None
        if final_root.exists():
            existing_verified = verify_canonical_manifest(
                final_root / "canonical_point_manifest.parquet", root
            )
            if existing_verified["logical_content_root"] is None:
                raise ValueError("pre-existing finalized root has no logical content root")
            existing_identity = R03RunIdentity.from_dict(
                dict(existing_verified["identity"])
            )
            # An exact prior build is already the immutable publication.  A
            # different build must still be realized in contained staging so
            # its logical content can be compared before any pointer change.
            if existing_identity.build_id == identity.build_id:
                if publication_mode == "project_record":
                    _atomic_json(
                        output / "current.json",
                        _pointer_record(root, final_root, existing_verified),
                    )
                return R03RunResult(
                    existing_identity,
                    "PUBLISHED" if publication_mode == "project_record" else "FINALIZED",
                    output, final_root,
                    final_root / "canonical_point_manifest.parquet",
                    str(existing_verified["logical_content_root"]),
                    existing_identity.population_kind == "full_population",
                    False, False, preflight,
                )
        if staging.exists() and not resume:
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        if selection_bytes is not None:
            selection_path = staging / "capacity_selection_manifest.json"
            if selection_path.exists() and selection_path.read_bytes() != selection_bytes:
                raise ValueError("staged capacity selection manifest mismatch")
            _atomic_bytes(selection_path, selection_bytes)
        checkpoint_path = staging / "checkpoint.json"
        if resume and checkpoint_path.is_file():
            checkpoint = RunCheckpoint.load(checkpoint_path)
            reused = _checkpoint_entry_set(checkpoint, staging, identity, parameters.shard_count)
            # A resumed attempt re-enumerates deterministically, then continues
            # from the only reusable boundary: checkpoint-named sealed shards.
            checkpoint = RunCheckpoint(
                _CHECKPOINT_FORMAT, attempt_id, identity,
                3 + len(reused), "SHARDING", None, "",
                tuple(checkpoint.sealed_shards),
            )
            checkpoint = _write_checkpoint(checkpoint_path, checkpoint)
            resumed_checkpoint = True
        else:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            checkpoint = RunCheckpoint(
                _CHECKPOINT_FORMAT, attempt_id, identity, 0, "NEW", None, "", tuple()
            )
            checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("PREFLIGHTED"))
            reused = {}
            checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("ENUMERATED"))
            resumed_checkpoint = False
        runs, bad_paths, results = _build_shard_runs(
            sources, root, staging, parameters, selected_ordinals
        )
        if not resumed_checkpoint:
            checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("SHARDING"))
        checkpoint, shards = _seal_missing_shards(
            staging, identity, runs, bad_paths, parameters, checkpoint, checkpoint_path,
            reused, observer,
        )
        checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("MERGING"))
        final_partitions = merge_final_partitions(
            [shards[index] for index in sorted(shards)], staging,
            parameters.max_rows_per_run, parameters.final_sort_version,
        )
        bad_path = _merge_bad_rows([shards[index] for index in sorted(shards)], staging, parameters.max_rows_per_run)
        if observer is not None:
            observer.sample("merge_complete", staging)
        final_paths = _ensure_empty_final_roles(
            staging, (*final_partitions, FinalPartition(
                artifact_role="bad_rows", airport_id="", partition_year=0,
                partition_month=0, path=bad_path,
                relative_path=bad_path.relative_to(staging).as_posix(),
                row_count=0, schema_fingerprint=_schema_fingerprint(_bad_rows_schema()),
                sha256=_sha256_file(bad_path), sort_contract=_SORT_CONTRACTS["bad_rows"],
            ))
        )
        enumeration_rows = enumeration_manifest(
            registry, _bind_source_results(registry, results), selected_ordinals
        )
        enumeration_path = staging / "source_enumeration.parquet"
        _write_parquet_atomic(enumeration_path, pyarrow.Table.from_pylist(enumeration_rows, schema=_enumeration_schema()))
        # `bad_path` is already included in `final_paths`; this assert catches an accidental role omission.
        if not any(role == "bad_rows" and path == bad_path for role, path in final_paths):
            raise RuntimeError("bad_rows was not included in final artifacts")
        preliminary = [
            {"artifact_role": role, "relative_path": path.relative_to(staging).as_posix()}
            for role, path in [("source_enumeration", enumeration_path), *final_paths]
        ]
        logical_root = _logical_content_root(preliminary, staging)
        parent_hashes = {
            "raw_manifest": registry.raw_manifest_sha256,
            "archive_comparison": registry.archive_comparison_sha256,
            "r02_run_manifest": registry.r02_run_manifest_sha256,
            "r02_decision": registry.r02_decision_sha256,
            "source_sequence": registry.source_sequence_manifest_sha256,
            "source_enumeration": _sha256_file(enumeration_path),
        }
        if selection_bytes is not None:
            parent_hashes["capacity_selection"] = hashlib.sha256(
                selection_bytes
            ).hexdigest()
        manifest_rows = [
            _final_artifact_record(staging, path, role, identity, logical_root, parent_hashes, root)
            for role, path in [("source_enumeration", enumeration_path), *final_paths]
        ]
        manifest_path = staging / "canonical_point_manifest.parquet"
        _write_parquet_atomic(manifest_path, pyarrow.Table.from_pylist(manifest_rows, schema=_manifest_schema()))
        checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("VERIFYING"))
        verified_staged = verify_canonical_manifest(manifest_path, root)
        if verified_staged["logical_content_root"] != logical_root:
            raise ValueError("staging logical content root changed during verification")
        checkpoint = _write_checkpoint(checkpoint_path, checkpoint.next("FINALIZED"))
        _discard_prepublication_intermediates(staging)
        if existing_verified is not None and existing_identity is not None:
            if verified_staged["logical_content_root"] != existing_verified["logical_content_root"]:
                raise ValueError(
                    "immutable dataset root has different logical content for the requested build"
                )
            # The content is genuinely equivalent, so preserve the original
            # immutable root and publish its authenticated identity.
            shutil.rmtree(staging)
            if publication_mode == "project_record":
                _atomic_json(
                    output / "current.json", _pointer_record(root, final_root, existing_verified)
                )
            return R03RunResult(
                existing_identity,
                "PUBLISHED" if publication_mode == "project_record" else "FINALIZED",
                output, final_root,
                final_root / "canonical_point_manifest.parquet",
                str(existing_verified["logical_content_root"]),
                existing_identity.population_kind == "full_population",
                False, False, preflight,
            )
        _termination_seam("dataset-root-promotion-pre")
        _termination_seam("dataset-root-promotion")
        os.replace(staging, final_root)
        _termination_seam("dataset-root-promotion-post")
        _sync_directory(output)
        if observer is not None:
            observer.sample("final_promotion", final_root)
        verified = verify_canonical_manifest(final_root / "canonical_point_manifest.parquet", root)
        if publication_mode == "project_record":
            _atomic_json(output / "current.json", _pointer_record(root, final_root, verified))
        published_identity = R03RunIdentity.from_dict(dict(verified["identity"]))
        return R03RunResult(
            published_identity,
            "PUBLISHED" if publication_mode == "project_record" else "FINALIZED",
            output, final_root,
            final_root / "canonical_point_manifest.parquet", logical_root,
            published_identity.population_kind == "full_population",
            False, False, preflight,
        )
    except Exception as exc:
        if 'checkpoint_path' in locals() and checkpoint_path.exists():
            try:
                checkpoint = RunCheckpoint.load(checkpoint_path)
                _write_checkpoint(checkpoint_path, checkpoint.next("FAILED", terminal_error=type(exc).__name__))
            except Exception:
                pass
        raise
    finally:
        _release_lock(lock)


_TASK6_EVIDENCE_FORMAT = "r03_run_evidence_v1"
_TASK6_FORBIDDEN_ARTIFACTS = frozenset({
    "trajectory.parquet", "scene.parquet", "split.json", "checkpoint.pt",
})


def _task6_scope_guard(output_root: Path) -> bool:
    """Keep this publication boundary restricted to canonical point ledgers."""
    output = Path(output_root)
    for path in output.rglob("*"):
        if path.name.casefold() in _TASK6_FORBIDDEN_ARTIFACTS:
            raise ValueError(f"out-of-scope R04/model artifact found: {path}")
    return True


def _task6_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _task6_write_contained_atomic(path.parent, path, buffer.getvalue().encode("utf-8"))


def _task6_write_contained_atomic(root: Path, path: Path, payload: bytes) -> None:
    """Replace one contained evidence file without following a static link."""
    destination = _assert_no_reparse_escape(root, path)
    temporary = _task6_write_temp_bytes(root, destination, payload)
    try:
        _assert_no_reparse_escape(root, destination)
        os.replace(temporary, destination)
        if _task6_read_pinned_bytes(root, destination) != payload:
            raise ValueError("run-local R03 evidence fixity mismatch")
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _task6_stats(final_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Produce small, deterministic census tables while streaming all ledgers."""
    totals = {
        "parsed_rows": 0,
        "unique_reports": 0,
        "report_membership": 0,
        "duplicate_conflicts": 0,
        "bad_rows": 0,
        "enumerated_source_records": 0,
    }
    monthly: dict[tuple[str, int, int], dict[str, object]] = {}

    def row_for(airport: object, year: object, month: object) -> dict[str, object]:
        if not isinstance(airport, str) or isinstance(year, bool) or not isinstance(year, int) or isinstance(month, bool) or not isinstance(month, int):
            raise ValueError("final ledger partition columns are invalid")
        key = (airport, year, month)
        return monthly.setdefault(
            key,
            {
                "airport_id": airport,
                "partition_year": year,
                "partition_month": month,
                "parsed_rows": 0,
                "unique_reports": 0,
                "report_membership": 0,
                "duplicate_conflicts": 0,
                "bad_rows": 0,
            },
        )

    for role, total_name, columns in (
        ("parsed_rows", "parsed_rows", ("airport_id", "partition_year", "partition_month")),
        ("unique_reports", "unique_reports", ("airport_id", "partition_year", "partition_month")),
        ("report_membership", "report_membership", ("airport_id", "partition_year", "partition_month")),
        ("duplicate_conflicts", "duplicate_conflicts", ("airport_id", "partition_year", "partition_month")),
        ("bad_rows", "bad_rows", ("airport_id",)),
    ):
        for row in stream_final_artifact_rows(final_root, role, columns):
            totals[total_name] += 1
            if role == "bad_rows":
                row_for(row["airport_id"], 0, 0)[total_name] = int(
                    row_for(row["airport_id"], 0, 0)[total_name]
                ) + 1
            else:
                summary = row_for(row["airport_id"], row["partition_year"], row["partition_month"])
                summary[total_name] = int(summary[total_name]) + 1
    for row in stream_final_artifact_rows(
        final_root, "source_enumeration", ("logical_record_count",),
    ):
        count = row["logical_record_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("source enumeration count is invalid")
        totals["enumerated_source_records"] += count
    canonical = [
        {"metric": metric, "value": value}
        for metric, value in sorted(totals.items())
    ]
    airport_month = [
        monthly[key] for key in sorted(monthly)
    ]
    return canonical, airport_month


def _task6_role_summary(
    role: str, root: Path, paths: Sequence[Path], *, parquet: bool,
) -> dict[str, object]:
    """Reduce a sorted artifact stream to one bounded, domain-separated root."""
    digest = hashlib.sha256()
    digest.update(canonical_json({"schema_version": "r03_role_root_v1", "role": role}))
    file_count = 0
    row_count = 0
    byte_count = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        contained = _assert_no_reparse_escape(root, path)
        if not contained.is_file():
            raise ValueError(f"required R03 evidence artifact is missing: {contained}")
        relative = contained.relative_to(root).as_posix()
        size = contained.stat().st_size
        rows = pq.ParquetFile(contained).metadata.num_rows if parquet else 0
        record = {
            "relative_path": relative,
            "row_count": rows,
            "byte_count": size,
            "sha256": _sha256_file(contained),
        }
        digest.update(b"\n")
        digest.update(canonical_json(record))
        file_count += 1
        row_count += rows
        byte_count += size
    return {
        "file_count": file_count,
        "row_count": row_count,
        "byte_count": byte_count,
        "root_sha256": digest.hexdigest(),
    }


def _task6_artifact_hashes(
    output_root: Path, final_root: Path | None,
) -> dict[str, dict[str, object]]:
    """Return fixed-cardinality role roots without exposing partition hashes."""
    output = Path(output_root).resolve()
    final = Path(final_root).resolve() if final_root is not None else None
    role_paths: dict[str, tuple[Path, Sequence[Path], bool]] = {
        "canonical_stats": (output, (output / "canonical_stats.csv",), False),
        "airport_month_stats": (output, (output / "airport_month_stats.csv",), False),
        "report": (output, (output / "REPORT_R03_CANONICAL_POINTS.md",), False),
        "current_pointer": (
            output, (output / "current.json",) if (output / "current.json").is_file() else (), False,
        ),
    }
    for role in _MANIFEST_ROLES:
        role_paths[role] = (
            final or output,
            _artifact_paths(final, role) if final is not None else (),
            True,
        )
    role_paths["canonical_manifest"] = (
        final or output,
        (final / "canonical_point_manifest.parquet",) if final is not None else (),
        True,
    )
    if final is not None:
        for role in (*_MANIFEST_ROLES, "canonical_manifest"):
            if not role_paths[role][1]:
                raise ValueError(f"required R03 artifact role is missing: {role}")
    return {
        role: _task6_role_summary(role, root, paths, parquet=parquet)
        for role, (root, paths, parquet) in sorted(role_paths.items())
    }


def _task6_report(
    result: R03RunResult, decision: R03Decision, canonical: Sequence[Mapping[str, object]],
) -> str:
    observed = "\n".join(
        f"- `{row['metric']}`: {row['value']}" for row in canonical
    ) or "- No canonical corpus was published."
    if result.state == "BLOCKED_NEEDS_HUMAN_DECISION":
        preflight = dict(result.preflight or {})
        capacity = (
            "This blocked preflight records historical observations only: "
            f"available disk `{preflight.get('available_disk_bytes')}` bytes and "
            f"process RSS `{preflight.get('rss_bytes')}` bytes. These volatile "
            "values cannot publish or authorize downstream work."
        )
    elif result.preflight is not None:
        preflight = _task6_deterministic_capacity_record(dict(result.preflight or {}))
        capacity = (
            "This non-blocked evidence record retains deterministic estimates only: "
            f"disk `{preflight['estimated_disk_bytes']}` bytes and working set "
            f"`{preflight['estimated_working_set_bytes']}` bytes. Volatile "
            "build-time disk/RSS observations are omitted; artifact verification "
            "evaluates capacity again from verifier-current measurements."
        )
    else:
        capacity = (
            "This capacity-sample record has no full-registry R03 preflight. "
            "It is evidence-only and cannot authorize downstream work."
        )
    return (
        "# R03 Canonical Points Report\n\n"
        "## Run identity\n\n"
        f"- Dataset ID: `{result.identity.dataset_id}`\n"
        f"- Build ID: `{result.identity.build_id}`\n"
        f"- Run state: `{result.state}`\n"
        f"- Full population: `{str(result.is_full_population).lower()}`\n\n"
        "## Observed census\n\n"
        f"{observed}\n\n"
        "## Expectations and limits\n\n"
        "The observed census is an audit result, not a substituted expectation. "
        "A bounded or preflight run is evidence-only and cannot authorize R04.\n\n"
        "## Capacity evidence\n\n"
        f"{capacity}\n\n"
        "## Decision\n\n"
        f"- Decision: `{decision.decision}`\n"
        f"- R04 authorized: `{str(decision.r04_authorized).lower()}`\n"
        "- Model training authorized: `false`\n"
    )


def _task6_checks(
    result: R03RunResult, output_root: Path,
) -> tuple[dict[str, bool], dict[str, object], Path | None, list[dict[str, object]], list[dict[str, object]]]:
    """Derive every decision gate from independent current observations."""
    output = Path(output_root).resolve()
    if result.output_root.resolve() != output:
        raise ValueError("run evidence output root does not match the R03 result")
    raw_preflight = dict(result.preflight or {})
    capacity = bool(
        raw_preflight.get("disk_capacity") is True
        and raw_preflight.get("memory_capacity") is True
    )
    preflight = (
        raw_preflight
        if result.state == "BLOCKED_NEEDS_HUMAN_DECISION" or result.preflight is None
        else _task6_deterministic_capacity_record(raw_preflight)
    )
    full_population = (
        result.is_full_population
        and result.identity.run_parameters.max_records_per_airport is None
    )
    try:
        scope_guard = _task6_scope_guard(output)
    except ValueError:
        scope_guard = False
    final_root: Path | None = None
    canonical_ok = False
    canonical: list[dict[str, object]] = []
    airport_month: list[dict[str, object]] = []
    zero_bad_rows_accounted = False
    artifact_state = result.state in {"PUBLISHED", "FINALIZED"}
    if artifact_state and result.manifest_path is not None:
        try:
            manifest_rows = _iter_mappings(result.manifest_path, _manifest_schema())
            first = next(manifest_rows)
            project_value = first.get("project_root")
            if not isinstance(project_value, str) or not project_value:
                raise ValueError("canonical manifest lacks a project root")
            if result.state == "PUBLISHED":
                verified = verify_current_pointer(output, Path(project_value))
            else:
                if (output / "current.json").exists():
                    raise ValueError("external evidence cannot have a current pointer")
                if result.identity.population_kind != "capacity_sample":
                    raise ValueError("only a capacity sample may be externally finalized")
                verified = verify_canonical_manifest(
                    result.manifest_path, Path(project_value)
                )
            if verified["dataset_id"] != result.identity.dataset_id:
                raise ValueError("published dataset identity does not match the run result")
            final_root = Path(str(verified["manifest_path"])).parent
            canonical, airport_month = _task6_stats(final_root)
            canonical_ok = True
            zero_bad_rows_accounted = (
                any(row["metric"] == "bad_rows" and row["value"] == 0 for row in canonical)
                and (final_root / "source_enumeration.parquet").is_file()
            ) or any(
                row["metric"] == "bad_rows" and row["value"] != 0 for row in canonical
            )
        except (OSError, StopIteration, ValueError):
            canonical_ok = False
            final_root = None
            canonical = []
            airport_month = []
    checks = {
        "canonical_verification": canonical_ok,
        "artifact_hashes": False,
        "statistics": not artifact_state or canonical_ok,
        "scope_guard": scope_guard,
        "capacity": capacity,
        "full_population": full_population,
        "zero_bad_rows_accounted": zero_bad_rows_accounted,
    }
    return checks, preflight, final_root, canonical, airport_month


def write_r03_run_evidence(result: R03RunResult, output_root: Path) -> R03Decision:
    """Write immutable-run evidence and a fail-closed R03 decision record."""
    if not isinstance(result, R03RunResult):
        raise TypeError("result must be an R03RunResult")
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(output, result.identity, f"evidence-{uuid.uuid4().hex}")
    try:
        return _task6_write_r03_run_evidence_locked(result, output)
    finally:
        _release_lock(lock)


def _task6_write_r03_run_evidence_locked(
    result: R03RunResult, output: Path,
) -> R03Decision:
    """Write one four-file evidence record while the output writer lock is held."""
    checks, preflight, final_root, canonical, airport_month = _task6_checks(result, output)
    _task6_write_csv(output / "canonical_stats.csv", ("metric", "value"), canonical)
    _task6_write_csv(
        output / "airport_month_stats.csv",
        (
            "airport_id", "partition_year", "partition_month", "parsed_rows",
            "unique_reports", "report_membership", "duplicate_conflicts", "bad_rows",
        ),
        airport_month,
    )
    blocked = result.state == "BLOCKED_NEEDS_HUMAN_DECISION"
    # A missing report or failed hash computation aborts before decision.json
    # exists, so the report can state the same final gate decision.
    checks = dict(checks)
    checks["artifact_hashes"] = True
    decision = R03Decision.from_checks(
        result.identity.dataset_id, checks,
        is_full_population=result.is_full_population, blocked=blocked,
    )
    report = output / "REPORT_R03_CANONICAL_POINTS.md"
    _task6_write_contained_atomic(
        output, report, _task6_report(result, decision, canonical).encode("utf-8"),
    )
    hashes = _task6_artifact_hashes(output, final_root)
    document = {
        "schema_version": _TASK6_EVIDENCE_FORMAT,
        "decision": decision.decision,
        "dataset_id": decision.dataset_id,
        "checks": dict(decision.checks),
        "artifact_hashes": hashes,
        "is_full_population": decision.is_full_population,
        "r04_authorized": decision.r04_authorized,
        "model_training_authorized": False,
        "run_state": result.state,
        "preflight": preflight,
    }
    _task6_write_contained_atomic(
        output,
        output / "decision.json",
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return decision


def _task6_read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    rows = _task6_read_csv_unordered(path, fields)
    if rows != sorted(rows, key=lambda row: tuple(row[field] for field in fields)):
        raise ValueError(f"CSV evidence is not sorted: {path}")
    return rows


def _task6_read_airport_month_csv(path: Path) -> list[dict[str, object]]:
    """Parse and validate airport/month statistics in numeric canonical order."""
    fields = (
        "airport_id", "partition_year", "partition_month", "parsed_rows",
        "unique_reports", "report_membership", "duplicate_conflicts", "bad_rows",
    )
    raw_rows = _task6_read_csv_unordered(path, fields)
    rows: list[dict[str, object]] = []
    numeric_fields = fields[1:]
    for raw in raw_rows:
        airport = raw["airport_id"]
        if not airport:
            raise ValueError("airport-month statistics has an empty airport")
        row: dict[str, object] = {"airport_id": airport}
        for field in numeric_fields:
            text = raw[field]
            if not text.isascii() or not text.isdecimal():
                raise ValueError("airport-month statistics has a non-integer value")
            value = int(text)
            if value < 0 or str(value) != text:
                raise ValueError("airport-month statistics has a non-canonical integer")
            row[field] = value
        rows.append(row)
    if rows != sorted(
        rows,
        key=lambda row: (
            str(row["airport_id"]), int(row["partition_year"]), int(row["partition_month"]),
        ),
    ):
        raise ValueError("airport-month statistics is not in canonical numeric order")
    return rows


def _task6_read_csv_unordered(
    path: Path, fields: Sequence[str],
) -> list[dict[str, str]]:
    """Read an exact CSV schema without imposing an inappropriate text order."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(fields):
                raise ValueError(f"unexpected CSV schema: {path}")
            return list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read CSV evidence: {path}") from exc


def _task6_assert_document_shape(document: Mapping[str, object]) -> None:
    required = {
        "schema_version", "decision", "dataset_id", "checks", "artifact_hashes",
        "is_full_population", "r04_authorized", "model_training_authorized",
        "run_state", "preflight",
    }
    if set(document) != required or document.get("schema_version") != _TASK6_EVIDENCE_FORMAT:
        raise ValueError("R03 decision evidence shape mismatch")
    if document.get("model_training_authorized") is not False:
        raise ValueError("R03 decision has an invalid training authorization claim")
    if not isinstance(document.get("dataset_id"), str) or len(str(document["dataset_id"])) != 64:
        raise ValueError("R03 decision dataset identifier is invalid")
    if not isinstance(document.get("checks"), dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in dict(document["checks"]).items()
    ):
        raise ValueError("R03 decision checks are invalid")
    artifact_hashes = document.get("artifact_hashes")
    expected_roles = {
        "airport_month_stats", "bad_rows", "canonical_manifest",
        "canonical_stats", "current_pointer", "duplicate_conflicts",
        "parsed_rows", "report", "report_membership", "source_enumeration",
        "unique_reports",
    }
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_roles:
        raise ValueError("R03 artifact hash evidence is invalid")
    for role, raw_summary in artifact_hashes.items():
        if not isinstance(role, str) or not isinstance(raw_summary, dict):
            raise ValueError("R03 artifact hash evidence is invalid")
        summary = dict(raw_summary)
        if set(summary) != {"file_count", "row_count", "byte_count", "root_sha256"}:
            raise ValueError("R03 artifact hash evidence is invalid")
        for field in ("file_count", "row_count", "byte_count"):
            value = summary[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("R03 artifact hash evidence is invalid")
        root = summary["root_sha256"]
        if (
            not isinstance(root, str) or len(root) != 64
            or any(character not in "0123456789abcdef" for character in root)
        ):
            raise ValueError("R03 artifact hash evidence is invalid")
    if not isinstance(document.get("is_full_population"), bool) or not isinstance(document.get("r04_authorized"), bool):
        raise ValueError("R03 decision authorization fields are invalid")
    if not isinstance(document.get("run_state"), str) or not isinstance(document.get("preflight"), dict):
        raise ValueError("R03 decision run evidence is invalid")


_TASK6_DETERMINISTIC_CAPACITY_FIELDS = (
    "enumerated_source_records", "record_batch_rows", "max_group_rows_in_memory",
    "estimated_batch_bytes", "estimated_working_set_bytes", "estimated_disk_bytes",
    "memory_limit_bytes",
)


def _task6_deterministic_capacity_record(
    capacity: Mapping[str, object],
) -> dict[str, object]:
    """Project capacity evidence onto values reproducible from sources/parameters."""
    record: dict[str, object] = {"schema_version": "r03_deterministic_capacity_v1"}
    for field in _TASK6_DETERMINISTIC_CAPACITY_FIELDS:
        value = capacity.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"R03 preflight evidence has an invalid {field}")
        record[field] = value
    return record


def _task6_validate_preflight_evidence(
    claimed: Mapping[str, object], recomputed: Mapping[str, object],
) -> dict[str, object]:
    """Require exact deterministic evidence; volatile values are verifier-local."""
    expected = _task6_deterministic_capacity_record(recomputed)
    if dict(claimed) != expected:
        raise ValueError("R03 preflight evidence is not exactly reproducible")
    return expected


def verify_r03_outputs(project_root: Path, output_root: Path) -> dict[str, object]:
    """Independently re-check R03 data and evidence without materializing ledgers."""
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    _task6_scope_guard(output)
    document = _read_json_object(output / "decision.json")
    _task6_assert_document_shape(document)
    run_state = document["run_state"]
    if run_state == "PUBLISHED":
        verified = verify_current_pointer(output, project)
    elif run_state == "FINALIZED":
        if (output / "current.json").exists():
            raise ValueError("external evidence cannot have a current pointer")
        manifests = sorted(output.glob("r03-*/canonical_point_manifest.parquet"))
        if len(manifests) != 1:
            raise ValueError("external evidence must contain exactly one finalized manifest")
        verified = verify_canonical_manifest(manifests[0], project)
        external_identity = R03RunIdentity.from_dict(dict(verified["identity"]))
        if external_identity.population_kind != "capacity_sample":
            raise ValueError("only a capacity sample may be externally finalized")
    else:
        raise ValueError("R03 decision run state is not independently verifiable")
    final_root = Path(str(verified["manifest_path"])).parent
    canonical, airport_month = _task6_stats(final_root)
    canonical_rows = _task6_read_csv(output / "canonical_stats.csv", ("metric", "value"))
    expected_canonical = [
        {"metric": str(row["metric"]), "value": str(row["value"])}
        for row in canonical
    ]
    if canonical_rows != expected_canonical:
        raise ValueError("canonical statistics evidence disagrees with the ledgers")
    airport_rows = _task6_read_airport_month_csv(output / "airport_month_stats.csv")
    if airport_rows != airport_month:
        raise ValueError("airport-month statistics evidence disagrees with the ledgers")
    identity = R03RunIdentity.from_dict(dict(verified["identity"]))
    registry = R03SourceRegistry.from_lineage(project)
    if identity.population_kind == "capacity_sample":
        if document["preflight"] != {}:
            raise ValueError("capacity-sample R03 evidence cannot claim preflight")
        capacity_record: Mapping[str, object] | None = None
        preflight = {}
    else:
        roots = _load_airport_roots(project)
        sources = tuple(verify_source(spec, roots) for spec in registry.sources)
        capacity_record = _preflight_capacity(sources, identity.run_parameters, output)
        preflight = _task6_validate_preflight_evidence(
            dict(document["preflight"]), capacity_record,
        )
    bad_rows = next(row["value"] for row in canonical if row["metric"] == "bad_rows")
    checks = {
        "canonical_verification": True,
        "artifact_hashes": True,
        "statistics": True,
        "scope_guard": True,
        "capacity": bool(
            capacity_record is not None
            and capacity_record["disk_capacity"]
            and capacity_record["memory_capacity"]
        ),
        "full_population": identity.population_kind == "full_population",
        "zero_bad_rows_accounted": bool(
            bad_rows != 0 or (final_root / "source_enumeration.parquet").is_file()
        ),
    }
    expected_hashes = _task6_artifact_hashes(output, final_root)
    if dict(document["artifact_hashes"]) != expected_hashes:
        raise ValueError("R03 evidence artifact hashes do not match current artifacts")
    if not (output / "REPORT_R03_CANONICAL_POINTS.md").is_file():
        raise ValueError("R03 report is missing")
    expected = R03Decision.from_checks(
        str(verified["dataset_id"]), checks,
        is_full_population=identity.population_kind == "full_population",
        blocked=False,
    )
    expected_result = R03RunResult(
        identity=identity,
        state=str(run_state),
        output_root=output,
        final_root=final_root,
        manifest_path=Path(str(verified["manifest_path"])),
        logical_content_root=str(verified["logical_content_root"]),
        is_full_population=expected.is_full_population,
        r04_authorized=False,
        model_training_authorized=False,
        preflight=(
            None if identity.population_kind == "capacity_sample" else preflight
        ),
    )
    expected_report = _task6_report(expected_result, expected, canonical).encode("utf-8")
    if _task6_read_pinned_bytes(output, output / "REPORT_R03_CANONICAL_POINTS.md") != expected_report:
        raise ValueError("R03 report does not match independently regenerated evidence")
    if dict(document["checks"]) != checks:
        raise ValueError("R03 decision checks are not independently reproducible")
    if (
        document["decision"] != expected.decision
        or document["r04_authorized"] != expected.r04_authorized
        or document["model_training_authorized"] is not False
    ):
        raise ValueError("R03 decision has an invalid authorization claim")
    if document["dataset_id"] != verified["dataset_id"]:
        raise ValueError("R03 decision dataset identifier does not match the canonical corpus")
    if document["is_full_population"] is not expected.is_full_population:
        raise ValueError("R03 decision population claim does not match the canonical corpus")
    return {
        "decision": expected.decision,
        "dataset_id": str(verified["dataset_id"]),
        "checks": checks,
        "artifact_hashes": expected_hashes,
        "publication_hashes": {
            name: _sha256_file(output / name)
            for name in (
                "canonical_stats.csv", "airport_month_stats.csv",
                "decision.json", "REPORT_R03_CANONICAL_POINTS.md",
            )
        },
        "is_full_population": expected.is_full_population,
        "r04_authorized": expected.r04_authorized,
        "model_training_authorized": False,
    }


_TASK6_PUBLICATION_TARGETS = {
    "canonical_stats.csv": Path("outputs/r03/canonical_stats.csv"),
    "airport_month_stats.csv": Path("outputs/r03/airport_month_stats.csv"),
    "decision.json": Path("outputs/r03/decision.json"),
    "REPORT_R03_CANONICAL_POINTS.md": Path("reports/REPORT_R03_CANONICAL_POINTS.md"),
}


def _task6_read_pinned_bytes(root: Path, path: Path) -> bytes:
    """Read one contained regular file through a no-follow descriptor."""
    safe = _assert_no_reparse_escape(root, path)
    try:
        before = os.lstat(safe)
    except OSError as exc:
        raise ValueError(f"publication source cannot be inspected: {safe}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"publication source is not a regular file: {safe}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(safe, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"publication source is not a regular file: {safe}")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                payload = handle.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        after = os.lstat(safe)
    except OSError as exc:
        raise ValueError(f"publication source changed while being pinned: {safe}") from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field, None) != getattr(after, field, None) for field in identity_fields):
        raise ValueError(f"publication source changed while being pinned: {safe}")
    return payload


def _task6_validate_pinned_publication(files: Mapping[str, bytes]) -> None:
    """Parse all pinned repository records before any project destination write."""
    csv_fields = {
        "canonical_stats.csv": ("metric", "value"),
        "airport_month_stats.csv": (
            "airport_id", "partition_year", "partition_month", "parsed_rows",
            "unique_reports", "report_membership", "duplicate_conflicts", "bad_rows",
        ),
    }
    for name, fields in csv_fields.items():
        try:
            reader = csv.DictReader(io.StringIO(files[name].decode("utf-8"), newline=""))
            if reader.fieldnames != list(fields):
                raise ValueError
            list(reader)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"verified publication CSV is not canonical: {name}") from exc
    try:
        decision = json.loads(files["decision.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("verified publication decision is invalid") from exc
    if not isinstance(decision, dict):
        raise ValueError("verified publication decision is invalid")
    _task6_assert_document_shape(decision)
    if (
        decision.get("decision") != "PROCEED_TO_R04"
        or decision.get("r04_authorized") is not True
        or decision.get("model_training_authorized") is not False
    ):
        raise ValueError("verified publication decision is not authorized")
    try:
        report = files["REPORT_R03_CANONICAL_POINTS.md"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("verified publication report is not UTF-8") from exc
    if not report.startswith("# "):
        raise ValueError("verified publication report is not canonical Markdown")


def _task6_pin_publication_sources(
    output: Path, verified: Mapping[str, object],
) -> dict[str, bytes]:
    expected = verified.get("publication_hashes")
    if not isinstance(expected, dict) or set(expected) != set(_TASK6_PUBLICATION_TARGETS):
        raise ValueError("verified publication hashes are incomplete")
    pinned: dict[str, bytes] = {}
    for name in _TASK6_PUBLICATION_TARGETS:
        payload = _task6_read_pinned_bytes(output, output / name)
        digest = hashlib.sha256(payload).hexdigest()
        if expected.get(name) != digest:
            raise ValueError(f"publication source changed after verification: {name}")
        pinned[name] = payload
    _task6_validate_pinned_publication(pinned)
    return pinned


def _task6_write_temp_bytes(root: Path, destination: Path, payload: bytes) -> Path:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    safe = _assert_no_reparse_escape(root, temporary)
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(safe, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if hashlib.sha256(_task6_read_pinned_bytes(root, safe)).digest() != hashlib.sha256(payload).digest():
        safe.unlink(missing_ok=True)
        raise ValueError("publication temporary fixity mismatch")
    return safe


def _task6_publish_pinned_records(project: Path, pinned: Mapping[str, bytes]) -> None:
    """Atomically replace four fixed records and roll back any partial failure."""
    destinations = {
        name: project / relative for name, relative in _TASK6_PUBLICATION_TARGETS.items()
    }
    for destination in destinations.values():
        parent = _assert_no_reparse_escape(project, destination.parent)
        parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_escape(project, parent)
        _assert_no_reparse_escape(project, destination)
    previous = {
        name: (_task6_read_pinned_bytes(project, destination) if destination.exists() else None)
        for name, destination in destinations.items()
    }
    temporaries: dict[str, Path] = {}
    committed: list[str] = []
    try:
        for name, destination in destinations.items():
            temporaries[name] = _task6_write_temp_bytes(project, destination, pinned[name])
        for name, destination in destinations.items():
            _assert_no_reparse_escape(project, destination)
            os.replace(temporaries.pop(name), destination)
            committed.append(name)
        for name, destination in destinations.items():
            if _task6_read_pinned_bytes(project, destination) != pinned[name]:
                raise ValueError("published R03 record fixity mismatch")
        for destination in {path.parent for path in destinations.values()}:
            _sync_directory(destination)
    except Exception:
        for name in reversed(committed):
            destination = destinations[name]
            old = previous[name]
            if old is None:
                destination.unlink(missing_ok=True)
            else:
                rollback = _task6_write_temp_bytes(project, destination, old)
                os.replace(rollback, destination)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def publish_verified_r03_record(project_root: Path, output_root: Path) -> None:
    """Copy only the verified R03 decision record and census summaries into the project."""
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    verified = verify_r03_outputs(project, output)
    if verified["decision"] != "PROCEED_TO_R04" or verified["r04_authorized"] is not True:
        raise ValueError("only a full-population PROCEED_TO_R04 record may be published")
    pinned = _task6_pin_publication_sources(output, verified)
    _task6_publish_pinned_records(project, pinned)
