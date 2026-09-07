"""Strict crash-recoverable terminal journal for the R03 capacity gate.

This module deliberately models only transaction state and direct filesystem
shape.  It is not a trust root for any scientific result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Literal, Mapping


JOURNAL_SCHEMA = "r03_task8a_capacity_terminal_journal_v1"
JOURNAL_NAME = ".capacity-terminal-journal.json"
NEXT_NAME = ".capacity-terminal-journal.next"
FORMAL_EVIDENCE_ORDER = (
    "capacity_selection_manifest.json",
    "capacity_metrics.json",
    "capacity_verifier_result.json",
    "command_environment.json",
    "hashes.json",
)

_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_FAILURE_STAGES = frozenset(
    {
        "rename_attempt",
        "remove_attempt",
        "remove_attempts_root",
        "measure_free_space",
    }
)


class TerminalState(StrEnum):
    METRICS_READY = "METRICS_READY"
    PREPARED = "PREPARED"
    OUTCOME_CLEAN = "OUTCOME_CLEAN"
    OUTCOME_FAILED = "OUTCOME_FAILED"
    SEALING = "SEALING"
    COMMITTED = "COMMITTED"


@dataclass(frozen=True)
class TerminalExpectations:
    capacity_selection_id: str
    selection_root: str
    project_root: str
    external_root: str
    invocation_sha256: str
    capacity_selection_manifest_sha256: str
    capacity_metrics_sha256: str


@dataclass(frozen=True)
class AttemptBinding:
    public_relative_path: str
    tombstone_relative_path: str
    path_identity: tuple[int, int, int]
    volume_identity: int

    def __post_init__(self) -> None:
        _attempt_paths(self.public_relative_path, self.tombstone_relative_path)
        _path_identity(self.path_identity)
        _exact_int("volume_identity", self.volume_identity)


@dataclass(frozen=True)
class CleanupFailure:
    stage: str
    exception_type: str

    def __post_init__(self) -> None:
        _validate_cleanup_failure(self)


@dataclass(frozen=True)
class TerminalJournal:
    schema_version: str
    journal_id: str
    generation: int
    state: TerminalState
    capacity_selection_id: str
    selection_root: str
    project_root: str
    external_root: str
    invocation_sha256: str
    capacity_selection_manifest_sha256: str
    capacity_metrics_sha256: str
    started_utc: str
    attempt_mode: Literal["NONE", "OWNED"]
    attempt_public_relative_path: str | None
    attempt_tombstone_relative_path: str | None
    attempt_path_identity: tuple[int, int, int] | None
    attempt_volume_identity: int | None
    cleanup_status: Literal["PENDING", "CLEAN", "FAILED"]
    retained_attempt_relative_path: str | None
    cleanup_failure: CleanupFailure | None
    free_bytes_after_cleanup: int | None
    decision: str | None
    pilot_capacity_pass: bool | None
    full_capacity_ready: bool | None
    performance_review_required: bool | None
    evidence_order: tuple[str, ...] | None
    evidence_payload_sha256: tuple[tuple[str, str], ...] | None
    ended_utc: str | None
    committed_envelope_sha256: str | None


@dataclass(frozen=True)
class TerminalShape:
    evidence_prefix: tuple[str, ...]
    attempt_kind: Literal["NONE", "PUBLIC", "TOMBSTONE"]
    attempt_relative_path: str | None


_GENERATION = {
    TerminalState.METRICS_READY: 0,
    TerminalState.PREPARED: 1,
    TerminalState.OUTCOME_CLEAN: 2,
    TerminalState.OUTCOME_FAILED: 2,
    TerminalState.SEALING: 3,
    TerminalState.COMMITTED: 4,
}
_TRANSITIONS = frozenset(
    {
        (TerminalState.METRICS_READY, TerminalState.PREPARED),
        (TerminalState.PREPARED, TerminalState.OUTCOME_CLEAN),
        (TerminalState.PREPARED, TerminalState.OUTCOME_FAILED),
        (TerminalState.OUTCOME_CLEAN, TerminalState.SEALING),
        (TerminalState.OUTCOME_FAILED, TerminalState.SEALING),
        (TerminalState.SEALING, TerminalState.COMMITTED),
    }
)
_IDENTITY_FIELDS = (
    "schema_version",
    "journal_id",
    "capacity_selection_id",
    "selection_root",
    "project_root",
    "external_root",
    "invocation_sha256",
    "capacity_selection_manifest_sha256",
    "capacity_metrics_sha256",
    "started_utc",
    "attempt_mode",
    "attempt_public_relative_path",
    "attempt_tombstone_relative_path",
    "attempt_path_identity",
    "attempt_volume_identity",
)


def allowed_terminal_transition(before: TerminalState, after: TerminalState) -> bool:
    return isinstance(before, TerminalState) and isinstance(after, TerminalState) and (
        before, after
    ) in _TRANSITIONS


def _exact_int(name: str, value: object, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _exact_bool_or_none(name: str, value: object) -> bool | None:
    if value is not None and type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean or null")
    return value


def _hash(name: str, value: object) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _timestamp(name: str, value: object) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be UTC")
    return value


def _normalized_root(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    normalized = os.path.normpath(os.path.abspath(value))
    if not os.path.isabs(value) or value != normalized:
        raise ValueError(f"{name} must be a normalized absolute path")
    return value


def _attempt_paths(public: object, tombstone: object) -> tuple[str, str]:
    if not isinstance(public, str) or not isinstance(tombstone, str):
        raise TypeError("owned attempt paths must be strings")
    public_match = re.fullmatch(r"attempts[/\\]([0-9a-f]{32})", public)
    tombstone_match = re.fullmatch(
        r"attempts[/\\]\.owned-cleanup-([0-9a-f]{32})", tombstone
    )
    if (
        public_match is None
        or tombstone_match is None
        or public_match.group(1) != tombstone_match.group(1)
        or ":" in public
        or ":" in tombstone
        or public != "attempts/" + public_match.group(1)
        or tombstone != "attempts/.owned-cleanup-" + tombstone_match.group(1)
    ):
        raise ValueError("attempt paths do not use the frozen relative grammar")
    return public, tombstone


def _path_identity(value: object) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("attempt_path_identity must be a three-integer tuple")
    return tuple(_exact_int("attempt_path_identity", item) for item in value)  # type: ignore[return-value]


def _journal_id_bytes(record: TerminalJournal) -> bytes:
    fields_to_bind = (
        record.schema_version,
        record.capacity_selection_id,
        record.selection_root,
        record.project_root,
        record.external_root,
        record.invocation_sha256,
    )
    payload = bytearray(b"r03_task8a_capacity_terminal_journal_id_v1\x00")
    for item in fields_to_bind:
        encoded = item.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return bytes(payload)


def derive_terminal_journal_id(record: TerminalJournal) -> str:
    """Return the domain-separated identity for immutable invocation fields."""
    return hashlib.sha256(_journal_id_bytes(record)).hexdigest()


def committed_terminal_envelope_sha256(record: TerminalJournal) -> str:
    """Bind the frozen five-file envelope to its terminal cleanup shape.

    The journal is deliberately not part of the immutable evidence envelope,
    but its final generation must still prove which clean/failed shape was
    independently checked before it may be removed.
    """
    if not isinstance(record, TerminalJournal) or record.state is not TerminalState.COMMITTED:
        raise ValueError("only a committed journal has an envelope digest")
    assert record.evidence_payload_sha256 is not None
    payload = {
        "schema_version": "r03_task8a_capacity_terminal_envelope_v1",
        "evidence": list(record.evidence_payload_sha256),
        "cleanup_status": record.cleanup_status,
        "retained_attempt_relative_path": record.retained_attempt_relative_path,
        "attempt_path_identity": list(record.attempt_path_identity)
        if record.cleanup_status == "FAILED" else None,
        "attempt_volume_identity": record.attempt_volume_identity
        if record.cleanup_status == "FAILED" else None,
    }
    return hashlib.sha256(
        b"r03_task8a_capacity_terminal_committed_envelope_v1\x00"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _validate_cleanup_failure(value: object) -> CleanupFailure:
    if not isinstance(value, CleanupFailure):
        raise TypeError("cleanup_failure must be CleanupFailure")
    if value.stage not in _FAILURE_STAGES:
        raise ValueError("cleanup failure stage is not allowed")
    if not isinstance(value.exception_type, str) or re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", value.exception_type
    ) is None:
        raise ValueError("cleanup failure exception type is not sanitized")
    return value


def _validate_record(record: TerminalJournal) -> None:
    if not isinstance(record, TerminalJournal):
        raise TypeError("terminal journal has the wrong type")
    if record.schema_version != JOURNAL_SCHEMA:
        raise ValueError("terminal journal schema is unsupported")
    if not isinstance(record.state, TerminalState):
        raise ValueError("terminal journal state is unknown")
    generation = _exact_int("generation", record.generation)
    if generation != _GENERATION[record.state]:
        raise ValueError("terminal journal generation disagrees with state")
    _hash("capacity_selection_id", record.capacity_selection_id)
    selection_root = _normalized_root("selection_root", record.selection_root)
    project_root = _normalized_root("project_root", record.project_root)
    external_root = _normalized_root("external_root", record.external_root)
    del project_root
    if Path(selection_root) != Path(external_root) / "capacity" / record.capacity_selection_id:
        raise ValueError("selection_root is not the frozen external capacity root")
    _hash("invocation_sha256", record.invocation_sha256)
    _hash("capacity_selection_manifest_sha256", record.capacity_selection_manifest_sha256)
    _hash("capacity_metrics_sha256", record.capacity_metrics_sha256)
    _timestamp("started_utc", record.started_utc)
    _hash("journal_id", record.journal_id)
    if record.journal_id != derive_terminal_journal_id(record):
        raise ValueError("terminal journal identity digest is inconsistent")

    if record.attempt_mode not in {"NONE", "OWNED"}:
        raise ValueError("attempt_mode is invalid")
    if record.attempt_mode == "NONE":
        if any(
            value is not None
            for value in (
                record.attempt_public_relative_path,
                record.attempt_tombstone_relative_path,
                record.attempt_path_identity,
                record.attempt_volume_identity,
            )
        ):
            raise ValueError("NONE attempt mode requires null attempt fields")
    else:
        _attempt_paths(record.attempt_public_relative_path, record.attempt_tombstone_relative_path)
        _path_identity(record.attempt_path_identity)
        _exact_int("attempt_volume_identity", record.attempt_volume_identity)

    if record.cleanup_status not in {"PENDING", "CLEAN", "FAILED"}:
        raise ValueError("cleanup_status is invalid")
    before_outcome = record.state in {TerminalState.METRICS_READY, TerminalState.PREPARED}
    outcome_fields = (
        record.retained_attempt_relative_path,
        record.cleanup_failure,
        record.free_bytes_after_cleanup,
        record.decision,
        record.pilot_capacity_pass,
        record.full_capacity_ready,
        record.performance_review_required,
    )
    if before_outcome:
        if record.cleanup_status != "PENDING" or any(value is not None for value in outcome_fields):
            raise ValueError("pre-outcome journal contains outcome data")
    else:
        if record.cleanup_status not in {"CLEAN", "FAILED"}:
            raise ValueError("outcome journal requires a terminal cleanup status")
        if (
            record.state is TerminalState.OUTCOME_CLEAN
            and record.cleanup_status != "CLEAN"
        ) or (
            record.state is TerminalState.OUTCOME_FAILED
            and record.cleanup_status != "FAILED"
        ):
            raise ValueError("outcome state and cleanup status must be an exact pair")
        _exact_int("free_bytes_after_cleanup", record.free_bytes_after_cleanup)
        for name in (
            "pilot_capacity_pass",
            "full_capacity_ready",
            "performance_review_required",
        ):
            if _exact_bool_or_none(name, getattr(record, name)) is None:
                raise ValueError(f"{name} is required after cleanup outcome")
        if record.cleanup_status == "CLEAN":
            if record.retained_attempt_relative_path is not None or record.cleanup_failure is not None:
                raise ValueError("clean outcome cannot retain attempt failure data")
            if record.decision not in {"CAPACITY_PASS_PILOT", "BLOCKED_NEEDS_HUMAN_DECISION"}:
                raise ValueError("clean outcome decision is invalid")
            if (record.decision == "CAPACITY_PASS_PILOT") != record.pilot_capacity_pass:
                raise ValueError("clean outcome decision disagrees with pilot result")
        else:
            if record.attempt_mode != "OWNED":
                raise ValueError("failed cleanup requires an owned attempt")
            public, tombstone = _attempt_paths(
                record.attempt_public_relative_path, record.attempt_tombstone_relative_path
            )
            if record.retained_attempt_relative_path not in {public, tombstone}:
                raise ValueError("failed cleanup must retain exactly one bound attempt path")
            _validate_cleanup_failure(record.cleanup_failure)
            if (
                record.decision != "BLOCKED_CLEANUP_REQUIRED"
                or record.pilot_capacity_pass is not False
                or record.full_capacity_ready is not False
            ):
                raise ValueError("failed cleanup outcome is not forced blocked")

    sealing = record.state in {TerminalState.SEALING, TerminalState.COMMITTED}
    if not sealing:
        if any(
            value is not None
            for value in (
                record.evidence_order,
                record.evidence_payload_sha256,
                record.ended_utc,
                record.committed_envelope_sha256,
            )
        ):
            raise ValueError("evidence sealing data is premature")
    else:
        if record.evidence_order != FORMAL_EVIDENCE_ORDER:
            raise ValueError("formal evidence order is invalid")
        if (
            not isinstance(record.evidence_payload_sha256, tuple)
            or tuple(name for name, _ in record.evidence_payload_sha256) != FORMAL_EVIDENCE_ORDER
        ):
            raise ValueError("formal evidence digest map is invalid")
        for name, digest in record.evidence_payload_sha256:
            if not isinstance(name, str):
                raise TypeError("evidence digest name must be a string")
            _hash(f"evidence payload {name}", digest)
        _timestamp("ended_utc", record.ended_utc)
        if record.state is TerminalState.SEALING:
            if record.committed_envelope_sha256 is not None:
                raise ValueError("committed digest is premature")
        else:
            _hash("committed_envelope_sha256", record.committed_envelope_sha256)
            if record.committed_envelope_sha256 != committed_terminal_envelope_sha256(record):
                raise ValueError("committed envelope digest is inconsistent")


def _json_record(record: TerminalJournal) -> dict[str, object]:
    result = asdict(record)
    result["state"] = record.state.value
    result["attempt_path_identity"] = (
        list(record.attempt_path_identity) if record.attempt_path_identity is not None else None
    )
    result["evidence_order"] = list(record.evidence_order) if record.evidence_order is not None else None
    result["evidence_payload_sha256"] = (
        dict(record.evidence_payload_sha256)
        if record.evidence_payload_sha256 is not None
        else None
    )
    return result


def canonical_terminal_journal_bytes(record: TerminalJournal) -> bytes:
    _validate_record(record)
    return (
        json.dumps(
            _json_record(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON scalar is forbidden: {value}")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_record(payload: bytes) -> TerminalJournal:
    try:
        text = payload.decode("utf-8")
        raw = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("terminal journal JSON is invalid") from exc
    if not isinstance(raw, dict):
        raise ValueError("terminal journal must be a JSON object")
    expected_fields = {field.name for field in fields(TerminalJournal)}
    if set(raw) != expected_fields:
        raise ValueError("terminal journal fields are not exactly the frozen schema")
    try:
        raw["state"] = TerminalState(raw["state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal journal state is unknown") from exc
    identity = raw["attempt_path_identity"]
    if identity is not None:
        if not isinstance(identity, list):
            raise TypeError("attempt_path_identity JSON value must be an array")
        raw["attempt_path_identity"] = tuple(identity)
    order = raw["evidence_order"]
    if order is not None:
        if not isinstance(order, list):
            raise TypeError("evidence_order JSON value must be an array")
        raw["evidence_order"] = tuple(order)
    digests = raw["evidence_payload_sha256"]
    if digests is not None:
        if not isinstance(digests, dict) or set(digests) != set(FORMAL_EVIDENCE_ORDER):
            raise ValueError("evidence digest JSON object has invalid fields")
        raw["evidence_payload_sha256"] = tuple(
            (name, digests[name]) for name in FORMAL_EVIDENCE_ORDER
        )
    failure = raw["cleanup_failure"]
    if failure is not None:
        if not isinstance(failure, dict) or set(failure) != {"stage", "exception_type"}:
            raise ValueError("cleanup_failure fields are invalid")
        raw["cleanup_failure"] = CleanupFailure(**failure)
    record = TerminalJournal(**raw)
    _validate_record(record)
    if canonical_terminal_journal_bytes(record) != payload:
        raise ValueError("terminal journal bytes are not canonical")
    return record


def _is_reparse(item: os.stat_result) -> bool:
    return bool(
        getattr(item, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _direct_root(root: Path) -> Path:
    absolute = Path(os.path.normpath(os.path.abspath(root)))
    for ancestor in (absolute, *absolute.parents):
        item = os.lstat(ancestor)
        if stat.S_ISLNK(item.st_mode) or _is_reparse(item):
            raise ValueError("selection root has a reparse lexical ancestor")
        if ancestor == absolute and not stat.S_ISDIR(item.st_mode):
            raise ValueError("selection root must be a direct non-reparse directory")
    return absolute


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and stat.S_IFMT(before.st_mode) == stat.S_IFMT(after.st_mode)
    )


def _direct_regular_bytes(root: Path, name: str) -> bytes:
    safe_root = _direct_root(root)
    path = safe_root / name
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"terminal entry is not a direct regular file: {name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_identity(before, opened):
            raise ValueError(f"terminal entry does not match pinned path: {name}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = os.lstat(path)
    if not _same_identity(before, after) or any(
        getattr(before, field, None) != getattr(after, field, None)
        for field in ("st_size", "st_mtime_ns")
    ):
        raise ValueError(f"terminal entry changed during pinned read: {name}")
    return payload


def _authenticate_e1_e2(root: Path, expectations: TerminalExpectations) -> None:
    for name, expected in (
        (FORMAL_EVIDENCE_ORDER[0], expectations.capacity_selection_manifest_sha256),
        (FORMAL_EVIDENCE_ORDER[1], expectations.capacity_metrics_sha256),
    ):
        _hash(name, expected)
        try:
            payload = _direct_regular_bytes(root, name)
        except FileNotFoundError as exc:
            raise ValueError(f"terminal input is missing: {name}") from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"terminal input hash mismatch: {name}")
        _validate_canonical_json_payload(name, payload)


def _validate_canonical_json_payload(name: str, payload: bytes) -> None:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"terminal input JSON is invalid: {name}") from exc
    # E1 predates the terminal journal and its project-wide canonical encoder
    # is UTF-8 without a terminal newline.  E2 is gate-local canonical JSON,
    # which is ASCII-safe and newline terminated.  Both byte grammars remain
    # exact; accepting the wrong one would invalidate their frozen hashes.
    selection_canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    gate_canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    allowed = (
        {selection_canonical, gate_canonical}
        if name == FORMAL_EVIDENCE_ORDER[0]
        else {gate_canonical}
    )
    if payload not in allowed:
        raise ValueError(f"terminal input JSON is not canonical: {name}")


def _validate_expectations(expectations: TerminalExpectations) -> None:
    if not isinstance(expectations, TerminalExpectations):
        raise TypeError("terminal expectations have the wrong type")
    _hash("capacity_selection_id", expectations.capacity_selection_id)
    selection = _normalized_root("selection_root", expectations.selection_root)
    _normalized_root("project_root", expectations.project_root)
    external = _normalized_root("external_root", expectations.external_root)
    if Path(selection) != Path(external) / "capacity" / expectations.capacity_selection_id:
        raise ValueError("expected selection root is inconsistent")
    _hash("invocation_sha256", expectations.invocation_sha256)
    _hash("capacity_selection_manifest_sha256", expectations.capacity_selection_manifest_sha256)
    _hash("capacity_metrics_sha256", expectations.capacity_metrics_sha256)


def _unlink_regular_next(root: Path) -> None:
    path = root / NEXT_NAME
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("journal temporary is not a direct regular file")
    _transition_hook("before_next_unlink")
    after = os.lstat(path)
    if not _same_identity(before, after):
        raise ValueError("journal temporary changed before unlink")
    path.unlink()


def _validate_reconstructible_prefix(
    root: Path, expectations: TerminalExpectations, *, require_next: bool
) -> None:
    children = _direct_children(root)
    required = set(FORMAL_EVIDENCE_ORDER[:2])
    if require_next:
        required.add(NEXT_NAME)
    allowed = required | {"attempts"}
    if not required.issubset(children) or set(children) - allowed:
        raise ValueError("journal temporary lacks an exact reconstructible E1+E2 prefix")
    for name in required:
        if not stat.S_ISREG(children[name].st_mode):
            raise ValueError("reconstructible terminal prefix contains a non-file entry")
    if "attempts" not in children:
        return
    if not stat.S_ISDIR(children["attempts"].st_mode):
        raise ValueError("reconstructible attempts entry is not a directory")
    attempts_root = root / "attempts"
    attempt_children = _direct_children(attempts_root)
    if len(attempt_children) != 1:
        raise ValueError("reconstructible prefix requires at most one public attempt")
    attempt_name, attempt_stat = next(iter(attempt_children.items()))
    if _ATTEMPT_ID_RE.fullmatch(attempt_name) is None or not stat.S_ISDIR(attempt_stat.st_mode):
        raise ValueError("reconstructible prefix has a foreign attempt")
    attempt_root = attempts_root / attempt_name
    _direct_root(attempt_root)
    invocation = _direct_regular_bytes(attempt_root, "invocation.json")
    if hashlib.sha256(invocation).hexdigest() != expectations.invocation_sha256:
        raise ValueError("public attempt invocation does not match the expected invocation")
    _validate_canonical_json_payload("invocation.json", invocation)


def load_terminal_journal(
    selection_root: Path, expectations: TerminalExpectations
) -> TerminalJournal | None:
    _validate_expectations(expectations)
    root = _direct_root(selection_root)
    if str(root) != expectations.selection_root:
        raise ValueError("selection root does not match terminal expectations")
    authority = root / JOURNAL_NAME
    try:
        authority_stat = os.lstat(authority)
    except FileNotFoundError:
        authority_stat = None
    if authority_stat is None:
        if (root / NEXT_NAME).exists() or os.path.lexists(root / NEXT_NAME):
            _authenticate_e1_e2(root, expectations)
            _validate_reconstructible_prefix(root, expectations, require_next=True)
            _unlink_regular_next(root)
        else:
            _authenticate_e1_e2(root, expectations)
            _validate_reconstructible_prefix(root, expectations, require_next=False)
        return None
    if stat.S_ISLNK(authority_stat.st_mode) or _is_reparse(authority_stat) or not stat.S_ISREG(
        authority_stat.st_mode
    ):
        raise ValueError("journal authority is not a direct regular file")
    record = _parse_record(_direct_regular_bytes(root, JOURNAL_NAME))
    for field in fields(TerminalExpectations):
        if getattr(record, field.name) != getattr(expectations, field.name):
            raise ValueError(f"journal does not match expected {field.name}")
    _authenticate_e1_e2(root, expectations)
    if (root / NEXT_NAME).exists() or os.path.lexists(root / NEXT_NAME):
        classify_terminal_shape(root, record)
        _unlink_regular_next(root)
    return record


def _direct_children(path: Path) -> dict[str, os.stat_result]:
    result: dict[str, os.stat_result] = {}
    safe_path = _direct_root(path)
    before_root = os.lstat(safe_path)
    with os.scandir(path) as entries:
        for entry in entries:
            # Windows DirEntry can expose zero inode/device values.  Re-lstat
            # through the already pinned lexical directory for the identity we
            # retain and revalidate below.
            item = os.lstat(safe_path / entry.name)
            if stat.S_ISLNK(item.st_mode) or _is_reparse(item):
                raise ValueError("terminal shape contains a link or reparse entry")
            result[entry.name] = item
    after_root = os.lstat(safe_path)
    if not _same_identity(before_root, after_root):
        raise ValueError("terminal directory changed during enumeration")
    for name, item in result.items():
        try:
            current = os.lstat(safe_path / name)
        except FileNotFoundError as exc:
            raise ValueError("terminal entry disappeared during enumeration") from exc
        if not _same_identity(item, current):
            raise ValueError("terminal entry changed during enumeration")
    return result


def classify_terminal_shape(selection_root: Path, journal: TerminalJournal) -> TerminalShape:
    _validate_record(journal)
    root = _direct_root(selection_root)
    if str(root) != journal.selection_root:
        raise ValueError("terminal shape root disagrees with journal")
    children = _direct_children(root)
    permitted_controls = {JOURNAL_NAME, NEXT_NAME}
    evidence_present: list[str] = []
    for name in FORMAL_EVIDENCE_ORDER:
        if name in children:
            if not stat.S_ISREG(children[name].st_mode):
                raise ValueError("formal evidence entry is not a direct regular file")
            evidence_present.append(name)
    allowed_names = set(evidence_present) | permitted_controls | {"attempts"}
    if set(children) - allowed_names:
        raise ValueError("terminal shape contains an unknown or legacy entry")
    if NEXT_NAME in children and not stat.S_ISREG(children[NEXT_NAME].st_mode):
        raise ValueError("journal temporary is not a direct regular file")
    if JOURNAL_NAME in children and not stat.S_ISREG(children[JOURNAL_NAME].st_mode):
        raise ValueError("journal authority is not a direct regular file")
    prefix = tuple(evidence_present)
    if prefix != FORMAL_EVIDENCE_ORDER[: len(prefix)]:
        raise ValueError("formal evidence is not an ordered prefix")

    attempts_exists = "attempts" in children
    attempt_kind: Literal["NONE", "PUBLIC", "TOMBSTONE"] = "NONE"
    attempt_relative: str | None = None
    if attempts_exists:
        if not stat.S_ISDIR(children["attempts"].st_mode):
            raise ValueError("attempts entry is not a direct directory")
        attempt_children = _direct_children(root / "attempts")
        if len(attempt_children) > 1:
            raise ValueError("terminal shape contains multiple attempt children")
        if attempt_children:
            name, item = next(iter(attempt_children.items()))
            if not stat.S_ISDIR(item.st_mode):
                raise ValueError("attempt entry is not a direct directory")
            relative = "attempts/" + name
            if relative == journal.attempt_public_relative_path:
                attempt_kind = "PUBLIC"
            elif relative == journal.attempt_tombstone_relative_path:
                attempt_kind = "TOMBSTONE"
            else:
                raise ValueError("terminal shape contains a foreign attempt")
            attempt_relative = relative
            if journal.attempt_mode == "OWNED":
                pinned_attempt = os.lstat(root / "attempts" / name)
                identity = (
                    int(pinned_attempt.st_dev),
                    int(pinned_attempt.st_ino),
                    stat.S_IFMT(pinned_attempt.st_mode),
                )
                if identity != journal.attempt_path_identity:
                    raise ValueError("terminal attempt path identity changed")
                if int(pinned_attempt.st_dev) != journal.attempt_volume_identity:
                    raise ValueError("terminal attempt volume identity changed")

    if journal.state in {
        TerminalState.METRICS_READY,
        TerminalState.PREPARED,
        TerminalState.OUTCOME_CLEAN,
        TerminalState.OUTCOME_FAILED,
    }:
        if prefix != FORMAL_EVIDENCE_ORDER[:2]:
            raise ValueError("journal state requires exactly E1 and E2")
    elif journal.state is TerminalState.SEALING:
        if len(prefix) < 2:
            raise ValueError("SEALING requires at least E1 and E2")
    elif prefix != FORMAL_EVIDENCE_ORDER:
        raise ValueError("COMMITTED requires the exact five-file envelope")

    if journal.state is TerminalState.METRICS_READY:
        if attempts_exists and journal.attempt_mode == "NONE":
            raise ValueError("NONE attempt mode cannot have an attempts root")
        if journal.attempt_mode == "OWNED" and attempt_kind != "PUBLIC":
            raise ValueError("METRICS_READY owned attempt must be public")
    elif journal.state is TerminalState.PREPARED:
        if journal.attempt_mode == "NONE" and attempts_exists:
            raise ValueError("NONE attempt mode cannot have an attempts root")
    elif journal.cleanup_status == "CLEAN":
        if attempts_exists:
            raise ValueError("clean outcome cannot retain an attempts root")
    else:
        if attempt_relative != journal.retained_attempt_relative_path:
            raise ValueError("failed outcome does not have its bound retained attempt")
    return TerminalShape(prefix, attempt_kind, attempt_relative)


def _expectations_from(record: TerminalJournal) -> TerminalExpectations:
    return TerminalExpectations(
        capacity_selection_id=record.capacity_selection_id,
        selection_root=record.selection_root,
        project_root=record.project_root,
        external_root=record.external_root,
        invocation_sha256=record.invocation_sha256,
        capacity_selection_manifest_sha256=record.capacity_selection_manifest_sha256,
        capacity_metrics_sha256=record.capacity_metrics_sha256,
    )


def _validate_transition(previous: TerminalJournal, next_record: TerminalJournal) -> None:
    _validate_record(previous)
    _validate_record(next_record)
    if not allowed_terminal_transition(previous.state, next_record.state):
        raise ValueError("terminal journal transition is not allowed")
    if next_record.generation != previous.generation + 1:
        raise ValueError("terminal journal generation must advance by exactly one")
    for name in _IDENTITY_FIELDS:
        if getattr(previous, name) != getattr(next_record, name):
            raise ValueError(f"terminal journal transition mutates frozen {name}")
    if previous.state is TerminalState.PREPARED:
        return
    mutable_for_transition = {
        TerminalState.OUTCOME_CLEAN: {
            "evidence_order", "evidence_payload_sha256", "ended_utc"
        },
        TerminalState.OUTCOME_FAILED: {
            "evidence_order", "evidence_payload_sha256", "ended_utc"
        },
        TerminalState.SEALING: {"committed_envelope_sha256"},
        TerminalState.METRICS_READY: set(),
    }[previous.state]
    for field in fields(TerminalJournal):
        if field.name in {"state", "generation"} | set(_IDENTITY_FIELDS) | mutable_for_transition:
            continue
        if getattr(previous, field.name) != getattr(next_record, field.name):
            raise ValueError(f"terminal transition mutates previously frozen {field.name}")


def _sync_directory(root: Path) -> None:
    try:
        descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _transition_hook(stage: str) -> None:
    """Test seam for process interruption after durable transition actions."""
    del stage


def replace_terminal_journal(
    selection_root: Path,
    previous: TerminalJournal | None,
    next_record: TerminalJournal,
) -> None:
    _validate_record(next_record)
    root = _direct_root(selection_root)
    if str(root) != next_record.selection_root:
        raise ValueError("journal replacement root disagrees with record")
    authority = root / JOURNAL_NAME
    temporary = root / NEXT_NAME
    if previous is None:
        if next_record.state is not TerminalState.METRICS_READY or next_record.generation != 0:
            raise ValueError("first journal generation must be METRICS_READY/0")
        if os.path.lexists(authority):
            raise FileExistsError("terminal journal authority already exists")
        if os.path.lexists(temporary):
            raise FileExistsError("terminal journal temporary already exists")
        _authenticate_e1_e2(root, _expectations_from(next_record))
        classify_terminal_shape(root, next_record)
    else:
        _validate_transition(previous, next_record)
        current = load_terminal_journal(root, _expectations_from(previous))
        if current != previous:
            raise ValueError("authoritative journal changed before replacement")
        classify_terminal_shape(root, previous)
        if os.path.lexists(temporary):
            raise FileExistsError("terminal journal temporary already exists")

    payload = canonical_terminal_journal_bytes(next_record)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if _direct_regular_bytes(root, NEXT_NAME) != payload:
            raise ValueError("journal temporary bytes changed after flush")
        _transition_hook("after_next_flush")
        if previous is None:
            if os.path.lexists(authority):
                raise FileExistsError("terminal journal authority appeared before creation")
        elif _direct_regular_bytes(root, JOURNAL_NAME) != canonical_terminal_journal_bytes(previous):
            raise ValueError("authoritative journal changed before atomic replacement")
        if previous is None:
            try:
                os.link(temporary, authority)
            except FileExistsError as exc:
                raise FileExistsError(
                    "terminal journal authority appeared before creation"
                ) from exc
            temporary.unlink()
        else:
            os.replace(temporary, authority)
        _transition_hook("after_replace")
        _sync_directory(root)
        if _direct_regular_bytes(root, JOURNAL_NAME) != payload:
            raise ValueError("journal authority differs after atomic replacement")
    except Exception:
        try:
            _unlink_regular_next(root)
        except (FileNotFoundError, ValueError):
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
