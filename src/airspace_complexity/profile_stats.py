"""Deterministic, mergeable primitives for bounded field profiling."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Iterable, Mapping
from typing import Any


def _stable_hash(value: Any) -> int:
    """Return a process-independent SHA-256 priority for a scalar value."""
    value_type = type(value)
    token = f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest(), "big")


def _finite_number(value: Any) -> float | None:
    """Return a finite scalar numeric value, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_exact_state(
    name: str,
    state: Mapping[str, Any],
    expected: set[str],
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} state must be a mapping")
    if set(state) != expected:
        raise ValueError(f"{name} state fields mismatch")


def _state_integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class StableHyperLogLog:
    """A SHA-256-based HyperLogLog sketch with register-wise merging."""

    def __init__(self, precision: int = 12) -> None:
        if isinstance(precision, bool) or not isinstance(precision, int):
            raise TypeError("precision must be an integer")
        if not 4 <= precision <= 18:
            raise ValueError("precision must be between 4 and 18")
        self.precision = precision
        self._register_count = 1 << precision
        self._registers = [0] * self._register_count

    @property
    def relative_error(self) -> float:
        return 1.04 / math.sqrt(self._register_count)

    def add(self, value: Any) -> None:
        hashed = _stable_hash(value)
        index = hashed & (self._register_count - 1)
        remainder = hashed >> self.precision
        remaining_bits = 256 - self.precision
        rank = (
            remaining_bits - remainder.bit_length() + 1
            if remainder
            else remaining_bits + 1
        )
        self._registers[index] = max(self._registers[index], rank)

    def merge(self, other: "StableHyperLogLog") -> None:
        if not isinstance(other, StableHyperLogLog):
            raise TypeError("can only merge another StableHyperLogLog")
        if self.precision != other.precision:
            raise ValueError("cannot merge sketches with different precisions")
        self._registers = [
            max(left, right)
            for left, right in zip(self._registers, other._registers)
        ]

    def estimate(self) -> float:
        register_sum = sum(2.0**-register for register in self._registers)
        alpha = {
            16: 0.673,
            32: 0.697,
            64: 0.709,
        }.get(
            self._register_count,
            0.7213 / (1.0 + 1.079 / self._register_count),
        )
        estimate = alpha * self._register_count**2 / register_sum
        empty_registers = self._registers.count(0)
        if estimate <= 2.5 * self._register_count and empty_registers:
            return self._register_count * math.log(self._register_count / empty_registers)
        return estimate

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": "stable_hll_v1",
            "precision": self.precision,
            "registers": list(self._registers),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "StableHyperLogLog":
        _require_exact_state(
            "StableHyperLogLog",
            state,
            {"schema_version", "precision", "registers"},
        )
        if state["schema_version"] != "stable_hll_v1":
            raise ValueError("unsupported StableHyperLogLog state schema")
        precision = _state_integer("precision", state["precision"], minimum=4)
        profile = cls(precision=precision)
        registers = state["registers"]
        if not isinstance(registers, list):
            raise TypeError("registers must be a list")
        if len(registers) != profile._register_count:
            raise ValueError("register count does not match precision")
        maximum_rank = 256 - precision + 1
        validated = [
            _state_integer("register", value) for value in registers
        ]
        if any(value > maximum_rank for value in validated):
            raise ValueError("register exceeds the maximum HLL rank")
        profile._registers = validated
        return profile


class BottomHashSample:
    """A deterministic bottom-k sample, keyed by SHA-256 identity priority."""

    def __init__(self, capacity: int = 8192) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._entries: dict[int, tuple[str, float]] = {}
        self._priority_heap: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def _discard_stale_maxima(self) -> None:
        while self._priority_heap:
            priority = self._priority_heap[0][1]
            if priority in self._entries:
                return
            heapq.heappop(self._priority_heap)

    def _largest_priority(self) -> int:
        self._discard_stale_maxima()
        return self._priority_heap[0][1]

    def _add_priority(self, priority: int, value: float) -> None:
        tie_breaker = f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
        existing = self._entries.get(priority)
        if existing is not None:
            if tie_breaker < existing[0]:
                self._entries[priority] = (tie_breaker, value)
            return
        if len(self._entries) >= self.capacity:
            largest = self._largest_priority()
            if priority > largest:
                return
            heapq.heappop(self._priority_heap)
            del self._entries[largest]
        self._entries[priority] = (tie_breaker, value)
        heapq.heappush(self._priority_heap, (-priority, priority))

    def add(self, identity: Any, value: Any) -> None:
        number = _finite_number(value)
        if number is not None:
            self._add_priority(_stable_hash(identity), number)

    def merge(self, other: "BottomHashSample") -> None:
        if not isinstance(other, BottomHashSample):
            raise TypeError("can only merge another BottomHashSample")
        if self.capacity != other.capacity:
            raise ValueError("cannot merge samples with different capacities")
        for priority, (_, value) in other._entries.items():
            self._add_priority(priority, value)

    def quantiles(self, probabilities: Iterable[float]) -> dict[float, float | None]:
        requested = list(probabilities)
        for probability in requested:
            if not 0.0 <= probability <= 1.0:
                raise ValueError("quantile probabilities must be between 0 and 1")
        values = sorted(value for _, value in self._entries.values())
        if not values:
            return {probability: None for probability in requested}
        last_index = len(values) - 1
        results: dict[float, float] = {}
        for probability in requested:
            position = probability * last_index
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                results[probability] = values[lower]
            else:
                weight = position - lower
                results[probability] = values[lower] + weight * (
                    values[upper] - values[lower]
                )
        return results

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": "bottom_hash_sample_v1",
            "capacity": self.capacity,
            "entries": [
                {
                    "priority": priority,
                    "tie_breaker": tie_breaker,
                    "value": value,
                }
                for priority, (tie_breaker, value) in sorted(
                    self._entries.items()
                )
            ],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BottomHashSample":
        _require_exact_state(
            "BottomHashSample",
            state,
            {"schema_version", "capacity", "entries"},
        )
        if state["schema_version"] != "bottom_hash_sample_v1":
            raise ValueError("unsupported BottomHashSample state schema")
        capacity = _state_integer("capacity", state["capacity"], minimum=1)
        entries = state["entries"]
        if not isinstance(entries, list):
            raise TypeError("entries must be a list")
        if len(entries) > capacity:
            raise ValueError("sample entries exceed capacity")
        sample = cls(capacity=capacity)
        seen: set[int] = set()
        for entry in entries:
            _require_exact_state(
                "BottomHashSample entry",
                entry,
                {"priority", "tie_breaker", "value"},
            )
            priority = _state_integer("priority", entry["priority"])
            if priority >= 1 << 256:
                raise ValueError("priority exceeds SHA-256 range")
            if priority in seen:
                raise ValueError("duplicate sample priority")
            seen.add(priority)
            value = _finite_number(entry["value"])
            if value is None:
                raise ValueError("sample value must be finite numeric")
            expected_tie = (
                f"{type(value).__module__}.{type(value).__qualname__}:"
                f"{value!r}"
            )
            if entry["tie_breaker"] != expected_tie:
                raise ValueError("sample tie-breaker does not match value")
            sample._add_priority(priority, value)
        return sample


class FieldProfile:
    """Counts field categories while retaining mergeable unique and numeric summaries."""

    def __init__(self) -> None:
        self.total = 0
        self.missing = 0
        self.invalid = 0
        self._minimum: float | None = None
        self._maximum: float | None = None
        self._unique = StableHyperLogLog()
        self._sample = BottomHashSample()

    def add(self, identity: Any, value: Any, error: Any = None) -> None:
        self.total += 1
        if error is not None:
            self.invalid += 1
            return
        if value is None:
            self.missing += 1
            return
        numeric_value = _finite_number(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if numeric_value is None:
                self.invalid += 1
                return
        self._unique.add(value)
        if numeric_value is None:
            return
        self._minimum = (
            numeric_value
            if self._minimum is None
            else min(self._minimum, numeric_value)
        )
        self._maximum = (
            numeric_value
            if self._maximum is None
            else max(self._maximum, numeric_value)
        )
        self._sample.add(identity, numeric_value)

    def merge(self, other: "FieldProfile") -> None:
        if not isinstance(other, FieldProfile):
            raise TypeError("can only merge another FieldProfile")
        self.total += other.total
        self.missing += other.missing
        self.invalid += other.invalid
        if other._minimum is not None:
            self._minimum = (
                other._minimum
                if self._minimum is None
                else min(self._minimum, other._minimum)
            )
        if other._maximum is not None:
            self._maximum = (
                other._maximum
                if self._maximum is None
                else max(self._maximum, other._maximum)
            )
        self._unique.merge(other._unique)
        self._sample.merge(other._sample)

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": "field_profile_v1",
            "total": self.total,
            "missing": self.missing,
            "invalid": self.invalid,
            "minimum": self._minimum,
            "maximum": self._maximum,
            "unique": self._unique.to_state(),
            "sample": self._sample.to_state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FieldProfile":
        _require_exact_state(
            "FieldProfile",
            state,
            {
                "schema_version",
                "total",
                "missing",
                "invalid",
                "minimum",
                "maximum",
                "unique",
                "sample",
            },
        )
        if state["schema_version"] != "field_profile_v1":
            raise ValueError("unsupported FieldProfile state schema")
        total = _state_integer("total", state["total"])
        missing = _state_integer("missing", state["missing"])
        invalid = _state_integer("invalid", state["invalid"])
        if missing + invalid > total:
            raise ValueError("profile category counts exceed total")
        minimum = state["minimum"]
        maximum = state["maximum"]
        if (minimum is None) != (maximum is None):
            raise ValueError("profile extrema must both be present or absent")
        if minimum is not None:
            minimum = _finite_number(minimum)
            maximum = _finite_number(maximum)
            if minimum is None or maximum is None:
                raise ValueError("profile extrema must be finite numeric")
            if minimum > maximum:
                raise ValueError("profile minimum exceeds maximum")
        profile = cls()
        profile.total = total
        profile.missing = missing
        profile.invalid = invalid
        profile._minimum = minimum
        profile._maximum = maximum
        profile._unique = StableHyperLogLog.from_state(state["unique"])
        profile._sample = BottomHashSample.from_state(state["sample"])
        return profile

    def to_record(self, stratum: Any) -> dict[str, Any]:
        probabilities = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        quantiles = self._sample.quantiles(probabilities)
        return {
            "stratum": stratum,
            "total": self.total,
            "missing": self.missing,
            "invalid": self.invalid,
            "finite_numeric_min": self._minimum,
            "finite_numeric_max": self._maximum,
            "approx_unique_count": round(self._unique.estimate()),
            "hll_relative_error": self._unique.relative_error,
            "sample_size": len(self._sample),
            "p01": quantiles[0.01],
            "p05": quantiles[0.05],
            "p25": quantiles[0.25],
            "p50": quantiles[0.5],
            "p75": quantiles[0.75],
            "p95": quantiles[0.95],
            "p99": quantiles[0.99],
        }
