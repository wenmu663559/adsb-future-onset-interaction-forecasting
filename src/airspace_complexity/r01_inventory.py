from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
import zlib
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import zipfile_deflate64  # noqa: F401  # Registers ZIP compression method 9.


DATE_MDY_RE = re.compile(r"(?<!\d)(\d{2}-\d{2}-\d{2})(?!\d)")
DATE_YMD_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def extract_date_token(text: str) -> str | None:
    """Extract one documented date token without assigning timezone semantics."""
    mdy_match = DATE_MDY_RE.search(text)
    if mdy_match:
        token = mdy_match.group(1)
        try:
            datetime.strptime(token, "%m-%d-%y")
        except ValueError:
            return None
        return token

    ymd_match = DATE_YMD_RE.search(text)
    if not ymd_match:
        return None
    try:
        parsed = datetime.strptime(ymd_match.group(1), "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.strftime("%m-%d-%y")


def _extract_all_date_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in DATE_MDY_RE.finditer(text):
        token = match.group(1)
        try:
            datetime.strptime(token, "%m-%d-%y")
        except ValueError:
            continue
        tokens.add(token)
    for match in DATE_YMD_RE.finditer(text):
        token = match.group(1)
        try:
            parsed = datetime.strptime(token, "%Y-%m-%d")
        except ValueError:
            continue
        tokens.add(parsed.strftime("%m-%d-%y"))
    return tokens


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest while reading a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the unsigned CRC32 used by ZIP central directories."""
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def compression_method_supported(method: int) -> bool:
    """Report whether the active ZIP implementation can decompress a method."""
    try:
        zipfile._get_decompressor(method)
    except NotImplementedError:
        return False
    return True


def _ends_with_newline(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) in {b"\n", b"\r"}


def _tolerant_csv_parse(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            for _ in csv.reader(handle, strict=False):
                pass
    except (OSError, csv.Error) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def audit_csv(path: Path, expected_date: str | None) -> dict[str, Any]:
    """Audit CSV structure without interpreting undocumented field semantics."""
    size_bytes = path.stat().st_size
    result: dict[str, Any] = {
        "is_empty": size_bytes == 0,
        "header_only": False,
        "strict_parse_ok": True,
        "tolerant_parse_ok": True,
        "header": [],
        "header_column_count": 0,
        "logical_record_count": 0,
        "data_row_count": 0,
        "blank_row_count": 0,
        "column_mismatch_count": 0,
        "ends_with_newline": _ends_with_newline(path),
        "last_five_complete": True,
        "parse_error": None,
        "decode_error": None,
        "tolerant_parse_error": None,
        "date_status": "not_checked" if expected_date is None else "not_assessable",
        "observed_date_tokens": [],
        "last_five_records": [],
        "terminal_truncation_candidate": False,
    }

    if size_bytes == 0:
        return result

    header: list[str] | None = None
    date_index: int | None = None
    observed_dates: set[str] = set()
    last_records: deque[list[str]] = deque(maxlen=5)

    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            for row in reader:
                result["logical_record_count"] += 1
                if not row or all(not value.strip() for value in row):
                    result["blank_row_count"] += 1
                    continue

                if header is None:
                    header = row
                    result["header"] = header
                    result["header_column_count"] = len(header)
                    try:
                        date_index = header.index("Date")
                    except ValueError:
                        date_index = None
                    continue

                result["data_row_count"] += 1
                last_records.append(row)
                if len(row) != len(header):
                    result["column_mismatch_count"] += 1
                if date_index is not None and date_index < len(row):
                    observed_dates.update(_extract_all_date_tokens(row[date_index]))
    except UnicodeDecodeError as exc:
        result["strict_parse_ok"] = False
        result["decode_error"] = (
            f"UnicodeDecodeError at byte {exc.start}: {exc.reason}"
        )
    except (OSError, csv.Error) as exc:
        result["strict_parse_ok"] = False
        result["parse_error"] = f"{type(exc).__name__}: {exc}"

    if not result["strict_parse_ok"]:
        tolerant_ok, tolerant_error = _tolerant_csv_parse(path)
        result["tolerant_parse_ok"] = tolerant_ok
        result["tolerant_parse_error"] = tolerant_error

    result["header_only"] = (
        result["strict_parse_ok"]
        and bool(result["header"])
        and result["data_row_count"] == 0
    )
    result["last_five_records"] = list(last_records)
    if header is not None:
        result["last_five_complete"] = all(
            len(row) == len(header) for row in last_records
        )

    result["observed_date_tokens"] = sorted(observed_dates)
    if expected_date is not None and date_index is not None:
        if not observed_dates:
            result["date_status"] = "no_date_values"
        elif observed_dates == {expected_date}:
            result["date_status"] = "match"
        else:
            result["date_status"] = "mismatch"

    parse_error = str(result.get("parse_error") or "").lower()
    result["terminal_truncation_candidate"] = (
        result["ends_with_newline"] is False
        and int(result["data_row_count"]) > 0
        and (
            (
                result["strict_parse_ok"] is False
                and result["decode_error"] is None
                and (
                    "unexpected end of data" in parse_error
                    or "field larger than field limit" in parse_error
                )
            )
            or (
                result["strict_parse_ok"] is True
                and int(result["column_mismatch_count"]) == 1
                and result["last_five_complete"] is False
            )
        )
    )

    return result


def audit_zip(path: Path) -> dict[str, Any]:
    """Validate a ZIP archive's directory and member CRCs without extraction."""
    result: dict[str, Any] = {
        "zip_valid": False,
        "zip_entry_count": 0,
        "zip_bad_member": None,
        "zip_error": None,
        "zip_entries": [],
        "zip_members": [],
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            entries = [
                info.filename.replace("\\", "/")
                for info in archive.infolist()
                if not info.is_dir()
            ]
            result["zip_entries"] = entries
            result["zip_members"] = [
                {
                    "filename": info.filename.replace("\\", "/"),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "compress_type": info.compress_type,
                    "crc32": f"{info.CRC:08x}",
                }
                for info in archive.infolist()
                if not info.is_dir()
            ]
            result["zip_entry_count"] = len(entries)
            result["zip_bad_member"] = archive.testzip()
            result["zip_valid"] = result["zip_bad_member"] is None
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        result["zip_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _path_year(relative_path: Path) -> str | None:
    for part in relative_path.parts:
        if re.fullmatch(r"\d{4}", part):
            return part
    return None


def _path_date_directory(relative_path: Path) -> str | None:
    for part in relative_path.parts:
        if re.fullmatch(r"\d{2}-\d{2}-\d{2}", part):
            return extract_date_token(part)
    return None


def _empty_type_specific_fields() -> dict[str, Any]:
    return {
        "is_empty": None,
        "header_only": None,
        "strict_parse_ok": None,
        "tolerant_parse_ok": None,
        "header": [],
        "header_column_count": None,
        "logical_record_count": None,
        "data_row_count": None,
        "blank_row_count": None,
        "column_mismatch_count": None,
        "ends_with_newline": None,
        "last_five_complete": None,
        "parse_error": None,
        "decode_error": None,
        "tolerant_parse_error": None,
        "date_status": "not_applicable",
        "observed_date_tokens": [],
        "last_five_records": [],
        "terminal_truncation_candidate": None,
        "zip_valid": None,
        "zip_entry_count": None,
        "zip_bad_member": None,
        "zip_error": None,
        "zip_entries": [],
        "zip_members": [],
    }


def audit_file(
    airport_id: str,
    airport_root: Path,
    path: Path,
) -> dict[str, Any]:
    """Audit one file and retain an airport-relative source identity."""
    relative_path = path.relative_to(airport_root)
    stat = path.stat()
    extension = path.suffix.lower()
    row: dict[str, Any] = {
        "airport_id": airport_id.lower(),
        "relative_path": relative_path.as_posix(),
        "filename": path.name,
        "extension": extension,
        "year": _path_year(relative_path),
        "date_directory": _path_date_directory(relative_path),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(
            stat.st_mtime
        ).astimezone().isoformat(),
        "modified_time_ns": stat.st_mtime_ns,
        "sha256": None,
        "crc32": None,
        "readable": False,
        "audit_type": "binary",
        "audit_error": None,
        "is_symlink": path.is_symlink(),
    }
    row.update(_empty_type_specific_fields())

    try:
        row["sha256"] = sha256_file(path)
        row["crc32"] = crc32_file(path)
        row["readable"] = True
        if extension == ".csv":
            row["audit_type"] = "csv"
            row.update(audit_csv(path, row["date_directory"]))
        elif extension == ".zip":
            row["audit_type"] = "zip"
            row.update(audit_zip(path))
    except (OSError, ValueError) as exc:
        row["audit_error"] = f"{type(exc).__name__}: {exc}"
        row["readable"] = False
    return row


def discover_files(
    airport_roots: dict[str, Path],
    progress: Callable[[int, int, Path], None] | None = None,
) -> list[dict[str, Any]]:
    """Discover and audit all regular files in stable airport/path order."""
    discovered: list[tuple[str, Path, Path]] = []
    for airport_id in sorted(airport_roots):
        root = airport_roots[airport_id]
        if not root.is_dir():
            raise FileNotFoundError(f"Airport raw directory not found: {root}")
        for path in root.rglob("*"):
            if path.is_file():
                discovered.append((airport_id, root, path))

    discovered.sort(
        key=lambda item: (item[0].lower(), item[2].relative_to(item[1]).as_posix())
    )
    total = len(discovered)
    rows: list[dict[str, Any]] = []
    for index, (airport_id, root, path) in enumerate(discovered, start=1):
        rows.append(audit_file(airport_id, root, path))
        if progress is not None:
            progress(index, total, path)
    return rows


def summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize structural findings without inferring dataset adequacy."""
    hash_counts = Counter(
        str(row["sha256"])
        for row in rows
        if isinstance(row.get("sha256"), str) and row.get("sha256")
    )
    return {
        "files_discovered": len(rows),
        "files_audited": sum(
            1
            for row in rows
            if isinstance(row.get("sha256"), str)
            and len(str(row["sha256"])) == 64
        ),
        "total_size_bytes": sum(int(row.get("size_bytes") or 0) for row in rows),
        "csv_count": sum(row.get("audit_type") == "csv" for row in rows),
        "zip_count": sum(row.get("audit_type") == "zip" for row in rows),
        "other_file_count": sum(
            row.get("audit_type") not in {"csv", "zip"} for row in rows
        ),
        "missing_hash_count": sum(not row.get("sha256") for row in rows),
        "unreadable_count": sum(row.get("readable") is not True for row in rows),
        "strict_parse_failures": sum(
            row.get("audit_type") == "csv"
            and row.get("strict_parse_ok") is not True
            for row in rows
        ),
        "decode_failures": sum(
            row.get("audit_type") == "csv" and bool(row.get("decode_error"))
            for row in rows
        ),
        "empty_csv_count": sum(
            row.get("audit_type") == "csv" and row.get("is_empty") is True
            for row in rows
        ),
        "header_only_csv_count": sum(
            row.get("audit_type") == "csv" and row.get("header_only") is True
            for row in rows
        ),
        "blank_row_files": sum(
            row.get("audit_type") == "csv"
            and int(row.get("blank_row_count") or 0) > 0
            for row in rows
        ),
        "column_mismatch_files": sum(
            row.get("audit_type") == "csv"
            and int(row.get("column_mismatch_count") or 0) > 0
            for row in rows
        ),
        "missing_final_newline_files": sum(
            row.get("audit_type") == "csv"
            and row.get("ends_with_newline") is False
            and row.get("is_empty") is False
            for row in rows
        ),
        "date_mismatch_files": sum(
            row.get("audit_type") == "csv"
            and row.get("date_status") == "mismatch"
            for row in rows
        ),
        "invalid_zip_count": sum(
            row.get("audit_type") == "zip" and row.get("zip_valid") is not True
            for row in rows
        ),
        "duplicate_hash_groups": sum(count > 1 for count in hash_counts.values()),
        "duplicate_file_count": sum(
            count for count in hash_counts.values() if count > 1
        ),
        "symlink_count": sum(row.get("is_symlink") is True for row in rows),
        "airport_counts": dict(
            sorted(Counter(str(row["airport_id"]) for row in rows).items())
        ),
        "extension_counts": dict(
            sorted(Counter(str(row["extension"]) for row in rows).items())
        ),
    }


def build_header_variants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group exact CSV headers and retain source coverage."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("audit_type") != "csv":
            continue
        header = row.get("header")
        if isinstance(header, list) and header:
            groups[tuple(str(value) for value in header)].append(row)

    variants = []
    for header, members in groups.items():
        variants.append(
            {
                "header": list(header),
                "file_count": len(members),
                "airports": sorted({str(row["airport_id"]) for row in members}),
                "sample_paths": sorted(
                    str(row["relative_path"]) for row in members
                )[:10],
            }
        )
    variants.sort(
        key=lambda item: (
            -int(item["file_count"]),
            json.dumps(item["header"], ensure_ascii=False),
        )
    )
    return {"variant_count": len(variants), "variants": variants}


def build_date_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build calendar coverage from directory tokens only."""
    grouped: dict[str, dict[datetime, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"file_count": 0, "size_bytes": 0})
    )
    for row in rows:
        token = row.get("date_directory")
        if row.get("audit_type") != "csv" or not isinstance(token, str):
            continue
        try:
            parsed = datetime.strptime(token, "%m-%d-%y")
        except ValueError:
            continue
        airport = str(row["airport_id"])
        grouped[airport][parsed]["file_count"] += 1
        grouped[airport][parsed]["size_bytes"] += int(row.get("size_bytes") or 0)

    output: list[dict[str, Any]] = []
    for airport in sorted(grouped):
        dates = grouped[airport]
        if not dates:
            continue
        current = min(dates)
        end = max(dates)
        while current <= end:
            stats = dates.get(current, {"file_count": 0, "size_bytes": 0})
            output.append(
                {
                    "airport_id": airport,
                    "date": current.strftime("%m-%d-%y"),
                    "year": current.year,
                    "file_count": stats["file_count"],
                    "size_bytes": stats["size_bytes"],
                    "missing": stats["file_count"] == 0,
                }
            )
            current = current.fromordinal(current.toordinal() + 1)
    return output


def _percentile(values: list[int], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return float(ordered[0])
    position = proportion * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_size_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute transparent size summaries by airport and extension."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["airport_id"]), str(row["extension"]))].append(
            int(row.get("size_bytes") or 0)
        )

    output = []
    for (airport, extension), values in sorted(grouped.items()):
        output.append(
            {
                "airport_id": airport,
                "extension": extension,
                "file_count": len(values),
                "min_bytes": min(values),
                "p01_bytes": _percentile(values, 0.01),
                "p25_bytes": _percentile(values, 0.25),
                "median_bytes": median(values),
                "p75_bytes": _percentile(values, 0.75),
                "p99_bytes": _percentile(values, 0.99),
                "max_bytes": max(values),
                "total_bytes": sum(values),
            }
        )
    return output


def _archive_member_key(path_text: str) -> tuple[str, str] | None:
    normalized = path_text.replace("\\", "/")
    date_token = extract_date_token(normalized)
    filename = Path(normalized).name.lower()
    if date_token is None or not filename.endswith(".csv"):
        return None
    return date_token, filename


def build_archive_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare ZIP member size/CRC evidence with extracted CSV files."""
    extracted_by_airport_year: dict[
        tuple[str, str | None], dict[tuple[str, str], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("audit_type") != "csv":
            continue
        key = _archive_member_key(str(row["relative_path"]))
        if key is not None:
            extracted_by_airport_year[
                (str(row["airport_id"]), row.get("year"))
            ][key].append(row)

    output: list[dict[str, Any]] = []
    for archive_row in rows:
        if archive_row.get("audit_type") != "zip":
            continue
        extracted = extracted_by_airport_year[
            (str(archive_row["airport_id"]), archive_row.get("year"))
        ]
        for member in archive_row.get("zip_members") or []:
            member_path = str(member.get("filename") or "")
            key = _archive_member_key(member_path)
            if key is None:
                continue
            candidates = sorted(
                extracted.get(key, []),
                key=lambda row: str(row["relative_path"]),
            )
            exact = next(
                (
                    row
                    for row in candidates
                    if int(row.get("size_bytes") or -1)
                    == int(member.get("file_size") or -2)
                    and str(row.get("crc32") or "").lower()
                    == str(member.get("crc32") or "").lower()
                ),
                None,
            )
            selected = exact or (candidates[0] if candidates else None)
            if archive_row.get("zip_valid") is not True:
                status = "archive_unverified"
            elif exact is not None:
                status = "exact_match"
            elif selected is None:
                status = "missing_extracted_recoverable"
            else:
                status = "content_mismatch_recoverable"
            output.append(
                {
                    "airport_id": archive_row["airport_id"],
                    "archive_relative_path": archive_row["relative_path"],
                    "archive_year": archive_row.get("year"),
                    "zip_valid": archive_row.get("zip_valid"),
                    "member_path": member_path,
                    "member_date": key[0],
                    "member_filename": key[1],
                    "member_size_bytes": member.get("file_size"),
                    "member_crc32": member.get("crc32"),
                    "member_compress_type": member.get("compress_type"),
                    "extracted_relative_path": (
                        selected.get("relative_path") if selected else None
                    ),
                    "extracted_size_bytes": (
                        selected.get("size_bytes") if selected else None
                    ),
                    "extracted_crc32": (
                        selected.get("crc32") if selected else None
                    ),
                    "status": status,
                }
            )
    return output


def decide_r01(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply conservative, deterministic R01 blocking rules."""
    blockers: list[str] = []
    discovered = int(summary.get("files_discovered") or 0)
    audited = int(summary.get("files_audited") or 0)
    if discovered == 0:
        blockers.append("No raw files were discovered.")
    if audited != discovered:
        blockers.append(
            f"Audit coverage mismatch: {audited} hashed of {discovered} discovered."
        )

    checks = [
        ("missing_hash_count", "files without SHA-256"),
        ("unreadable_count", "unreadable files"),
        ("unrecoverable_csv_count", "unrecoverable non-terminal CSV files"),
        ("invalid_zip_count", "invalid ZIP archives"),
        ("archive_unverified_count", "unverified archive CSV members"),
        ("symlink_count", "symbolic-link files"),
    ]
    for key, label in checks:
        count = int(summary.get(key) or 0)
        if count:
            blockers.append(f"{count} {label}.")

    if blockers:
        return "REVISE_R01", blockers
    return "PROCEED_TO_R02", []


def _serializable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({key for row in rows for key in row})
    output = []
    for row in rows:
        serialized: dict[str, Any] = {}
        for key in keys:
            value = row.get(key)
            if isinstance(value, (list, dict, tuple)):
                serialized[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            else:
                serialized[key] = value
        output.append(serialized)
    return output


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 CSV with structured values encoded as JSON."""
    serialized = _serializable_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in serialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized)


def write_manifest_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the same scalarized manifest rows to Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    serialized = _serializable_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(serialized)
    pq.write_table(table, path, compression="zstd")
