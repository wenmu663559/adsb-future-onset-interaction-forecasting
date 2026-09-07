from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "project_root",
    "raw_data_root",
    "raw_airport_dirs",
    "local_output_root",
    "local_interim_root",
    "local_processed_research_root",
}

PATH_KEYS = {
    "project_root",
    "official_dataset_root",
    "raw_data_root",
    "processed_official_root",
    "legacy_project_root",
    "local_output_root",
    "local_interim_root",
    "local_processed_research_root",
}


def load_local_paths(config_path: Path) -> dict[str, Any]:
    """Load and validate local path configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Local path config not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Local path config root must be a mapping")

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing required path keys: {sorted(missing)}")

    airports = data["raw_airport_dirs"]
    if not isinstance(airports, dict):
        raise ValueError("raw_airport_dirs must be a mapping")

    missing_airports = {"kagc", "kbtp"} - airports.keys()
    if missing_airports:
        raise ValueError(f"Missing airport path keys: {sorted(missing_airports)}")

    result = dict(data)
    for key in PATH_KEYS:
        value = data.get(key)
        result[key] = Path(value) if value is not None else None
    result["raw_airport_dirs"] = {
        key: Path(value) for key, value in airports.items()
    }
    return result
