import hashlib
import json
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "r03_nb2_review_protocol.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nb2_review_protocol_freezes_corrected_datasets_and_scope() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "r03_nb2_review_protocol_v1"
    assert protocol["target"] == "future_onset_proximity_event_count"
    assert protocol["feature_ladder"] == "B"
    assert protocol["horizons_seconds"] == [30, 120, 300]
    assert protocol["threshold_quantiles"] == [0.7, 0.75, 0.8, 0.9]
    assert protocol["threshold_interpolation"] == "higher"
    assert protocol["fit_scope"] == "KBTP development train split only"
    assert protocol["allow_site_tuning"] is False
    assert protocol["primary_group_unit"] == "source_group"
    assert protocol["fallback"] == "supplementary_only_retain_current_primary_models"
    assert protocol["prohibited"] == [
        "KAGC_parameter_fitting",
        "site_specific_thresholds",
        "representation_reselection",
        "additional_model_families",
    ]


def test_nb2_review_protocol_hashes_match_the_three_scene_files() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    for record in protocol["datasets"].values():
        path = PROJECT_ROOT / record["relative_path"]
        if not path.is_file():
            pytest.skip("derived ADS-B scene artifacts are not redistributed")
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_nb2_review_gate_requires_distributional_improvement_at_two_horizons() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    gate = protocol["main_text_gate"]

    assert gate["minimum_horizons"] == 2
    assert gate["count_distribution"] == "NB2 group-equal NLL < Poisson group-equal NLL"
    assert gate["exceedance"] == (
        "NB2 group-equal Brier or log loss < Logistic at the frozen q75 threshold"
    )
    assert gate["direction"] == "strict majority of source groups on the relevant evaluation set"
