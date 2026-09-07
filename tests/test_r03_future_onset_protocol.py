import json
import hashlib
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "configs" / "r03_future_onset_protocol.json"


def test_future_onset_protocol_freezes_scientific_decisions() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "r03_future_onset_protocol_v1"
    assert protocol["frozen_at"] == "2026-09-03"
    assert protocol["horizons_seconds"] == [30, 120, 300]
    assert protocol["independent_evaluation_unit"] == "source_day"
    assert protocol["primary_target"]["target_id"] == "future_onset_proximity_event_count"
    assert protocol["primary_target"]["exclude_pairs_active_at_cutoff"] is True
    assert protocol["primary_target"]["horizontal_nm"] == 5
    assert protocol["primary_target"]["vertical_ft"] == 1000
    assert protocol["feature_ladder"] == {
        "A": "existing_eight_current_state_features",
        "B": "A_plus_observed_pairwise_geometry",
        "C": "B_plus_constant_velocity_CPA_features",
    }
    assert protocol["point_models"] == ["ridge_log1p", "hist_gradient_boosting_poisson"]
    assert protocol["probability_models"] == ["prevalence", "logistic_ridge"]
    assert protocol["count_models"] == ["plain_poisson"]
    assert protocol["threshold_sensitivity"]["selection_from_outcomes"] is False
    assert protocol["confirmation"]["allow_retuning"] is False
    assert protocol["external_validation"]["airport_id"] == "KAGC"
    assert protocol["external_validation"]["allow_site_tuning"] is False
    assert len(protocol["stopping_rules"]) >= 3


def test_protocol_training_artifact_is_present_and_hash_frozen() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    artifact = Path(protocol["development_dataset"]["scene_path"])

    if not artifact.is_file():
        pytest.skip("derived ADS-B scene artifact is not redistributed")
    assert artifact.is_file()
    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert actual == protocol["development_dataset"]["scene_sha256"]
