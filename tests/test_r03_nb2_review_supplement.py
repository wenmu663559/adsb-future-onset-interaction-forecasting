import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.run_r03_nb2_review_supplement import (
    _dataset_decision,
    _frozen_thresholds,
    _group_comparison,
    _verify_dataset_hashes,
)


def test_frozen_thresholds_use_training_counts_only_and_all_protocol_quantiles() -> None:
    train = np.asarray([0, 0, 1, 2, 3, 5, 8, 13], dtype=np.float64)

    thresholds = _frozen_thresholds(train, [0.7, 0.75, 0.8, 0.9])

    assert thresholds == {"0.70": 5, "0.75": 8, "0.80": 8, "0.90": 13}


def test_group_comparison_uses_equal_group_weight_and_counts_strict_wins() -> None:
    groups = np.asarray(["large", "large", "large", "small"])
    truth = np.asarray([0.0, 0.0, 0.0, 0.0])
    candidate = np.asarray([0.0, 0.0, 0.0, 2.0])
    reference = np.asarray([1.0, 1.0, 1.0, 1.0])

    result = _group_comparison(groups, truth, candidate, reference, metric="mae")

    assert result["candidate_equal_group_mean"] == 1.0
    assert result["reference_equal_group_mean"] == 1.0
    assert result["candidate_wins"] == 1
    assert result["reference_wins"] == 1


def test_dataset_decision_requires_both_endpoints_at_two_horizons() -> None:
    rows = {
        "30": {"count_distribution_pass": True, "q75_exceedance_pass": True},
        "120": {"count_distribution_pass": True, "q75_exceedance_pass": True},
        "300": {"count_distribution_pass": True, "q75_exceedance_pass": False},
    }

    result = _dataset_decision(rows, minimum_horizons=2)

    assert result == {
        "passed_horizons": [30, 120],
        "passed_horizon_count": 2,
        "supports_dataset_specific_distributional_value": True,
    }


def test_dataset_hash_verification_rejects_changed_scene_file(tmp_path: Path) -> None:
    scene = tmp_path / "pilot_scenes.jsonl"
    scene.write_text("original\n", encoding="utf-8")
    expected = hashlib.sha256(scene.read_bytes()).hexdigest()
    records = {"development": {"relative_path": scene.name, "sha256": expected}}

    _verify_dataset_hashes(tmp_path, records)
    scene.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_dataset_hashes(tmp_path, records)
