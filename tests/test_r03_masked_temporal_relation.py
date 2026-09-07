from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airspace_complexity.r03_masked_temporal_relation import (
    MaskedTemporalRelation,
    load_masked_temporal_relation_checkpoint,
)
from airspace_complexity.r03_masked_temporal_ab import decide_masked_temporal_ab


def _model() -> MaskedTemporalRelation:
    torch.manual_seed(7)
    return MaskedTemporalRelation(
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
    )


def _values() -> torch.Tensor:
    values = torch.zeros((1, 3, 6, 7), dtype=torch.float32)
    values[:, :2, :, 6] = 1.0
    values[:, 0, :, 0] = torch.arange(6, dtype=torch.float32)
    values[:, 1, :, 0] = torch.arange(6, dtype=torch.float32).flip(0)
    return values


def test_masked_temporal_relation_returns_prediction_and_joint_representation() -> None:
    output = _model()(_values(), mask_probability=0.2, deterministic=True)

    assert output["prediction"].shape == (1,)
    assert output["representation"].shape == (1, 97)
    assert output["reconstruction"].shape == (1, 3, 6, 6)
    assert output["selected"].shape == (1, 3, 6)
    assert output["selected"].sum().item() == 3
    assert not output["selected"][:, 2].any()


def test_temporal_encoder_changes_when_only_history_order_changes() -> None:
    model = _model().eval()
    forward = _values()
    reversed_history = forward.clone()
    reversed_history[:, :, :-1] = forward[:, :, :-1].flip(2)

    with torch.no_grad():
        forward_temporal = model.encode_temporal(forward)
        reversed_temporal = model.encode_temporal(reversed_history)

    assert not torch.equal(forward_temporal, reversed_temporal)


def test_masked_temporal_checkpoint_loader_reproduces_predictions(tmp_path: Path) -> None:
    original = _model().eval()
    with torch.no_grad():
        expected = original(_values())["prediction"]
    checkpoint_path = tmp_path / "masked_temporal_relation__seed-7.pt"
    torch.save(
        {
            "model": "masked_temporal_relation",
            "seed": 7,
            "state_dict": original.state_dict(),
            "target": "log1p(120_second_proximity_event_count)",
        },
        checkpoint_path,
    )

    loaded, metadata = load_masked_temporal_relation_checkpoint(checkpoint_path)
    with torch.no_grad():
        actual = loaded(_values())["prediction"]

    assert metadata["seed"] == 7
    assert loaded.training is False
    assert torch.equal(actual, expected)


def test_joint_prediction_loss_updates_relation_and_temporal_encoders() -> None:
    model = _model().train()
    output = model(_values(), mask_probability=0.2, deterministic=True)
    reconstruction_loss = torch.nn.functional.mse_loss(
        output["reconstruction"][output["selected"]],
        output["reconstruction_target"][output["selected"]],
    )
    loss = output["prediction"].square().mean() + 0.1 * reconstruction_loss

    loss.backward()

    assert model.edge[0].weight.grad.abs().sum().item() > 0
    assert model.temporal_gru.weight_ih_l0.grad.abs().sum().item() > 0


def test_masked_temporal_ab_cli_has_fixed_comparison_inputs() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_r03_masked_temporal_ab.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset-root" in completed.stdout
    assert "--baseline-checkpoint" in completed.stdout
    assert "--output-root" in completed.stdout
    assert "--seeds" in completed.stdout
    assert "--epochs" in completed.stdout
    assert "--prepare" not in completed.stdout


def test_ab_decision_requires_prediction_and_secondary_stability() -> None:
    promoted = decide_masked_temporal_ab(
        baseline_mae_by_seed={17: 2.0, 29: 2.0, 43: 2.0},
        candidate_mae_by_seed={17: 1.8, 29: 1.9, 43: 2.1},
        day_mae_delta={"d1": -0.2, "d2": -0.1, "d3": -0.1, "d4": 0.1},
        closing_edge_delta_change=0.03,
        maneuver_representation_delta_change=0.01,
    )
    rejected = decide_masked_temporal_ab(
        baseline_mae_by_seed={17: 2.0, 29: 2.0, 43: 2.0},
        candidate_mae_by_seed={17: 1.5, 29: 2.1, 43: 2.1},
        day_mae_delta={"d1": -0.2, "d2": 0.1, "d3": 0.1, "d4": 0.1},
        closing_edge_delta_change=0.03,
        maneuver_representation_delta_change=0.01,
    )

    assert promoted["decision"] == "PROMOTE_B"
    assert promoted["passed_gates"] == 4
    assert rejected["decision"] == "KEEP_A"
    assert rejected["passed_gates"] == 2
