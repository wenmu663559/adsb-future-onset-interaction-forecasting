from __future__ import annotations

import torch

from airspace_complexity.r03_interaction_models import (
    DeepSetsRegressor,
    RelationRegressor,
    TokenRegressor,
)


def test_scene_models_are_permutation_invariant_and_relations_are_used() -> None:
    torch.manual_seed(1701)
    nodes = torch.tensor([[[1.0, 2.0, 0.5, 0.1],
                           [3.0, 1.0, 0.2, 0.4],
                           [2.0, 4.0, 0.9, 0.3]]])
    valid = torch.tensor([[True, True, True]])
    target = torch.tensor([0])
    relations = torch.tensor([[[0.0, 0.0, 0.0],
                               [2.0, -1.0, 0.3],
                               [1.0, 2.0, -0.2]]])
    permutation = torch.tensor([0, 2, 1])

    token = TokenRegressor(node_dim=4, hidden_dim=8, output_dim=2)
    sets = DeepSetsRegressor(node_dim=4, hidden_dim=8, output_dim=2)
    graph = RelationRegressor(
        node_dim=4, relation_dim=3, hidden_dim=8, output_dim=2
    )

    assert torch.equal(
        token(nodes, valid, target),
        token(nodes[:, permutation], valid[:, permutation], target),
    )
    assert torch.allclose(
        sets(nodes, valid, target),
        sets(nodes[:, permutation], valid[:, permutation], target),
    )
    original = graph(nodes, valid, target, relations)
    assert torch.allclose(
        original,
        graph(
            nodes[:, permutation], valid[:, permutation], target,
            relations[:, permutation],
        ),
    )
    assert not torch.allclose(
        original,
        graph(nodes, valid, target, relations[:, torch.tensor([0, 2, 1])]),
    )
