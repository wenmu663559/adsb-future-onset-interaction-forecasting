"""Small CPU models for the non-scientific R03 interaction screen."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _target_rows(nodes: Tensor, target_indices: Tensor) -> Tensor:
    batch = torch.arange(nodes.shape[0], device=nodes.device)
    return nodes[batch, target_indices]


def _neighbor_mask(valid: Tensor, target_indices: Tensor) -> Tensor:
    result = valid.clone()
    batch = torch.arange(valid.shape[0], device=valid.device)
    result[batch, target_indices] = False
    return result


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / count


class TokenRegressor(nn.Module):
    def __init__(self, *, node_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, nodes: Tensor, valid: Tensor, target_indices: Tensor) -> Tensor:
        del valid
        return self.network(_target_rows(nodes, target_indices))


class DeepSetsRegressor(nn.Module):
    def __init__(self, *, node_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, nodes: Tensor, valid: Tensor, target_indices: Tensor) -> Tensor:
        neighbors = _masked_mean(
            self.neighbor_encoder(nodes), _neighbor_mask(valid, target_indices)
        )
        return self.predictor(torch.cat([
            _target_rows(nodes, target_indices), neighbors,
        ], dim=-1))


class RelationRegressor(nn.Module):
    def __init__(
        self, *, node_dim: int, relation_dim: int, hidden_dim: int, output_dim: int,
    ) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(node_dim + relation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        nodes: Tensor,
        valid: Tensor,
        target_indices: Tensor,
        relations: Tensor,
    ) -> Tensor:
        edges = self.edge_encoder(torch.cat([nodes, relations], dim=-1))
        context = _masked_mean(edges, _neighbor_mask(valid, target_indices))
        return self.predictor(torch.cat([
            _target_rows(nodes, target_indices), context,
        ], dim=-1))
