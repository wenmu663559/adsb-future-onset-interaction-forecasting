"""Joint masked-temporal and invariant-relation model for the bounded R03 A/B test."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from airspace_complexity.r03_frozen_probe import (
    InvariantRelation,
    reverse_gradient,
)


class MaskedTemporalRelation(InvariantRelation):
    """Combine frozen-design pair geometry with a causal temporal GRU branch."""

    def __init__(self, feature_mean, feature_scale):
        super().__init__(feature_mean, feature_scale)
        self.temporal_gru = nn.GRU(7, 32, batch_first=True)
        self.temporal_decoder = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 6)
        )
        self.head = nn.Sequential(nn.Linear(97, 48), nn.ReLU(), nn.Linear(48, 1))
        self.domain_head = nn.Sequential(
            nn.Linear(97, 24), nn.ReLU(), nn.Linear(24, 2)
        )

    @staticmethod
    def _selected_tokens(observed, probability: float, deterministic: bool):
        if probability <= 0.0:
            return torch.zeros_like(observed)
        if deterministic:
            nodes, steps = observed.shape[1:]
            token = torch.arange(nodes * steps, device=observed.device).reshape(
                1, nodes, steps
            )
            period = max(1, round(1.0 / probability))
            selected = (token % period) == 0
            return selected.expand_as(observed) & observed
        return (torch.rand(observed.shape, device=observed.device) < probability) & observed

    def _temporal_outputs(self, value, selected=None):
        batch, nodes, steps, features = value.shape
        temporal_input = value.clone()
        if selected is not None:
            temporal_input[selected] = 0.0
        sequence = temporal_input.reshape(batch * nodes, steps, features)
        encoded_sequence, hidden = self.temporal_gru(sequence)
        node_representation = hidden[-1].reshape(batch, nodes, 32)
        node_present = (value[:, :, :, 6].sum(2) > 0).float().unsqueeze(-1)
        temporal_representation = (
            (node_representation * node_present).sum(1)
            / node_present.sum(1).clamp_min(1)
        )
        reconstruction = self.temporal_decoder(encoded_sequence).reshape(
            batch, nodes, steps, 6
        )
        return temporal_representation, reconstruction

    def encode_temporal(self, value):
        """Return the causal temporal scene representation without masking."""

        return self._temporal_outputs(value)[0]

    def forward(self, value, mask_probability: float = 0.0, deterministic: bool = False):
        observed = value[:, :, :, 6] > 0
        selected = self._selected_tokens(observed, mask_probability, deterministic)
        temporal_representation, reconstruction = self._temporal_outputs(
            value, selected
        )
        pair, valid, node_mask = self.pair_inputs(value)
        relation_representation = self.aggregate_edges(
            self.edge(pair), valid, node_mask
        )
        representation = torch.cat(
            (
                relation_representation[:, :-1],
                temporal_representation,
                relation_representation[:, -1:],
            ),
            dim=-1,
        )
        return {
            "prediction": self.head(representation).squeeze(-1),
            "representation": representation,
            "reconstruction": reconstruction,
            "reconstruction_target": value[:, :, :, :6],
            "selected": selected,
        }

    def domain_logits(self, representation):
        return self.domain_head(reverse_gradient(representation))


def load_masked_temporal_relation_checkpoint(path: Path):
    """Load one frozen masked-temporal checkpoint without updating parameters."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model") != "masked_temporal_relation":
        raise ValueError("checkpoint is not a masked_temporal_relation model")
    state = checkpoint["state_dict"]
    model = MaskedTemporalRelation(state["feature_mean"], state["feature_scale"])
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, metadata
