"""Frozen relation-model loading and held-out task-indicator probes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def reverse_gradient(value):
    """Preserve values in the forward pass and reverse their gradients."""

    return _GradientReverse.apply(value)


class InvariantRelation(nn.Module):
    """Checkpoint-compatible invariant pair-relation network."""

    def __init__(self, feature_mean, feature_scale):
        super().__init__()
        self.register_buffer(
            "feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32).clone()
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(feature_scale, dtype=torch.float32).clone()
        )
        self.edge = nn.Sequential(
            nn.Linear(6, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(65, 48), nn.ReLU(), nn.Linear(48, 1))
        self.domain_head = nn.Sequential(
            nn.Linear(65, 24), nn.ReLU(), nn.Linear(24, 2)
        )

    def pair_inputs(self, value):
        raw = value[:, :, :, :6] * self.feature_scale + self.feature_mean
        current = raw[:, :, -1]
        previous_index = max(0, value.shape[2] - 4)
        previous = raw[:, :, previous_index]
        mask = value[:, :, -1, 6] > 0
        previous_mask = value[:, :, previous_index, 6] > 0
        delta = current[:, :, None, :] - current[:, None, :, :]
        horizontal = torch.sqrt((delta[..., :2] ** 2).sum(-1) + 1e-6)
        old_delta = previous[:, :, None, :2] - previous[:, None, :, :2]
        old_horizontal = torch.sqrt((old_delta ** 2).sum(-1) + 1e-6)
        heading_similarity = (
            current[:, :, None, 4] * current[:, None, :, 4]
            + current[:, :, None, 5] * current[:, None, :, 5]
        )
        pair = torch.stack(
            (
                torch.log1p(horizontal) / 10.0,
                torch.log1p(delta[..., 2].abs()) / 8.0,
                delta[..., 3].abs() / 100.0,
                heading_similarity,
                (torch.log1p(old_horizontal) - torch.log1p(horizontal)) / 3.0,
                (
                    (previous[:, :, None, 2] - previous[:, None, :, 2]).abs()
                    - delta[..., 2].abs()
                )
                / 1000.0,
            ),
            dim=-1,
        )
        nodes = value.shape[1]
        off_diagonal = ~torch.eye(
            nodes, dtype=torch.bool, device=value.device
        )[None]
        valid = mask[:, :, None] & mask[:, None, :] & off_diagonal
        valid = valid & (previous_mask[:, :, None] & previous_mask[:, None, :])
        return pair, valid, mask

    def aggregate_edges(self, encoded, valid, mask):
        weights = valid.float().unsqueeze(-1)
        mean_edge = (encoded * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)
        pair_count = valid.sum((1, 2))
        max_edge = encoded.masked_fill(~valid.unsqueeze(-1), -1e9).amax((1, 2))
        max_edge = torch.where(
            pair_count.unsqueeze(-1) > 0,
            max_edge,
            torch.zeros_like(max_edge),
        )
        count = torch.log1p(mask.float().sum(1)).unsqueeze(-1) / 5.0
        return torch.cat((mean_edge, max_edge, count), dim=-1)

    def forward(self, value):
        pair, valid, mask = self.pair_inputs(value)
        encoded = self.edge(pair)
        representation = self.aggregate_edges(encoded, valid, mask)
        return self.head(representation).squeeze(-1), representation

    def domain_logits(self, representation):
        return self.domain_head(reverse_gradient(representation))


def load_invariant_relation_checkpoint(path: Path):
    """Load one frozen checkpoint without updating any parameter."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model") != "invariant_relation":
        raise ValueError("checkpoint is not an invariant_relation model")
    state = checkpoint["state_dict"]
    model = InvariantRelation(state["feature_mean"], state["feature_scale"])
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, metadata


def _subset_metrics(y_true, y_pred, days):
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    daily = []
    for day in sorted(set(days.tolist())):
        selected = days == day
        if selected.sum() < 2 or np.ptp(y_true[selected]) == 0 or np.ptp(y_pred[selected]) == 0:
            continue
        daily.append(float(spearmanr(y_true[selected], y_pred[selected]).statistic))
    correlation = None
    if len(y_true) >= 2 and np.ptp(y_true) > 0 and np.ptp(y_pred) > 0:
        correlation = float(spearmanr(y_true, y_pred).statistic)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": correlation,
        "daily_spearman_median": float(np.median(daily)) if daily else None,
        "evaluable_days": len(daily),
    }


def _day_count_residual_correlation(y_true, y_pred, days, counts):
    from scipy.stats import pearsonr

    target_residuals = np.zeros(len(y_true), dtype=float)
    prediction_residuals = np.zeros(len(y_true), dtype=float)
    for day in sorted(set(days.tolist())):
        for count in sorted(set(counts[days == day].tolist())):
            selected = (days == day) & (counts == count)
            target_residuals[selected] = y_true[selected] - y_true[selected].mean()
            prediction_residuals[selected] = y_pred[selected] - y_pred[selected].mean()
    if np.ptp(target_residuals) == 0 or np.ptp(prediction_residuals) == 0:
        return None
    return float(pearsonr(target_residuals, prediction_residuals).statistic)


def fit_frozen_representation_probes(
    *,
    representations: Mapping[int, np.ndarray],
    predictions: Mapping[int, np.ndarray],
    indicators: Mapping[str, np.ndarray],
    current_aircraft_count: np.ndarray,
    split: np.ndarray,
    days: np.ndarray,
) -> dict[str, object]:
    """Fit fixed linear probes on train rows and evaluate held-out rows."""

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = split == "train"
    subsets = tuple(name for name in ("validation", "test") if np.any(split == name))
    count_feature = np.log1p(np.asarray(current_aircraft_count, dtype=float))[:, None]
    result: dict[str, object] = {"probe": "StandardScaler + Ridge(alpha=1.0)", "indicators": {}}
    for indicator_name, raw_target in indicators.items():
        target = np.asarray(raw_target, dtype=float)
        indicator_result = {"seeds": {}}
        count_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
            count_feature[train], target[train]
        )
        count_prediction = count_model.predict(count_feature)
        for seed in sorted(representations):
            representation = np.asarray(representations[seed], dtype=float)
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                representation[train], target[train]
            )
            representation_prediction = model.predict(representation)
            edge_only = representation[:, :-1]
            edge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                edge_only[train], target[train]
            )
            edge_prediction = edge_model.predict(edge_only)
            seed_result = {}
            for subset_name in subsets:
                selected = split == subset_name
                representation_metrics = _subset_metrics(
                    target[selected], representation_prediction[selected], days[selected]
                )
                count_metrics = _subset_metrics(
                    target[selected], count_prediction[selected], days[selected]
                )
                edge_metrics = _subset_metrics(
                    target[selected], edge_prediction[selected], days[selected]
                )
                prediction_metrics = _subset_metrics(
                    target[selected], np.asarray(predictions[seed])[selected], days[selected]
                )
                seed_result[subset_name] = {
                    "scene_count": int(selected.sum()),
                    "representation_r2": representation_metrics["r2"],
                    "representation_mae": representation_metrics["mae"],
                    "representation_spearman": representation_metrics["spearman"],
                    "representation_daily_spearman_median": representation_metrics[
                        "daily_spearman_median"
                    ],
                    "count_r2": count_metrics["r2"],
                    "count_mae": count_metrics["mae"],
                    "edge_only_r2": edge_metrics["r2"],
                    "edge_only_mae": edge_metrics["mae"],
                    "representation_minus_count_r2": (
                        representation_metrics["r2"] - count_metrics["r2"]
                    ),
                    "edge_only_minus_count_r2": (
                        edge_metrics["r2"] - count_metrics["r2"]
                    ),
                    "prediction_spearman": prediction_metrics["spearman"],
                    "prediction_day_count_residual_correlation": (
                        _day_count_residual_correlation(
                            target[selected],
                            np.asarray(predictions[seed])[selected],
                            days[selected],
                            np.asarray(current_aircraft_count)[selected],
                        )
                    ),
                }
            indicator_result["seeds"][str(seed)] = seed_result
        result["indicators"][indicator_name] = indicator_result
    return result
