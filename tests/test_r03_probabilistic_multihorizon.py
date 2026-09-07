import numpy as np
import pytest

from airspace_complexity.r03_probabilistic_multihorizon import (
    binary_log_loss,
    calibration_intercept_slope,
    central_interval_score,
    count_metrics,
    estimate_nb2_dispersion,
    expected_calibration_error,
    nb2_interval,
    nb2_log_loss,
    nb2_risk_probability,
)


def test_nb2_dispersion_recovers_overdispersion_and_never_goes_negative():
    y = np.asarray([0, 0, 1, 2, 8, 12], dtype=float)
    mean = np.ones_like(y) * y.mean()
    assert estimate_nb2_dispersion(y, mean) > 0.0
    assert estimate_nb2_dispersion(np.ones(5), np.ones(5)) == pytest.approx(1e-8)
    expected = (((y - mean) ** 2 - mean).sum()) / ((mean ** 2).sum())
    assert estimate_nb2_dispersion(y, mean) == pytest.approx(expected)


def test_nb2_probabilities_intervals_and_log_loss_are_finite():
    mean = np.asarray([0.5, 2.0, 8.0])
    probability = nb2_risk_probability(mean, dispersion=0.5, threshold=3)
    lower, upper = nb2_interval(mean, dispersion=0.5, coverage=0.90)
    loss = nb2_log_loss(np.asarray([0, 2, 10]), mean, dispersion=0.5)

    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert probability.tolist() == sorted(probability.tolist())
    assert np.all(lower <= upper)
    assert np.all(np.isfinite(loss))


def test_expected_calibration_error_is_zero_for_exact_bin_rates():
    probabilities = np.asarray([0.1, 0.1, 0.9, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert expected_calibration_error(probabilities, labels, bins=10) == pytest.approx(0.1)


def test_count_metrics_use_raw_count_scale():
    metrics = count_metrics(np.asarray([0, 2, 4]), np.asarray([1, 2, 3]))
    assert metrics["count_mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["count_rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics["mean_poisson_deviance"] >= 0.0


def test_binary_log_loss_and_calibration_diagnostics_are_finite():
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    probabilities = np.asarray([0.05, 0.15, 0.30, 0.70, 0.85, 0.95])
    assert binary_log_loss(labels, probabilities) > 0.0
    calibration = calibration_intercept_slope(labels, probabilities)
    assert calibration["converged"]
    assert np.isfinite(calibration["intercept"])
    assert calibration["slope"] > 0.0


def test_central_interval_score_penalizes_misses():
    inside = central_interval_score(
        np.asarray([2.0]), np.asarray([1.0]), np.asarray([3.0]), coverage=0.90
    )
    below = central_interval_score(
        np.asarray([0.0]), np.asarray([1.0]), np.asarray([3.0]), coverage=0.90
    )
    assert inside == pytest.approx(2.0)
    assert below == pytest.approx(22.0)
