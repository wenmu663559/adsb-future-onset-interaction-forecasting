import numpy as np
import pytest

from airspace_complexity.r03_probabilistic_multihorizon import (
    fit_nb2_regression,
    nb2_log_loss,
    predict_nb2_mean,
)


def test_nb2_regression_learns_a_deterministic_positive_conditional_mean() -> None:
    x = np.asarray(
        [[-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0]],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 1, 0, 3, 5, 8, 13], dtype=np.float64)

    first = fit_nb2_regression(x, y, l2=0.1)
    second = fit_nb2_regression(x, y, l2=0.1)
    prediction = predict_nb2_mean(first, x)

    assert first["converged"] is True
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0.0)
    assert prediction[-1] > prediction[0]
    np.testing.assert_allclose(first["coefficients"], second["coefficients"], atol=1e-10)
    np.testing.assert_allclose(prediction, predict_nb2_mean(second, x), atol=1e-10)


def test_nb2_regression_beats_an_intercept_only_nb2_fit_when_feature_is_informative() -> None:
    x = np.asarray(
        [[-2.0], [-1.5], [-1.0], [-0.5], [0.5], [1.0], [1.5], [2.0]],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 1, 0, 3, 5, 8, 13], dtype=np.float64)

    conditional = fit_nb2_regression(x, y, l2=0.1)
    intercept_only = fit_nb2_regression(np.zeros((len(y), 0)), y, l2=0.1)
    conditional_nll = nb2_log_loss(
        y, predict_nb2_mean(conditional, x), conditional["dispersion"]
    ).mean()
    intercept_nll = nb2_log_loss(
        y,
        predict_nb2_mean(intercept_only, np.zeros((len(y), 0))),
        intercept_only["dispersion"],
    ).mean()

    assert conditional_nll < intercept_nll


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.ones((2, 1)), np.asarray([1.0])),
        (np.ones((2, 1)), np.asarray([1.0, -1.0])),
        (np.asarray([[1.0], [np.nan]]), np.asarray([1.0, 2.0])),
        (np.ones((2, 1)), np.asarray([1.0, 1.5])),
    ],
)
def test_nb2_regression_rejects_misaligned_nonfinite_or_noninteger_counts(x, y) -> None:
    with pytest.raises(ValueError):
        fit_nb2_regression(x, y)
