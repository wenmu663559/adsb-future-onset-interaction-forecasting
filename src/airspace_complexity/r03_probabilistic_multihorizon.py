"""Metrics and NB2 uncertainty helpers for frozen multi-horizon experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson


MIN_MEAN = 1e-8
MIN_DISPERSION = 1e-8


def fit_nb2_regression(x, y_true, *, l2: float = 1.0) -> dict[str, object]:
    """Fit a standardized NB2 regression with a log link.

    The conditional mean coefficients and dispersion are optimized jointly.
    Only non-intercept coefficients receive the L2 penalty.
    """

    from sklearn.linear_model import PoissonRegressor

    features = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if (
        features.ndim != 2
        or y.ndim != 1
        or len(y) == 0
        or len(features) != len(y)
        or not np.isfinite(features).all()
        or not np.isfinite(y).all()
        or np.any(y < 0.0)
        or not np.equal(y, np.floor(y)).all()
        or not np.isfinite(l2)
        or l2 < 0.0
    ):
        raise ValueError("features and nonnegative integer counts must be finite and aligned")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (features - mean) / scale
    if standardized.shape[1]:
        initializer = PoissonRegressor(alpha=l2, max_iter=2000).fit(standardized, y)
        intercept0 = float(initializer.intercept_)
        coefficients0 = np.asarray(initializer.coef_, dtype=np.float64)
        initial_mean = np.maximum(initializer.predict(standardized), MIN_MEAN)
    else:
        intercept0 = float(np.log(max(float(y.mean()), MIN_MEAN)))
        coefficients0 = np.empty(0, dtype=np.float64)
        initial_mean = np.full(len(y), np.exp(intercept0), dtype=np.float64)
    dispersion0 = estimate_nb2_dispersion(y, initial_mean)
    parameters0 = np.concatenate(
        ([intercept0], coefficients0, [np.log(dispersion0)])
    )

    def objective(parameters) -> float:
        intercept = float(parameters[0])
        coefficients = np.asarray(parameters[1:-1], dtype=np.float64)
        dispersion = float(np.exp(parameters[-1]))
        linear = np.clip(intercept + standardized @ coefficients, -20.0, 20.0)
        conditional_mean = np.exp(linear)
        penalty = 0.5 * float(l2) * float(coefficients @ coefficients)
        return float(nb2_log_loss(y, conditional_mean, dispersion).mean() + penalty)

    fitted = minimize(
        objective,
        x0=parameters0,
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * (len(parameters0) - 1) + [(-18.0, 6.0)],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    parameters = np.asarray(fitted.x, dtype=np.float64)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "intercept": float(parameters[0]),
        "coefficients": parameters[1:-1],
        "dispersion": float(np.exp(parameters[-1])),
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "objective": float(fitted.fun),
        "message": str(fitted.message),
        "l2": float(l2),
    }


def predict_nb2_mean(model: dict[str, object], x) -> np.ndarray:
    """Predict positive conditional means from a fitted NB2 regression."""

    features = np.asarray(x, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != len(coefficients)
        or mean.shape != coefficients.shape
        or scale.shape != coefficients.shape
        or not np.isfinite(features).all()
    ):
        raise ValueError("prediction features do not match the fitted NB2 model")
    standardized = (features - mean) / scale
    linear = np.clip(
        float(model["intercept"]) + standardized @ coefficients, -20.0, 20.0
    )
    return np.maximum(np.exp(linear), MIN_MEAN)


def estimate_nb2_dispersion(y_true, mean_prediction) -> float:
    """Estimate Var(Y)=mu+alpha*mu^2 by training-only method of moments."""

    y = np.asarray(y_true, dtype=np.float64)
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), MIN_MEAN)
    if y.shape != mean.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("y_true and mean_prediction must be aligned nonempty vectors")
    numerator = float(np.sum((y - mean) ** 2 - mean))
    denominator = float(np.sum(mean ** 2))
    return max(MIN_DISPERSION, numerator / denominator if denominator else MIN_DISPERSION)


def nb2_log_loss(y_true, mean_prediction, dispersion: float) -> np.ndarray:
    """Return per-row negative log likelihood under an NB2 distribution."""

    y = np.asarray(y_true, dtype=np.float64)
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), MIN_MEAN)
    if y.shape != mean.shape or np.any(y < 0):
        raise ValueError("counts and means must be aligned and nonnegative")
    alpha = max(float(dispersion), MIN_DISPERSION)
    if alpha <= 1e-7:
        log_probability = y * np.log(mean) - mean - gammaln(y + 1.0)
    else:
        size = 1.0 / alpha
        probability = size / (size + mean)
        log_probability = (
            gammaln(y + size)
            - gammaln(size)
            - gammaln(y + 1.0)
            + size * np.log(probability)
            + y * np.log1p(-probability)
        )
    return -log_probability


def nb2_risk_probability(mean_prediction, *, dispersion: float, threshold: int):
    """Probability that a count is at least the frozen threshold."""

    if threshold < 1:
        raise ValueError("threshold must be positive")
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), MIN_MEAN)
    alpha = max(float(dispersion), MIN_DISPERSION)
    if alpha <= 1e-7:
        values = poisson.sf(threshold - 1, mean)
    else:
        size = 1.0 / alpha
        probability = size / (size + mean)
        values = nbinom.sf(threshold - 1, size, probability)
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def nb2_interval(mean_prediction, *, dispersion: float, coverage: float = 0.90):
    """Central integer prediction interval under NB2."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), MIN_MEAN)
    alpha = max(float(dispersion), MIN_DISPERSION)
    tail = (1.0 - coverage) / 2.0
    if alpha <= 1e-7:
        lower = poisson.ppf(tail, mean)
        upper = poisson.ppf(1.0 - tail, mean)
    else:
        size = 1.0 / alpha
        probability = size / (size + mean)
        lower = nbinom.ppf(tail, size, probability)
        upper = nbinom.ppf(1.0 - tail, size, probability)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def expected_calibration_error(probabilities, labels, *, bins: int = 10) -> float:
    """Equal-width expected calibration error."""

    probability = np.asarray(probabilities, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    if probability.shape != label.shape or probability.ndim != 1 or len(label) == 0:
        raise ValueError("probabilities and labels must be aligned nonempty vectors")
    if bins <= 0 or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("invalid bins or probabilities")
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(label[selected].mean())
            )
    return error


def binary_log_loss(labels, probabilities) -> float:
    """Mean Bernoulli negative log likelihood with numerical clipping."""

    label = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != label.shape or label.ndim != 1 or len(label) == 0:
        raise ValueError("probabilities and labels must be aligned nonempty vectors")
    if np.any((label != 0.0) & (label != 1.0)):
        raise ValueError("labels must be binary")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be between zero and one")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(label * np.log(clipped) + (1.0 - label) * np.log1p(-clipped))
    )


def calibration_intercept_slope(labels, probabilities) -> dict[str, float | bool]:
    """Estimate pooled logistic calibration intercept and slope.

    This is a descriptive diagnostic. It does not make overlapping scene rows
    independent and should therefore be interpreted alongside day-level results.
    """

    label = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != label.shape or label.ndim != 1 or len(label) == 0:
        raise ValueError("probabilities and labels must be aligned nonempty vectors")
    if np.any((label != 0.0) & (label != 1.0)) or len(np.unique(label)) != 2:
        raise ValueError("both binary label classes are required")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be between zero and one")
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    logit = np.log(clipped) - np.log1p(-clipped)

    def objective(parameters):
        linear = parameters[0] + parameters[1] * logit
        return float(np.mean(np.logaddexp(0.0, linear) - label * linear))

    fitted = minimize(
        objective,
        x0=np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=((-20.0, 20.0), (-10.0, 10.0)),
    )
    return {
        "intercept": float(fitted.x[0]),
        "slope": float(fitted.x[1]),
        "converged": bool(fitted.success),
    }


def central_interval_score(y_true, lower, upper, *, coverage: float = 0.90) -> float:
    """Mean proper interval score for a central prediction interval."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    y = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if y.shape != low.shape or y.shape != high.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("targets and interval bounds must be aligned nonempty vectors")
    if np.any(low > high):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    alpha = 1.0 - coverage
    score = (
        high
        - low
        + (2.0 / alpha) * (low - y) * (y < low)
        + (2.0 / alpha) * (y - high) * (y > high)
    )
    return float(np.mean(score))


def count_metrics(y_true, mean_prediction) -> dict[str, float]:
    """Point metrics on the original count scale."""

    y = np.asarray(y_true, dtype=np.float64)
    mean = np.maximum(np.asarray(mean_prediction, dtype=np.float64), MIN_MEAN)
    if y.shape != mean.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("counts and predictions must be aligned nonempty vectors")
    error = mean - y
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance_term = np.where(y > 0.0, y * np.log(y / mean), 0.0)
    return {
        "count_mae": float(np.mean(np.abs(error))),
        "count_rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_poisson_deviance": float(2.0 * np.mean(deviance_term - (y - mean))),
    }
