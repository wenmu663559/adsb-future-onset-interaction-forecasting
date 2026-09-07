import importlib.util

import numpy as np
import pytest


def api():
    spec = importlib.util.find_spec('scripts.analyze_r03_review_statistics')
    assert spec is not None, 'review statistics implementation is missing'
    from scripts import analyze_r03_review_statistics
    return analyze_r03_review_statistics


def test_all_negative_groups_have_exact_sign_probability_and_constant_ci():
    result = api().paired_summary([-1.] * 8, repetitions=1000, seed=1)
    assert result['mean_difference'] == -1.
    assert result['ci95'] == [-1., -1.]
    assert result['sign_p_two_sided'] == pytest.approx(0.0078125)


def test_sign_test_excludes_ties_and_is_two_sided():
    result = api().paired_summary([-2., -1., 0., 3.], repetitions=1000, seed=1)
    assert result['mean_difference'] == 0.
    assert result['ties'] == 1
    assert result['sign_p_two_sided'] == 1.
    assert result['ci95'][0] < 0 < result['ci95'][1]


def test_holm_preserves_original_order_and_monotonic_adjustment():
    assert api().holm([.03, .01, .2]) == pytest.approx([.06, .03, .2])


def test_invalid_or_empty_differences_are_rejected():
    with pytest.raises(ValueError):
        api().paired_summary([], repetitions=100, seed=1)
    with pytest.raises(ValueError):
        api().paired_summary([float('nan')], repetitions=100, seed=1)


def test_bootstrap_uses_groups_equally_and_is_reproducible():
    a = api().paired_summary([-1., 3.], repetitions=20000, seed=77)
    b = api().paired_summary([-1., 3.], repetitions=20000, seed=77)
    assert a == b
    assert a['mean_difference'] == 1.
    assert a['ci95'] == [-1., 3.]
