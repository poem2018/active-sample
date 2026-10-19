import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp

from experiments.adaptive_candidate_weights import (
    effective_sample_size, ess_temperature, normalized_log_weights, recursive_update)


def test_zero_temperature_and_constant_scores():
    s = np.array([-100000., -1., 0., 4.])
    np.testing.assert_allclose(np.exp(normalized_log_weights(s, 0)), .25)
    out = ess_temperature(np.ones(16) * -10000, 8)
    assert out['beta'] == 1
    np.testing.assert_allclose(np.exp(out['log_weights']), 1/16)


def test_use_one_when_untempered_is_feasible():
    s = np.linspace(-.1, .1, 16)
    out = ess_temperature(s, 8)
    assert out['beta'] == 1 and out['search_iterations'] == 0
    np.testing.assert_allclose(out['log_weights'], s-logsumexp(s))


def test_beta_matches_independent_root_and_is_maximal():
    s = np.array([0., -1., -2., -3., -4., -6., -10., -15.,
                  -20., -30., -40., -55., -70., -90., -120., -200.])
    def ess(beta):
        p = np.exp(beta*s); p /= p.sum()
        return 1 / (p @ p)
    reference = brentq(lambda b: ess(b)-8, 0., 1., xtol=1e-14)
    out = ess_temperature(s, 8)
    assert abs(out['beta']-reference) < 2e-13
    assert ess(out['beta']) >= 8-2e-13
    assert ess(out['beta']+1e-8) < 8
    assert out['beta_upper']-out['beta'] < 1e-14
    grid = [ess(b) for b in np.linspace(0, 1, 101)]
    assert np.all(np.diff(grid) <= 1e-12)


def test_entire_previous_log_weight_is_tempered():
    old = normalized_log_weights(np.linspace(-20, 0, 16), 1)
    me = np.linspace(0, 6, 16); pe = np.linspace(2, 50, 16)**2
    out = recursive_update(old, me, pe, .1, 8)
    score = old - me - .1*pe
    expected = out['beta']*score
    np.testing.assert_allclose(out['log_weights'], expected-logsumexp(expected), atol=1e-12)
    wrong = old-out['beta']*(me+.1*pe); wrong -= logsumexp(wrong)
    assert np.max(np.abs(wrong-out['log_weights'])) > 1


def test_underflowed_old_probability_can_recover():
    old = normalized_log_weights(np.array([0.]+[-2000.]*15), 1)
    assert np.exp(old[1]) == 0
    out = recursive_update(old, np.zeros(16), np.zeros(16), 0, 8)
    assert np.isfinite(out['log_weights']).all()
    assert out['weights'][1] > .01
    assert out['ess'] >= 8-1e-12


def test_each_variant_uses_its_own_recursive_history():
    energy = np.arange(48.).reshape(3,16)
    logw = np.full(16, -np.log(16.))
    first = recursive_update(logw, energy[0], energy[1], 0, 8)
    # Noncollinear scores are needed: with collinear scores, adaptive beta can
    # exactly cancel the scale change caused by the previous log weights.
    current = energy[2][::-1] + np.linspace(0,10,16)**2
    second = recursive_update(first['log_weights'], current, energy[1], 0, 8)
    reset = recursive_update(logw, current, energy[1], 0, 8)
    assert not np.allclose(second['weights'], reset['weights'])
    np.testing.assert_allclose(second['scores'], first['log_weights']-current)


def test_adaptive_temperature_cancels_affine_score_scale():
    scores = -np.arange(16.)
    a = ess_temperature(scores, 8)
    b = ess_temperature(3*scores+5, 8)
    np.testing.assert_allclose(a['log_weights'], b['log_weights'], atol=1e-12)
    assert abs(a['beta']/3-b['beta']) < 1e-13


def test_original_lambda_and_score_offset_invariance():
    old = normalized_log_weights(np.arange(16.), 1)
    me = np.arange(16.)**2; pe = np.arange(16.)[::-1]**2
    out = recursive_update(old, me, pe)
    expected = old-me-pe; expected -= logsumexp(expected)
    np.testing.assert_allclose(out['log_weights'], expected, atol=1e-12)
    a = ess_temperature(old-me-pe, 8)
    b = ess_temperature(old-me-pe+12345., 8)
    np.testing.assert_allclose(a['log_weights'], b['log_weights'], atol=1e-12)
    assert abs(a['beta']-b['beta']) < 1e-13


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError): ess_temperature(np.zeros(16), 17)
    with pytest.raises(ValueError): ess_temperature(np.array([0., np.inf]), 1)
    with pytest.raises(ValueError): recursive_update(np.zeros(16), np.zeros(15), np.zeros(16))
    with pytest.raises(ValueError): recursive_update(np.zeros(16), -np.ones(16), np.zeros(16))
