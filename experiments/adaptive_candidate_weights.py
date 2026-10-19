"""ESS-constrained regularization of the entire recursive candidate score.

This module has no image, GT, prior, or candidate-generation inputs.
"""
import numpy as np
from scipy.special import logsumexp


def normalized_log_weights(scores, beta):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError('Expected a finite vector of candidate scores')
    if not 0 <= beta <= 1:
        raise ValueError('Inverse temperature must be in [0,1]')
    if beta == 0:
        return np.full(len(scores), -np.log(len(scores)), dtype=np.float64)
    scaled = beta * (scores - scores.max())
    return scaled - logsumexp(scaled)


def effective_sample_size(log_weights):
    return float(np.exp(-logsumexp(2 * np.asarray(log_weights, dtype=np.float64))))


def ess_temperature(scores, target_ess):
    """Largest feasible beta in [0,1], bracketed to floating-point precision.

    For Z(beta)=sum exp(beta*s), ESS=Z(beta)^2/Z(2*beta). Its log
    derivative is 2(E_beta[s]-E_2beta[s])<=0, so bisection is valid.
    """
    scores = np.asarray(scores, dtype=np.float64)
    untempered = normalized_log_weights(scores, 1.)
    if not 1 <= target_ess <= len(scores):
        raise ValueError('ESS target must be between 1 and K')
    raw_ess = effective_sample_size(untempered)
    if raw_ess >= target_ess or np.ptp(scores) == 0:
        return {'beta': 1., 'beta_upper': 1., 'log_weights': untempered,
                'ess_untempered': raw_ess, 'search_iterations': 0}
    if target_ess == len(scores):
        return {'beta': 0., 'beta_upper': 0.,
                'log_weights': normalized_log_weights(scores, 0.),
                'ess_untempered': raw_ess, 'search_iterations': 0}
    lo, hi = 0., 1.
    for iteration in range(1, 81):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if effective_sample_size(normalized_log_weights(scores, mid)) >= target_ess:
            lo = mid
        else:
            hi = mid
    return {'beta': lo, 'beta_upper': hi,
            'log_weights': normalized_log_weights(scores, lo),
            'ess_untempered': raw_ess, 'search_iterations': iteration}


def recursive_update(previous_log_weights, measurement_energy, proximity_energy,
                     lambda_w=1., target_ess=None):
    old, me, pe = [np.asarray(a, dtype=np.float64)
                   for a in (previous_log_weights, measurement_energy, proximity_energy)]
    if old.ndim != 1 or me.shape != old.shape or pe.shape != old.shape:
        raise ValueError('Candidate energies and log weights must have identical vector shapes')
    if not all(np.isfinite(a).all() for a in (old, me, pe)):
        raise ValueError('Finite log weights and energies required')
    if lambda_w < 0 or not np.isfinite(lambda_w) or np.any(me < 0) or np.any(pe < 0):
        raise ValueError('Nonnegative finite weighting coefficient and energies required')
    scores = old - me - lambda_w * pe
    if target_ess is None:
        logw = normalized_log_weights(scores, 1.)
        result = {'beta': 1., 'beta_upper': 1., 'log_weights': logw,
                  'ess_untempered': effective_sample_size(logw), 'search_iterations': 0}
    else:
        result = ess_temperature(scores, target_ess)
    result.update(scores=scores, weights=np.exp(result['log_weights']),
                  ess=effective_sample_size(result['log_weights']))
    return result
