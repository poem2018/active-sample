"""Clean-space FlowDPS adaptation with native old-candidate guidance and Eq.(8) scoring."""
import math
import numpy as np
from scipy.special import logsumexp
import torch
from .flowdps_pixel import decompose
from .official_face_prior import NATIVE_EPS
from .official_face_reuse import ObservationFlow, downsample


class CandidateEnergy(ObservationFlow):
    """Summed acquired likelihood plus a native-image Gaussian proximity energy."""
    def __init__(self, template, ids, y, sigma=.05, old=None, tau=1., factor=4):
        super().__init__(template, ids, y, sigma, factor)
        if tau <= 0 or (old is not None and old.shape != template.shape):
            raise ValueError('Invalid corresponding old candidates or tau')
        self.old = old
        self.lam = 0. if old is None else 1 / tau**2
        self.lipschitz = self.rho + self.lam

    def proximity_gradient(self, x):
        return torch.zeros_like(x) if self.old is None else self.lam * (x - self.old)

    def total_gradient(self, x):
        gradient = self.gradient(x)
        return gradient if self.old is None else gradient + self.proximity_gradient(x)


def clean_update(clean, energy, iterations=3, step_fraction=.5):
    if iterations < 0 or not 0 <= step_fraction <= 1:
        raise ValueError('Invalid clean-space optimizer')
    x = clean.clone()
    for _ in range(iterations):
        x = x - (step_fraction / energy.lipschitz) * energy.total_gradient(x)
    return x


@torch.no_grad()
def sample(prior, epsilon, history_ids, history_y, old=None, sigma=.05, tau=1.,
           steps=128, factor=4, generator=None, diagnostics=True):
    """Fresh Gaussian start, all acquired measurements, corresponding old candidate.

    No GT or weights enter this fixed-history sampler. Extra-noise generator is
    reset per round to the same state for sequential and scratch comparisons.
    """
    if steps < 1 or generator is None:
        raise ValueError('Positive steps and an explicit generator are required')
    if history_ids.ndim != 2 or history_ids.shape != history_y.shape or len(history_ids) < 1:
        raise ValueError('Invalid acquired history')
    ids, y = history_ids.flatten(), history_y.flatten()
    energy = CandidateEnergy(epsilon, ids, y, sigma, old, tau, factor)
    z = epsilon.clone()
    h = (1 - NATIVE_EPS) / steps
    points = {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}
    trace = {k: [] for k in ['times', 'states', 'clean_before', 'clean_guided',
                             'measurement_l2', 'proximity_l2', 'prior_l2',
                             'observed_rmse_before', 'observed_rmse_after']}
    rng_states = []
    for j in range(steps):
        s = NATIVE_EPS + j * h
        snext = 1. if j == steps - 1 else NATIVE_EPS + (j + 1) * h
        velocity = prior(z, s)
        clean, noise = decompose(z, velocity, s)
        corrected = clean_update(clean, energy)
        guided = s * clean + (1 - s) * corrected
        if diagnostics and j in points:
            trace['times'].append(s)
            for name, value in [('states', z), ('clean_before', clean), ('clean_guided', guided)]:
                trace[name].append(downsample(value, factor).cpu().numpy())
            for name, value in [('measurement_l2', energy.gradient(clean)),
                                ('proximity_l2', energy.proximity_gradient(clean)), ('prior_l2', velocity)]:
                trace[name].append(value.flatten(1).double().norm(dim=1).cpu().numpy())
            for name, value in [('observed_rmse_before', clean), ('observed_rmse_after', corrected)]:
                residual = downsample(value, factor).flatten(1)[:, ids] - y
                trace[name].append(residual.double().square().mean(1).sqrt().cpu().numpy())
        rng_states.append(generator.get_state().cpu().numpy())
        fresh = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        mixed_noise = math.sqrt(1 - snext) * noise + math.sqrt(snext) * fresh
        z = snext * guided + (1 - snext) * mixed_noise
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite sequential FlowDPS endpoint')
    return z, {k: np.asarray(v) for k, v in trace.items()}, np.stack(rng_states)


def recursive_score(previous_log_weights, new_native, old_native, new_ids, new_y,
                    sigma=.05, tau=1., factor=4):
    """Math-note Eq.(8): old weight times endpoint measurement N times proximity N.

    This is the note's finite-candidate score, not a FlowDPS transition density.
    Retain normalized log weights, even if their displayed exponentials underflow.
    """
    x, old = np.asarray(new_native, dtype=np.float64), np.asarray(old_native, dtype=np.float64)
    log_old = np.asarray(previous_log_weights, dtype=np.float64)
    ids = np.asarray(new_ids).flatten()
    y = np.asarray(new_y, dtype=np.float64).flatten()
    if x.shape != old.shape or x.ndim != 4 or log_old.shape != (len(x),) or ids.shape != y.shape:
        raise ValueError('Invalid paired candidates, weights or new measurements')
    if sigma <= 0 or tau <= 0 or not np.isfinite(x).all() or not np.isfinite(old).all() or not np.isfinite(log_old).all():
        raise ValueError('Nonfinite candidates/weights or invalid Gaussian scale')
    k, c, h, w = x.shape
    task = x.reshape(k, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))
    residual = task.reshape(k, -1)[:, ids] - y
    measurement_energy = np.square(residual).sum(1) / (2 * sigma**2)
    proximity_energy = np.square(x - old).reshape(k, -1).sum(1) / (2 * tau**2)
    log_measurement_N = -measurement_energy - len(ids) / 2 * np.log(2 * np.pi * sigma**2)
    log_proximity_N = -proximity_energy - np.prod(x.shape[1:]) / 2 * np.log(2 * np.pi * tau**2)
    # Gaussian constants are identical across i, so cancel exactly on normalization.
    logits = log_old - measurement_energy - proximity_energy
    log_weights = logits - logsumexp(logits)
    weights = np.exp(log_weights)
    if not np.isfinite(log_weights).all() or not np.isclose(weights.sum(), 1, atol=1e-10):
        raise FloatingPointError('Invalid recursive normalization')
    return {'log_weights': log_weights, 'weights': weights,
            'measurement_energy': measurement_energy, 'proximity_energy': proximity_energy,
            'log_measurement_N': log_measurement_N, 'log_proximity_N': log_proximity_N,
            'log_increment_without_constants': -measurement_energy - proximity_energy,
            'ess': float(1 / np.square(weights).sum())}
