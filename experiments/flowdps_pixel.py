"""FlowDPS-inspired clean/noise decomposition for the frozen pixel-space RF.

Reference: FlowDPS sd3_sampler.py, commit 83618433cece8cbcd056344cf11c021c37f5e186.
This is an adaptation, not an official SD3 reproduction: retain the local
summed Gaussian likelihood, true pooling adjoint and uniform native time grid.
No prior Jacobian is used, following the decomposed clean-estimate update.
"""
import math
import numpy as np
import torch
from .official_face_prior import NATIVE_EPS
from .official_face_reuse import ObservationFlow, downsample


def decompose(z, velocity, s):
    """Our convention is noise at s=0, clean data at s=1."""
    return z + (1 - s) * velocity, z - s * velocity


def data_consistency(clean, operator, iterations=3, step_fraction=.5):
    """Stable GD on SUM ||M D x-y||^2/(2 sigma^2), in clean-image space.

    For these disjoint measurements, the gradient Lipschitz constant is rho.
    eta=step_fraction/rho halves the observed residual per default GD step.
    No clipping, projection to exact readings, or loss averaging is applied.
    """
    if iterations < 0 or not 0 <= step_fraction <= 1:
        raise ValueError('Invalid data consistency optimizer')
    eta = step_fraction / operator.rho
    x = clean.clone()
    for _ in range(iterations):
        x = x - eta * operator.gradient(x)
    return x


@torch.no_grad()
def sample(prior, epsilon, history_ids, history_y, sigma=.05, steps=128,
           factor=4, dc_iterations=3, dc_step_fraction=.5, stochastic=True,
           generator=None, diagnostics=True):
    if steps < 1 or history_ids.ndim != 2 or history_ids.shape != history_y.shape:
        raise ValueError('Invalid steps or history')
    if stochastic and generator is None:
        raise ValueError('An explicit generator is required for stochastic renoising')
    ids, y = history_ids.flatten(), history_y.flatten()
    op = ObservationFlow(epsilon, ids, y, sigma, factor)
    z = epsilon.clone()
    h = (1 - NATIVE_EPS) / steps
    points = set([0, steps // 4, steps // 2, 3 * steps // 4, steps - 1])
    saved = {name: [] for name in ['times', 'states', 'clean_before', 'clean_dc', 'clean_guided',
                                   'clean_observed_rmse', 'dc_observed_rmse', 'guided_observed_rmse',
                                   'measurement_gradient_l2', 'prior_l2']}
    rng_states = []
    for j in range(steps):
        s = NATIVE_EPS + j * h
        snext = 1. if j == steps - 1 else NATIVE_EPS + (j + 1) * h
        velocity = prior(z, s)
        clean, noise = decompose(z, velocity, s)
        corrected = data_consistency(clean, op, dc_iterations, dc_step_fraction)
        # Upstream sigma=1-s, gamma=sigma: interpolate clean prediction and DC result.
        guided = s * clean + (1 - s) * corrected
        if diagnostics and j in points:
            saved['times'].append(s)
            for name, tensor in [('states', z), ('clean_before', clean), ('clean_dc', corrected), ('clean_guided', guided)]:
                saved[name].append(downsample(tensor, factor).cpu().numpy())
            for name, tensor in [('clean_observed_rmse', clean), ('dc_observed_rmse', corrected), ('guided_observed_rmse', guided)]:
                residual = downsample(tensor, factor).flatten(1)[:, ids] - y
                saved[name].append(residual.double().square().mean(1).sqrt().cpu().numpy())
            saved['measurement_gradient_l2'].append(op.gradient(clean).flatten(1).double().norm(dim=1).cpu().numpy())
            saved['prior_l2'].append(velocity.flatten(1).double().norm(dim=1).cpu().numpy())
        if stochastic:
            # Archive every generator state, including the unused last noise draw.
            rng_states.append(generator.get_state().cpu().numpy())
            fresh = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
            # Official implementation: sqrt(sigma_next)*noise_hat + sqrt(1-sigma_next)*fresh.
            noise = math.sqrt(1 - snext) * noise + math.sqrt(snext) * fresh
        z = snext * guided + (1 - snext) * noise
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite FlowDPS-inspired endpoint')
    trace = {name: np.asarray(value) for name, value in saved.items()}
    states = np.stack(rng_states) if rng_states else np.empty((0, 0), dtype=np.uint8)
    return z, trace, states
