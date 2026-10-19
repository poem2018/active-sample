import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
import torch
from experiments.sequential_flowdps import CandidateEnergy, clean_update, sample, recursive_score
from experiments.flowdps_pixel import sample as initial_sample
from experiments.official_face_reuse import downsample


def fixture():
    g = torch.Generator().manual_seed(1729)
    z = torch.randn(2, 3, 8, 8, generator=g, dtype=torch.float64)
    old = torch.randn(2, 3, 8, 8, generator=g, dtype=torch.float64)
    ids = torch.tensor([[0, 3], [6, 11]])
    y = torch.tensor([[.2, -.4], [.5, -.1]], dtype=z.dtype)
    return z, old, ids, y


def test_combined_clean_gradient_matches_gaussian_energy_autograd():
    x, old, ids, y = fixture(); x.requires_grad_(True)
    sigma, tau = .13, .8
    E = CandidateEnergy(x, ids.flatten(), y.flatten(), sigma, old, tau)
    loss = (downsample(x).flatten(1)[:, ids.flatten()] - y.flatten()).square().sum() / (2 * sigma**2)
    loss = loss + (x - old).square().sum() / (2 * tau**2)
    expected, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(E.total_gradient(x), expected)
    assert E.lipschitz == 1 / (16 * sigma**2) + 1 / tau**2


def test_clean_update_matches_independent_dense_quadratic_iteration():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 1, 4, 4)); old = rng.normal(size=x.shape)
    A = np.zeros((2, 16))
    A[0, [0, 1, 4, 5]] = .25; A[1, [10, 11, 14, 15]] = .25
    y = np.array([.3, -.6]); sigma, tau = .17, .9
    H = A.T @ A / sigma**2 + np.eye(16) / tau**2
    force = A.T @ y / sigma**2 + old.flatten() / tau**2
    eta = .5 / np.linalg.eigvalsh(H).max()
    optimum = np.linalg.solve(H, force)
    expected = optimum + np.linalg.matrix_power(np.eye(16) - eta * H, 3) @ (x.flatten() - optimum)
    E = CandidateEnergy(torch.from_numpy(x), torch.tensor([0, 3]), torch.from_numpy(y),
                        sigma, torch.from_numpy(old), tau, factor=2)
    result = clean_update(torch.from_numpy(x), E).numpy().flatten()
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_no_old_candidate_exactly_reproduces_verified_initial_sampler():
    z, _, ids, y = fixture(); prior = lambda x, s: .1 * x + s
    a, _, _ = sample(prior, z, ids, y, steps=8, generator=torch.Generator().manual_seed(55))
    b, _, _ = initial_sample(prior, z, ids, y, steps=8, generator=torch.Generator().manual_seed(55))
    assert torch.equal(a, b)


def test_corresponding_old_candidate_changes_update_and_start_is_fresh():
    z, old, ids, y = fixture(); starts = []
    def prior(x, s):
        starts.append(x.clone()); return .1 * x + s
    a, trace, _ = sample(prior, z, ids, y, old, steps=8, generator=torch.Generator().manual_seed(91))
    assert torch.equal(starts[0], z) and np.all(trace['proximity_l2'] > 0)
    b, _, _ = sample(prior, z, ids, y, old.flip(0), steps=8, generator=torch.Generator().manual_seed(91))
    assert not torch.allclose(a, b)
    c, _, _ = sample(prior, z, ids, y + .2, old, steps=8, generator=torch.Generator().manual_seed(91))
    assert not torch.allclose(a, c)


def test_recursive_score_matches_two_full_gaussian_densities_and_single_old_weight():
    z, old, ids, y = fixture()
    sigma, tau = .13, .8
    log_old = np.log([.15, .85]); x, o = z.numpy(), old.numpy()
    result = recursive_score(log_old, x, o, ids[-1].numpy(), y[-1].numpy(), sigma, tau)
    pooled = x.reshape(2, 3, 2, 4, 2, 4).mean(axis=(3, 5)).reshape(2, -1)
    log_m = norm.logpdf(y[-1].numpy(), loc=pooled[:, ids[-1].numpy()], scale=sigma).sum(1)
    log_p = norm.logpdf(x, loc=o, scale=tau).reshape(2, -1).sum(1)
    expected = log_old + log_m + log_p; expected -= logsumexp(expected)
    np.testing.assert_allclose(result['log_measurement_N'], log_m)
    np.testing.assert_allclose(result['log_proximity_N'], log_p)
    np.testing.assert_allclose(result['log_weights'], expected, atol=1e-11)


def test_log_weights_can_recover_after_displayed_underflow():
    x = np.zeros((2, 1, 4, 4)); x[1] = np.sqrt(40.)
    previous_log = np.array([-1000., 0.])
    assert np.exp(previous_log[0]) == 0
    out = recursive_score(previous_log, x, x, np.array([0]), np.array([0.]), sigma=.1, tau=1.)
    assert np.isfinite(out['log_weights']).all() and out['weights'][0] > .999
    assert 1 <= out['ess'] <= 2


def test_batch_scaling_and_candidate_alignment():
    z, old, ids, y = fixture()
    full = CandidateEnergy(z, ids.flatten(), y.flatten(), old=old)
    one = CandidateEnergy(z[:1], ids.flatten(), y.flatten(), old=old[:1])
    torch.testing.assert_close(full.total_gradient(z)[0], one.total_gradient(z[:1])[0])
    with pytest.raises(ValueError):
        CandidateEnergy(z, ids.flatten(), y.flatten(), old=old[:1])
