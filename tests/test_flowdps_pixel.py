import pytest
import torch

from experiments.flowdps_pixel import decompose, data_consistency, sample
from experiments.official_face_reuse import ObservationFlow, downsample


def fixture():
    g = torch.Generator().manual_seed(1729)
    z = torch.randn(2, 3, 8, 8, generator=g, dtype=torch.float64)
    ids = torch.tensor([[0, 3], [6, 11]])
    y = torch.tensor([[.2, -.4], [.5, -.1]], dtype=z.dtype)
    return z, ids, y


@pytest.mark.parametrize('s', [.001, .25, .8, 1.])
def test_decomposition_recovers_known_clean_and_noise(s):
    clean, _, _ = fixture()
    noise = clean.flip(-1) + .31
    z = s * clean + (1 - s) * noise
    predicted_clean, predicted_noise = decompose(z, clean - noise, s)
    torch.testing.assert_close(predicted_clean, clean)
    torch.testing.assert_close(predicted_noise, noise)


def test_clean_gradient_is_true_gaussian_gradient_with_pooling_adjoint():
    clean, ids, y = fixture()
    clean.requires_grad_(True)
    op = ObservationFlow(clean, ids.flatten(), y.flatten(), sigma=.05)
    loss = (downsample(clean).flatten(1)[:, ids.flatten()] - y.flatten()).square().sum() / (2 * .05**2)
    expected, = torch.autograd.grad(loss, clean)
    torch.testing.assert_close(op.gradient(clean), expected)
    # Batch size must not change a candidate's guidance (no averaging).
    torch.testing.assert_close(op.gradient(clean)[0], op.gradient(clean[:1])[0])


def test_three_stable_steps_contract_measurements_and_preserve_nullspace():
    clean, ids, y = fixture()
    op = ObservationFlow(clean, ids.flatten(), y.flatten(), sigma=.05)
    result = data_consistency(clean, op, iterations=3, step_fraction=.5)
    residual = downsample(clean).flatten(1)[:, ids.flatten()] - y.flatten()
    corrected = downsample(result).flatten(1)[:, ids.flatten()] - y.flatten()
    torch.testing.assert_close(corrected, residual / 8)
    delta = result - clean
    # Native update is constant within each observed pooling cell, zero elsewhere.
    pooled_delta = downsample(delta)
    lifted = pooled_delta.repeat_interleave(4, -2).repeat_interleave(4, -1)
    torch.testing.assert_close(delta, lifted)
    mask = torch.ones(12, dtype=torch.bool); mask[ids.flatten()] = False
    torch.testing.assert_close(pooled_delta.flatten(1)[:, mask], torch.zeros(2, 8, dtype=clean.dtype))


def test_unguided_deterministic_limit_equals_euler_and_preserves_start():
    epsilon, ids, y = fixture()
    original = epsilon.clone()
    calls = []
    def prior(z, s):
        calls.append((z.clone(), s))
        return .2 * z + s
    result, _, _ = sample(prior, epsilon, ids, y, steps=8, stochastic=False, dc_step_fraction=0.)
    expected = epsilon.clone()
    h = .999 / 8
    for j in range(8):
        expected = expected + h * (.2 * expected + .001 + j * h)
    torch.testing.assert_close(result, expected)
    assert len(calls) == 8
    assert torch.equal(calls[0][0], original) and torch.equal(epsilon, original)


def test_stochastic_replay_diagnostics_and_observation_dependence():
    epsilon, ids, y = fixture()
    prior = lambda z, s: .15 * z + s
    def run(readings, seed, diagnostics):
        return sample(prior, epsilon, ids, readings, steps=8, stochastic=True,
                      generator=torch.Generator().manual_seed(seed), diagnostics=diagnostics)
    a, trace, states = run(y, 19, True)
    b, _, _ = run(y, 19, False)
    c, _, _ = run(y, 20, True)
    d, _, _ = run(y + .1, 19, True)
    assert torch.equal(a, b) and not torch.equal(a, c) and not torch.equal(a, d)
    assert trace['clean_before'].shape == (5, 2, 3, 2, 2)
    assert states.shape[0] == 8


def test_stochastic_mode_requires_explicit_rng():
    epsilon, ids, y = fixture()
    with pytest.raises(ValueError, match='generator'):
        sample(lambda z, s: z, epsilon, ids, y, stochastic=True)
