import numpy as np
import pytest
from experiments.active_blocks import block_grid, select_block, guidance_inputs, pixel_ridge


def test_grid_is_exact_partition_and_selects_maximum_disagreement():
    coords, ids = block_grid()
    assert coords.shape == (256, 2) and ids.shape == (256, 48)
    np.testing.assert_array_equal(np.sort(ids.ravel()), np.arange(3 * 64**2))
    x = np.zeros((2, 3, 8, 8)); x[1, :, 0:4, 4:8] = 2
    selected, scores = select_block(x, [.5, .5], [False] * 4)
    assert selected == 1 and scores[1] == 48
    assert select_block(x, [.5, .5], [False, True, False, False])[0] == 0


def test_weights_really_change_acquisition_and_permutations_preserve_it():
    # One variable candidate per informative block.
    x = np.zeros((3, 1, 8, 8))
    x[0, 0, :4, :4] = 2
    x[2, 0, :4, 4:] = 3
    assert select_block(x, [1/3]*3, [False]*4)[0] == 1
    weights = np.array([.5, .49, .01])
    chosen, scores = select_block(x, weights, [False]*4)
    assert chosen == 0
    perm = [2, 0, 1]
    other, other_scores = select_block(x[perm], weights[perm], [False]*4)
    assert other == chosen
    np.testing.assert_allclose(other_scores, scores)


def test_selector_uses_unclipped_images_and_excludes_all_acquired_blocks():
    x = np.zeros((2, 1, 8, 8)); x[:, 0, :4, :4] = np.array([3, 8])[:, None, None]
    x[1, 0, 4:, 4:] = 1
    assert select_block(x, [.5, .5], [False]*4)[0] == 0
    assert select_block(x, [.5, .5], [True, False, False, False])[0] == 3
    with pytest.raises(ValueError): select_block(x, [.5, .5], [True]*4)


def test_guidance_visibility_matches_user_design_and_retains_old_identity():
    ids = np.arange(12).reshape(3, 4); y = np.arange(12.).reshape(3, 4); old = object()
    for method in ('SEQ_UNIFORM', 'SEQ_WEIGHTED'):
        ii, yy, oo = guidance_inputs(method, ids, y, old)
        np.testing.assert_array_equal(ii, ids[-1:]); np.testing.assert_array_equal(yy, y[-1:])
        assert oo is old
        # Historical sensor values cannot influence the new-only guidance.
        altered = y.copy(); altered[:-1] += 10000
        np.testing.assert_array_equal(guidance_inputs(method, ids, altered, old)[1], yy)
    ii, yy, oo = guidance_inputs('SCRATCH', ids, y, old)
    assert ii is ids and yy is y and oo is None


def test_ridge_agrees_with_independent_dense_normal_equations():
    ids = np.array([0, 3, 6]); y = np.array([.2, -.8, 1.1]); sigma = .23
    A = np.eye(9)[ids]
    expected = np.linalg.solve(A.T @ A / sigma**2 + np.eye(9), A.T @ y / sigma**2)
    result = pixel_ridge(ids, y, shape=(1, 3, 3), sigma=sigma)
    np.testing.assert_allclose(result.ravel(), expected, rtol=1e-14, atol=1e-14)
