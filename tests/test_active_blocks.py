import numpy as np
import pytest
from experiments.active_blocks import block_grid, select_block


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
