"""Active block acquisition: the policy sees candidates, weights and a mask only."""
import numpy as np


def block_grid(resolution=64, block=4, channels=3):
    if resolution % block or block < 1:
        raise ValueError('Blocks must tile the image')
    coordinates = np.array([(r, c) for r in range(0, resolution, block)
                            for c in range(0, resolution, block)], dtype=np.int64)
    indices = np.array([[ch * resolution**2 + rr * resolution + cc
                         for ch in range(channels) for rr in range(r, r + block)
                         for cc in range(c, c + block)] for r, c in coordinates])
    return coordinates, indices


def select_block(candidates, weights, observed, block=4):
    """Argmax SUM of weighted predictive variance; row-major tie breaking.

    There is deliberately no GT, measurement value, or sensor-oracle argument.
    Scores use raw model-scale candidates, without display clipping.
    """
    x = np.asarray(candidates, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 4 or x.shape[-1] != x.shape[-2] or not np.isfinite(x).all():
        raise ValueError('Finite square candidate images required')
    if w.shape != (len(x),) or np.any(w < 0) or not np.isfinite(w).all() or not np.isclose(w.sum(), 1):
        raise ValueError('Normalized nonnegative candidate weights required')
    _, ids = block_grid(x.shape[-1], block, x.shape[1])
    seen = np.asarray(observed, dtype=bool)
    if seen.shape != (len(ids),) or seen.all():
        raise ValueError('At least one unobserved block required')
    mean = np.einsum('i,ichw->chw', w, x)
    variance = np.einsum('i,ichw->chw', w, (x - mean)**2)
    scores = variance.flatten()[ids].sum(1)
    eligible = np.flatnonzero(~seen)
    chosen = int(eligible[np.argmax(scores[eligible])])
    return chosen, scores
