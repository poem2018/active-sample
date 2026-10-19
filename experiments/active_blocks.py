"""Active block acquisition: the policy sees candidates, weights and a mask only."""
import numpy as np

METHODS = ('SEQ_UNIFORM', 'SEQ_WEIGHTED', 'SCRATCH')


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


def guidance_inputs(method, ids, readings, old):
    """Enforce the requested new-only sequential / all-history scratch design."""
    if method not in METHODS or len(ids) < 1 or ids.shape != readings.shape:
        raise ValueError('Invalid method or acquired observations')
    if method == 'SCRATCH':
        return ids, readings, None
    if old is None:
        raise ValueError('Sequential guidance needs corresponding old candidates')
    return ids[-1:], readings[-1:], old


def pixel_ridge(indices, readings, shape=(3, 64, 64), sigma=.05):
    """min_x ||M x-y||^2/(2 sigma^2) + ||x||^2/2.

    Direct pixel-space ridge, prior N(0,I). Unknown values are zero in [-1,1]
    (display gray). It is a weak linear reference, not a learned face model.
    """
    ix = np.asarray(indices).flatten()
    y = np.asarray(readings, dtype=np.float64).flatten()
    if ix.shape != y.shape or len(np.unique(ix)) != len(ix) or sigma <= 0:
        raise ValueError('Unique acquired measurements and positive noise required')
    x = np.zeros(shape, dtype=np.float64)
    x.flat[ix] = y / (1 + sigma**2)
    return x
