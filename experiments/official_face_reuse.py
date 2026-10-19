"""R25 and matched scratch controls with a native 256px frozen flow.

All measurement inputs live in the original 64px space. D is non-overlapping
4x4 averaging, so the induced 64px prior is D#p_256. No high-resolution GT is
used. Full-history Gaussian NLL is SUM ||M D z-y||^2/(2 sigma^2).
"""
import math
import torch
import torch.nn.functional as F
from .official_face_prior import NATIVE_EPS

METHODS = {'C': {'steps': 128, 'label': 'From-scratch 128'},
           'F96': {'steps': 96, 'label': 'From-scratch 96'},
           'F32': {'steps': 32, 'label': 'From-scratch 32'},
           'R25': {'steps': 96, 'start': .25, 'label': 'R25 sequential reuse'}}


def downsample(x, factor=4):
    return F.avg_pool2d(x, factor, factor) if factor != 1 else x


def lift(x, factor=4):
    return x.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


class ObservationFlow:
    """Exact affine semigroup on the observed block-mean subspace.

DD*=I/factor^2; P=factor^2 D* M* M D is an orthogonal projector.
This retains the true adjoint 1/factor^2, without rescaling the likelihood.
"""
    def __init__(self, template, ids, readings, sigma=.05, factor=4):
        h, w = template.shape[-2:]
        if h % factor or w % factor or sigma <= 0:
            raise ValueError('Invalid resolution or noise')
        if ids.ndim != 1 or ids.shape != readings.shape or ids.unique().numel() != ids.numel():
            raise ValueError('Unique flat acquired measurements required')
        self.factor = factor
        self.mask = torch.zeros((1, template.shape[1], h//factor, w//factor), device=template.device, dtype=template.dtype)
        self.y = torch.zeros_like(self.mask)
        self.mask.flatten()[ids] = 1
        self.y.flatten()[ids] = readings
        self.rho = 1 / (sigma**2 * factor**2)

    def gradient(self, z):
        return self.rho * lift(self.mask * (downsample(z, self.factor) - self.y), self.factor)

    def step(self, z, velocity, duration):
        beta = -math.expm1(-self.rho * duration)
        gamma = beta / self.rho - duration
        correction = self.mask * (beta * (self.y - downsample(z, self.factor))
                                  + gamma * downsample(velocity, self.factor))
        return z + duration * velocity + lift(correction, self.factor)


@torch.no_grad()
def integrate(prior, initial, ids, readings, sigma=.05, start=NATIVE_EPS,
              end=1., steps=128, factor=4, observer=None):
    if steps < 1 or not NATIVE_EPS <= start < end <= 1:
        raise ValueError('Invalid integration interval')
    operator = ObservationFlow(initial, ids, readings, sigma, factor)
    h = (end-start)/steps
    z = initial.clone()
    for j in range(steps):
        t = start+j*h
        v = prior(z, t)
        if observer is not None:
            observer(z, v, operator.gradient(z), t, j)
        mid = operator.step(z, v, h/2)
        vm = prior(mid, t+h/2)
        z = operator.step(z, vm, h)
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite flow endpoint')
    return z


@torch.no_grad()
def update(prior, method, epsilon, history_ids, history_y, old=None, sigma=.05,
           observer=None, step_multiplier=1):
    if method not in METHODS or history_ids.ndim != 2 or history_ids.shape != history_y.shape:
        raise ValueError('Unknown method or invalid acquired history')
    if not len(history_ids):
        raise ValueError('At least one observed block is required')
    if old is None:
        start, steps, z = NATIVE_EPS, 128, epsilon
    elif method == 'R25':
        if old.shape != epsilon.shape:
            raise ValueError('Old candidate and noise shapes must match')
        start, steps, z = .25, 96, .25*old + .75*epsilon
    else:
        start, steps, z = NATIVE_EPS, METHODS[method]['steps'], epsilon
    return integrate(prior, z, history_ids.flatten(), history_y.flatten(), sigma,
                     start=start, steps=steps*step_multiplier, observer=observer)
