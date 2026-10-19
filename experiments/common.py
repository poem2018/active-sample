"""Data splits, reproducibility, and the literal quadratic guidance in Eq. (6)."""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_data():
    dataset = MNIST(str(ROOT / "data/mnist"), train=True, download=False)
    path = ROOT / "data/mnist/splits_v1.json"
    if path.exists():
        split = json.loads(path.read_text())
    else:
        order = np.random.default_rng(20261010).permutation(len(dataset)).tolist()
        train, validation = order[:55000], order[55000:]
        tuning, pilot = [], []
        for label in range(10):
            matches = [i for i in validation if int(dataset.targets[i]) == label]
            tuning.extend(matches[:1])
            pilot.extend(matches[1:3])
        split = {"seed":20261010,"source":"torchvision MNIST official training set",
                 "train":train,"validation":validation,"tuning":tuning,"pilot":pilot,
                 "test":"All 10000 official test examples remain unused.",
                 "preprocessing":"uint8 / 255; bilinear resize to 32x32, align_corners=False, antialias=True; scale to [-1,1]"}
        write_json(path, split)
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["tuning"]) & set(split["pilot"])
    return dataset, split


def preprocess(images):
    x = images[:, None].float() / 255
    return F.interpolate(x, size=(32,32), mode="bilinear", align_corners=False, antialias=True) * 2 - 1


def fixed_history(image, total=32, sigma=0.05, seed=0):
    """Only this simulator accesses GT; inference receives indices and readings."""
    if total > image.numel() or total <= 0:
        raise ValueError("Invalid measurement budget")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(image.numel(), generator=rng)[:total].to(image.device)
    noise = torch.randn(total, generator=rng).to(image.device) * sigma
    readings = image.flatten()[indices] + noise
    return indices, readings, noise


def quadratic_terms(shape, indices, readings, sigma, old=None, tau=None):
    if sigma <= 0 or indices.ndim != 1 or readings.shape != indices.shape:
        raise ValueError("Positive sigma and matching one-dimensional observations required")
    if len(torch.unique(indices)) != len(indices):
        raise ValueError("Duplicate pixel observations are forbidden")
    d = torch.zeros(shape, device=readings.device, dtype=readings.dtype)
    c = torch.zeros_like(d)
    d.flatten(1)[:,indices] = 1 / sigma**2
    c.flatten(1)[:,indices] = readings / sigma**2
    if old is not None:
        if tau is None or tau <= 0 or old.shape != d.shape:
            raise ValueError("Positive tau and one old image per candidate required")
        d += 1 / tau**2
        c += old / tau**2
    return d, c


@torch.no_grad()
def sample_guided(prior, initial, indices, readings, sigma, steps=128, old=None, tau=None):
    """Exponential midpoint for z'=v(z,s)-D*z+c, exactly Eq. (6).

    The constant quadratic term is integrated analytically; there is no clipping,
    observation overwrite, annealing, warm start, resampling or terminal denoiser.
    Two prior evaluations per step; s runs from Gaussian noise (0) to image (1).
    """
    if steps < 1:
        raise ValueError("At least one integration step required")
    d, c = quadratic_terms(initial.shape, indices, readings, sigma, old, tau)
    h = 1 / steps
    def coefficients(dt):
        decay = torch.exp(-dt*d)
        factor = torch.where(d > 0, -torch.expm1(-dt*d) / d.clamp_min(1e-30), torch.full_like(d,dt))
        return decay, factor
    decay_half, factor_half = coefficients(h/2)
    decay, factor = coefficients(h)
    z = initial.clone()
    for j in range(steps):
        s = j*h
        middle = decay_half*z + factor_half*(c + prior(z,s))
        z = decay*z + factor*(c + prior(middle,s+h/2))
    if not torch.isfinite(z).all():
        raise FloatingPointError("Non-finite ODE output")
    return z


def recursive_weights(previous, new, old, index, reading, sigma, tau):
    """Eq. (8): exactly one old weight, full-image SUM proximity penalty."""
    return recursive_log_weights(previous.log(),new,old,index,reading,sigma,tau).exp()


def recursive_log_weights(previous_log, new, old, index, reading, sigma, tau):
    """Keep normalized float64 log weights across rounds, even after exp underflows."""
    residual = new.flatten(1)[:,index].double() - reading.double()
    proximity = (new.double()-old.double()).flatten(1).square().sum(1)
    scores = previous_log.double() - residual.square()/(2*sigma**2) - proximity/(2*tau**2)
    return torch.log_softmax(scores,dim=0)


def evaluate(candidates, weights, image, indices, readings, initial_count=16):
    flat = candidates.flatten(1)
    mean = (weights[:,None]*flat).sum(0)
    truth = image.flatten()
    error = (mean-truth)/2  # standardized [0,1] image scale, without clipping
    unseen = torch.ones_like(truth,dtype=torch.bool)
    unseen[indices] = False
    mse = error.square().mean()
    pair_distance = torch.pdist(flat,p=2)/math.sqrt(flat.shape[1])
    residual = flat[:,indices] - readings
    count = len(indices)
    old_count = count-1 if count>initial_count else count
    result = {
        "mse_01":float(mse),"mae_01":float(error.abs().mean()),
        "unobserved_mse_01":float(error[unseen].square().mean()),
        "psnr_db":float(-10*torch.log10(mse.clamp_min(1e-30))),
        "ess":float(1/weights.square().sum()),
        "weight_entropy":float(-(weights*weights.clamp_min(1e-30).log()).sum()),
        "max_weight":float(weights.max()),
        "pairwise_rmse":float(pair_distance.mean()),
        "weighted_pixel_variance":float((weights[:,None]*(flat-mean).square()).sum(0).mean()),
        "all_observation_candidate_rmse":float(residual.square().mean().sqrt()),
        "old_observation_candidate_rmse":float(residual[:,:old_count].square().mean().sqrt()),
        "old_observation_weighted_candidate_rmse":float((weights*residual[:,:old_count].square().mean(1)).sum().sqrt()),
        "old_observation_mean_rmse":float((mean[indices[:old_count]]-readings[:old_count]).square().mean().sqrt()),
        "initial_history_candidate_rmse":float(residual[:,:initial_count].square().mean().sqrt()),
        "new_observation_candidate_rmse":None if count==initial_count else float(residual[:,-1].square().mean().sqrt()),
        "new_observation_mean_rmse":None if count==initial_count else float((mean[indices[-1]]-readings[-1]).abs()),
    }
    return result,mean.reshape_as(image)
