"""Frozen official CelebA-HQ pixel-space Rectified Flow, without retraining.

Upstream source is pinned in artifacts/external/rectified_flow_official.
The native model uses Fourier log-time features, labels=999*t, t>=.001.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .common import ROOT, sha256

UPSTREAM = ROOT / 'artifacts/external/rectified_flow_official'
CHECKPOINT = ROOT / 'artifacts/prior/celebahq256_rectified_official/checkpoint_10.pth'
CHECKPOINT_SHA256 = '890198b78860ca1eb679d01485231f6fcb47161a215aab4a94de7fd7156d6d00'
NATIVE_EPS = .001


def load_official_prior(device='cuda', verify_hash=True):
    if verify_hash:
        assert sha256(CHECKPOINT) == CHECKPOINT_SHA256
    sys.path.insert(0, str(ROOT / 'artifacts/external/python_deps'))
    sys.path.insert(0, str(UPSTREAM / 'ImageGeneration'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / 'artifacts/external/torch_extensions'))
    os.environ.setdefault('MAX_JOBS', '4')
    from configs.rectified_flow.celeba_hq_pytorch_rf_gaussian import get_config
    from models.ncsnpp import NCSNpp

    config = get_config()
    model = NCSNpp(config)
    # Old NumPy module name is explicitly mapped; no unrestricted unpickling.
    allowed = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    state = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    shadow = checkpoint['ema']['shadow_params']
    assert len(parameters) == len(shadow)
    with torch.no_grad():
        for (name, p), source in zip(parameters, shadow):
            assert p.shape == source.shape, name
            p.copy_(source)
            assert torch.equal(p, source), name
    metadata = {'source': json.loads((UPSTREAM / 'upstream.json').read_text()),
                'checkpoint_sha256': CHECKPOINT_SHA256,
                'training_step': int(checkpoint['step']),
                'ema_updates': int(checkpoint['ema']['num_updates']),
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_ema_tensors': len(parameters),
                'native_resolution': 256, 'model_scale': '[-1,1]',
                'native_start_time': NATIVE_EPS, 'time_labels': '999*t',
                'ema_loaded_exactly': True, 'strict_model_load': True,
                'config': config.to_dict(),
                'training_split': 'Upstream does not publish checkpoint training-image IDs; no held-out claim.'}
    metadata['config']['device'] = str(metadata['config']['device'])
    model = model.eval().requires_grad_(False).to(device)
    return OfficialPrior(model), metadata


class OfficialPrior:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, t):
        if not NATIVE_EPS <= t <= 1:
            raise ValueError('Official Fourier model requires native t in [.001,1]')
        labels = torch.full((len(x),), t * 999, device=x.device, dtype=x.dtype)
        return self.model(x, labels)
