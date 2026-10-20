"""Average-pooling measurement operator and its native-image likelihood gradient."""
import torch
import torch.nn.functional as F

def downsample(x, factor=4):
    return F.avg_pool2d(x, factor, factor) if factor != 1 else x


def lift(x, factor=4):
    return x.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


class ObservationFlow:
    """Summed Gaussian likelihood gradient with the true pooling adjoint.

    D D* = I / factor**2; rho retains this factor without loss averaging.
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
