"""Pure-torch spectral shaping primitives for colored noise.

Imports only ``torch`` so it can be unit-tested without a running ComfyUI.
"""
import torch

# Cache of radial frequency -> bin-index maps, keyed by (kind, *shape, num_bins).
# Stored on CPU; moved to the working device per call.
_BIN_CACHE = {}


def radial_bin_map_2d(h, w, num_bins, device):
    """Integer [h, w] map from each 2D FFT frequency to a radial bin in [0, num_bins-1]."""
    key = ("2d", h, w, num_bins)
    idx = _BIN_CACHE.get(key)
    if idx is None:
        fy = torch.fft.fftfreq(h).view(h, 1)
        fx = torch.fft.fftfreq(w).view(1, w)
        r = torch.sqrt(fx * fx + fy * fy)
        r = r / r.max()
        idx = (r * (num_bins - 1)).round().long().clamp(0, num_bins - 1)
        _BIN_CACHE[key] = idx
    return idx.to(device)


def radial_bin_map_1d(length, num_bins, device):
    """Integer [length] map from each 1D FFT frequency to a radial bin in [0, num_bins-1]."""
    key = ("1d", length, num_bins)
    idx = _BIN_CACHE.get(key)
    if idx is None:
        f = torch.fft.fftfreq(length).abs()
        r = f / f.max()
        idx = (r * (num_bins - 1)).round().long().clamp(0, num_bins - 1)
        _BIN_CACHE[key] = idx
    return idx.to(device)


def parametric_scaling(num_bins, alpha, device=torch.device("cpu")):
    """Per-bin amplitude multiplier following a power-law PSD ~ 1/f^alpha.

    amplitude(r) ~ r^(-alpha/2): alpha 0 -> white, +ve -> red/pink (low-freq),
    -ve -> blue/violet (high-freq). DC is killed for alpha>0 (it would diverge;
    the constant component is removed by the zero-mean renorm anyway).
    """
    r = torch.linspace(0.0, 1.0, num_bins, device=device)
    eps = 1.0 / (2 * num_bins)
    scaling = r.clamp(min=eps).pow(-alpha / 2.0)
    if alpha > 0:
        scaling[0] = 0.0
    return scaling
