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


def _renorm_unit(t, energy_scale):
    t = t - t.mean()
    std = t.std()
    if float(std) > 1e-9:
        t = t / std
    return t * energy_scale


def color_tensor(white, scaling_vector, energy_scale=1.0):
    """Spectrally shape ``white`` by a per-bin ``scaling_vector``.

    Returns zero-mean, unit-variance (times ``energy_scale``) noise of the same
    shape/dtype. FFT axes are derived from the rank: >=4D -> 2D spatial FFT over
    the last two dims (image [B,C,H,W] and video [B,C,T,H,W]); 3D -> 1D FFT over
    the last dim (audio [B,C,L]); <3D -> returned unchanged. Nested-tensor safe.
    The FFT runs in float32 (half precision is unsupported on most backends) with
    a CPU fallback for backends (DirectML/MPS) that reject ``torch.fft``.
    """
    if white.is_nested:
        import comfy.nested_tensor as nt
        return nt.NestedTensor(
            [color_tensor(t, scaling_vector, energy_scale) for t in white.unbind()]
        )

    num_bins = scaling_vector.shape[0]
    ndim = white.dim()
    if ndim >= 4:
        h, w = white.shape[-2], white.shape[-1]
        grid = scaling_vector.to(white.device)[radial_bin_map_2d(h, w, num_bins, white.device)]
        dims = (-2, -1)
    elif ndim == 3:
        length = white.shape[-1]
        grid = scaling_vector.to(white.device)[radial_bin_map_1d(length, num_bins, white.device)]
        dims = (-1,)
    else:
        return white  # cannot meaningfully shape (<3d, e.g. flat audio not in [B,C,L] form)

    orig_dtype = white.dtype
    wf = white.float()
    try:
        spec = torch.fft.fftn(wf, dim=dims)
        shaped = torch.fft.ifftn(spec * grid, dim=dims).real
    except RuntimeError:
        # DirectML / MPS reject torch.fft -> CPU fallback (mirrors comfy_extras FreeU/FreSca).
        spec = torch.fft.fftn(wf.cpu(), dim=dims)
        shaped = torch.fft.ifftn(spec * grid.detach().cpu(), dim=dims).real.to(white.device)

    return _renorm_unit(shaped, energy_scale).to(orig_dtype)
