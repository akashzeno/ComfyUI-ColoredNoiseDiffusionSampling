"""Trajectory schedule helpers: sigma->progress, alpha interpolation, and gamma shaping.

Imports only ``torch`` so it can be unit-tested without a running ComfyUI.
"""
import math

import torch


def progress_from_sigma(sigma, sigma_max, sigma_min):
    """Continuous trajectory progress in [0, 1] from the current sigma (log-sigma space).

    0 at the start of sampling (high sigma), 1 at the end (low sigma). Derived from the
    passed sigma rather than a step index or call count, so it is robust to rectified-flow
    sampler dispatch, off-grid sub-step sigmas, and non-uniform per-step call counts.
    """
    s = float(sigma)
    smax = float(sigma_max)
    smin = max(float(sigma_min), 1e-9)
    if smax <= smin:
        return 0.0
    s = min(max(s, smin), smax)
    return (math.log(smax) - math.log(s)) / (math.log(smax) - math.log(smin))


def interp_alpha(alpha_start, alpha_end, progress, mode="linear", sharpness=4.0):
    """Interpolate the spectral exponent alpha between start/end over progress in [0, 1]."""
    p = min(max(float(progress), 0.0), 1.0)
    if mode == "exponential":
        p = (math.exp(sharpness * p) - 1.0) / (math.exp(sharpness) - 1.0)
    return alpha_start + p * (alpha_end - alpha_start)


def gamma_row_at(gamma_matrix, progress):
    """Linearly interpolate a [num_bins] gamma row at continuous progress.

    Decouples the matrix's native step count from the actual sampler step count.
    """
    n = gamma_matrix.shape[0]
    f = min(max(float(progress), 0.0), 1.0) * (n - 1)
    i0 = int(f)
    i1 = min(i0 + 1, n - 1)
    frac = f - i0
    return gamma_matrix[i0] * (1.0 - frac) + gamma_matrix[i1] * frac


def gamma_scaling(gamma_row, divider=1.0, shaping="none", power=1.0, alpha_tilting=0.0):
    """Per-bin amplitude from a gamma row: residual = (1 - gamma/divider), with optional
    frequency tilt and sqrt/power shaping.

    Faithful to colored-noise-sampling ``transport/integrators.py``:
    ``cns_sde.__generate_matrix_scaled_noise`` (multiplicative-tilt variant).
    """
    residual = (1.0 - gamma_row / divider).clamp(min=0.0)
    if alpha_tilting != 0.0:
        f_norm = torch.linspace(0.0, 1.0, gamma_row.shape[0], device=gamma_row.device)
        residual = torch.exp(alpha_tilting * f_norm) * residual
    if shaping == "sqrt":
        return residual.clamp(min=0.0).sqrt()
    if shaping == "power" and power != 1.0:
        return residual.clamp(min=0.0).pow(power)
    return residual


def load_gamma_matrix(path):
    """Load a [steps, bins] gamma tensor saved as a raw tensor .pt/.pth file."""
    g = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(g, torch.Tensor) or g.dim() != 2:
        raise ValueError(
            f"gamma matrix at {path} must be a 2D tensor [steps, bins]; got {type(g)}"
        )
    return g.float()
