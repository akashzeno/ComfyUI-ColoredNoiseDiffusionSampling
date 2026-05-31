"""Bridges the schedule + spectral primitives into colored-noise providers.

Provides:
- ``ColorParams``: the full set of coloring controls.
- ``scaling_for_progress``: the per-bin amplitude vector for a given trajectory progress.
- ``make_colored_noise_sampler``: a lazily-built ``noise_sampler(sigma, sigma_next)`` closed
  over the latent ``x``, returning unit-variance colored noise for the k-diffusion seam.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from .schedule import gamma_row_at, gamma_scaling, interp_alpha, progress_from_sigma
from .spectral import color_tensor, parametric_scaling

# Resolution of the parametric per-frequency profile (gamma mode uses the matrix's bin count).
PARAMETRIC_BINS = 256


@dataclass
class ColorParams:
    mode: str = "parametric"                  # "parametric" | "gamma_matrix"
    alpha_start: float = 0.0
    alpha_end: float = -1.0
    interpolation: str = "linear"             # "linear" | "exponential"
    exp_sharpness: float = 4.0
    gamma_matrix: Optional[torch.Tensor] = None
    gamma_divider: float = 1.0
    gamma_shaping: str = "none"               # "none" | "sqrt" | "power"
    power_gamma: float = 1.0
    alpha_tilting: float = 0.0
    energy_scale: float = 1.0


def scaling_for_progress(params: ColorParams, progress, device=torch.device("cpu")):
    """Per-bin amplitude vector for this trajectory progress (mode-dispatched)."""
    if params.mode == "gamma_matrix" and params.gamma_matrix is not None:
        row = gamma_row_at(params.gamma_matrix.to(device), progress)
        return gamma_scaling(
            row, params.gamma_divider, params.gamma_shaping, params.power_gamma, params.alpha_tilting
        )
    alpha = interp_alpha(
        params.alpha_start, params.alpha_end, progress, params.interpolation, params.exp_sharpness
    )
    return parametric_scaling(PARAMETRIC_BINS, alpha, device)


def _make_generator(x, seed):
    if seed is None:
        return None
    # Mirror comfy.k_diffusion.sampling.default_noise_sampler's device/seed handling.
    s = int(seed) + (1 if x.device == torch.device("cpu") else 0)
    g = torch.Generator(device=x.device)
    g.manual_seed(s)
    return g


def _randn_like_x(x, generator):
    if x.is_nested:
        return torch.nested.nested_tensor(
            [
                torch.randn(t.shape, dtype=t.dtype, layout=t.layout, device=t.device, generator=generator)
                for t in x.unbind()
            ]
        )
    return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


def make_colored_noise_sampler(x, sigmas, params: ColorParams, seed):
    """Build a ``noise_sampler(sigma, sigma_next) -> unit-variance colored noise``, closed over x.

    The closure must be built here (where x is available) rather than passed via static
    sampler ``extra_options`` — the noise must match x's shape/dtype/device at call time.
    """
    sigma_max = float(sigmas[0])
    positives = [float(s) for s in sigmas if float(s) > 0]
    sigma_min = min(positives) if positives else 1e-3
    generator = _make_generator(x, seed)

    def noise_sampler(sigma, sigma_next):
        white = _randn_like_x(x, generator)
        p = progress_from_sigma(sigma, sigma_max, sigma_min)
        scaling = scaling_for_progress(params, p, device=torch.device("cpu"))
        return color_tensor(white, scaling, energy_scale=params.energy_scale)

    return noise_sampler
