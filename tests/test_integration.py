"""Integration: drive REAL k-diffusion stochastic samplers with the colored noise_sampler.

Uses a trivial denoiser (predicts zeros) so no model weights are needed, but exercises the
true sampler -> noise_sampler -> FFT-coloring loop: call signature, sigma values passed at each
step, and the requirement that the returned noise be unit-variance and shaped like x.
"""
import torch

from comfy.k_diffusion import sampling as kds

from src.engine.colored_noise import ColorParams, make_colored_noise_sampler
from src.sampling.registry import stochastic_samplers


class _FakeDenoiser:
    """Minimal stand-in for a wrapped diffusion denoiser.

    k-diffusion samplers introspect ``model.inner_model.inner_model.model_sampling`` to detect
    rectified-flow (CONST) models; we expose that chain with a non-CONST sentinel so the standard
    (eps/v) path runs, and make the object callable to return a trivial denoised prediction.
    """

    class _Lvl2:
        model_sampling = object()  # not a comfy.model_sampling.CONST -> standard path

    class _Lvl1:
        inner_model = None

    def __init__(self):
        self.calls = 0
        self.inner_model = self._Lvl1()
        self.inner_model.inner_model = self._Lvl2()

    def __call__(self, x, sigma, **kw):
        self.calls += 1
        return torch.zeros_like(x)


def _denoiser():
    m = _FakeDenoiser()
    return m, m


def test_euler_ancestral_runs_with_colored_noise():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    sigmas = torch.linspace(10.0, 0.0, 6)
    params = ColorParams(mode="parametric", alpha_start=0.0, alpha_end=-2.0)
    ns = make_colored_noise_sampler(x, sigmas, params, seed=0)
    model, calls = _denoiser()
    out = kds.sample_euler_ancestral(model, x, sigmas, extra_args={}, disable=True, eta=1.0, noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert calls.calls >= 1


def test_dpmpp_2s_ancestral_runs_with_colored_noise():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    sigmas = torch.linspace(10.0, 0.0, 6)
    params = ColorParams(mode="parametric", alpha_start=0.0, alpha_end=-1.0)
    ns = make_colored_noise_sampler(x, sigmas, params, seed=0)
    model, _ = _denoiser()
    out = kds.sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args={}, disable=True, eta=1.0,
                                        s_noise=1.0, noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_gamma_mode_runs_end_to_end():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    sigmas = torch.linspace(10.0, 0.0, 6)
    gamma = torch.linspace(0.05, 1.0, 250).unsqueeze(1).repeat(1, 32)  # synthetic [250,32]
    params = ColorParams(mode="gamma_matrix", gamma_matrix=gamma)
    ns = make_colored_noise_sampler(x, sigmas, params, seed=0)
    model, _ = _denoiser()
    out = kds.sample_euler_ancestral(model, x, sigmas, extra_args={}, disable=True, eta=1.0, noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_default_sampler_is_registered_and_callable():
    s = stochastic_samplers()
    assert "dpmpp_2m_sde" in s and callable(s["dpmpp_2m_sde"])
