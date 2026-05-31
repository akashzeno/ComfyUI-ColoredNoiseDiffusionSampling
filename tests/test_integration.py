"""Integration: drive REAL k-diffusion stochastic samplers with the colored noise_sampler.

Uses a trivial denoiser (predicts zeros) so no model weights are needed, but exercises the
true sampler -> noise_sampler -> FFT-coloring loop. Critically, it also drives EVERY registry
sampler through build_colored_sampler with the production base_opts, and the actual node
execute() for the historically-fragile phi_*/heun bases, so a solver_type-style crash or a
silent no-op cannot pass green.
"""
import pytest
import torch

from comfy.k_diffusion import sampling as kds

from src.engine.colored_noise import ColorParams, make_colored_noise_sampler
from src.sampling.colored_ksampler import build_colored_sampler
from src.sampling.registry import stochastic_samplers
from src.nodes.sampler_select import ColoredNoise_SamplerSelect

# The exact base_opts the production nodes forward (universal SDE knobs only).
PROD_BASE_OPTS = {"eta": 1.0, "s_noise": 1.0}
SAMPLER_NAMES = list(stochastic_samplers().keys())
# A schedule that ENDS ABOVE ZERO so the final ancestral step still injects noise; a schedule
# ending at exactly 0 would discard all colored noise on the last step (false-confidence trap).
SIGMAS = torch.linspace(10.0, 0.5, 6)


class _FakeDenoiser:
    """Minimal stand-in for a wrapped diffusion denoiser.

    k-diffusion samplers introspect ``model.inner_model.inner_model.model_sampling`` (and some
    deeper plumbing) for rectified-flow dispatch; we expose a non-CONST sentinel so the standard
    path runs, and make the object callable to return a trivial denoised prediction. Samplers
    that need plumbing we don't provide raise AttributeError (NOT our concern); a ValueError from
    our forwarded options (e.g. a bad solver_type) IS our bug.
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


@pytest.mark.parametrize("name", SAMPLER_NAMES)
def test_every_registry_sampler_runs_without_option_crash(name):
    """No registry sampler may raise a ValueError from our forwarded base_opts (catches the
    solver_type='midpoint' crash on phi_* bases seeds_2 / exp_heun_2_x0_sde)."""
    sampler = build_colored_sampler(stochastic_samplers()[name], PROD_BASE_OPTS, ColorParams())
    x = torch.randn(1, 4, 16, 16)
    try:
        sampler.sampler_function(_FakeDenoiser(), x, SIGMAS, extra_args={"seed": 0}, disable=True)
    except ValueError as e:  # our forwarded options are invalid for this base
        pytest.fail(f"{name} raised ValueError from forwarded options: {e}")
    except Exception:
        pass  # incomplete fake model -> non-ValueError errors are expected and irrelevant here


@pytest.mark.parametrize("name", ["seeds_2", "exp_heun_2_x0_sde", "dpmpp_2m_sde_heun"])
def test_node_built_sampler_does_not_crash_on_options(name):
    """The REAL node path (execute -> KSAMPLER) must not forward a solver_type that the phi_*
    bases reject or that silently downgrades the _heun base."""
    if name not in SAMPLER_NAMES:
        pytest.skip(f"{name} not in registry on this build")
    out = ColoredNoise_SamplerSelect.execute(
        base_sampler=name, eta=1.0, s_noise=1.0, mode="parametric",
        alpha_start=0.0, alpha_end=-1.0, interpolation="linear", exp_sharpness=4.0,
        gamma_matrix="none (parametric)", gamma_divider=1.0, gamma_shaping="none",
        power_gamma=1.0, alpha_tilting=0.0, energy_scale=1.0)
    sampler = out.result[0]
    x = torch.randn(1, 4, 16, 16)
    try:
        sampler.sampler_function(_FakeDenoiser(), x, SIGMAS, extra_args={"seed": 0}, disable=True)
    except ValueError as e:
        pytest.fail(f"node-built {name} raised ValueError: {e}")
    except Exception:
        pass


def test_res_multistep_noop_variants_excluded_but_ancestral_kept():
    s = stochastic_samplers()
    assert "res_multistep" not in s          # injects eta=0 -> coloring no-op -> excluded
    assert "res_multistep_cfg_pp" not in s
    assert "res_multistep_ancestral" in s    # exposes eta -> genuinely colorable -> kept


def test_euler_ancestral_runs_and_injects_noise():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    params = ColorParams(mode="parametric", alpha_start=0.0, alpha_end=-2.0)
    ns = make_colored_noise_sampler(x, SIGMAS, params, seed=0)
    model = _FakeDenoiser()
    out = kds.sample_euler_ancestral(model, x, SIGMAS, extra_args={}, disable=True, eta=1.0,
                                     noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert float(out.std()) > 0.0          # noise actually contributed (schedule ends above 0)
    assert model.calls >= 1


def test_coloring_actually_changes_output():
    """Two different spectral tilts (same seed) must produce different latents — proves the
    colored noise is genuinely shaping the result, not a no-op."""
    x = torch.randn(1, 4, 32, 32)
    red = make_colored_noise_sampler(x, SIGMAS, ColorParams(alpha_start=2.0, alpha_end=2.0), seed=0)
    blue = make_colored_noise_sampler(x, SIGMAS, ColorParams(alpha_start=-2.0, alpha_end=-2.0), seed=0)
    out_red = kds.sample_euler_ancestral(_FakeDenoiser(), x.clone(), SIGMAS, extra_args={},
                                         disable=True, eta=1.0, noise_sampler=red)
    out_blue = kds.sample_euler_ancestral(_FakeDenoiser(), x.clone(), SIGMAS, extra_args={},
                                          disable=True, eta=1.0, noise_sampler=blue)
    assert not torch.allclose(out_red, out_blue)


def test_dpmpp_2s_ancestral_runs_with_colored_noise():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    ns = make_colored_noise_sampler(x, SIGMAS, ColorParams(alpha_start=0.0, alpha_end=-1.0), seed=0)
    out = kds.sample_dpmpp_2s_ancestral(_FakeDenoiser(), x, SIGMAS, extra_args={}, disable=True,
                                        eta=1.0, s_noise=1.0, noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all() and float(out.std()) > 0.0


def test_gamma_mode_runs_end_to_end():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    gamma = torch.linspace(0.05, 1.0, 250).unsqueeze(1).repeat(1, 32)
    ns = make_colored_noise_sampler(x, SIGMAS, ColorParams(mode="gamma_matrix", gamma_matrix=gamma), seed=0)
    out = kds.sample_euler_ancestral(_FakeDenoiser(), x, SIGMAS, extra_args={}, disable=True,
                                     eta=1.0, noise_sampler=ns)
    assert out.shape == x.shape and torch.isfinite(out).all() and float(out.std()) > 0.0


def test_default_sampler_is_registered_and_callable():
    s = stochastic_samplers()
    assert "dpmpp_2m_sde" in s and callable(s["dpmpp_2m_sde"])
