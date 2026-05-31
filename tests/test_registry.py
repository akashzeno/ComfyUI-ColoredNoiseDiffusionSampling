import torch

import comfy.samplers

from src.sampling import registry, colored_ksampler
from src.engine.colored_noise import ColorParams


def test_stochastic_includes_known_sde_excludes_deterministic_and_gpu():
    s = registry.stochastic_samplers()
    assert "euler_ancestral" in s
    assert "dpmpp_2m_sde" in s and "dpmpp_3m_sde" in s and "dpmpp_sde" in s
    assert "euler" not in s                          # deterministic -> excluded
    assert all(not k.endswith("_gpu") for k in s)    # gpu variants deduped
    # non-standard-signature samplers (sigma_min/sigma_max/n) must be excluded:
    assert "dpm_fast" not in s and "dpm_adaptive" not in s
    # rectified-flow variants are included (Flux/SD3 support):
    assert "euler_ancestral_RF" in s
    assert callable(next(iter(s.values())))


def test_build_colored_sampler_injects_unit_variance_noise_sampler():
    captured = {}

    def fake_base(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  noise_sampler=None, eta=1.0, s_noise=1.0):
        captured["ns"] = noise_sampler
        captured["eta"] = eta
        return x

    sampler = colored_ksampler.build_colored_sampler(
        fake_base, {"eta": 0.7, "s_noise": 1.0, "r": 0.5}, ColorParams())
    assert isinstance(sampler, comfy.samplers.KSAMPLER)

    x = torch.randn(1, 4, 16, 16)
    sigmas = torch.tensor([14.6, 1.0, 0.0])
    sampler.sampler_function(object(), x, sigmas, extra_args={"seed": 1}, disable=True)
    assert captured["eta"] == 0.7                    # accepted kwarg forwarded
    n = captured["ns"](3.0, 1.0)                      # injected sampler returns unit-variance
    assert n.shape == x.shape and abs(float(n.std()) - 1.0) < 1e-2
    assert "r" not in captured                        # unaccepted kwarg filtered out
