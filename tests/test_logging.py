import logging

import torch

from src.sampling.colored_ksampler import build_colored_sampler
from src.sampling.initial_noise import Noise_ColoredInitial
from src.engine.colored_noise import ColorParams, make_colored_noise_sampler

LOGGER = "ComfyUI-ColoredNoiseDiffusionSampling"


def _fake_base(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
               eta=1.0, s_noise=1.0):
    return x


def test_per_run_info_log(caplog):
    sampler = build_colored_sampler(_fake_base, {"eta": 1.0},
                                    ColorParams(alpha_start=0.0, alpha_end=-1.5))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        sampler.sampler_function(object(), torch.randn(1, 4, 8, 8), torch.tensor([10.0, 1.0, 0.0]),
                                 extra_args={"seed": 0}, disable=True)
    msgs = " ".join(caplog.messages)
    assert "colored noise ACTIVE" in msgs
    assert "base=_fake_base" in msgs
    assert "alpha 0.00->-1.50" in msgs
    # the per-run line must be at INFO (visible by default), not DEBUG/WARNING
    assert any(r.levelno == logging.INFO and "colored noise ACTIVE" in r.getMessage()
               for r in caplog.records)


def test_gamma_mode_info_log(caplog):
    gamma = torch.linspace(0.05, 1.0, 250).unsqueeze(1).repeat(1, 32)
    sampler = build_colored_sampler(_fake_base, {"eta": 1.0},
                                    ColorParams(mode="gamma_matrix", gamma_matrix=gamma))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        sampler.sampler_function(object(), torch.randn(1, 4, 8, 8), torch.tensor([10.0, 1.0, 0.0]),
                                 extra_args={"seed": 0}, disable=True)
    assert "gamma_matrix" in " ".join(caplog.messages)


def test_initial_noise_info_log(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        Noise_ColoredInitial(7, ColorParams(alpha_start=1.0, alpha_end=1.0)).generate_noise(
            {"samples": torch.zeros(1, 4, 8, 8)})
    assert any("colored initial noise" in m for m in caplog.messages)


def test_debug_per_step_log(caplog):
    x = torch.randn(1, 4, 8, 8)
    sigmas = torch.tensor([10.0, 1.0, 0.0])
    ns = make_colored_noise_sampler(x, sigmas, ColorParams(), seed=0)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ns(3.0, 1.0)
    assert any("step sigma=" in m for m in caplog.messages)


def test_no_debug_when_level_is_info(caplog):
    x = torch.randn(1, 4, 8, 8)
    sigmas = torch.tensor([10.0, 1.0, 0.0])
    ns = make_colored_noise_sampler(x, sigmas, ColorParams(), seed=0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ns(3.0, 1.0)
    assert not any("step sigma=" in m for m in caplog.messages)
