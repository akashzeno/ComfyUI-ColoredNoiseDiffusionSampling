import asyncio
import os

import pytest
import torch

from comfy.samplers import KSAMPLER

from src.nodes.sampler_select import ColoredNoise_SamplerSelect, resolve_color_params, NONE
from src.nodes.noise import ColoredNoise_Noise, COLOR_ALPHA
from src.sampling.initial_noise import Noise_ColoredInitial


def test_noise_node_execute_returns_colored_noise_object():
    out = ColoredNoise_Noise.execute(noise_seed=123, color="blue", alpha=0.0, energy_scale=1.0)
    obj = out.result[0]
    assert isinstance(obj, Noise_ColoredInitial)
    assert obj.seed == 123
    assert obj.params.alpha_start == COLOR_ALPHA["blue"] == -1.0
    n = obj.generate_noise({"samples": torch.zeros(1, 4, 8, 8)})
    assert n.shape == (1, 4, 8, 8) and n.dtype == torch.float32


def test_noise_node_custom_color_uses_alpha():
    out = ColoredNoise_Noise.execute(noise_seed=0, color="custom", alpha=1.7, energy_scale=1.0)
    assert out.result[0].params.alpha_start == 1.7


def test_sampler_select_execute_returns_ksampler():
    out = ColoredNoise_SamplerSelect.execute(
        base_sampler="dpmpp_2m_sde", eta=1.0, s_noise=1.0, mode="parametric",
        alpha_start=0.0, alpha_end=-1.0, interpolation="linear", exp_sharpness=4.0,
        gamma_matrix=NONE, gamma_divider=1.0, gamma_shaping="none",
        power_gamma=1.0, alpha_tilting=0.0, energy_scale=1.0)
    assert isinstance(out.result[0], KSAMPLER)


def test_resolve_color_params_parametric_and_none_fallback():
    p = resolve_color_params("parametric", 0.0, -1.0, "linear", 4.0, NONE, 1.0, "none", 1.0, 0.0, 1.0)
    assert p.mode == "parametric" and p.gamma_matrix is None
    # gamma mode but 'none' selected -> must fall back to parametric (no crash)
    p2 = resolve_color_params("gamma_matrix", 0.0, -1.0, "linear", 4.0, NONE, 1.0, "none", 1.0, 0.0, 1.0)
    assert p2.mode == "parametric" and p2.gamma_matrix is None


def test_real_loader_registers_three_nodes_v3_only():
    try:
        import nodes  # ComfyUI's real node loader
        import folder_paths
    except Exception:
        pytest.skip("ComfyUI nodes module unavailable in this environment")
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = asyncio.run(nodes.load_custom_node(pkg))
    assert ok is True
    for nid in ["ColoredNoise_SamplerSelect", "ColoredNoise_Noise", "ColoredNoise_KSampler"]:
        assert nid in nodes.NODE_CLASS_MAPPINGS
    assert "colored_noise_gamma" in folder_paths.folder_names_and_paths
