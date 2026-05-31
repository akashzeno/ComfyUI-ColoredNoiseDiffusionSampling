import torch

from src.sampling.initial_noise import Noise_ColoredInitial
from src.engine.colored_noise import ColorParams


def _latent(shape=(3, 4, 16, 16), dtype=torch.float32):
    return {"samples": torch.zeros(shape, dtype=dtype)}


def test_contract_shape_dtype_device_cpu():
    n = Noise_ColoredInitial(seed=42, params=ColorParams())
    out = n.generate_noise(_latent())
    assert out.shape == (3, 4, 16, 16)
    assert out.device == torch.device("cpu")
    assert out.dtype == torch.float32
    assert n.seed == 42


def test_batch_index_subset_matches_full_draw():
    # ComfyUI's determinism guarantee holds within the per-item (batch_index) regime:
    # regenerating a subset of indices reproduces those items bit-for-bit.
    p = ColorParams(mode="parametric", alpha_start=-1.0, alpha_end=-1.0)
    full_lat = _latent((4, 4, 16, 16))
    full_lat["batch_index"] = [0, 1, 2, 3]
    full = Noise_ColoredInitial(7, p).generate_noise(full_lat)
    sub_lat = _latent((2, 4, 16, 16))
    sub_lat["batch_index"] = [1, 3]
    sub = Noise_ColoredInitial(7, p).generate_noise(sub_lat)
    assert torch.allclose(sub[0], full[1], atol=1e-5)
    assert torch.allclose(sub[1], full[3], atol=1e-5)


def test_unit_variance_per_item():
    out = Noise_ColoredInitial(1, ColorParams(alpha_start=2.0, alpha_end=2.0)).generate_noise(_latent())
    for i in range(out.shape[0]):
        assert abs(float(out[i].std()) - 1.0) < 1e-2


def test_dtype_fp16_latent():
    out = Noise_ColoredInitial(1, ColorParams()).generate_noise(_latent(dtype=torch.float16))
    assert out.dtype == torch.float16
