import math
import torch

from src.engine import spectral


def test_radial_bin_map_2d_shape_and_range():
    idx = spectral.radial_bin_map_2d(16, 24, num_bins=32, device=torch.device("cpu"))
    assert idx.shape == (16, 24)
    assert int(idx.min()) == 0 and int(idx.max()) <= 31
    assert idx[0, 0] == 0  # DC is bin 0


def test_radial_bin_map_1d_shape():
    idx = spectral.radial_bin_map_1d(40, num_bins=16, device=torch.device("cpu"))
    assert idx.shape == (40,)
    assert idx[0] == 0


def test_parametric_white_is_flat():
    s = spectral.parametric_scaling(64, alpha=0.0)
    assert torch.allclose(s, torch.ones_like(s))


def test_parametric_red_emphasizes_low_freq():
    s = spectral.parametric_scaling(64, alpha=2.0)
    assert s[0] == 0.0           # DC killed for alpha>0
    assert s[1] > s[-1]          # low-freq amplitude > high-freq


def test_parametric_blue_emphasizes_high_freq():
    s = spectral.parametric_scaling(64, alpha=-2.0)
    assert s[-1] > s[1]          # high-freq amplitude > low-freq
    assert torch.isfinite(s).all()


def _unit(x):
    return abs(float(x.mean())) < 1e-4 and abs(float(x.std()) - 1.0) < 1e-3


def test_color_tensor_white_is_unit_variance_4d():
    w = torch.randn(2, 4, 32, 32)
    s = spectral.parametric_scaling(64, 0.0)
    out = spectral.color_tensor(w, s)
    assert out.shape == w.shape and _unit(out)


def test_color_tensor_5d_video():
    w = torch.randn(1, 16, 3, 24, 24)
    s = spectral.parametric_scaling(64, -1.0)
    out = spectral.color_tensor(w, s)
    assert out.shape == w.shape and _unit(out)


def test_color_tensor_3d_audio_uses_1d():
    w = torch.randn(1, 64, 256)
    s = spectral.parametric_scaling(32, 1.0)
    out = spectral.color_tensor(w, s)
    assert out.shape == w.shape and _unit(out)


def test_color_tensor_2d_returns_unchanged():
    w = torch.randn(8, 8)
    s = spectral.parametric_scaling(16, 1.0)
    out = spectral.color_tensor(w, s)
    assert torch.equal(out, w)  # <3d -> passthrough


def test_color_tensor_dtype_roundtrip_fp16():
    w = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    s = spectral.parametric_scaling(32, 2.0)
    out = spectral.color_tensor(w, s)
    assert out.dtype == torch.float16


def test_color_tensor_red_has_more_lowfreq_energy_than_blue():
    torch.manual_seed(0)
    w = torch.randn(1, 4, 64, 64)
    red = spectral.color_tensor(w, spectral.parametric_scaling(128, 2.0))
    blue = spectral.color_tensor(w, spectral.parametric_scaling(128, -2.0))

    def lowfreq_frac(t):
        m = torch.fft.fftshift(torch.fft.fft2(t[0, 0].float()).abs())
        h, w_ = m.shape
        cy, cx = h // 2, w_ // 2
        center = m[cy - 4:cy + 4, cx - 4:cx + 4].sum()
        return float(center / m.sum())

    assert lowfreq_frac(red) > lowfreq_frac(blue)
