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
