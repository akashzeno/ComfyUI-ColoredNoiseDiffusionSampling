import torch

from src.engine import colored_noise as cn


def test_scaling_for_progress_parametric_constant_color():
    p = cn.ColorParams(mode="parametric", alpha_start=1.0, alpha_end=1.0)
    s0 = cn.scaling_for_progress(p, 0.0)
    s1 = cn.scaling_for_progress(p, 1.0)
    assert torch.allclose(s0, s1)               # constant alpha => progress-independent


def test_scaling_for_progress_gamma():
    g = torch.stack([torch.zeros(32), torch.ones(32)])
    p = cn.ColorParams(mode="gamma_matrix", gamma_matrix=g)
    s = cn.scaling_for_progress(p, 0.0)
    assert s.shape == (32,) and torch.allclose(s, torch.ones(32))  # residual at gamma=0 is 1


def test_make_colored_noise_sampler_unit_variance():
    x = torch.randn(2, 4, 32, 32)
    sig = torch.tensor([14.6, 5.0, 1.0, 0.0])
    p = cn.ColorParams(mode="parametric", alpha_start=0.0, alpha_end=-2.0)
    ns = cn.make_colored_noise_sampler(x, sig, p, seed=123)
    out = ns(5.0, 1.0)
    assert out.shape == x.shape
    assert abs(float(out.mean())) < 1e-3 and abs(float(out.std()) - 1.0) < 1e-2


def test_make_colored_noise_sampler_deterministic():
    x = torch.randn(1, 4, 16, 16)
    sig = torch.tensor([14.6, 1.0, 0.0])
    p = cn.ColorParams(mode="parametric")
    a = cn.make_colored_noise_sampler(x, sig, p, seed=7)(3.0, 1.0)
    b = cn.make_colored_noise_sampler(x, sig, p, seed=7)(3.0, 1.0)
    assert torch.allclose(a, b)


def test_colored_noise_sampler_sequence_matches_and_advances():
    x = torch.randn(1, 4, 16, 16)
    sig = torch.tensor([14.6, 5.0, 1.0, 0.0])
    p = cn.ColorParams(mode="parametric")
    nsa = cn.make_colored_noise_sampler(x, sig, p, seed=11)
    nsb = cn.make_colored_noise_sampler(x, sig, p, seed=11)
    a1, a2 = nsa(5.0, 1.0), nsa(1.0, 0.0)
    b1, b2 = nsb(5.0, 1.0), nsb(1.0, 0.0)
    assert torch.allclose(a1, b1) and torch.allclose(a2, b2)  # same seed -> identical sequence
    assert not torch.allclose(a1, a2)                          # generator advances between steps
