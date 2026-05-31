import math
import os

import torch

from src.engine import schedule


def test_progress_monotonic_and_clamped():
    assert schedule.progress_from_sigma(14.6, 14.6, 0.03) == 0.0
    assert schedule.progress_from_sigma(0.03, 14.6, 0.03) == 1.0
    mid = schedule.progress_from_sigma(0.6, 14.6, 0.03)
    assert 0.0 < mid < 1.0
    assert schedule.progress_from_sigma(99.0, 14.6, 0.03) == 0.0   # clamp high
    assert schedule.progress_from_sigma(0.0, 14.6, 0.03) == 1.0    # clamp low/zero


def test_interp_alpha_endpoints_and_exp():
    assert schedule.interp_alpha(0.0, -2.0, 0.0) == 0.0
    assert schedule.interp_alpha(0.0, -2.0, 1.0) == -2.0
    assert math.isclose(schedule.interp_alpha(0.0, -2.0, 0.5), -1.0)
    e = schedule.interp_alpha(0.0, -2.0, 0.5, mode="exponential", sharpness=4.0)
    assert -2.0 <= e <= 0.0 and e != -1.0


def test_gamma_row_at_interpolates():
    g = torch.stack([torch.zeros(32), torch.ones(32)])  # 2 rows
    assert torch.allclose(schedule.gamma_row_at(g, 0.0), torch.zeros(32))
    assert torch.allclose(schedule.gamma_row_at(g, 1.0), torch.ones(32))
    assert torch.allclose(schedule.gamma_row_at(g, 0.5), torch.full((32,), 0.5))


def test_gamma_scaling_residual():
    row = torch.tensor([0.0, 0.5, 1.0])
    s = schedule.gamma_scaling(row, divider=1.0, shaping="none")
    assert torch.allclose(s, torch.tensor([1.0, 0.5, 0.0]))


def test_load_real_gamma_matrix():
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # pkg above resolves to .../ComfyUI/custom_nodes ; rebuild to the package dir:
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(pkg_dir, "gamma_matrices", "gamma_matrix_scaled.pt")
    g = schedule.load_gamma_matrix(p)
    assert g.shape == (250, 32) and g.dtype == torch.float32
