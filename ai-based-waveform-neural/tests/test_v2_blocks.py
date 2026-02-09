from __future__ import annotations

import torch

from moe_waveform_v2.losses import (
    alpha_tv_loss,
    alpha_uniform_loss,
    occupied_bandwidth_fraction_per_frame,
    papr_db_per_frame,
    papr_hinge_loss,
    residual_energy_loss,
    spectral_oob_loss,
)
from moe_waveform_v2.models import FusionNetV2
from moe_waveform_v2.training import _calibrate_papr_target_db


def test_fusion_v2_output_shape_simplex_and_power(tiny_cfg):
    cfg = tiny_cfg.v2
    cfg.use_v2_fusion = True

    fusion = FusionNetV2(cfg)
    batch = 5
    n = tiny_cfg.signal.frame_len
    x_ofdm = torch.randn(batch, n)
    x_ook = torch.randn(batch, n)
    weights = torch.softmax(torch.randn(batch, 2), dim=-1)
    snr_norm = torch.randn(batch, 1).clamp(-1.0, 1.0)

    out = fusion(x_ofdm, x_ook, weights, snr_norm=snr_norm)
    assert out["x_fused"].shape == (batch, n)
    assert out["x_fused_pre"].shape == (batch, n)
    assert out["alpha_logits"].shape == (batch, 4, n)
    assert out["alpha"].shape == (batch, 4, n)
    assert out["operators"].shape == (batch, 4, n)
    assert out["residual"].shape == (batch, n)
    assert torch.isfinite(out["x_fused"]).all()
    assert torch.isfinite(out["alpha"]).all()
    assert torch.all(out["alpha"] >= 0.0)
    assert torch.allclose(out["alpha"].sum(dim=1), torch.ones(batch, n), atol=1e-5)

    power = out["x_fused"].pow(2).mean(dim=-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4, rtol=1e-4)


def test_v2_regularizers_and_spectral_metrics_have_finite_gradients():
    alpha = torch.softmax(torch.randn(4, 4, 64, requires_grad=True), dim=1)
    residual = torch.randn(4, 64, requires_grad=True)
    x_fused_pre = torch.randn(4, 64, requires_grad=True)
    x_ofdm = torch.randn(4, 64)
    x_ook = torch.randn(4, 64)

    loss = (
        residual_energy_loss(residual)
        + alpha_uniform_loss(alpha)
        + alpha_tv_loss(alpha)
        + papr_hinge_loss(x_fused_pre, target_db=6.0)
        + spectral_oob_loss(x_fused_pre, x_ofdm, x_ook, spectral_mask_db=-25.0)
        + papr_db_per_frame(x_fused_pre).mean() * 0.0
        + occupied_bandwidth_fraction_per_frame(x_fused_pre, power_frac=0.99).mean() * 0.0
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert x_fused_pre.grad is not None and torch.isfinite(x_fused_pre.grad).all()

    bw_occ = occupied_bandwidth_fraction_per_frame(x_fused_pre.detach(), power_frac=0.99)
    assert torch.isfinite(bw_occ).all()
    assert torch.all(bw_occ > 0.0)
    assert torch.all(bw_occ <= 1.0)


def test_papr_calibration_bounds(tiny_cfg):
    tiny_cfg.v2.papr_calibration_quantile = 0.7
    tiny_cfg.v2.papr_target_margin_db = 0.3
    tiny_cfg.v2.papr_target_min_db = 4.0
    tiny_cfg.v2.papr_target_max_db = 8.0
    target = _calibrate_papr_target_db([4.5, 5.0, 5.5, 6.0, 6.5], tiny_cfg)
    assert 4.0 <= target <= 8.0
