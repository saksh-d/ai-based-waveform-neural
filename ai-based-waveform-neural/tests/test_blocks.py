from __future__ import annotations

import torch

from moe_waveform.channel import AWGN
from moe_waveform.models import FusionNet, MultiTaskReceiver, Router


def test_awgn_empirical_snr_close():
    torch.manual_seed(0)
    awgn = AWGN()
    batch = 64
    n = 2000
    x = torch.randn(batch, n)
    x = x / x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    target_snr_db = torch.full((batch, 1), 10.0)

    y = awgn(x, target_snr_db)
    noise = y - x
    snr_est = 10.0 * torch.log10(x.pow(2).mean() / noise.pow(2).mean())
    assert abs(float(snr_est.item()) - 10.0) < 0.7


def test_router_outputs_simplex():
    router = Router(hidden_dim=12, residual_scale=0.2, min_weight=0.2)
    inp = torch.randn(10, 1)
    w = router(inp)
    assert w.shape == (10, 2)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_fusion_output_unit_power():
    torch.manual_seed(1)
    fusion = FusionNet(hidden_channels=8)
    x1 = torch.randn(6, 128)
    x2 = torch.randn(6, 128)
    w = torch.softmax(torch.randn(6, 2), dim=-1)
    out = fusion(x1, x2, w)
    power = out.pow(2).mean(dim=-1)
    assert torch.allclose(power, torch.ones_like(power), atol=1e-4, rtol=1e-4)


def test_receiver_head_dimensions(tiny_cfg):
    model = MultiTaskReceiver(tiny_cfg.model, tiny_cfg.signal.ofdm_bits, tiny_cfg.signal.ook_bits)
    y = torch.randn(5, tiny_cfg.signal.frame_len)
    ofdm_logits, ook_logits = model(y)
    assert ofdm_logits.shape == (5, tiny_cfg.signal.ofdm_bits)
    assert ook_logits.shape == (5, tiny_cfg.signal.ook_bits)
