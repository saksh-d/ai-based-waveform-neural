from __future__ import annotations

import torch

from moe_waveform.experts import OFDMExpert, OOKExpert, random_bits


def test_experts_shape_and_power(tiny_cfg):
    device = torch.device("cpu")
    ofdm = OFDMExpert(tiny_cfg.signal).to(device)
    ook = OOKExpert(tiny_cfg.signal).to(device)

    b_ofdm = random_bits(4, tiny_cfg.signal.ofdm_bits, device)
    b_ook = random_bits(4, tiny_cfg.signal.ook_bits, device)

    x_ofdm = ofdm.modulate(b_ofdm)
    x_ook = ook.modulate(b_ook)

    assert x_ofdm.shape == (4, tiny_cfg.signal.frame_len)
    assert x_ook.shape == (4, tiny_cfg.signal.frame_len)
    assert x_ofdm.dtype == torch.float32
    assert x_ook.dtype == torch.float32

    pow_ofdm = x_ofdm.pow(2).mean(dim=-1)
    pow_ook = x_ook.pow(2).mean(dim=-1)
    assert torch.allclose(pow_ofdm, torch.ones_like(pow_ofdm), atol=1e-4, rtol=1e-4)
    assert torch.allclose(pow_ook, torch.ones_like(pow_ook), atol=1e-4, rtol=1e-4)
