from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_waveform.channel import AWGN
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.losses import router_balance_loss, router_prior_loss
from moe_waveform.system import MoEWaveformModel


def test_single_training_step_grad_flow(tiny_cfg):
    device = torch.device("cpu")
    ofdm = OFDMExpert(tiny_cfg.signal).to(device)
    ook = OOKExpert(tiny_cfg.signal).to(device)
    awgn = AWGN().to(device)
    model = MoEWaveformModel(tiny_cfg).to(device)

    batch = sample_batch(
        batch_size=tiny_cfg.train.batch_size,
        signal_cfg=tiny_cfg.signal,
        ofdm=ofdm,
        ook=ook,
        awgn=awgn,
        snr_min_db=-5.0,
        snr_max_db=10.0,
        device=device,
    )

    out = model(batch, awgn)
    loss = (
        F.binary_cross_entropy_with_logits(out["logits_ofdm"], batch.b_ofdm)
        + F.binary_cross_entropy_with_logits(out["logits_ook"], batch.b_ook)
        + 0.01 * router_balance_loss(out["weights"])
        + 0.05
        * router_prior_loss(
            out["weights"],
            batch.snr_db,
            tiny_cfg.train.router_prior_center_db,
            tiny_cfg.train.router_prior_temp_db,
        )
    )
    loss.backward()

    grads = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.append((name, p.grad))

    assert grads, "No gradients found"
    assert all(torch.isfinite(g).all() for _, g in grads)
    assert torch.isfinite(loss)
