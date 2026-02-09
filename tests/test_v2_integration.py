from __future__ import annotations

import torch
import torch.nn.functional as F

from moe_waveform.channel import AWGN
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.losses import router_balance_loss, router_prior_loss
from moe_waveform_v2.losses import alpha_tv_loss, alpha_uniform_loss, papr_hinge_loss, residual_energy_loss, spectral_oob_loss
from moe_waveform_v2.models import MoEWaveformModelV2


def test_v2_single_training_step_grad_flow(tiny_cfg):
    tiny_cfg.v2.use_v2_fusion = True
    device = torch.device("cpu")
    ofdm = OFDMExpert(tiny_cfg.signal).to(device)
    ook = OOKExpert(tiny_cfg.signal).to(device)
    awgn = AWGN().to(device)
    model = MoEWaveformModelV2(tiny_cfg).to(device)

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
        + 0.01 * residual_energy_loss(out["residual"])
        + 0.01 * papr_hinge_loss(out["x_fused_pre"], target_db=6.0)
        + 0.01 * alpha_uniform_loss(out["alpha"])
        + 0.01 * alpha_tv_loss(out["alpha"])
        + 0.01 * spectral_oob_loss(out["x_fused_pre"], batch.x_ofdm, batch.x_ook, spectral_mask_db=-25.0)
    )
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "No gradients found for V2 model"
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(loss)
