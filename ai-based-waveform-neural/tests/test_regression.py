from __future__ import annotations

import torch

from moe_waveform.channel import AWGN
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.system import AdditiveMultiTaskModel
from moe_waveform.utils import compute_ber_from_logits, set_seed


def _run_once(cfg):
    set_seed(123)
    device = torch.device("cpu")
    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)
    model = AdditiveMultiTaskModel(cfg).to(device)
    model.eval()

    batch = sample_batch(
        batch_size=cfg.eval.batch_size,
        signal_cfg=cfg.signal,
        ofdm=ofdm,
        ook=ook,
        awgn=awgn,
        snr_min_db=5.0,
        snr_max_db=5.0,
        device=device,
    )
    with torch.no_grad():
        out = model(batch, awgn)
    ber = 0.5 * (
        compute_ber_from_logits(out["logits_ofdm"], batch.b_ofdm)
        + compute_ber_from_logits(out["logits_ook"], batch.b_ook)
    )
    return float(ber)


def test_deterministic_seed_reproducibility(tiny_cfg):
    ber1 = _run_once(tiny_cfg)
    ber2 = _run_once(tiny_cfg)
    assert abs(ber1 - ber2) < 1e-8


def test_baseline_metric_in_valid_range(tiny_cfg):
    ber = _run_once(tiny_cfg)
    assert 0.0 <= ber <= 1.0
