from __future__ import annotations

from dataclasses import dataclass
import torch

from .channel import AWGN
from .configs import SignalConfig
from .experts import OFDMExpert, OOKExpert, random_bits
from .utils import sample_uniform_snr_db


@dataclass
class Batch:
    b_ofdm: torch.Tensor
    b_ook: torch.Tensor
    x_ofdm: torch.Tensor
    x_ook: torch.Tensor
    snr_db: torch.Tensor


@torch.no_grad()
def sample_batch(
    batch_size: int,
    signal_cfg: SignalConfig,
    ofdm: OFDMExpert,
    ook: OOKExpert,
    awgn: AWGN,
    snr_min_db: float,
    snr_max_db: float,
    device: torch.device,
) -> Batch:
    b_ofdm = random_bits(batch_size, signal_cfg.ofdm_bits, device)
    b_ook = random_bits(batch_size, signal_cfg.ook_bits, device)

    x_ofdm = ofdm.modulate(b_ofdm)
    x_ook = ook.modulate(b_ook)

    snr_db = sample_uniform_snr_db(batch_size, snr_min_db, snr_max_db, device)

    return Batch(
        b_ofdm=b_ofdm,
        b_ook=b_ook,
        x_ofdm=x_ofdm,
        x_ook=x_ook,
        snr_db=snr_db,
    )
