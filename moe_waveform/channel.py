from __future__ import annotations

import torch
from torch import nn

from .utils import frame_power


class AWGN(nn.Module):
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        if snr_db.ndim == 1:
            snr_db = snr_db.unsqueeze(-1)
        sig_pow = frame_power(x)
        snr_linear = torch.pow(10.0, snr_db / 10.0)
        noise_pow = sig_pow / snr_linear
        noise = torch.randn_like(x) * noise_pow.sqrt()
        return x + noise
