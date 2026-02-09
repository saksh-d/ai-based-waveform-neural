from __future__ import annotations

import math
import torch
from torch import nn

from .configs import SignalConfig
from .utils import unit_power_normalize


class OFDMExpert(nn.Module):
    def __init__(self, cfg: SignalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        data_indices = torch.arange(1, 1 + cfg.active_subcarriers)
        self.register_buffer("data_indices", data_indices, persistent=False)

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        batch = bits.shape[0]
        cfg = self.cfg
        pair = bits.view(batch, cfg.n_ofdm_symbols, cfg.active_subcarriers, 2)
        re = 2.0 * pair[..., 0] - 1.0
        im = 2.0 * pair[..., 1] - 1.0
        qpsk = torch.complex(re, im) / math.sqrt(2.0)

        spectrum = torch.zeros(
            batch,
            cfg.n_ofdm_symbols,
            cfg.n_fft,
            dtype=torch.complex64,
            device=bits.device,
        )
        spectrum[:, :, self.data_indices] = qpsk.to(torch.complex64)

        x_td = torch.fft.ifft(spectrum, dim=-1)
        cp = x_td[..., -cfg.cp_len :]
        x_cp = torch.cat([cp, x_td], dim=-1)
        x_bb = x_cp.reshape(batch, cfg.frame_len)

        n = torch.arange(cfg.frame_len, device=bits.device, dtype=torch.float32).unsqueeze(0)
        carrier = torch.exp(1j * 2.0 * math.pi * cfg.ofdm_carrier_freq * n)
        x_passband = torch.real(x_bb * carrier)
        return unit_power_normalize(x_passband.float())


class OOKExpert(nn.Module):
    def __init__(self, cfg: SignalConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # True OOK pulse train: each bit is held for `ook_sps` samples.
        # Use bipolar levels so power normalization remains stable.
        levels = 2.0 * bits - 1.0
        pulse_train = levels.repeat_interleave(cfg.ook_sps, dim=1)
        return unit_power_normalize(pulse_train.float())


def random_bits(batch_size: int, n_bits: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, n_bits), device=device).float()
