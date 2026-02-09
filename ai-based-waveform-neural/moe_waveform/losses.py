from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_bits_loss(logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, bits)


def router_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(weights.mean(dim=0), 1.0 / weights.shape[-1])
    return F.mse_loss(weights.mean(dim=0), target)


def router_prior_loss(
    weights: torch.Tensor,
    snr_db: torch.Tensor,
    center_db: float,
    temp_db: float,
) -> torch.Tensor:
    temp_db = max(float(temp_db), 1e-3)
    target_ofdm = torch.sigmoid((snr_db - center_db) / temp_db)
    target = torch.cat([target_ofdm, 1.0 - target_ofdm], dim=-1)
    return F.mse_loss(weights, target)
