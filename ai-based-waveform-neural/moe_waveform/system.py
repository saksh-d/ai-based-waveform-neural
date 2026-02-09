from __future__ import annotations

import torch
from torch import nn

from .configs import ExperimentConfig
from .models import FusionNet, MultiTaskReceiver, Router, SingleHeadReceiver
from .utils import unit_power_normalize


def additive_fusion(x_ofdm: torch.Tensor, x_ook: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return unit_power_normalize(alpha * x_ofdm + (1.0 - alpha) * x_ook)


class MoEWaveformModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.snr_min_db = cfg.train.snr_min_db
        self.snr_max_db = cfg.train.snr_max_db
        self.router = Router(
            cfg.model.router_hidden,
            residual_scale=cfg.model.router_residual_scale,
            min_weight=cfg.model.min_expert_weight,
        )
        self.fusion = FusionNet(cfg.model.fusion_hidden)
        self.receiver = MultiTaskReceiver(cfg.model, cfg.signal.ofdm_bits, cfg.signal.ook_bits)

    def _normalize_snr(self, snr_db: torch.Tensor) -> torch.Tensor:
        den = max(self.snr_max_db - self.snr_min_db, 1e-6)
        return ((snr_db - self.snr_min_db) / den) * 2.0 - 1.0

    def forward(self, batch, awgn) -> dict[str, torch.Tensor]:
        snr_norm = self._normalize_snr(batch.snr_db)
        weights = self.router(snr_norm)

        x_fused = self.fusion(batch.x_ofdm, batch.x_ook, weights)
        y = awgn(x_fused, batch.snr_db)
        logits_ofdm, logits_ook = self.receiver(y)

        return {
            "snr_norm": snr_norm,
            "weights": weights,
            "x_fused": x_fused,
            "y": y,
            "logits_ofdm": logits_ofdm,
            "logits_ook": logits_ook,
        }


class AdditiveMultiTaskModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.receiver = MultiTaskReceiver(cfg.model, cfg.signal.ofdm_bits, cfg.signal.ook_bits)

    def forward(self, batch, awgn) -> dict[str, torch.Tensor]:
        x_fused = additive_fusion(batch.x_ofdm, batch.x_ook, alpha=0.5)
        y = awgn(x_fused, batch.snr_db)
        logits_ofdm, logits_ook = self.receiver(y)
        return {
            "x_fused": x_fused,
            "y": y,
            "logits_ofdm": logits_ofdm,
            "logits_ook": logits_ook,
        }


class AdditiveSingleHeadModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        total_bits = cfg.signal.ofdm_bits + cfg.signal.ook_bits
        self.receiver = SingleHeadReceiver(cfg.model, total_bits)

    def forward(self, batch, awgn) -> dict[str, torch.Tensor]:
        x_fused = additive_fusion(batch.x_ofdm, batch.x_ook, alpha=0.5)
        y = awgn(x_fused, batch.snr_db)
        logits = self.receiver(y)
        return {
            "x_fused": x_fused,
            "y": y,
            "logits": logits,
        }
