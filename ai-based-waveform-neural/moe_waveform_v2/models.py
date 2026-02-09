from __future__ import annotations

import torch
from torch import nn

from moe_waveform.configs import ExperimentConfig, FusionV2Config
from moe_waveform.models import MultiTaskReceiver, Router
from moe_waveform.utils import unit_power_normalize


class FusionNetV2(nn.Module):
    def __init__(self, cfg: FusionV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_ops = 4
        in_channels = 5 + (1 if cfg.include_snr_in_fusion else 0)

        self.conv_interaction = nn.Sequential(
            nn.Conv1d(in_channels, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, 1, kernel_size=1),
        )
        self.alpha_net = nn.Sequential(
            nn.Conv1d(in_channels, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, self.num_ops, kernel_size=1),
        )
        self.residual_net = nn.Sequential(
            nn.Conv1d(in_channels, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, cfg.fusion_operator_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cfg.fusion_operator_hidden, 1, kernel_size=1),
        )
        self.residual_gain = nn.Parameter(torch.tensor(float(cfg.residual_gain_init)))

    def _build_features(
        self,
        x_ofdm: torch.Tensor,
        x_ook: torch.Tensor,
        weights: torch.Tensor,
        snr_norm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_ofdm = weights[:, 0:1].expand(-1, x_ofdm.shape[-1])
        w_ook = weights[:, 1:2].expand(-1, x_ofdm.shape[-1])
        weighted_sum = w_ofdm * x_ofdm + w_ook * x_ook

        channels = [x_ofdm, x_ook, weighted_sum, w_ofdm, w_ook]
        if self.cfg.include_snr_in_fusion:
            if snr_norm is None:
                snr_ch = torch.zeros_like(x_ofdm)
            else:
                snr_ch = snr_norm.expand(-1, x_ofdm.shape[-1])
            channels.append(snr_ch)

        z = torch.stack(channels, dim=1)
        return z, w_ofdm, w_ook, weighted_sum

    def forward(
        self,
        x_ofdm: torch.Tensor,
        x_ook: torch.Tensor,
        weights: torch.Tensor,
        snr_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z, w_ofdm, w_ook, weighted_sum = self._build_features(x_ofdm, x_ook, weights, snr_norm)

        o_add = weighted_sum
        o_mul = x_ofdm * x_ook
        o_diff = torch.abs(x_ofdm - x_ook)
        o_conv = torch.tanh(self.conv_interaction(z).squeeze(1))

        operators = torch.stack([o_add, o_mul, o_diff, o_conv], dim=1)
        alpha_logits = self.alpha_net(z)
        alpha = torch.softmax(alpha_logits, dim=1)

        x_core = (alpha * operators).sum(dim=1)
        residual = torch.tanh(self.residual_net(z).squeeze(1))
        x_fused_pre = x_core + self.residual_gain * residual
        x_fused = unit_power_normalize(x_fused_pre)

        return {
            "x_fused": x_fused,
            "x_fused_pre": x_fused_pre,
            "alpha_logits": alpha_logits,
            "alpha": alpha,
            "operators": operators,
            "x_core": x_core,
            "residual": residual,
        }


class MoEWaveformModelV2(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.snr_min_db = cfg.train.snr_min_db
        self.snr_max_db = cfg.train.snr_max_db
        self.router = Router(
            cfg.model.router_hidden,
            residual_scale=cfg.model.router_residual_scale,
            min_weight=cfg.model.min_expert_weight,
        )
        self.fusion = FusionNetV2(cfg.v2)
        self.receiver = MultiTaskReceiver(cfg.model, cfg.signal.ofdm_bits, cfg.signal.ook_bits)

    def _normalize_snr(self, snr_db: torch.Tensor) -> torch.Tensor:
        den = max(self.snr_max_db - self.snr_min_db, 1e-6)
        return ((snr_db - self.snr_min_db) / den) * 2.0 - 1.0

    def forward(self, batch, awgn) -> dict[str, torch.Tensor]:
        snr_norm = self._normalize_snr(batch.snr_db)
        weights = self.router(snr_norm)

        fused = self.fusion(batch.x_ofdm, batch.x_ook, weights, snr_norm=snr_norm)
        y = awgn(fused["x_fused"], batch.snr_db)
        logits_ofdm, logits_ook = self.receiver(y)

        return {
            "snr_norm": snr_norm,
            "weights": weights,
            "x_fused": fused["x_fused"],
            "x_fused_pre": fused["x_fused_pre"],
            "y": y,
            "logits_ofdm": logits_ofdm,
            "logits_ook": logits_ook,
            "alpha_logits": fused["alpha_logits"],
            "alpha": fused["alpha"],
            "operators": fused["operators"],
            "x_core": fused["x_core"],
            "residual": fused["residual"],
        }
