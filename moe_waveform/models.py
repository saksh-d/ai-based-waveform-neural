from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .configs import ModelConfig
from .utils import unit_power_normalize


class Router(nn.Module):
    def __init__(self, hidden_dim: int = 32, residual_scale: float = 0.2, min_weight: float = 0.2) -> None:
        super().__init__()
        self.base_gain = nn.Parameter(torch.tensor(2.0))
        self.base_bias = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = residual_scale
        self.min_weight = float(min_weight)
        self.residual = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, snr_norm: torch.Tensor) -> torch.Tensor:
        base_logit = self.base_gain * snr_norm + self.base_bias
        base_logits = torch.cat([base_logit, -base_logit], dim=-1)
        delta = self.residual_scale * self.residual(snr_norm)
        w = F.softmax(base_logits + delta, dim=-1)
        if self.min_weight > 0.0:
            w = self.min_weight + (1.0 - 2.0 * self.min_weight) * w
        return w


class FusionNet(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(5, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        self.residual_gain = nn.Parameter(torch.tensor(0.2))

    def forward(self, x_ofdm: torch.Tensor, x_ook: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w_ofdm = weights[:, 0:1].expand(-1, x_ofdm.shape[-1])
        w_ook = weights[:, 1:2].expand(-1, x_ofdm.shape[-1])

        weighted_sum = w_ofdm * x_ofdm + w_ook * x_ook

        z = torch.stack(
            [
                x_ofdm,
                x_ook,
                weighted_sum,
                w_ofdm,
                w_ook,
            ],
            dim=1,
        )
        residual = torch.tanh(self.net(z).squeeze(1))
        out = weighted_sum + self.residual_gain * residual
        return unit_power_normalize(out)


class SharedReceiverBackbone(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=1, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = y.unsqueeze(1)
        return self.project(self.features(x))


class MultiTaskReceiver(nn.Module):
    def __init__(self, model_cfg: ModelConfig, ofdm_bits: int, ook_bits: int) -> None:
        super().__init__()
        self.backbone = SharedReceiverBackbone(model_cfg.receiver_channels, model_cfg.receiver_hidden)
        self.ofdm_head = nn.Linear(model_cfg.receiver_hidden, ofdm_bits)
        self.ook_head = nn.Linear(model_cfg.receiver_hidden, ook_bits)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(y)
        return self.ofdm_head(feat), self.ook_head(feat)


class SingleHeadReceiver(nn.Module):
    def __init__(self, model_cfg: ModelConfig, total_bits: int) -> None:
        super().__init__()
        self.backbone = SharedReceiverBackbone(model_cfg.receiver_channels, model_cfg.receiver_hidden)
        self.head = nn.Linear(model_cfg.receiver_hidden, total_bits)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(y)
        return self.head(feat)
