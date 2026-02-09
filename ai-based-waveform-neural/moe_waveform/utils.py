from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frame_power(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)


def unit_power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = frame_power(x, eps=eps).sqrt()
    return x / scale


def sample_uniform_snr_db(batch_size: int, min_db: float, max_db: float, device: torch.device) -> torch.Tensor:
    return (min_db + (max_db - min_db) * torch.rand(batch_size, 1, device=device)).float()


def compute_ber_from_logits(logits: torch.Tensor, bits: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).to(bits.dtype)
    errors = (pred != bits).float().mean().item()
    return float(errors)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
