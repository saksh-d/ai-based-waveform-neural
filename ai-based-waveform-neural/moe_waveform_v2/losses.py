from __future__ import annotations

import torch


def residual_energy_loss(residual: torch.Tensor) -> torch.Tensor:
    return residual.pow(2).mean()


def alpha_uniform_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # KL(alpha || uniform) over operator axis.
    k = alpha.shape[1]
    alpha_safe = alpha.clamp_min(eps)
    log_uniform = -torch.log(torch.tensor(float(k), device=alpha.device, dtype=alpha.dtype))
    kl = alpha_safe * (torch.log(alpha_safe) - log_uniform)
    return kl.sum(dim=1).mean()


def alpha_tv_loss(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.shape[-1] <= 1:
        return alpha.abs().mean() * 0.0
    return torch.abs(alpha[:, :, 1:] - alpha[:, :, :-1]).mean()


def papr_db_per_frame(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = x.pow(2).amax(dim=-1)
    mean = x.pow(2).mean(dim=-1).clamp_min(eps)
    papr_linear = peak / mean
    return 10.0 * torch.log10(papr_linear.clamp_min(eps))


def papr_hinge_loss(x: torch.Tensor, target_db: float = 8.0) -> torch.Tensor:
    papr_db = papr_db_per_frame(x)
    return torch.relu(papr_db - float(target_db)).pow(2).mean()


def spectral_oob_ratio_per_frame(
    x_fused_pre: torch.Tensor,
    x_ofdm: torch.Tensor,
    x_ook: torch.Tensor,
    spectral_mask_db: float = -25.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    fused_pow = torch.fft.rfft(x_fused_pre, dim=-1).abs().pow(2)
    ofdm_pow = torch.fft.rfft(x_ofdm, dim=-1).abs().pow(2)
    ook_pow = torch.fft.rfft(x_ook, dim=-1).abs().pow(2)

    rel = torch.pow(torch.tensor(10.0, device=x_fused_pre.device, dtype=x_fused_pre.dtype), spectral_mask_db / 10.0)
    thr_ofdm = ofdm_pow.amax(dim=-1, keepdim=True) * rel
    thr_ook = ook_pow.amax(dim=-1, keepdim=True) * rel
    inband = (ofdm_pow >= thr_ofdm) | (ook_pow >= thr_ook)

    total = fused_pow.sum(dim=-1).clamp_min(eps)
    oob = fused_pow.masked_fill(inband, 0.0).sum(dim=-1)
    return oob / total


def spectral_oob_loss(
    x_fused_pre: torch.Tensor,
    x_ofdm: torch.Tensor,
    x_ook: torch.Tensor,
    spectral_mask_db: float = -25.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    return spectral_oob_ratio_per_frame(
        x_fused_pre=x_fused_pre,
        x_ofdm=x_ofdm,
        x_ook=x_ook,
        spectral_mask_db=spectral_mask_db,
        eps=eps,
    ).mean()


def alpha_entropy(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    alpha_safe = alpha.clamp_min(eps)
    return -(alpha_safe * torch.log(alpha_safe)).sum(dim=1).mean()


def occupied_bandwidth_fraction_per_frame(
    x: torch.Tensor,
    power_frac: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    pow_spec = torch.fft.rfft(x, dim=-1).abs().pow(2)
    total = pow_spec.sum(dim=-1, keepdim=True).clamp_min(eps)
    sorted_pow, _ = torch.sort(pow_spec, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_pow, dim=-1) / total
    threshold = torch.tensor(power_frac, dtype=x.dtype, device=x.device)
    bins_needed = (cdf < threshold).sum(dim=-1) + 1
    n_bins = pow_spec.shape[-1]
    return bins_needed.to(x.dtype) / float(n_bins)
