from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

from moe_waveform.channel import AWGN
from moe_waveform.configs import ExperimentConfig
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.losses import bce_bits_loss, router_balance_loss, router_prior_loss
from moe_waveform.system import additive_fusion
from moe_waveform.utils import compute_ber_from_logits, ensure_dir, set_seed

from .losses import (
    alpha_entropy,
    alpha_tv_loss,
    alpha_uniform_loss,
    occupied_bandwidth_fraction_per_frame,
    papr_db_per_frame,
    papr_hinge_loss,
    residual_energy_loss,
    spectral_oob_loss,
    spectral_oob_ratio_per_frame,
)
from .models import MoEWaveformModelV2


@dataclass
class TrainArtifactsV2:
    v2_ckpt: Path
    history_path: Path


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _resolve_device(device_name: str) -> tuple[torch.device, str]:
    requested = (device_name or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if _mps_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if requested == "mps":
        if _mps_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    return torch.device("cpu"), "cpu"


def _save_checkpoint(path: Path, model: torch.nn.Module, cfg: ExperimentConfig, best_metric: float) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def _snr_bounds_for_epoch(cfg: ExperimentConfig, epoch: int) -> Tuple[float, float]:
    if cfg.train.curriculum_epochs <= 0:
        return cfg.train.snr_min_db, cfg.train.snr_max_db

    high_min = max(cfg.train.curriculum_high_snr_min_db, cfg.train.snr_min_db)
    if epoch >= cfg.train.curriculum_epochs:
        return cfg.train.snr_min_db, cfg.train.snr_max_db

    if cfg.train.curriculum_epochs == 1:
        return high_min, cfg.train.snr_max_db

    frac = epoch / float(cfg.train.curriculum_epochs - 1)
    min_db = high_min - frac * (high_min - cfg.train.snr_min_db)
    return float(min_db), cfg.train.snr_max_db


def _log_epoch(metrics: Dict[str, float], show_progress: bool) -> None:
    msg = (
        f"[v2] epoch={metrics['epoch'] + 1} loss={metrics['train_loss']:.4f} "
        f"ber_avg={metrics['ber_avg']:.6f} ber_ofdm={metrics['ber_ofdm']:.6f} ber_ook={metrics['ber_ook']:.6f} "
        f"papr={metrics['papr_db_mean']:.3f}dB oob={metrics['oob_ratio']:.4f} "
        f"bw_occ={metrics['bw_occ_frac']:.4f} se={metrics['spectral_eff_proxy']:.4f}"
    )
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg)


def _calibrate_papr_target_db(samples: List[float], cfg: ExperimentConfig) -> float:
    if not samples:
        return float(cfg.v2.papr_target_db)
    values = torch.tensor(samples, dtype=torch.float32)
    q = torch.quantile(values, float(cfg.v2.papr_calibration_quantile)).item()
    target = q - float(cfg.v2.papr_target_margin_db)
    target = max(float(cfg.v2.papr_target_min_db), min(float(cfg.v2.papr_target_max_db), float(target)))
    return float(target)


def _validate_v2(model: MoEWaveformModelV2, cfg: ExperimentConfig, ofdm, ook, awgn, device) -> Dict[str, float]:
    model.eval()
    ber_ofdm: List[float] = []
    ber_ook: List[float] = []
    papr_db_vals: List[float] = []
    oob_vals: List[float] = []
    alpha_entropy_vals: List[float] = []
    bw_occ_vals: List[float] = []
    w_ofdm: List[float] = []
    w_ook: List[float] = []

    with torch.no_grad():
        for _ in range(cfg.train.val_steps):
            batch = sample_batch(
                cfg.train.batch_size,
                cfg.signal,
                ofdm,
                ook,
                awgn,
                cfg.train.snr_min_db,
                cfg.train.snr_max_db,
                device,
            )
            out = model(batch, awgn)
            ber_ofdm.append(compute_ber_from_logits(out["logits_ofdm"], batch.b_ofdm))
            ber_ook.append(compute_ber_from_logits(out["logits_ook"], batch.b_ook))
            papr_db_vals.append(float(papr_db_per_frame(out["x_fused_pre"]).mean().item()))
            oob_vals.append(
                float(
                    spectral_oob_ratio_per_frame(
                        out["x_fused_pre"],
                        batch.x_ofdm,
                        batch.x_ook,
                        spectral_mask_db=cfg.v2.spectral_mask_db,
                    ).mean().item()
                )
            )
            alpha_entropy_vals.append(float(alpha_entropy(out["alpha"]).item()))
            bw_occ_vals.append(
                float(
                    occupied_bandwidth_fraction_per_frame(
                        out["x_fused_pre"],
                        power_frac=cfg.v2.occupied_bw_power_frac,
                    ).mean().item()
                )
            )
            w_ofdm.append(float(out["weights"][:, 0].mean().item()))
            w_ook.append(float(out["weights"][:, 1].mean().item()))

    mean_ofdm = float(sum(ber_ofdm) / len(ber_ofdm))
    mean_ook = float(sum(ber_ook) / len(ber_ook))
    mean_bw_occ = float(sum(bw_occ_vals) / len(bw_occ_vals))
    good_bits = (1.0 - mean_ofdm) * cfg.signal.ofdm_bits + (1.0 - mean_ook) * cfg.signal.ook_bits
    goodput_per_sample = good_bits / float(cfg.signal.frame_len)
    spectral_eff_proxy = goodput_per_sample / max(mean_bw_occ, 1e-8)

    return {
        "ber_ofdm": mean_ofdm,
        "ber_ook": mean_ook,
        "ber_avg": 0.5 * (mean_ofdm + mean_ook),
        "papr_db_mean": float(sum(papr_db_vals) / len(papr_db_vals)),
        "oob_ratio": float(sum(oob_vals) / len(oob_vals)),
        "alpha_entropy": float(sum(alpha_entropy_vals) / len(alpha_entropy_vals)),
        "bw_occ_frac": mean_bw_occ,
        "goodput_per_sample": float(goodput_per_sample),
        "spectral_eff_proxy": float(spectral_eff_proxy),
        "w_ofdm": float(sum(w_ofdm) / len(w_ofdm)),
        "w_ook": float(sum(w_ook) / len(w_ook)),
    }


def train_v2(
    cfg: ExperimentConfig,
    output_dir: str | Path,
    show_progress: bool = True,
) -> TrainArtifactsV2:
    cfg.validate()
    output_dir = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    set_seed(cfg.train.seed)
    requested_device = (cfg.train.device or "auto").lower()
    device, resolved_name = _resolve_device(requested_device)
    if requested_device != "auto" and resolved_name != requested_device:
        print(f"[train_v2] Requested device '{requested_device}' unavailable; using '{resolved_name}'.")
    cfg.train.device = resolved_name

    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)

    model = MoEWaveformModelV2(cfg).to(device)
    rx_opt = torch.optim.Adam(
        model.receiver.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    joint_opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    history: Dict[str, list | float | str | int] = {"v2": []}
    best_metric = float("inf")
    best_state = None
    patience = 0
    papr_target_current_db = float(cfg.v2.papr_target_db)
    papr_target_source = "fallback"
    papr_calibration_samples: List[float] = []

    epoch_iter = tqdm(range(cfg.train.epochs), desc="v2 epochs", leave=True) if show_progress else range(cfg.train.epochs)
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        snr_min_epoch, snr_max_epoch = _snr_bounds_for_epoch(cfg, epoch)
        train_papr_vals: List[float] = []
        train_oob_vals: List[float] = []

        step_iter = (
            tqdm(range(cfg.train.steps_per_epoch), desc=f"v2 e{epoch + 1}", total=cfg.train.steps_per_epoch, leave=False)
            if show_progress
            else range(cfg.train.steps_per_epoch)
        )
        for _ in step_iter:
            batch = sample_batch(
                cfg.train.batch_size,
                cfg.signal,
                ofdm,
                ook,
                awgn,
                snr_min_epoch,
                snr_max_epoch,
                device,
            )

            if epoch < cfg.train.warmup_epochs:
                rx_opt.zero_grad(set_to_none=True)
                x_fused = additive_fusion(batch.x_ofdm, batch.x_ook, alpha=0.5)
                y = awgn(x_fused, batch.snr_db)
                logits_ofdm, logits_ook = model.receiver(y)
                loss = bce_bits_loss(logits_ofdm, batch.b_ofdm) + bce_bits_loss(logits_ook, batch.b_ook)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.receiver.parameters(), cfg.train.grad_clip_norm)
                rx_opt.step()
            else:
                joint_opt.zero_grad(set_to_none=True)
                out = model(batch, awgn)

                if cfg.v2.papr_calibration_batches > 0 and len(papr_calibration_samples) < cfg.v2.papr_calibration_batches:
                    papr_vals = papr_db_per_frame(out["x_fused_pre"]).detach().cpu().tolist()
                    needed = cfg.v2.papr_calibration_batches - len(papr_calibration_samples)
                    papr_calibration_samples.extend(float(v) for v in papr_vals[:needed])
                    if len(papr_calibration_samples) >= cfg.v2.papr_calibration_batches:
                        papr_target_current_db = _calibrate_papr_target_db(papr_calibration_samples, cfg)
                        papr_target_source = "calibrated"

                loss_bits = bce_bits_loss(out["logits_ofdm"], batch.b_ofdm) + bce_bits_loss(out["logits_ook"], batch.b_ook)
                loss_balance = router_balance_loss(out["weights"])
                loss_prior = router_prior_loss(
                    out["weights"],
                    batch.snr_db,
                    cfg.train.router_prior_center_db,
                    cfg.train.router_prior_temp_db,
                )
                loss_res_energy = residual_energy_loss(out["residual"])
                loss_papr = papr_hinge_loss(out["x_fused_pre"], target_db=papr_target_current_db)
                loss_alpha_uniform = alpha_uniform_loss(out["alpha"])
                loss_alpha_tv = alpha_tv_loss(out["alpha"])
                loss_oob = spectral_oob_loss(
                    out["x_fused_pre"],
                    batch.x_ofdm,
                    batch.x_ook,
                    spectral_mask_db=cfg.v2.spectral_mask_db,
                )

                loss = (
                    loss_bits
                    + cfg.train.lambda_router_balance * loss_balance
                    + cfg.train.lambda_router_prior * loss_prior
                    + cfg.v2.lambda_res_energy * loss_res_energy
                    + cfg.v2.lambda_papr * loss_papr
                    + cfg.v2.lambda_alpha_uniform * loss_alpha_uniform
                    + cfg.v2.lambda_alpha_tv * loss_alpha_tv
                    + cfg.v2.lambda_oob * loss_oob
                    + cfg.v2.lambda_gate_tv * loss_alpha_tv
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                joint_opt.step()

                train_papr_vals.append(float(papr_db_per_frame(out["x_fused_pre"]).mean().item()))
                train_oob_vals.append(
                    float(
                        spectral_oob_ratio_per_frame(
                            out["x_fused_pre"],
                            batch.x_ofdm,
                            batch.x_ook,
                            spectral_mask_db=cfg.v2.spectral_mask_db,
                        ).mean().item()
                    )
                )

            running_loss += float(loss.item())
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(loss=f"{loss.item():.4f}")

        metrics = _validate_v2(model, cfg, ofdm, ook, awgn, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / cfg.train.steps_per_epoch
        metrics["snr_min_epoch"] = snr_min_epoch
        metrics["papr_target_db"] = float(papr_target_current_db)
        metrics["train_papr_db"] = float(sum(train_papr_vals) / len(train_papr_vals)) if train_papr_vals else 0.0
        metrics["train_oob_ratio"] = float(sum(train_oob_vals) / len(train_oob_vals)) if train_oob_vals else 0.0
        history["v2"].append(metrics)

        _save_checkpoint(ckpt_dir / "v2_moe_latest.pt", model, cfg, metrics["ber_avg"])
        _log_epoch(metrics, show_progress)

        if metrics["ber_avg"] < best_metric:
            best_metric = metrics["ber_avg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(ckpt_dir / "v2_moe_best.pt", model, cfg, best_metric)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if show_progress:
                    tqdm.write("[v2] early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_ckpt = ckpt_dir / "v2_moe_best.pt"
    _save_checkpoint(final_ckpt, model, cfg, best_metric)

    history["papr_target_db"] = float(papr_target_current_db)
    history["papr_target_source"] = papr_target_source
    history["papr_calibration_samples"] = int(len(papr_calibration_samples))

    history_path = output_dir / "training_history_v2.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return TrainArtifactsV2(v2_ckpt=final_ckpt, history_path=history_path)
