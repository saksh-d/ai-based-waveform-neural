from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm

from .channel import AWGN
from .configs import ExperimentConfig
from .data import sample_batch
from .experts import OFDMExpert, OOKExpert
from .losses import bce_bits_loss, router_balance_loss, router_prior_loss
from .system import AdditiveMultiTaskModel, AdditiveSingleHeadModel, MoEWaveformModel, additive_fusion
from .utils import compute_ber_from_logits, ensure_dir, set_seed


@dataclass
class TrainArtifacts:
    moe_ckpt: Path
    baseline_multi_ckpt: Path | None
    baseline_single_ckpt: Path | None
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


def _validate_moe(model: MoEWaveformModel, cfg: ExperimentConfig, ofdm, ook, awgn, device) -> Dict[str, float]:
    model.eval()
    ber_ofdm: List[float] = []
    ber_ook: List[float] = []
    router_ofdm: List[float] = []
    router_ook: List[float] = []

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
            router_ofdm.append(float(out["weights"][:, 0].mean().item()))
            router_ook.append(float(out["weights"][:, 1].mean().item()))

    mean_ofdm = float(sum(ber_ofdm) / len(ber_ofdm))
    mean_ook = float(sum(ber_ook) / len(ber_ook))
    return {
        "ber_ofdm": mean_ofdm,
        "ber_ook": mean_ook,
        "ber_avg": 0.5 * (mean_ofdm + mean_ook),
        "w_ofdm": float(sum(router_ofdm) / len(router_ofdm)),
        "w_ook": float(sum(router_ook) / len(router_ook)),
    }


def _validate_additive_multitask(model: AdditiveMultiTaskModel, cfg: ExperimentConfig, ofdm, ook, awgn, device) -> Dict[str, float]:
    model.eval()
    ber_ofdm: List[float] = []
    ber_ook: List[float] = []
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
    mean_ofdm = float(sum(ber_ofdm) / len(ber_ofdm))
    mean_ook = float(sum(ber_ook) / len(ber_ook))
    return {"ber_ofdm": mean_ofdm, "ber_ook": mean_ook, "ber_avg": 0.5 * (mean_ofdm + mean_ook)}


def _validate_additive_single(model: AdditiveSingleHeadModel, cfg: ExperimentConfig, ofdm, ook, awgn, device) -> Dict[str, float]:
    model.eval()
    ber_ofdm: List[float] = []
    ber_ook: List[float] = []
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
            logits = out["logits"]
            logits_ofdm = logits[:, : cfg.signal.ofdm_bits]
            logits_ook = logits[:, cfg.signal.ofdm_bits :]
            ber_ofdm.append(compute_ber_from_logits(logits_ofdm, batch.b_ofdm))
            ber_ook.append(compute_ber_from_logits(logits_ook, batch.b_ook))
    mean_ofdm = float(sum(ber_ofdm) / len(ber_ofdm))
    mean_ook = float(sum(ber_ook) / len(ber_ook))
    return {"ber_ofdm": mean_ofdm, "ber_ook": mean_ook, "ber_avg": 0.5 * (mean_ofdm + mean_ook)}


def _epoch_iterator(model_label: str, total_epochs: int, show_progress: bool):
    if not show_progress:
        return range(total_epochs)
    return tqdm(range(total_epochs), desc=f"{model_label} epochs", leave=True)


def _step_iterator(model_label: str, epoch: int, total_steps: int, show_progress: bool):
    if not show_progress:
        return range(total_steps)
    return tqdm(
        range(total_steps),
        desc=f"{model_label} e{epoch + 1}",
        total=total_steps,
        leave=False,
    )


def _log_epoch(model_label: str, metrics: Dict[str, float], show_progress: bool) -> None:
    msg = (
        f"[{model_label}] epoch={metrics['epoch'] + 1} "
        f"loss={metrics['train_loss']:.4f} ber_avg={metrics['ber_avg']:.6f} "
        f"ber_ofdm={metrics['ber_ofdm']:.6f} ber_ook={metrics['ber_ook']:.6f}"
    )
    if "w_ofdm" in metrics:
        msg += f" w_ofdm={metrics['w_ofdm']:.3f} w_ook={metrics['w_ook']:.3f}"

    if show_progress:
        tqdm.write(msg)
    else:
        print(msg)


def train_all(
    cfg: ExperimentConfig,
    output_dir: str | Path,
    moe_only: bool = False,
    show_progress: bool = True,
) -> TrainArtifacts:
    cfg.validate()
    output_dir = ensure_dir(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    set_seed(cfg.train.seed)
    requested_device = (cfg.train.device or "auto").lower()
    device, resolved_name = _resolve_device(requested_device)
    if requested_device != "auto" and resolved_name != requested_device:
        print(f"[train] Requested device '{requested_device}' unavailable; using '{resolved_name}'.")
    cfg.train.device = resolved_name

    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)

    history: Dict[str, list] = {"moe": []}

    moe_model = MoEWaveformModel(cfg).to(device)
    moe_model, moe_hist, moe_best = _train_moe(cfg, moe_model, ofdm, ook, awgn, device, ckpt_dir, show_progress)
    history["moe"] = moe_hist
    moe_ckpt = ckpt_dir / "moe_best.pt"
    _save_checkpoint(moe_ckpt, moe_model, cfg, moe_best)

    bm_ckpt: Path | None = None
    bs_ckpt: Path | None = None

    if not moe_only:
        history["baseline_multitask"] = []
        history["baseline_singlehead"] = []

        baseline_multi = AdditiveMultiTaskModel(cfg).to(device)
        baseline_multi, bm_hist, bm_best = _train_baseline_multitask(
            cfg,
            baseline_multi,
            ofdm,
            ook,
            awgn,
            device,
            ckpt_dir,
            show_progress,
        )
        history["baseline_multitask"] = bm_hist
        bm_ckpt = ckpt_dir / "baseline_multitask_best.pt"
        _save_checkpoint(bm_ckpt, baseline_multi, cfg, bm_best)

        baseline_single = AdditiveSingleHeadModel(cfg).to(device)
        baseline_single, bs_hist, bs_best = _train_baseline_singlehead(
            cfg,
            baseline_single,
            ofdm,
            ook,
            awgn,
            device,
            ckpt_dir,
            show_progress,
        )
        history["baseline_singlehead"] = bs_hist
        bs_ckpt = ckpt_dir / "baseline_singlehead_best.pt"
        _save_checkpoint(bs_ckpt, baseline_single, cfg, bs_best)

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return TrainArtifacts(
        moe_ckpt=moe_ckpt,
        baseline_multi_ckpt=bm_ckpt,
        baseline_single_ckpt=bs_ckpt,
        history_path=history_path,
    )


def _train_moe(cfg, model, ofdm, ook, awgn, device, ckpt_dir: Path, show_progress: bool):
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

    best_metric = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in _epoch_iterator("moe", cfg.train.epochs, show_progress):
        model.train()
        running_loss = 0.0
        snr_min_epoch, snr_max_epoch = _snr_bounds_for_epoch(cfg, epoch)

        step_iter = _step_iterator("moe", epoch, cfg.train.steps_per_epoch, show_progress)
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
                loss_bits = bce_bits_loss(out["logits_ofdm"], batch.b_ofdm) + bce_bits_loss(out["logits_ook"], batch.b_ook)
                loss_balance = router_balance_loss(out["weights"])
                loss_prior = router_prior_loss(
                    out["weights"],
                    batch.snr_db,
                    cfg.train.router_prior_center_db,
                    cfg.train.router_prior_temp_db,
                )
                loss = (
                    loss_bits
                    + cfg.train.lambda_router_balance * loss_balance
                    + cfg.train.lambda_router_prior * loss_prior
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                joint_opt.step()

            running_loss += float(loss.item())
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(loss=f"{loss.item():.4f}")

        metrics = _validate_moe(model, cfg, ofdm, ook, awgn, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / cfg.train.steps_per_epoch
        metrics["snr_min_epoch"] = snr_min_epoch
        history.append(metrics)

        _save_checkpoint(ckpt_dir / "moe_latest.pt", model, cfg, metrics["ber_avg"])
        _log_epoch("moe", metrics, show_progress)

        if metrics["ber_avg"] < best_metric:
            best_metric = metrics["ber_avg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(ckpt_dir / "moe_best.pt", model, cfg, best_metric)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if show_progress:
                    tqdm.write("[moe] early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_metric


def _train_baseline_multitask(cfg, model, ofdm, ook, awgn, device, ckpt_dir: Path, show_progress: bool):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_metric = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in _epoch_iterator("baseline_multitask", cfg.train.epochs, show_progress):
        model.train()
        running_loss = 0.0
        snr_min_epoch, snr_max_epoch = _snr_bounds_for_epoch(cfg, epoch)

        step_iter = _step_iterator("baseline_multitask", epoch, cfg.train.steps_per_epoch, show_progress)
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
            opt.zero_grad(set_to_none=True)
            out = model(batch, awgn)
            loss = bce_bits_loss(out["logits_ofdm"], batch.b_ofdm) + bce_bits_loss(out["logits_ook"], batch.b_ook)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            opt.step()
            running_loss += float(loss.item())
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(loss=f"{loss.item():.4f}")

        metrics = _validate_additive_multitask(model, cfg, ofdm, ook, awgn, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / cfg.train.steps_per_epoch
        metrics["snr_min_epoch"] = snr_min_epoch
        history.append(metrics)

        _save_checkpoint(ckpt_dir / "baseline_multitask_latest.pt", model, cfg, metrics["ber_avg"])
        _log_epoch("baseline_multitask", metrics, show_progress)

        if metrics["ber_avg"] < best_metric:
            best_metric = metrics["ber_avg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(ckpt_dir / "baseline_multitask_best.pt", model, cfg, best_metric)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if show_progress:
                    tqdm.write("[baseline_multitask] early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_metric


def _train_baseline_singlehead(cfg, model, ofdm, ook, awgn, device, ckpt_dir: Path, show_progress: bool):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_metric = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in _epoch_iterator("baseline_singlehead", cfg.train.epochs, show_progress):
        model.train()
        running_loss = 0.0
        snr_min_epoch, snr_max_epoch = _snr_bounds_for_epoch(cfg, epoch)

        step_iter = _step_iterator("baseline_singlehead", epoch, cfg.train.steps_per_epoch, show_progress)
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
            labels = torch.cat([batch.b_ofdm, batch.b_ook], dim=-1)
            opt.zero_grad(set_to_none=True)
            out = model(batch, awgn)
            loss = bce_bits_loss(out["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            opt.step()
            running_loss += float(loss.item())
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(loss=f"{loss.item():.4f}")

        metrics = _validate_additive_single(model, cfg, ofdm, ook, awgn, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / cfg.train.steps_per_epoch
        metrics["snr_min_epoch"] = snr_min_epoch
        history.append(metrics)

        _save_checkpoint(ckpt_dir / "baseline_singlehead_latest.pt", model, cfg, metrics["ber_avg"])
        _log_epoch("baseline_singlehead", metrics, show_progress)

        if metrics["ber_avg"] < best_metric:
            best_metric = metrics["ber_avg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(ckpt_dir / "baseline_singlehead_best.pt", model, cfg, best_metric)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                if show_progress:
                    tqdm.write("[baseline_singlehead] early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_metric
