from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from moe_waveform.channel import AWGN
from moe_waveform.configs import ExperimentConfig, load_experiment_config
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.system import MoEWaveformModel
from moe_waveform.utils import set_seed
from moe_waveform_v2.models import MoEWaveformModelV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize time-domain waveforms for MoE fused model")
    p.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Pipeline version")
    p.add_argument("--config", type=str, default="", help="Path to config JSON")
    p.add_argument("--ckpt", type=str, default="", help="Path to checkpoint")
    p.add_argument("--output", type=str, default="", help="Output plot path")
    p.add_argument("--snr-db", type=float, default=5.0)
    p.add_argument("--window-samples", type=int, default=256, help="Number of initial samples to plot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_output_dir_for(version: str) -> Path:
    return Path("outputs") / version


def _load_cfg(version: str, config_arg: str, output_dir: Path) -> ExperimentConfig:
    if config_arg:
        return load_experiment_config(config_arg)
    if version == "v2":
        cfg_candidates = [output_dir / "config_used.json", output_dir / "config_used_v2.json"]
    else:
        cfg_candidates = [output_dir / "config_used.json"]

    for path in cfg_candidates:
        if path.exists():
            return load_experiment_config(path)
    return ExperimentConfig()


def main() -> None:
    args = parse_args()
    output_dir = _default_output_dir_for(args.version)
    cfg = _load_cfg(args.version, args.config, output_dir)
    if args.device is not None:
        cfg.train.device = args.device
    if args.version == "v2":
        cfg.v2.use_v2_fusion = True
    cfg.validate()

    set_seed(args.seed)
    device = _resolve_device(cfg.train.device)

    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)

    default_ckpt = (
        output_dir / "checkpoints" / ("v2_moe_best.pt" if args.version == "v2" else "moe_best.pt")
    )
    ckpt_path = Path(args.ckpt) if args.ckpt else default_ckpt
    if args.version == "v2":
        model = MoEWaveformModelV2(cfg).to(device)
    else:
        model = MoEWaveformModel(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    batch = sample_batch(
        batch_size=1,
        signal_cfg=cfg.signal,
        ofdm=ofdm,
        ook=ook,
        awgn=awgn,
        snr_min_db=args.snr_db,
        snr_max_db=args.snr_db,
        device=device,
    )

    with torch.no_grad():
        out = model(batch, awgn)

    x_ofdm = batch.x_ofdm[0].detach().cpu().numpy()
    x_ook = batch.x_ook[0].detach().cpu().numpy()
    x_fused = out["x_fused"][0].detach().cpu().numpy()
    y_rx = out["y"][0].detach().cpu().numpy()

    n = min(args.window_samples, len(x_ofdm))
    t = np.arange(n)
    x_ofdm_w = x_ofdm[:n]
    x_ook_w = x_ook[:n]
    x_fused_w = x_fused[:n]
    y_rx_w = y_rx[:n]

    output_path = Path(args.output) if args.output else (
        output_dir
        / "plots"
        / ("time_domain_waveforms_v2.png" if args.version == "v2" else "time_domain_waveforms.png")
    )

    if args.version == "v2":
        alpha = out["alpha"][0].detach().cpu().numpy()
        operators = out["operators"][0].detach().cpu().numpy()
        contrib = alpha * operators
        op_names = ["add", "mul", "diff", "conv"]

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
        axes[0].plot(t, x_ofdm_w, lw=1.1)
        axes[0].set_title("OFDM Expert")
        axes[1].step(t, x_ook_w, where="post", lw=1.3)
        axes[1].set_title("OOK Expert")

        # for i, name in enumerate(op_names):
        #     axes[2].plot(t, alpha[i, :n], lw=1.0, label=f"alpha_{name}")
        # axes[2].set_ylim(0.0, 1.0)
        # axes[2].set_title("Operator Usage Weights alpha_k(t)")
        # axes[2].legend(loc="upper right", ncol=2, fontsize=8)

        # for i, name in enumerate(op_names):
        #     axes[3].plot(t, contrib[i, :n], lw=0.9, label=f"{name}_contrib")
        # axes[3].set_title("Operator Contributions alpha_k(t) * o_k(t)")
        # axes[3].legend(loc="upper right", ncol=2, fontsize=8)

        axes[2].plot(t, x_fused_w, lw=1.0)
        axes[2].set_title("Fused Waveform")
        axes[3].plot(t, y_rx_w, lw=1.0)
        axes[3].set_title(f"Received Waveform (SNR={args.snr_db} dB)")
        axes[3].set_xlabel("Sample")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(t, x_ofdm_w, lw=1.1)
        axes[0].set_title("OFDM Expert")
        axes[1].step(t, x_ook_w, where="post", lw=1.4)
        axes[1].set_title("OOK Expert")
        axes[2].plot(t, x_fused_w, lw=1.0, label="fused raw")
        axes[2].set_title("Fused Waveform")
        axes[2].legend(loc="upper right")
        axes[3].plot(t, y_rx_w, lw=1.0)
        axes[3].set_title(f"Received Waveform (SNR={args.snr_db} dB)")
        axes[3].set_xlabel("Sample")

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    weights = out["weights"][0].detach().cpu().tolist()
    print(f"Saved waveform plot ({args.version}) to: {output_path}")
    print(f"Router weights [OFDM, OOK]: {weights}")
    print(f"True SNR (dB): {args.snr_db:.3f}")
    if args.version == "v2":
        alpha_mean = out["alpha"][0].mean(dim=-1).detach().cpu().tolist()
        print(f"Operator mean usage [add, mul, diff, conv]: {alpha_mean}")


if __name__ == "__main__":
    main()
