from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_waveform.configs import ExperimentConfig, load_experiment_config
from moe_waveform.evaluation import evaluate_checkpoints, summarize_acceptance
from moe_waveform.plotting import plot_ber_curves, plot_moe_task_ber_curves, plot_router_weights
from moe_waveform_v2.evaluation import evaluate_v2_checkpoints
from moe_waveform_v2.plotting import plot_papr_vs_snr, plot_spectral_efficiency_vs_snr, plot_v2_task_ber


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MoE model")
    p.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Pipeline version")
    p.add_argument("--config", type=str, default="", help="Path to config JSON")
    p.add_argument("--output-dir", type=str, default="", help="Output directory")
    p.add_argument("--moe-ckpt", type=str, default="", help="V1 MoE checkpoint")
    p.add_argument("--baseline-multi-ckpt", type=str, default="", help="V1 baseline multitask checkpoint")
    p.add_argument("--baseline-single-ckpt", type=str, default="", help="V1 baseline single-head checkpoint")
    p.add_argument("--v2-ckpt", type=str, default="", help="V2 checkpoint")
    p.add_argument("--v1-ckpt", type=str, default="", help="Optional V1 checkpoint for V2 comparison")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def _default_output_dir_for(version: str) -> Path:
    return Path("outputs") / version


def _load_cfg_for_eval(version: str, config_arg: str, output_dir: Path) -> ExperimentConfig:
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
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir_for(args.version)
    cfg = _load_cfg_for_eval(args.version, args.config, output_dir)
    if args.device is not None:
        cfg.train.device = args.device
    if args.version == "v2":
        cfg.v2.use_v2_fusion = True
    cfg.validate()

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.version == "v1":
        ckpt_dir = output_dir / "checkpoints"
        ckpts = {
            "moe": args.moe_ckpt or str(ckpt_dir / "moe_best.pt"),
            "baseline_multitask": args.baseline_multi_ckpt or str(ckpt_dir / "baseline_multitask_best.pt"),
            "baseline_singlehead": args.baseline_single_ckpt or str(ckpt_dir / "baseline_singlehead_best.pt"),
        }
        available_ckpts = {k: v for k, v in ckpts.items() if Path(v).exists()}
        if not available_ckpts:
            raise SystemExit("No checkpoints found. Provide at least one valid checkpoint path.")

        results = evaluate_checkpoints(cfg, available_ckpts, metrics_dir)

        plot_ber_curves(results, plots_dir / "ber_vs_snr.png")
        if "moe" in results:
            plot_router_weights(results["moe"], plots_dir / "router_weights_vs_snr.png")
            plot_moe_task_ber_curves(results["moe"], plots_dir / "ber_vs_snr_moe_tasks.png", include_avg=True)

        summary = summarize_acceptance(results, target_ber=cfg.train.target_ber)
        with (metrics_dir / "acceptance_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Evaluation complete (v1)")
        print(json.dumps(summary, indent=2))
        return

    v2_ckpt = Path(args.v2_ckpt) if args.v2_ckpt else (output_dir / "checkpoints" / "v2_moe_best.pt")
    if not v2_ckpt.exists():
        raise SystemExit(f"Missing V2 checkpoint: {v2_ckpt}")
    default_v1 = Path("outputs") / "v1" / "checkpoints" / "moe_best.pt"
    v1_ckpt = Path(args.v1_ckpt) if args.v1_ckpt else default_v1
    v1_ckpt_arg = v1_ckpt if v1_ckpt.exists() else None

    results, _, summary = evaluate_v2_checkpoints(
        cfg=cfg,
        v2_ckpt=v2_ckpt,
        output_dir=metrics_dir,
        v1_ckpt=v1_ckpt_arg,
    )

    plot_v2_task_ber(results.get("v2", []), plots_dir / "ber_vs_snr_v2_tasks.png", include_avg=True)
    plot_router_weights(results.get("v2", []), plots_dir / "router_weights_vs_snr_v2.png")
    plot_papr_vs_snr(results.get("v2", []), plots_dir / "papr_vs_snr_v2.png")
    plot_spectral_efficiency_vs_snr(results.get("v2", []), plots_dir / "spectral_efficiency_vs_snr_v2.png")

    with (metrics_dir / "acceptance_summary_v2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete (v2)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
