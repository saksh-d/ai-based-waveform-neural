from __future__ import annotations

import argparse
from pathlib import Path

from moe_waveform.configs import ExperimentConfig, load_experiment_config
from moe_waveform.training import train_all
from moe_waveform_v2.training import train_v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MoE waveform model")
    p.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Pipeline version")
    p.add_argument("--config", type=str, default="", help="Path to config JSON file")
    p.add_argument("--output-dir", type=str, default="", help="Output directory")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--moe-only", action="store_true", help="Train only the MoE model (v1 only), doesn't compare against baselines")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    return p.parse_args()


def _default_config_for(version: str) -> str:
    if version == "v2":
        return "configs/v2.json"
    return "configs/default.json"


def _default_output_dir_for(version: str) -> Path:
    return Path("outputs") / version


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config else _default_config_for(args.version)
    cfg = load_experiment_config(config_path) if config_path else ExperimentConfig()
    cfg.train.device = args.device
    if args.version == "v2":
        cfg.v2.use_v2_fusion = True
    cfg.validate()

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir_for(args.version)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.version == "v1":
        artifacts = train_all(cfg, output_dir, moe_only=args.moe_only, show_progress=not args.no_progress)
        cfg.save_json(output_dir / "config_used.json")

        print("Training complete (v1)")
        print(f"MoE checkpoint: {artifacts.moe_ckpt}")
        if artifacts.baseline_multi_ckpt is not None:
            print(f"Baseline multitask checkpoint: {artifacts.baseline_multi_ckpt}")
        if artifacts.baseline_single_ckpt is not None:
            print(f"Baseline single-head checkpoint: {artifacts.baseline_single_ckpt}")
        print(f"History: {artifacts.history_path}")
        return

    if args.moe_only:
        raise SystemExit("--moe-only is only supported for --version v1")

    artifacts_v2 = train_v2(cfg, output_dir, show_progress=not args.no_progress)
    # Save both names so old and new docs/commands work.
    cfg.save_json(output_dir / "config_used.json")
    cfg.save_json(output_dir / "config_used_v2.json")

    print("Training complete (v2)")
    print(f"V2 checkpoint: {artifacts_v2.v2_ckpt}")
    print(f"History: {artifacts_v2.history_path}")


if __name__ == "__main__":
    main()
