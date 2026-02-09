from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import torch

from .channel import AWGN
from .configs import ExperimentConfig
from .data import sample_batch
from .experts import OFDMExpert, OOKExpert
from .system import AdditiveMultiTaskModel, AdditiveSingleHeadModel, MoEWaveformModel
from .utils import compute_ber_from_logits, ensure_dir, set_seed


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

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps":
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


def _load_model(model_name: str, ckpt_path: str | Path, cfg: ExperimentConfig, device: torch.device):
    if model_name == "moe":
        model = MoEWaveformModel(cfg).to(device)
    elif model_name == "baseline_multitask":
        model = AdditiveMultiTaskModel(cfg).to(device)
    elif model_name == "baseline_singlehead":
        model = AdditiveSingleHeadModel(cfg).to(device)
    else:
        raise ValueError(f"unknown model_name: {model_name}")

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate_checkpoints(
    cfg: ExperimentConfig,
    ckpt_paths: Dict[str, str | Path],
    output_dir: str | Path,
) -> Dict[str, List[dict]]:
    cfg.validate()
    output_dir = ensure_dir(output_dir)
    set_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)

    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)

    all_results: Dict[str, List[dict]] = {}

    for model_name, ckpt_path in ckpt_paths.items():
        model = _load_model(model_name, ckpt_path, cfg, device)
        rows: List[dict] = []
        with torch.no_grad():
            for snr_db in cfg.eval.snr_grid_db:
                ber_ofdm_vals = []
                ber_ook_vals = []
                router_ofdm_vals = []
                router_ook_vals = []

                for _ in range(cfg.eval.eval_batches):
                    batch = sample_batch(
                        cfg.eval.batch_size,
                        cfg.signal,
                        ofdm,
                        ook,
                        awgn,
                        snr_db,
                        snr_db,
                        device,
                    )

                    if model_name == "moe":
                        out = model(batch, awgn)
                        ber_ofdm_vals.append(compute_ber_from_logits(out["logits_ofdm"], batch.b_ofdm))
                        ber_ook_vals.append(compute_ber_from_logits(out["logits_ook"], batch.b_ook))
                        router_ofdm_vals.append(float(out["weights"][:, 0].mean().item()))
                        router_ook_vals.append(float(out["weights"][:, 1].mean().item()))
                    elif model_name == "baseline_multitask":
                        out = model(batch, awgn)
                        ber_ofdm_vals.append(compute_ber_from_logits(out["logits_ofdm"], batch.b_ofdm))
                        ber_ook_vals.append(compute_ber_from_logits(out["logits_ook"], batch.b_ook))
                    else:
                        out = model(batch, awgn)
                        logits = out["logits"]
                        logits_ofdm = logits[:, : cfg.signal.ofdm_bits]
                        logits_ook = logits[:, cfg.signal.ofdm_bits :]
                        ber_ofdm_vals.append(compute_ber_from_logits(logits_ofdm, batch.b_ofdm))
                        ber_ook_vals.append(compute_ber_from_logits(logits_ook, batch.b_ook))

                ber_ofdm = float(sum(ber_ofdm_vals) / len(ber_ofdm_vals))
                ber_ook = float(sum(ber_ook_vals) / len(ber_ook_vals))
                row = {
                    "model": model_name,
                    "snr_db": snr_db,
                    "ber_ofdm": ber_ofdm,
                    "ber_ook": ber_ook,
                    "ber_avg": 0.5 * (ber_ofdm + ber_ook),
                }
                if model_name == "moe":
                    row["w_ofdm"] = float(sum(router_ofdm_vals) / len(router_ofdm_vals))
                    row["w_ook"] = float(sum(router_ook_vals) / len(router_ook_vals))
                rows.append(row)
        all_results[model_name] = rows

    with (Path(output_dir) / "ber_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    summary = summarize_acceptance(all_results, target_ber=cfg.train.target_ber)
    with (Path(output_dir) / "acceptance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return all_results


def _avg_ber(rows: List[dict]) -> float:
    return float(sum(float(r["ber_avg"]) for r in rows) / max(1, len(rows)))


def _map(rows: List[dict]) -> Dict[float, float]:
    return {float(r["snr_db"]): float(r["ber_avg"]) for r in rows}


def summarize_acceptance(results: Dict[str, List[dict]], target_ber: float = 1e-3) -> dict:
    present = sorted(results.keys())
    moe_rows = results.get("moe", [])

    if not moe_rows:
        return {
            "status": "no_moe_results",
            "models_present": present,
        }

    moe_map = _map(moe_rows)
    best_moe_ber = min(moe_map.values()) if moe_map else float("inf")

    summary = {
        "models_present": present,
        "target": {
            "ber_threshold": target_ber,
            "best_moe_ber": best_moe_ber,
            "target_met": bool(best_moe_ber <= target_ber),
        },
    }

    b1_rows = results.get("baseline_multitask")
    b2_rows = results.get("baseline_singlehead")

    if not b1_rows or not b2_rows:
        summary.update(
            {
                "comparison_status": "baselines_missing",
                "avg_ber": {
                    "moe": _avg_ber(moe_rows),
                    "baseline_multitask": _avg_ber(b1_rows) if b1_rows else None,
                    "baseline_singlehead": _avg_ber(b2_rows) if b2_rows else None,
                },
                "wins": {
                    "vs_baseline_multitask": None,
                    "vs_baseline_singlehead": None,
                    "total_points": len(moe_rows),
                },
                "acceptance_pass": None,
                "criteria": {
                    "avg_ber_better_than_both": None,
                    "wins_at_least_4_of_6": None,
                },
            }
        )
        return summary

    b1_map = _map(b1_rows)
    b2_map = _map(b2_rows)

    common = sorted(set(moe_map) & set(b1_map) & set(b2_map))
    if not common:
        summary.update(
            {
                "comparison_status": "insufficient_overlap",
                "acceptance_pass": None,
            }
        )
        return summary

    avg_moe = sum(moe_map[s] for s in common) / len(common)
    avg_b1 = sum(b1_map[s] for s in common) / len(common)
    avg_b2 = sum(b2_map[s] for s in common) / len(common)

    wins_vs_b1 = sum(1 for s in common if moe_map[s] < b1_map[s])
    wins_vs_b2 = sum(1 for s in common if moe_map[s] < b2_map[s])

    pass_avg = avg_moe < avg_b1 and avg_moe < avg_b2
    pass_pointwise = wins_vs_b1 >= 4 and wins_vs_b2 >= 4

    summary.update(
        {
            "comparison_status": "full",
            "avg_ber": {
                "moe": avg_moe,
                "baseline_multitask": avg_b1,
                "baseline_singlehead": avg_b2,
            },
            "wins": {
                "vs_baseline_multitask": wins_vs_b1,
                "vs_baseline_singlehead": wins_vs_b2,
                "total_points": len(common),
            },
            "acceptance_pass": bool(pass_avg and pass_pointwise),
            "criteria": {
                "avg_ber_better_than_both": pass_avg,
                "wins_at_least_4_of_6": pass_pointwise,
            },
        }
    )
    return summary
