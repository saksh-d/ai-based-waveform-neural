from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import torch

from moe_waveform.channel import AWGN
from moe_waveform.configs import ExperimentConfig
from moe_waveform.data import sample_batch
from moe_waveform.experts import OFDMExpert, OOKExpert
from moe_waveform.system import MoEWaveformModel
from moe_waveform.utils import compute_ber_from_logits, ensure_dir, set_seed

from .losses import (
    alpha_entropy,
    occupied_bandwidth_fraction_per_frame,
    papr_db_per_frame,
    spectral_oob_ratio_per_frame,
)
from .models import MoEWaveformModelV2


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


def _load_v2(ckpt_path: str | Path, cfg: ExperimentConfig, device: torch.device) -> MoEWaveformModelV2:
    model = MoEWaveformModelV2(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _load_v1(ckpt_path: str | Path, cfg: ExperimentConfig, device: torch.device) -> MoEWaveformModel:
    model = MoEWaveformModel(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate_v2_checkpoints(
    cfg: ExperimentConfig,
    v2_ckpt: str | Path,
    output_dir: str | Path,
    v1_ckpt: str | Path | None = None,
) -> tuple[Dict[str, List[dict]], Dict[str, list], dict]:
    cfg.validate()
    output_dir = ensure_dir(output_dir)
    set_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)

    ofdm = OFDMExpert(cfg.signal).to(device)
    ook = OOKExpert(cfg.signal).to(device)
    awgn = AWGN().to(device)

    model_v2 = _load_v2(v2_ckpt, cfg, device)
    model_v1 = _load_v1(v1_ckpt, cfg, device) if v1_ckpt else None

    results: Dict[str, List[dict]] = {"v2": []}
    if model_v1 is not None:
        results["v1"] = []

    diagnostics: Dict[str, list] = {"per_snr": []}

    with torch.no_grad():
        for snr_db in cfg.eval.snr_grid_db:
            v2_ber_ofdm_vals: List[float] = []
            v2_ber_ook_vals: List[float] = []
            v2_w_ofdm_vals: List[float] = []
            v2_w_ook_vals: List[float] = []
            papr_vals: List[float] = []
            oob_vals: List[float] = []
            bw_occ_vals: List[float] = []
            alpha_entropy_vals: List[float] = []
            alpha_usage_vals: List[List[float]] = []

            v1_ber_ofdm_vals: List[float] = []
            v1_ber_ook_vals: List[float] = []

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

                out_v2 = model_v2(batch, awgn)
                v2_ber_ofdm_vals.append(compute_ber_from_logits(out_v2["logits_ofdm"], batch.b_ofdm))
                v2_ber_ook_vals.append(compute_ber_from_logits(out_v2["logits_ook"], batch.b_ook))
                v2_w_ofdm_vals.append(float(out_v2["weights"][:, 0].mean().item()))
                v2_w_ook_vals.append(float(out_v2["weights"][:, 1].mean().item()))
                papr_vals.append(float(papr_db_per_frame(out_v2["x_fused_pre"]).mean().item()))
                oob_vals.append(
                    float(
                        spectral_oob_ratio_per_frame(
                            out_v2["x_fused_pre"],
                            batch.x_ofdm,
                            batch.x_ook,
                            spectral_mask_db=cfg.v2.spectral_mask_db,
                        ).mean().item()
                    )
                )
                bw_occ_vals.append(
                    float(
                        occupied_bandwidth_fraction_per_frame(
                            out_v2["x_fused_pre"],
                            power_frac=cfg.v2.occupied_bw_power_frac,
                        ).mean().item()
                    )
                )
                alpha_entropy_vals.append(float(alpha_entropy(out_v2["alpha"]).item()))
                alpha_usage_vals.append(out_v2["alpha"].mean(dim=(0, 2)).detach().cpu().tolist())

                if model_v1 is not None:
                    out_v1 = model_v1(batch, awgn)
                    v1_ber_ofdm_vals.append(compute_ber_from_logits(out_v1["logits_ofdm"], batch.b_ofdm))
                    v1_ber_ook_vals.append(compute_ber_from_logits(out_v1["logits_ook"], batch.b_ook))

            ber_ofdm = float(sum(v2_ber_ofdm_vals) / len(v2_ber_ofdm_vals))
            ber_ook = float(sum(v2_ber_ook_vals) / len(v2_ber_ook_vals))
            bw_occ_frac = float(sum(bw_occ_vals) / len(bw_occ_vals))
            good_bits = (1.0 - ber_ofdm) * cfg.signal.ofdm_bits + (1.0 - ber_ook) * cfg.signal.ook_bits
            goodput_per_sample = good_bits / float(cfg.signal.frame_len)
            spectral_eff_proxy = goodput_per_sample / max(bw_occ_frac, 1e-8)

            n_ops = len(alpha_usage_vals[0]) if alpha_usage_vals else 0
            alpha_usage_mean = [
                float(sum(v[i] for v in alpha_usage_vals) / len(alpha_usage_vals)) for i in range(n_ops)
            ] if n_ops > 0 else []

            row_v2 = {
                "model": "v2",
                "snr_db": snr_db,
                "ber_ofdm": ber_ofdm,
                "ber_ook": ber_ook,
                "ber_avg": 0.5 * (ber_ofdm + ber_ook),
                "w_ofdm": float(sum(v2_w_ofdm_vals) / len(v2_w_ofdm_vals)),
                "w_ook": float(sum(v2_w_ook_vals) / len(v2_w_ook_vals)),
                "papr_db": float(sum(papr_vals) / len(papr_vals)),
                "oob_ratio": float(sum(oob_vals) / len(oob_vals)),
                "alpha_entropy": float(sum(alpha_entropy_vals) / len(alpha_entropy_vals)),
                "alpha_usage_mean": alpha_usage_mean,
                "bw_occ_frac": bw_occ_frac,
                "goodput_per_sample": float(goodput_per_sample),
                "spectral_eff_proxy": float(spectral_eff_proxy),
            }
            results["v2"].append(row_v2)

            diagnostics["per_snr"].append(
                {
                    "snr_db": float(snr_db),
                    "alpha_usage_mean": alpha_usage_mean,
                    "alpha_entropy": row_v2["alpha_entropy"],
                    "oob_ratio": row_v2["oob_ratio"],
                    "papr_db": row_v2["papr_db"],
                    "bw_occ_frac": row_v2["bw_occ_frac"],
                    "spectral_eff_proxy": row_v2["spectral_eff_proxy"],
                }
            )

            if model_v1 is not None:
                row_v1 = {
                    "model": "v1",
                    "snr_db": snr_db,
                    "ber_ofdm": float(sum(v1_ber_ofdm_vals) / len(v1_ber_ofdm_vals)),
                    "ber_ook": float(sum(v1_ber_ook_vals) / len(v1_ber_ook_vals)),
                    "ber_avg": 0.5
                    * (
                        float(sum(v1_ber_ofdm_vals) / len(v1_ber_ofdm_vals))
                        + float(sum(v1_ber_ook_vals) / len(v1_ber_ook_vals))
                    ),
                }
                results["v1"].append(row_v1)

    with (Path(output_dir) / "ber_results_v2.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (Path(output_dir) / "fusion_diagnostics_v2.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    summary = summarize_v2(results, target_ber=cfg.train.target_ber)
    with (Path(output_dir) / "acceptance_summary_v2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return results, diagnostics, summary


def summarize_v2(results: Dict[str, List[dict]], target_ber: float = 1e-3) -> dict:
    v2 = results.get("v2", [])
    if not v2:
        return {"status": "no_v2_results"}

    v2_map = {float(r["snr_db"]): float(r["ber_avg"]) for r in v2}
    best_v2 = min(v2_map.values())
    avg_v2 = sum(v2_map.values()) / len(v2_map)

    summary = {
        "models_present": sorted(results.keys()),
        "target": {
            "ber_threshold": target_ber,
            "best_v2_ber": best_v2,
            "target_met": bool(best_v2 <= target_ber),
        },
        "avg_ber": {"v2": avg_v2},
    }

    v1 = results.get("v1")
    if not v1:
        summary.update(
            {
                "comparison_status": "v1_missing",
                "wins_vs_v1": None,
                "total_points": len(v2_map),
            }
        )
        return summary

    v1_map = {float(r["snr_db"]): float(r["ber_avg"]) for r in v1}
    common = sorted(set(v2_map) & set(v1_map))
    if not common:
        summary.update({"comparison_status": "insufficient_overlap"})
        return summary

    avg_v1 = sum(v1_map[s] for s in common) / len(common)
    wins = sum(1 for s in common if v2_map[s] < v1_map[s])

    summary.update(
        {
            "comparison_status": "full",
            "avg_ber": {
                "v2": sum(v2_map[s] for s in common) / len(common),
                "v1": avg_v1,
            },
            "wins_vs_v1": wins,
            "total_points": len(common),
        }
    )
    return summary
