from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_ber_curves(results: Dict[str, List[dict]], output_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, rows in results.items():
        xs = [r["snr_db"] for r in rows]
        ys = [r["ber_avg"] for r in rows]
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.yscale("log")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average BER")
    plt.title("BER vs SNR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_router_weights(moe_rows: List[dict], output_path: str | Path) -> None:
    xs = [r["snr_db"] for r in moe_rows]
    w_ofdm = [r.get("w_ofdm", 0.5) for r in moe_rows]
    w_ook = [r.get("w_ook", 0.5) for r in moe_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, w_ofdm, marker="o", label="w_ofdm")
    plt.plot(xs, w_ook, marker="s", label="w_ook")
    plt.xlabel("True SNR (dB)")
    plt.ylabel("Average Router Weight")
    plt.title("Router Weights vs True SNR")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_moe_task_ber_curves(moe_rows: List[dict], output_path: str | Path, include_avg: bool = True) -> None:
    xs = [r["snr_db"] for r in moe_rows]
    ber_ofdm = [r["ber_ofdm"] for r in moe_rows]
    ber_ook = [r["ber_ook"] for r in moe_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ber_ofdm, marker="o", label="ber_ofdm")
    plt.plot(xs, ber_ook, marker="s", label="ber_ook")
    if include_avg:
        ber_avg = [r["ber_avg"] for r in moe_rows]
        plt.plot(xs, ber_avg, marker="^", label="ber_avg")
    plt.yscale("log")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("MTL: Task BER vs SNR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
