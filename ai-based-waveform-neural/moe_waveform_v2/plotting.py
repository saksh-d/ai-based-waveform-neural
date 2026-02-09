from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_v2_task_ber(rows: List[dict], output_path: str | Path, include_avg: bool = True) -> None:
    if not rows:
        return
    xs = [r["snr_db"] for r in rows]
    ber_ofdm = [r["ber_ofdm"] for r in rows]
    ber_ook = [r["ber_ook"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ber_ofdm, marker="o", label="ofdm")
    plt.plot(xs, ber_ook, marker="s", label="ook")
    if include_avg:
        ber_avg = [r["ber_avg"] for r in rows]
        plt.plot(xs, ber_avg, marker="^", label="avg")
    plt.yscale("log")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Task BER vs SNR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_papr_vs_snr(rows: List[dict], output_path: str | Path) -> None:
    if not rows:
        return
    xs = [r["snr_db"] for r in rows]
    papr = [r["papr_db"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, papr, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PAPR (dB)")
    plt.title("PAPR vs SNR")
    plt.grid(True, ls="--", alpha=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_spectral_efficiency_vs_snr(rows: List[dict], output_path: str | Path) -> None:
    if not rows:
        return
    xs = [r["snr_db"] for r in rows]
    eff = [r["spectral_eff_proxy"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, eff, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Goodput/Hz Proxy")
    plt.title("Spectral Efficiency Proxy vs SNR")
    plt.grid(True, ls="--", alpha=0.4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
