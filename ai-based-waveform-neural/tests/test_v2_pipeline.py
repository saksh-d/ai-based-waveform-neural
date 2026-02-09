from __future__ import annotations

from pathlib import Path

from moe_waveform.plotting import plot_router_weights
from moe_waveform_v2.evaluation import evaluate_v2_checkpoints
from moe_waveform_v2.plotting import plot_papr_vs_snr, plot_spectral_efficiency_vs_snr, plot_v2_task_ber
from moe_waveform_v2.training import train_v2


def test_v2_tiny_train_and_eval_pipeline(tiny_cfg, tmp_path: Path):
    cfg = tiny_cfg
    cfg.v2.use_v2_fusion = True
    cfg.train.epochs = 1
    cfg.train.steps_per_epoch = 1
    cfg.train.warmup_epochs = 0
    cfg.train.val_steps = 1
    cfg.train.batch_size = 4
    cfg.eval.snr_grid_db = [-5.0, 5.0]
    cfg.eval.eval_batches = 1
    cfg.eval.batch_size = 4

    artifacts = train_v2(cfg, tmp_path, show_progress=False)
    assert artifacts.v2_ckpt.exists()
    assert artifacts.history_path.exists()

    results, diagnostics, summary = evaluate_v2_checkpoints(
        cfg=cfg,
        v2_ckpt=artifacts.v2_ckpt,
        output_dir=tmp_path / "metrics",
        v1_ckpt=None,
    )

    assert "v2" in results
    assert len(results["v2"]) == len(cfg.eval.snr_grid_db)
    assert diagnostics["per_snr"]
    assert summary["comparison_status"] == "v1_missing"

    plots_dir = tmp_path / "plots"
    plot_v2_task_ber(results["v2"], plots_dir / "ber_vs_snr_v2_tasks.png", include_avg=True)
    plot_router_weights(results["v2"], plots_dir / "router_weights_vs_snr_v2.png")
    plot_papr_vs_snr(results["v2"], plots_dir / "papr_vs_snr_v2.png")
    plot_spectral_efficiency_vs_snr(results["v2"], plots_dir / "spectral_efficiency_vs_snr_v2.png")

    assert (tmp_path / "metrics" / "ber_results_v2.json").exists()
    assert (tmp_path / "metrics" / "fusion_diagnostics_v2.json").exists()
    assert (tmp_path / "metrics" / "acceptance_summary_v2.json").exists()
    assert (plots_dir / "ber_vs_snr_v2_tasks.png").exists()
    assert (plots_dir / "router_weights_vs_snr_v2.png").exists()
    assert (plots_dir / "papr_vs_snr_v2.png").exists()
    assert (plots_dir / "spectral_efficiency_vs_snr_v2.png").exists()
