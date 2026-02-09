from __future__ import annotations

import pytest

from moe_waveform.configs import EvalConfig, ExperimentConfig, ModelConfig, SignalConfig, TrainConfig


@pytest.fixture()
def tiny_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig(
        signal=SignalConfig(
            n_fft=32,
            cp_len=8,
            n_ofdm_symbols=4,
            active_subcarriers=8,
            ook_sps=8,
            ofdm_carrier_freq=0.18,
            ook_carrier_freq=0.06,
        ),
        model=ModelConfig(
            router_hidden=16,
            router_residual_scale=0.2,
            min_expert_weight=0.2,
            fusion_hidden=16,
            receiver_channels=8,
            receiver_hidden=64,
        ),
        train=TrainConfig(
            seed=7,
            device="cpu",
            batch_size=8,
            epochs=3,
            steps_per_epoch=3,
            warmup_epochs=1,
            lr=1e-3,
            val_steps=2,
            early_stop_patience=3,
            curriculum_epochs=2,
            curriculum_high_snr_min_db=8.0,
            lambda_router_balance=0.01,
            lambda_router_prior=0.05,
            router_prior_center_db=6.0,
            router_prior_temp_db=4.0,
            target_ber=1e-3,
        ),
        eval=EvalConfig(
            snr_grid_db=[-5.0, 0.0, 5.0],
            eval_batches=2,
            batch_size=8,
        ),
    )
    cfg.validate()
    return cfg
