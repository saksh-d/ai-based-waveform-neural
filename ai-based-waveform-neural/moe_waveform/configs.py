from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List
import json


@dataclass
class SignalConfig:
    n_fft: int = 64
    cp_len: int = 16
    n_ofdm_symbols: int = 8
    active_subcarriers: int = 16
    ook_sps: int = 8
    ofdm_carrier_freq: float = 0.18
    ook_carrier_freq: float = 0.06

    @property
    def ofdm_symbol_len(self) -> int:
        return self.n_fft + self.cp_len

    @property
    def frame_len(self) -> int:
        return self.n_ofdm_symbols * self.ofdm_symbol_len

    @property
    def ofdm_bits(self) -> int:
        return self.active_subcarriers * self.n_ofdm_symbols * 2

    @property
    def ook_bits(self) -> int:
        return self.frame_len // self.ook_sps

    def validate(self) -> None:
        if self.active_subcarriers <= 0:
            raise ValueError("active_subcarriers must be > 0")
        if self.active_subcarriers >= self.n_fft // 2:
            raise ValueError("active_subcarriers must be < n_fft/2")
        if self.frame_len % self.ook_sps != 0:
            raise ValueError("frame_len must be divisible by ook_sps")
        if not 0.0 < self.ofdm_carrier_freq < 0.5:
            raise ValueError("ofdm_carrier_freq must be in (0, 0.5)")
        if not 0.0 < self.ook_carrier_freq < 0.5:
            raise ValueError("ook_carrier_freq must be in (0, 0.5)")


@dataclass
class ModelConfig:
    router_hidden: int = 32
    router_residual_scale: float = 0.2
    min_expert_weight: float = 0.2
    fusion_hidden: int = 32
    receiver_channels: int = 32
    receiver_hidden: int = 256

    def validate(self) -> None:
        if not 0.0 <= self.min_expert_weight < 0.5:
            raise ValueError("min_expert_weight must be in [0, 0.5)")


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 64
    epochs: int = 40
    steps_per_epoch: int = 200
    warmup_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    snr_min_db: float = -5.0
    snr_max_db: float = 20.0
    curriculum_epochs: int = 8
    curriculum_high_snr_min_db: float = 10.0
    lambda_router_balance: float = 0.01
    lambda_router_prior: float = 0.05
    router_prior_center_db: float = 6.0
    router_prior_temp_db: float = 4.0
    early_stop_patience: int = 8
    val_steps: int = 40
    target_ber: float = 1e-3


@dataclass
class FusionV2Config:
    use_v2_fusion: bool = False
    fusion_operator_hidden: int = 64
    lambda_gate_tv: float = 0.01
    lambda_res_energy: float = 0.01
    lambda_alpha_uniform: float = 0.01
    lambda_alpha_tv: float = 0.005
    lambda_oob: float = 0.05
    spectral_mask_db: float = -25.0
    occupied_bw_power_frac: float = 0.99
    lambda_papr: float = 0.01
    papr_target_db: float = 8.0
    papr_calibration_batches: int = 300
    papr_calibration_quantile: float = 0.7
    papr_target_margin_db: float = 0.3
    papr_target_min_db: float = 4.0
    papr_target_max_db: float = 8.0
    residual_gain_init: float = 0.1
    include_snr_in_fusion: bool = True

    def validate(self) -> None:
        if self.fusion_operator_hidden <= 0:
            raise ValueError("fusion_operator_hidden must be > 0")
        if self.lambda_gate_tv < 0.0:
            raise ValueError("lambda_gate_tv must be >= 0")
        if self.lambda_res_energy < 0.0:
            raise ValueError("lambda_res_energy must be >= 0")
        if self.lambda_alpha_uniform < 0.0:
            raise ValueError("lambda_alpha_uniform must be >= 0")
        if self.lambda_alpha_tv < 0.0:
            raise ValueError("lambda_alpha_tv must be >= 0")
        if self.lambda_oob < 0.0:
            raise ValueError("lambda_oob must be >= 0")
        if self.spectral_mask_db >= 0.0:
            raise ValueError("spectral_mask_db must be < 0")
        if not 0.0 < self.occupied_bw_power_frac < 1.0:
            raise ValueError("occupied_bw_power_frac must be in (0, 1)")
        if self.lambda_papr < 0.0:
            raise ValueError("lambda_papr must be >= 0")
        if self.papr_target_db <= 0.0:
            raise ValueError("papr_target_db must be > 0")
        if self.papr_calibration_batches < 0:
            raise ValueError("papr_calibration_batches must be >= 0")
        if not 0.0 < self.papr_calibration_quantile < 1.0:
            raise ValueError("papr_calibration_quantile must be in (0, 1)")
        if self.papr_target_margin_db < 0.0:
            raise ValueError("papr_target_margin_db must be >= 0")
        if self.papr_target_min_db <= 0.0:
            raise ValueError("papr_target_min_db must be > 0")
        if self.papr_target_max_db <= 0.0:
            raise ValueError("papr_target_max_db must be > 0")
        if self.papr_target_min_db > self.papr_target_max_db:
            raise ValueError("papr_target_min_db must be <= papr_target_max_db")
        if self.residual_gain_init < 0.0:
            raise ValueError("residual_gain_init must be >= 0")


@dataclass
class EvalConfig:
    snr_grid_db: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    eval_batches: int = 60
    batch_size: int = 128


@dataclass
class ExperimentConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    v2: FusionV2Config = field(default_factory=FusionV2Config)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        self.signal.validate()
        self.model.validate()
        self.v2.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = ExperimentConfig(
        signal=SignalConfig(**raw.get("signal", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        v2=FusionV2Config(**raw.get("v2", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )
    cfg.validate()
    return cfg
