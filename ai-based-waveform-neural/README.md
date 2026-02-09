# AI-Based MoE Waveform Neural PoC

PyTorch proof-of-concept for waveform Mixture-of-Experts with two maintained pipelines:
- `v1`: base MoE fusion + baselines,
- `v2`: operator-bank fusion + additional diagnostics (PAPR/OOB/spectral efficiency proxy).

## Project Layout

- `moe_waveform/`: shared core + v1 pipeline implementation.
- `moe_waveform_v2/`: v2 operator-bank models, losses, training, evaluation, plotting.
- `configs/default.json`: v1 default config.
- `configs/v2.json`: v2 default config.
- `train.py`: unified train entrypoint for `v1` and `v2` via `--version`.
- `evaluate.py`: unified evaluate entrypoint for `v1` and `v2` via `--version`.
- `visualize.py`: unified visualization entrypoint for `v1` and `v2` via `--version`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandline Organization

Use the same scripts for both pipelines and select the pipeline with `--version`:

- `--version v1` for base MoE + baselines + MTL decoder.
- `--version v2` for operator-bank fusion on top of v1.

Default output roots are organized as:
- `outputs/v1/...`
- `outputs/v2/...`

## v1 vs v2

| Area | v1 (`moe_waveform/`) | v2 (`moe_waveform_v2/`) |
|---|---|---|
| Experts | Deterministic OFDM + OOK experts | Same OFDM + OOK experts |
| Router | 2-expert SNR-conditioned router (`w_ofdm`, `w_ook`) | Same router design as v1 |
| Fusion block | `FusionNet`: weighted sum + learned residual conv correction | `FusionNetV2`: operator bank (`add`, `mul`, `diff`, learned `conv`) with time-varying operator mixture `alpha_k(t)` |
| Receiver | Multi-task learnign: Shared backbone + two heads (OFDM/OOK) | Same receiver structure |
| Warmup stage | Train receiver on fixed additive fusion | Same warmup idea |
| Joint loss | Bit BCE + router balance + router prior | v1 losses + residual energy + PAPR hinge + alpha uniformity + alpha TV + gate TV + spectral OOB penalty |
| PAPR handling | No explicit PAPR regularization | Explicit PAPR target with optional calibration from early training batches |
| Spectral regularization | None | Explicit OOB spectral penalty and occupied-bandwidth diagnostics |
| Training scope | Trains MoE and optional baselines (`multitask`, `singlehead`) | Trains v2 model only |
| Eval outputs | BER curves + router weights + acceptance summary | BER + router weights + fusion diagnostics (`papr`, `oob`, `alpha`, bandwidth occupancy, spectral efficiency proxy) + v2 acceptance summary |
| Visualization | Expert/fused/received waveforms | v1 waveforms plus operator usage `alpha_k(t)` and operator contribution traces |

In short:
- `v1` is the baseline MoE fusion pipeline with simpler objectives and baseline-model comparisons.
- `v2` keeps the same experts/router/receiver but upgrades fusion to an operator-bank formulation with additional waveform-shaping regularizers and richer diagnostics.

## Train

### Train v1

```bash
python train.py --version v1 --config configs/default.json --output-dir outputs/v1 --device auto
```

Optional MoE-only mode:

```bash
python train.py --version v1 --config configs/default.json --output-dir outputs/v1 --device auto --moe-only
```

v1 artifacts:
- `outputs/v1/checkpoints/moe_best.pt`
- `outputs/v1/checkpoints/baseline_multitask_best.pt`
- `outputs/v1/checkpoints/baseline_singlehead_best.pt`
- `outputs/v1/training_history.json`
- `outputs/v1/config_used.json`

### Train v2

```bash
python train.py --version v2 --config configs/v2.json --output-dir outputs/v2 --device auto
```

v2 artifacts:
- `outputs/v2/checkpoints/v2_moe_best.pt`
- `outputs/v2/training_history_v2.json`
- `outputs/v2/config_used.json`

## Evaluate

### Evaluate v1

```bash
python evaluate.py --version v1 --config outputs/v1/config_used.json --output-dir outputs/v1
```

v1 outputs:
- `outputs/v1/metrics/ber_results.json`
- `outputs/v1/metrics/acceptance_summary.json`
- `outputs/v1/plots/ber_vs_snr.png`
- `outputs/v1/plots/ber_vs_snr_moe_tasks.png`
- `outputs/v1/plots/router_weights_vs_snr.png`

### Evaluate v2

```bash
python evaluate.py --version v2 \
  --config outputs/v2/config_used.json \
  --output-dir outputs/v2 \
  --v2-ckpt outputs/v2/checkpoints/v2_moe_best.pt \
  --v1-ckpt outputs/v1/checkpoints/moe_best.pt
```

v2 outputs:
- `outputs/v2/metrics/ber_results_v2.json`
- `outputs/v2/metrics/fusion_diagnostics_v2.json`
- `outputs/v2/metrics/acceptance_summary_v2.json`
- `outputs/v2/plots/ber_vs_snr_v2_tasks.png`
- `outputs/v2/plots/router_weights_vs_snr_v2.png`
- `outputs/v2/plots/papr_vs_snr_v2.png`
- `outputs/v2/plots/spectral_efficiency_vs_snr_v2.png`

## Visualize

### Visualize v1

```bash
python visualize.py --version v1 \
  --config outputs/v1/config_used.json \
  --ckpt outputs/v1/checkpoints/moe_best.pt \
  --snr-db 5 \
  --window-samples 256 \
  --output outputs/v1/plots/time_domain_waveforms.png
```

### Visualize v2

```bash
python visualize.py --version v2 \
  --config outputs/v2/config_used.json \
  --ckpt outputs/v2/checkpoints/v2_moe_best.pt \
  --snr-db 5 \
  --window-samples 256 \
  --output outputs/v2/plots/time_domain_waveforms_v2.png
```

## Testing

```bash
pytest -q
```

## Notes

- Real passband representation is used for transmitted/fused waveforms.
- OOK expert is a rectangular high/low pulse train held for `ook_sps` samples.
- Per-frame unit-power normalization is enforced for fair BER comparison.
- Router input is true SNR in this PoC to simplify training/debugging.
- V2 adds spectral OOB regularization and reports a goodput-per-Hz proxy.
