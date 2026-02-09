from __future__ import annotations

from moe_waveform.evaluation import summarize_acceptance


def test_summarize_acceptance_moe_only():
    results = {
        "moe": [
            {"snr_db": -5.0, "ber_avg": 0.2},
            {"snr_db": 0.0, "ber_avg": 0.1},
            {"snr_db": 5.0, "ber_avg": 0.01},
        ]
    }
    summary = summarize_acceptance(results, target_ber=1e-3)

    assert summary["comparison_status"] == "baselines_missing"
    assert summary["target"]["best_moe_ber"] == 0.01
    assert summary["acceptance_pass"] is None


def test_summarize_acceptance_full_comparison():
    results = {
        "moe": [
            {"snr_db": -5.0, "ber_avg": 0.3},
            {"snr_db": 0.0, "ber_avg": 0.2},
            {"snr_db": 5.0, "ber_avg": 0.1},
            {"snr_db": 10.0, "ber_avg": 0.01},
            {"snr_db": 15.0, "ber_avg": 0.005},
            {"snr_db": 20.0, "ber_avg": 0.001},
        ],
        "baseline_multitask": [
            {"snr_db": -5.0, "ber_avg": 0.35},
            {"snr_db": 0.0, "ber_avg": 0.25},
            {"snr_db": 5.0, "ber_avg": 0.2},
            {"snr_db": 10.0, "ber_avg": 0.02},
            {"snr_db": 15.0, "ber_avg": 0.01},
            {"snr_db": 20.0, "ber_avg": 0.002},
        ],
        "baseline_singlehead": [
            {"snr_db": -5.0, "ber_avg": 0.4},
            {"snr_db": 0.0, "ber_avg": 0.3},
            {"snr_db": 5.0, "ber_avg": 0.2},
            {"snr_db": 10.0, "ber_avg": 0.03},
            {"snr_db": 15.0, "ber_avg": 0.015},
            {"snr_db": 20.0, "ber_avg": 0.003},
        ],
    }
    summary = summarize_acceptance(results, target_ber=1e-3)

    assert summary["comparison_status"] == "full"
    assert summary["acceptance_pass"] is True
    assert summary["wins"]["vs_baseline_multitask"] == 6
    assert summary["wins"]["vs_baseline_singlehead"] == 6
