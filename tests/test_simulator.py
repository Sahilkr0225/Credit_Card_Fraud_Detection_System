"""
Unit tests for Synthetic Transaction Generator (src/security/simulator.py).
"""

import pytest
from src.security.simulator import SyntheticTransactionGenerator


@pytest.fixture
def generator():
    return SyntheticTransactionGenerator(seed=123)


def test_generate_normal_transaction(generator):
    """Check normal transaction schema and feature bounds."""
    tx = generator.generate_normal_transaction()
    assert tx["scenario_label"] == "NORMAL"
    assert tx["is_ground_truth_fraud"] == 0
    assert tx["sim_swap_detected"] is False
    assert tx["otp_failures_count"] == 0
    assert tx["amount"] > 0
    assert "V1" in tx and "V28" in tx
    assert "Time" in tx and "Amount" in tx


def test_generate_sim_swap_attack(generator):
    """Check SIM-swap attack profile."""
    tx = generator.generate_sim_swap_attack()
    assert tx["scenario_label"] == "SIM_SWAP_TAKEOVER"
    assert tx["is_ground_truth_fraud"] == 1
    assert tx["sim_swap_detected"] is True
    assert tx["sim_swap_age_hours"] < 24.0
    assert tx["amount"] >= 500.0
    assert tx["otp_verified"] is True  # Intercepted OTP!


def test_generate_carding_burst(generator):
    """Check carding burst generator."""
    burst = generator.generate_carding_burst(burst_size=4)
    assert len(burst) == 4
    for tx in burst:
        assert tx["scenario_label"] == "CARDING_BURST"
        assert tx["is_ground_truth_fraud"] == 1
    # Check micro amounts on test charges
    assert burst[0]["amount"] <= 15.0


def test_generate_impossible_travel(generator):
    """Check impossible travel pair."""
    pair = generator.generate_impossible_travel()
    assert len(pair) == 2
    assert pair[0]["city"] != pair[1]["city"]
    assert pair[1]["scenario_label"] == "IMPOSSIBLE_TRAVEL"


def test_generate_otp_brute_force_attack(generator):
    """Check OTP brute force transaction."""
    tx = generator.generate_otp_brute_force_attack()
    assert tx["scenario_label"] == "OTP_BRUTE_FORCE"
    assert tx["otp_failures_count"] >= 3
    assert not tx["otp_verified"]


def test_generate_batch(generator):
    """Check batch generation and fraud prevalence."""
    batch = generator.generate_batch(size=20, fraud_ratio=0.20)
    assert len(batch) >= 20
    scenarios = set(tx["scenario_label"] for tx in batch)
    assert "NORMAL" in scenarios
