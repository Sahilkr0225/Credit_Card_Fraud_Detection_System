"""
Unit tests for SIM-Swap Detection Engine (src/security/sim_swap_engine.py).
"""

import pytest
from src.security.sim_swap_engine import SimSwapEngine


@pytest.fixture
def engine():
    return SimSwapEngine(
        critical_window_hours=24.0,
        high_risk_window_hours=48.0,
        medium_risk_window_hours=72.0,
        high_value_threshold=500.0,
    )


def test_baseline_no_sim_swap(engine):
    """When no SIM swap has occurred, risk must be minimal."""
    result = engine.evaluate(sim_swap_detected=False)
    assert result["risk_score"] == 0.0
    assert result["risk_tier"] == "LOW"
    assert result["recommendation"] == "ALLOW"
    assert len(result["flags"]) == 0


def test_old_sim_swap_beyond_72h(engine):
    """SIM swap older than 72 hours should be treated as baseline."""
    result = engine.evaluate(sim_swap_detected=True, sim_swap_age_hours=120.0)
    assert result["risk_score"] <= 0.05
    assert result["risk_tier"] == "LOW"
    assert result["recommendation"] == "ALLOW"


def test_critical_sim_swap_under_24h(engine):
    """SIM swap under 24 hours must trigger critical risk tier."""
    result = engine.evaluate(
        sim_swap_detected=True,
        sim_swap_age_hours=8.0,
        amount=100.0,
    )
    assert result["risk_score"] >= 0.85
    assert result["risk_tier"] == "CRITICAL"
    assert "SIM_SWAP_UNDER_24H" in result["flags"]


def test_compounded_high_value_and_new_device(engine):
    """Recent SIM swap compounded with large transfer and new device caps at 1.0."""
    result = engine.evaluate(
        sim_swap_detected=True,
        sim_swap_age_hours=4.5,
        amount=1500.0,
        device_is_new=True,
        otp_verified=True,
    )
    assert result["risk_score"] == 1.0
    assert result["risk_tier"] == "CRITICAL"
    assert result["recommendation"] == "DECLINE"
    assert "SIM_SWAP_HIGH_VALUE_COMPOUND" in result["flags"]
    assert "SIM_SWAP_NEW_DEVICE" in result["flags"]
    assert "SIM_SWAP_OTP_INTERCEPTION_RISK" in result["flags"]


def test_medium_and_high_risk_windows(engine):
    """Test 24-48h (High) and 48-72h (Medium) transition windows."""
    res_high = engine.evaluate(sim_swap_detected=True, sim_swap_age_hours=36.0, amount=50.0)
    assert res_high["risk_tier"] == "HIGH"
    assert "SIM_SWAP_24_TO_48H" in res_high["flags"]

    res_med = engine.evaluate(sim_swap_detected=True, sim_swap_age_hours=60.0, amount=50.0)
    assert res_med["risk_tier"] == "MEDIUM"
    assert "SIM_SWAP_48_TO_72H" in res_med["flags"]
