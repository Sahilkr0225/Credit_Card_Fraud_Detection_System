"""
Unit tests for OTP Failure Monitor (src/security/otp_monitor.py).
"""

import pytest
from src.security.otp_monitor import OtpMonitor


@pytest.fixture
def otp_monitor():
    return OtpMonitor(
        consecutive_warning_threshold=2,
        consecutive_block_threshold=3,
        window_minutes=15.0,
    )


def test_otp_baseline_success(otp_monitor):
    """Successful OTP verification with 0 prior failures has zero risk."""
    res = otp_monitor.evaluate(user_id="USR_OK", otp_failures_count=0)
    assert res["otp_risk_score"] == 0.0
    assert not res["brute_force_detected"]
    assert res["recommendation"] == "PROCEED"


def test_single_otp_failure_warning(otp_monitor):
    """Single failed OTP creates slight retry penalty."""
    res = otp_monitor.evaluate(user_id="USR_TYPO", otp_failures_count=1)
    assert 0.15 <= res["otp_risk_score"] <= 0.30
    assert "OTP_SINGLE_FAILURE_RETRY" in res["flags"]
    assert not res["brute_force_detected"]


def test_consecutive_otp_brute_force_exceeded(otp_monitor):
    """3 or more consecutive failures triggers brute force lockout."""
    res = otp_monitor.evaluate(user_id="USR_ATTACK", otp_failures_count=3)
    assert res["otp_risk_score"] >= 0.85
    assert res["brute_force_detected"] is True
    assert "OTP_BRUTE_FORCE_EXCEEDED" in res["flags"]
    assert res["recommendation"] == "LOCK_AUTH_CHANNEL"


def test_stateful_otp_attempt_recording(otp_monitor):
    """Recording failures increments counter, success resets it."""
    u = "USR_STATEFUL"
    assert otp_monitor.record_attempt(u, success=False) == 1
    assert otp_monitor.record_attempt(u, success=False) == 2

    eval_2 = otp_monitor.evaluate(u)
    assert "OTP_ELEVATED_FAILURES" in eval_2["flags"]

    # Success resets consecutive count
    assert otp_monitor.record_attempt(u, success=True) == 0
    eval_reset = otp_monitor.evaluate(u)
    assert eval_reset["brute_force_detected"] is False
