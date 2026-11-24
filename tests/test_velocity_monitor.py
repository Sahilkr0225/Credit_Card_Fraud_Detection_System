"""
Unit tests for Velocity and Carding Monitor (src/security/velocity_monitor.py).
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.security.velocity_monitor import VelocityMonitor


@pytest.fixture
def monitor():
    return VelocityMonitor(
        burst_1m_limit=3,
        burst_5m_limit=5,
        carding_amount_max=15.0,
        carding_count_threshold=3,
        hourly_spend_limit=2000.0,
    )


def test_single_transaction_baseline(monitor):
    """A single isolated transaction should have zero velocity risk."""
    res = monitor.record_and_evaluate(
        user_id="USR_001",
        transaction_id="TXN_001",
        amount=45.0,
    )
    assert res["velocity_score"] == 0.0
    assert not res["carding_flag"]
    assert not res["burst_flag"]
    assert res["tx_count_1m"] == 1
    assert res["tx_count_5m"] == 1


def test_velocity_burst_detection(monitor):
    """Three rapid transactions in 30 seconds triggers velocity burst."""
    now = datetime.now(timezone.utc)
    user = "USR_BURST"

    monitor.record_and_evaluate(user, "T1", 20.0, timestamp=now)
    monitor.record_and_evaluate(user, "T2", 25.0, timestamp=now + timedelta(seconds=10))
    res3 = monitor.record_and_evaluate(user, "T3", 30.0, timestamp=now + timedelta(seconds=20))

    assert res3["burst_flag"] is True
    assert "VELOCITY_BURST_EXCEEDED" in res3["flags"]
    assert res3["velocity_score"] >= 0.40


def test_carding_micro_transaction_detection(monitor):
    """Multiple rapid low-value transactions trigger carding flag."""
    now = datetime.now(timezone.utc)
    user = "USR_CARDING"

    monitor.record_and_evaluate(user, "C1", 2.50, timestamp=now)
    monitor.record_and_evaluate(user, "C2", 4.00, timestamp=now + timedelta(seconds=15))
    res3 = monitor.record_and_evaluate(user, "C3", 3.75, timestamp=now + timedelta(seconds=30))

    assert res3["carding_flag"] is True
    assert "CARDING_MICRO_TRANSACTIONS" in res3["flags"]
    assert res3["velocity_score"] >= 0.50


def test_hourly_spend_spike(monitor):
    """Exceeding hourly spending threshold raises hourly spend spike flag."""
    res = monitor.record_and_evaluate(
        user_id="USR_SPEND",
        transaction_id="S1",
        amount=3500.0,
    )
    assert "HOURLY_SPEND_SPIKE" in res["flags"]
    assert res["spend_1h"] == 3500.0


def test_user_history_isolation_and_clear(monitor):
    """Ensure history is segregated per user and can be cleared."""
    monitor.record_and_evaluate("USER_A", "A1", 100.0)
    monitor.record_and_evaluate("USER_B", "B1", 200.0)

    res_a = monitor.record_and_evaluate("USER_A", "A2", 50.0)
    res_b = monitor.record_and_evaluate("USER_B", "B2", 50.0)

    assert res_a["tx_count_5m"] == 2
    assert res_b["tx_count_5m"] == 2

    monitor.clear("USER_A")
    res_a_fresh = monitor.record_and_evaluate("USER_A", "A3", 50.0)
    assert res_a_fresh["tx_count_5m"] == 1
