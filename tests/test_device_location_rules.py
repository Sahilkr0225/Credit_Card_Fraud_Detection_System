"""
Unit tests for Device and Location Rules (src/security/device_location_rules.py).
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.security.device_location_rules import DeviceLocationRules


@pytest.fixture
def rules():
    return DeviceLocationRules(
        max_physical_speed_kmh=850.0,
        geo_jump_min_distance_km=100.0,
    )


def test_haversine_distance_accuracy(rules):
    """Test Great-Circle distance between New York and London (~5,570 km)."""
    # NYC: 40.7128, -74.0060; London: 51.5074, -0.1278
    dist = rules.haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5650


def test_first_device_trusted(rules):
    """First device registered for a user should not be flagged as new."""
    res = rules.evaluate(
        user_id="USR_DEV_1",
        device_id="DEV_PRIMARY",
    )
    assert not res["new_device_detected"]
    assert "NEW_UNRECOGNIZED_DEVICE" not in res["flags"]


def test_second_device_unrecognized(rules):
    """Subsequent different device for same user must be flagged as unrecognized."""
    rules.evaluate(user_id="USR_DEV_2", device_id="DEV_KNOWN")
    res = rules.evaluate(user_id="USR_DEV_2", device_id="DEV_NEW_UNKNOWN")

    assert res["new_device_detected"] is True
    assert "NEW_UNRECOGNIZED_DEVICE" in res["flags"]


def test_impossible_travel_detection(rules):
    """NYC to London in 15 minutes implies speed > 20,000 km/h and must be blocked."""
    now = datetime.now(timezone.utc)
    u = "USR_TRAVELER"

    # Transaction 1 in New York
    rules.evaluate(
        user_id=u,
        latitude=40.7128,
        longitude=-74.0060,
        city="New York",
        country="USA",
        timestamp=now,
    )

    # Transaction 2 in London 15 minutes later
    res2 = rules.evaluate(
        user_id=u,
        latitude=51.5074,
        longitude=-0.1278,
        city="London",
        country="UK",
        timestamp=now + timedelta(minutes=15),
    )

    assert res2["impossible_travel_detected"] is True
    assert res2["distance_km"] > 5000.0
    assert res2["implied_speed_kmh"] > 10000.0
    assert "IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED" in res2["flags"]
    assert res2["device_location_risk_score"] >= 0.70


def test_plausible_travel_allowed(rules):
    """NYC to Chicago (1150 km) in 3.5 hours (~328 km/h) is plausible via flight."""
    now = datetime.now(timezone.utc)
    u = "USR_FLIGHT"

    rules.evaluate(
        user_id=u,
        latitude=40.7128,
        longitude=-74.0060,
        city="New York",
        country="USA",
        timestamp=now,
    )

    res2 = rules.evaluate(
        user_id=u,
        latitude=41.8781,
        longitude=-87.6298,
        city="Chicago",
        country="USA",
        timestamp=now + timedelta(hours=3.5),
    )

    assert not res2["impossible_travel_detected"]
    assert res2["implied_speed_kmh"] < 850.0


def test_high_risk_merchant_category(rules):
    """Crypto or offshore gambling merchant raises category risk flag."""
    res = rules.evaluate(
        user_id="USR_CRYPTO",
        merchant_category="Crypto Exchange",
    )
    assert "HIGH_RISK_MERCHANT_CATEGORY" in res["flags"]
