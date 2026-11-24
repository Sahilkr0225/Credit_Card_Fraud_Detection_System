"""
Unit tests for Unified Risk Assessment Engine (src/risk_engine.py).
"""

import pytest
from src.risk_engine import UnifiedRiskEngine


@pytest.fixture
def risk_engine():
    return UnifiedRiskEngine(w_ml=0.45, w_anomaly=0.15, w_security=0.40)


def test_low_risk_approve(risk_engine):
    """Benign transaction must result in LOW risk tier and APPROVE decision."""
    ml_result = {
        "transaction_id": "TXN_OK",
        "fraud_probability": 0.02,
        "is_anomaly": False,
        "anomaly_score": 0.15,
    }
    sec_result = {
        "transaction_id": "TXN_OK",
        "sim_swap_detected": False,
        "sim_swap_age_hours": 720.0,
        "otp_failures_count": 0,
        "velocity_score": 0.05,
        "carding_flag": False,
        "security_risk_score": 0.05,
        "flags": [],
    }

    assessment = risk_engine.assess(ml_result, sec_result)

    assert assessment["transaction_id"] == "TXN_OK"
    assert assessment["risk_level"] == "LOW"
    assert assessment["decision"] == "APPROVE"
    assert assessment["unified_risk_score"] < 30.0


def test_medium_risk_challenge_otp(risk_engine):
    """Moderately suspicious event results in MEDIUM tier and CHALLENGE_OTP."""
    ml_result = {
        "transaction_id": "TXN_MED",
        "fraud_probability": 0.45,
        "is_anomaly": False,
        "anomaly_score": 0.0,
    }
    sec_result = {
        "transaction_id": "TXN_MED",
        "sim_swap_detected": False,
        "otp_failures_count": 1,
        "velocity_score": 0.40,
        "carding_flag": False,
        "security_risk_score": 0.35,
        "flags": ["OTP_SINGLE_FAILURE_RETRY"],
    }

    assessment = risk_engine.assess(ml_result, sec_result)
    assert assessment["risk_level"] == "MEDIUM"
    assert assessment["decision"] == "CHALLENGE_OTP"
    assert 30.0 <= assessment["unified_risk_score"] < 65.0


def test_critical_hard_override_for_sim_swap(risk_engine):
    """A recent SIM swap under 24 hours forces DECLINE even if ML score is low."""
    ml_result = {
        "transaction_id": "TXN_OVERRIDE",
        "fraud_probability": 0.10,  # ML model hasn't seen this attack vector
        "is_anomaly": False,
        "anomaly_score": 0.0,
    }
    sec_result = {
        "transaction_id": "TXN_OVERRIDE",
        "sim_swap_detected": True,
        "sim_swap_age_hours": 3.5,
        "otp_failures_count": 0,
        "velocity_score": 0.20,
        "carding_flag": False,
        "security_risk_score": 0.90,
        "flags": ["SIM_SWAP_UNDER_24H", "SIM_SWAP_OTP_INTERCEPTION_RISK"],
    }

    assessment = risk_engine.assess(ml_result, sec_result)

    assert assessment["risk_level"] == "CRITICAL"
    assert assessment["decision"] == "DECLINE"
    assert assessment["unified_risk_score"] >= 85.0
    assert "SIM_SWAP_RECENT" in assessment["breakdown"]["critical_flags"]


def test_output_schema_contract_compliance(risk_engine):
    """Verify exact schema keys match API_CONTRACT.md section 4.1."""
    ml_result = {"fraud_probability": 0.5, "is_anomaly": True, "anomaly_score": -0.2}
    sec_result = {"security_risk_score": 0.5, "flags": []}

    assessment = risk_engine.assess(ml_result, sec_result)

    assert "transaction_id" in assessment
    assert "timestamp" in assessment
    assert "unified_risk_score" in assessment
    assert "risk_level" in assessment
    assert "decision" in assessment
    assert "breakdown" in assessment

    breakdown = assessment["breakdown"]
    assert "ml_fraud_probability" in breakdown
    assert "ml_contribution" in breakdown
    assert "security_risk_score" in breakdown
    assert "security_contribution" in breakdown
    assert "critical_flags" in breakdown
