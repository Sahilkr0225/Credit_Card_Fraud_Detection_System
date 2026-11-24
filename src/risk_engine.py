"""
Unified Risk Assessment Engine.

Fuses Machine Learning probabilities, unsupervised anomaly metrics, and
telecom/behavioral security rule signals into a single unified risk score
and automated enforcement decision (APPROVE, CHALLENGE_OTP, MANUAL_REVIEW, DECLINE).
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class UnifiedRiskEngine:
    """
    Weighted multi-factor fraud risk aggregator and decision engine.

    Parameters
    ----------
    w_ml : float, default=0.45
        Weight assigned to supervised ML probability (0.0 to 1.0).
    w_anomaly : float, default=0.15
        Weight assigned to unsupervised anomaly score (0.0 to 1.0).
    w_security : float, default=0.40
        Weight assigned to behavioral/security rules (0.0 to 1.0).
    """

    def __init__(
        self,
        w_ml: float = 0.45,
        w_anomaly: float = 0.15,
        w_security: float = 0.40,
    ) -> None:
        total = w_ml + w_anomaly + w_security
        self.w_ml = w_ml / total
        self.w_anomaly = w_anomaly / total
        self.w_security = w_security / total

    def assess(
        self,
        ml_result: Dict[str, Any],
        security_result: Dict[str, Any],
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute unified risk score and render enforcement decision.

        Parameters
        ----------
        ml_result : dict
            Prediction output matching MLPredictionResult schema.
        security_result : dict
            Telemetry output matching SecuritySignalsResult schema.
        timestamp : datetime, str, or None
            Assessment timestamp.

        Returns
        -------
        dict
            Assessment matching UnifiedRiskAssessment schema.
        """
        tx_id = (
            ml_result.get("transaction_id")
            or security_result.get("transaction_id")
            or "TXN_UNKNOWN"
        )

        # 1. Extract and normalize component scores (all mapped to [0, 100])
        # Supervised ML probability [0.0, 1.0] -> [0, 100]
        ml_prob = float(ml_result.get("fraud_probability", 0.0))
        s_ml = ml_prob * 100.0

        # Unsupervised Anomaly Score
        # If is_anomaly is bool or anomaly_score is negative Isolation Forest score
        is_anomaly = bool(ml_result.get("is_anomaly", False))
        raw_anomaly = ml_result.get("anomaly_score", 0.0)
        if isinstance(raw_anomaly, (int, float)):
            # Isolation forest typical scores: negative means outlier
            if raw_anomaly < 0:
                s_anomaly = min(100.0, abs(raw_anomaly) * 200.0)
            else:
                s_anomaly = 100.0 if is_anomaly else 0.0
        else:
            s_anomaly = 100.0 if is_anomaly else 0.0

        # Security Risk Score [0.0, 1.0] -> [0, 100]
        sec_score = float(security_result.get("security_risk_score", 0.0))
        s_sec = sec_score * 100.0

        # 2. Weighted Multi-Factor Score Calculation
        ml_contrib = round(self.w_ml * s_ml, 2)
        anomaly_contrib = round(self.w_anomaly * s_anomaly, 2)
        sec_contrib = round(self.w_security * s_sec, 2)

        base_score = ml_contrib + anomaly_contrib + sec_contrib

        # 3. Critical Flags and Escalation Rules
        critical_flags: List[str] = []
        flags = security_result.get("flags", [])

        # Security rule flags check
        if security_result.get("sim_swap_detected") and float(security_result.get("sim_swap_age_hours", 999.0)) < 24.0:
            critical_flags.append("SIM_SWAP_RECENT")

        if security_result.get("carding_flag"):
            critical_flags.append("CARDING_PATTERN")

        if "VELOCITY_BURST_EXCEEDED" in flags:
            critical_flags.append("VELOCITY_BURST_EXCEEDED")

        if "IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED" in flags or security_result.get("impossible_travel_detected"):
            critical_flags.append("IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED")

        if int(security_result.get("otp_failures_count", 0)) >= 3 or "OTP_BRUTE_FORCE_EXCEEDED" in flags:
            critical_flags.append("OTP_BRUTE_FORCE_EXCEEDED")

        if ml_prob >= 0.80:
            critical_flags.append("HIGH_ML_FRAUD_PROBABILITY")

        if is_anomaly:
            critical_flags.append("UNSUPERVISED_ANOMALY_OUTLIER")

        # 4. Critical Hard Override Protection
        # A recent SIM swap under 24 hours, physically impossible travel, or OTP brute-force
        # must never be diluted by an innocent ML score on unfamiliar zero-day vectors.
        final_score = base_score
        hard_override_triggers = {
            "SIM_SWAP_RECENT",
            "IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED",
            "OTP_BRUTE_FORCE_EXCEEDED",
        }

        if any(trigger in critical_flags for trigger in hard_override_triggers):
            final_score = max(final_score, 85.0)

        unified_risk_score = round(min(100.0, max(0.0, final_score)), 1)

        # 5. Decision Matrix Mapping
        if unified_risk_score > 80.0:
            risk_level = "CRITICAL"
            decision = "DECLINE"
        elif unified_risk_score >= 65.0:
            risk_level = "HIGH"
            decision = "MANUAL_REVIEW"
        elif unified_risk_score >= 30.0:
            risk_level = "MEDIUM"
            decision = "CHALLENGE_OTP"
        else:
            risk_level = "LOW"
            decision = "APPROVE"

        # Format ISO timestamp
        if isinstance(timestamp, datetime):
            ts_str = timestamp.isoformat()
        elif isinstance(timestamp, str):
            ts_str = timestamp
        else:
            ts_str = datetime.now(timezone.utc).isoformat()

        return {
            "transaction_id": tx_id,
            "timestamp": ts_str,
            "unified_risk_score": unified_risk_score,
            "risk_level": risk_level,
            "decision": decision,
            "breakdown": {
                "ml_fraud_probability": round(ml_prob, 4),
                "ml_contribution": ml_contrib,
                "anomaly_score": round(s_anomaly, 2),
                "anomaly_contribution": anomaly_contrib,
                "security_risk_score": round(sec_score, 4),
                "security_contribution": sec_contrib,
                "critical_flags": critical_flags,
            },
        }
