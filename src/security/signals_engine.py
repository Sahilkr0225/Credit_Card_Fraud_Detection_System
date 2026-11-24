"""
Security Signals Aggregator Engine.

Fuses telemetry from SIM-swap detection, velocity monitoring, OTP abuse tracking,
and device/location rules into the unified SecuritySignalsResult schema.
"""

from typing import Dict, Any, List, Optional
from src.security.sim_swap_engine import SimSwapEngine
from src.security.velocity_monitor import VelocityMonitor
from src.security.otp_monitor import OtpMonitor
from src.security.device_location_rules import DeviceLocationRules


class SecuritySignalsEngine:
    """
    Orchestrates all Member 2 security sub-engines.
    """

    def __init__(
        self,
        sim_swap_engine: Optional[SimSwapEngine] = None,
        velocity_monitor: Optional[VelocityMonitor] = None,
        otp_monitor: Optional[OtpMonitor] = None,
        device_location_rules: Optional[DeviceLocationRules] = None,
    ) -> None:
        self.sim_swap_engine = sim_swap_engine or SimSwapEngine()
        self.velocity_monitor = velocity_monitor or VelocityMonitor()
        self.otp_monitor = otp_monitor or OtpMonitor()
        self.device_location_rules = device_location_rules or DeviceLocationRules()

    def evaluate_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate full security telemetry for a transaction.

        Parameters
        ----------
        tx_data : dict
            Transaction dictionary containing security and metadata fields.

        Returns
        -------
        dict
            Matches the API Contract SecuritySignalsResult schema.
        """
        tx_id = tx_data.get("transaction_id", "TXN_UNKNOWN")
        user_id = tx_data.get("user_id", "USR_UNKNOWN")
        amount = float(tx_data.get("Amount", tx_data.get("amount", 0.0)))
        timestamp = tx_data.get("timestamp")

        # 1. Device and Location Evaluation
        dev_res = self.device_location_rules.evaluate(
            user_id=user_id,
            device_id=tx_data.get("device_id"),
            latitude=tx_data.get("latitude"),
            longitude=tx_data.get("longitude"),
            city=tx_data.get("city", "Unknown"),
            country=tx_data.get("country", "Unknown"),
            merchant_category=tx_data.get("merchant_category", "general"),
            timestamp=timestamp,
            update_state=True,
        )

        # 2. SIM Swap Evaluation
        sim_detected = bool(tx_data.get("sim_swap_detected", False))
        sim_age = tx_data.get("sim_swap_age_hours")
        otp_verified = bool(tx_data.get("otp_verified", True))
        sim_res = self.sim_swap_engine.evaluate(
            sim_swap_detected=sim_detected,
            sim_swap_age_hours=sim_age,
            amount=amount,
            device_is_new=dev_res.get("new_device_detected", False),
            otp_verified=otp_verified,
        )

        # 3. Velocity and Carding Evaluation
        vel_res = self.velocity_monitor.record_and_evaluate(
            user_id=user_id,
            transaction_id=tx_id,
            amount=amount,
            timestamp=timestamp,
            mutate_history=True,
        )

        # 4. OTP Abuse Evaluation
        otp_failures = tx_data.get("otp_failures_count", 0)
        otp_res = self.otp_monitor.evaluate(
            user_id=user_id,
            otp_failures_count=otp_failures,
            rapid_resend_count=tx_data.get("rapid_resend_count", 0),
            timestamp=timestamp,
        )

        # 5. Composite Security Risk Score Computation
        # Uses hazard-dominant aggregation so critical flags (e.g. SIM swap or impossible travel)
        # cannot be masked by zeros in other checks, while multiple triggers compound.
        subscores = [
            sim_res["risk_score"],
            vel_res["velocity_score"],
            otp_res["otp_risk_score"],
            dev_res["device_location_risk_score"],
        ]
        max_sub = max(subscores)
        mean_sub = sum(subscores) / len(subscores)

        # Base composite: 70% highest severity risk + 30% cumulative context
        composite_score = (max_sub * 0.70) + (mean_sub * 0.30)

        # Bonus escalation if 2 or more distinct vectors are activated
        activated_vectors = sum(1 for s in subscores if s >= 0.40)
        if activated_vectors >= 2:
            composite_score = min(1.0, composite_score + 0.15)

        security_risk_score = round(min(1.0, max(0.0, composite_score)), 3)

        # Merge all security flags
        all_flags: List[str] = (
            sim_res["flags"]
            + vel_res["flags"]
            + otp_res["flags"]
            + dev_res["flags"]
        )

        # Contract compliant output schema
        return {
            "transaction_id": tx_id,
            "sim_swap_detected": sim_res["sim_swap_detected"],
            "sim_swap_age_hours": sim_res["sim_swap_age_hours"],
            "otp_failures_count": otp_res["otp_failures_count"],
            "velocity_score": vel_res["velocity_score"],
            "carding_flag": vel_res["carding_flag"],
            "security_risk_score": security_risk_score,
            # Enhanced telemetry
            "flags": all_flags,
            "impossible_travel_detected": dev_res["impossible_travel_detected"],
            "implied_speed_kmh": dev_res["implied_speed_kmh"],
            "subscores": {
                "sim_swap": sim_res["risk_score"],
                "velocity": vel_res["velocity_score"],
                "otp": otp_res["otp_risk_score"],
                "device_location": dev_res["device_location_risk_score"],
            },
        }
