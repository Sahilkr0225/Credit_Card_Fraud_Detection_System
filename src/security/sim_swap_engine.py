"""
SIM-Swap Detection Engine.

Detects telecom account takeovers where attackers port a victim's phone number
to intercept SMS One-Time Passwords (OTPs). Even when OTP authentication succeeds,
a recent SIM replacement combined with device anomalies or high-value transfers
triggers high-severity defensive risk scoring.
"""

from typing import Dict, Any, List, Optional


class SimSwapEngine:
    """
    Evaluates telephony telemetry and carrier signals to assess SIM-swap risk.

    Parameters
    ----------
    critical_window_hours : float, default=24.0
        SIM swap age under this threshold is treated as Critical risk.
    high_risk_window_hours : float, default=48.0
        SIM swap age under this threshold is treated as High risk.
    medium_risk_window_hours : float, default=72.0
        SIM swap age under this threshold is treated as Medium risk.
    high_value_threshold : float, default=500.0
        Transaction amount threshold above which risk is compounded.
    """

    def __init__(
        self,
        critical_window_hours: float = 24.0,
        high_risk_window_hours: float = 48.0,
        medium_risk_window_hours: float = 72.0,
        high_value_threshold: float = 500.0,
    ) -> None:
        self.critical_window_hours = critical_window_hours
        self.high_risk_window_hours = high_risk_window_hours
        self.medium_risk_window_hours = medium_risk_window_hours
        self.high_value_threshold = high_value_threshold

    def evaluate(
        self,
        sim_swap_detected: bool = False,
        sim_swap_age_hours: Optional[float] = None,
        amount: float = 0.0,
        device_is_new: bool = False,
        otp_verified: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate SIM-swap risk for an incoming transaction.

        Parameters
        ----------
        sim_swap_detected : bool
            Whether telecom carrier recorded an ICCID / IMSI change.
        sim_swap_age_hours : float or None
            Hours elapsed since the SIM card was swapped.
        amount : float
            Transaction value in USD.
        device_is_new : bool
            Whether the transaction originates from an unrecognized device.
        otp_verified : bool
            Whether OTP challenge was successfully verified.

        Returns
        -------
        dict
            Structured assessment containing risk score, tier, flags, and explanation.
        """
        flags: List[str] = []
        age = float(sim_swap_age_hours) if sim_swap_age_hours is not None else 999.0

        if not sim_swap_detected or age > self.medium_risk_window_hours:
            return {
                "sim_swap_detected": sim_swap_detected,
                "sim_swap_age_hours": age,
                "risk_score": 0.05 if sim_swap_detected else 0.0,
                "risk_tier": "LOW",
                "flags": [],
                "recommendation": "ALLOW",
                "explanation": "No recent SIM-swap activity detected. Telecom baseline normal.",
            }

        # Calculate base score by age window
        if age < self.critical_window_hours:
            base_score = 0.85
            tier = "CRITICAL"
            flags.append("SIM_SWAP_UNDER_24H")
        elif age < self.high_risk_window_hours:
            base_score = 0.65
            tier = "HIGH"
            flags.append("SIM_SWAP_24_TO_48H")
        else:
            base_score = 0.35
            tier = "MEDIUM"
            flags.append("SIM_SWAP_48_TO_72H")

        # Compounding factors
        score = base_score

        # High-value transfer escalation
        if amount >= self.high_value_threshold:
            score += 0.15
            flags.append("SIM_SWAP_HIGH_VALUE_COMPOUND")

        # New device escalation
        if device_is_new:
            score += 0.10
            flags.append("SIM_SWAP_NEW_DEVICE")

        # OTP Interception warning: Successful OTP + Recent SIM swap = dangerous takeover indicator
        if otp_verified and age < self.high_risk_window_hours:
            flags.append("SIM_SWAP_OTP_INTERCEPTION_RISK")

        # Clamp score between 0.0 and 1.0
        final_score = round(min(1.0, max(0.0, score)), 3)

        # Update tier if score reached critical
        if final_score >= 0.85:
            tier = "CRITICAL"
            recommendation = "DECLINE"
        elif final_score >= 0.60:
            tier = "HIGH"
            recommendation = "CHALLENGE_OUT_OF_BAND"
        else:
            tier = "MEDIUM"
            recommendation = "CHALLENGE_OTP"

        explanation = (
            f"SIM swapped {age:.1f}h ago (Threshold < {self.critical_window_hours:.0f}h). "
            f"Risk compounded by {len(flags)} security indicator(s). "
            f"Note: Successful OTP is untrusted due to carrier SIM takeover."
            if "SIM_SWAP_OTP_INTERCEPTION_RISK" in flags
            else f"SIM swapped {age:.1f}h ago. Monitored for potential takeover."
        )

        return {
            "sim_swap_detected": True,
            "sim_swap_age_hours": age,
            "risk_score": final_score,
            "risk_tier": tier,
            "flags": flags,
            "recommendation": recommendation,
            "explanation": explanation,
        }
