"""
OTP Failure and Authentication Abuse Monitoring.

Tracks consecutive and clustered One-Time Password verification failures
to defend against credential stuffing, automated OTP brute-forcing,
and social engineering OTP interception pressure.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


class OtpMonitor:
    """
    Tracks OTP verification lifecycle and abuse metrics per user/card.

    Parameters
    ----------
    consecutive_warning_threshold : int, default=2
        Consecutive failures that elevate risk to Medium/High.
    consecutive_block_threshold : int, default=3
        Consecutive failures that trigger critical brute-force flags.
    window_minutes : float, default=15.0
        Sliding window in minutes for monitoring clustered OTP errors.
    """

    def __init__(
        self,
        consecutive_warning_threshold: int = 2,
        consecutive_block_threshold: int = 3,
        window_minutes: float = 15.0,
    ) -> None:
        self.consecutive_warning_threshold = consecutive_warning_threshold
        self.consecutive_block_threshold = consecutive_block_threshold
        self.window_seconds = window_minutes * 60.0

        # Storage: user_id -> list of (timestamp_epoch, success: bool)
        self._history: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        # Storage: user_id -> current consecutive failure count
        self._consecutive_failures: Dict[str, int] = defaultdict(int)

    def _parse_timestamp(self, ts: Any) -> float:
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=timezone.utc).timestamp() if ts.tzinfo is None else ts.timestamp()
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                return datetime.now(timezone.utc).timestamp()
        return datetime.now(timezone.utc).timestamp()

    def record_attempt(
        self,
        user_id: str,
        success: bool,
        timestamp: Optional[Any] = None,
    ) -> int:
        """Record an OTP verification result and return updated consecutive failure count."""
        t_now = self._parse_timestamp(timestamp)
        self._history[user_id].append((t_now, success))

        if success:
            self._consecutive_failures[user_id] = 0
        else:
            self._consecutive_failures[user_id] += 1

        return self._consecutive_failures[user_id]

    def evaluate(
        self,
        user_id: str,
        otp_failures_count: Optional[int] = None,
        rapid_resend_count: int = 0,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate OTP abuse risk for the user/transaction.

        Parameters
        ----------
        user_id : str
            Identifier for the account.
        otp_failures_count : int or None
            Explicit failure count passed with transaction (if stateless / pre-aggregated).
            If None, uses internal state.
        rapid_resend_count : int
            Number of OTP resend requests within the last 60 seconds.
        timestamp : datetime, str, float, or None
            Current event timestamp.

        Returns
        -------
        dict
            OTP risk evaluation containing score, flags, and recommendation.
        """
        t_now = self._parse_timestamp(timestamp)

        if otp_failures_count is not None:
            consecutive = otp_failures_count
        else:
            consecutive = self._consecutive_failures[user_id]

        # Calculate failures within recent time window
        records = self._history[user_id]
        cutoff = t_now - self.window_seconds
        recent_records = [r for r in records if r[0] >= cutoff]
        recent_failures = sum(1 for r in recent_records if not r[1])

        effective_failures = max(consecutive, recent_failures)

        flags: List[str] = []
        score = 0.0

        if effective_failures >= self.consecutive_block_threshold:
            score = 0.95
            flags.append("OTP_BRUTE_FORCE_EXCEEDED")
            flags.append(f"CONSECUTIVE_FAILURES_{effective_failures}")
        elif effective_failures >= self.consecutive_warning_threshold:
            score = 0.55
            flags.append("OTP_ELEVATED_FAILURES")
            flags.append(f"CONSECUTIVE_FAILURES_{effective_failures}")
        elif effective_failures == 1:
            score = 0.20
            flags.append("OTP_SINGLE_FAILURE_RETRY")
        else:
            score = 0.0

        if rapid_resend_count >= 3:
            score = min(1.0, score + 0.30)
            flags.append("OTP_FLOODING_RAPID_RESEND")

        score = round(min(1.0, max(0.0, score)), 3)

        if score >= 0.85:
            recommendation = "LOCK_AUTH_CHANNEL"
        elif score >= 0.50:
            recommendation = "STEP_UP_BIOMETRIC"
        else:
            recommendation = "PROCEED"

        return {
            "user_id": user_id,
            "otp_failures_count": effective_failures,
            "otp_risk_score": score,
            "brute_force_detected": effective_failures >= self.consecutive_block_threshold,
            "flags": flags,
            "recommendation": recommendation,
        }

    def clear(self, user_id: Optional[str] = None) -> None:
        """Reset OTP history for a user or all users."""
        if user_id:
            self._history.pop(user_id, None)
            self._consecutive_failures.pop(user_id, None)
        else:
            self._history.clear()
            self._consecutive_failures.clear()
