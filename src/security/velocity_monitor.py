"""
Transaction Velocity and Carding Detection Monitor.

Tracks sliding time windows (1 min, 5 min, 1 hour, 24 hours) per card/account
to detect high-frequency payment bursts, automated card testing (carding),
and sudden spending velocity spikes.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


class VelocityMonitor:
    """
    In-memory sliding-window velocity tracker per card/account.

    Parameters
    ----------
    burst_1m_limit : int, default=3
        Max transactions allowed in a 1-minute window before burst flag.
    burst_5m_limit : int, default=5
        Max transactions allowed in a 5-minute window before burst flag.
    carding_amount_max : float, default=15.0
        Upper limit for micro-amounts indicative of card-testing.
    carding_count_threshold : int, default=3
        Minimum small-value transactions in 5 minutes to trigger carding flag.
    hourly_spend_limit : float, default=2500.0
        Cumulative spend in 1 hour before raising velocity risk.
    """

    def __init__(
        self,
        burst_1m_limit: int = 3,
        burst_5m_limit: int = 5,
        carding_amount_max: float = 15.0,
        carding_count_threshold: int = 3,
        hourly_spend_limit: float = 2500.0,
    ) -> None:
        self.burst_1m_limit = burst_1m_limit
        self.burst_5m_limit = burst_5m_limit
        self.carding_amount_max = carding_amount_max
        self.carding_count_threshold = carding_count_threshold
        self.hourly_spend_limit = hourly_spend_limit

        # Storage: user_id -> list of (timestamp_epoch, amount, transaction_id)
        self._history: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)

    def _parse_timestamp(self, ts: Any) -> float:
        """Convert datetime or ISO string to epoch float seconds."""
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

    def record_and_evaluate(
        self,
        user_id: str,
        transaction_id: str,
        amount: float,
        timestamp: Optional[Any] = None,
        mutate_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Record the transaction and evaluate velocity metrics across sliding windows.

        Parameters
        ----------
        user_id : str
            Identifier for the cardholder or account.
        transaction_id : str
            Unique identifier for the transaction.
        amount : float
            Transaction value in USD.
        timestamp : datetime, str, float, or None
            Event timestamp. Defaults to current UTC time.
        mutate_history : bool, default=True
            Whether to permanently save this transaction into the account history.

        Returns
        -------
        dict
            Velocity assessment including score, carding flag, window metrics, and flags.
        """
        t_now = self._parse_timestamp(timestamp)
        records = self._history[user_id]

        # Purge records older than 24 hours (86400 seconds)
        cutoff_24h = t_now - 86400.0
        active_records = [r for r in records if r[0] >= cutoff_24h]

        # Calculate metrics for sliding windows
        window_1m = [r for r in active_records if r[0] >= t_now - 60.0]
        window_5m = [r for r in active_records if r[0] >= t_now - 300.0]
        window_1h = [r for r in active_records if r[0] >= t_now - 3600.0]

        # Include current transaction in calculations
        count_1m = len(window_1m) + 1
        count_5m = len(window_5m) + 1
        count_1h = len(window_1h) + 1
        count_24h = len(active_records) + 1

        spend_1m = sum(r[1] for r in window_1m) + amount
        spend_5m = sum(r[1] for r in window_5m) + amount
        spend_1h = sum(r[1] for r in window_1h) + amount
        spend_24h = sum(r[1] for r in active_records) + amount

        # Carding analysis (rapid small test charges)
        micro_tx_5m = [r for r in window_5m if r[1] <= self.carding_amount_max]
        if amount <= self.carding_amount_max:
            micro_count_5m = len(micro_tx_5m) + 1
        else:
            micro_count_5m = len(micro_tx_5m)

        carding_flag = micro_count_5m >= self.carding_count_threshold
        burst_flag = count_1m >= self.burst_1m_limit or count_5m >= self.burst_5m_limit

        flags: List[str] = []
        score = 0.0

        if carding_flag:
            score += 0.55
            flags.append("CARDING_MICRO_TRANSACTIONS")

        if burst_flag:
            score += 0.40
            flags.append("VELOCITY_BURST_EXCEEDED")

        if count_1h > 8:
            score += 0.20
            flags.append("HIGH_HOURLY_FREQUENCY")

        if spend_1h > self.hourly_spend_limit:
            score += 0.25
            flags.append("HOURLY_SPEND_SPIKE")

        # Gradual baseline velocity penalty
        if count_5m > 2 and not burst_flag:
            score += 0.15

        # Normalize score
        velocity_score = round(min(1.0, max(0.0, score)), 3)

        if mutate_history:
            active_records.append((t_now, amount, transaction_id))
            self._history[user_id] = active_records

        return {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "velocity_score": velocity_score,
            "carding_flag": carding_flag,
            "burst_flag": burst_flag,
            "tx_count_1m": count_1m,
            "tx_count_5m": count_5m,
            "tx_count_1h": count_1h,
            "tx_count_24h": count_24h,
            "spend_5m": round(spend_5m, 2),
            "spend_1h": round(spend_1h, 2),
            "spend_24h": round(spend_24h, 2),
            "flags": flags,
        }

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear recorded transactions for a specific user or all users."""
        if user_id:
            self._history.pop(user_id, None)
        else:
            self._history.clear()
