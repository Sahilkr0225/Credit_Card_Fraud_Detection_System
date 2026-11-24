"""
Device and Location Risk Rules Engine.

Evaluates device fingerprint mismatches, unknown hardware identifiers,
high-risk merchant categories, and impossible travel patterns (geo-velocity).
"""

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set


class DeviceLocationRules:
    """
    Evaluates device trust and geographic velocity between consecutive transactions.

    Parameters
    ----------
    max_physical_speed_kmh : float, default=850.0
        Max plausible speed for commercial passenger aviation.
    geo_jump_min_distance_km : float, default=100.0
        Minimum distance threshold to evaluate geo-velocity anomalies.
    high_risk_categories : list of str, optional
        Merchant categories deemed high-risk.
    """

    DEFAULT_HIGH_RISK_CATEGORIES = {
        "crypto",
        "cryptocurrency",
        "offshore_gambling",
        "casino",
        "wire_transfer",
        "p2p_cash",
        "foreign_atm",
    }

    def __init__(
        self,
        max_physical_speed_kmh: float = 850.0,
        geo_jump_min_distance_km: float = 100.0,
        high_risk_categories: Optional[Set[str]] = None,
    ) -> None:
        self.max_physical_speed_kmh = max_physical_speed_kmh
        self.geo_jump_min_distance_km = geo_jump_min_distance_km
        self.high_risk_categories = (
            set(c.lower() for c in high_risk_categories)
            if high_risk_categories
            else self.DEFAULT_HIGH_RISK_CATEGORIES
        )

        # Storage: user_id -> set of known device_ids
        self._user_devices: Dict[str, Set[str]] = defaultdict(set)
        # Storage: user_id -> last (timestamp_epoch, lat, lon, city, country)
        self._last_location: Dict[str, Tuple[float, float, float, str, str]] = {}

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

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points on Earth in kilometers.
        """
        radius = 6371.0  # Earth radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return radius * c

    def evaluate(
        self,
        user_id: str,
        device_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        city: str = "Unknown",
        country: str = "Unknown",
        merchant_category: str = "general",
        timestamp: Optional[Any] = None,
        update_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate device novelty, impossible travel, and merchant risk.

        Returns
        -------
        dict
            Assessment with risk score, impossible travel metrics, and flags.
        """
        t_now = self._parse_timestamp(timestamp)
        flags: List[str] = []
        score = 0.0

        # 1. Device Novelty Check
        is_new_device = False
        if device_id:
            known_devices = self._user_devices[user_id]
            if known_devices and device_id not in known_devices:
                is_new_device = True
                score += 0.25
                flags.append("NEW_UNRECOGNIZED_DEVICE")

        # 2. Geo-Velocity / Impossible Travel Check
        distance_km = 0.0
        implied_speed_kmh = 0.0
        impossible_travel = False

        if latitude is not None and longitude is not None:
            last_loc = self._last_location.get(user_id)
            if last_loc is not None:
                last_time, last_lat, last_lon, last_city, last_country = last_loc
                time_delta_hours = max((t_now - last_time) / 3600.0, 0.001)  # Prevent div-by-zero

                distance_km = round(self.haversine_distance(last_lat, last_lon, latitude, longitude), 2)
                implied_speed_kmh = round(distance_km / time_delta_hours, 2)

                if (
                    distance_km >= self.geo_jump_min_distance_km
                    and implied_speed_kmh > self.max_physical_speed_kmh
                ):
                    impossible_travel = True
                    score += 0.70
                    flags.append("IMPOSSIBLE_TRAVEL_SPEED_EXCEEDED")
                    flags.append(f"SPEED_{implied_speed_kmh:.0f}KMH")

                    if last_country != "Unknown" and country != "Unknown" and last_country != country:
                        score += 0.15
                        flags.append(f"SUDDEN_CROSS_BORDER_{last_country}_TO_{country}")

        # 3. High-Risk Merchant Category Check
        clean_category = merchant_category.strip().lower()
        if any(hrc in clean_category for hrc in self.high_risk_categories):
            score += 0.20
            flags.append("HIGH_RISK_MERCHANT_CATEGORY")

        final_score = round(min(1.0, max(0.0, score)), 3)

        # State updates
        if update_state:
            if device_id:
                self._user_devices[user_id].add(device_id)
            if latitude is not None and longitude is not None:
                self._last_location[user_id] = (t_now, latitude, longitude, city, country)

        return {
            "user_id": user_id,
            "device_id": device_id,
            "new_device_detected": is_new_device,
            "impossible_travel_detected": impossible_travel,
            "distance_km": distance_km,
            "implied_speed_kmh": implied_speed_kmh,
            "merchant_category": merchant_category,
            "device_location_risk_score": final_score,
            "flags": flags,
        }

    def clear(self, user_id: Optional[str] = None) -> None:
        """Reset known devices and location state."""
        if user_id:
            self._user_devices.pop(user_id, None)
            self._last_location.pop(user_id, None)
        else:
            self._user_devices.clear()
            self._last_location.clear()
