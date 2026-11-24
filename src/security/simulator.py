"""
Synthetic Transaction and Fraud Attack Scenario Generator.

Generates realistic normal transaction streams and high-fidelity attack vectors
(SIM-swap takeovers, carding bursts, impossible travel, and OTP brute-force)
with both tabular ML features (V1-V28, Time, Amount) and security telemetry.
"""

import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional


# Representative global cities with coordinates for geo-velocity simulation
CITIES = [
    {"city": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
    {"city": "San Francisco", "country": "USA", "lat": 37.7749, "lon": -122.4194},
    {"city": "Chicago", "country": "USA", "lat": 41.8781, "lon": -87.6298},
    {"city": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
    {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"city": "Frankfurt", "country": "Germany", "lat": 50.1109, "lon": 8.6821},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
]

MERCHANTS = {
    "grocery": ["Whole Foods Market", "Trader Joe's", "Safeway", "Kroger Supermarket"],
    "dining": ["Starbucks Coffee", "Chipotle Mexican Grill", "Shake Shack", "Local Bistro"],
    "retail": ["Target", "Amazon Retail", "Apple Store", "Nike Official Store", "Best Buy"],
    "travel": ["Delta Airlines", "Uber Rides", "Marriott Hotels", "Expedia Bookings"],
    "crypto": ["Binance Exchange", "Coinbase Pro", "Kraken Digital", "Crypto P2P Desk"],
    "gambling": ["Offshore Bet365", "Vegas Gold Casino", "Monte Carlo VIP", "Crypto Dice"],
    "p2p": ["Zelle Transfer", "Venmo Instant", "CashApp Direct", "Wire Remittance"],
}


class SyntheticTransactionGenerator:
    """
    Generator for realistic payment events and cyber fraud scenarios.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        if seed is not None:
            random.seed(seed)
        self._user_homes: Dict[str, Dict[str, Any]] = {}
        self._user_devices: Dict[str, str] = {}

    def _get_user_home(self, user_id: str) -> Dict[str, Any]:
        """Assign or retrieve a consistent home city for a user."""
        if user_id not in self._user_homes:
            self._user_homes[user_id] = random.choice(CITIES[:5])  # Default to US/Europe cities
        return self._user_homes[user_id]

    def _get_user_device(self, user_id: str) -> str:
        """Assign or retrieve a trusted primary device for a user."""
        if user_id not in self._user_devices:
            self._user_devices[user_id] = f"DEV_TRUSTED_{user_id[-4:]}_{random.randint(100, 999)}"
        return self._user_devices[user_id]

    def _generate_pca_features(self, is_fraud: bool = False) -> Dict[str, float]:
        """
        Generate synthetic PCA V1-V28 vectors mirroring real credit card distributions.
        EDA showed V14, V17, V12, V10 exhibit strong negative shifts for fraud,
        while V4 and V11 exhibit positive shifts.
        """
        features: Dict[str, float] = {}
        for i in range(1, 29):
            col = f"V{i}"
            if not is_fraud:
                features[col] = round(random.gauss(0.0, 1.0), 6)
            else:
                # Characteristic fraud vector shifts
                if col in ("V14", "V17"):
                    features[col] = round(random.gauss(-7.5, 2.0), 6)
                elif col in ("V12", "V10"):
                    features[col] = round(random.gauss(-5.0, 1.5), 6)
                elif col in ("V4", "V11"):
                    features[col] = round(random.gauss(4.5, 1.5), 6)
                elif col in ("V2", "V3"):
                    features[col] = round(random.gauss(2.5, 1.8), 6)
                else:
                    features[col] = round(random.gauss(0.0, 1.5), 6)
        return features

    def generate_normal_transaction(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate a single legitimate normal cardholder transaction."""
        uid = user_id or f"USR_{random.randint(1000, 9999)}"
        home = self._get_user_home(uid)
        device = self._get_user_device(uid)
        dt = timestamp or datetime.now(timezone.utc)

        category = random.choice(["grocery", "dining", "retail", "travel"])
        merchant = random.choice(MERCHANTS[category])
        amount = round(random.expovariate(1 / 45.0) + 3.50, 2)
        amount = min(amount, 850.0)  # Bound standard transaction

        tx: Dict[str, Any] = {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "user_id": uid,
            "timestamp": dt.isoformat(),
            "Time": round(dt.hour * 3600 + dt.minute * 60 + dt.second + random.random(), 1),
            "amount": amount,
            "Amount": amount,
            "merchant": merchant,
            "merchant_category": category,
            "device_id": device,
            "city": home["city"],
            "country": home["country"],
            "latitude": home["lat"] + random.uniform(-0.03, 0.03),
            "longitude": home["lon"] + random.uniform(-0.03, 0.03),
            "sim_swap_detected": False,
            "sim_swap_age_hours": 720.0,
            "otp_failures_count": 0,
            "otp_verified": True,
            "scenario_label": "NORMAL",
            "is_ground_truth_fraud": 0,
        }
        tx.update(self._generate_pca_features(is_fraud=False))
        return tx

    def generate_sim_swap_attack(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a SIM-swap account takeover transaction.
        High transfer amount, recent SIM replacement (< 24h), brand-new device,
        and OTP succeeds because fraudster intercepted SMS token on ported SIM.
        """
        uid = user_id or f"USR_{random.randint(1000, 9999)}"
        home = self._get_user_home(uid)
        dt = timestamp or datetime.now(timezone.utc)

        category = random.choice(["p2p", "crypto", "retail"])
        merchant = random.choice(MERCHANTS[category])
        # High value transfer right after takeover
        amount = round(random.uniform(950.0, 4800.0), 2)
        sim_age = round(random.uniform(1.2, 14.5), 1)  # Recent SIM swap under 24 hours
        new_device = f"DEV_ATTACKER_{uuid.uuid4().hex[:6].upper()}"

        tx: Dict[str, Any] = {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "user_id": uid,
            "timestamp": dt.isoformat(),
            "Time": round(dt.hour * 3600 + dt.minute * 60 + dt.second, 1),
            "amount": amount,
            "Amount": amount,
            "merchant": merchant,
            "merchant_category": category,
            "device_id": new_device,
            "city": home["city"],
            "country": home["country"],
            "latitude": home["lat"] + random.uniform(-0.05, 0.05),
            "longitude": home["lon"] + random.uniform(-0.05, 0.05),
            "sim_swap_detected": True,
            "sim_swap_age_hours": sim_age,
            "otp_failures_count": 0,
            "otp_verified": True,  # Key viva point: OTP is verified because SMS was intercepted!
            "scenario_label": "SIM_SWAP_TAKEOVER",
            "is_ground_truth_fraud": 1,
        }
        tx.update(self._generate_pca_features(is_fraud=True))
        return tx

    def generate_carding_burst(
        self,
        user_id: Optional[str] = None,
        burst_size: int = 5,
        base_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate rapid card-testing micro-transactions within 2 minutes.
        Followed by an elevated cash-out attempt.
        """
        uid = user_id or f"USR_{random.randint(1000, 9999)}"
        home = self._get_user_home(uid)
        dt = base_time or datetime.now(timezone.utc)
        bot_device = f"DEV_BOT_{uuid.uuid4().hex[:6].upper()}"

        batch: List[Dict[str, Any]] = []
        for i in range(burst_size):
            tx_time = dt + timedelta(seconds=i * 18)  # 18 seconds apart
            is_final_cashout = i == burst_size - 1

            if not is_final_cashout:
                amount = round(random.uniform(1.10, 8.50), 2)  # Micro carding amounts
                category = "retail"
                merchant = f"Online Micro-merchant #{i + 1}"
            else:
                amount = round(random.uniform(650.0, 1800.0), 2)  # Drain after successful tests
                category = "crypto"
                merchant = "Binance Instant Cashout"

            tx: Dict[str, Any] = {
                "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
                "user_id": uid,
                "timestamp": tx_time.isoformat(),
                "Time": round(tx_time.hour * 3600 + tx_time.minute * 60 + tx_time.second, 1),
                "amount": amount,
                "Amount": amount,
                "merchant": merchant,
                "merchant_category": category,
                "device_id": bot_device,
                "city": home["city"],
                "country": home["country"],
                "latitude": home["lat"],
                "longitude": home["lon"],
                "sim_swap_detected": False,
                "sim_swap_age_hours": 720.0,
                "otp_failures_count": 0,
                "otp_verified": True,
                "scenario_label": "CARDING_BURST",
                "is_ground_truth_fraud": 1,
            }
            tx.update(self._generate_pca_features(is_fraud=True))
            batch.append(tx)

        return batch

    def generate_impossible_travel(
        self,
        user_id: Optional[str] = None,
        base_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate two transactions separated by vast distance in negligible time.
        e.g., Transaction in New York followed 15 minutes later in Tokyo.
        """
        uid = user_id or f"USR_{random.randint(1000, 9999)}"
        dt = base_time or datetime.now(timezone.utc)

        city_a = CITIES[0]  # New York
        city_b = CITIES[6]  # Tokyo (distance ~10,800 km)

        # Legitimate first transaction in City A
        tx1 = self.generate_normal_transaction(user_id=uid, timestamp=dt)
        tx1["city"] = city_a["city"]
        tx1["country"] = city_a["country"]
        tx1["latitude"] = city_a["lat"]
        tx1["longitude"] = city_a["lon"]

        # Fraudulent second transaction 15 minutes later in City B
        dt2 = dt + timedelta(minutes=15)
        tx2_amount = round(random.uniform(450.0, 2200.0), 2)
        tx2: Dict[str, Any] = {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "user_id": uid,
            "timestamp": dt2.isoformat(),
            "Time": round(dt2.hour * 3600 + dt2.minute * 60 + dt2.second, 1),
            "amount": tx2_amount,
            "Amount": tx2_amount,
            "merchant": "Tokyo Duty Free Luxury",
            "merchant_category": "retail",
            "device_id": f"DEV_FOREIGN_{uuid.uuid4().hex[:6].upper()}",
            "city": city_b["city"],
            "country": city_b["country"],
            "latitude": city_b["lat"],
            "longitude": city_b["lon"],
            "sim_swap_detected": False,
            "sim_swap_age_hours": 720.0,
            "otp_failures_count": 0,
            "otp_verified": True,
            "scenario_label": "IMPOSSIBLE_TRAVEL",
            "is_ground_truth_fraud": 1,
        }
        tx2.update(self._generate_pca_features(is_fraud=True))

        return [tx1, tx2]

    def generate_otp_brute_force_attack(
        self,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a transaction subject to repeated failed OTP attempts.
        """
        uid = user_id or f"USR_{random.randint(1000, 9999)}"
        home = self._get_user_home(uid)
        dt = timestamp or datetime.now(timezone.utc)

        amount = round(random.uniform(300.0, 1500.0), 2)
        failures = random.randint(3, 5)

        tx: Dict[str, Any] = {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:8].upper()}",
            "user_id": uid,
            "timestamp": dt.isoformat(),
            "Time": round(dt.hour * 3600 + dt.minute * 60 + dt.second, 1),
            "amount": amount,
            "Amount": amount,
            "merchant": "High-Risk Peer Transfer",
            "merchant_category": "p2p",
            "device_id": f"DEV_SUSPECT_{uuid.uuid4().hex[:6].upper()}",
            "city": home["city"],
            "country": home["country"],
            "latitude": home["lat"],
            "longitude": home["lon"],
            "sim_swap_detected": False,
            "sim_swap_age_hours": 720.0,
            "otp_failures_count": failures,
            "otp_verified": False,
            "scenario_label": "OTP_BRUTE_FORCE",
            "is_ground_truth_fraud": 1,
        }
        tx.update(self._generate_pca_features(is_fraud=True))
        return tx

    def generate_batch(
        self,
        size: int = 50,
        fraud_ratio: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """
        Generate a mixed stream of transactions with normal events and cyber attacks.
        """
        batch: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for i in range(size):
            tx_time = now - timedelta(minutes=(size - i) * 3)
            is_attack = random.random() < fraud_ratio

            if not is_attack:
                batch.append(self.generate_normal_transaction(timestamp=tx_time))
            else:
                attack_type = random.choice([
                    "sim_swap",
                    "carding",
                    "impossible_travel",
                    "otp_brute_force",
                ])

                if attack_type == "sim_swap":
                    batch.append(self.generate_sim_swap_attack(timestamp=tx_time))
                elif attack_type == "carding":
                    burst = self.generate_carding_burst(burst_size=3, base_time=tx_time)
                    batch.extend(burst)
                elif attack_type == "impossible_travel":
                    travel_pair = self.generate_impossible_travel(base_time=tx_time)
                    batch.extend(travel_pair)
                else:
                    batch.append(self.generate_otp_brute_force_attack(timestamp=tx_time))

        return batch
