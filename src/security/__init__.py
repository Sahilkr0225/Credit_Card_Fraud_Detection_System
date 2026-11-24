"""
Security & Behavioral Signals Package (Member 2 Deliverables).

Provides:
- SimSwapEngine: Telecom and SIM replacement risk detection.
- VelocityMonitor: Sliding-window transaction frequency & carding detection.
- OtpMonitor: OTP verification failures and brute-force tracking.
- DeviceLocationRules: Device fingerprinting and geo-velocity / impossible travel.
- SecuritySignalsEngine: Unified security signals aggregator.
- SyntheticTransactionGenerator: Realistic attack and normal transaction simulation.
"""

from src.security.sim_swap_engine import SimSwapEngine
from src.security.velocity_monitor import VelocityMonitor
from src.security.otp_monitor import OtpMonitor
from src.security.device_location_rules import DeviceLocationRules
from src.security.signals_engine import SecuritySignalsEngine
from src.security.simulator import SyntheticTransactionGenerator

__all__ = [
    "SimSwapEngine",
    "VelocityMonitor",
    "OtpMonitor",
    "DeviceLocationRules",
    "SecuritySignalsEngine",
    "SyntheticTransactionGenerator",
]
