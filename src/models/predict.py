"""
Real-Time Machine Learning Prediction Interface.

Exposes a thread-safe, high-performance prediction service conforming strictly to
the MLPredictionResult schema defined in docs/API_CONTRACT.md.
Combines Supervised Random Forest probabilities and Unsupervised Isolation Forest anomaly scores.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import joblib
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import EXPECTED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = "rf_iforest_v1.0"


class FraudPredictor:
    """
    Inference service for Credit Card Fraud Detection.

    Loads the serialized Random Forest classifier, Isolation Forest anomaly detector,
    and RobustScaler, providing single-transaction and batch scoring.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        decision_threshold: float = 0.50,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.decision_threshold = decision_threshold

        self.supervised_model = None
        self.scaler = None
        self.isolation_forest = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained models and scaler from disk."""
        rf_path = self.model_dir / "best_fraud_model.joblib"
        scaler_path = self.model_dir / "robust_scaler.joblib"
        iforest_path = self.model_dir / "isolation_forest.joblib"

        if rf_path.exists():
            self.supervised_model = joblib.load(rf_path)
            logger.info("Loaded Supervised Random Forest model.")
        else:
            logger.warning(f"Supervised model not found at {rf_path}.")

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded RobustScaler.")
        else:
            logger.warning(f"Scaler not found at {scaler_path}.")

        if iforest_path.exists():
            self.isolation_forest = joblib.load(iforest_path)
            logger.info("Loaded Unsupervised Isolation Forest.")
        else:
            logger.warning(f"Isolation Forest not found at {iforest_path}.")

    def _prepare_features(self, tx: Dict[str, Any], strict: bool = False) -> pd.DataFrame:
        """
        Extract and scale features from input transaction dictionary.
        """
        tx_dict = dict(tx)
        # Normalize Amount/amount and Time/time
        if "Amount" not in tx_dict and "amount" in tx_dict:
            tx_dict["Amount"] = tx_dict["amount"]
        if "Time" not in tx_dict and "time" in tx_dict:
            tx_dict["Time"] = tx_dict["time"]

        missing = [col for col in EXPECTED_FEATURE_COLUMNS if col not in tx_dict]
        if missing and strict:
            raise ValueError(f"Missing required transaction features: {missing}")

        # Construct single-row DataFrame with correct column order, defaulting to 0.0
        row_dict = {
            col: [float(tx_dict.get(col, 0.0))] for col in EXPECTED_FEATURE_COLUMNS
        }
        df_row = pd.DataFrame(row_dict)

        # Scale Time and Amount if scaler is available
        if self.scaler is not None:
            df_row[["Time", "Amount"]] = self.scaler.transform(df_row[["Time", "Amount"]])

        return df_row

    def predict_single(
        self, transaction: Dict[str, Any], strict: bool = False
    ) -> Dict[str, Any]:
        """
        Score a single incoming transaction matching API_CONTRACT.md schema.

        Parameters
        ----------
        transaction : dict
            Transaction dictionary with keys: Time, Amount, V1-V28, and optional transaction_id.
        strict : bool, default=False
            If True, raise ValueError if any required features are missing.

        Returns
        -------
        dict
            MLPredictionResult dictionary.
        """
        tx_id = transaction.get("transaction_id", "TXN_UNKNOWN")
        df_features = self._prepare_features(transaction, strict=strict)

        # 1. Supervised Fraud Probability (Random Forest)
        if self.supervised_model is not None:
            probs = self.supervised_model.predict_proba(df_features)[0]
            fraud_prob = round(float(probs[1]), 4)
            prediction = 1 if fraud_prob >= self.decision_threshold else 0
        else:
            fraud_prob = 0.05
            prediction = 0

        # 2. Unsupervised Anomaly Score (Isolation Forest)
        if self.isolation_forest is not None:
            raw_score = float(self.isolation_forest.decision_function(df_features)[0])
            is_anomaly = bool(raw_score < 0.0)
            anomaly_score = round(raw_score, 4)
        else:
            anomaly_score = 0.10
            is_anomaly = False

        # 3. Determine Risk Tier per contract enum
        if fraud_prob >= 0.85:
            risk_tier = "CRITICAL"
        elif fraud_prob >= 0.60:
            risk_tier = "HIGH"
        elif fraud_prob >= 0.20:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        return {
            "transaction_id": tx_id,
            "fraud_probability": fraud_prob,
            "ml_prediction": int(prediction),
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "ml_risk_tier": risk_tier,
            "model_version": MODEL_VERSION,
        }


# Global cached singleton instance
_GLOBAL_PREDICTOR: Optional[FraudPredictor] = None


def get_fraud_predictor() -> FraudPredictor:
    """Return singleton FraudPredictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = FraudPredictor()
    return _GLOBAL_PREDICTOR


def predict_transaction(tx_dict: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """Convenience functional interface for scoring a single transaction."""
    predictor = get_fraud_predictor()
    return predictor.predict_single(tx_dict, strict=strict)
