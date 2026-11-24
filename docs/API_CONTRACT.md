# API Contract & Module Interface Specification

**Project Title:** Intelligent Credit Card Fraud Detection and Risk Assessment System  
**Version:** 1.0.0  
**Status:** Approved for independent module development  

---

## 1. Overview & Separation of Ownership

This document defines the interface boundary between **Member 1 (Machine Learning Pipeline)** and **Member 2 (Security Rules, Synthetic Generation & Dashboard)** to ensure independent development, zero merge conflicts, and seamless integration through the **Unified Risk Assessment Engine**.

```
+-------------------------------------------------------------------------------+
|                               Incoming Event                                  |
+---------------------------------------+---------------------------------------+
                                        |
            +---------------------------+---------------------------+
            |                                                       |
            v                                                       v
+-------------------------------+               +-------------------------------+
|     Member 1: ML Pipeline     |               |  Member 2: Security Signals   |
| - Supervised Classifier       |               | - SIM Swap Monitoring         |
| - Unsupervised Anomaly Engine |               | - OTP Verification Failure    |
| (Input: Kaggle features)      |               | - Velocity & Carding Detector |
+---------------+---------------+               +---------------+---------------+
                |                                               |
                | MLPredictionResult                            | SecuritySignalsResult
                +-----------------------+-----------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |    Unified Risk Assessment Engine     |
                    | (Shared Module / Integration Layer)   |
                    | - Weighted Multi-Factor Score         |
                    | - Decision Engine (APPROVE/CHALLENGE) |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |       Streamlit Dashboard (M2)        |
                    +---------------------------------------+
```

---

## 2. Member 1: Machine Learning Prediction Interface

The ML model is trained on the European credit card fraud feature schema. It does not ingest device or telephony signals directly.

### 2.1 Input Schema: `MLTransactionInput`
```json
{
  "transaction_id": "TXN_982341",
  "Time": 406.0,
  "V1": -2.312226542,
  "V2": 1.951992011,
  "V3": -1.609850732,
  "...": "V4 through V27",
  "V28": 0.133558377,
  "Amount": 149.62
}
```

### 2.2 Output Schema: `MLPredictionResult`
```json
{
  "transaction_id": "TXN_982341",
  "fraud_probability": 0.892,
  "ml_prediction": 1,
  "anomaly_score": -0.245,
  "is_anomaly": true,
  "ml_risk_tier": "HIGH",
  "model_version": "rf_smote_v1.0"
}
```

* Field definitions:
  - `fraud_probability`: `float` in range `[0.0, 1.0]`. Calibrated probability from the supervised model.
  - `ml_prediction`: `int` (`0` = Legitimate, `1` = Fraudulent).
  - `anomaly_score`: `float`. Continuous anomaly metric from Isolation Forest.
  - `is_anomaly`: `bool`. True if identified as outlier by unsupervised model.
  - `ml_risk_tier`: `string` enum: `LOW` (<0.20), `MEDIUM` (0.20-0.60), `HIGH` (0.60-0.85), `CRITICAL` (>0.85).
  - `model_version`: `string`. Identifier of the trained pipeline.

---

## 3. Member 2: Security & Behavioral Signals Interface

Independently produced by Member 2's rule engines and synthetic generator.

### 3.1 Output Schema: `SecuritySignalsResult`
```json
{
  "transaction_id": "TXN_982341",
  "sim_swap_detected": true,
  "sim_swap_age_hours": 14.5,
  "otp_failures_count": 2,
  "velocity_score": 0.78,
  "carding_flag": true,
  "security_risk_score": 0.82
}
```

* Field definitions:
  - `sim_swap_detected`: `bool`. Flag if SIM was swapped within the last 48 hours.
  - `sim_swap_age_hours`: `float`. Hours since SIM change event.
  - `otp_failures_count`: `int`. Consecutive failed OTP attempts.
  - `velocity_score`: `float` in range `[0.0, 1.0]`. Rapid repeated transaction metric.
  - `carding_flag`: `bool`. Detected low-value rapid probing pattern.
  - `security_risk_score`: `float` in range `[0.0, 1.0]`. Aggregate heuristic score from security detectors.

---

## 4. Shared Integration: Unified Risk Assessment Interface

Both modules feed into the Unified Risk Assessment Engine.

### 4.1 Output Schema: `UnifiedRiskAssessment`
```json
{
  "transaction_id": "TXN_982341",
  "timestamp": "2026-11-22T23:30:00Z",
  "unified_risk_score": 88.5,
  "risk_level": "CRITICAL",
  "decision": "DECLINE",
  "breakdown": {
    "ml_fraud_probability": 0.892,
    "ml_contribution": 53.5,
    "security_risk_score": 0.82,
    "security_contribution": 35.0,
    "critical_flags": [
      "SIM_SWAP_RECENT",
      "HIGH_ML_FRAUD_PROBABILITY",
      "CARDING_PATTERN"
    ]
  }
}
```

* Decision Matrix:
  - `APPROVE`: Score < 30 (Low risk)
  - `CHALLENGE_OTP`: Score 30 - 65 (Suspicious, step-up authentication required)
  - `MANUAL_REVIEW`: Score 65 - 80 (Elevated risk, flagged for analyst queue)
  - `DECLINE`: Score > 80 (High probability of fraud or severe rule violation)

---

## 5. Mock Implementations for Independent Testing

Each member can mock the other's module using the following helper factories:

```python
# Mock for Member 2 to test dashboard without waiting for trained ML model
def get_mock_ml_prediction(transaction_id: str = "MOCK_001") -> dict:
    return {
        "transaction_id": transaction_id,
        "fraud_probability": 0.82,
        "ml_prediction": 1,
        "anomaly_score": -0.18,
        "is_anomaly": True,
        "ml_risk_tier": "HIGH",
        "model_version": "mock_v1"
    }

# Mock for Member 1 to test Unified Risk Engine without waiting for Security modules
def get_mock_security_signals(transaction_id: str = "MOCK_001") -> dict:
    return {
        "transaction_id": transaction_id,
        "sim_swap_detected": False,
        "sim_swap_age_hours": 120.0,
        "otp_failures_count": 0,
        "velocity_score": 0.15,
        "carding_flag": False,
        "security_risk_score": 0.12
    }
```

