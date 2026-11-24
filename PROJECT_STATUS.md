# Project Status & Milestone Tracker

**Project:** Intelligent Credit Card Fraud Detection and Risk Assessment System  
**Current Branch:** `feature/phase2-supervised-classification`  
**Current Phase:** Phase 2 (ML) & Member 2 (Security) Merged -> Ready for Phase 3  
**Last Updated:** November 24, 2026  

---

## 1. Team Allocation & Ownership Boundaries

| Module / Component | Owner | Status | Deliverables |
| :--- | :--- | :--- | :--- |
| **Dataset Acquisition & Validation** | **Member 1 (ML)** | Completed | `data/raw/creditcard.csv`, `src/data/load_data.py` |
| **Exploratory Data Analysis (EDA)** | **Member 1 (ML)** | Completed | `notebooks/01_eda.ipynb` |
| **Data Preprocessing & Splitting** | **Member 1 (ML)** | Completed | `src/data/preprocessing.py` |
| **Data & Preprocessing Unit Tests** | **Member 1 (ML)** | Completed | `tests/test_load_data.py`, `tests/test_preprocessing.py` (14/14 Passed) |
| **Supervised Classification (LR, RF)** | **Member 1 (ML)** | Completed (Phase 2) | `src/models/train_supervised.py`, `models/best_fraud_model.joblib` |
| **Supervised Model Unit Tests** | **Member 1 (ML)** | Completed (Phase 2) | `tests/test_supervised_models.py` (5/5 Passed) |
| **Unsupervised Anomaly Detection** | **Member 1 (ML)** | Next Up (Phase 3) | `src/models/train_unsupervised.py` |
| **Model Evaluation & Metric Analysis** | **Member 1 (ML)** | Planned (Phase 4) | `notebooks/02_model_evaluation.ipynb` |
| **ML Prediction Interface & API** | **Member 1 (ML)** | Planned (Phase 5) | `src/models/predict.py` |
| **Synthetic Event Generation** | **Member 2 (Security)** | Completed | `src/security/simulator.py`, `app/simulator/transaction_generator.py` |
| **SIM Swap Detection Engine** | **Member 2 (Security)** | Completed | `src/security/sim_swap_engine.py`, `app/security/sim_swap_engine.py` |
| **OTP Failure Monitoring** | **Member 2 (Security)** | Completed | `src/security/otp_monitor.py`, `app/security/otp_monitor.py` |
| **Velocity & Carding Detection** | **Member 2 (Security)** | Completed | `src/security/velocity_monitor.py`, `app/security/velocity_monitor.py` |
| **Device & Location Rules** | **Member 2 (Security)** | Completed | `src/security/device_location_rules.py`, `app/security/device_location_rules.py` |
| **Security Signals Aggregator** | **Member 2 (Security)** | Completed | `src/security/signals_engine.py` |
| **Streamlit Interactive Dashboard** | **Member 2 (Security)** | Completed | `app/dashboard.py`, `app/components/` |
| **Security Unit Tests (30/30 Passed)** | **Member 2 (Security)** | Completed | `tests/test_sim_swap_engine.py`, `tests/test_velocity_monitor.py`, `tests/test_otp_monitor.py`, `tests/test_device_location_rules.py`, `tests/test_simulator.py`, `tests/test_risk_engine.py` |
| **API Contract Specification** | **Shared** | Completed (v1.0.0) | `docs/API_CONTRACT.md` |
| **Unified Risk Assessment Engine** | **Shared** | Scaffolded / Integrating | `src/risk_engine.py`, `core/risk_engine.py` |

---

## 2. Member 1 Implementation Summary (Machine Learning Core)

### Phase 1: Data Ingestion, EDA, Preprocessing
- Schema validation enforcing certified 31 columns and zero missing values.
- Leakage-free `RobustScaler` on `Time` and `Amount` (fit strictly on `X_train`).
- 80/20 Stratified train/test partition maintaining exact 0.1727% fraud prevalence.
- Exploratory data analysis in `notebooks/01_eda.ipynb` with embedded diurnal and correlation charts.
- 14 automated unit tests in `tests/test_load_data.py` and `tests/test_preprocessing.py`.

### Phase 2: Supervised Classification Benchmark Results
Trained on 227,845 records and evaluated on 56,962 untouched holdout test records (98 fraud cases):

| Candidate Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | F1-Score | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Class-Weighted)** | 0.7411 | 0.7175 | 0.9720 | 91.84% | 6.09% | 0.1142 | 1,388 | 8 |
| **Logistic Regression (SMOTE Pipeline)** | 0.7412 | 0.7251 | 0.9712 | 91.84% | 5.89% | 0.1106 | 1,439 | 8 |
| **Random Forest (Class-Weighted)** 🏆 | **0.8215** | **0.8311** | 0.9766 | 80.61% | **84.04%** | **0.8229** | **15** | 19 |
| **Random Forest (SMOTE Pipeline)** | 0.8150 | 0.8258 | **0.9792** | 85.71% | 51.22% | 0.6412 | 80 | 14 |

### Artifacts Saved to `models/`:
- `best_fraud_model.joblib`: Winning Random Forest model.
- `robust_scaler.joblib`: Preprocessing scaler for real-time inference.
- `evaluation_summary.json`: Complete benchmark metrics and confusion matrices.
- 19 automated unit tests passing across Phase 1 & 2.

---

## 3. Member 2 Implementation Summary (Security & Application Layer)

All Member 2 deliverables have been fully constructed, tested, and integrated:

### Component 1: SIM-Swap Detection Engine (`src/security/sim_swap_engine.py`)
- Evaluates carrier ICCID/IMSI change signals.
- Multi-tier risk windows: <24h (Critical), 24-48h (High), 48-72h (Medium), >72h (Baseline).
- Detects account takeover *even when OTP succeeds* via `SIM_SWAP_OTP_INTERCEPTION_RISK`.
- Compounding risk logic for high-value transfers (>= $500) and new devices.

### Component 2: Velocity & Carding Monitor (`src/security/velocity_monitor.py`)
- In-memory sliding time windows (1-minute, 5-minute, 1-hour, 24-hour).
- Detects card-testing bots via micro-transactions (amounts <= $15, 3+ within 5m).
- Detects transaction frequency bursts and hourly spending velocity spikes.

### Component 3: OTP Abuse Tracker (`src/security/otp_monitor.py`)
- Tracks consecutive and rolling-window OTP verification failures.
- Threshold-based escalation: warning at 1-2 failures, automated channel lockout at >= 3 failures.
- Identifies OTP resend flooding and credential stuffing patterns.

### Component 4: Device & Location Rules (`src/security/device_location_rules.py`)
- Device fingerprint novelty detection per user account.
- Great-Circle Haversine distance and geo-velocity calculations.
- Detects physically impossible supersonic travel (> 850 km/h commercial flight limit).
- High-risk merchant category evaluation (crypto, offshore gambling, P2P cash).

### Component 5: Synthetic Transaction Generator (`src/security/simulator.py`)
- High-fidelity synthetic event generator producing both tabular PCA features (V1-V28, Time, Amount) and security telemetry.
- Attack profiles: SIM-Swap Takeover, Carding Burst, Impossible Travel, OTP Brute-Force, and Normal batch streams.

### Component 6: Unified Risk Assessment Engine (`src/risk_engine.py`)
- Weighted multi-factor score: $R_{\text{total}} = w_1 S_{\text{ML}} + w_2 S_{\text{Anomaly}} + w_3 S_{\text{Security}}$.
- Dynamic decision mapping: `APPROVE`, `CHALLENGE_OTP`, `MANUAL_REVIEW`, `DECLINE`.
- Hard override protection preventing unfamiliar attack vectors from diluting critical telecom/geo violations.

### Component 7: Streamlit Interactive Dashboard (`app/dashboard.py`)
- Modern, dark-fintech user interface with real-time KPI tiles.
- 4 interactive views:
  1. **Live Transaction Feed & Monitor:** Filterable real-time table, CSV export, and deep telemetry inspector.
  2. **Security Visualizations:** Interactive Plotly charts (Risk Distribution, Amount vs Risk Scatter, Rule Trigger Donut).
  3. **Live Transaction Sandbox:** Interactive custom transaction evaluator with real-time radial risk gauge and factor breakdown.
  4. **Risk Engine Configurator:** Adjustable weights and decision thresholds.
- Sidebar Attack Simulator Studio for one-click scenario testing.

### Component 8: Security Unit Test Suite (30/30 Passed)
- `tests/test_sim_swap_engine.py`: 5 passed
- `tests/test_velocity_monitor.py`: 5 passed
- `tests/test_otp_monitor.py`: 4 passed
- `tests/test_device_location_rules.py`: 6 passed
- `tests/test_simulator.py`: 6 passed
- `tests/test_risk_engine.py`: 4 passed
