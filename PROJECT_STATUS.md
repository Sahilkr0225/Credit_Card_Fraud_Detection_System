# Project Status & Milestone Tracker

**Project:** Intelligent Credit Card Fraud Detection and Risk Assessment System  
**Current Branch:** `develop`  
**Current Phase:** Phase 1 Completed -> Ready for Phase 2  
**Last Updated:** November 23, 2026  

---

## 1. Team Allocation & Ownership Boundaries

| Module / Component | Owner | Status | Deliverables |
| :--- | :--- | :--- | :--- |
| **Dataset Acquisition & Validation** | **Member 1 (ML)** | Completed | `data/raw/creditcard.csv`, `src/data/load_data.py` |
| **Exploratory Data Analysis (EDA)** | **Member 1 (ML)** | Completed | `notebooks/01_eda.ipynb` |
| **Data Preprocessing & Splitting** | **Member 1 (ML)** | Completed | `src/data/preprocessing.py` |
| **Data & Preprocessing Unit Tests** | **Member 1 (ML)** | Completed | `tests/test_load_data.py`, `tests/test_preprocessing.py` (11/11 Passed) |
| **Supervised Classification (LR, RF)** | **Member 1 (ML)** | Next Up (Phase 2) | `src/models/train_supervised.py` |
| **Unsupervised Anomaly Detection** | **Member 1 (ML)** | Planned (Phase 3) | `src/models/train_unsupervised.py` |
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
| **Unified Risk Assessment Engine** | **Shared / Member 2** | Completed | `src/risk_engine.py`, `core/risk_engine.py` |

---

## 2. Phase 1 Implementation Summary (Member 1)

All Phase 1 requirements have been successfully built, tested, and verified:

### Step 1: Repository Audit
- Current branch: `develop` (clean, up to date).
- Project skeleton initialized: `src/data/`, `notebooks/`, `tests/`, `docs/`, `data/raw/`, `data/processed/`, `models/`.
- Git protection: `.gitignore` configured to prevent committing raw data, virtual environments, or compiled files.
- Python 3.10 virtual environment (`venv`) established with all dependencies pinned.

### Step 2: Data Loading Module (`src/data/load_data.py`)
- Automated validation of expected 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`).
- Pure read-only loading logic; original raw dataset is never altered.
- Informative `FileNotFoundError` guidance if raw CSV is missing.
- Verified on live Kaggle dataset: 284,807 rows, 31 columns, 0 nulls, 492 fraud cases (0.1727%).

### Step 3: Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
- Interactive analysis of dataset dimensions, missing cells, and duplicate rows.
- Quantitative class distribution visualization (log-scale bar chart and proportion donut chart).
- Transaction amount analysis (demonstrating bimodal distribution of fraud and rationale for `RobustScaler`).
- Diurnal time analysis (demonstrating off-peak fraud activity).
- Correlation analysis of PCA components (`V14`, `V17`, `V12`, `V10` vs `V4`, `V11`).
- Viva defense section on the Accuracy Paradox with mathematical proof.

### Step 4: Preprocessing Pipeline (`src/data/preprocessing.py`)
- Independent target and feature separation.
- Stratified train/test splitting (80/20) preserving the exact 0.17% fraud prevalence.
- Leakage-free `RobustScaler` fitting strictly on `X_train`.
- Dual imbalance handling strategies:
  1. Balanced cost-sensitive class weights for standard loss functions.
  2. Train-only SMOTE oversampling (test data remains 100% authentic).
- Reproducible execution with fixed `random_state=42`.

### Step 5: Unit Tests (`tests/test_load_data.py`, `tests/test_preprocessing.py`)
- **11 out of 11 tests passing**:
  - `test_missing_file_raises_filenotfound`: PASSED
  - `test_validate_schema_success`: PASSED
  - `test_validate_schema_missing_column`: PASSED
  - `test_validate_schema_invalid_target`: PASSED
  - `test_summarize_dataset`: PASSED
  - `test_separate_features_and_target`: PASSED
  - `test_stratified_split_preserves_ratio`: PASSED
  - `test_no_data_leakage_in_scaling`: PASSED
  - `test_smote_applied_only_to_train`: PASSED
  - `test_reproducibility`: PASSED
  - `test_balanced_class_weights`: PASSED

### Step 6: Documentation
- Updated `README.md` with complete architecture, setup commands, execution instructions, and viva answers.
- Updated `PROJECT_STATUS.md` with milestone checklist.

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

### Component 8: Unit Test Suite
- **44 out of 44 tests passing** (14 Member 1 + 30 Member 2):
  - `tests/test_sim_swap_engine.py`: 5 passed
  - `tests/test_velocity_monitor.py`: 5 passed
  - `tests/test_otp_monitor.py`: 4 passed
  - `tests/test_device_location_rules.py`: 6 passed
  - `tests/test_simulator.py`: 6 passed
  - `tests/test_risk_engine.py`: 4 passed
  - `tests/test_load_data.py`: 8 passed
  - `tests/test_preprocessing.py`: 6 passed

