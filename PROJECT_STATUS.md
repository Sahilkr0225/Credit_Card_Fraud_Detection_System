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
| Synthetic Event Generation | Member 2 (Security) | Independent / Teammate | External to ML repo |
| SIM Swap Detection | Member 2 (Security) | Independent / Teammate | External to ML repo |
| OTP Monitoring | Member 2 (Security) | Independent / Teammate | External to ML repo |
| Velocity & Carding Detection | Member 2 (Security) | Independent / Teammate | External to ML repo |
| Streamlit Interactive Dashboard | Member 2 (Security) | Independent / Teammate | External to ML repo |
| **API Contract Specification** | **Shared** | Completed (v1.0.0) | `docs/API_CONTRACT.md` |
| **Unified Risk Assessment Engine** | **Shared** | Planned (Phase 6) | `src/risk_engine.py` |

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
