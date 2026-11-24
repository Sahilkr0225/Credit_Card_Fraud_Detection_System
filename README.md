# Intelligent Credit Card Fraud Detection and Risk Assessment System

An enterprise-grade, multi-layered machine learning and behavioral security system designed to detect fraudulent transactions and combat carding attacks in real time.

---

## 1. Team Allocation & Ownership Boundaries

This is a two-member collaborative engineering project integrated through strict API boundaries defined in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

```
+-------------------------------------------------------------------------------+
|                             Incoming Transaction                              |
+---------------------------------------+---------------------------------------+
                                        |
            +---------------------------+---------------------------+
            |                                                       |
            v                                                       v
+-------------------------------+               +-------------------------------+
|      Member 1: ML Engine      |               |  Member 2: Security Signals   |
| - Supervised Fraud Classifier |               | - SIM Swap Detector           |
| - Unsupervised Anomaly Engine |               | - OTP Failure Monitor         |
| - Data & Preprocessing Core   |               | - Velocity & Carding Detector |
| (Input: Kaggle Schema)        |               | - Interactive Streamlit UI    |
+---------------+---------------+               +---------------+---------------+
                |                                               |
                | MLPredictionResult                            | SecuritySignalsResult
                +-----------------------+-----------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |    Unified Risk Assessment Engine     |
                    |       (Collaborative Layer)           |
                    | - Multi-Factor Score (0 - 100)        |
                    | - Decision: APPROVE / REVIEW / DECLINE|
                    +---------------------------------------+
```

* **Member 1 (Machine Learning Core):**
  - Dataset acquisition, validation, and exploratory data analysis.
  - Leakage-free preprocessing, stratified partitioning, and scaling.
  - Class imbalance handling (Cost-sensitive weighting, Train-only SMOTE).
  - Supervised models (Logistic Regression, Random Forest) & Anomaly detection (Isolation Forest).
  - Unit testing suite & ML prediction API.
* **Member 2 (Security Rules & Presentation):**
  - Synthetic event generation & scenario simulations.
  - SIM Swap detection & OTP failure tracking.
  - Velocity checks & carding-pattern detection.
  - Streamlit dashboard.

---

## 2. Dataset Information

* **Source:** [Kaggle European Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group).
* **Expected Location:** `data/raw/creditcard.csv`
* **Characteristics:**
  - **Total Transactions:** 284,807
  - **Legitimate Transactions:** 284,315 (99.827%)
  - **Fraudulent Transactions:** 492 (0.1727%)
  - **Features:** 31 numeric columns (`Time`, `V1`–`V28` via PCA transformation, `Amount`, `Class`).
  - **Missing Values:** 0 null cells.

---

## 3. Environment Setup & Installation

### Prerequisites
* Windows / Linux / macOS
* Python 3.10 or higher

### Setup Commands
```bash
# 1. Clone repository and navigate to root
cd D:\Credit_Card_Fraud_Detection_System

# 2. Activate existing virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate

# 3. Verify / Install Dependencies
pip install -r requirements.txt
```

---

## 4. Execution & Verification

### Running the Unit Test Suite
To verify data validation, schema integrity, and leakage-free preprocessing:
```bash
pytest -v
```

### Running Data Loader Summary
```bash
python src/data/load_data.py
```

### Running Supervised Model Training & Benchmarking
```bash
python src/models/train_supervised.py
```

---

## 5. Machine Learning Viva Defense Highlights

### Why is Accuracy Misleading in Fraud Detection?
In this dataset, 99.83% of transactions are legitimate. A naive "dummy classifier" that predicts every transaction is legitimate achieves **99.827% accuracy** while detecting **zero fraud cases (0% Recall)**. 
Therefore, models must be evaluated using:
* **Recall (Sensitivity):** Fraction of actual frauds detected.
* **Precision:** Fraction of flagged transactions that are truly fraud.
* **PR-AUC (Precision-Recall Area Under Curve):** The gold standard for extreme class imbalances.

### Why RobustScaler over StandardScaler?
The `Amount` feature has severe positive skew with values ranging from 0 to over 25,000 EUR. `StandardScaler` calculates mean and standard deviation, which are heavily distorted by high-value outliers. `RobustScaler` scales data using the **median and Interquartile Range (IQR)**, preventing outlier distortion.

### How is Data Leakage Prevented?
* All scaling transformations are fit **strictly on training data** (`X_train`) and applied downstream to testing data (`X_test`).
* SMOTE synthetic oversampling is **strictly applied only to training folds**, keeping the test set 100% representative of real-world operational distributions.

---

## 6. Project Roadmap

- [x] **Phase 1:** Repository Audit, Data Loading, EDA, Preprocessing Pipeline, and Unit Tests.
- [x] **Phase 2:** Supervised Baseline Classification (Logistic Regression vs. Random Forest with SMOTE and Class Weighting).
- [ ] **Phase 3:** Unsupervised Anomaly Detection (Isolation Forest).
- [ ] **Phase 4:** Model Evaluation & Precision-Recall Calibration.
- [ ] **Phase 5:** ML Prediction Service Interface (`src/models/predict.py`).
- [ ] **Phase 6:** Integration with Member 2 via Unified Risk Assessment Engine.
