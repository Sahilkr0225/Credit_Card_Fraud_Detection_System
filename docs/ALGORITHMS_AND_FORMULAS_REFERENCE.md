# 📐 Algorithms, Mathematical Formulas & Architecture Trade-Offs Reference
## Intelligent Credit Card Fraud Detection & Risk Assessment System

> **Purpose:** Comprehensive mathematical, algorithmic, and architectural reference detailing every formula, hyperparameter selection, rationale ("Why Used"), and competitive trade-off ("Why NOT Alternatives") across all project components.

---

## Table of Contents
1. [Mathematical Formulations Summary Table](#1-mathematical-formulations-summary-table)
2. [Data Scaling & Transformation](#2-data-scaling--transformation)
   - [RobustScaler vs. StandardScaler vs. MinMaxScaler](#robustscaler-vs-standardscaler-vs-minmaxscaler)
3. [Class Imbalance Mitigation](#3-class-imbalance-mitigation)
   - [Balanced Cost-Sensitive Weighting vs. SMOTE vs. Random Undersampling](#balanced-cost-sensitive-weighting-vs-smote-vs-random-undersampling)
4. [Supervised Classification Algorithms](#4-supervised-classification-algorithms)
   - [Random Forest Classifier (Champion)](#random-forest-classifier-champion)
   - [Logistic Regression (Baseline)](#logistic-regression-baseline)
   - [Why NOT Gradient Boosting (XGBoost / LightGBM)?](#why-not-gradient-boosting-xgboost--lightgbm)
   - [Why NOT Deep Neural Networks (MLP)?](#why-not-deep-neural-networks-mlp)
   - [Why NOT Naive Bayes or Support Vector Machines (SVM)?](#why-not-naive-bayes-or-support-vector-machines-svm)
5. [Unsupervised Anomaly Detection](#5-unsupervised-anomaly-detection)
   - [Isolation Forest](#isolation-forest)
   - [Why NOT One-Class SVM (OC-SVM)?](#why-not-one-class-svm-oc-svm)
   - [Why NOT Local Outlier Factor (LOF)?](#why-not-local-outlier-factor-lof)
   - [Why NOT Autoencoders?](#why-not-autoencoders)
6. [Cybersecurity & Geolocation Telemetry](#6-cybersecurity--geolocation-telemetry)
   - [The Haversine Distance Formula](#the-haversine-distance-formula)
   - [Why NOT Euclidean or Manhattan Distance?](#why-not-euclidean-or-manhattan-distance)
   - [Implied Geo-Velocity Calculation](#implied-geo-velocity-calculation)
   - [In-Memory Sliding Time-Window Velocity Buffers](#in-memory-sliding-time-window-velocity-buffers)
7. [The Unified Risk Assessment Fusion Engine](#7-the-unified-risk-assessment-fusion-engine)
   - [Weighted Multi-Factor Fusion Formulation](#weighted-multi-factor-fusion-formulation)
   - [Non-Linear Security Hard Overrides](#non-linear-security-hard-overrides)
   - [Why NOT Stacking Classifier or Simple Average?](#why-not-stacking-classifier-or-simple-average)
8. [Evaluation Metrics & Financial Loss Modeling](#8-evaluation-metrics--financial-loss-modeling)
   - [Precision-Recall AUC vs. ROC-AUC vs. Accuracy](#precision-recall-auc-vs-roc-auc-vs-accuracy)
   - [Total Operational Financial Impact Function](#total-operational-financial-impact-function)

---

## 1. Mathematical Formulations Summary Table

| Category | Component / Formula | Primary Equation | Core Parameter Values |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | **RobustScaler** | $x_{\text{scaled}} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$ | $Q_2 = \text{median}, \text{IQR} = Q_3 - Q_1$ |
| **Imbalance** | **Balanced Class Weighting** | $w_j = \frac{N}{2 \cdot N_j}$ | $w_0 \approx 0.5009, w_1 \approx 289.44$ |
| **Imbalance** | **SMOTE Interpolation** | $x_{\text{new}} = x_i + \lambda(x_{zi} - x_i)$ | $\lambda \sim U(0,1), k=5$ nearest neighbors |
| **Supervised** | **Random Forest Ensemble** | $\hat{P}(Y=1 \mid X) = \frac{1}{B} \sum_{b=1}^{B} f_b(X)$ | $B=100$ trees, Gini split criterion |
| **Supervised** | **Logistic Regression** | $P(Y=1 \mid X) = \frac{1}{1 + e^{-(w^T X + b)}}$ | L2 regularization, $C=1.0$, balanced weights |
| **Anomaly** | **Isolation Forest Score** | $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$ | $n=256$ sub-samples, 100 trees, $\text{contam}=0.0017$ |
| **Telemetry** | **Haversine Distance** | $d = 2R \arcsin\left(\sqrt{\sin^2\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\frac{\Delta\lambda}{2}}\right)$ | Earth radius $R = 6,371\text{ km}$ |
| **Telemetry** | **Implied Geo-Velocity** | $v = \frac{d}{\Delta t}$ | Max threshold $v_{\text{max}} = 850\text{ km/h}$ |
| **Fusion** | **Unified Risk Score** | $R_{\text{total}} = 0.45 S_{\text{ML}} + 0.15 S_{\text{Anom}} + 0.40 S_{\text{Sec}}$ | $w_1=0.45, w_2=0.15, w_3=0.40$ |
| **Operations** | **Financial Cost Function** | $\text{Cost} = (FN \times C_{\text{fraud}}) + (FP \times C_{\text{friction}})$ | $C_{\text{fraud}} = \$500, C_{\text{friction}} = \$25$ |

---

## 2. Data Scaling & Transformation

### RobustScaler vs. StandardScaler vs. MinMaxScaler

#### The Formula:
$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$$

#### Why We Used RobustScaler:
- Features $V_1$ through $V_{28}$ are PCA-transformed variables with zero mean and near-unit variance. 
- However, `Time` (0 to 172,792 seconds) and `Amount` (€0.00 to €25,691.00) operate on completely unscaled domains.
- `Amount` is heavily right-skewed with extreme positive outliers ($75\text{th percentile} = €77.165$, but $\text{max} = €25,691.00$).
- `RobustScaler` maps values relative to the **median and Interquartile Range ($IQR$)**, rendering the scaling completely insensitive to extreme high-value fraud or luxury outliers.

#### Why NOT StandardScaler?
$$\mu = \frac{1}{N}\sum x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}, \quad z = \frac{x - \mu}{\sigma}$$
- The mean ($\mu$) and standard deviation ($\sigma$) are quadratic with respect to distance and are violently skewed by the €25,691 outlier. This results in an artificially compressed scale where 98% of valid transactions sit in a narrow band ($z \in [-0.3, 0.2]$).

#### Why NOT MinMaxScaler?
$$x_{\text{norm}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
- Because $x_{\text{max}} = €25,691$, a standard €50 grocery purchase maps to $\frac{50}{25691} \approx 0.0019$. This compresses normal transaction variance into near-zero values, destroying numerical precision in tree splitting and gradient computations.

---

## 3. Class Imbalance Mitigation

### Balanced Cost-Sensitive Weighting vs. SMOTE vs. Random Undersampling

#### 1. Balanced Class Weighting (Chosen Champion):
$$w_j = \frac{N}{2 \cdot N_j}$$
For our training set ($N = 227,845$, $N_0 = 227,451$ legitimate, $N_1 = 394$ fraud):
$$w_0 = \frac{227845}{2 \times 227451} \approx 0.5009, \quad w_1 = \frac{227845}{2 \times 394} \approx 289.14$$

- **Why Used:** Modifies the loss function during tree impurity calculation without creating artificial samples or deleting real data:
  $$\text{Loss}(y, \hat{y}) = - \left[ w_1 y \log(\hat{y}) + w_0 (1-y) \log(1 - \hat{y}) \right]$$
  Achieved **84.04% precision with only 15 False Positives**.

#### 2. Synthetic Minority Over-sampling Technique (SMOTE):
$$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$
- **Why NOT Chosen as Champion:** While SMOTE yielded slightly higher Recall in Random Forest (85.71% vs 80.61%), it generated **80 False Positives** compared to only 15 for Class Weighting. This caused Precision to plummet from **84.0% down to 51.2%**, introducing excessive false alarms.

#### 3. Why NOT Random Undersampling (RUS)?
- Reducing the legitimate majority from $227,451$ down to $394$ to match fraud throws away **99.8% of available training data**. This discards valuable legitimate behavioral variety and results in extreme variance on unseen test data.

---

## 4. Supervised Classification Algorithms

### Random Forest Classifier (Champion)

#### The Formulation:
An ensemble of $B = 100$ randomized decision trees trained via bootstrap aggregation (bagging) with random feature subspace sampling ($m = \sqrt{p} \approx \sqrt{30} \approx 5$):
$$\hat{P}(Y=1 \mid X) = \frac{1}{B} \sum_{b=1}^{B} f_b(X)$$

Split criterion is Gini Impurity weighted by cost weights $w_j$:
$$I_G(p) = 1 - \sum_{j=0}^{1} p_j^2$$

#### Why Chosen:
1. **Non-Linear Decision Boundaries:** Highly effective at capturing non-linear interactions among latent PCA features (e.g., negative shifts in $V_{14}$ combined with positive shifts in $V_4$).
2. **Empirical Supremacy:** Highest PR-AUC (**0.8311**), highest F1-score (**0.8229**), and lowest False Positive count (**15** out of 56,864 legitimate transactions).
3. **Resilience to Overfitting:** The combination of bagging and feature subspace randomization prevents individual trees from memorizing localized noise.

---

### Logistic Regression (Baseline)

#### The Formulation:
$$P(Y=1 \mid X) = \sigma(w^T X + b) = \frac{1}{1 + e^{-(w^T X + b)}}$$

#### Why Evaluated & Why NOT Chosen:
- **Evaluated because:** It serves as the standard, interpretable linear baseline required in institutional financial modeling.
- **Why NOT Chosen:** Because credit card fraud is inherently non-linear, Logistic Regression's linear hyperplane cannot isolate localized fraud clusters without aggressive threshold shifts. Shifting the threshold to catch 91.8% of fraud forced it to generate **1,388 False Positives**, rendering it commercially unviable ($6.09\%$ precision).

---

### Why NOT Gradient Boosting (XGBoost / LightGBM)?
- **Overfitting Risk on Rare Classes:** In datasets with 0.17% minority prevalence, iterative gradient boosting continually concentrates residual weights on misclassified boundary outliers, frequently fitting noise in the fraud class unless aggressively constrained.
- **Inference Latency & Portability:** Random Forest allows embarrassingly parallel tree evaluation and serializes into lightweight, deterministic C-level pipelines without requiring OpenMP or custom native runtime dependencies in production microservices.

---

### Why NOT Deep Neural Networks (MLP)?
- **Tabular Data Superiority:** Extensive empirical studies (e.g., Grinsztajn et al., NeurIPS 2022 *"Why do tree-based models still outperform deep learning on tabular data?"*) demonstrate that tree ensembles consistently outperform Deep Learning architectures on tabular data with uninformative or rotated coordinates.
- **Explainability:** Neural networks function as black boxes. In banking, adverse action notices require explaining *why* a card was blocked (e.g., Gini feature importances showing $V_{14}$ deviations).

---

### Why NOT Naive Bayes or Support Vector Machines (SVM)?
- **Naive Bayes Assumption Violation:** Naive Bayes assumes conditional feature independence:
  $$P(X \mid Y) = \prod_{i=1}^{30} P(X_i \mid Y)$$
  While PCA features $V_1$–$V_{28}$ are linearly uncorrelated, they exhibit strong non-linear and high-order dependencies, and `Amount` correlates with multiple latent factors.
- **SVM Computational Infeasibility:** Non-linear kernel SVMs (RBF kernel) scale quadratically with sample size: $O(N^2)$ to $O(N^3)$. Training on $N = 227,845$ records requires hours of computation and huge memory overhead.

---

## 5. Unsupervised Anomaly Detection

### Isolation Forest

#### The Formulation:
Isolates anomalies by recursively partitioning data using random axis-parallel cuts:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
Where $E(h(x))$ is the average path length across all isolation trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree:
$$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler-Mascheroni constant)} - \frac{2(n - 1)}{n}$$
- **When $E(h(x)) \to 0 \implies s \to 1$:** Instance isolates at shallow depth $\to$ **Anomaly**.
- **When $E(h(x)) \to n - 1 \implies s \to 0$:** Instance requires deep recursive splits $\to$ **Inlier**.

#### Why Chosen:
1. **Zero-Day Catching:** Trained without any labels ($unsupervised$). It isolates outliers based strictly on structural density, catching novel attack techniques never recorded in past training labels.
2. **Computational Efficiency:** Operates with linear time complexity $O(t \cdot \psi \log \psi)$ ($t=100$ trees, $\psi=256$ sub-sample size), enabling sub-millisecond real-time scoring.
3. **High Empirical Discrimination:** Achieved an **ROC-AUC of 0.9528** on unseen test data.

---

### Why NOT One-Class SVM (OC-SVM)?
- Standard OC-SVM with RBF kernel requires solving a quadratic programming optimization problem with complexity $O(N^2 \cdot p)$. For $284,807$ instances, this requires over 50 GB of RAM and tens of minutes of compute time.
- Highly sensitive to hyperparameter tuning ($\nu$ and $\gamma$).

### Why NOT Local Outlier Factor (LOF)?
- LOF computes $k$-nearest-neighbor local densities. At inference time, scoring a single real-time transaction requires querying pairwise Euclidean distances against all 227,000 reference points ($O(N \cdot p)$), causing unacceptable multi-second latency for live payment processing.

### Why NOT Autoencoders?
- Autoencoders (reconstruction error via $|x - \hat{x}|^2$) require extensive hyperparameter search (bottleneck dimension, activation functions, learning rate schedules) and are prone to reconstructing anomalous patterns if trained without strict regularization.

---

## 6. Cybersecurity & Geolocation Telemetry

### The Haversine Distance Formula

#### The Mathematical Formulation:
Computes the great-circle distance between two geographic coordinate pairs on a spherical Earth of radius $R = 6,371\text{ km}$:
$$\Delta \phi = \text{lat}_2 - \text{lat}_1, \quad \Delta \lambda = \text{lon}_2 - \text{lon}_1$$
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\text{lat}_1) \cdot \cos(\text{lat}_2) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$
$$d = R \cdot c$$

#### Implied Velocity:
$$v = \frac{d}{\Delta t \text{ (hours)}} = \frac{d}{(t_2 - t_1) / 3600}$$

#### Rule Boundary:
$$v > 850.0\text{ km/h} \implies \text{IMPOSSIBLE\_TRAVEL\_SPEED\_EXCEEDED}$$

---

### Why NOT Euclidean or Manhattan Distance?
- **Planar Distortion:** Euclidean distance ($d = \sqrt{(\Delta x)^2 + (\Delta y)^2}$) treats latitude and longitude as Cartesian coordinates on a flat plane. 
- Because meridians of longitude converge at the poles (where 1 degree of longitude shrinks from 111 km at the equator to 0 km at the poles), Euclidean distance introduces catastrophic errors ($> 40\%$) over long-distance flights (e.g., New York to Tokyo).

---

### In-Memory Sliding Time-Window Velocity Buffers
Thread-safe sliding ring buffers maintain transaction event history for each active cardholder over multiple temporal windows:
$$\text{Window Volume}(u, \Delta t) = \sum_{i} I(t_i > t_{\text{now}} - \Delta t)$$
$$\text{Window Spend}(u, \Delta t) = \sum_{i} \text{Amount}_i \cdot I(t_i > t_{\text{now}} - \Delta t)$$
- **1-Minute Window:** Detects automated API carding loops ($> 3$ transactions $\implies$ `VELOCITY_BURST_1M`).
- **5-Minute Window:** Detects bot-driven micro-transaction card-testing ($\le \$15.00$ each, $\ge 3$ transactions).
- **1-Hour Window:** Detects sudden spending velocity spikes ($> 3\times$ user historical average).

---

## 7. The Unified Risk Assessment Fusion Engine

### Weighted Multi-Factor Fusion Formulation

#### The Formula:
$$R_{\text{total}} = w_{\text{ML}} S_{\text{ML}} + w_{\text{Anomaly}} S_{\text{Anomaly}} + w_{\text{Security}} S_{\text{Security}}$$
Where:
- $S_{\text{ML}} = P(\text{Fraud} \mid X) \in [0, 1]$ (Random Forest output).
- $S_{\text{Anomaly}} = \text{clip}\left(\frac{- \text{Score}_{\text{iforest}} + 0.10}{0.50}, 0.0, 1.0\right)$ (Normalized Isolation Forest outlier intensity).
- $S_{\text{Security}} = \min\left(1.0, \sum \text{Rule Penalty Weights}\right)$.
- **Default System Weights:**
  $$w_{\text{ML}} = 0.45, \quad w_{\text{Anomaly}} = 0.15, \quad w_{\text{Security}} = 0.40$$
  $$\sum w = 1.00$$

---

### Non-Linear Security Hard Overrides

#### Formulation:
$$\text{If } \Big( (\text{SIM\_SWAP\_AGE} < 24\text{h}) \lor (v > 850\text{ km/h}) \lor (\text{OTP\_FAILURES} \ge 3) \Big) \implies R_{\text{total}} = \max(R_{\text{total}}, 0.90) \land \text{Decision} = \text{DECLINE}$$

#### Why Hard Overrides Are Mathematically Necessary:
A linear weighted average has a critical flaw: if a fraudster executes a transaction that appears normal to the ML model ($S_{\text{ML}} = 0.10$), the weighted sum would be:
$$R = (0.45 \times 0.10) + (0.15 \times 0.20) + (0.40 \times 1.00) = 0.045 + 0.030 + 0.400 = 0.475$$
A score of $0.475$ falls in the `CHALLENGE_OTP` range. But if a SIM swap occurred, **the fraudster has already intercepted the OTP**! A challenge is completely useless.
Therefore, the non-linear **Hard Override** prevents ML dilution and guarantees an instant `DECLINE`.

---

### Why NOT a Stacking Meta-Classifier?
- **Auditability and Regulatory Transparency:** In banking (under Fair Credit Reporting Act and Basel Accord guidelines), every decline must be accompanied by explicit adverse reason codes. A linear fusion engine with rule-based hard overrides provides full auditability, while a non-linear second-stage meta-classifier would re-introduce black-box unexplainability.

---

## 8. Evaluation Metrics & Financial Loss Modeling

### Precision-Recall AUC vs. ROC-AUC vs. Accuracy

| Metric | Mathematical Formula | Sensitivity to Class Imbalance | Operational Meaning in Banking |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **Catastrophic Failure** ($99.82\%$ for naive dummy) | Irrelevant; hides missed fraud completely. |
| **ROC-AUC** | $\int_0^1 \text{TPR} \, d(\text{FPR})$ | **Misleadingly Optimistic** (Large $TN$ masks large $FP$) | Measures separability across all possible thresholds. |
| **Recall** | $\frac{TP}{TP + FN}$ | Focuses purely on fraud coverage | Percentage of fraudulent transactions stopped. |
| **Precision** | $\frac{TP}{TP + FP}$ | **Directly penalizes false alarms** | Percentage of alerted transactions that are actually fraud. |
| **PR-AUC** | $\int_0^1 \text{Precision} \, d(\text{Recall})$ | **The Gold Standard for Imbalance** | Evaluates true business precision across all recall targets. |

---

### Total Operational Financial Impact Function

#### The Formula:
$$\text{Cost}_{\text{Total}} = (FN \times C_{\text{fraud}}) + (FP \times C_{\text{friction}})$$
Where:
- $FN$ = Missed fraudulent transactions (chargebacks, merchant disputes, direct capital loss).
- $FP$ = Falsely declined legitimate transactions (customer service call center costs, card re-issuance, brand damage).
- Benchmark empirical parameters: $C_{\text{fraud}} = \$500.00$, $C_{\text{friction}} = \$25.00$.

#### Comparative Financial Analysis (56,962 Holdout Transactions):
- **Logistic Regression (Class-Weighted):**
  $$\text{Cost} = (8 \times \$500) + (1,388 \times \$25) = \$4,000 + \$34,700 = \mathbf{\$38,700.00}$$
- **Random Forest Champion (Class-Weighted):**
  $$\text{Cost} = (19 \times \$500) + (15 \times \$25) = \$9,500 + \$375 = \mathbf{\$9,875.00}$$
- **Net Operational Financial Savings:**
  $$\text{Savings} = \$38,700 - \$9,875 = \mathbf{\$28,825.00} \quad \mathbf{(74.5\% \text{ Cost Reduction})}$$
