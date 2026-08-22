# 🛡️ AI-Powered Network Security & DDoS Detection Platform

An end-to-end machine learning system that classifies network traffic as **Attack**
or **Benign**, explains *why* in plain English, scores risk on a 0–100 scale, and
surfaces everything through an interactive Streamlit dashboard — built and
benchmarked on a 10,000-row network traffic dataset.

---

## 🌟 Features

- **Exploratory Data Analysis** — distribution, outlier, and correlation analysis of raw traffic
- **Multi-model training & comparison** — Logistic Regression, Decision Tree, Random Forest, and XGBoost, evaluated head-to-head on held-out test data
- **Leakage-safe feature engineering** — URL-based SQL injection/XSS/command-injection pattern detection, IP privacy flags, time-based features, port-frequency behavioral features
- **Threshold optimization** — F1-optimal and recall-at-precision-floor threshold selection, validated via cross-validated out-of-fold predictions
- **Extended model search** — LightGBM, CatBoost, and stacked ensembles benchmarked against the baseline
- **Live Detection** — manually enter a traffic record and get an instant Attack/Benign verdict with probability, risk score, and rule-based explanation
- **Batch Analysis** — upload a CSV (or use the built-in demo sample) and score hundreds of records at once
- **Rule-based explainability** — every flagged record comes with plain-English reasons (e.g. *"SQL injection pattern detected in URL"*, *"User agent identifies as 'sqlmap'"*), not just a black-box score
- **Security report generation** — downloadable TXT/PDF reports summarizing findings
- **Interactive dashboard** — dark-themed, gauge charts, risk-band visualizations, filterable results table

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Dashboard | Streamlit |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Report Generation | fpdf2 |

---

## 📊 Dataset

**`cybersecurity.csv`** — 10,000 rows of simulated network traffic, ~4% labeled malicious.

### Raw columns

| Column | Description |
|---|---|
| `timestamp` | Date/time of the traffic event |
| `src_ip`, `dst_ip` | Source and destination IP addresses |
| `src_port`, `dst_port` | Source and destination ports |
| `protocol` | TCP / UDP / ICMP |
| `bytes_sent`, `bytes_received` | Traffic volume |
| `user_agent` | Client user-agent string |
| `url` | Requested URL (missing for non-HTTP traffic) |
| `is_internal_traffic` | Whether traffic stayed within the internal network |
| `label` | 0 = Benign, 1 = Attack (target variable) |
| `attack_type` | Ground-truth attack category (benign, sql-injection, xss, command-injection, brute-force, port-scan, ddos, credential-stuffing, exploit-attempt, c2) |

### Class balance

```
label
0    9,600   (96.0%)
1      400   (4.0%)
```

### Attack type breakdown

Roughly **31%** of attacks (SQL injection, XSS, command injection) leave a detectable
trace in the URL. The remaining **69%** (brute-force, port-scan, DDoS,
credential-stuffing, exploit-attempt, C2) have no URL signature and must be caught
through traffic volume/port anomalies instead — this split is central to
understanding the model's precision/recall ceiling (see below).

### Engineered features (42 total, via `feature_pipeline.py`)

- **Time-based**: hour, day, month, day-of-week, is_weekend, is_business_hours
- **IP-based**: src/dst IP privacy flags
- **User-agent**: categorized into browser / curl / python_requests / sqlmap / zgrab / googlebot / other
- **URL security signatures**: SQL/XSS/command-injection keyword counts, suspicious symbol density, path traversal detection, SQL comment detection, and 8 more binary attack-pattern flags
- **Traffic volume**: total bytes, bytes sent/received ratio
- **Behavioral**: source/destination port frequency (computed leakage-safe — fit on training data only)

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip

---

## 🚀 Installation

```bash
# 1. Get the project files (adjust if hosted elsewhere)
cd network-attack-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Place `cybersecurity.csv` in the project root.

---

## 🎯 Usage

### Run the core ML pipeline

```bash
python 01_EDA.py                    # exploratory analysis
python 02_PREPROCESSING.py          # feature engineering walkthrough
python 03_MODEL_TRAINING.py         # trains & selects the final model
python 06_ALL_MODELS_TEST_EVAL.py   # confirms model choice on test data
```

### Run live/interactive prediction (console)

```bash
python 09_TRAIN_FINAL_MODEL.py      # run once — saves model_artifacts/
python 10_DDOS_PREDICTION.py        # prompts for traffic details
python 11_PROBABILITY.py
python 12_RISK_SCORE_AND_EXPLANATION.py
python 13_SECURITY_REPORT.py
```

### Launch the dashboard

```bash
streamlit run 14_DASHBOARD.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`). From there:
1. **Security Overview** — see attack percentage and risk breakdown across the demo dataset automatically
2. **Live Detection** — enter protocol, ports, byte counts, IPs, and optionally a user-agent/URL, then click **Analyze Traffic**
3. **Batch Analysis** — upload your own CSV or use the demo sample; filter results and download a report

---

## 📈 Actual Test Results

Sample output from a real run of `03_MODEL_TRAINING.py`:

```
===== FINAL XGBOOST CONFUSION MATRIX =====
[[1890   30]
 [  41   39]]

======================================================================
FINAL XGBOOST TEST RESULTS
======================================================================
Selected Threshold : 0.76

Accuracy : 0.9645
Precision: 0.5652
Recall   : 0.4875
F1-Score : 0.5235
ROC-AUC  : 0.8931
PR-AUC   : 0.5448

              precision    recall  f1-score   support
           0       0.98      0.98      0.98      1920
           1       0.57      0.49      0.52        80
```

Sample output from `09_TRAIN_FINAL_MODEL.py`:

```json
{
  "threshold": 0.88,
  "val_accuracy": 0.96875,
  "val_precision": 0.5946,
  "val_recall": 0.6875,
  "val_f1": 0.6377,
  "val_roc_auc": 0.8688,
  "test_accuracy": 0.962,
  "test_precision": 0.525,
  "test_recall": 0.525,
  "test_f1": 0.525,
  "test_roc_auc": 0.9190
}
```

Sample output from a **live detection** run (`10_DDOS_PREDICTION.py`, sqlmap
SQL-injection attempt):

```
Prediction : Attack
Model probability : 96.48%

🚨 WARNING: MALICIOUS / ATTACK TRAFFIC DETECTED

Risk Score : 96.5 / 100   (Critical)
Reasons:
  - SQL injection pattern detected in URL (e.g. UNION SELECT, OR 1=1, SQL comments)
  - User agent identifies as 'sqlmap' -- a known automated SQL injection tool
```

### Results table (across all experiments)

| Approach | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.320 | 0.487 | 0.386 | 0.884 |
| Decision Tree | 0.442 | 0.425 | 0.433 | 0.743 |
| Random Forest | 0.402 | 0.562 | 0.469 | 0.891 |
| **XGBoost (baseline, selected model)** | **0.53–0.57** | **0.49–0.54** | **0.52–0.54** | **0.91–0.93** |
| LightGBM (expanded search) | 0.594 | 0.512 | 0.550 | 0.910 |
| XGBoost + targeted anomaly features* | 0.540–0.57 | 0.588–0.613 | 0.563–0.59 | 0.913–0.916 |


## ❓ Why Are Precision & Recall Limited?

This was investigated in depth during development. Three separate, compounding
reasons — not a modeling mistake:

### 1. Extreme class imbalance with a small positive sample
Only **400 malicious rows out of 10,000** (4%), split further into train/val/test —
as few as **64–80 malicious examples** in validation/test. At this sample size,
precision/recall/F1 naturally swing several points between runs due to
GridSearchCV/SMOTE randomness. This isn't unique to this project; it's inherent to
any rare-event detection task at this data scale.

### 2. Most attack types have no row-level signal, by construction
Every `src_ip`/`dst_ip` in this dataset is **unique across all 10,000 rows** — no IP
ever repeats. That means attack types requiring session/time-window context
(brute-force, port-scan, DDoS, credential-stuffing, exploit-attempt, C2 — **69% of
all attacks**) cannot be detected the way they would be in real deployment, where
seeing the *same* IP hit many ports/endpoints in a short window is the actual
signal. Only SQL injection/XSS/command-injection (~31% of attacks) leave a
detectable trace in a single row's URL. This caps achievable recall regardless of
model choice — confirmed by testing wider hyperparameter search and three
different gradient boosting libraries (07), none of which meaningfully broke past
this ceiling.

### 3. The best-performing engineered feature is likely a dataset artifact
`src_port_equals_dst_port` (used in the improved-F1 experiment, `08`) is the single
strongest predictor found — but real ephemeral source ports have no mechanism to
equal the destination port; a real attacker doesn't control their OS-assigned
source port. This strongly suggests the synthetic dataset's generator took a
shortcut when producing brute-force/credential-stuffing rows, rather than this
being a genuine attacker signature. The improvement is real *on this dataset* but
should not be assumed to generalize to real-world traffic without independent
validation.

**Bottom line:** given points 1–2, an F1 around 0.52–0.59 represents close to what's
extractable from this feature set — the modeling and threshold-selection choices
were stress-tested (wider search, alternate algorithms, cross-validated thresholds,
recall-floor optimization) and none moved the ceiling substantially, which is itself
evidence the constraint is in the data, not the model.

---

## 📁 Project Structure

| # | File | Purpose |
|---|---|---|
| 1 | `01_EDA.py` | Exploratory data analysis |
| 2 | `02_PREPROCESSING.py` | Preprocessing walkthrough |
| — | `feature_pipeline.py` | Shared feature engineering module |
| 3 | `03_MODEL_TRAINING.py` | Trains & selects the final model |
| 4 | `04_RECALL_AT_PRECISION_FLOOR.py` | Precision/recall trade-off exploration |
| 5 | `05_CV_THRESHOLD_SELECTION.py` | Cross-validated threshold stability check |
| 6 | `06_ALL_MODELS_TEST_EVAL.py` | All 4 models compared on held-out test |
| 7 | `07_EXPANDED_SEARCH_AND_ALT_MODELS.py` | Wider search + LightGBM/CatBoost/ensemble |
| 8 | `08_TARGETED_ANOMALY_FEATURES.py` | Best-F1 experiment (see caveat above) |
| — | `inference_engine.py` | Shared prediction/risk-scoring/explanation module |
| 9 | `09_TRAIN_FINAL_MODEL.py` | Trains & persists the production model |
| 10 | `10_DDOS_PREDICTION.py` | Interactive manual traffic entry → prediction |
| 11 | `11_PROBABILITY.py` | Adds attack probability |
| 12 | `12_RISK_SCORE_AND_EXPLANATION.py` | Adds risk score + rule-based explanation |
| 13 | `13_SECURITY_REPORT.py` | Generates the final text report |
| 14 | `14_DASHBOARD.py` | Full interactive Streamlit dashboard |
| — | `requirements.txt` | Pinned dependencies |

---

## ⚠️ Limitations & Honest Caveats

- Metrics vary by several points run-to-run due to the small positive sample size — treat reported numbers as approximate operating points, not fixed values.
- Accuracy (~96%) is not a meaningful headline metric given the class imbalance; F1/Recall/ROC-AUC are the metrics that reflect actual detection quality.
- The dataset's structure (unique IPs per row) means this model cannot detect session-based attacks the way a production system with real traffic history could.
- `src_port_equals_dst_port` (used in the `08` experiment) should be validated against real traffic before trusting it in production — see caveat above.
- Live Detection inputs (IP addresses, user-agent, URL) default to realistic placeholders when left blank, but predictions are most accurate when real observed values are entered.

---

## 🔮 Future Improvements

- Collect more malicious traffic samples — the single highest-leverage fix for recall
- Add session/time-window aggregation features (requires traffic with repeated IPs over time, unlike the current synthetic dataset)
- Validate the `src_port_equals_dst_port` signal against real-world traffic before relying on it
- Integrate with a live traffic feed / SIEM for real-time alerting
- Add model explainability via SHAP for a more granular, per-feature attribution view alongside the current rule-based explanations

---

## 🙏 Acknowledgments

- XGBoost, scikit-learn, and imbalanced-learn teams for the core ML tooling
- Streamlit for the dashboard framework
- Plotly for interactive visualizations

---

## 📝 Note

This is an academic/internship project built for learning and demonstration
purposes. Model results should not be used for production security decisions
without further validation on real-world traffic data.