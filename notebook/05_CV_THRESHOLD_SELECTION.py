
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)

from feature_pipeline import build_dataset, make_preprocessor


data = build_dataset()
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]

best_params = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=18.0,
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    **best_params,
)

pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("model", model),
])


# ============================================================
# STEP 1 - OUT-OF-FOLD PROBABILITIES ON TRAINING DATA
# ============================================================
print("Generating out-of-fold predictions on x_train (5-fold CV)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_prob = cross_val_predict(
    pipeline, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

print(f"Out-of-fold predictions collected for {len(oof_prob)} rows "
      f"({y_train.sum()} positive).")


# ============================================================
# STEP 2 - THRESHOLD SWEEP AGAINST OOF PREDICTIONS
# ============================================================
thresholds = np.arange(0.10, 0.91, 0.01)
rows = []
for t in thresholds:
    pred = (oof_prob >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_train, pred, zero_division=0),
        "recall": recall_score(y_train, pred, zero_division=0),
        "f1": f1_score(y_train, pred, zero_division=0),
    })
sweep = pd.DataFrame(rows)

best_row = sweep.loc[sweep["f1"].idxmax()]
cv_threshold = best_row["threshold"]

print("\n===== BEST THRESHOLD FROM CROSS-VALIDATED (OOF) SWEEP =====")
print(best_row)


# ============================================================
# STEP 3 - FIT ON FULL TRAINING SET, CHECK VAL AS A SANITY CHECK
# ============================================================
pipeline.fit(x_train, y_train)

val_prob = pipeline.predict_proba(x_val)[:, 1]
val_pred = (val_prob >= cv_threshold).astype(int)

print(f"\n===== SANITY CHECK ON VALIDATION @ CV THRESHOLD {cv_threshold:.2f} =====")
print(confusion_matrix(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred, zero_division=0))
print("Recall   :", recall_score(y_val, val_pred, zero_division=0))
print("F1-Score :", f1_score(y_val, val_pred, zero_division=0))


# ============================================================
# STEP 4 - FINAL TEST EVALUATION
# ============================================================
test_prob = pipeline.predict_proba(x_test)[:, 1]
test_pred = (test_prob >= cv_threshold).astype(int)

print(f"\n===== TEST RESULTS @ CV THRESHOLD {cv_threshold:.2f} =====")
print(confusion_matrix(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred, zero_division=0))
print("F1-Score :", f1_score(y_test, test_pred, zero_division=0))


# ============================================================
# COMPARISON: OOF-DERIVED THRESHOLD VS. THE ORIGINAL VAL-DERIVED ONE
# ============================================================
val_sweep_rows = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    val_sweep_rows.append({
        "threshold": t,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
    })
val_sweep = pd.DataFrame(val_sweep_rows)
val_only_threshold = val_sweep.loc[val_sweep["f1"].idxmax(), "threshold"]

print("\n\n===== COMPARISON: threshold source vs. test performance =====")
for label, t in [("Validation-only (03_MODEL_TRAINING.py approach)", val_only_threshold),
                  ("Cross-validated OOF (this script)", cv_threshold)]:
    pred = (test_prob >= t).astype(int)
    print(f"{label:52s} threshold={t:.2f}  "
          f"precision={precision_score(y_test, pred, zero_division=0):.3f}  "
          f"recall={recall_score(y_test, pred, zero_division=0):.3f}  "
          f"f1={f1_score(y_test, pred, zero_division=0):.3f}")
