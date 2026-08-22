import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)

from feature_pipeline import build_dataset, make_preprocessor


PRECISION_FLOOR = 0.40  # <-- tune this: minimum acceptable precision


data = build_dataset()
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]


# ============================================================
# FIT XGBOOST WITH THE BEST HYPERPARAMETERS FROM 03's GRID SEARCH
# ============================================================
# Reusing these instead of re-running GridSearchCV -- swap in your
# own best_params_ here if you re-tune.
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

pipeline.fit(x_train, y_train)

val_prob = pipeline.predict_proba(x_val)[:, 1]
test_prob = pipeline.predict_proba(x_test)[:, 1]


# ============================================================
# THRESHOLD SWEEP ON VALIDATION
# ============================================================
thresholds = np.arange(0.10, 0.91, 0.01)
rows = []
for t in thresholds:
    pred = (val_prob >= t).astype(int)
    rows.append({
        "threshold": t,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
    })
sweep = pd.DataFrame(rows)


def evaluate_at_threshold(name, t, x, y):
    pred = (pipeline.predict_proba(x)[:, 1] >= t).astype(int)
    print(f"\n--- {name} @ threshold {t:.2f} ---")
    print(confusion_matrix(y, pred))
    print("Precision:", precision_score(y, pred, zero_division=0))
    print("Recall   :", recall_score(y, pred, zero_division=0))
    print("F1-Score :", f1_score(y, pred, zero_division=0))


# ---- Option A: F1-optimal threshold (what 03_MODEL_TRAINING.py does) ----
f1_optimal_row = sweep.loc[sweep["f1"].idxmax()]
f1_optimal_threshold = f1_optimal_row["threshold"]

print("=" * 70)
print("F1-OPTIMAL THRESHOLD (baseline, same criterion as 03)")

print(f"\nThreshold: {f1_optimal_threshold:.2f}")
print(f"Validation -> precision={f1_optimal_row['precision']:.3f}, "
      f"recall={f1_optimal_row['recall']:.3f}, f1={f1_optimal_row['f1']:.3f}")
evaluate_at_threshold("TEST (F1-optimal threshold)", f1_optimal_threshold, x_test, y_test)


# ---- Option B: max recall subject to precision >= PRECISION_FLOOR ----
candidates = sweep[sweep["precision"] >= PRECISION_FLOOR]

print("\n\n" + "=" * 70)
print(f"RECALL-OPTIMAL THRESHOLD (precision floor = {PRECISION_FLOOR})")

if candidates.empty:
    print(f"\nNo threshold in the sweep reaches precision >= {PRECISION_FLOOR}. "
          f"Try lowering PRECISION_FLOOR.")
else:
    recall_optimal_row = candidates.loc[candidates["recall"].idxmax()]
    recall_optimal_threshold = recall_optimal_row["threshold"]

    print(f"\nThreshold: {recall_optimal_threshold:.2f}")
    print(f"Validation -> precision={recall_optimal_row['precision']:.3f}, "
          f"recall={recall_optimal_row['recall']:.3f}, f1={recall_optimal_row['f1']:.3f}")
    evaluate_at_threshold("TEST (recall-optimal threshold)", recall_optimal_threshold, x_test, y_test)

    print("\n\n" + "=" * 70)
    print("SIDE-BY-SIDE SUMMARY (test set)")
    print("=" * 70)
    for name, t in [("F1-optimal", f1_optimal_threshold),
                     ("Recall-optimal (floor)", recall_optimal_threshold)]:
        pred = (test_prob >= t).astype(int)
        print(f"{name:28s} threshold={t:.2f}  "
              f"precision={precision_score(y_test, pred, zero_division=0):.3f}  "
              f"recall={recall_score(y_test, pred, zero_division=0):.3f}  "
              f"f1={f1_score(y_test, pred, zero_division=0):.3f}")
