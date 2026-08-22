import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from feature_pipeline import build_dataset, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


data = build_dataset()
x_train, y_train = data["x_train"].copy(), data["y_train"]
x_val, y_val = data["x_val"].copy(), data["y_val"]
x_test, y_test = data["x_test"].copy(), data["y_test"]


def add_anomaly_flags(x):
    x = x.copy()

    # Brute-force / credential-stuffing pattern found in this dataset.
    x["src_port_equals_dst_port"] = (x["src_port"] == x["dst_port"]).astype(int)

    # DDoS / port-scan pattern: extreme high/low byte volume.
    # Thresholds are derived from TRAIN quantiles only (not val/test),
    # to avoid leaking test-set distribution info into the feature.
    return x


# Compute thresholds from x_train only (leakage-safe).
total_bytes_train = x_train["bytes_sent"] + x_train["bytes_received"]
high_bytes_threshold = total_bytes_train.quantile(0.99)   # extreme outlier = DDoS-like
low_bytes_threshold = total_bytes_train.quantile(0.02)    # tiny volume = port-scan-like


def add_byte_flags(x):
    x = x.copy()
    total = x["bytes_sent"] + x["bytes_received"]
    x["is_ddos_like_bytes"] = (total >= high_bytes_threshold).astype(int)
    x["is_portscan_like_bytes"] = (total <= low_bytes_threshold).astype(int)
    return x


x_train = add_byte_flags(add_anomaly_flags(x_train))
x_val = add_byte_flags(add_anomaly_flags(x_val))
x_test = add_byte_flags(add_anomaly_flags(x_test))

new_numerical_columns = NUMERICAL_COLUMNS + [
    "src_port_equals_dst_port", "is_ddos_like_bytes", "is_portscan_like_bytes",
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ("numerical", StandardScaler(), new_numerical_columns),
])

best_params = dict(
    n_estimators=150, max_depth=6, learning_rate=0.03, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1, scale_pos_weight=18.0,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_jobs=-1, tree_method="hist", verbosity=0, **best_params,
    )),
])

pipeline.fit(x_train, y_train)

val_prob = pipeline.predict_proba(x_val)[:, 1]
test_prob = pipeline.predict_proba(x_test)[:, 1]


def best_threshold(y_true, prob):
    thresholds = np.arange(0.10, 0.91, 0.01)
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    return sweep.loc[sweep["f1"].idxmax()]


row = best_threshold(y_val, val_prob)
threshold = row["threshold"]

print("XGBOOST + TARGETED ANOMALY FLAGS")
print(f"\nNew features added: src_port_equals_dst_port, "
      f"is_ddos_like_bytes (>= {high_bytes_threshold:,.0f} total bytes), "
      f"is_portscan_like_bytes (<= {low_bytes_threshold:,.0f} total bytes)")
print(f"\nThreshold (val, F1-optimal): {threshold:.2f}")
print(f"Validation -> precision={row['precision']:.3f}, "
      f"recall={row['recall']:.3f}, f1={row['f1']:.3f}")

test_pred = (test_prob >= threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_prob)

print(f"\n----- TEST RESULTS (with anomaly flags) -----")
print(confusion_matrix(y_test, test_pred))
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-Score : {f1:.3f}")
print(f"ROC-AUC  : {auc:.3f}")

print("\nCompare against the 03_MODEL_TRAINING.py baseline (no anomaly flags): "
      "test F1 was ~0.52-0.54.")

# Feature importance -- check whether the new flags actually matter,
# or whether the model ignored them.
feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)

print("\n----- Where do the new anomaly flags rank in feature importance? -----")
new_flag_names = [f"numerical__{c}" for c in
                   ["src_port_equals_dst_port", "is_ddos_like_bytes", "is_portscan_like_bytes"]]
imp_df["rank"] = range(1, len(imp_df) + 1)
print(imp_df[imp_df["feature"].isin(new_flag_names)].to_string(index=False))
print(f"\n(out of {len(imp_df)} total features)")
