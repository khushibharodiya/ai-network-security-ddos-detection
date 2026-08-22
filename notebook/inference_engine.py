
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)

from notebook.feature_pipeline import (
    build_dataset, make_preprocessor, engineer_features,
    add_behavioral_features, FEATURE_COLUMNS,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts"

# Best hyperparameters found by GridSearchCV in 03_MODEL_TRAINING.py.
BEST_PARAMS = dict(
    n_estimators=150, max_depth=6, learning_rate=0.03, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1, scale_pos_weight=18.0,
)

RISK_BANDS = [
    (0.00, 0.25, "Low", "green"),
    (0.25, 0.50, "Medium", "orange"),
    (0.50, 0.75, "High", "orangered"),
    (0.75, 1.01, "Critical", "red"),
]


# ============================================================
# TRAINING / PERSISTENCE
# ============================================================
def train_and_save_final_model(csv_path="cybersecurity.csv", out_dir=MODEL_DIR):
    """
    Trains the final XGBoost pipeline (same feature set + best
    hyperparameters as 03_MODEL_TRAINING.py), picks a threshold on
    validation, evaluates on test, and saves everything the
    dashboard needs to score new traffic later:
      - the fitted sklearn/imblearn pipeline
      - the chosen decision threshold
      - the train-set port-frequency lookups (needed to compute
        src_port_frequency/dst_port_frequency for brand-new rows)
      - the feature column list
      - a small metrics summary
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    data = build_dataset(csv_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", random_state=42,
            n_jobs=-1, tree_method="hist", verbosity=0, **BEST_PARAMS,
        )),
    ])
    pipeline.fit(x_train, y_train)

    val_prob = pipeline.predict_proba(x_val)[:, 1]
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
    best_row = sweep.loc[sweep["f1"].idxmax()]
    threshold = float(best_row["threshold"])

    val_pred = (val_prob >= threshold).astype(int)
    test_prob = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_precision": float(best_row["precision"]),
        "val_recall": float(best_row["recall"]),
        "val_f1": float(best_row["f1"]),
        "val_roc_auc": float(roc_auc_score(y_val, val_prob)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    }

    joblib.dump(pipeline, out_dir / "final_model_pipeline.joblib")
    joblib.dump(data["src_port_counts"], out_dir / "src_port_counts.joblib")
    joblib.dump(data["dst_port_counts"], out_dir / "dst_port_counts.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print("Final model trained and saved to:", out_dir.resolve())
    print(json.dumps(metrics, indent=2))

    return pipeline, threshold, metrics


# ============================================================
# LOADING SAVED ARTIFACTS
# ============================================================
def load_artifacts(model_dir=MODEL_DIR):
    """Loads everything saved by train_and_save_final_model()."""
    model_dir = Path(model_dir)
    if not (model_dir / "final_model_pipeline.joblib").exists():
        raise FileNotFoundError(
            f"No saved model found in {model_dir.resolve()}. "
            f"Run 09_TRAIN_FINAL_MODEL.py first."
        )

    pipeline = joblib.load(model_dir / "final_model_pipeline.joblib")
    src_port_counts = joblib.load(model_dir / "src_port_counts.joblib")
    dst_port_counts = joblib.load(model_dir / "dst_port_counts.joblib")
    with open(model_dir / "metrics.json") as f:
        metrics = json.load(f)

    return {
        "pipeline": pipeline,
        "threshold": metrics["threshold"],
        "src_port_counts": src_port_counts,
        "dst_port_counts": dst_port_counts,
        "metrics": metrics,
    }


# ============================================================
# RISK SCORING
# ============================================================
def risk_score_from_probability(probability):
    """Maps a model probability (0-1) to a 0-100 risk score and a
    Low/Medium/High/Critical band. The score itself is just the
    probability rescaled to 0-100 -- the bands are what give it
    interpretable meaning for a dashboard/report."""
    score = round(float(probability) * 100, 1)
    for low, high, label, color in RISK_BANDS:
        if low * 100 <= score < high * 100:
            return score, label, color
    return score, "Critical", "red"


# ============================================================
# RULE-BASED EXPLANATION
# ============================================================
def generate_explanation(row):
    """
    Given one engineered feature row (a pandas Series, matching the
    columns produced by feature_pipeline.engineer_features +
    add_behavioral_features), returns a list of human-readable
    reasons the model likely flagged (or didn't flag) this traffic.

    This is intentionally simple/transparent (if/else rules on the
    same engineered features the model sees) rather than a SHAP/LIME
    explanation -- appropriate for a security-report audience who
    wants a plain-English reason, not a feature-attribution plot.
    """
    reasons = []

    if row.get("has_sql_pattern", 0) == 1:
        reasons.append("SQL injection pattern detected in URL (e.g. UNION SELECT, OR 1=1, SQL comments)")
    if row.get("has_xss_pattern", 0) == 1:
        reasons.append("Cross-site scripting (XSS) pattern detected in URL (script tags or event handlers)")
    if row.get("has_command_pattern", 0) == 1:
        reasons.append("Command injection pattern detected in URL (shell operators or system commands)")
    if row.get("has_path_traversal", 0) == 1:
        reasons.append("Path traversal pattern detected in URL (../ sequences)")
    if row.get("has_sql_function", 0) == 1:
        reasons.append("SQL exploitation function detected (e.g. SLEEP, LOAD_FILE, EXTRACTVALUE)")

    user_agent_type = row.get("user_agent_type", "")
    if user_agent_type == "sqlmap":
        reasons.append("User agent identifies as 'sqlmap' -- a known automated SQL injection tool")
    elif user_agent_type == "zgrab":
        reasons.append("User agent identifies as 'zgrab' -- a known network scanning tool")

    total_bytes = row.get("total_bytes", 0)
    if total_bytes >= 1_000_000:
        reasons.append(f"Unusually high data volume ({total_bytes:,.0f} bytes) -- possible DDoS/exfiltration")
    elif 0 < total_bytes <= 200:
        reasons.append(f"Unusually low data volume ({total_bytes:,.0f} bytes) -- possible port/service scan")

    if row.get("suspicious_symbol_count", 0) >= 5:
        reasons.append("High density of suspicious symbols in URL (quotes, semicolons, brackets)")

    if row.get("url_missing", 0) == 1 and row.get("dst_port", 0) not in (80, 443, 8080, 8443):
        reasons.append("No URL present on a non-standard destination port")

    if not reasons:
        reasons.append("No specific rule triggered -- flagged primarily on statistical pattern "
                        "(port/byte-volume profile) rather than a single identifiable signature")

    return reasons


# ============================================================
# END-TO-END PREDICTION
# ============================================================
def predict(raw_df, artifacts):
    """
    Full pipeline for new/raw traffic rows (same columns as the
    original cybersecurity.csv): engineer features -> score with
    the saved model -> risk score -> rule-based explanation.

    Returns a DataFrame with one row per input row, containing:
      probability, risk_score, risk_band, prediction (Attack/Benign),
      explanation (semicolon-joined reasons)
    """
    engineered = engineer_features(raw_df)
    engineered = add_behavioral_features(
        engineered, artifacts["src_port_counts"], artifacts["dst_port_counts"]
    )

    x = engineered[FEATURE_COLUMNS]
    probabilities = artifacts["pipeline"].predict_proba(x)[:, 1]
    threshold = artifacts["threshold"]

    results = raw_df.copy().reset_index(drop=True)
    results["probability"] = probabilities
    results["prediction"] = np.where(probabilities >= threshold, "Attack", "Benign")

    risk_scores, risk_bands, risk_colors = [], [], []
    explanations = []
    for i in range(len(engineered)):
        score, band, color = risk_score_from_probability(probabilities[i])
        risk_scores.append(score)
        risk_bands.append(band)
        risk_colors.append(color)
        explanations.append("; ".join(generate_explanation(engineered.iloc[i])))

    results["risk_score"] = risk_scores
    results["risk_band"] = risk_bands
    results["risk_color"] = risk_colors
    results["explanation"] = explanations

    return results


# ============================================================
# SECURITY REPORT
# ============================================================
def generate_security_report(results_df):
    """Produces a plain-text security report summarizing a batch of
    scored traffic (the output of predict())."""
    total = len(results_df)
    attacks = (results_df["prediction"] == "Attack").sum()
    benign = total - attacks
    attack_rate = (attacks / total * 100) if total else 0

    band_counts = results_df["risk_band"].value_counts().reindex(
        ["Critical", "High", "Medium", "Low"], fill_value=0
    )

    top_reasons = (
        results_df.loc[results_df["prediction"] == "Attack", "explanation"]
        .str.split("; ")
        .explode()
        .value_counts()
        .head(5)
    )

    lines = []
    lines.append("=" * 60)
    lines.append("SECURITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Total records analyzed : {total}")
    lines.append(f"Flagged as Attack       : {attacks} ({attack_rate:.1f}%)")
    lines.append(f"Flagged as Benign       : {benign} ({100 - attack_rate:.1f}%)")
    lines.append("")
    lines.append("Risk band breakdown:")
    for band, count in band_counts.items():
        lines.append(f"  {band:10s}: {count}")
    lines.append("")
    if len(top_reasons) > 0:
        lines.append("Top reasons for Attack flags:")
        for reason, count in top_reasons.items():
            lines.append(f"  ({count:4d}) {reason}")
    else:
        lines.append("No attacks flagged in this batch.")
    lines.append("=" * 60)

    return "\n".join(lines)