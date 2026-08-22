import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt  # ADDED: needed for plotting
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from feature_pipeline import build_dataset, make_preprocessor


data = build_dataset()
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]


# ============================================================
# BUILD EACH PIPELINE WITH ITS BEST HYPERPARAMETERS FROM 03
# ============================================================
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42, C=100)),
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(
            random_state=42, max_depth=15, min_samples_leaf=2, min_samples_split=5,
        )),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            random_state=42, n_jobs=-1, max_depth=10,
            min_samples_leaf=2, min_samples_split=5, n_estimators=100,
        )),
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", random_state=42,
            n_jobs=-1, tree_method="hist", verbosity=0,
            n_estimators=150, max_depth=6, learning_rate=0.03,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, scale_pos_weight=18.0,
        )),
    ]),
}


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


results = []
roc_curves = {}   # ADDED: store fpr/tpr per model for combined ROC plot
pr_curves = {}     # ADDED: store precision/recall per model for combined PR plot
confusions = {} 
for name, pipeline in models.items():
    print("=" * 70)
    print(name)
    print("=" * 70)

    pipeline.fit(x_train, y_train)

    val_prob = pipeline.predict_proba(x_val)[:, 1]
    row = best_threshold(y_val, val_prob)
    threshold = row["threshold"]

    print(f"\nThreshold (tuned on validation, F1-optimal): {threshold:.2f}")
    print(f"Validation -> precision={row['precision']:.3f}, "
          f"recall={row['recall']:.3f}, f1={row['f1']:.3f}")

    test_prob = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_prob)

    print(f"\n----- TEST RESULTS -----")
    print(confusion_matrix(y_test, test_pred))
    print(f"Accuracy : {test_accuracy:.3f}")
    print(f"Precision: {test_precision:.3f}")
    print(f"Recall   : {test_recall:.3f}")
    print(f"F1-Score : {test_f1:.3f}")
    print(f"ROC-AUC  : {test_roc_auc:.3f}")
    print()

    # ============================================================
    # ADDED: Confusion matrix plot for this model
    # ============================================================
    cm = confusion_matrix(y_test, test_pred)
    confusions[name] = cm
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    # ============================================================
    # ADDED: store ROC curve and PR curve data for combined plots
    # ============================================================
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    roc_curves[name] = (fpr, tpr, test_roc_auc)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)
    pr_curves[name] = (prec_curve, rec_curve, pr_auc)

    results.append({
        "model": name,
        "threshold": threshold,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
    })


# ============================================================
# SUMMARY TABLE
# ============================================================
summary = pd.DataFrame(results).sort_values("test_f1", ascending=False)

print("=" * 70)
print("SUMMARY: ALL MODELS ON HELD-OUT TEST SET")
print(summary.to_string(index=False))

best_by_f1 = summary.iloc[0]
best_by_recall = summary.sort_values("test_recall", ascending=False).iloc[0]

print(f"\nBest by test F1     : {best_by_f1['model']} "
      f"(recall={best_by_f1['test_recall']:.3f}, precision={best_by_f1['test_precision']:.3f})")
print(f"Best by test recall : {best_by_recall['model']} "
      f"(recall={best_by_recall['test_recall']:.3f}, precision={best_by_recall['test_precision']:.3f})")



fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, (name, cm) in zip(axes.ravel(), confusions.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("final_test_confusion_matrix.png", dpi=150)
plt.show()
# ============================================================
#Final combined ROC-AUC plot (all models on one figure)
# ============================================================
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc_val) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_auc_final.png", dpi=150)
plt.show()


# ============================================================
#  Final combined PR-AUC plot (all models on one figure)
# ============================================================
plt.figure(figsize=(7, 6))
for name, (prec_curve, rec_curve, pr_auc) in pr_curves.items():
    plt.plot(rec_curve, prec_curve, label=f"{name} (AP = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - All Models (Test Set)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("pr_auc_final.png", dpi=150)
plt.show()