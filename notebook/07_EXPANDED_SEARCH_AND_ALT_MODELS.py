import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from feature_pipeline import build_dataset, make_preprocessor


data = build_dataset()
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos


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


def evaluate(name, pipeline, threshold):
    val_prob = pipeline.predict_proba(x_val)[:, 1]
    test_prob = pipeline.predict_proba(x_test)[:, 1]

    test_pred = (test_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    auc = roc_auc_score(y_test, test_prob)

    print(f"\n----- {name} (threshold={threshold:.2f}) -----")
    print(confusion_matrix(y_test, test_pred))
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")

    return {"model": name, "threshold": threshold, "test_accuracy": acc,
            "test_precision": prec, "test_recall": rec, "test_f1": f1,
            "test_roc_auc": auc}


results = []


# ============================================================
# 1. WIDER RANDOMIZEDSEARCHCV OVER XGBOOST
# ============================================================
print("=" * 70)
print("XGBOOST - EXPANDED RANDOMIZED SEARCH")
print("=" * 70)

xgb_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_jobs=-1, tree_method="hist", verbosity=0,
    )),
])

xgb_param_distributions = {
    "model__n_estimators": [100, 150, 200, 300, 400],
    "model__max_depth": [3, 4, 5, 6, 8, 10],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    "model__min_child_weight": [1, 2, 3, 5, 7],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.05, 0.1, 0.2, 0.3],
    "model__reg_alpha": [0, 0.01, 0.1, 1],
    "model__reg_lambda": [0.5, 1, 1.5, 2],
    "model__scale_pos_weight": [scale_pos_weight * m for m in (0.5, 0.75, 1.0, 1.25)],
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_distributions,
    n_iter=80,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
xgb_search.fit(x_train, y_train)

print("\nBest params:", xgb_search.best_params_)
print("Best CV PR-AUC:", xgb_search.best_score_)

xgb_best = xgb_search.best_estimator_
xgb_val_prob = xgb_best.predict_proba(x_val)[:, 1]
xgb_row = best_threshold(y_val, xgb_val_prob)
results.append(evaluate("XGBoost (expanded search)", xgb_best, xgb_row["threshold"]))


# ============================================================
# 2. LIGHTGBM
# ============================================================
print("\n\n" + "=" * 70)
print("LIGHTGBM")
print("=" * 70)

lgbm_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("model", LGBMClassifier(
        random_state=42, n_jobs=-1, verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )),
])

lgbm_param_distributions = {
    "model__n_estimators": [100, 150, 200, 300, 400],
    "model__max_depth": [3, 4, 5, 6, 8, -1],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "model__num_leaves": [15, 31, 63, 127],
    "model__min_child_samples": [5, 10, 20, 30],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1],
    "model__reg_lambda": [0.5, 1, 1.5, 2],
}

lgbm_search = RandomizedSearchCV(
    estimator=lgbm_pipeline,
    param_distributions=lgbm_param_distributions,
    n_iter=60,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
lgbm_search.fit(x_train, y_train)

print("\nBest params:", lgbm_search.best_params_)
print("Best CV PR-AUC:", lgbm_search.best_score_)

lgbm_best = lgbm_search.best_estimator_
lgbm_val_prob = lgbm_best.predict_proba(x_val)[:, 1]
lgbm_row = best_threshold(y_val, lgbm_val_prob)
results.append(evaluate("LightGBM", lgbm_best, lgbm_row["threshold"]))


# ============================================================
# 3. CATBOOST
# ============================================================
print("\n\n" + "=" * 70)
print("CATBOOST")
print("=" * 70)

cat_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("model", CatBoostClassifier(
        random_state=42, verbose=0, scale_pos_weight=scale_pos_weight,
    )),
])

cat_param_distributions = {
    "model__iterations": [200, 300, 500],
    "model__depth": [4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "model__l2_leaf_reg": [1, 3, 5, 7, 9],
}

cat_search = RandomizedSearchCV(
    estimator=cat_pipeline,
    param_distributions=cat_param_distributions,
    n_iter=20,
    scoring="average_precision",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
cat_search.fit(x_train, y_train)

print("\nBest params:", cat_search.best_params_)
print("Best CV PR-AUC:", cat_search.best_score_)

cat_best = cat_search.best_estimator_
cat_val_prob = cat_best.predict_proba(x_val)[:, 1]
cat_row = best_threshold(y_val, cat_val_prob)
results.append(evaluate("CatBoost", cat_best, cat_row["threshold"]))


# ============================================================
# 4. STACKED ENSEMBLE (soft-vote: average of XGBoost + RF probabilities)
# ============================================================
print("\n\n" + "=" * 70)
print("STACKED ENSEMBLE (XGBoost + Random Forest, averaged probabilities)")
print("=" * 70)

rf_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        random_state=42, n_jobs=-1, max_depth=10,
        min_samples_leaf=2, min_samples_split=5, n_estimators=100,
    )),
])
rf_pipeline.fit(x_train, y_train)

# Use the tuned XGBoost from step 1 as the other half of the ensemble.
rf_val_prob = rf_pipeline.predict_proba(x_val)[:, 1]
rf_test_prob = rf_pipeline.predict_proba(x_test)[:, 1]

ensemble_val_prob = (xgb_val_prob + rf_val_prob) / 2
ensemble_row = best_threshold(y_val, ensemble_val_prob)
ensemble_threshold = ensemble_row["threshold"]

ensemble_test_prob = (xgb_best.predict_proba(x_test)[:, 1] + rf_test_prob) / 2
ensemble_pred = (ensemble_test_prob >= ensemble_threshold).astype(int)

acc = accuracy_score(y_test, ensemble_pred)
prec = precision_score(y_test, ensemble_pred, zero_division=0)
rec = recall_score(y_test, ensemble_pred, zero_division=0)
f1 = f1_score(y_test, ensemble_pred, zero_division=0)
auc = roc_auc_score(y_test, ensemble_test_prob)

print(f"\n----- Stacked Ensemble (threshold={ensemble_threshold:.2f}) -----")
print(confusion_matrix(y_test, ensemble_pred))
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-Score : {f1:.3f}")
print(f"ROC-AUC  : {auc:.3f}")

results.append({"model": "Stacked Ensemble (XGB+RF avg)", "threshold": ensemble_threshold,
                 "test_accuracy": acc, "test_precision": prec, "test_recall": rec,
                 "test_f1": f1, "test_roc_auc": auc})


# ============================================================
# SUMMARY
# ============================================================
summary = pd.DataFrame(results).sort_values("test_f1", ascending=False)

print("\n\n" + "=" * 70)
print("SUMMARY: EXPANDED SEARCH / ALTERNATE MODELS ON HELD-OUT TEST SET")
print("=" * 70)
print(summary.to_string(index=False))

print("\nCompare against the 03_MODEL_TRAINING.py / 06_ALL_MODELS_TEST_EVAL.py "
      "baseline (XGBoost, small grid): test F1 was ~0.52-0.54.")
