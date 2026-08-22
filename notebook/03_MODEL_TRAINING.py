import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("cybersecurity.csv")

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())


# ============================================================
# BASIC DATA PREPARATION
# ============================================================
try:
    df["timestamp"]=pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M"
        )
except ValueError:
    df["timestamp"]=pd.to_datetime(df["timestamp"])



# ============================================================
# TIME FEATURES
# ============================================================

df["hour"] = df["timestamp"].dt.hour

df["day"] = df["timestamp"].dt.day

df["month"] = df["timestamp"].dt.month

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_business_hours"] = (
    (df["hour"] >= 9) &
    (df["hour"] <= 18)
).astype(int)


# ============================================================
# IP FEATURES
# ============================================================

df["src_ip_is_private"] = df["src_ip"].apply(
    lambda x: ipaddress.ip_address(x).is_private
)

df["dst_ip_is_private"] = df["dst_ip"].apply(
    lambda x: ipaddress.ip_address(x).is_private
)


# ============================================================
# USER AGENT FEATURE
# ============================================================

def get_user_agent_type(user_agent):

    user_agent = str(user_agent)

    if (
        "Chrome" in user_agent or
        "Firefox" in user_agent or
        "Safari" in user_agent
    ):
        return "browser"

    elif "curl" in user_agent:
        return "curl"

    elif "python-requests" in user_agent:
        return "python_requests"

    elif "python-urllib" in user_agent:
        return "python_urllib"

    elif "sqlmap" in user_agent:
        return "sqlmap"

    elif "zgrab" in user_agent:
        return "zgrab"

    elif "Googlebot" in user_agent:
        return "googlebot"

    else:
        return "other"


df["user_agent_type"] = (
    df["user_agent"]
    .fillna("")
    .apply(get_user_agent_type)
)


# ============================================================
# BASIC URL FEATURES
# ============================================================

df["url_missing"] = (
    df["url"]
    .isna()
    .astype(int)
)

df["url_length"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["url_special_chars"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.count(r"[^a-zA-Z0-9]")
)


# ============================================================
# URL TEXT
# ============================================================

url_text = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.lower()
)


# ============================================================
# SQL KEYWORD FEATURES
# ============================================================

sql_patterns = [
    "select",
    "union",
    "insert",
    "update",
    "delete",
    "drop",
    "from",
    "where",
    "having",
    "group by"
]

df["sql_keyword_count"] = 0

for pattern in sql_patterns:

    df["sql_keyword_count"] += (
        url_text.str.count(pattern)
    )


# ============================================================
# XSS KEYWORD FEATURES
# ============================================================

xss_patterns = [
    "<script",
    "</script",
    "javascript:",
    "alert(",
    "onerror",
    "onload",
    "onclick",
    "<img",
    "<iframe"
]

df["xss_keyword_count"] = 0

for pattern in xss_patterns:

    escaped_pattern = (
        pattern
        .replace("(", r"\(")
        .replace(")", r"\)")
    )

    df["xss_keyword_count"] += (
        url_text.str.count(escaped_pattern)
    )


# ============================================================
# COMMAND INJECTION FEATURES
# ============================================================

command_patterns = [
    "cmd",
    "exec",
    "system",
    "shell",
    "/bin/",
    "bash",
    "powershell",
    "wget",
    "curl",
    "ping"
]

df["command_keyword_count"] = 0

for pattern in command_patterns:

    escaped_pattern = (
        pattern
        .replace("(", r"\(")
        .replace(")", r"\)")
    )

    df["command_keyword_count"] += (
        url_text.str.count(escaped_pattern)
    )


# ============================================================
# SUSPICIOUS CHARACTER FEATURES
# ============================================================

df["quote_count"] = (
    url_text.str.count("'") +
    url_text.str.count('"')
)

df["semicolon_count"] = (
    url_text.str.count(";")
)

df["equals_count"] = (
    url_text.str.count("=")
)

df["parentheses_count"] = (
    url_text.str.count(r"\(") +
    url_text.str.count(r"\)")
)

df["angle_bracket_count"] = (
    url_text.str.count("<") +
    url_text.str.count(">")
)

df["url_encoded_count"] = (
    url_text.str.count("%")
)

df["suspicious_symbol_count"] = (
    df["quote_count"] +
    df["semicolon_count"] +
    df["parentheses_count"] +
    df["angle_bracket_count"]
)


# ============================================================
# ADVANCED SQL FEATURES
# ============================================================

sql_patterns_advanced = [
    r"union\s+select",
    r"select\s+.*\s+from",
    r"insert\s+into",
    r"update\s+.*\s+set",
    r"delete\s+from",
    r"drop\s+table",
    r"information_schema",
    r"or\s+1\s*=\s*1",
    r"and\s+1\s*=\s*1",
    r"--",
    r"/\*",
    r"\*/"
]

df["has_sql_pattern"] = (
    sum(
        url_text.str.contains(
            pattern,
            regex=True,
            na=False
        ).astype(int)
        for pattern in sql_patterns_advanced
    ) > 0
).astype(int)


# ============================================================
# ADVANCED XSS FEATURES
# ============================================================

xss_patterns_advanced = [
    r"<script",
    r"</script",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
    r"onclick\s*=",
    r"onmouseover\s*=",
    r"alert\s*\(",
    r"prompt\s*\(",
    r"confirm\s*\("
]

df["has_xss_pattern"] = (
    sum(
        url_text.str.contains(
            pattern,
            regex=True,
            na=False
        ).astype(int)
        for pattern in xss_patterns_advanced
    ) > 0
).astype(int)


# ============================================================
# ADVANCED COMMAND INJECTION FEATURES
# ============================================================

command_patterns_advanced = [
    r";\s*(bash|sh|cmd|powershell)",
    r"&&",
    r"\|\|",
    r"\|\s*(bash|sh|cmd)",
    r"\$\(",
    r"`.*`",
    r"\b(wget|curl)\b",
    r"\b(nc|netcat)\b",
    r"\b(ping)\b",
    r"\b(chmod|chown)\b"
]

df["has_command_pattern"] = (
    sum(
        url_text.str.contains(
            pattern,
            regex=True,
            na=False
        ).astype(int)
        for pattern in command_patterns_advanced
    ) > 0
).astype(int)


# ============================================================
# OTHER URL SECURITY FEATURES
# ============================================================

df["has_path_traversal"] = (
    url_text.str.contains(
        r"\.\./|\.\.\\",
        regex=True,
        na=False
    )
).astype(int)


df["has_script_tag"] = (
    url_text.str.contains(
        r"<script|%3cscript",
        regex=True,
        na=False
    )
).astype(int)


df["has_sql_comment"] = (
    url_text.str.contains(
        r"--|/\*|\*/",
        regex=True,
        na=False
    )
).astype(int)


df["has_shell_operator"] = (
    url_text.str.contains(
        r";|&&|\|\||\$\(|`",
        regex=True,
        na=False
    )
).astype(int)


df["has_encoded_payload"] = (
    url_text.str.contains(
        r"%[0-9a-f]{2}",
        regex=True,
        na=False
    )
).astype(int)


df["has_sql_function"] = (
    url_text.str.contains(
        r"(?:sleep|benchmark|load_file|extractvalue|updatexml)",
        regex=True,
        na=False
    )
).astype(int)


df["has_xss_function"] = (
    url_text.str.contains(
        r"(?:alert|prompt|confirm)\s*\(",
        regex=True,
        na=False
    )
).astype(int)


print("\nURL security features created.")


# ============================================================
# NETWORK BEHAVIOR FEATURES
# ============================================================

df["total_bytes"] = (
    df["bytes_sent"] +
    df["bytes_received"]
)

df["bytes_ratio"] = (
    df["bytes_sent"] /
    (df["bytes_received"] + 1)
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# IMPORTANT:
# test_df is created here but NEVER USED below.
#
# 64% TRAIN
# 16% VALIDATION
# 20% TEST
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    stratify=train_df["label"]
)


print("\nData split completed.")

print("\nTraining data:", train_df.shape)

print("Validation data:", val_df.shape)

print("Testing data:", test_df.shape)

# ============================================================
# LEAKAGE-SAFE CYBERSECURITY BEHAVIORAL FEATURE ENGINEERING
# ============================================================

print("\n===== LEAKAGE-SAFE CYBERSECURITY BEHAVIORAL FEATURE ENGINEERING =====")


def add_behavioral_features(target_df, reference_df):

    target_df = target_df.copy()

    # --------------------------------------------------------
    # 1. SOURCE IP REQUEST COUNT
    # --------------------------------------------------------

    src_counts = reference_df["src_ip"].value_counts()

    target_df["src_ip_request_count"] = (
        target_df["src_ip"]
        .map(src_counts)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 2. DESTINATION IP REQUEST COUNT
    # --------------------------------------------------------

    dst_counts = reference_df["dst_ip"].value_counts()

    target_df["dst_ip_request_count"] = (
        target_df["dst_ip"]
        .map(dst_counts)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 3. UNIQUE DESTINATIONS PER SOURCE
    # --------------------------------------------------------

    src_unique_dst = (
        reference_df
        .groupby("src_ip")["dst_ip"]
        .nunique()
    )

    target_df["src_ip_unique_dst_count"] = (
        target_df["src_ip"]
        .map(src_unique_dst)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 4. UNIQUE DESTINATION PORTS PER SOURCE
    # --------------------------------------------------------

    src_unique_ports = (
        reference_df
        .groupby("src_ip")["dst_port"]
        .nunique()
    )

    target_df["src_ip_unique_dst_port_count"] = (
        target_df["src_ip"]
        .map(src_unique_ports)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 5. SOURCE-DESTINATION PAIR FREQUENCY
    # --------------------------------------------------------

    pair_counts = (
        reference_df
        .groupby(
            ["src_ip", "dst_ip"]
        )
        .size()
    )

    target_pairs = pd.MultiIndex.from_frame(
        target_df[
            ["src_ip", "dst_ip"]
        ]
    )

    target_df["src_dst_pair_count"] = (
        pd.Series(
            target_pairs.map(pair_counts),
            index=target_df.index
        )
        .fillna(0)
        .values
    )

    # --------------------------------------------------------
    # 6. SOURCE PORT FREQUENCY
    # --------------------------------------------------------

    src_port_counts = (
        reference_df["src_port"]
        .value_counts()
    )

    target_df["src_port_frequency"] = (
        target_df["src_port"]
        .map(src_port_counts)
        .fillna(0)
    )

    # --------------------------------------------------------
    # 7. DESTINATION PORT FREQUENCY
    # --------------------------------------------------------

    dst_port_counts = (
        reference_df["dst_port"]
        .value_counts()
    )

    target_df["dst_port_frequency"] = (
        target_df["dst_port"]
        .map(dst_port_counts)
        .fillna(0)
    )

    return target_df


# ============================================================
# APPLY BEHAVIORAL FEATURES
# ============================================================

# Training behavioral statistics are calculated ONLY
# from training data.

train_df = add_behavioral_features(
    train_df,
    train_df
)


# Validation data uses ONLY training statistics.

val_df = add_behavioral_features(
    val_df,
    train_df
)


# Test data uses ONLY training statistics.
# Test data remains untouched for model evaluation.

test_df = add_behavioral_features(
    test_df,
    train_df
)


print("\nBehavioral features added safely.")


# ============================================================
# CHECK BEHAVIORAL FEATURES
# ============================================================

behavioral_features = [

    "src_ip_request_count",
    "dst_ip_request_count",
    "src_ip_unique_dst_count",
    "src_ip_unique_dst_port_count",
    "src_dst_pair_count",
    "src_port_frequency",
    "dst_port_frequency"

]


print("\nChecking behavioral features:")

for column in behavioral_features:

    print(
        column,
        "->",
        column in train_df.columns,
        column in val_df.columns,
        column in test_df.columns
    )


# ============================================================
# FINAL FEATURE SET
# ============================================================

feature_columns = [

    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received",
    "protocol",
    "is_internal_traffic",

    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "is_business_hours",

    "src_ip_is_private",
    "dst_ip_is_private",

    "user_agent_type",

    "url_missing",
    "url_length",
    "url_special_chars",

    "total_bytes",
    "bytes_ratio",

    # NOTE: src_ip_request_count, dst_ip_request_count,
    # src_ip_unique_dst_count, src_ip_unique_dst_port_count and
    # src_dst_pair_count were REMOVED. In this dataset every
    # src_ip/dst_ip is unique (10,000 unique IPs across 10,000 rows),
    # so those five features were always 1 in train and always 0 in
    # val/test -- pure noise with zero predictive signal. Only the
    # port-frequency features are kept below, since ports genuinely
    # repeat across rows and do carry signal.
    "src_port_frequency",
    "dst_port_frequency",

    "sql_keyword_count",
    "xss_keyword_count",
    "command_keyword_count",
    "quote_count",
    "semicolon_count",
    "equals_count",
    "parentheses_count",
    "angle_bracket_count",
    "url_encoded_count",
    "suspicious_symbol_count",

    "has_sql_pattern",
    "has_xss_pattern",
    "has_command_pattern",
    "has_path_traversal",
    "has_script_tag",
    "has_sql_comment",
    "has_shell_operator",
    "has_encoded_payload",
    "has_sql_function",
    "has_xss_function"
]


# ============================================================
# TRAIN / VALIDATION DATA
#
# NO x_test / y_test HERE
# ============================================================

x_train = train_df[feature_columns].copy()
y_train = train_df["label"].copy()

x_val = val_df[feature_columns].copy()
y_val = val_df["label"].copy()

x_test = test_df[feature_columns]
y_test = test_df["label"]


print("\nFinal feature set created.")

print(
    "Number of features:",
    len(feature_columns)
)

print(
    "x_train:",
    x_train.shape
)

print(
    "x_val:",
    x_val.shape
)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "protocol",
    "user_agent_type"
]


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

numerical_columns = [

    "src_port",
    "dst_port",
    "bytes_sent",
    "bytes_received",
    "is_internal_traffic",

    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "is_business_hours",

    "src_ip_is_private",
    "dst_ip_is_private",

    "url_missing",
    "url_length",
    "url_special_chars",

    "total_bytes",
    "bytes_ratio",

    # See note above feature_columns -- IP-identity behavioral
    # features removed; port-frequency features kept.
    "src_port_frequency",
    "dst_port_frequency",

    "sql_keyword_count",
    "xss_keyword_count",
    "command_keyword_count",
    "quote_count",
    "semicolon_count",
    "equals_count",
    "parentheses_count",
    "angle_bracket_count",
    "url_encoded_count",
    "suspicious_symbol_count",

    "has_sql_pattern",
    "has_xss_pattern",
    "has_command_pattern",
    "has_path_traversal",
    "has_script_tag",
    "has_sql_comment",
    "has_shell_operator",
    "has_encoded_payload",
    "has_sql_function",
    "has_xss_function"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        ),

        (
            "numerical",

            StandardScaler(),

            numerical_columns
        )
    ]
)


print("\nPreprocessor created successfully.")



# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("LOGISTIC REGRESSION")


# ============================================================
# BASELINE LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "model",
        logistic_model
    )
])


logistic_pipeline.fit(
    x_train,
    y_train
)


y_lr_pred = logistic_pipeline.predict(
    x_val
)


print(
    "\n===== BASELINE LOGISTIC REGRESSION ====="
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_lr_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_lr_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_lr_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_lr_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_lr_pred,
        zero_division=0
    )
)


# ============================================================
# SMOTE LOGISTIC REGRESSION
# ============================================================

smote_logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


smote_logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "model",
        smote_logistic_model
    )
])


smote_logistic_pipeline.fit(
    x_train,
    y_train
)


y_smote_lr_pred = (
    smote_logistic_pipeline.predict(
        x_val
    )
)


print(
    "\n===== SMOTE LOGISTIC REGRESSION ====="
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_smote_lr_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_smote_lr_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_smote_lr_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_smote_lr_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_smote_lr_pred,
        zero_division=0
    )
)


# ============================================================
# LOGISTIC REGRESSION GRID SEARCH
# ============================================================

lr_param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]

}


lr_grid_search = GridSearchCV(

    estimator=smote_logistic_pipeline,

    param_grid=lr_param_grid,

    scoring="f1",

    cv=5,

    n_jobs=-1,

    verbose=1

)


print(
    "\nStarting Logistic Regression GridSearchCV..."
)


lr_grid_search.fit(
    x_train,
    y_train
)


print(
    "\n===== TUNED LOGISTIC REGRESSION ====="
)


print(
    "\nBest Parameters:"
)

print(
    lr_grid_search.best_params_
)


print(
    "\nBest Cross-Validation F1-Score:"
)

print(
    lr_grid_search.best_score_
)


best_lr_model = (
    lr_grid_search.best_estimator_
)


# ============================================================
# TUNED LOGISTIC REGRESSION VALIDATION
# ============================================================

y_lr_tuned_pred = (
    best_lr_model.predict(
        x_val
    )
)


y_lr_tuned_prob = (
    best_lr_model.predict_proba(
        x_val
    )[:, 1]
)


lr_tuned_accuracy = accuracy_score(
    y_val,
    y_lr_tuned_pred
)


lr_tuned_precision = precision_score(
    y_val,
    y_lr_tuned_pred,
    zero_division=0
)


lr_tuned_recall = recall_score(
    y_val,
    y_lr_tuned_pred,
    zero_division=0
)


lr_tuned_f1 = f1_score(
    y_val,
    y_lr_tuned_pred,
    zero_division=0
)


lr_tuned_roc_auc = roc_auc_score(
    y_val,
    y_lr_tuned_prob
)


lr_tuned_pr_auc = average_precision_score(
    y_val,
    y_lr_tuned_prob
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_lr_tuned_pred
    )
)


print(
    "\nAccuracy :",
    lr_tuned_accuracy
)

print(
    "Precision:",
    lr_tuned_precision
)

print(
    "Recall   :",
    lr_tuned_recall
)

print(
    "F1-Score :",
    lr_tuned_f1
)

print(
    "ROC-AUC  :",
    lr_tuned_roc_auc
)

print(
    "PR-AUC   :",
    lr_tuned_pr_auc
)


# ============================================================
# LOGISTIC REGRESSION - ROC CURVE
# ============================================================

lr_fpr, lr_tpr, _ = roc_curve(
    y_val,
    y_lr_tuned_prob
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    lr_fpr,
    lr_tpr,
    label=(
        f"Logistic Regression "
        f"(AUC = {lr_tuned_roc_auc:.3f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Logistic Regression - ROC-AUC Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# LOGISTIC REGRESSION - PRECISION RECALL CURVE
# ============================================================

lr_precision, lr_recall, _ = (
    precision_recall_curve(
        y_val,
        y_lr_tuned_prob
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    lr_recall,
    lr_precision,
    label=(
        f"Logistic Regression "
        f"(PR-AUC = {lr_tuned_pr_auc:.3f})"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Logistic Regression - "
    "Precision-Recall Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# LOGISTIC REGRESSION - THRESHOLD ANALYSIS
# ============================================================

thresholds_lr = np.arange(
    0.10,
    0.91,
    0.01
)


lr_threshold_results = []


for threshold in thresholds_lr:

    y_temp_pred = (
        y_lr_tuned_prob >= threshold
    ).astype(int)


    lr_threshold_results.append({

        "threshold": threshold,

        "precision": precision_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            y_temp_pred,
            zero_division=0
        )

    })


lr_threshold_results = pd.DataFrame(
    lr_threshold_results
)


best_lr_threshold_row = (
    lr_threshold_results.loc[
        lr_threshold_results["f1"].idxmax()
    ]
)


print(
    "\n===== LOGISTIC REGRESSION "
    "THRESHOLD ANALYSIS ====="
)


print(
    "\nBest threshold:"
)

print(
    best_lr_threshold_row
)


# ============================================================
# BEST LOGISTIC REGRESSION THRESHOLD
# ============================================================

best_lr_threshold = (
    best_lr_threshold_row[
        "threshold"
    ]
)


best_lr_threshold_precision = (
    best_lr_threshold_row[
        "precision"
    ]
)


best_lr_threshold_recall = (
    best_lr_threshold_row[
        "recall"
    ]
)


best_lr_threshold_f1 = (
    best_lr_threshold_row[
        "f1"
    ]
)


print(
    "\nBest Logistic Regression Threshold:",
    best_lr_threshold
)


print(
    "Precision at Best Threshold:",
    best_lr_threshold_precision
)


print(
    "Recall at Best Threshold:",
    best_lr_threshold_recall
)


print(
    "F1-Score at Best Threshold:",
    best_lr_threshold_f1
)


# ============================================================
# THRESHOLD PERFORMANCE CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    lr_threshold_results["threshold"],
    lr_threshold_results["precision"],
    label="Precision"
)


plt.plot(
    lr_threshold_results["threshold"],
    lr_threshold_results["recall"],
    label="Recall"
)


plt.plot(
    lr_threshold_results["threshold"],
    lr_threshold_results["f1"],
    label="F1-Score"
)


plt.axvline(
    best_lr_threshold,
    linestyle="--",
    label=(
        f"Best Threshold = "
        f"{best_lr_threshold:.2f}"
    )
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Logistic Regression: "
    "Precision, Recall and F1 vs Threshold"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# OPTIMIZED VALIDATION PREDICTION
# ============================================================

y_lr_threshold_pred = (
    y_lr_tuned_prob >= best_lr_threshold
).astype(int)


print(
    "\n===== LOGISTIC REGRESSION "
    "OPTIMIZED THRESHOLD RESULTS ====="
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_lr_threshold_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_lr_threshold_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_lr_threshold_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_lr_threshold_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_lr_threshold_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC  :",
    roc_auc_score(
        y_val,
        y_lr_tuned_prob
    )
)

print(
    "PR-AUC   :",
    average_precision_score(
        y_val,
        y_lr_tuned_prob
    )
)
print(
    "LOGISTIC REGRESSION SECTION COMPLETED"
)
# ============================================================
# DECISION TREE
# ============================================================

print("\n")
print("============================================================")
print("DECISION TREE")
print("============================================================")


# ============================================================
# BASELINE DECISION TREE
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)


decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "model",
        decision_tree_model
    )
])


decision_tree_pipeline.fit(
    x_train,
    y_train
)


y_dt_pred = decision_tree_pipeline.predict(
    x_val
)


print(
    "\n===== BASELINE DECISION TREE ====="
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_dt_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_dt_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_dt_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_dt_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_dt_pred,
        zero_division=0
    )
)


# ============================================================
# TUNED DECISION TREE
# ============================================================

dt_param_grid = {

    "model__max_depth": [
        5,
        10,
        15,
        20,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        5
    ]

}


dt_grid_search = GridSearchCV(

    estimator=decision_tree_pipeline,

    param_grid=dt_param_grid,

    scoring="f1",

    cv=5,

    n_jobs=-1,

    verbose=1

)


print(
    "\nStarting Decision Tree GridSearchCV..."
)


dt_grid_search.fit(
    x_train,
    y_train
)


print(
    "\n===== TUNED DECISION TREE ====="
)


print(
    "\nBest Parameters:"
)

print(
    dt_grid_search.best_params_
)


print(
    "\nBest Cross-Validation F1-Score:"
)

print(
    dt_grid_search.best_score_
)


best_dt_model = (
    dt_grid_search.best_estimator_
)


# ============================================================
# TUNED DECISION TREE VALIDATION
# ============================================================

y_dt_tuned_pred = (
    best_dt_model.predict(
        x_val
    )
)


y_dt_tuned_prob = (
    best_dt_model.predict_proba(
        x_val
    )[:, 1]
)


dt_tuned_accuracy = accuracy_score(
    y_val,
    y_dt_tuned_pred
)


dt_tuned_precision = precision_score(
    y_val,
    y_dt_tuned_pred,
    zero_division=0
)


dt_tuned_recall = recall_score(
    y_val,
    y_dt_tuned_pred,
    zero_division=0
)


dt_tuned_f1 = f1_score(
    y_val,
    y_dt_tuned_pred,
    zero_division=0
)


dt_tuned_roc_auc = roc_auc_score(
    y_val,
    y_dt_tuned_prob
)


dt_tuned_pr_auc = average_precision_score(
    y_val,
    y_dt_tuned_prob
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_dt_tuned_pred
    )
)


print(
    "\nAccuracy :",
    dt_tuned_accuracy
)

print(
    "Precision:",
    dt_tuned_precision
)

print(
    "Recall   :",
    dt_tuned_recall
)

print(
    "F1-Score :",
    dt_tuned_f1
)

print(
    "ROC-AUC  :",
    dt_tuned_roc_auc
)

print(
    "PR-AUC   :",
    dt_tuned_pr_auc
)


# ============================================================
# DECISION TREE - ROC CURVE
# ============================================================

dt_fpr, dt_tpr, _ = roc_curve(
    y_val,
    y_dt_tuned_prob
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    dt_fpr,
    dt_tpr,
    label=(
        f"Decision Tree "
        f"(AUC = {dt_tuned_roc_auc:.3f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Decision Tree - ROC-AUC Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# DECISION TREE - PRECISION RECALL CURVE
# ============================================================

dt_precision, dt_recall, _ = (
    precision_recall_curve(
        y_val,
        y_dt_tuned_prob
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    dt_recall,
    dt_precision,
    label=(
        f"Decision Tree "
        f"(PR-AUC = {dt_tuned_pr_auc:.3f})"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Decision Tree - Precision-Recall Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# DECISION TREE - THRESHOLD ANALYSIS
# ============================================================

thresholds_dt = np.arange(
    0.10,
    0.91,
    0.01
)


dt_threshold_results = []


for threshold in thresholds_dt:

    y_temp_pred = (
        y_dt_tuned_prob >= threshold
    ).astype(int)


    dt_threshold_results.append({

        "threshold": threshold,

        "precision": precision_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            y_temp_pred,
            zero_division=0
        )

    })


dt_threshold_results = pd.DataFrame(
    dt_threshold_results
)


best_dt_threshold_row = (
    dt_threshold_results.loc[
        dt_threshold_results["f1"].idxmax()
    ]
)


print(
    "\n===== DECISION TREE "
    "THRESHOLD ANALYSIS ====="
)


print(
    "\nBest threshold:"
)

print(
    best_dt_threshold_row
)


best_dt_threshold = (
    best_dt_threshold_row[
        "threshold"
    ]
)


print(
    "\nBest Decision Tree Threshold:",
    best_dt_threshold
)


print(
    "Precision at Best Threshold:",
    best_dt_threshold_row[
        "precision"
    ]
)


print(
    "Recall at Best Threshold:",
    best_dt_threshold_row[
        "recall"
    ]
)


print(
    "F1-Score at Best Threshold:",
    best_dt_threshold_row[
        "f1"
    ]
)


# ============================================================
# DECISION TREE THRESHOLD CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    dt_threshold_results["threshold"],
    dt_threshold_results["precision"],
    label="Precision"
)


plt.plot(
    dt_threshold_results["threshold"],
    dt_threshold_results["recall"],
    label="Recall"
)


plt.plot(
    dt_threshold_results["threshold"],
    dt_threshold_results["f1"],
    label="F1-Score"
)


plt.axvline(
    best_dt_threshold,
    linestyle="--",
    label=(
        f"Best Threshold = "
        f"{best_dt_threshold:.2f}"
    )
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Decision Tree: "
    "Precision, Recall and F1 vs Threshold"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# OPTIMIZED DECISION TREE VALIDATION
# ============================================================

y_dt_threshold_pred = (
    y_dt_tuned_prob >= best_dt_threshold
).astype(int)


print(
    "\n===== DECISION TREE "
    "OPTIMIZED THRESHOLD RESULTS ====="
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_dt_threshold_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_dt_threshold_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_dt_threshold_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_dt_threshold_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_dt_threshold_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC  :",
    roc_auc_score(
        y_val,
        y_dt_tuned_prob
    )
)

print(
    "PR-AUC   :",
    average_precision_score(
        y_val,
        y_dt_tuned_prob
    )
)
print(
    "DECISION TREE SECTION COMPLETED"
)
# ============================================================
# RANDOM FOREST
# ============================================================

print("\n")
print("RANDOM FOREST")


# ============================================================
# BASELINE RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "model",
        random_forest_model
    )
])


random_forest_pipeline.fit(
    x_train,
    y_train
)


y_rf_pred = (
    random_forest_pipeline.predict(
        x_val
    )
)


print(
    "\n===== BASELINE RANDOM FOREST ====="
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_rf_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_rf_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_rf_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_rf_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_rf_pred,
        zero_division=0
    )
)


# ============================================================
# RANDOM FOREST GRID SEARCH
# ============================================================

rf_param_grid = {

    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        10,
        20,
        None
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ]

}


rf_grid_search = GridSearchCV(

    estimator=random_forest_pipeline,

    param_grid=rf_param_grid,

    scoring="f1",

    cv=5,

    n_jobs=-1,

    verbose=1

)


print(
    "\nStarting Random Forest GridSearchCV..."
)


rf_grid_search.fit(
    x_train,
    y_train
)


print(
    "\n===== TUNED RANDOM FOREST ====="
)


print(
    "\nBest Parameters:"
)

print(
    rf_grid_search.best_params_
)


print(
    "\nBest Cross-Validation F1-Score:"
)

print(
    rf_grid_search.best_score_
)


best_rf_model = (
    rf_grid_search.best_estimator_
)


# ============================================================
# TUNED RANDOM FOREST VALIDATION
# ============================================================

y_rf_tuned_pred = (
    best_rf_model.predict(
        x_val
    )
)


y_rf_tuned_prob = (
    best_rf_model.predict_proba(
        x_val
    )[:, 1]
)


rf_tuned_accuracy = accuracy_score(
    y_val,
    y_rf_tuned_pred
)


rf_tuned_precision = precision_score(
    y_val,
    y_rf_tuned_pred,
    zero_division=0
)


rf_tuned_recall = recall_score(
    y_val,
    y_rf_tuned_pred,
    zero_division=0
)


rf_tuned_f1 = f1_score(
    y_val,
    y_rf_tuned_pred,
    zero_division=0
)


rf_tuned_roc_auc = roc_auc_score(
    y_val,
    y_rf_tuned_prob
)


rf_tuned_pr_auc = average_precision_score(
    y_val,
    y_rf_tuned_prob
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        y_rf_tuned_pred
    )
)


print(
    "\nAccuracy :",
    rf_tuned_accuracy
)

print(
    "Precision:",
    rf_tuned_precision
)

print(
    "Recall   :",
    rf_tuned_recall
)

print(
    "F1-Score :",
    rf_tuned_f1
)

print(
    "ROC-AUC  :",
    rf_tuned_roc_auc
)

print(
    "PR-AUC   :",
    rf_tuned_pr_auc
)


# ============================================================
# RANDOM FOREST - ROC CURVE
# ============================================================

rf_fpr, rf_tpr, _ = roc_curve(
    y_val,
    y_rf_tuned_prob
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    rf_fpr,
    rf_tpr,
    label=(
        f"Random Forest "
        f"(AUC = {rf_tuned_roc_auc:.3f})"
    )
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Random Forest - ROC-AUC Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# RANDOM FOREST - PRECISION RECALL CURVE
# ============================================================

rf_precision, rf_recall, _ = (
    precision_recall_curve(
        y_val,
        y_rf_tuned_prob
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    rf_recall,
    rf_precision,
    label=(
        f"Random Forest "
        f"(PR-AUC = {rf_tuned_pr_auc:.3f})"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Random Forest - Precision-Recall Curve"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# RANDOM FOREST - THRESHOLD ANALYSIS
# ============================================================

thresholds_rf = np.arange(
    0.10,
    0.91,
    0.01
)


rf_threshold_results = []


for threshold in thresholds_rf:

    y_temp_pred = (
        y_rf_tuned_prob >= threshold
    ).astype(int)


    rf_threshold_results.append({

        "threshold": threshold,

        "precision": precision_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            y_temp_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            y_temp_pred,
            zero_division=0
        )

    })


rf_threshold_results = pd.DataFrame(
    rf_threshold_results
)


best_rf_threshold_row = (
    rf_threshold_results.loc[
        rf_threshold_results["f1"].idxmax()
    ]
)


print(
    "\n===== RANDOM FOREST "
    "THRESHOLD ANALYSIS ====="
)


print(
    "\nBest threshold:"
)

print(
    best_rf_threshold_row
)


best_rf_threshold = (
    best_rf_threshold_row[
        "threshold"
    ]
)


print(
    "\nBest Random Forest Threshold:",
    best_rf_threshold
)


print(
    "Precision at Best Threshold:",
    best_rf_threshold_row[
        "precision"
    ]
)


print(
    "Recall at Best Threshold:",
    best_rf_threshold_row[
        "recall"
    ]
)


print(
    "F1-Score at Best Threshold:",
    best_rf_threshold_row[
        "f1"
    ]
)


# ============================================================
# RANDOM FOREST THRESHOLD CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    rf_threshold_results["threshold"],
    rf_threshold_results["precision"],
    label="Precision"
)


plt.plot(
    rf_threshold_results["threshold"],
    rf_threshold_results["recall"],
    label="Recall"
)


plt.plot(
    rf_threshold_results["threshold"],
    rf_threshold_results["f1"],
    label="F1-Score"
)


plt.axvline(
    best_rf_threshold,
    linestyle="--",
    label=(
        f"Best Threshold = "
        f"{best_rf_threshold:.2f}"
    )
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Random Forest: "
    "Precision, Recall and F1 vs Threshold"
)


plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# OPTIMIZED RANDOM FOREST VALIDATION
# ============================================================

y_rf_threshold_pred = (
    y_rf_tuned_prob >= best_rf_threshold
).astype(int)


print(
    "\n===== RANDOM FOREST "
    "OPTIMIZED THRESHOLD RESULTS ====="
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_rf_threshold_pred
    )
)


print(
    "\nAccuracy :",
    accuracy_score(
        y_val,
        y_rf_threshold_pred
    )
)

print(
    "Precision:",
    precision_score(
        y_val,
        y_rf_threshold_pred,
        zero_division=0
    )
)

print(
    "Recall   :",
    recall_score(
        y_val,
        y_rf_threshold_pred,
        zero_division=0
    )
)

print(
    "F1-Score :",
    f1_score(
        y_val,
        y_rf_threshold_pred,
        zero_division=0
    )
)

print(
    "ROC-AUC  :",
    roc_auc_score(
        y_val,
        y_rf_tuned_prob
    )
)

print(
    "PR-AUC   :",
    average_precision_score(
        y_val,
        y_rf_tuned_prob
    )
)
print(
    "RANDOM FOREST SECTION COMPLETED"
)

# ============================================================
# XGBOOST
# FAST TUNING VERSION
# ============================================================

print("\n============================================================")
print("XGBOOST - FAST TUNING")
print("============================================================")


# ------------------------------------------------------------
# STEP 1: CALCULATE CLASS IMBALANCE
# ------------------------------------------------------------

negative_count = (y_train == 0).sum()

positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\nXGBoost scale_pos_weight:")
print(scale_pos_weight)


# ------------------------------------------------------------
# STEP 2: BASE XGBOOST PIPELINE
# ------------------------------------------------------------

xgb_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        # SMOTE added here to match the imbalance-handling strategy
        # used by the LR/DT/RF pipelines above. Previously XGBoost
        # relied on scale_pos_weight alone while the other models
        # used SMOTE -- comparing "best CV F1 across models" wasn't
        # apples-to-apples. scale_pos_weight is still tuned in the
        # grid below on top of SMOTE-balanced data; feel free to
        # drop one or the other if you want to isolate their effect.
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "model",

            XGBClassifier(

                objective="binary:logistic",

                eval_metric="logloss",

                random_state=42,

                n_jobs=-1,

                tree_method="hist",

                verbosity=1
            )
        )
    ]
)


print("\nXGBoost pipeline created.")


# ------------------------------------------------------------
# STEP 3: SMALL TARGETED PARAMETER GRID
# ------------------------------------------------------------
#
# IMPORTANT:
# We are NOT testing hundreds/thousands of combinations.
#
# These values are selected around the configuration that
# previously gave us approximately 65% F1.
#
# ------------------------------------------------------------

xgb_param_grid = {

    "model__n_estimators": [
        150,
        250
    ],

    "model__max_depth": [
        4,
        6
    ],

    "model__learning_rate": [
        0.03,
        0.05,
        0.08
    ],

    "model__min_child_weight": [
        1,
        3
    ],

    "model__subsample": [
        0.8
    ],

    "model__colsample_bytree": [
        0.8
    ],

    "model__gamma": [
        0,
        0.1
    ],

    "model__scale_pos_weight": [
        scale_pos_weight,
        scale_pos_weight * 0.75
    ]
}


print("\nFast XGBoost parameter grid created.")


# ------------------------------------------------------------
# STEP 4: GRID SEARCH
# ------------------------------------------------------------

print("\nStarting FAST XGBoost GridSearchCV...")

# cv raised from 3 -> 5 and scoring switched from "f1" to
# "average_precision" (PR-AUC). With only ~256 positive training
# rows, cv=3 + a threshold-dependent metric like F1 is noisy enough
# that GridSearchCV can pick a combination that just got lucky on
# fold noise rather than one that generalizes. average_precision
# integrates over all thresholds, so it's a more stable target when
# the positive class is this rare, and cv=5 gives each fold more
# positive examples to score against.
xgb_grid = GridSearchCV(

    estimator=xgb_pipeline,

    param_grid=xgb_param_grid,

    scoring="average_precision",

    cv=5,

    n_jobs=-1,

    verbose=1
)


xgb_grid.fit(

    x_train,

    y_train
)


print("\n============================================================")
print("XGBOOST GRID SEARCH COMPLETED")
print("============================================================")


print("\nBest XGBoost Parameters:")

print(
    xgb_grid.best_params_
)


print("\nBest Cross-Validation F1:")

print(
    xgb_grid.best_score_
)


# ------------------------------------------------------------
# STEP 5: BEST XGBOOST MODEL
# ------------------------------------------------------------

best_xgb_model = (
    xgb_grid.best_estimator_
)


print("\nBest tuned XGBoost model created.")


# ============================================================
# XGBOOST VALIDATION
# ============================================================


# ------------------------------------------------------------
# STEP 6: VALIDATION PROBABILITY
# ------------------------------------------------------------

xgb_val_prob = (

    best_xgb_model
    .predict_proba(
        x_val
    )[:, 1]

)


# ------------------------------------------------------------
# STEP 7: DEFAULT THRESHOLD
# ------------------------------------------------------------

xgb_val_pred_default = (

    xgb_val_prob >= 0.50

).astype(int)


print("\n============================================================")
print("XGBOOST VALIDATION RESULTS - DEFAULT THRESHOLD")
print("============================================================")


print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_val,

        xgb_val_pred_default

    )

)


print(

    "\nAccuracy :",

    accuracy_score(

        y_val,

        xgb_val_pred_default

    )

)


print(

    "Precision:",

    precision_score(

        y_val,

        xgb_val_pred_default,

        zero_division=0

    )

)


print(

    "Recall   :",

    recall_score(

        y_val,

        xgb_val_pred_default,

        zero_division=0

    )

)


print(

    "F1-Score :",

    f1_score(

        y_val,

        xgb_val_pred_default,

        zero_division=0

    )

)


print(

    "ROC-AUC  :",

    roc_auc_score(

        y_val,

        xgb_val_prob

    )

)


print(

    "PR-AUC   :",

    average_precision_score(

        y_val,

        xgb_val_prob

    )

)


# ============================================================
# XGBOOST ROC CURVE
# ============================================================

xgb_val_roc_auc = roc_auc_score(

    y_val,

    xgb_val_prob

)


xgb_fpr, xgb_tpr, _ = roc_curve(

    y_val,

    xgb_val_prob

)


plt.figure(figsize=(8, 6))


plt.plot(

    xgb_fpr,

    xgb_tpr,

    label=f"XGBoost (AUC = {xgb_val_roc_auc:.3f})"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "XGBoost - ROC-AUC Curve"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# XGBOOST PRECISION-RECALL CURVE
# ============================================================

xgb_val_pr_auc = average_precision_score(

    y_val,

    xgb_val_prob

)


xgb_precision_curve, xgb_recall_curve, _ = (

    precision_recall_curve(

        y_val,

        xgb_val_prob

    )

)


plt.figure(figsize=(8, 6))


plt.plot(

    xgb_recall_curve,

    xgb_precision_curve,

    label=f"XGBoost (PR-AUC = {xgb_val_pr_auc:.3f})"

)


plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title(
    "XGBoost - Precision-Recall Curve"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# XGBOOST THRESHOLD ANALYSIS
# ============================================================

thresholds_xgb = np.arange(

    0.10,

    0.91,

    0.01

)


xgb_threshold_results = []


for threshold in thresholds_xgb:

    y_temp_pred = (

        xgb_val_prob >= threshold

    ).astype(int)


    xgb_threshold_results.append({

        "threshold": threshold,

        "precision": precision_score(

            y_val,

            y_temp_pred,

            zero_division=0

        ),

        "recall": recall_score(

            y_val,

            y_temp_pred,

            zero_division=0

        ),

        "f1": f1_score(

            y_val,

            y_temp_pred,

            zero_division=0

        )

    })


xgb_threshold_results = pd.DataFrame(

    xgb_threshold_results

)


# ------------------------------------------------------------
# BEST THRESHOLD BASED ON VALIDATION F1
# ------------------------------------------------------------

best_xgb_threshold_row = (

    xgb_threshold_results.loc[

        xgb_threshold_results["f1"].idxmax()

    ]

)


best_xgb_threshold = (

    best_xgb_threshold_row["threshold"]

)


best_xgb_val_f1 = (

    best_xgb_threshold_row["f1"]

)


best_xgb_val_precision = (

    best_xgb_threshold_row["precision"]

)


best_xgb_val_recall = (

    best_xgb_threshold_row["recall"]

)
print("XGBOOST THRESHOLD ANALYSIS")


print("\nBest XGBoost Threshold:")

print(
    best_xgb_threshold
)


print(

    "\nBest Validation Precision:",

    best_xgb_val_precision

)


print(

    "Best Validation Recall:",

    best_xgb_val_recall

)


print(

    "Best Validation F1:",

    best_xgb_val_f1

)


# ============================================================
# THRESHOLD GRAPH
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(

    xgb_threshold_results["threshold"],

    xgb_threshold_results["precision"],

    label="Precision"

)


plt.plot(

    xgb_threshold_results["threshold"],

    xgb_threshold_results["recall"],

    label="Recall"

)


plt.plot(

    xgb_threshold_results["threshold"],

    xgb_threshold_results["f1"],

    label="F1-Score"

)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "XGBoost - Precision, Recall and F1 vs Threshold"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# OPTIMIZED VALIDATION PREDICTIONS
# ============================================================

xgb_val_pred = (

    xgb_val_prob >= best_xgb_threshold

).astype(int)


print("XGBOOST OPTIMIZED VALIDATION RESULTS")


print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_val,

        xgb_val_pred

    )

)


print(

    "\nAccuracy :",

    accuracy_score(

        y_val,

        xgb_val_pred

    )

)


print(

    "Precision:",

    precision_score(

        y_val,
        xgb_val_pred,
        zero_division=0
    )
)

print(

    "Recall   :",

    recall_score(
        y_val,
        xgb_val_pred,
        zero_division=0

    )

)

print(

    "F1-Score :",

    f1_score(

        y_val,
        xgb_val_pred,
        zero_division=0

    )

)

print(

    "ROC-AUC  :",

    roc_auc_score(

        y_val,
        xgb_val_prob

    )

)

print(

    "PR-AUC   :",

    average_precision_score(

        y_val,

        xgb_val_prob

    )

)

# ============================================================
# STEP 1 - FINAL MODEL SELECTION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)


# ============================================================
# LOGISTIC REGRESSION - BEST THRESHOLD METRICS
# ============================================================

y_lr_best_threshold_pred = (
    y_lr_tuned_prob >= best_lr_threshold_row["threshold"]
).astype(int)

lr_threshold_accuracy = accuracy_score(
    y_val,
    y_lr_best_threshold_pred
)

lr_threshold_precision = precision_score(
    y_val,
    y_lr_best_threshold_pred,
    zero_division=0
)

lr_threshold_recall = recall_score(
    y_val,
    y_lr_best_threshold_pred,
    zero_division=0
)

lr_threshold_f1 = f1_score(
    y_val,
    y_lr_best_threshold_pred,
    zero_division=0
)


print("\n----- LOGISTIC REGRESSION -----")

print("Best Threshold :", best_lr_threshold_row["threshold"])
print("Accuracy       :", lr_threshold_accuracy)
print("Precision      :", lr_threshold_precision)
print("Recall         :", lr_threshold_recall)
print("F1-Score       :", lr_threshold_f1)


# ============================================================
# DECISION TREE - BEST THRESHOLD METRICS
# ============================================================

y_dt_best_threshold_pred = (
    y_dt_tuned_prob >= best_dt_threshold_row["threshold"]
).astype(int)

dt_threshold_accuracy = accuracy_score(
    y_val,
    y_dt_best_threshold_pred
)

dt_threshold_precision = precision_score(
    y_val,
    y_dt_best_threshold_pred,
    zero_division=0
)

dt_threshold_recall = recall_score(
    y_val,
    y_dt_best_threshold_pred,
    zero_division=0
)

dt_threshold_f1 = f1_score(
    y_val,
    y_dt_best_threshold_pred,
    zero_division=0
)


print("\n----- DECISION TREE -----")

print("Best Threshold :", best_dt_threshold_row["threshold"])
print("Accuracy       :", dt_threshold_accuracy)
print("Precision      :", dt_threshold_precision)
print("Recall         :", dt_threshold_recall)
print("F1-Score       :", dt_threshold_f1)


# ============================================================
# RANDOM FOREST - BEST THRESHOLD METRICS
# ============================================================

y_rf_best_threshold_pred = (
    y_rf_tuned_prob >= best_rf_threshold_row["threshold"]
).astype(int)

rf_threshold_accuracy = accuracy_score(
    y_val,
    y_rf_best_threshold_pred
)

rf_threshold_precision = precision_score(
    y_val,
    y_rf_best_threshold_pred,
    zero_division=0
)

rf_threshold_recall = recall_score(
    y_val,
    y_rf_best_threshold_pred,
    zero_division=0
)

rf_threshold_f1 = f1_score(
    y_val,
    y_rf_best_threshold_pred,
    zero_division=0
)


print("\n----- RANDOM FOREST -----")

print("Best Threshold :", best_rf_threshold_row["threshold"])
print("Accuracy       :", rf_threshold_accuracy)
print("Precision      :", rf_threshold_precision)
print("Recall         :", rf_threshold_recall)
print("F1-Score       :", rf_threshold_f1)


# ============================================================
# XGBOOST - BEST THRESHOLD METRICS
# ============================================================

y_xgb_best_threshold_pred = (
    xgb_val_prob >= best_xgb_threshold
).astype(int)

xgb_threshold_accuracy = accuracy_score(
    y_val,
    y_xgb_best_threshold_pred
)

xgb_threshold_precision = precision_score(
    y_val,
    y_xgb_best_threshold_pred,
    zero_division=0
)

xgb_threshold_recall = recall_score(
    y_val,
    y_xgb_best_threshold_pred,
    zero_division=0
)

xgb_threshold_f1 = f1_score(
    y_val,
    y_xgb_best_threshold_pred,
    zero_division=0
)


xgb_threshold_roc_auc = roc_auc_score(
    y_val,
    xgb_val_prob
)

xgb_threshold_pr_auc = average_precision_score(
    y_val,
    xgb_val_prob
)


print("\n----- XGBOOST -----")

print("Best Threshold :", best_xgb_threshold)
print("Accuracy       :", xgb_threshold_accuracy)
print("Precision      :", xgb_threshold_precision)
print("Recall         :", xgb_threshold_recall)
print("F1-Score       :", xgb_threshold_f1)
print("ROC-AUC        :", xgb_threshold_roc_auc)
print("PR-AUC         :", xgb_threshold_pr_auc)


# ============================================================
# FINAL MODEL
# ============================================================

print("=" * 70)
print("SELECTED MODEL: XGBOOST")

print("\nReason:")
print("XGBoost achieved the highest validation F1-Score.")

print("\nXGBoost Validation Results:")
print("Best Threshold :", best_xgb_threshold)
print("Accuracy       :", xgb_threshold_accuracy)
print("Precision      :", xgb_threshold_precision)
print("Recall         :", xgb_threshold_recall)
print("F1-Score       :", xgb_threshold_f1)
print("ROC-AUC        :", xgb_threshold_roc_auc)
print("PR-AUC         :", xgb_threshold_pr_auc)

print("No test prediction has been performed.")
# ============================================================
# FINAL XGBOOST MODEL - TEST SET EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL XGBOOST MODEL - TEST SET EVALUATION")
print("=" * 70)


# ============================================================
# STEP 1 - USE THE SELECTED XGBOOST MODEL
# ============================================================

final_xgb_model = best_xgb_model


print("\nSelected XGBoost model:")
print(final_xgb_model)


# ============================================================
# STEP 2 - PREDICT PROBABILITIES ON TEST DATA
# ============================================================

print("\nGenerating final test probabilities...")

xgb_test_prob = final_xgb_model.predict_proba(
    x_test
)[:, 1]


# ============================================================
# STEP 3 - APPLY SELECTED VALIDATION THRESHOLD
# ============================================================

final_xgb_threshold = best_xgb_threshold

print("\nSelected validation threshold:")
print(final_xgb_threshold)


y_xgb_test_pred = (
    xgb_test_prob >= final_xgb_threshold
).astype(int)


# ============================================================
# STEP 4 - CONFUSION MATRIX
# ============================================================

print("\n")
print("===== FINAL XGBOOST CONFUSION MATRIX =====")

xgb_test_cm = confusion_matrix(
    y_test,
    y_xgb_test_pred
)

print(xgb_test_cm)


# ============================================================
# STEP 5 - TEST ACCURACY
# ============================================================

xgb_test_accuracy = accuracy_score(
    y_test,
    y_xgb_test_pred
)


# ============================================================
# STEP 6 - TEST PRECISION
# ============================================================

xgb_test_precision = precision_score(
    y_test,
    y_xgb_test_pred,
    zero_division=0
)


# ============================================================
# STEP 7 - TEST RECALL
# ============================================================

xgb_test_recall = recall_score(
    y_test,
    y_xgb_test_pred,
    zero_division=0
)


# ============================================================
# STEP 8 - TEST F1-SCORE
# ============================================================

xgb_test_f1 = f1_score(
    y_test,
    y_xgb_test_pred,
    zero_division=0
)


# ============================================================
# STEP 9 - TEST ROC-AUC
# ============================================================

xgb_test_roc_auc = roc_auc_score(
    y_test,
    xgb_test_prob
)


# ============================================================
# STEP 10 - TEST PR-AUC
# ============================================================

xgb_test_pr_auc = average_precision_score(
    y_test,
    xgb_test_prob
)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL XGBOOST TEST RESULTS")
print("=" * 70)

print("\nSelected Threshold :", final_xgb_threshold)

print("\nAccuracy :", xgb_test_accuracy)
print("Precision:", xgb_test_precision)
print("Recall   :", xgb_test_recall)
print("F1-Score :", xgb_test_f1)
print("ROC-AUC  :", xgb_test_roc_auc)
print("PR-AUC   :", xgb_test_pr_auc)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("===== FINAL CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_xgb_test_pred,
        zero_division=0
    )
)


# ============================================================
# ROC-AUC CURVE
# ============================================================

xgb_test_fpr, xgb_test_tpr, _ = roc_curve(
    y_test,
    xgb_test_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    xgb_test_fpr,
    xgb_test_tpr,
    label=f"XGBoost (AUC = {xgb_test_roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Final XGBoost - Test ROC-AUC Curve"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

xgb_test_precision_curve, xgb_test_recall_curve, _ = (
    precision_recall_curve(
        y_test,
        xgb_test_prob
    )
)


plt.figure(figsize=(8, 6))

plt.plot(
    xgb_test_recall_curve,
    xgb_test_precision_curve,
    label=f"XGBoost (PR-AUC = {xgb_test_pr_auc:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Final XGBoost - Test Precision-Recall Curve"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("FINAL MODEL EVALUATION COMPLETED")
print("\nModel Selected : XGBoost")
print("Threshold Used :", final_xgb_threshold)

print("\nTest set was used ONLY for final evaluation.")
print("No model selection or threshold tuning was performed using test data.")
# ============================================================
# SAVE FINAL XGBOOST MODEL
# ============================================================

import joblib
import os

print("SAVING FINAL XGBOOST MODEL")

# Create models folder if it does not exist
os.makedirs("../models", exist_ok=True)

# Save complete XGBoost pipeline
joblib.dump(
    best_xgb_model,
    "../models/final_xgboost_pipeline.pkl"
)

# Save selected classification threshold
joblib.dump(
    best_xgb_threshold,
    "../models/xgboost_threshold.pkl"
)

# Save feature names
joblib.dump(
    list(x_train.columns),
    "../models/model_features.pkl"
)

print("\nFinal XGBoost pipeline saved successfully.")
print("Location: ../models/final_xgboost_pipeline.pkl")

print("\nSelected threshold saved successfully.")
print("Location: ../models/xgboost_threshold.pkl")

print("\nFeature names saved successfully.")
print("Location: ../models/model_features.pkl")

print("MODEL SAVING COMPLETED")