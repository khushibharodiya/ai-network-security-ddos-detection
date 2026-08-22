
import pandas as pd
import ipaddress

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


CATEGORICAL_COLUMNS = ["protocol", "user_agent_type"]

# IP-identity behavioral features (src_ip_request_count,
# dst_ip_request_count, src_ip_unique_dst_count,
# src_ip_unique_dst_port_count, src_dst_pair_count) are deliberately
# NOT included here -- every src_ip/dst_ip in this dataset is
# unique, so those columns are always 1 in train / 0 in val+test and
# carry no signal. Port-frequency features are kept since ports
# genuinely repeat and do carry signal.
NUMERICAL_COLUMNS = [
    "src_port", "dst_port", "bytes_sent", "bytes_received",
    "is_internal_traffic", "hour", "day", "month", "day_of_week",
    "is_weekend", "is_business_hours", "src_ip_is_private",
    "dst_ip_is_private", "url_missing", "url_length", "url_special_chars",
    "total_bytes", "bytes_ratio",
    "src_port_frequency", "dst_port_frequency",
    "sql_keyword_count", "xss_keyword_count", "command_keyword_count",
    "quote_count", "semicolon_count", "equals_count", "parentheses_count",
    "angle_bracket_count", "url_encoded_count", "suspicious_symbol_count",
    "has_sql_pattern", "has_xss_pattern", "has_command_pattern",
    "has_path_traversal", "has_script_tag", "has_sql_comment",
    "has_shell_operator", "has_encoded_payload", "has_sql_function",
    "has_xss_function",
]

FEATURE_COLUMNS = ["protocol", "user_agent_type"] + NUMERICAL_COLUMNS
# (protocol/user_agent_type appear first here just for readability;
# order doesn't matter since ColumnTransformer selects by name.)


def _engineer_features(df):
    df = df.copy()

    # Try the original DD-MM-YYYY HH:MM format first (fast path);
    # fall back to flexible parsing if the source CSV uses a
    # different format (e.g. ISO "YYYY-MM-DD HH:MM:SS"). Different
    # exports/regenerations of this dataset have used both.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M")
    except ValueError:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)

    df["src_ip_is_private"] = df["src_ip"].apply(lambda x: ipaddress.ip_address(x).is_private)
    df["dst_ip_is_private"] = df["dst_ip"].apply(lambda x: ipaddress.ip_address(x).is_private)

    def get_user_agent_type(user_agent):
        user_agent = str(user_agent)
        if "Chrome" in user_agent or "Firefox" in user_agent or "Safari" in user_agent:
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

    df["user_agent_type"] = df["user_agent"].fillna("").apply(get_user_agent_type)

    df["url_missing"] = df["url"].isna().astype(int)
    df["url_length"] = df["url"].fillna("").astype(str).str.len()
    df["url_special_chars"] = df["url"].fillna("").astype(str).str.count(r"[^a-zA-Z0-9]")

    url_text = df["url"].fillna("").astype(str).str.lower()

    sql_patterns = ["select", "union", "insert", "update", "delete", "drop",
                     "from", "where", "having", "group by"]
    df["sql_keyword_count"] = 0
    for p in sql_patterns:
        df["sql_keyword_count"] += url_text.str.count(p)

    xss_patterns = ["<script", "</script", "javascript:", "alert(", "onerror",
                     "onload", "onclick", "<img", "<iframe"]
    df["xss_keyword_count"] = 0
    for p in xss_patterns:
        ep = p.replace("(", r"\(").replace(")", r"\)")
        df["xss_keyword_count"] += url_text.str.count(ep)

    command_patterns = ["cmd", "exec", "system", "shell", "/bin/", "bash",
                         "powershell", "wget", "curl", "ping"]
    df["command_keyword_count"] = 0
    for p in command_patterns:
        ep = p.replace("(", r"\(").replace(")", r"\)")
        df["command_keyword_count"] += url_text.str.count(ep)

    df["quote_count"] = url_text.str.count("'") + url_text.str.count('"')
    df["semicolon_count"] = url_text.str.count(";")
    df["equals_count"] = url_text.str.count("=")
    df["parentheses_count"] = url_text.str.count(r"\(") + url_text.str.count(r"\)")
    df["angle_bracket_count"] = url_text.str.count("<") + url_text.str.count(">")
    df["url_encoded_count"] = url_text.str.count("%")
    df["suspicious_symbol_count"] = (
        df["quote_count"] + df["semicolon_count"]
        + df["parentheses_count"] + df["angle_bracket_count"]
    )

    sql_adv = [r"union\s+select", r"select\s+.*\s+from", r"insert\s+into",
               r"update\s+.*\s+set", r"delete\s+from", r"drop\s+table",
               r"information_schema", r"or\s+1\s*=\s*1", r"and\s+1\s*=\s*1",
               r"--", r"/\*", r"\*/"]
    df["has_sql_pattern"] = (
        sum(url_text.str.contains(p, regex=True, na=False).astype(int) for p in sql_adv) > 0
    ).astype(int)

    xss_adv = [r"<script", r"</script", r"javascript:", r"onerror\s*=",
               r"onload\s*=", r"onclick\s*=", r"onmouseover\s*=",
               r"alert\s*\(", r"prompt\s*\(", r"confirm\s*\("]
    df["has_xss_pattern"] = (
        sum(url_text.str.contains(p, regex=True, na=False).astype(int) for p in xss_adv) > 0
    ).astype(int)

    cmd_adv = [r";\s*(bash|sh|cmd|powershell)", r"&&", r"\|\|",
               r"\|\s*(bash|sh|cmd)", r"\$\(", r"`.*`", r"\b(wget|curl)\b",
               r"\b(nc|netcat)\b", r"\b(ping)\b", r"\b(chmod|chown)\b"]
    df["has_command_pattern"] = (
        sum(url_text.str.contains(p, regex=True, na=False).astype(int) for p in cmd_adv) > 0
    ).astype(int)

    df["has_path_traversal"] = url_text.str.contains(r"\.\./|\.\.\\", regex=True, na=False).astype(int)
    df["has_script_tag"] = url_text.str.contains(r"<script|%3cscript", regex=True, na=False).astype(int)
    df["has_sql_comment"] = url_text.str.contains(r"--|/\*|\*/", regex=True, na=False).astype(int)
    df["has_shell_operator"] = url_text.str.contains(r";|&&|\|\||\$\(|`", regex=True, na=False).astype(int)
    df["has_encoded_payload"] = url_text.str.contains(r"%[0-9a-f]{2}", regex=True, na=False).astype(int)
    df["has_sql_function"] = url_text.str.contains(
        r"(?:sleep|benchmark|load_file|extractvalue|updatexml)", regex=True, na=False
    ).astype(int)
    df["has_xss_function"] = url_text.str.contains(
        r"(?:alert|prompt|confirm)\s*\(", regex=True, na=False
    ).astype(int)

    df["total_bytes"] = df["bytes_sent"] + df["bytes_received"]
    df["bytes_ratio"] = df["bytes_sent"] / (df["bytes_received"] + 1)

    return df


def _add_behavioral_features(target_df, reference_df):
    """Port-frequency features only -- fit on reference_df (train),
    applied to target_df. Leakage-safe: val/test only ever see
    stats computed from train."""
    target_df = target_df.copy()

    src_port_counts = reference_df["src_port"].value_counts()
    target_df["src_port_frequency"] = target_df["src_port"].map(src_port_counts).fillna(0)

    dst_port_counts = reference_df["dst_port"].value_counts()
    target_df["dst_port_frequency"] = target_df["dst_port"].map(dst_port_counts).fillna(0)

    return target_df


def engineer_features(df):
    """
    Public wrapper around the feature engineering used by
    build_dataset(). Exposed separately so 02_PREPROCESSING.py can
    run the SAME engineering step-by-step with diagnostic prints
    (crosstabs, value_counts, etc.) between steps, without
    duplicating the feature logic itself -- guaranteeing 02 always
    matches whatever 03_MODEL_TRAINING.py actually uses.
    """
    return _engineer_features(df)


def build_dataset(csv_path="cybersecurity.csv", random_state=42):
    """
    Returns a dict with:
      x_train, x_val, x_test, y_train, y_val, y_test
    using the same 64/16/20 stratified split as 03_MODEL_TRAINING.py.
    """
    df = pd.read_csv(csv_path)
    df = _engineer_features(df)

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.20, random_state=random_state, stratify=train_df["label"]
    )

    train_df = _add_behavioral_features(train_df, train_df)
    val_df = _add_behavioral_features(val_df, train_df)
    test_df = _add_behavioral_features(test_df, train_df)

    x_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["label"].copy()
    x_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df["label"].copy()
    x_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["label"].copy()

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        # attack_type is kept alongside for optional per-attack-type
        # breakdowns; not used as a model feature.
        "attack_type_test": test_df["attack_type"].copy(),
        # Reference port-frequency lookups computed from train_df only
        # (leakage-safe). Exposed here so a deployed model can be
        # scored consistently on brand-new incoming traffic later --
        # without these, there'd be no way to compute
        # src_port_frequency/dst_port_frequency for new rows.
        "src_port_counts": train_df["src_port"].value_counts(),
        "dst_port_counts": train_df["dst_port"].value_counts(),
    }


def add_behavioral_features(target_df, src_port_counts, dst_port_counts):
    """
    Public version of _add_behavioral_features that takes already-
    computed reference lookups (e.g. loaded from a saved model
    bundle) instead of a reference dataframe -- used at inference
    time to score new traffic without needing the original training
    data on hand.
    """
    target_df = target_df.copy()
    target_df["src_port_frequency"] = target_df["src_port"].map(src_port_counts).fillna(0)
    target_df["dst_port_frequency"] = target_df["dst_port"].map(dst_port_counts).fillna(0)
    return target_df


def make_preprocessor():
    """Fresh (unfitted) ColumnTransformer -- build a new one per
    pipeline rather than sharing a fitted instance across models."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numerical", StandardScaler(), NUMERICAL_COLUMNS),
        ]
    )