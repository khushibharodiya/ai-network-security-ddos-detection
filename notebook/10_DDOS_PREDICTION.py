import warnings
warnings.filterwarnings("ignore")

import ipaddress
from pathlib import Path
from datetime import datetime

import pandas as pd

from inference_engine import load_artifacts
from feature_pipeline import (
    engineer_features,
    add_behavioral_features,
    FEATURE_COLUMNS
)


# ============================================================
# SETTINGS
# ============================================================

STAGE_DIR = Path("stage_outputs")
OUTPUT_PATH = STAGE_DIR / "01_ddos_prediction.csv"


# ============================================================
# INPUT HELPERS
# ============================================================

def prompt_ip(label, default):
    """Prompts for an IP address, validating it, with a default if
    left blank. Loops until a valid address is entered."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        raw = raw or default
        try:
            ipaddress.ip_address(raw)
            return raw
        except ValueError:
            print(f"  '{raw}' is not a valid IP address -- try again.")


# ============================================================
# USER INPUT
# ============================================================

def get_user_input():

    print("\n" + "=" * 70)
    print("        AI-POWERED NETWORK SECURITY - DDoS DETECTION")
    print("=" * 70)

    print("\nEnter the network traffic information.")
    print("Fields in [brackets] show the default if you press Enter.")
    print("Press Ctrl+C to exit.\n")

    protocol = input("Protocol (TCP/UDP/ICMP): ").strip().upper()

    src_port = int(input("Source Port: "))
    dst_port = int(input("Destination Port: "))

    bytes_sent = float(input("Bytes Sent: "))
    bytes_received = float(input("Bytes Received: "))

    internal_input = input(
        "Is Internal Traffic? (yes/no): "
    ).strip().lower()

    is_internal_traffic = (
        1 if internal_input in ["yes", "y", "1", "true"]
        else 0
    )

    # Real IP addresses -- these directly affect
    # src_ip_is_private/dst_ip_is_private, which the model uses.
    src_ip = prompt_ip("Source IP", default="203.0.113.10")
    dst_ip = prompt_ip("Destination IP", default="192.168.1.10")

    user_agent = input(
        "User-Agent string (optional, press Enter to skip): "
    ).strip()

    url = input(
        "Request URL (optional, press Enter if none): "
    ).strip()

    # --------------------------------------------------------
    # Build the raw row using the same basic columns expected
    # by the project's feature pipeline.
    # --------------------------------------------------------

    row = {
        "src_port": src_port,
        "dst_port": dst_port,
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "is_internal_traffic": is_internal_traffic,
        "protocol": protocol,

        # Timestamp defaults to "now" -- if left blank/empty,
        # hour/day/month/day_of_week silently come out as NaN
        # (see the fix note at the top of this file).
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "src_ip": src_ip,
        "dst_ip": dst_ip,

        # None (not "") so url_missing correctly reads as 1 when
        # left blank, matching how missing URLs are represented in
        # the training data (NaN, not empty string).
        "user_agent": user_agent if user_agent else None,
        "url": url if url else None,
    }

    return pd.DataFrame([row])


# ============================================================
# PREDICTION
# ============================================================

def predict_user_traffic(raw_df, artifacts):

    # --------------------------------------------------------
    # STEP 1: Feature engineering
    # --------------------------------------------------------

    engineered = engineer_features(raw_df)

    engineered = add_behavioral_features(
        engineered,
        artifacts["src_port_counts"],
        artifacts["dst_port_counts"]
    )

    # --------------------------------------------------------
    # STEP 2: Select final model features
    # --------------------------------------------------------

    x = engineered[FEATURE_COLUMNS]

    # --------------------------------------------------------
    # STEP 3: Generate probability internally
    # --------------------------------------------------------

    probability = artifacts["pipeline"].predict_proba(x)[:, 1][0]

    # --------------------------------------------------------
    # STEP 4: Apply selected threshold
    # --------------------------------------------------------

    threshold = artifacts["threshold"]

    if probability >= threshold:
        prediction = "Attack"
    else:
        prediction = "Benign"

    return prediction, probability


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(raw_df, prediction):

    STAGE_DIR.mkdir(exist_ok=True)

    result = raw_df.copy()

    result["prediction"] = prediction

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading final model...")

    artifacts = load_artifacts()

    print("Final model loaded successfully.")

    # --------------------------------------------------------
    # USER ENTERS TRAFFIC
    # --------------------------------------------------------

    raw_df = get_user_input()

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction, probability = predict_user_traffic(
        raw_df,
        artifacts
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("PREDICTION RESULT")
    print("=" * 70)

    print(f"\nPrediction : {prediction}")

    # Probability is displayed here only for immediate
    # user feedback. It is NOT saved in Stage 1.
    print(f"Model probability : {probability:.2%}")

    if prediction == "Attack":
        print("\n WARNING: MALICIOUS / ATTACK TRAFFIC DETECTED")
    else:
        print("\nTraffic appears to be BENIGN")

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE STAGE 1 OUTPUT
    # --------------------------------------------------------

    save_result(
        raw_df,
        prediction
    )

    print(
        f"\nStage 1 result saved to:\n"
        f"{OUTPUT_PATH.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()