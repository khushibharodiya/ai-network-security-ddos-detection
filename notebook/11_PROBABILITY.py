import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

from inference_engine import load_artifacts
from feature_pipeline import (
    engineer_features,
    add_behavioral_features,
    FEATURE_COLUMNS
)


# ============================================================
# PATHS
# ============================================================

STAGE_DIR = Path("stage_outputs")

INPUT_PATH = STAGE_DIR / "01_ddos_prediction.csv"
OUTPUT_PATH = STAGE_DIR / "02_probability.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check Stage 1 output
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. "
            f"Run 10_DDOS_PREDICTION.py first."
        )

    # --------------------------------------------------------
    # Load final model
    # --------------------------------------------------------

    print("\nLoading final model...")

    artifacts = load_artifacts()

    print("Final model loaded successfully.")

    # --------------------------------------------------------
    # Load Stage 1 prediction
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print(f"Records loaded: {len(df)}")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    engineered = engineer_features(df)

    engineered = add_behavioral_features(
        engineered,
        artifacts["src_port_counts"],
        artifacts["dst_port_counts"]
    )

    # --------------------------------------------------------
    # Select final model features
    # --------------------------------------------------------

    x = engineered[FEATURE_COLUMNS]

    # --------------------------------------------------------
    # Generate probability
    # --------------------------------------------------------

    probability = artifacts["pipeline"].predict_proba(x)[:, 1]

    df["probability"] = probability

    # --------------------------------------------------------
    # Save Stage 2 output
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROBABILITY RESULT")

    for i in range(len(df)):

        prob = float(df["probability"].iloc[i])
        prediction = df["prediction"].iloc[i]

        print(f"\nRecord {i + 1}")
        print(f"Prediction       : {prediction}")
        print(f"Attack Probability: {prob:.2%}")
        print(f"Model Threshold  : {artifacts['threshold']:.2f}")

        if prob >= artifacts["threshold"]:
            print("Status: 🚨 ATTACK")
        else:
            print("Status: ✅ BENIGN")

    print("\n" + "=" * 70)

    print(
        f"\nSaved Stage 2 output to:\n"
        f"{OUTPUT_PATH.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()