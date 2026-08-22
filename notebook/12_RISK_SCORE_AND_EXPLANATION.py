import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

from inference_engine import load_artifacts, risk_score_from_probability, generate_explanation
from feature_pipeline import engineer_features, add_behavioral_features


# ============================================================
# PATHS
# ============================================================

STAGE_DIR = Path("stage_outputs")

INPUT_PATH = STAGE_DIR / "02_probability.csv"
OUTPUT_PATH = STAGE_DIR / "03_risk_score_and_explanation.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. "
            f"Run 11_PROBABILITY.py first."
        )

    print("\nLoading final model...")

    artifacts = load_artifacts()

    print("Final model loaded successfully.")

    df = pd.read_csv(INPUT_PATH)

    print(f"Records loaded: {len(df)}")

    # --------------------------------------------------------
    # Rule-based explanations need the engineered feature flags
    # (has_sql_pattern, user_agent_type, total_bytes, etc.) --
    # those aren't in the saved CSV, so they're recomputed here
    # from the original raw columns still present in df.
    # --------------------------------------------------------

    engineered = engineer_features(df)

    engineered = add_behavioral_features(
        engineered,
        artifacts["src_port_counts"],
        artifacts["dst_port_counts"]
    )

    risk_scores, risk_bands, explanations = [], [], []

    for i in range(len(df)):
        score, band, color = risk_score_from_probability(df["probability"].iloc[i])
        risk_scores.append(score)
        risk_bands.append(band)
        explanations.append(generate_explanation(engineered.iloc[i]))

    df["risk_score"] = risk_scores
    df["risk_band"] = risk_bands
    df["explanation"] = ["; ".join(reasons) for reasons in explanations]

    df.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("RISK SCORE & RULE-BASED EXPLANATION")
    print("=" * 70)

    for i in range(len(df)):

        print(f"\nRecord {i + 1}")
        print(f"Prediction  : {df['prediction'].iloc[i]}")
        print(f"Probability : {df['probability'].iloc[i]:.2%}")
        print(f"Risk Score  : {risk_scores[i]} / 100")
        print(f"Risk Band   : {risk_bands[i]}")
        print("Reasons     :")
        for reason in explanations[i]:
            print(f"  - {reason}")
    print(
        f"\nSaved Stage 3 output to:\n"
        f"{OUTPUT_PATH.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()