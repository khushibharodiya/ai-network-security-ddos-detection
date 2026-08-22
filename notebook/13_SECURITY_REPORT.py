from pathlib import Path

import pandas as pd

from inference_engine import generate_security_report


STAGE_DIR = Path("stage_outputs")

INPUT_PATH = STAGE_DIR / "03_risk_score_and_explanation.csv"
OUTPUT_PATH = STAGE_DIR / "04_security_report.txt"


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. "
            f"Run 12_RISK_SCORE_AND_EXPLANATION.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nRecords loaded: {len(df)}")

    report = generate_security_report(df)

    print("\n" + report)

    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()