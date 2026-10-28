
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]


def check_file(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    errors = []

    if df.empty:
        errors.append("Empty dataset")
    else:
        missing = set(REQUIRED) - set(df.columns)

        if missing:
            errors.append(f"Missing columns: {missing}")
        else:
            if df[REQUIRED].isna().any().any():
                errors.append("Missing values")

            if df["Date"].duplicated().any():
                errors.append("Duplicate dates")

            if not df["Date"].is_monotonic_increasing:
                errors.append("Dates are not sorted")

            if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
                errors.append("Non-positive prices")

            if (df["Volume"] < 0).any():
                errors.append("Negative volume")

            if (df["High"] < df[["Open", "Low", "Close"]].max(axis=1) - 1e-6).any():
                errors.append("Invalid high price")

            if (df["Low"] > df[["Open", "High", "Close"]].min(axis=1) + 1e-6).any():
                errors.append("Invalid low price")

    status = "PASS" if not errors else "FAIL"

    print(f"\n[{status}] {path.name}")
    print(f"  Rows: {len(df):,}")

    if not df.empty and "Date" in df.columns:
        print(f"  Period: {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    for error in errors:
        print(f"  ERROR: {error}")

    return not errors


def main():
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise SystemExit("[ERROR] No CSV files found")

    results = [check_file(path) for path in files]

    print(f"\nSummary: {sum(results)}/{len(results)} passed")

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
