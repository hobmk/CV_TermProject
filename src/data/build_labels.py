
import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {"Date", "Open", "High", "Low", "Close", "Volume"}

SPLIT_BOUNDS = {
    "train": ("1900-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}


def load_stock(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = REQUIRED - set(df.columns)

    if missing:
        raise ValueError(f"{path.stem}: Missing {missing}")

    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty or df["Date"].isna().any():
        raise ValueError(f"{path.stem}: Invalid dates")

    if df["Date"].duplicated().any():
        raise ValueError(f"{path.stem}: Duplicate dates")

    if df[list(REQUIRED - {"Date"})].isna().any().any():
        raise ValueError(f"{path.stem}: Missing OHLCV")

    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError(f"{path.stem}: Invalid prices")

    return df


def build_samples(raw_dir: Path, window: int, horizon: int):
    records = []
    purged = 0
    files = sorted(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files: {raw_dir}")

    for path in files:
        ticker = path.stem.upper()
        df = load_stock(path)

        dates = df["Date"].tolist()
        closes = df["Close"].to_numpy(dtype=float)
        count_before = len(records)

        for i in range(window - 1, len(df) - horizon):
            start = dates[i - window + 1]
            decision = dates[i]
            target = dates[i + horizon]

            for split, (left, right) in SPLIT_BOUNDS.items():
                left = pd.Timestamp(left)
                right = pd.Timestamp(right)

                if not left <= decision <= right:
                    continue

                # Entire input and target must remain in one split.
                if start < left or target > right:
                    purged += 1
                    break

                future_return = closes[i + horizon] / closes[i] - 1.0

                records.append({
                    "ticker": ticker,
                    "split": split,
                    "window_start": start.strftime("%Y-%m-%d"),
                    "decision_date": decision.strftime("%Y-%m-%d"),
                    "target_date": target.strftime("%Y-%m-%d"),
                    f"return_{horizon}d": future_return,
                    "label": int(future_return > 0),
                })
                break

        count = len(records) - count_before
        print(f"[SUCCESS] {ticker}: {count:,} samples")

    if not records:
        raise RuntimeError("No samples generated")

    result = pd.DataFrame(records)
    result = result.sort_values(
        ["split", "ticker", "decision_date"]
    ).reset_index(drop=True)

    if result.duplicated(["ticker", "decision_date"]).any():
        raise AssertionError("Duplicate samples detected")

    for split, (left, right) in SPLIT_BOUNDS.items():
        part = result[result["split"] == split]

        if part.empty:
            raise RuntimeError(f"Empty split: {split}")

        assert (part["window_start"] >= left).all()
        assert (part["target_date"] <= right).all()
        assert (part["target_date"] > part["decision_date"]).all()

    print(f"\n[PURGED] {purged:,} boundary-overlapping samples")
    print("[CHECK] No cross-split input/target overlap")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/samples.csv"))
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    if args.window < 2 or args.horizon < 1:
        parser.error("window >= 2 and horizon >= 1 required")

    samples = build_samples(args.raw_dir, args.window, args.horizon)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output, index=False)

    print("\n[SUMMARY]")
    for split in ["train", "val", "test"]:
        part = samples[samples["split"] == split]
        print(
            f"{split:5s} | N={len(part):,} "
            f"| Positive={part['label'].mean():.2%}"
        )

    print(f"\n[SAVED] {args.output}")
    print(f"[TOTAL] {len(samples):,} samples")


if __name__ == "__main__":
    main()
