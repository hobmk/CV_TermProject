
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
JUMP_THRESHOLD = 0.40

results = []
jumps = []

files = sorted(RAW_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DIR}"
    )

for path in files:
    ticker = path.stem.upper()
    df = pd.read_csv(path)

    required = {
        "Date", "Open", "High",
        "Low", "Close", "Volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{ticker}: Missing columns {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(
        drop=True
    )

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    invalid_ohlc = (
        (h < o)
        | (h < c)
        | (l > o)
        | (l > c)
        | (h < l)
    )

    returns = c.pct_change(
        fill_method=None
    )

    extreme = (
        returns.abs() >= JUMP_THRESHOLD
    )

    results.append({
        "ticker": ticker,
        "rows": len(df),
        "duplicate_dates": int(
            df["Date"].duplicated().sum()
        ),
        "invalid_ohlc": int(
            invalid_ohlc.sum()
        ),
        "missing_values": int(
            df[list(required - {"Date"})]
            .isna().sum().sum()
        ),
        "has_adj_close": (
            "Adj Close" in df.columns
        ),
        "extreme_daily_moves": int(
            extreme.sum()
        ),
    })

    for idx in np.flatnonzero(
        extreme.to_numpy()
    ):
        jumps.append({
            "ticker": ticker,
            "date": df.at[
                idx, "Date"
            ].strftime("%Y-%m-%d"),
            "previous_close": (
                float(c.iloc[idx - 1])
            ),
            "close": float(
                c.iloc[idx]
            ),
            "daily_return": float(
                returns.iloc[idx]
            ),
        })


summary = pd.DataFrame(results)

print("\n[DATA QUALITY SUMMARY]")

print(
    summary.to_string(
        index=False
    )
)

print("\n[EXTREME DAILY MOVES]")

if jumps:
    jump_df = pd.DataFrame(jumps)

    jump_df = jump_df.sort_values(
        "daily_return"
    )

    print(
        jump_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )
else:
    print(
        "No daily close-to-close moves "
        "of 40% or more found."
    )

print("\n[OVERALL]")

print(
    f"Tickers: {len(files)}"
)

print(
    f"Duplicate dates: "
    f"{summary['duplicate_dates'].sum()}"
)

print(
    f"Invalid OHLC rows: "
    f"{summary['invalid_ohlc'].sum()}"
)

print(
    f"Missing OHLCV values: "
    f"{summary['missing_values'].sum()}"
)

print(
    f"Extreme daily moves: "
    f"{summary['extreme_daily_moves'].sum()}"
)

print(
    "\n[NOTE] Extreme moves are flags "
    "for inspection, not proof of "
    "incorrect adjustment."
)

print(
    "[NOTE] No raw files were modified."
)
