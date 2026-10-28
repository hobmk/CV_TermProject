
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DIR = Path("data/raw")
REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def download_stock(ticker: str, start: str, end: str) -> None:
    print(f"\n[DOWNLOAD] {ticker}: {start} ~ {end}")

    df = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
    )

    if df.empty:
        raise RuntimeError(f"{ticker}: No data returned")

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: Missing columns {missing}")

    if df[REQUIRED].isna().any().any():
        raise ValueError(f"{ticker}: Missing OHLCV values")

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = (
        pd.to_datetime(df["Date"])
        .dt.strftime("%Y-%m-%d")
    )

    columns = [
        "Date", "Open", "High", "Low", "Close",
        "Volume", "Adj Close", "Dividends",
        "Stock Splits",
    ]
    columns = [col for col in columns if col in df.columns]
    df = df[columns].sort_values("Date")

    if df["Date"].duplicated().any():
        raise ValueError(f"{ticker}: Duplicate dates found")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DIR / f"{ticker}.csv"
    df.to_csv(output, index=False)

    print(f"[SUCCESS] {ticker}")
    print(f"  Samples : {len(df):,}")
    print(f"  First   : {df['Date'].iloc[0]}")
    print(f"  Last    : {df['Date'].iloc[-1]}")
    print(f"  Saved   : {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    args = parser.parse_args()

    failed = []

    for ticker in args.tickers:
        ticker = ticker.upper()
        try:
            download_stock(ticker, args.start, args.end)
        except Exception as exc:
            failed.append(ticker)
            print(f"[FAILED] {ticker}: {exc}")

    if failed:
        raise SystemExit(f"Download failed: {failed}")

    print("\n[COMPLETE] All downloads successful")


if __name__ == "__main__":
    main()
