
"""Audit frozen 200-stock universe coverage for 2022-2025.

Never modifies existing 2010-2021 data or trained models.
Caches new Yahoo histories in a separate directory.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [
    "Date", "Open", "High", "Low",
    "Close", "Adj Close", "Volume",
]

YEARS = (2022, 2023, 2024, 2025)


def clean(df):
    if not set(COLS).issubset(df.columns):
        raise ValueError("Missing price columns")

    df = df[COLS].copy().reset_index(drop=True)

    df["Date"] = (
        pd.to_datetime(df["Date"], errors="raise")
        .dt.strftime("%Y-%m-%d")
    )

    for col in COLS[1:]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=COLS[1:])
    )

    prices = [
        "Open", "High", "Low",
        "Close", "Adj Close",
    ]

    df = df[
        (df[prices] > 0).all(axis=1)
        & (df.Volume >= 0)
    ]

    return (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def fetch(ticker, path, retries, delay):
    if path.exists():
        return clean(pd.read_csv(path)), "cache"

    import yfinance as yf

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            history = yf.Ticker(ticker).history(
                start="2021-11-01",
                end="2026-01-01",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )

            if (
                history is None
                or history.empty
                or "Adj Close" not in history
            ):
                raise ValueError(
                    "Empty history or missing Adj Close"
                )

            dates = pd.DatetimeIndex(history.index)

            if dates.tz is not None:
                dates = dates.tz_localize(None)

            frame = history[COLS[1:]].copy()

            frame.insert(
                0,
                "Date",
                dates.strftime("%Y-%m-%d"),
            )

            frame = clean(frame)

            if frame.empty:
                raise ValueError("No valid price rows")

            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            frame.to_csv(
                path,
                index=False,
                float_format="%.12g",
            )

            if delay:
                time.sleep(delay)

            return frame, "download"

        except Exception as exc:
            last_error = str(exc)

            time.sleep(
                max(delay, attempt * 1.5)
            )

    raise RuntimeError(last_error)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
        ),
    )

    parser.add_argument(
        "--old-raw",
        type=Path,
        default=Path("data/exp200/raw"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/exp200_holdout_audit"),
    )

    parser.add_argument(
        "--min-days",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()

    if (
        args.retries < 1
        or args.min_days < 1
        or args.delay < 0
    ):
        parser.error("Invalid arguments")

    manifest = pd.read_csv(
        args.manifest,
        dtype={
            "ticker": str,
            "group": str,
        },
    )

    if (
        len(manifest) != 200
        or manifest.ticker.nunique() != 200
    ):
        raise ValueError(
            "Expected frozen 200-stock universe"
        )

    if (manifest.group == "legacy").sum() != 100:
        raise ValueError(
            "Expected 100 legacy stocks"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failed = []

    for n, item in enumerate(
        manifest.itertuples(index=False), 1
    ):
        ticker = item.ticker
        group = item.group

        future_path = (
            args.out
            / "future_raw"
            / f"{ticker}.csv"
        )

        try:
            old = clean(
                pd.read_csv(
                    args.old_raw / f"{ticker}.csv"
                )
            )

            if (
                old.empty
                or old.Date.iloc[-1] < "2021-12-01"
            ):
                raise ValueError(
                    "Original 2021 history missing"
                )

            future, source = fetch(
                ticker,
                future_path,
                args.retries,
                args.delay,
            )

            # Check overlapping adjusted-close prices.
            old_overlap = old.loc[
                old.Date < "2022-01-01",
                ["Date", "Adj Close"],
            ]

            new_overlap = future.loc[
                future.Date < "2022-01-01",
                ["Date", "Adj Close"],
            ]

            overlap = old_overlap.merge(
                new_overlap,
                on="Date",
                suffixes=("_old", "_new"),
            )

            if len(overlap) < 5:
                raise ValueError(
                    f"Insufficient overlap: {len(overlap)}"
                )

            ratios = (
                overlap["Adj Close_old"].to_numpy()
                / overlap["Adj Close_new"].to_numpy()
            )

            scale = float(np.median(ratios))

            deviation = float(
                np.max(
                    np.abs(ratios / scale - 1)
                )
            )

            if deviation > 0.005:
                raise ValueError(
                    "Historical adjustment mismatch: "
                    f"{deviation:.4%}"
                )

            for year in YEARS:
                year_data = future.loc[
                    (future.Date >= f"{year}-01-01")
                    & (
                        future.Date
                        < f"{year + 1}-01-01"
                    )
                ]

                rows.append({
                    "ticker": ticker,
                    "group": group,
                    "year": year,
                    "days": len(year_data),
                    "first": (
                        year_data.Date.min()
                        if len(year_data) else ""
                    ),
                    "last": (
                        year_data.Date.max()
                        if len(year_data) else ""
                    ),
                    "coverage_ok": (
                        len(year_data) >= args.min_days
                    ),
                    "overlap_scale": scale,
                    "source": source,
                })

        except Exception as exc:
            failed.append({
                "ticker": ticker,
                "group": group,
                "error": str(exc),
            })

            print(
                f"[CHECK FAILED] {ticker}: {exc}",
                flush=True,
            )

            for year in YEARS:
                rows.append({
                    "ticker": ticker,
                    "group": group,
                    "year": year,
                    "days": 0,
                    "first": "",
                    "last": "",
                    "coverage_ok": False,
                    "overlap_scale": np.nan,
                    "source": "failed",
                })

        if n % 10 == 0 or n == 200:
            print(
                f"[AUDIT] {n}/200 complete "
                f"| issues={len(failed)}",
                flush=True,
            )

    table = pd.DataFrame(rows)

    table.to_csv(
        args.out / "coverage_by_ticker_year.csv",
        index=False,
    )

    pd.DataFrame(
        failed,
        columns=["ticker", "group", "error"],
    ).to_csv(
        args.out / "download_or_overlap_issues.csv",
        index=False,
    )

    print("\n[ANNUAL COVERAGE]")

    for year in YEARS:
        part = table[table.year == year]

        all_ok = int(part.coverage_ok.sum())

        legacy_ok = int(
            part.loc[
                part.group == "legacy",
                "coverage_ok",
            ].sum()
        )

        missing = part.loc[
            ~part.coverage_ok,
            "ticker",
        ].tolist()

        print(
            f"{year}: "
            f"200-stock {all_ok}/200 | "
            f"legacy100 {legacy_ok}/100"
        )

        print(
            "  insufficient:",
            ", ".join(missing) if missing else "none",
        )

    coverage = table.pivot(
        index="ticker",
        columns="year",
        values="coverage_ok",
    )

    print(
        "\n[ALL 4 YEARS]",
        int(coverage.all(axis=1).sum()),
        "/200",
    )

    print(f"[SAVED] {args.out}")

    print(
        "[NEXT] Review delistings and coverage "
        "before generating test images"
    )


if __name__ == "__main__":
    main()
