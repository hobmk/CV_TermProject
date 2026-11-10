
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.render_chart import render_window


START = "2015-01-01"
END = "2022-01-01"  # Exclusive: 2022-2025 not used
FIELDS = ["Open", "High", "Low", "Close", "Volume"]

SECTORS = {
    "Technology": (
        "MSFT AAPL NVDA AMD INTC QCOM AVGO TXN ADI "
        "MCHP MU AMAT LRCX KLAC ANET CSCO IBM ORCL "
        "ADBE CRM SNPS CDNS ADSK INTU ACN NOW FTNT "
        "PANW AKAM EPAM FICO"
    ),
    "Financials": (
        "JPM BAC WFC C GS MS BLK SCHW AXP V MA COF "
        "USB PNC BK STT MMC AJG TRV CB AON SPGI "
        "MCO CME ICE NDAQ CBOE AFL PRU MET ALL HIG"
    ),
    "Healthcare": (
        "JNJ UNH MRK PFE ABBV LLY AMGN GILD BMY "
        "CVS CI HUM MDT SYK BSX ISRG REGN VRTX "
        "ZTS BIIB TMO DHR ABT BDX HCA IDXX EW "
        "HOLX CAH MCK"
    ),
    "Industrials": (
        "CAT DE HON GE BA LMT NOC GD ETN EMR PH "
        "ITW MMM UPS FDX UNP CSX NSC WM RSG "
        "CTAS ROK CMI FAST JCI GWW PCAR PWR "
        "URI DOV IR"
    ),
    "Consumer": (
        "AMZN TSLA WMT COST TGT HD LOW MCD SBUX "
        "NKE PEP KO PG CL KMB GIS K KR MDLZ "
        "HSY DG DLTR TJX ROST BBY YUM MAR HLT "
        "DHI LEN PHM NVR BURL LULU ULTA EBAY"
    ),
    "Energy": (
        "XOM CVX COP EOG SLB HAL OXY VLO MPC "
        "PSX DVN APA FANG EQT KMI NOV BKR WMB OKE"
    ),
    "Utilities": (
        "NEE DUK SO AEP EXC XEL SRE PCG ED D "
        "DTE WEC PEG CNP ETR FE ATO CMS LNT"
    ),
    "RealEstate": (
        "AMT PLD EQIX SPG O PSA VTR WELL EQR "
        "AVB ESS MAA DLR IRM ARE BXP HST UDR REG FRT"
    ),
    "Materials": (
        "SHW ECL APD NUE FCX NEM ALB VMC MLM "
        "CF MOS IP PKG PPG EMN LYB AVY IFF FMC"
    ),
    "Communication": (
        "GOOGL META NFLX TMUS VZ T CHTR EA "
        "TTWO DIS CMCSA"
    ),
}


def read_stock(path):
    df = pd.read_csv(
        path,
        dtype={"Date": str},
    )

    required = {
        "Date",
        *FIELDS,
        "Adj Close",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: "
            f"{required - set(df.columns)}"
        )

    df = df[
        (df["Date"] >= START)
        & (df["Date"] < END)
    ].copy()

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    if (
        df.empty
        or df["Date"].isna().any()
        or df["Date"].duplicated().any()
    ):
        raise ValueError(
            "Invalid or duplicate dates"
        )

    values = df[
        FIELDS + ["Adj Close"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("NaN/Inf values")

    prices = [
        "Open", "High", "Low",
        "Close", "Adj Close",
    ]

    if (df[prices] <= 0).any().any():
        raise ValueError(
            "Nonpositive prices"
        )

    if (df["Volume"] < 0).any():
        raise ValueError(
            "Negative volume"
        )

    if (
        df["High"] + 1e-5
        < df[
            ["Open", "Close", "Low"]
        ].max(axis=1)
    ).any():
        raise ValueError(
            "Invalid High"
        )

    if (
        df["Low"] - 1e-5
        > df[
            ["Open", "Close", "High"]
        ].min(axis=1)
    ).any():
        raise ValueError(
            "Invalid Low"
        )

    return df


def check_dates(df, reference):
    dates = set(df["Date"])

    missing = len(
        reference - dates
    )

    extra = len(
        dates - reference
    )

    if df["Date"].iloc[0] > "2015-01-06":
        raise ValueError(
            "Not listed in early 2015"
        )

    if df["Date"].iloc[-1] < "2021-12-28":
        raise ValueError(
            "Incomplete 2021 coverage"
        )

    if missing > 3 or extra > 3:
        raise ValueError(
            f"Market date mismatch: "
            f"missing={missing}, extra={extra}"
        )

    return missing


def candidate_stream():
    lists = [
        text.split()
        for text in SECTORS.values()
    ]

    for group in itertools.zip_longest(
        *lists
    ):
        for sector, ticker in zip(
            SECTORS, group
        ):
            if ticker:
                yield sector, ticker


def yahoo_history(ticker):
    import yfinance as yf

    history = yf.Ticker(ticker).history(
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if history.empty:
        raise ValueError(
            "Yahoo returned no data"
        )

    df = history.reset_index()

    df = df.rename(
        columns={
            df.columns[0]: "Date"
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    ).dt.strftime("%Y-%m-%d")

    return df


def make_samples(raw_dir, tickers):
    records = []

    for ticker in tickers:
        df = read_stock(
            raw_dir / f"{ticker}.csv"
        )

        dates = df[
            "Date"
        ].to_numpy()

        close = df[
            "Close"
        ].to_numpy(dtype=float)

        for i in range(
            19,
            len(df) - 5,
        ):
            future_return = float(
                close[i + 5]
                / close[i]
                - 1.0
            )

            records.append((
                ticker,
                "train",
                dates[i - 19],
                dates[i],
                dates[i + 5],
                future_return,
                int(future_return > 0),
            ))

    result = pd.DataFrame(
        records,
        columns=[
            "ticker",
            "split",
            "window_start",
            "decision_date",
            "target_date",
            "return_5d",
            "label",
        ],
    )

    result = (
        result.sort_values(
            ["ticker", "decision_date"]
        )
        .reset_index(drop=True)
    )

    if result.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise RuntimeError(
            "Duplicate metadata samples"
        )

    return result


def protocol_masks(samples):
    train = (
        (
            samples["decision_date"]
            < "2021-01-01"
        )
        &
        (
            samples["target_date"]
            < "2021-01-01"
        )
    )

    dev = (
        (
            samples["window_start"]
            >= "2021-01-01"
        )
        &
        (
            samples["target_date"]
            < "2022-01-01"
        )
    )

    return train, dev


def prepare(args):
    old = args.existing_dir

    raw = args.root / "raw"

    processed = (
        args.root / "processed"
    )

    manifest_file = (
        processed / "universe100.csv"
    )

    if manifest_file.exists():
        print(
            "[EXISTS] Universe already built:",
            manifest_file,
        )
        print(
            "[SAFE] Refusing to silently "
            "reselect tickers."
        )
        return

    current = sorted(
        path.stem.upper()
        for path in old.glob("*.csv")
    )

    if len(current) != 30:
        raise RuntimeError(
            f"Expected 30 existing stocks, "
            f"found {len(current)} in {old}. "
            "Original data was not modified."
        )

    if "AAPL" not in current:
        raise RuntimeError(
            "AAPL must be in the "
            "existing 30 stocks"
        )

    reference = set(
        read_stock(
            old / "AAPL.csv"
        )["Date"]
    )

    raw.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.mkdir(
        parents=True,
        exist_ok=True,
    )

    chosen = []
    used = set()

    # Preserve the original 30 stocks.
    for ticker in current:
        df = read_stock(
            old / f"{ticker}.csv"
        )

        missing = check_dates(
            df, reference
        )

        df.to_csv(
            raw / f"{ticker}.csv",
            index=False,
        )

        chosen.append((
            ticker,
            "Existing30",
            "original",
            len(df),
            missing,
        ))

        used.add(ticker)

        print(
            f"[EXISTING] {ticker:7s} "
            f"rows={len(df)} "
            f"missing_days={missing}",
            flush=True,
        )

    failed = []

    # Select 70 additional stocks.
    for sector, ticker in candidate_stream():
        if len(chosen) >= 100:
            break

        if ticker in used:
            continue

        used.add(ticker)

        path = (
            raw / f"{ticker}.csv"
        )

        try:
            if path.exists():
                df = read_stock(path)

            else:
                downloaded = (
                    yahoo_history(ticker)
                )

                tmp = (
                    raw
                    / f"{ticker}.download.csv"
                )

                downloaded.to_csv(
                    tmp,
                    index=False,
                )

                try:
                    df = read_stock(tmp)

                    check_dates(
                        df, reference
                    )

                    df.to_csv(
                        path,
                        index=False,
                    )

                finally:
                    tmp.unlink(
                        missing_ok=True
                    )

                time.sleep(args.delay)

            missing = check_dates(
                df, reference
            )

            chosen.append((
                ticker,
                sector,
                "downloaded",
                len(df),
                missing,
            ))

            print(
                f"[ACCEPT {len(chosen):3d}/100] "
                f"{ticker:7s} "
                f"{sector:16s} "
                f"missing_days={missing}",
                flush=True,
            )

        except Exception as exc:
            failed.append((
                ticker,
                str(exc)[:160],
            ))

            print(
                f"[SKIP] {ticker}: {exc}",
                flush=True,
            )

    if failed:
        pd.DataFrame(
            failed,
            columns=[
                "ticker", "reason"
            ],
        ).to_csv(
            processed / "skipped_candidates.csv",
            index=False,
        )

    if len(chosen) != 100:
        raise RuntimeError(
            f"Only {len(chosen)} valid stocks; "
            "need 100. Review "
            "skipped_candidates.csv. "
            "Downloaded CSVs are cached."
        )

    manifest = pd.DataFrame(
        chosen,
        columns=[
            "ticker",
            "sector_bucket",
            "source",
            "raw_rows",
            "missing_ref_days",
        ],
    )

    samples = make_samples(
        raw,
        manifest["ticker"].tolist(),
    )

    train_mask, dev_mask = (
        protocol_masks(samples)
    )

    train = samples[train_mask]
    dev = samples[dev_mask]

    if train.empty or dev.empty:
        raise RuntimeError(
            "Empty Train or Dev"
        )

    reports = []

    for name, frame in (
        ("train", train),
        ("dev", dev),
    ):
        by_day = frame.groupby(
            "decision_date"
        )["ticker"].nunique()

        print(
            f"[SPLIT {name}] "
            f"rows={len(frame):,} "
            f"days={len(by_day)} "
            f"up={frame['label'].mean():.3f} "
            f"per_day_min={by_day.min()} "
            f"per_day_median="
            f"{by_day.median():.0f}",
            flush=True,
        )

        reports.append(
            pd.DataFrame({
                "decision_date": (
                    by_day.index
                ),
                "split_year": name,
                "stock_count": (
                    by_day.values
                ),
            })
        )

    samples.to_csv(
        processed / "samples.csv",
        index=False,
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )

    pd.concat(
        reports,
        ignore_index=True,
    ).to_csv(
        processed / "daily_groups.csv",
        index=False,
    )

    print(
        "[READY] 100 stocks:",
        manifest_file,
    )

    print(
        "[READY] Metadata:",
        processed / "samples.csv",
    )

    print(
        "[READY] Ranking groups:",
        processed / "daily_groups.csv",
    )

    print(
        "[NOTE] No 2022-2025 "
        "data used."
    )


def render(args):
    processed = (
        args.root / "processed"
    )

    manifest = pd.read_csv(
        processed / "universe100.csv"
    )

    if len(manifest) != 100:
        raise ValueError(
            "Manifest must contain "
            "exactly 100 stocks"
        )

    samples = pd.read_csv(
        processed / "samples.csv",
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    train, dev = protocol_masks(
        samples
    )

    selected = samples[
        train | dev
    ].copy()

    if args.limit:
        selected = selected.head(
            args.limit
        )

    created = 0
    skipped = 0

    for ticker, group in selected.groupby(
        "ticker",
        sort=True,
    ):
        df = read_stock(
            args.root
            / "raw"
            / f"{ticker}.csv"
        )

        date_pos = {
            date: i
            for i, date in enumerate(
                df["Date"]
            )
        }

        for item in group.itertuples(
            index=False
        ):
            output = (
                args.root
                / "charts"
                / "train"
                / ticker
                / f"{item.decision_date}.png"
            )

            if output.exists():
                skipped += 1
                continue

            i = date_pos[
                item.decision_date
            ]

            window = df.iloc[
                i - 19:i + 1
            ]

            if (
                len(window) != 20
                or window["Date"].iloc[0]
                != item.window_start
            ):
                raise RuntimeError(
                    f"Window mismatch: "
                    f"{ticker} "
                    f"{item.decision_date}"
                )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            render_window(
                window
            ).save(
                output,
                optimize=True,
            )

            created += 1

        print(
            f"[CHART] {ticker:7s} "
            f"created_so_far={created:,}",
            flush=True,
        )

    print(
        f"[DONE] created={created:,} "
        f"skipped_existing={skipped:,}"
    )

    print(
        "[OUTPUT]",
        args.root / "charts" / "train",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=["prepare", "render"],
        default="prepare",
    )

    parser.add_argument(
        "--existing-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/exp100"),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render limit; 0 means all",
    )

    args = parser.parse_args()

    if args.delay < 0 or args.limit < 0:
        parser.error(
            "delay and limit must "
            "be nonnegative"
        )

    if args.stage == "prepare":
        prepare(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
