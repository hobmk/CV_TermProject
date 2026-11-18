
"""Prepare 2022-2025 I20/R5 images separately.

Run:
    python -u -m scripts.prepare_exp200_holdout

Preserves all historical 2010-2021 data.
Supports safe resume after interruption.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from scripts.jkx_i20_reproduce import render_i20


COLUMNS = [
    "Date", "Open", "High", "Low",
    "Close", "Adj Close", "Volume",
]

START = "2022-01-01"
END = "2026-01-01"
WINDOW = 20
HORIZON = 5


def file_hash(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1 << 20), b""
        ):
            h.update(block)

    return h.hexdigest()


def read_future_prices(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing audited future prices: {path}"
        )

    data = pd.read_csv(
        path, dtype={"Date": str}
    )

    if not set(COLUMNS).issubset(data.columns):
        raise ValueError(
            f"Missing price columns: {path}"
        )

    data = (
        data[COLUMNS]
        .copy()
        .reset_index(drop=True)
    )

    data["Date"] = pd.to_datetime(
        data["Date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    for c in COLUMNS[1:]:
        data[c] = pd.to_numeric(
            data[c], errors="raise"
        )

    if (
        data[COLUMNS[1:]].isna().any().any()
        or not np.isfinite(
            data[COLUMNS[1:]].to_numpy(float)
        ).all()
    ):
        raise ValueError(
            f"Nonfinite price data: {path}"
        )

    price_cols = [
        "Open", "High", "Low",
        "Close", "Adj Close",
    ]

    if (
        (data[price_cols] <= 0).any().any()
        or (data["Volume"] < 0).any()
    ):
        raise ValueError(
            f"Invalid price or volume: {path}"
        )

    if data.Date.duplicated().any():
        raise ValueError(
            f"Duplicate dates: {path}"
        )

    data = (
        data.sort_values("Date")
        .reset_index(drop=True)
    )

    if data.Date.iloc[0] > "2021-11-15":
        raise ValueError(
            f"Insufficient pre-2022 lookback: {path}"
        )

    for year in range(2022, 2026):
        n = int(
            data.Date.between(
                f"{year}-01-01",
                f"{year}-12-31",
            ).sum()
        )

        if n < 200:
            raise ValueError(
                f"{path}: only {n} days in {year}"
            )

    return data


def adjusted(data):
    out = data.copy()

    factor = (
        out["Adj Close"].to_numpy(float)
        / out["Close"].to_numpy(float)
    )

    for c in ("Open", "High", "Low", "Close"):
        out["A_" + c] = (
            out[c].to_numpy(float) * factor
        )

    out["MA20"] = (
        out["A_Close"]
        .rolling(20, min_periods=20)
        .mean()
    )

    return out


def load_manifest(path):
    manifest = pd.read_csv(
        path,
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
            "Expected 200 unique stocks"
        )

    if (manifest.group == "legacy").sum() != 100:
        raise ValueError(
            "Expected 100 legacy stocks"
        )

    if "sector" not in manifest.columns:
        manifest["sector"] = "unknown"

    return (
        manifest.sort_values("ticker")
        .reset_index(drop=True)
    )


def build_metadata(manifest, raw):
    rows = []

    for j, item in enumerate(
        manifest.itertuples(index=False), 1
    ):
        d = adjusted(
            read_future_prices(
                raw / f"{item.ticker}.csv"
            )
        )

        dates = d.Date.to_numpy(str)
        closes = d.A_Close.to_numpy(float)

        for i in range(
            WINDOW + 19 - 1,
            len(d) - HORIZON,
        ):
            decision = dates[i]

            if not (START <= decision < END):
                continue

            target = dates[i + HORIZON]

            if target >= END:
                continue

            ret = float(
                closes[i + HORIZON]
                / closes[i]
                - 1.0
            )

            rows.append((
                item.ticker,
                item.group,
                item.sector,
                dates[i - 19],
                decision,
                target,
                ret,
                int(ret > 0),
                i,
            ))

        if j % 25 == 0:
            print(
                f"[META] {j}/200 tickers",
                flush=True,
            )

    meta = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "group",
            "sector",
            "window_start",
            "decision_date",
            "target_date",
            "return_5d",
            "label",
            "raw_i",
        ],
    )

    meta = (
        meta.sort_values([
            "ticker", "decision_date"
        ])
        .reset_index(drop=True)
    )

    if meta.empty or len(meta) < 150000:
        raise ValueError(
            f"Unexpected sample count: {len(meta)}"
        )

    if meta.duplicated([
        "ticker", "decision_date"
    ]).any():
        raise ValueError(
            "Duplicate ticker/date"
        )

    if not np.isfinite(
        meta.return_5d.to_numpy(float)
    ).all():
        raise ValueError(
            "Invalid future returns"
        )

    if not np.array_equal(
        meta.label.to_numpy(int),
        (
            meta.return_5d.to_numpy(float) > 0
        ).astype(int),
    ):
        raise ValueError(
            "Label/return mismatch"
        )

    if (
        meta.decision_date.min() < START
        or meta.target_date.max() >= END
    ):
        raise ValueError(
            "Metadata outside 2022-2025"
        )

    for year in range(2022, 2026):
        year_test = meta.loc[
            (
                meta.window_start
                >= f"{year}-01-01"
            )
            & (
                meta.target_date
                < f"{year+1}-01-01"
            )
        ]

        if (
            year_test.ticker.nunique() != 200
            or len(year_test) < 180 * 200
        ):
            raise ValueError(
                f"Incomplete test year {year}: "
                f"N={len(year_test)}"
            )

        print(
            f"[YEAR] {year} "
            f"test-ready={len(year_test):,} "
            f"tickers={year_test.ticker.nunique()}",
            flush=True,
        )

    return meta


def validate_existing_meta(path, new):
    old = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "group": str,
            "sector": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    cols = [
        "ticker",
        "group",
        "sector",
        "window_start",
        "decision_date",
        "target_date",
        "label",
        "raw_i",
    ]

    if (
        len(old) != len(new)
        or not old[cols].equals(new[cols])
    ):
        raise ValueError(
            "Existing metadata differs. "
            "Refusing to overwrite."
        )

    if not np.allclose(
        old.return_5d.to_numpy(float),
        new.return_5d.to_numpy(float),
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError(
            "Existing targets changed. "
            "Refusing to overwrite."
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
        ),
    )

    p.add_argument(
        "--raw",
        type=Path,
        default=Path(
            "data/exp200_holdout_audit/future_raw"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/exp200_holdout_jkx"
        ),
    )

    args = p.parse_args()

    manifest = load_manifest(args.manifest)

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    meta_file = args.out / "meta.csv"
    image_file = args.out / "images.npy"
    journal_file = args.out / "progress.json"
    ready_file = args.out / "READY.json"

    meta = build_metadata(
        manifest, args.raw
    )

    if meta_file.exists():
        validate_existing_meta(
            meta_file, meta
        )
    else:
        meta.to_csv(
            meta_file,
            index=False,
            float_format="%.12g",
        )

    fingerprint = file_hash(meta_file)

    shape = (
        len(meta), 1, 64, 60
    )

    if ready_file.exists():
        ready = json.loads(
            ready_file.read_text(
                encoding="utf-8"
            )
        )

        if ready.get("fingerprint") != fingerprint:
            raise ValueError(
                "READY fingerprint mismatch"
            )

        if (
            not image_file.exists()
            or np.load(
                image_file, mmap_mode="r"
            ).shape != shape
        ):
            raise ValueError(
                "READY images missing or invalid"
            )

        print(
            f"[DATA READY] {len(meta):,} "
            f"charts in {args.out}",
            flush=True,
        )
        return

    if image_file.exists():
        images = open_memmap(
            image_file, mode="r+"
        )

        if (
            images.shape != shape
            or images.dtype != np.uint8
        ):
            raise ValueError(
                "Existing images mismatch"
            )

    else:
        if journal_file.exists():
            raise ValueError(
                "Journal exists without image file"
            )

        images = open_memmap(
            image_file,
            mode="w+",
            dtype=np.uint8,
            shape=shape,
        )

    if journal_file.exists():
        progress = json.loads(
            journal_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            progress.get("fingerprint")
            != fingerprint
        ):
            raise ValueError(
                "Journal fingerprint mismatch"
            )

        finished = set(
            progress.get("done", [])
        )

    else:
        finished = set()

    print(
        f"[RENDER] {len(meta):,} charts "
        f"| done={len(finished)}/200 "
        f"| 2022-2025",
        flush=True,
    )

    for ticker, rows in meta.groupby(
        "ticker", sort=True
    ):
        if ticker in finished:
            continue

        d = adjusted(
            read_future_prices(
                args.raw / f"{ticker}.csv"
            )
        )

        for row in rows.itertuples():
            i = int(row.raw_i)

            if (
                i < 38
                or i + 5 >= len(d)
                or d.Date.iloc[i - 19]
                != row.window_start
                or d.Date.iloc[i]
                != row.decision_date
                or d.Date.iloc[i + 5]
                != row.target_date
            ):
                raise ValueError(
                    f"Index mismatch: "
                    f"{ticker} {row.decision_date}"
                )

            image = render_i20(
                d.iloc[i - 19 : i + 1]
            )

            if (
                image.shape != (64, 60)
                or image.dtype != np.uint8
            ):
                raise ValueError(
                    f"Chart format mismatch: {ticker}"
                )

            images[row.Index, 0] = image

        images.flush()

        finished.add(ticker)

        journal_file.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "done": sorted(finished),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if len(finished) % 10 == 0:
            print(
                f"[CHART] {len(finished)}/200",
                flush=True,
            )

    if len(finished) != 200:
        raise ValueError(
            "Rendering incomplete"
        )

    images.flush()
    del images

    ready_file.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "count": len(meta),
                "stocks": 200,
                "period": "2022-2025",
                "image_shape": list(shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    journal_file.unlink(
        missing_ok=True
    )

    print(
        f"[READY] {len(meta):,} "
        f"future charts. Saved: {args.out}",
        flush=True,
    )

    print(
        "[PRESERVED] Historical data "
        "and model outputs unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
