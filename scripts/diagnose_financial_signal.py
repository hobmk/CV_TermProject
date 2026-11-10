
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


WINDOW = 20
HORIZON = 5

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_19",
    "range_20",
    "vol_19",
    "close_position",
    "volume_ratio",
]


def get_samples(path):
    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    required = {
        "ticker", "split", "window_start",
        "decision_date", "target_date",
        "label", "return_5d",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # The original train split includes 2015-2021.
    original = df[df["split"] == "train"].copy()

    train = original[
        (original["decision_date"] <= "2020-12-31")
        & (original["target_date"] <= "2020-12-31")
    ].copy()

    dev = original[
        (original["window_start"] >= "2021-01-01")
        & (original["decision_date"] <= "2021-12-31")
        & (original["target_date"] <= "2021-12-31")
    ].copy()

    if train.empty or dev.empty:
        raise RuntimeError("Train or Dev is empty")

    train["period"] = "train"
    dev["period"] = "dev"

    result = pd.concat(
        [train, dev],
        ignore_index=True,
    )

    if result.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError("Duplicate ticker/date")

    return result


def extract_features(samples, raw_dir):
    records = []

    for ticker, rows in samples.groupby(
        "ticker", sort=True
    ):
        path = raw_dir / f"{ticker}.csv"

        raw = pd.read_csv(path)

        raw["Date"] = pd.to_datetime(
            raw["Date"]
        ).dt.strftime("%Y-%m-%d")

        raw = raw.sort_values(
            "Date"
        ).reset_index(drop=True)

        if raw["Date"].duplicated().any():
            raise ValueError(
                f"Duplicate dates: {ticker}"
            )

        date_to_idx = {
            date: i
            for i, date in enumerate(raw["Date"])
        }

        values = raw[
            ["Open", "High", "Low", "Close", "Volume"]
        ].to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
        ):
            raise ValueError(
                f"Invalid OHLCV: {ticker}"
            )

        for row in rows.itertuples(index=False):
            idx = date_to_idx.get(
                row.decision_date
            )

            if (
                idx is None
                or idx < WINDOW - 1
                or idx + HORIZON >= len(raw)
            ):
                raise ValueError(
                    f"Incomplete window: "
                    f"{ticker} {row.decision_date}"
                )

            start = raw.at[
                idx - WINDOW + 1, "Date"
            ]

            target = raw.at[
                idx + HORIZON, "Date"
            ]

            if (
                start != row.window_start
                or target != row.target_date
            ):
                raise ValueError(
                    f"Window mismatch: "
                    f"{ticker} {row.decision_date}"
                )

            window = values[
                idx - WINDOW + 1:idx + 1
            ]

            close = window[:, 3]

            highest = float(
                window[:, 1].max()
            )

            lowest = float(
                window[:, 2].min()
            )

            future_return = (
                values[idx + HORIZON, 3]
                / close[-1] - 1.0
            )

            if (
                int(future_return > 0) != int(row.label)
                or abs(
                    future_return - float(row.return_5d)
                ) > 1e-8
            ):
                raise ValueError(
                    f"Label mismatch: "
                    f"{ticker} {row.decision_date}"
                )

            avg_volume = max(
                float(window[:, 4].mean()),
                1e-8,
            )

            records.append({
                "ticker": ticker,
                "period": row.period,
                "year": int(row.decision_date[:4]),
                "decision_date": row.decision_date,
                "label": int(row.label),
                "future_return": future_return,

                "ret_1": np.log(
                    close[-1] / close[-2]
                ),

                "ret_5": np.log(
                    close[-1] / close[-6]
                ),

                "ret_10": np.log(
                    close[-1] / close[-11]
                ),

                "ret_19": np.log(
                    close[-1] / close[0]
                ),

                "range_20": np.log(
                    highest / lowest
                ),

                "vol_19": np.diff(
                    np.log(close)
                ).std(),

                "close_position": (
                    (close[-1] - lowest)
                    / max(highest - lowest, 1e-8)
                ),

                "volume_ratio": (
                    window[-5:, 4].mean()
                    / avg_volume
                ),
            })

        print(
            f"[EXTRACTED] {ticker}: "
            f"{len(rows):,}",
            flush=True,
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["decision_date", "ticker"]
        )
        .reset_index(drop=True)
    )


def compute_auc(data, feature):
    if data["label"].nunique() < 2:
        return float("nan")

    return roc_auc_score(
        data["label"],
        data[feature],
    )


def compute_daily_ic(data, feature):
    group = data.groupby(
        "decision_date",
        sort=True,
    )

    x = group[feature].rank(
        method="average"
    )

    y = group["future_return"].rank(
        method="average"
    )

    dates = data["decision_date"]

    xc = x - x.groupby(
        dates
    ).transform("mean")

    yc = y - y.groupby(
        dates
    ).transform("mean")

    numerator = (
        (xc * yc)
        .groupby(dates)
        .sum()
    )

    denominator = np.sqrt(
        xc.pow(2).groupby(dates).sum()
        * yc.pow(2).groupby(dates).sum()
    )

    ic = numerator / denominator.replace(
        0, np.nan
    )

    sample_counts = group.size()

    return ic[
        sample_counts >= 10
    ].dropna()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(
            "data/processed/samples.csv"
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/metrics"),
    )

    args = parser.parse_args()

    samples = get_samples(
        args.samples
    )

    print("[EXTRACTING FEATURES]")

    data = extract_features(
        samples,
        args.raw_dir,
    )

    train = data[
        data["period"] == "train"
    ]

    dev = data[
        data["period"] == "dev"
    ]

    print("\n[DATA DISTRIBUTION]")

    for name, part in [
        ("TRAIN", train),
        ("DEV2021", dev),
    ]:
        daily_up = part.groupby(
            "decision_date"
        )["label"].mean()

        near_zero_ratio = (
            part["future_return"].abs() < 0.005
        ).mean()

        print(
            f"{name} "
            f"N={len(part):,} "
            f"Days={len(daily_up)} "
            f"Up={part['label'].mean():.4f} "
            f"MedianDailyUp={daily_up.median():.4f} "
            f"NearZeroReturn={near_zero_ratio:.4f}"
        )

    feature_rows = []

    for feature in FEATURES:
        train_auc = compute_auc(
            train, feature
        )

        dev_auc = compute_auc(
            dev, feature
        )

        train_ic = compute_daily_ic(
            train, feature
        )

        dev_ic = compute_daily_ic(
            dev, feature
        )

        feature_rows.append({
            "feature": feature,
            "train_auc": train_auc,
            "dev_auc": dev_auc,
            "train_mean_daily_ic": train_ic.mean(),
            "dev_mean_daily_ic": dev_ic.mean(),
            "auc_crosses_0.5": (
                (train_auc - 0.5)
                * (dev_auc - 0.5) < 0
            ),
        })

    feature_report = pd.DataFrame(
        feature_rows
    )

    print("\n[FEATURE RELATIONSHIP]")

    print(
        feature_report.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    annual_rows = []

    for year, part in data.groupby(
        "year", sort=True
    ):
        ic = compute_daily_ic(
            part, "ret_5"
        )

        daily_up = part.groupby(
            "decision_date"
        )["label"].mean()

        annual_rows.append({
            "year": year,
            "samples": len(part),
            "days": len(daily_up),
            "up_rate": part["label"].mean(),
            "median_daily_up": daily_up.median(),
            "past5_auc": compute_auc(
                part, "ret_5"
            ),
            "past5_daily_ic": ic.mean(),
            "near_zero_return_ratio": (
                part["future_return"].abs()
                < 0.005
            ).mean(),
        })

    annual_report = pd.DataFrame(
        annual_rows
    )

    print("\n[ANNUAL RELATIONSHIP]")

    print(
        annual_report.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        )
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        args.out_dir
        / "regime_feature_summary.csv"
    )

    annual_path = (
        args.out_dir
        / "regime_annual_summary.csv"
    )

    feature_report.to_csv(
        feature_path,
        index=False,
    )

    annual_report.to_csv(
        annual_path,
        index=False,
    )

    print(f"\n[SAVED] {feature_path}")
    print(f"[SAVED] {annual_path}")

    print(
        "[NOTE] Descriptive diagnostics only. "
        "No 2022-2025 evaluation."
    )


if __name__ == "__main__":
    main()
