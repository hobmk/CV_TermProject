
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = [
    ("Day_Optuna", "CNN"),
    ("Day_Optuna", "Standard_ViT"),
    ("Day_Optuna", "LocalCNN_Best"),
    ("LocalCNN_Best", "CNN"),
]


def block_bootstrap_mean(
    values, years, reps, block, seed
):
    values = np.asarray(values, dtype=np.float64)
    years = np.asarray(years).astype(str)

    rng = np.random.default_rng(seed)

    totals = np.zeros(reps, dtype=np.float64)
    count = 0

    # Resample each calendar year separately
    for year in np.unique(years):
        segment = values[years == year]
        n = len(segment)

        if n == 0:
            continue

        n_blocks = (n + block - 1) // block

        starts = rng.integers(
            0, n, size=(reps, n_blocks)
        )

        offsets = np.arange(block)[
            None, None, :
        ]

        # Circular moving blocks
        indices = (
            starts[:, :, None] + offsets
        ) % n

        indices = indices.reshape(
            reps, -1
        )[:, :n]

        totals += segment[indices].sum(axis=1)
        count += n

    if count == 0:
        raise ValueError("No valid dates")

    bootstrap_means = totals / count

    low, high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/metrics/daily_signal_by_date.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/metrics/daily_ic_bootstrap.csv"
        ),
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--block-days",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.bootstraps < 100 or args.block_days < 1:
        parser.error(
            "bootstraps >= 100 and block-days >= 1 required"
        )

    data = pd.read_csv(
        args.input,
        dtype={
            "date": str,
            "year": str,
        },
    )

    required = {
        "date",
        "model",
        "rank_ic",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if data.duplicated(
        ["date", "model"]
    ).any():
        raise ValueError(
            "Duplicate date/model rows"
        )

    # Same dates are aligned across models
    wide = data.pivot(
        index="date",
        columns="model",
        values="rank_ic",
    ).sort_index()

    records = []

    def append_result(kind, name, series):
        frame = series.dropna().sort_index()

        years = (
            pd.Index(frame.index)
            .str[:4]
            .to_numpy()
        )

        for period in ("ALL", "2022", "2023"):
            if period == "ALL":
                mask = np.ones(
                    len(frame), dtype=bool
                )
            else:
                mask = years == period

            vals = frame.to_numpy(
                dtype=float
            )[mask]

            yr = years[mask]

            if len(vals) < 2:
                continue

            low, high = block_bootstrap_mean(
                vals,
                yr,
                args.bootstraps,
                args.block_days,
                args.seed,
            )

            records.append({
                "type": kind,
                "model_or_pair": name,
                "period": period,
                "n_dates": len(vals),
                "estimate": float(vals.mean()),
                "ci_low_95": low,
                "ci_high_95": high,
                "ci_excludes_zero": bool(
                    low > 0 or high < 0
                ),
            })

    # Model mean Rank IC
    for name in wide.columns:
        append_result(
            "mean_ic",
            name,
            wide[name],
        )

    # Paired model differences
    for first, second in PAIRS:
        if first not in wide or second not in wide:
            print(
                f"[SKIP PAIR] {first} vs {second}"
            )
            continue

        matching = wide[
            [first, second]
        ].dropna()

        difference = (
            matching[first] - matching[second]
        )

        append_result(
            "paired_difference",
            f"{first} - {second}",
            difference,
        )

    output = pd.DataFrame.from_records(records)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    print("\n[BLOCK BOOTSTRAP 95% INTERVALS]")

    print(
        output.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print(f"\n[SAVED] {args.output}")

    print(
        "[NOTE] Validation-based model selection "
        "is not corrected by these intervals."
    )

    print("[NOTE] Test set untouched.")


if __name__ == "__main__":
    main()
