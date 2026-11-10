
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "ticker",
    "decision_date",
    "target_date",
    "return_5d",
    "prob_up",
}


def read_predictions(folder, year):
    path = (
        folder
        / f"test{year}"
        / "seed42"
        / "predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    missing = REQUIRED - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing {sorted(missing)}"
        )

    if df.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError(
            f"{path}: duplicated ticker/date"
        )

    if not (
        df.decision_date.str[:4] == str(year)
    ).all():
        raise ValueError(
            f"{path}: unexpected years"
        )

    if not (
        df.decision_date < df.target_date
    ).all():
        raise ValueError(
            f"{path}: invalid target dates"
        )

    for col in ("return_5d", "prob_up"):
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"{path}: invalid {col}"
            )

    if not df.prob_up.between(0, 1).all():
        raise ValueError(
            f"{path}: invalid probability"
        )

    return df


def make_decile_series(frame, offset):
    dates = sorted(
        frame.decision_date.unique()
    )

    # Rebalance every 5 trading days.
    anchor_dates = dates[offset::5]

    results = []

    for date in anchor_dates:
        group = (
            frame.loc[
                frame.decision_date == date
            ]
            .sort_values(
                ["prob_up", "ticker"]
            )
            .reset_index(drop=True)
        )

        if len(group) < 20:
            raise ValueError(
                f"{date}: too few stocks"
            )

        if group.target_date.nunique() != 1:
            raise ValueError(
                f"{date}: target dates differ"
            )

        bins = np.array_split(
            np.arange(len(group)),
            10,
        )

        rec = {
            "date": date,
            "target_date": str(
                group.target_date.iloc[0]
            ),
            "stocks": len(group),
            "offset": offset,
        }

        for n, ix in enumerate(
            bins,
            start=1,
        ):
            rec[f"d{n}"] = float(
                group.loc[
                    ix, "return_5d"
                ].mean()
            )

        rec["hl"] = (
            rec["d10"] - rec["d1"]
        )

        rec["top_excess"] = (
            rec["d10"]
            - group.return_5d.mean()
        )

        results.append(rec)

    out = pd.DataFrame(results)

    # Verify non-overlapping holdings.
    if len(out) >= 2:
        ends = out.target_date.iloc[
            :-1
        ].to_numpy()

        next_starts = out.date.iloc[
            1:
        ].to_numpy()

        if not (
            ends <= next_starts
        ).all():
            raise ValueError(
                "Overlapping holding periods"
            )

    return out


def summarize(
    df,
    universe,
    year,
    offset,
    start_year,
):
    if len(df) < 3:
        raise ValueError(
            "Not enough rebalances"
        )

    mean_return = float(
        df.hl.mean()
    )

    std_return = float(
        df.hl.std(ddof=1)
    )

    sharpe = (
        mean_return
        / std_return
        * np.sqrt(252 / 5)
        if std_return > 0
        else np.nan
    )

    decile_returns = np.array([
        df[f"d{i}"].mean()
        for i in range(1, 11)
    ])

    monotonicity = float(
        pd.Series(
            np.arange(1, 11)
        ).corr(
            pd.Series(decile_returns),
            method="spearman",
        )
    )

    return {
        "year": year,
        "universe": universe,
        "train_start": start_year,
        "offset": offset,
        "rebalances": len(df),
        "mean_5d_hl": mean_return,
        "annualized_sharpe_gross": sharpe,
        "top_excess_5d": float(
            df.top_excess.mean()
        ),
        "d1_mean": float(
            df.d1.mean()
        ),
        "d10_mean": float(
            df.d10.mean()
        ),
        "decile_spearman": monotonicity,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "outputs/exp200_factorial"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/jkx_paper_portfolio_eval"
        ),
    )

    parser.add_argument(
        "--universes",
        type=int,
        nargs="+",
        default=[100, 200],
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021],
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    all_rebalances = []

    for universe in args.universes:
        folder = (
            args.base
            / f"u{universe}_start{args.start_year}"
        )

        for year in args.years:
            df = read_predictions(
                folder,
                year,
            )

            # Offset 0 is the predefined
            # primary evaluation.
            # Other offsets are sensitivity checks.
            for offset in range(5):
                weekly = make_decile_series(
                    df,
                    offset,
                )

                weekly.insert(
                    0, "year", year
                )

                weekly.insert(
                    1, "universe", universe
                )

                weekly.insert(
                    2, "train_start",
                    args.start_year,
                )

                summaries.append(
                    summarize(
                        weekly,
                        universe,
                        year,
                        offset,
                        args.start_year,
                    )
                )

                all_rebalances.append(
                    weekly
                )

    overview = pd.DataFrame(
        summaries
    )

    weekly = pd.concat(
        all_rebalances,
        ignore_index=True,
    )

    overview.to_csv(
        args.out / "summary_by_offset.csv",
        index=False,
    )

    weekly.to_csv(
        args.out / "weekly_deciles.csv",
        index=False,
    )

    paper_style = overview.loc[
        overview.offset == 0
    ].copy()

    paper_style.to_csv(
        args.out / "paper_style_offset0.csv",
        index=False,
    )

    print(
        "\n[PAPER-STYLE] "
        "Fixed offset 0, 5-day holding"
    )

    print(
        paper_style.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print(
        "\n[OFFSET SENSITIVITY] "
        "5-day H-L returns"
    )

    pivot = overview.pivot_table(
        index=["year", "universe"],
        columns="offset",
        values="mean_5d_hl",
    )

    print(
        pivot.to_string(
            float_format=lambda x: f"{x:+.5f}"
        )
    )

    print(
        f"\n[SAVED] {args.out}"
    )

    print(
        "[NOTE] No trading costs; "
        "Close-to-Close return assumption."
    )

    print(
        "[HOLDOUT] 2022-2025 untouched"
    )


if __name__ == "__main__":
    main()
