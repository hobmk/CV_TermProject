
"""Frozen 2022-2025 holdout Top20 EW backtest.

Inputs:
    Saved 10-seed ensemble predictions.
    Audited 2022-2025 adjusted prices.

Strategies:
    JKX CNN Top20 EW
    Day-Aligned ViT Top20 EW
    EW100

Decide at t close.
Buy at t+1 open.
Sell at t+6 open.
Five rebalance offsets.
5 bps transaction cost per side.

Run:
    python -u -m scripts.holdout_top20_eval --stage audit
    python -u -m scripts.holdout_top20_eval --stage backtest
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import mvo_compare as base


YEARS = (2022, 2023, 2024, 2025)

MODELS = (
    "JKX_CNN",
    "DayAligned_ViT",
)

KEYS = [
    "ticker",
    "group",
    "decision_date",
    "target_date",
    "label",
]

TOP_K = 20
COST_BPS = 5.0


def load_predictions(root, manifest):
    legacy = set(
        manifest.loc[
            manifest.group == "legacy",
            "ticker",
        ]
    )

    if (
        len(manifest) != 200
        or manifest.ticker.nunique() != 200
        or len(legacy) != 100
    ):
        raise ValueError(
            "Expected frozen 200-stock universe "
            "with 100 legacy stocks"
        )

    predictions = {}

    for year in YEARS:
        reference = None

        for model in MODELS:
            path = (
                root
                / "ensemble_predictions"
                / f"test{year}"
                / f"{model}.csv"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing ensemble predictions: {path}"
                )

            frame = pd.read_csv(
                path,
                dtype={
                    "ticker": str,
                    "group": str,
                    "decision_date": str,
                    "target_date": str,
                },
            )

            required = set(
                KEYS + ["prob_up", "return_5d"]
            )

            if not required.issubset(frame.columns):
                raise ValueError(
                    f"Missing columns: {path}"
                )

            frame = (
                frame.sort_values(
                    ["decision_date", "ticker"]
                )
                .reset_index(drop=True)
            )

            if frame.duplicated(
                ["ticker", "decision_date"]
            ).any():
                raise ValueError(
                    f"Duplicate predictions: {path}"
                )

            if not frame.decision_date.str.startswith(
                str(year)
            ).all():
                raise ValueError(
                    f"Wrong evaluation year: {path}"
                )

            if not frame.prob_up.between(
                0, 1
            ).all():
                raise ValueError(
                    f"Invalid probabilities: {path}"
                )

            values = frame[
                ["return_5d", "prob_up"]
            ].to_numpy(float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nonfinite predictions: {path}"
                )

            labels = (
                frame.return_5d.to_numpy(float) > 0
            ).astype(int)

            if not np.array_equal(
                frame.label.to_numpy(int),
                labels,
            ):
                raise ValueError(
                    f"Incorrect labels: {path}"
                )

            if reference is not None:
                if not frame[KEYS].equals(
                    reference[KEYS]
                ):
                    raise ValueError(
                        f"CNN/ViT alignment mismatch: {year}"
                    )

                if not np.allclose(
                    frame.return_5d,
                    reference.return_5d,
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"Return mismatch: {year}"
                    )
            else:
                reference = frame

            common = (
                frame.loc[
                    frame.ticker.isin(legacy)
                ]
                .copy()
                .reset_index(drop=True)
            )

            if common.ticker.nunique() != 100:
                raise ValueError(
                    f"Common100 missing: {model} {year}"
                )

            counts = (
                common.groupby("decision_date")
                .ticker.nunique()
            )

            if not counts.eq(100).all():
                raise ValueError(
                    f"Expected 100 stocks per day: "
                    f"{model} {year}"
                )

            targets = (
                common.groupby("decision_date")
                .target_date.nunique()
            )

            if not targets.eq(1).all():
                raise ValueError(
                    f"Inconsistent target dates: "
                    f"{model} {year}"
                )

            predictions[(year, model)] = common

    return predictions, sorted(legacy)


def audit_probabilities(predictions):
    rows = []

    for (year, model), frame in predictions.items():
        p = frame.prob_up.to_numpy(float)

        daily_std = (
            frame.groupby("decision_date")
            .prob_up.std()
            .mean()
        )

        rows.append({
            "year": year,
            "model": model,
            "mean_prob": float(np.mean(p)),
            "std_prob": float(np.std(p)),
            "daily_std": float(daily_std),
            "fraction_up_at_half": float(
                np.mean(p >= 0.5)
            ),
            "minimum": float(np.min(p)),
            "maximum": float(np.max(p)),
        })

    table = pd.DataFrame(rows)

    print(
        "\n[PROBABILITY DIAGNOSTIC: Common100]",
        flush=True,
    )

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    return table


def load_adjusted_opens(root, tickers):
    columns = {}

    for ticker in tickers:
        path = root / f"{ticker}.csv"

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(
            path,
            usecols=[
                "Date",
                "Open",
                "Close",
                "Adj Close",
            ],
        )

        frame["Date"] = pd.to_datetime(
            frame["Date"],
            errors="raise",
        )

        if frame.Date.duplicated().any():
            raise ValueError(
                f"Duplicate price dates: {ticker}"
            )

        price_columns = [
            "Open",
            "Close",
            "Adj Close",
        ]

        for col in price_columns:
            frame[col] = pd.to_numeric(
                frame[col],
                errors="raise",
            )

        prices = frame[
            price_columns
        ].to_numpy(float)

        if (
            not np.isfinite(prices).all()
            or (prices <= 0).any()
        ):
            raise ValueError(
                f"Invalid prices: {ticker}"
            )

        frame = (
            frame.set_index("Date")
            .sort_index()
        )

        factor = (
            frame["Adj Close"]
            / frame["Close"]
        )

        columns[ticker] = (
            frame["Open"] * factor
        )

    adj_open = (
        pd.DataFrame(columns)
        .sort_index()
    )

    # Same calendar convention as previous MVO:
    # at least 80 available common stocks.
    valid = (
        adj_open.notna().sum(axis=1) >= 80
    )

    calendar = adj_open.index[valid]

    adj_open = adj_open.reindex(calendar)

    if (
        calendar.min()
        > pd.Timestamp("2021-12-01")
        or calendar.max()
        < pd.Timestamp("2025-12-30")
    ):
        raise ValueError(
            "Price calendar does not cover "
            "the required evaluation period"
        )

    return adj_open, calendar


def run_offset(
    offset,
    predictions,
    tickers,
    adj_open,
    calendar,
):
    price_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    lookups = {}
    schedules = {}

    for year in YEARS:
        dates = None

        for model in MODELS:
            frame = predictions[
                (year, model)
            ]

            these = sorted(
                pd.to_datetime(
                    frame.decision_date.unique()
                )
            )

            if (
                dates is not None
                and dates != these
            ):
                raise ValueError(
                    f"CNN/ViT schedule mismatch: {year}"
                )

            dates = these

            lookups[(year, model)] = {
                pd.Timestamp(date): group
                for date, group in frame.groupby(
                    "decision_date"
                )
            }

        schedules[year] = dates[offset::5]

    records = []
    states = {}

    for year in YEARS:
        completed = 0

        for decision in schedules[year]:
            if decision not in price_index:
                raise ValueError(
                    f"Missing decision date: {decision}"
                )

            i = price_index[decision]

            if i + 6 >= len(calendar):
                continue

            entry = calendar[i + 1]
            exit_date = calendar[i + 6]

            # Match earlier year-end convention.
            if exit_date.year != year:
                continue

            reference = lookups[
                (year, MODELS[0])
            ][decision]

            target = pd.Timestamp(
                reference.target_date.iloc[0]
            )

            if target != calendar[i + 5]:
                raise ValueError(
                    f"Target/calendar mismatch: "
                    f"{decision.date()}"
                )

            # Equal-weight 100-stock benchmark.
            allocations = {
                "EW100": dict.fromkeys(
                    tickers,
                    1.0 / len(tickers),
                )
            }

            for model in MODELS:
                day = lookups[
                    (year, model)
                ][decision]

                # FIXED: Top20 selected by score.
                selected = (
                    day.sort_values(
                        ["prob_up", "ticker"],
                        ascending=[False, True],
                        kind="stable",
                    )
                    .head(TOP_K)
                )

                if len(selected) != TOP_K:
                    raise ValueError(
                        "Insufficient Top20 candidates"
                    )

                name = f"{model}_Top20_EW"

                allocations[name] = dict.fromkeys(
                    selected.ticker.tolist(),
                    1.0 / TOP_K,
                )

            # Future prices are accessed only
            # after stocks and weights are fixed.
            needed = sorted(
                set().union(
                    *(
                        set(a)
                        for a in allocations.values()
                    )
                )
            )

            entry_prices = adj_open.loc[
                entry, needed
            ]

            exit_prices = adj_open.loc[
                exit_date, needed
            ]

            if (
                entry_prices.isna().any()
                or exit_prices.isna().any()
            ):
                raise ValueError(
                    f"Missing execution prices: "
                    f"{decision.date()}"
                )

            realized = (
                exit_prices / entry_prices - 1.0
            ).to_dict()

            for name, allocation in allocations.items():
                state_key = (year, name)

                gross, net, turnover, states[
                    state_key
                ] = base.compute_cost_and_update(
                    states.get(state_key),
                    allocation,
                    entry,
                    exit_date,
                    realized,
                    COST_BPS,
                )

                records.append({
                    "year": year,
                    "date": decision.date().isoformat(),
                    "entry_open": entry.date().isoformat(),
                    "exit_open": exit_date.date().isoformat(),
                    "strategy": name,
                    "n_positions": len(allocation),
                    "gross_return": gross,
                    "net_return": net,
                    "turnover": turnover,
                })

            completed += 1

        print(
            f"[REBAL] offset={offset} "
            f"year={year} "
            f"completed={completed}",
            flush=True,
        )

        if completed < 35:
            raise ValueError(
                f"Too few rebalances: "
                f"{year} offset={offset}"
            )

    results = pd.DataFrame(records)

    # Apply the same annual liquidation
    # transaction cost as development backtest.
    for _, group in results.groupby(
        ["year", "strategy"]
    ):
        last = group.index[-1]

        results.loc[
            last,
            "net_return",
        ] -= COST_BPS / 10000.0

        results.loc[
            last,
            "turnover",
        ] += 1.0

    return results


def report(summaries, output):
    combined = pd.concat(
        summaries,
        ignore_index=True,
    )

    overall = combined.loc[
        combined.period == "ALL"
    ].copy()

    columns = [
        "cumulative_net",
        "sharpe_net",
        "max_drawdown_net",
        "mean_turnover",
    ]

    means = (
        overall.groupby(
            "strategy",
            as_index=False,
        )[columns]
        .mean()
    )

    yearly = (
        combined.loc[
            combined.period != "ALL"
        ]
        .groupby(
            ["period", "strategy"],
            as_index=False,
        )[columns]
        .mean()
    )

    pivot = overall.pivot(
        index="offset",
        columns="strategy",
        values="cumulative_net",
    )

    paired = pd.DataFrame({
        "offset": pivot.index,
        "vit_minus_cnn_pp": (
            100
            * (
                pivot["DayAligned_ViT_Top20_EW"]
                - pivot["JKX_CNN_Top20_EW"]
            ).to_numpy()
        ),
        "vit_minus_ew100_pp": (
            100
            * (
                pivot["DayAligned_ViT_Top20_EW"]
                - pivot["EW100"]
            ).to_numpy()
        ),
    })

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output / "all_offset_summaries.csv",
        index=False,
    )

    means.to_csv(
        output / "five_offset_means.csv",
        index=False,
    )

    yearly.to_csv(
        output / "yearly_offset_means.csv",
        index=False,
    )

    paired.to_csv(
        output / "paired_offset_deltas.csv",
        index=False,
    )

    shown = means.copy()

    shown["cumulative_net"] *= 100
    shown["max_drawdown_net"] *= 100

    print(
        "\n[HOLDOUT 5-OFFSET MEANS] "
        "Cumulative return and MDD in %",
        flush=True,
    )

    print(
        shown.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ),
        flush=True,
    )

    annual = yearly[
        [
            "period",
            "strategy",
            "cumulative_net",
            "sharpe_net",
        ]
    ].copy()

    annual["cumulative_net"] *= 100

    print(
        "\n[HOLDOUT ANNUAL 5-OFFSET MEANS]",
        flush=True,
    )

    print(
        annual.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ),
        flush=True,
    )

    print(
        "\n[PAIRED OFFSETS] "
        "Return differences in percentage points",
        flush=True,
    )

    print(
        paired.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ),
        flush=True,
    )

    print(
        f"\n[SAVED] {output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=["audit", "backtest"],
        default="audit",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
        ),
    )

    parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path(
            "outputs/exp200_holdout_10seed_eval"
        ),
    )

    parser.add_argument(
        "--prices",
        type=Path,
        default=Path(
            "data/exp200_holdout_audit/future_raw"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/exp200_holdout_top20_ew"
        ),
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest,
        dtype={
            "ticker": str,
            "group": str,
        },
    )

    predictions, tickers = load_predictions(
        args.ensemble,
        manifest,
    )

    audit = audit_probabilities(
        predictions
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        args.out / "probability_diagnostics.csv",
        index=False,
    )

    if args.stage == "audit":
        return

    adj_open, calendar = load_adjusted_opens(
        args.prices,
        tickers,
    )

    summaries = []

    print(
        "[FROZEN STRATEGY] "
        "Top20 Score EW | 5bp per side "
        "| offsets 0..4",
        flush=True,
    )

    for offset in range(5):
        results = run_offset(
            offset,
            predictions,
            tickers,
            adj_open,
            calendar,
        )

        folder = (
            args.out / f"offset{offset}"
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            folder / "period_returns.csv",
            index=False,
        )

        summary = base.summarize(
            results
        )

        summary.insert(
            0,
            "offset",
            offset,
        )

        summary.to_csv(
            folder / "summary.csv",
            index=False,
        )

        summaries.append(summary)

    report(
        summaries,
        args.out,
    )


if __name__ == "__main__":
    main()
