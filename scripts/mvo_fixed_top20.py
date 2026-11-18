
"""Score-fixed Top20 ablation for the existing MVO pipeline.

2017: initial out-of-time calibration.
2018-2021: portfolio evaluation.

The model score alone selects Top20 stocks.
EW, MinVar and MVO use the SAME selected stocks.
Calibration slope affects only MVO portfolio weights.

2022+ holdout remains untouched.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

from scripts import mvo_compare as base


def evaluate_offset(
    args,
    predictions,
    adj_open,
    daily_returns,
    calendar,
    calibrations,
    lookups,
    offset,
):
    price_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    schedule = base.rebalance_dates(
        predictions[base.MODELS[0]],
        offset,
    )

    for model in base.MODELS[1:]:
        other = base.rebalance_dates(
            predictions[model],
            offset,
        )

        if other != schedule:
            raise ValueError(
                "CNN/ViT rebalance dates differ"
            )

    records = []
    weights_rows = []
    slope_rows = []

    states = {}

    for num, decision in enumerate(schedule, 1):
        year = decision.year

        if decision not in price_index:
            raise ValueError(
                f"Missing decision date: {decision}"
            )

        idx = price_index[decision]

        if idx + 6 >= len(calendar):
            continue

        entry = calendar[idx + 1]
        exit_date = calendar[idx + 6]

        if (
            exit_date.year != year
            or exit_date >= pd.Timestamp("2022-01-01")
        ):
            continue

        # Only historical returns known at decision close.
        history = (
            daily_returns.loc[:decision]
            .tail(args.cov_lookback)
        )

        if len(history) != args.cov_lookback:
            raise ValueError(
                f"Insufficient history: {decision.date()}"
            )

        eligible = history.notna().all(axis=0)

        available = (
            eligible.index[eligible]
            .tolist()
        )

        if len(available) < 80:
            raise ValueError(
                f"Too few eligible stocks: {decision.date()}"
            )

        allocations = {
            "EW100": dict.fromkeys(
                available,
                1.0 / len(available),
            )
        }

        for model in base.MODELS:
            day = (
                lookups[model][decision]
                .copy()
            )

            slope, n_calib = base.slope_at(
                calibrations[model],
                decision,
                args.calib_lookback,
                args.min_calib_days,
            )

            day["rank_x"] = base.rank_features(
                day
            )

            day["mu_5d"] = (
                slope * day["rank_x"]
            )

            candidates = day.loc[
                day.index.intersection(available)
            ]

            # -----------------------------------------
            # CRITICAL ABLATION:
            # Select stocks by original model SCORE.
            # Do NOT select by signed calibrated mu.
            # -----------------------------------------
            selected = (
                candidates.sort_values(
                    "score",
                    ascending=False,
                    kind="stable",
                )
                .head(args.top_k)
            )

            if len(selected) != args.top_k:
                raise ValueError(
                    f"Insufficient Top20 candidates: "
                    f"{decision.date()}"
                )

            tickers = selected.index.tolist()

            # Same selected stocks for EW, MinVar, MVO.
            covariance = (
                LedoitWolf(
                    store_precision=False
                )
                .fit(
                    history[tickers].to_numpy(float)
                )
                .covariance_
                * 5.0
            )

            mu = (
                selected.mu_5d
                .to_numpy(float)
            )

            equal_weights = np.full(
                args.top_k,
                1.0 / args.top_k,
            )

            minvar_weights = base.mvo_weights(
                np.zeros_like(mu),
                covariance,
                1.0,
                args.max_weight,
            )

            mvo_weights = base.mvo_weights(
                mu,
                covariance,
                args.risk_aversion,
                args.max_weight,
            )

            slope_rows.append({
                "offset": offset,
                "year": year,
                "date": decision.date().isoformat(),
                "model": model,
                "slope": slope,
                "calibration_days": n_calib,
            })

            strategies = (
                ("Top20_EW", equal_weights),
                ("Top20_MinVar", minvar_weights),
                ("Top20_MVO", mvo_weights),
            )

            for strategy, weights in strategies:
                name = f"{model}_{strategy}"

                allocations[name] = dict(
                    zip(tickers, weights)
                )

                for ticker, weight in zip(
                    tickers,
                    weights,
                ):
                    weights_rows.append({
                        "offset": offset,
                        "year": year,
                        "date": decision.date().isoformat(),
                        "strategy": name,
                        "ticker": ticker,
                        "weight": float(weight),
                        "score": float(
                            selected.loc[ticker, "score"]
                        ),
                        "mu_5d": float(
                            selected.loc[ticker, "mu_5d"]
                        ),
                    })

        # Future returns are read only after allocation.
        needed = sorted(
            set().union(
                *(
                    set(allocation)
                    for allocation in allocations.values()
                )
            )
        )

        entry_prices = adj_open.loc[
            entry,
            needed,
        ]

        exit_prices = adj_open.loc[
            exit_date,
            needed,
        ]

        invalid = (
            entry_prices.isna().any()
            or exit_prices.isna().any()
            or (entry_prices <= 0).any()
            or (exit_prices <= 0).any()
        )

        if invalid:
            raise ValueError(
                f"Invalid execution prices: "
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
                args.cost_bps,
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

        if num == 1 or num % 50 == 0:
            print(
                f"[REBAL] offset={offset} "
                f"{num}/{len(schedule)} "
                f"date={decision.date()} "
                f"eligible={len(available)}",
                flush=True,
            )

    if not records:
        raise RuntimeError(
            f"No trades for offset={offset}"
        )

    results = pd.DataFrame(records)

    # Same year-end liquidation rule as original.
    if args.cost_bps:
        for _, group in results.groupby(
            ["year", "strategy"]
        ):
            last = group.index[-1]

            results.loc[
                last,
                "net_return",
            ] -= args.cost_bps / 10000.0

            results.loc[
                last,
                "turnover",
            ] += 1.0

    return (
        results,
        pd.DataFrame(weights_rows),
        pd.DataFrame(slope_rows),
    )


def report(all_summaries, output):
    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    overall = combined.loc[
        combined.period == "ALL"
    ].copy()

    cols = [
        "cumulative_net",
        "sharpe_net",
        "max_drawdown_net",
        "mean_turnover",
    ]

    means = (
        overall.groupby(
            "strategy",
            as_index=False,
        )[cols]
        .mean()
        .sort_values("strategy")
    )

    ranges = (
        overall.groupby("strategy")
        .cumulative_net.agg(["min", "max"])
    )

    means["min_cumulative_net"] = (
        means.strategy.map(ranges["min"])
    )

    means["max_cumulative_net"] = (
        means.strategy.map(ranges["max"])
    )

    yearly = (
        combined.loc[
            combined.period != "ALL"
        ]
        .groupby(
            ["period", "strategy"],
            as_index=False,
        )[cols]
        .mean()
    )

    pair_cols = [
        "cumulative_net",
        "sharpe_net",
        "max_drawdown_net",
    ]

    deltas = []

    comparisons = (
        (
            "DayAligned_ViT_Top20_EW",
            "JKX_CNN_Top20_EW",
            "ViT_EW-minus-CNN_EW",
        ),
        (
            "DayAligned_ViT_Top20_MVO",
            "JKX_CNN_Top20_MVO",
            "ViT_MVO-minus-CNN_MVO",
        ),
        (
            "DayAligned_ViT_Top20_MinVar",
            "DayAligned_ViT_Top20_EW",
            "ViT_MinVar-minus-EW",
        ),
        (
            "DayAligned_ViT_Top20_MVO",
            "DayAligned_ViT_Top20_EW",
            "ViT_MVO-minus-EW",
        ),
        (
            "DayAligned_ViT_Top20_MVO",
            "DayAligned_ViT_Top20_MinVar",
            "ViT_MVO-minus-MinVar",
        ),
    )

    for offset, rows in overall.groupby("offset"):
        indexed = rows.set_index("strategy")

        for left, right, label in comparisons:
            row = {
                "offset": offset,
                "contrast": label,
            }

            for metric in pair_cols:
                row["delta_" + metric] = (
                    float(indexed.loc[left, metric])
                    - float(indexed.loc[right, metric])
                )

            deltas.append(row)

    diffs = pd.DataFrame(deltas)

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

    diffs.to_csv(
        output / "paired_offset_deltas.csv",
        index=False,
    )

    print(
        "\n[5-OFFSET MEANS] "
        "percentage columns shown as %",
        flush=True,
    )

    shown = means.copy()

    for col in [
        "cumulative_net",
        "max_drawdown_net",
        "min_cumulative_net",
        "max_cumulative_net",
    ]:
        shown[col] *= 100

    print(
        shown.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ),
        flush=True,
    )

    print(
        "\n[PAIRED OFFSETS] "
        "cumulative return differences "
        "in percentage points",
        flush=True,
    )

    shown = diffs[
        [
            "offset",
            "contrast",
            "delta_cumulative_net",
            "delta_sharpe_net",
        ]
    ].copy()

    shown["delta_cumulative_net"] *= 100

    print(
        shown.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ),
        flush=True,
    )

    print(
        f"\n[SAVED] {output}",
        flush=True,
    )

    print(
        "[HOLDOUT] 2022-2025 untouched",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cnn",
        type=Path,
        default=Path("outputs/exp200_factorial"),
    )

    parser.add_argument(
        "--vit",
        type=Path,
        default=Path("outputs/exp200_day_vit"),
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/exp200/raw"),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/exp200/universe200.csv"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/exp200_mvo_fixed_top20_2018_2021"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--cov-lookback",
        type=int,
        default=126,
    )

    parser.add_argument(
        "--calib-lookback",
        type=int,
        default=252,
    )

    parser.add_argument(
        "--min-calib-days",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    invalid = (
        len(args.seeds) < 2
        or len(set(args.seeds)) != len(args.seeds)
        or args.top_k < 10
        or args.top_k > 100
        or args.max_weight <= 0
        or args.top_k * args.max_weight < 1 - 1e-8
        or args.cov_lookback < 10
        or args.min_calib_days < 1
        or args.calib_lookback < args.min_calib_days
        or args.risk_aversion <= 0
        or args.cost_bps < 0
    )

    if invalid:
        parser.error(
            "Invalid experiment parameters"
        )

    manifest = pd.read_csv(
        args.manifest,
        dtype={
            "ticker": str,
            "group": str,
        },
    )

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
            "Expected 200 stocks with "
            "exactly 100 legacy stocks"
        )

    predictions = base.load_ensembles(
        args,
        legacy,
    )

    adjusted_open, returns, calendar = (
        base.load_prices(
            args.raw,
            legacy,
        )
    )

    calibrations = {
        model: base.calibration_table(
            predictions[model],
            adjusted_open,
            calendar,
        )
        for model in base.MODELS
    }

    lookups = {
        model: {
            pd.Timestamp(date): group.set_index(
                "ticker"
            )
            for date, group in predictions[
                model
            ].groupby("decision_date")
        }
        for model in base.MODELS
    }

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(args).copy()

    config.update({
        "initial_calibration_year": 2017,
        "evaluation_years": [
            2018, 2019, 2020, 2021
        ],
        "selection":
            "score_descending_top20_fixed",
        "calibration_target":
            "adjusted_open_t1_to_t6",
        "cost_applied_each_side_bps":
            args.cost_bps,
        "offsets": list(range(5)),
        "year_end_liquidation": True,
    })

    (
        args.out / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    summaries = []

    print(
        "[RUN] Score-fixed Top20 | "
        "2017 calibration | "
        "2018-2021 evaluation | "
        "offsets=0..4",
        flush=True,
    )

    for offset in range(5):
        print(
            f"\n[OFFSET] {offset}",
            flush=True,
        )

        results, weights, slopes = (
            evaluate_offset(
                args,
                predictions,
                adjusted_open,
                returns,
                calendar,
                calibrations,
                lookups,
                offset,
            )
        )

        summary = base.summarize(results)

        summary.insert(
            0,
            "offset",
            offset,
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

        weights.to_csv(
            folder / "weights.csv",
            index=False,
        )

        slopes.to_csv(
            folder / "calibration_slopes.csv",
            index=False,
        )

        summary.to_csv(
            folder / "summary.csv",
            index=False,
        )

        summaries.append(summary)

        focus = summary.loc[
            (summary.period == "ALL")
            & summary.strategy.isin([
                "EW100",
                "JKX_CNN_Top20_EW",
                "DayAligned_ViT_Top20_EW",
                "JKX_CNN_Top20_MVO",
                "DayAligned_ViT_Top20_MVO",
            ])
        ]

        print(
            "[OFFSET RESULT]",
            flush=True,
        )

        print(
            focus[
                [
                    "strategy",
                    "cumulative_net",
                    "sharpe_net",
                    "max_drawdown_net",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:+.5f}",
            ),
            flush=True,
        )

    report(
        summaries,
        args.out,
    )


if __name__ == "__main__":
    main()
