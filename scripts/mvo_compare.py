
"""MVO comparison using saved JKX CNN and Day-Aligned ViT ensembles.

2019: historical out-of-time predictions for return calibration.
2020-2021: portfolio evaluation using information available at each date.

Signals are observed at the decision-day close.
Trades enter at the next trading day's open and exit five sessions later.
No 2022+ data are used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


MODELS = {
    "JKX_CNN": "exp200_factorial",
    "DayAligned_ViT": "exp200_day_vit",
}

KEYS = ["ticker", "decision_date", "target_date"]
REQUIRED = KEYS + ["group", "label", "return_5d", "prob_up"]


def read_predictions(path, year):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "group": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    if not set(REQUIRED).issubset(df.columns):
        raise ValueError(f"Missing prediction fields: {path}")

    df = (
        df[REQUIRED]
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )

    if df.duplicated(["ticker", "decision_date"]).any():
        raise ValueError(f"Duplicate ticker/date: {path}")

    if not df.decision_date.str.startswith(str(year)).all():
        raise ValueError(f"Test year mismatch: {path}")

    if not (
        (df.target_date > df.decision_date)
        & (df.target_date < "2022-01-01")
    ).all():
        raise ValueError(f"Invalid target date: {path}")

    for col in ("label", "return_5d", "prob_up"):
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {col}: {path}")

    if not df.prob_up.between(0, 1).all():
        raise ValueError(f"Invalid probability: {path}")

    expected = (df.return_5d.to_numpy(float) > 0).astype(int)

    if not np.array_equal(df.label.to_numpy(int), expected):
        raise ValueError(f"Incorrect return label: {path}")

    return df


def ensemble_preds(base, year, seeds, legacy):
    reference = None
    scores = []

    for seed in seeds:
        path = (
            base
            / "u200_start2010"
            / f"test{year}"
            / f"seed{seed}"
            / "predictions.csv"
        )

        df = read_predictions(path, year)

        if reference is None:
            reference = df
        else:
            columns = KEYS + ["group", "label"]

            if not reference[columns].equals(df[columns]):
                raise ValueError(
                    f"Seed alignment mismatch: {path}"
                )

            if not np.allclose(
                reference.return_5d,
                df.return_5d,
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(
                    f"Future-return mismatch: {path}"
                )

        scores.append(df.prob_up.to_numpy(float))

    mask = reference.ticker.isin(legacy).to_numpy()

    out = reference.loc[mask].copy()
    out["score"] = np.mean(scores, axis=0)[mask]

    if out.ticker.nunique() != 100:
        raise ValueError("Expected 100 common stocks")

    counts = out.groupby("decision_date").ticker.nunique()

    if not counts.eq(100).all():
        raise ValueError("Expected 100 stocks per decision date")

    if (
        out.groupby("decision_date").target_date.nunique() != 1
    ).any():
        raise ValueError("Inconsistent target dates")

    return out.reset_index(drop=True)


def load_ensembles(args, legacy):
    predictions = {}

    for model in MODELS:
        root = args.cnn if model == "JKX_CNN" else args.vit

        frames = [
            ensemble_preds(root, year, args.seeds, legacy)
            for year in (2019, 2020, 2021)
        ]

        predictions[model] = pd.concat(
            frames,
            ignore_index=True,
        )

    cnn = (
        predictions["JKX_CNN"]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    vit = (
        predictions["DayAligned_ViT"]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    if not cnn[KEYS + ["label"]].equals(vit[KEYS + ["label"]]):
        raise ValueError("CNN/ViT sample IDs differ")

    if not np.allclose(
        cnn.return_5d,
        vit.return_5d,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("CNN/ViT future returns differ")

    return predictions


def load_prices(raw_root, legacy):
    opens = {}
    closes = {}

    for ticker in sorted(legacy):
        path = raw_root / f"{ticker}.csv"

        if not path.is_file():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        needed = {"Date", "Open", "Close", "Adj Close"}

        if not needed.issubset(df.columns):
            raise ValueError(f"Missing price columns: {path}")

        df["Date"] = pd.to_datetime(df["Date"], errors="raise")

        df = df[
            (df.Date >= "2010-01-01")
            & (df.Date < "2022-01-01")
        ].copy()

        if df.Date.duplicated().any():
            raise ValueError(f"Duplicate dates: {path}")

        df = df.set_index("Date").sort_index()

        adjustment = df["Adj Close"] / df["Close"]

        opens[ticker] = df["Open"] * adjustment
        closes[ticker] = df["Adj Close"]

    adj_open = pd.DataFrame(opens).sort_index()
    adj_close = pd.DataFrame(closes).sort_index()

    # Common trading calendar. Never fill missing stock prices.
    valid_day = (
        ((adj_open > 0) & (adj_close > 0))
        .sum(axis=1) >= 80
    )

    calendar = adj_close.index[valid_day]

    adj_open = adj_open.reindex(calendar)
    adj_close = adj_close.reindex(calendar)

    if calendar.max() >= pd.Timestamp("2022-01-01"):
        raise ValueError("Holdout price access forbidden")

    returns = adj_close.pct_change(fill_method=None)

    return adj_open, returns, calendar


def rank_features(day):
    ranks = day.score.rank(method="average")
    n = len(day)

    return (ranks.to_numpy(float) - 0.5) / n - 0.5


def calibration_table(predictions):
    rows = []

    for date, day in predictions.groupby(
        "decision_date",
        sort=True,
    ):
        x = rank_features(day)

        y = day.return_5d.to_numpy(float)
        y = y - y.mean()

        rows.append({
            "decision_date": pd.Timestamp(date),
            "target_date": pd.Timestamp(
                day.target_date.iloc[0]
            ),
            "xy": float(np.dot(x, y)),
            "xx": float(np.dot(x, x)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("decision_date")
    )


def slope_at(calib, current_date, lookback_days, min_days):
    # Only previously matured future returns are eligible.
    history = calib[
        (calib.decision_date < current_date)
        & (calib.target_date <= current_date)
    ].tail(lookback_days)

    if len(history) < min_days:
        raise ValueError(
            f"Insufficient calibration history at {current_date}: "
            f"{len(history)} < {min_days}"
        )

    denominator = history.xx.sum()

    if denominator <= 0:
        raise ValueError("Degenerate calibration scores")

    slope = history.xy.sum() / denominator

    return float(np.clip(slope, -0.1, 0.1)), len(history)


def mvo_weights(mu, covariance, risk_aversion, max_weight):
    n = len(mu)

    if n * max_weight < 1 - 1e-10:
        raise ValueError("Infeasible weight constraint")

    initial = np.full(n, 1.0 / n)

    def objective(w):
        expected_return = float(mu @ w)
        variance = float(w @ covariance @ w)

        return (
            -expected_return
            + 0.5 * risk_aversion * variance
        )

    def gradient(w):
        return -mu + risk_aversion * (covariance @ w)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones(n),
        }],
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if (
        not result.success
        or abs(result.x.sum() - 1.0) > 1e-6
        or result.x.min() < -1e-7
        or result.x.max() > max_weight + 1e-6
    ):
        raise RuntimeError(
            f"MVO optimization failed: {result.message}"
        )

    return result.x


def rebalance_dates(predictions, offset):
    dates = sorted(
        d for d in predictions.decision_date.unique()
        if "2020-01-01" <= d < "2022-01-01"
    )

    schedules = []

    for year in (2020, 2021):
        annual_dates = [
            d for d in dates
            if d.startswith(str(year))
        ]

        schedules.extend(
            pd.Timestamp(d)
            for d in annual_dates[offset::5]
        )

    return schedules


def strategy_return(weights, returns):
    total = float(np.dot(weights, returns))

    if total <= -1:
        raise ValueError("Invalid portfolio return")

    final_weights = (
        weights * (1.0 + returns) / (1.0 + total)
    )

    return total, final_weights


def compute_cost_and_update(
    state,
    current_weights,
    start,
    end,
    ticker_returns,
    cost_bps,
):
    new_map = dict(current_weights)

    if state is None:
        turnover = sum(
            abs(w) for w in new_map.values()
        )

    else:
        previous_end, previous_weights = state

        if previous_end > start:
            raise ValueError("Overlapping trading periods")

        if previous_end < start:
            turnover = (
                sum(abs(w) for w in previous_weights.values())
                + sum(abs(w) for w in new_map.values())
            )

        else:
            tickers = set(previous_weights) | set(new_map)

            turnover = sum(
                abs(
                    new_map.get(t, 0.0)
                    - previous_weights.get(t, 0.0)
                )
                for t in tickers
            )

    tickers = list(new_map)

    weights = np.array([
        new_map[t] for t in tickers
    ])

    returns = np.array([
        ticker_returns[t] for t in tickers
    ])

    gross, final_weights = strategy_return(
        weights,
        returns,
    )

    net = gross - turnover * cost_bps / 10000.0

    new_state = (
        end,
        dict(zip(tickers, final_weights)),
    )

    return gross, net, turnover, new_state


def evaluate(args, predictions, adj_open, daily_returns, calendar):
    price_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    schedule = rebalance_dates(
        predictions["JKX_CNN"],
        args.offset,
    )

    for model in MODELS:
        dates = rebalance_dates(
            predictions[model],
            args.offset,
        )

        if dates != schedule:
            raise ValueError(
                "CNN/ViT decision dates differ"
            )

    calibrations = {
        model: calibration_table(predictions[model])
        for model in MODELS
    }

    lookups = {
        model: {
            pd.Timestamp(date): day.set_index("ticker")
            for date, day in predictions[model].groupby(
                "decision_date"
            )
        }
        for model in MODELS
    }

    records = []
    weights_data = []
    slopes = []
    state = {}

    for trade_number, date in enumerate(schedule, 1):
        year = date.year

        if date not in price_index:
            raise ValueError(
                f"Missing trading date: {date}"
            )

        loc = price_index[date]

        if loc + 6 >= len(calendar):
            continue

        entry = calendar[loc + 1]
        exit_date = calendar[loc + 6]

        if (
            exit_date.year != year
            or exit_date >= pd.Timestamp("2022-01-01")
        ):
            continue

        # Historical close-to-close returns known at decision close.
        history = (
            daily_returns.loc[:date]
            .tail(args.cov_lookback)
        )

        if len(history) != args.cov_lookback:
            raise ValueError(
                f"Insufficient price history: {date}"
            )

        eligible = history.notna().all(axis=0)

        available_tickers = eligible[
            eligible
        ].index.tolist()

        if len(available_tickers) < 80:
            raise ValueError(
                f"Too few eligible assets at {date}: "
                f"{len(available_tickers)}"
            )

        allocations = {
            "EW100": dict.fromkeys(
                available_tickers,
                1.0 / len(available_tickers),
            )
        }

        for model in MODELS:
            day = lookups[model][date].copy()

            slope, calibration_days = slope_at(
                calibrations[model],
                date,
                args.calib_lookback,
                args.min_calib_days,
            )

            day["rank_x"] = rank_features(day)
            day["mu_5d"] = slope * day["rank_x"]

            candidates = day.loc[
                day.index.intersection(available_tickers)
            ]

            # Signed historical calibration may reverse the score.
            selected = candidates.sort_values(
                ["mu_5d", "score"],
                ascending=[False, False],
                kind="stable",
            ).head(args.top_k)

            if len(selected) != args.top_k:
                raise ValueError(
                    "Insufficient portfolio candidates"
                )

            tickers = selected.index.tolist()

            # 126 historical daily returns only.
            covariance = (
                LedoitWolf(store_precision=False)
                .fit(history[tickers].to_numpy(float))
                .covariance_ * 5.0
            )

            mu = selected.mu_5d.to_numpy(float)

            equal_weights = np.full(
                args.top_k,
                1.0 / args.top_k,
            )

            minvar_weights = mvo_weights(
                np.zeros_like(mu),
                covariance,
                1.0,
                args.max_weight,
            )

            optimized_weights = mvo_weights(
                mu,
                covariance,
                args.risk_aversion,
                args.max_weight,
            )

            slopes.append({
                "date": date.date().isoformat(),
                "year": year,
                "model": model,
                "slope": slope,
                "calibration_days": calibration_days,
            })

            strategies = (
                ("Top20_EW", equal_weights),
                ("Top20_MinVar", minvar_weights),
                ("Top20_MVO", optimized_weights),
            )

            for strategy, weights in strategies:
                name = f"{model}_{strategy}"

                allocations[name] = dict(
                    zip(tickers, weights)
                )

                for ticker, weight in zip(tickers, weights):
                    weights_data.append({
                        "year": year,
                        "date": date.date().isoformat(),
                        "strategy": name,
                        "ticker": ticker,
                        "weight": float(weight),
                        "mu_5d": float(
                            selected.loc[ticker, "mu_5d"]
                        ),
                    })

        # Realized future prices are used only after
        # allocations have been finalized.
        needed = set().union(
            *(set(a) for a in allocations.values())
        )

        ordered = sorted(needed)

        entry_prices = adj_open.loc[entry, ordered]
        exit_prices = adj_open.loc[exit_date, ordered]

        if (
            entry_prices.isna().any()
            or exit_prices.isna().any()
            or (entry_prices <= 0).any()
            or (exit_prices <= 0).any()
        ):
            raise ValueError(
                f"Unexecutable stock at {date.date()}; "
                "do not exclude it using future data"
            )

        asset_returns = (
            exit_prices / entry_prices - 1.0
        ).to_dict()

        for name, allocation in allocations.items():
            key = (year, name)

            gross, net, turnover, state[key] = (
                compute_cost_and_update(
                    state.get(key),
                    allocation,
                    entry,
                    exit_date,
                    asset_returns,
                    args.cost_bps,
                )
            )

            records.append({
                "year": year,
                "date": date.date().isoformat(),
                "entry_open": entry.date().isoformat(),
                "exit_open": exit_date.date().isoformat(),
                "strategy": name,
                "n_positions": len(allocation),
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
            })

        if trade_number == 1 or trade_number % 10 == 0:
            print(
                f"[REBAL] {trade_number}/{len(schedule)} "
                f"{date.date()} "
                f"eligible={len(available_tickers)}",
                flush=True,
            )

    if not records:
        raise RuntimeError("No evaluated trading periods")

    results = pd.DataFrame(records)

    # Final liquidation cost at the end of each evaluation year.
    if args.cost_bps:
        for (_, _), group in results.groupby(
            ["year", "strategy"]
        ):
            last = group.index[-1]

            results.loc[last, "net_return"] -= (
                args.cost_bps / 10000.0
            )

            results.loc[last, "turnover"] += 1.0

    return (
        results,
        pd.DataFrame(weights_data),
        pd.DataFrame(slopes),
    )


def summarize(results):
    rows = []
    annualization = 252.0 / 5.0

    annual = [
        (str(year), strategy, group)
        for (year, strategy), group in results.groupby(
            ["year", "strategy"],
            sort=True,
        )
    ]

    combined = [
        (
            "ALL",
            strategy,
            group.sort_values(["year", "entry_open"]),
        )
        for strategy, group in results.groupby(
            "strategy",
            sort=True,
        )
    ]

    for period, strategy, group in annual + combined:
        returns = group.net_return.to_numpy(float)
        gross = group.gross_return.to_numpy(float)

        volatility = float(
            np.std(returns, ddof=1)
        )

        sharpe = (
            float(
                np.mean(returns)
                / volatility
                * np.sqrt(annualization)
            )
            if volatility > 0
            else np.nan
        )

        wealth = np.r_[
            1.0,
            np.cumprod(1.0 + returns),
        ]

        drawdown = (
            wealth / np.maximum.accumulate(wealth) - 1.0
        )

        rows.append({
            "period": period,
            "strategy": strategy,
            "periods": len(group),
            "cumulative_net": float(wealth[-1] - 1.0),
            "cumulative_gross": float(
                np.prod(1.0 + gross) - 1.0
            ),
            "sharpe_net": sharpe,
            "annualized_vol_net": (
                volatility * np.sqrt(annualization)
            ),
            "max_drawdown_net": float(drawdown.min()),
            "mean_turnover": float(
                group.turnover.mean()
            ),
        })

    return pd.DataFrame(rows)


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
        default=Path("outputs/exp200_mvo_compare"),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
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

    parser.add_argument(
        "--offset",
        type=int,
        choices=range(5),
        default=0,
    )

    args = parser.parse_args()

    if (
        len(args.seeds) != 3
        or len(set(args.seeds)) != 3
        or args.top_k < 10
        or args.max_weight <= 0
        or args.top_k * args.max_weight < 1 - 1e-8
        or args.cov_lookback < 10
        or args.min_calib_days < 1
        or args.calib_lookback < args.min_calib_days
        or args.risk_aversion <= 0
        or args.cost_bps < 0
    ):
        parser.error("Invalid strategy parameters or seeds")

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
        raise ValueError("Invalid 200-stock manifest")

    legacy = set(
        manifest.loc[
            manifest.group == "legacy",
            "ticker",
        ]
    )

    if len(legacy) != 100:
        raise ValueError("Expected exactly 100 legacy stocks")

    predictions = load_ensembles(args, legacy)

    adj_open, daily_returns, calendar = load_prices(
        args.raw,
        legacy,
    )

    print(
        "[DATA] Common100 | CNN/ViT 3-Seed ensembles | "
        "2019 calibration | 2020-2021 evaluation",
        flush=True,
    )

    print(
        f"[CONFIG] top_k={args.top_k} "
        f"max_weight={args.max_weight} "
        f"risk_aversion={args.risk_aversion} "
        f"cov_lookback={args.cov_lookback} "
        f"cost={args.cost_bps}bps/side "
        f"offset={args.offset}",
        flush=True,
    )

    results, weights, slopes = evaluate(
        args,
        predictions,
        adj_open,
        daily_returns,
        calendar,
    )

    summary = summarize(results)

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.out / "period_returns.csv",
        index=False,
    )

    weights.to_csv(
        args.out / "weights.csv",
        index=False,
    )

    slopes.to_csv(
        args.out / "calibration_slopes.csv",
        index=False,
    )

    summary.to_csv(
        args.out / "summary.csv",
        index=False,
    )

    focus = summary[
        summary.strategy.isin([
            "EW100",
            "JKX_CNN_Top20_MVO",
            "DayAligned_ViT_Top20_MVO",
        ])
    ]

    print("\n[MVO RESULTS]", flush=True)

    print(
        focus.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    benchmarks = summary[
        summary.strategy.str.contains(
            "Top20_EW|Top20_MinVar"
        )
    ]

    print("\n[TOP20 BENCHMARKS]", flush=True)

    print(
        benchmarks.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    print("\n[CALIBRATION]", flush=True)

    print(
        slopes.groupby(["year", "model"])
        .slope.agg(["mean", "min", "max"])
        .to_string(
            float_format=lambda x: f"{x:+.6f}"
        ),
        flush=True,
    )

    print(f"\n[SAVED] {args.out}", flush=True)
    print("[HOLDOUT] 2022-2025 untouched")


if __name__ == "__main__":
    main()
