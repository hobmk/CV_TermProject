
"""Point-in-time MVO comparison of JKX CNN and Day-Aligned ViT.

2017 out-of-time predictions provide initial calibration.
2018-2021 predictions are used for portfolio evaluation.

Signals are observed at the decision-day CLOSE.
Trades enter at the next trading day's adjusted OPEN.
Trades exit five trading sessions later at adjusted OPEN.

Calibration uses the same open-to-open return, and only
after the corresponding exit price is observable.

No 2022+ data are used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


MODELS = (
    "JKX_CNN",
    "DayAligned_ViT",
)

KEYS = [
    "ticker",
    "decision_date",
    "target_date",
]

REQUIRED = KEYS + [
    "group",
    "label",
    "return_5d",
    "prob_up",
]

YEARS = range(2017, 2022)
EVAL_YEARS = range(2018, 2022)


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

    missing = set(REQUIRED) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing {sorted(missing)}: {path}"
        )

    df = (
        df[REQUIRED]
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )

    if df.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError(
            f"Duplicated ticker/date: {path}"
        )

    if not df.decision_date.str.startswith(
        str(year)
    ).all():
        raise ValueError(
            f"Wrong year: {path}"
        )

    valid_dates = (
        (df.target_date > df.decision_date)
        & (df.target_date < "2022-01-01")
    )

    if not valid_dates.all():
        raise ValueError(
            f"Invalid future date: {path}"
        )

    for col in (
        "label",
        "return_5d",
        "prob_up",
    ):
        if not np.isfinite(
            df[col].to_numpy(float)
        ).all():
            raise ValueError(
                f"Nonfinite {col}: {path}"
            )

    if not df.prob_up.between(0, 1).all():
        raise ValueError(
            f"Invalid probabilities: {path}"
        )

    expected = (
        df.return_5d.to_numpy(float) > 0
    ).astype(int)

    if not np.array_equal(
        df.label.to_numpy(int),
        expected,
    ):
        raise ValueError(
            f"Return/label mismatch: {path}"
        )

    return df


def ensemble_preds(folder, year, seeds, legacy):
    reference = None
    probabilities = []

    for seed in seeds:
        path = (
            folder
            / "u200_start2010"
            / f"test{year}"
            / f"seed{seed}"
            / "predictions.csv"
        )

        frame = read_predictions(
            path,
            year,
        )

        if reference is None:
            reference = frame

        else:
            columns = KEYS + [
                "group",
                "label",
            ]

            if not reference[columns].equals(
                frame[columns]
            ):
                raise ValueError(
                    f"Seed sample mismatch: {path}"
                )

            if not np.allclose(
                reference.return_5d,
                frame.return_5d,
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    f"Seed return mismatch: {path}"
                )

        probabilities.append(
            frame.prob_up.to_numpy(float)
        )

    mask = (
        reference.ticker.isin(legacy)
        .to_numpy()
    )

    result = reference.loc[mask].copy()

    result["score"] = (
        np.stack(probabilities)
        .mean(axis=0)[mask]
    )

    counts = (
        result.groupby("decision_date")
        .ticker.nunique()
    )

    if (
        result.ticker.nunique() != 100
        or not counts.eq(100).all()
    ):
        raise ValueError(
            f"Not 100 stocks per day in {year}"
        )

    target_counts = (
        result.groupby("decision_date")
        .target_date.nunique()
    )

    if target_counts.ne(1).any():
        raise ValueError(
            f"Target-date mismatch in {year}"
        )

    return result.reset_index(drop=True)


def load_ensembles(args, legacy):
    result = {}

    for model, root in (
        (MODELS[0], args.cnn),
        (MODELS[1], args.vit),
    ):
        frames = [
            ensemble_preds(
                root,
                year,
                args.seeds,
                legacy,
            )
            for year in YEARS
        ]

        result[model] = pd.concat(
            frames,
            ignore_index=True,
        )

    cnn = (
        result[MODELS[0]]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    vit = (
        result[MODELS[1]]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    columns = KEYS + ["label"]

    if not cnn[columns].equals(
        vit[columns]
    ):
        raise ValueError(
            "CNN/ViT samples not aligned"
        )

    if not np.allclose(
        cnn.return_5d,
        vit.return_5d,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "CNN/ViT returns not aligned"
        )

    return result


def load_prices(raw_root, tickers):
    opens = {}
    closes = {}

    for ticker in sorted(tickers):
        path = raw_root / f"{ticker}.csv"

        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        needed = {
            "Date",
            "Open",
            "Close",
            "Adj Close",
        }

        if not needed.issubset(
            frame.columns
        ):
            raise ValueError(
                f"Missing price fields: {path}"
            )

        frame["Date"] = pd.to_datetime(
            frame.Date,
            errors="raise",
        )

        frame = frame.loc[
            (frame.Date >= "2010-01-01")
            & (frame.Date < "2022-01-01")
        ].copy()

        if frame.Date.duplicated().any():
            raise ValueError(
                f"Duplicate date: {ticker}"
            )

        frame = (
            frame.set_index("Date")
            .sort_index()
        )

        adjustment = (
            frame["Adj Close"]
            / frame["Close"]
        )

        opens[ticker] = (
            frame["Open"] * adjustment
        )

        closes[ticker] = (
            frame["Adj Close"]
        )

    adjusted_open = (
        pd.DataFrame(opens)
        .sort_index()
    )

    adjusted_close = (
        pd.DataFrame(closes)
        .sort_index()
    )

    valid = (
        (
            (adjusted_open > 0)
            & (adjusted_close > 0)
        )
        .sum(axis=1) >= 80
    )

    calendar = adjusted_close.index[valid]

    adjusted_open = (
        adjusted_open.reindex(calendar)
    )

    adjusted_close = (
        adjusted_close.reindex(calendar)
    )

    if (
        len(calendar) < 1000
        or calendar.max()
        >= pd.Timestamp("2022-01-01")
    ):
        raise ValueError(
            "Invalid price calendar "
            "or holdout contamination"
        )

    daily_returns = (
        adjusted_close.pct_change(
            fill_method=None
        )
    )

    return (
        adjusted_open,
        daily_returns,
        calendar,
    )


def rank_features(day):
    ranks = (
        day.score.rank(method="average")
        .to_numpy(float)
    )

    return (
        (ranks - 0.5) / len(day)
        - 0.5
    )


def calibration_table(
    predictions,
    adjusted_open,
    calendar,
):
    """Calculate matured open-to-open returns."""

    day_to_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    rows = []
    skipped = 0

    for date, day in predictions.groupby(
        "decision_date",
        sort=True,
    ):
        decision = pd.Timestamp(date)

        if decision not in day_to_index:
            raise ValueError(
                f"Missing calendar date: {date}"
            )

        index = day_to_index[decision]

        if index + 6 >= len(calendar):
            skipped += 1
            continue

        entry_date = calendar[index + 1]
        maturity_date = calendar[index + 6]

        tickers = day.ticker.tolist()

        entry = adjusted_open.loc[
            entry_date,
            tickers,
        ].to_numpy(float)

        exit_prices = adjusted_open.loc[
            maturity_date,
            tickers,
        ].to_numpy(float)

        valid = (
            np.isfinite(entry).all()
            and np.isfinite(exit_prices).all()
            and (entry > 0).all()
            and (exit_prices > 0).all()
        )

        if not valid:
            skipped += 1
            continue

        realized = (
            exit_prices / entry - 1.0
        )

        x = rank_features(day)

        y = (
            realized - realized.mean()
        )

        rows.append({
            "decision_date": decision,
            "maturity_date": maturity_date,
            "xy": float(np.dot(x, y)),
            "xx": float(np.dot(x, x)),
        })

    if not rows:
        raise ValueError(
            "Empty calibration history"
        )

    print(
        f"[CALIB DATA] "
        f"days={len(rows)} "
        f"skipped={skipped}",
        flush=True,
    )

    return (
        pd.DataFrame(rows)
        .sort_values("decision_date")
        .reset_index(drop=True)
    )


def slope_at(
    calibration,
    decision_date,
    lookback,
    minimum,
):
    history = calibration.loc[
        (
            calibration.decision_date
            < decision_date
        )
        & (
            calibration.maturity_date
            <= decision_date
        )
    ].tail(lookback)

    if len(history) < minimum:
        raise ValueError(
            f"Only {len(history)} matured "
            f"calibration dates at {decision_date}; "
            f"need {minimum}"
        )

    xx = float(history.xx.sum())

    if xx <= 0:
        raise ValueError(
            "Degenerate calibration denominator"
        )

    raw = float(
        history.xy.sum() / xx
    )

    return (
        float(np.clip(raw, -0.1, 0.1)),
        len(history),
    )


def mvo_weights(
    mu,
    covariance,
    risk_aversion,
    max_weight,
):
    n = len(mu)

    if n * max_weight < 1 - 1e-10:
        raise ValueError(
            "Infeasible weight bound"
        )

    initial = np.full(
        n,
        1.0 / n,
    )

    def objective(w):
        expected_return = float(mu @ w)

        variance = float(
            w @ covariance @ w
        )

        return (
            -expected_return
            + 0.5 * risk_aversion * variance
        )

    def gradient(w):
        return (
            -mu
            + risk_aversion * (covariance @ w)
        )

    optimum = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[
            (0.0, max_weight)
        ] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones(n),
        }],
        options={
            "maxiter": 500,
            "ftol": 1e-12,
        },
    )

    if (
        not optimum.success
        or abs(optimum.x.sum() - 1) > 1e-6
        or optimum.x.min() < -1e-7
        or optimum.x.max() > max_weight + 1e-6
    ):
        raise RuntimeError(
            f"MVO failed: {optimum.message}"
        )

    return optimum.x


def rebalance_dates(
    predictions,
    offset,
):
    candidates = sorted(
        d
        for d in predictions.decision_date.unique()
        if "2018-01-01" <= d < "2022-01-01"
    )

    schedule = []

    for year in EVAL_YEARS:
        year_days = [
            d
            for d in candidates
            if d.startswith(str(year))
        ]

        schedule.extend(
            pd.Timestamp(d)
            for d in year_days[offset::5]
        )

    return schedule


def strategy_return(
    weights,
    asset_returns,
):
    gross = float(
        np.dot(weights, asset_returns)
    )

    if gross <= -1:
        raise ValueError(
            "Portfolio gross return <= -100%"
        )

    end_weights = (
        weights
        * (1.0 + asset_returns)
        / (1.0 + gross)
    )

    return gross, end_weights


def compute_cost_and_update(
    state,
    allocation,
    entry,
    exit_date,
    asset_returns,
    cost_bps,
):
    new = dict(allocation)

    if state is None:
        turnover = sum(
            abs(w)
            for w in new.values()
        )

    else:
        last_exit, previous = state

        if last_exit > entry:
            raise ValueError(
                "Overlapping holding windows"
            )

        if last_exit < entry:
            turnover = (
                sum(abs(w) for w in previous.values())
                + sum(abs(w) for w in new.values())
            )

        else:
            tickers = (
                set(previous) | set(new)
            )

            turnover = sum(
                abs(
                    new.get(t, 0.0)
                    - previous.get(t, 0.0)
                )
                for t in tickers
            )

    tickers = list(new)

    weights = np.array(
        [new[t] for t in tickers],
        dtype=float,
    )

    returns = np.array(
        [asset_returns[t] for t in tickers],
        dtype=float,
    )

    gross, end_weights = strategy_return(
        weights,
        returns,
    )

    net = (
        gross
        - turnover * cost_bps / 10000.0
    )

    new_state = (
        exit_date,
        dict(zip(tickers, end_weights)),
    )

    return (
        gross,
        net,
        turnover,
        new_state,
    )


def evaluate(
    args,
    predictions,
    adjusted_open,
    daily_returns,
    calendar,
):
    day_index = {
        date: i
        for i, date in enumerate(calendar)
    }

    schedule = rebalance_dates(
        predictions[MODELS[0]],
        args.offset,
    )

    for model in MODELS[1:]:
        dates = rebalance_dates(
            predictions[model],
            args.offset,
        )

        if dates != schedule:
            raise ValueError(
                "Model rebalancing calendars differ"
            )

    calibration = {
        model: calibration_table(
            predictions[model],
            adjusted_open,
            calendar,
        )
        for model in MODELS
    }

    lookups = {
        model: {
            pd.Timestamp(date): day.set_index(
                "ticker"
            )
            for date, day in predictions[
                model
            ].groupby("decision_date")
        }
        for model in MODELS
    }

    records = []
    weights_rows = []
    slope_rows = []
    states = {}

    for count, decision in enumerate(
        schedule,
        1,
    ):
        year = decision.year

        if decision not in day_index:
            raise ValueError(
                f"Missing rebalance date: {decision}"
            )

        i = day_index[decision]

        if i + 6 >= len(calendar):
            continue

        entry = calendar[i + 1]
        exit_date = calendar[i + 6]

        if (
            exit_date.year != year
            or exit_date
            >= pd.Timestamp("2022-01-01")
        ):
            continue

        history = (
            daily_returns.loc[:decision]
            .tail(args.cov_lookback)
        )

        if len(history) != args.cov_lookback:
            raise ValueError(
                f"Insufficient covariance "
                f"history at {decision}"
            )

        eligible = (
            history.notna().all(axis=0)
        )

        available = (
            eligible.index[eligible]
            .tolist()
        )

        if len(available) < 80:
            raise ValueError(
                f"Fewer than 80 eligible "
                f"stocks at {decision}"
            )

        allocations = {
            "EW100": dict.fromkeys(
                available,
                1.0 / len(available),
            )
        }

        for model in MODELS:
            day = lookups[
                model
            ][decision].copy()

            slope, n_calib = slope_at(
                calibration[model],
                decision,
                args.calib_lookback,
                args.min_calib_days,
            )

            day["rank_x"] = rank_features(
                day
            )

            day["mu_5d"] = (
                slope * day.rank_x
            )

            candidates = day.loc[
                day.index.intersection(
                    available
                )
            ]

            selected = candidates.sort_values(
                ["mu_5d", "score"],
                ascending=[False, False],
                kind="stable",
            ).head(args.top_k)

            if len(selected) != args.top_k:
                raise ValueError(
                    f"Not enough candidates at "
                    f"{decision}"
                )

            names = selected.index.tolist()

            cov = (
                LedoitWolf(
                    store_precision=False
                )
                .fit(
                    history[names].to_numpy(
                        float
                    )
                )
                .covariance_
                * 5.0
            )

            mu = (
                selected.mu_5d
                .to_numpy(float)
            )

            strategies = {
                "Top20_EW": np.full(
                    args.top_k,
                    1.0 / args.top_k,
                ),

                "Top20_MinVar": mvo_weights(
                    np.zeros_like(mu),
                    cov,
                    1.0,
                    args.max_weight,
                ),

                "Top20_MVO": mvo_weights(
                    mu,
                    cov,
                    args.risk_aversion,
                    args.max_weight,
                ),
            }

            slope_rows.append({
                "year": year,
                "date": (
                    decision.date().isoformat()
                ),
                "model": model,
                "slope": slope,
                "calibration_days": n_calib,
            })

            for strategy, weights in (
                strategies.items()
            ):
                full_name = (
                    f"{model}_{strategy}"
                )

                allocations[full_name] = dict(
                    zip(names, weights)
                )

                for ticker, weight in zip(
                    names,
                    weights,
                ):
                    weights_rows.append({
                        "year": year,
                        "date": (
                            decision.date()
                            .isoformat()
                        ),
                        "strategy": full_name,
                        "ticker": ticker,
                        "weight": float(weight),
                        "mu_5d": float(
                            selected.loc[
                                ticker,
                                "mu_5d",
                            ]
                        ),
                    })

        needed = sorted(
            set().union(
                *(
                    set(a)
                    for a in allocations.values()
                )
            )
        )

        entry_px = adjusted_open.loc[
            entry,
            needed,
        ]

        exit_px = adjusted_open.loc[
            exit_date,
            needed,
        ]

        invalid_prices = (
            entry_px.isna().any()
            or exit_px.isna().any()
            or (entry_px <= 0).any()
            or (exit_px <= 0).any()
        )

        if invalid_prices:
            raise ValueError(
                f"Unexecutable trade on "
                f"{decision.date()}; "
                f"do not drop positions "
                f"using future prices"
            )

        realized = (
            exit_px / entry_px - 1.0
        ).to_dict()

        for name, allocation in (
            allocations.items()
        ):
            state_key = (
                year,
                name,
            )

            gross, net, turnover, states[
                state_key
            ] = compute_cost_and_update(
                states.get(state_key),
                allocation,
                entry,
                exit_date,
                realized,
                args.cost_bps,
            )

            records.append({
                "year": year,
                "date": (
                    decision.date().isoformat()
                ),
                "entry_open": (
                    entry.date().isoformat()
                ),
                "exit_open": (
                    exit_date.date().isoformat()
                ),
                "strategy": name,
                "n_positions": len(allocation),
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
            })

        if (
            count == 1
            or count % 10 == 0
        ):
            print(
                f"[REBAL] "
                f"{count}/{len(schedule)} "
                f"{decision.date()} "
                f"eligible={len(available)}",
                flush=True,
            )

    if not records:
        raise RuntimeError(
            "No portfolio rebalances"
        )

    results = pd.DataFrame(records)

    # Charge year-end liquidation cost.
    if args.cost_bps:
        grouped = results.groupby(
            ["year", "strategy"]
        )

        for _, group in grouped:
            last = group.index[-1]

            results.loc[
                last,
                "net_return",
            ] -= (
                args.cost_bps / 10000.0
            )

            results.loc[
                last,
                "turnover",
            ] += 1.0

    return (
        results,
        pd.DataFrame(weights_rows),
        pd.DataFrame(slope_rows),
    )


def summarize(results):
    rows = []

    annual = [
        (str(year), strategy, group)
        for (year, strategy), group
        in results.groupby(
            ["year", "strategy"]
        )
    ]

    combined = [
        (
            "ALL",
            strategy,
            group.sort_values(
                ["year", "entry_open"]
            ),
        )
        for strategy, group
        in results.groupby("strategy")
    ]

    for period, strategy, group in (
        annual + combined
    ):
        net = (
            group.net_return
            .to_numpy(float)
        )

        gross = (
            group.gross_return
            .to_numpy(float)
        )

        sd = (
            float(
                np.std(net, ddof=1)
            )
            if len(net) > 1
            else np.nan
        )

        wealth = np.r_[
            1.0,
            np.cumprod(1.0 + net),
        ]

        drawdown = (
            wealth
            / np.maximum.accumulate(wealth)
            - 1.0
        )

        sharpe = (
            float(
                net.mean()
                / sd
                * np.sqrt(252.0 / 5.0)
            )
            if sd > 0
            else np.nan
        )

        rows.append({
            "period": period,
            "strategy": strategy,
            "periods": len(group),
            "cumulative_net": float(
                wealth[-1] - 1.0
            ),
            "cumulative_gross": float(
                np.prod(1.0 + gross) - 1.0
            ),
            "sharpe_net": sharpe,
            "annualized_vol_net": float(
                sd * np.sqrt(252.0 / 5.0)
            ),
            "max_drawdown_net": float(
                drawdown.min()
            ),
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
        default=Path(
            "outputs/exp200_factorial"
        ),
    )

    parser.add_argument(
        "--vit",
        type=Path,
        default=Path(
            "outputs/exp200_day_vit"
        ),
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(
            "data/exp200/raw"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/exp200_mvo_2018_2021"
        ),
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

    invalid = (
        len(args.seeds) != 3
        or len(set(args.seeds)) != 3
        or args.top_k < 10
        or args.top_k > 100
        or args.max_weight <= 0
        or (
            args.top_k * args.max_weight
            < 1 - 1e-8
        )
        or args.cov_lookback < 10
        or args.min_calib_days < 1
        or args.calib_lookback
        < args.min_calib_days
        or args.risk_aversion <= 0
        or args.cost_bps < 0
    )

    if invalid:
        parser.error(
            "Invalid portfolio parameters"
        )

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
            "Expected locked "
            "200-stock manifest"
        )

    legacy = set(
        manifest.loc[
            manifest.group == "legacy",
            "ticker",
        ]
    )

    if len(legacy) != 100:
        raise ValueError(
            "Expected 100 common stocks"
        )

    predictions = load_ensembles(
        args,
        legacy,
    )

    adjusted_open, returns, calendar = (
        load_prices(
            args.raw,
            legacy,
        )
    )

    print(
        "[DATA] Common100 | "
        "CNN/ViT 3-seed ensembles | "
        "2017 initial calibration | "
        "2018-2021 evaluation",
        flush=True,
    )

    print(
        f"[CONFIG] "
        f"top_k={args.top_k} "
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
        adjusted_open,
        returns,
        calendar,
    )

    summary = summarize(results)

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = vars(args).copy()

    config.update({
        "initial_calibration_year": 2017,
        "evaluation_years": list(EVAL_YEARS),
        "calibration_target":
            "adjusted_open_t1_to_t6",
        "year_end_liquidation": True,
    })

    config_path = (
        args.out / "config.json"
    )

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
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

    print(
        "\n[MVO RESULTS]",
        flush=True,
    )

    print(
        focus.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    benchmark = summary[
        summary.strategy.str.contains(
            "Top20_EW|Top20_MinVar"
        )
    ]

    print(
        "\n[TOP20 BENCHMARKS]",
        flush=True,
    )

    print(
        benchmark.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    print(
        "\n[CALIBRATION]",
        flush=True,
    )

    print(
        slopes.groupby(
            ["year", "model"]
        )
        .slope.agg([
            "mean",
            "min",
            "max",
        ])
        .to_string(
            float_format=lambda x: f"{x:+.6f}"
        ),
        flush=True,
    )

    print(
        f"\n[SAVED] {args.out}",
        flush=True,
    )

    print(
        "[LIMITATIONS] Survivor-only stocks; "
        "yearly trading gaps; "
        "MDD based on 5-session "
        "portfolio returns",
        flush=True,
    )

    print(
        "[HOLDOUT] 2022-2025 untouched",
        flush=True,
    )


if __name__ == "__main__":
    main()
