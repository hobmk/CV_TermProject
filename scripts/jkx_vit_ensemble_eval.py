
"""Evaluate saved 3-seed JKX CNN and Day-Aligned ViT predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEYS = ["ticker", "decision_date", "target_date"]
DATA_COLS = KEYS + ["group", "label", "return_5d"]
MODELS = {"JKX_CNN": "cnn", "DayAligned_ViT": "vit"}


def read_one(path, year):
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "group": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    required = set(DATA_COLS + ["prob_up"])
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns {sorted(required - set(df.columns))}: {path}"
        )

    if df.duplicated(["ticker", "decision_date"]).any():
        raise ValueError(f"Duplicate predictions: {path}")

    if not (df.decision_date.str[:4] == str(year)).all():
        raise ValueError(f"Unexpected test year: {path}")

    if not (
        (df.target_date > df.decision_date)
        & (df.target_date < "2022-01-01")
    ).all():
        raise ValueError(f"Invalid target dates: {path}")

    for col in ("prob_up", "return_5d", "label"):
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite {col}: {path}")

    if not df.prob_up.between(0, 1).all():
        raise ValueError(f"Invalid probabilities: {path}")

    expected = (df.return_5d.to_numpy(float) > 0).astype(int)

    if not np.array_equal(expected, df.label.to_numpy(int)):
        raise ValueError(f"Label mismatch: {path}")

    return (
        df.sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )


def aligned(reference, candidate, path):
    if not reference[KEYS].equals(candidate[KEYS]):
        raise ValueError(f"Sample IDs differ: {path}")

    if not reference[["group", "label"]].equals(
        candidate[["group", "label"]]
    ):
        raise ValueError(f"Label/group mismatch: {path}")

    if not np.allclose(
        reference.return_5d,
        candidate.return_5d,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"Future returns differ: {path}")


def make_ensemble(root, year, seeds, reference):
    arrays = []
    singles = {}

    for seed in seeds:
        path = (
            root
            / "u200_start2010"
            / f"test{year}"
            / f"seed{seed}"
            / "predictions.csv"
        )

        frame = read_one(path, year)
        aligned(reference, frame, path)

        scores = frame.prob_up.to_numpy(float)
        arrays.append(scores)
        singles[seed] = scores

    ensemble = np.stack(arrays).mean(axis=0)

    return ensemble, singles


def daily_metrics(frame):
    auc = float(
        roc_auc_score(
            frame.label.to_numpy(int),
            frame.prob_up,
        )
    )

    rank_ics = []
    daily_aucs = []
    top_excesses = []
    spreads = []

    for date, day in frame.groupby("decision_date", sort=True):
        if len(day) < 20:
            raise ValueError(f"Too few stocks: {date}")

        if day.target_date.nunique() != 1:
            raise ValueError(f"Target-date mismatch: {date}")

        ic = day.prob_up.corr(
            day.return_5d,
            method="spearman",
        )

        if np.isfinite(ic):
            rank_ics.append(float(ic))

        if day.label.nunique() == 2:
            daily_aucs.append(
                float(roc_auc_score(day.label, day.prob_up))
            )

        ranked = day.sort_values(
            ["prob_up", "ticker"],
            kind="stable",
        )

        top = ranked.tail(10).return_5d.mean()
        bottom = ranked.head(10).return_5d.mean()

        top_excesses.append(
            float(top - day.return_5d.mean())
        )
        spreads.append(float(top - bottom))

    if not rank_ics or not daily_aucs:
        raise ValueError("Insufficient evaluation observations")

    return {
        "auc": auc,
        "daily_auc": float(np.mean(daily_aucs)),
        "rank_ic": float(np.mean(rank_ics)),
        "top10_excess_5d": float(np.mean(top_excesses)),
        "top10_bottom10_spread_5d": float(np.mean(spreads)),
        "days": int(frame.decision_date.nunique()),
        "samples": int(len(frame)),
    }


def weekly_deciles(frame, year, model, evaluation, offset):
    dates = sorted(frame.decision_date.unique())
    records = []

    for date in dates[offset::5]:
        day = frame.loc[frame.decision_date == date]

        ordered = (
            day.sort_values(["prob_up", "ticker"], kind="stable")
            .reset_index(drop=True)
        )

        bins = np.array_split(
            np.arange(len(ordered)),
            10,
        )

        if any(len(b) == 0 for b in bins):
            raise ValueError(f"Insufficient stocks: {date}")

        rec = {
            "year": year,
            "model": model,
            "evaluation": evaluation,
            "offset": offset,
            "decision_date": date,
            "target_date": str(ordered.target_date.iloc[0]),
            "n_stocks": len(ordered),
        }

        for d, idx in enumerate(bins, 1):
            rec[f"d{d}"] = float(
                ordered.iloc[idx].return_5d.mean()
            )

        rec["hl_5d"] = rec["d10"] - rec["d1"]
        rec["top_decile_excess_5d"] = (
            rec["d10"] - float(day.return_5d.mean())
        )

        records.append(rec)

    out = pd.DataFrame(records)

    if len(out) < 3:
        raise ValueError("Too few rebalances")

    if not (
        out.target_date.iloc[:-1].to_numpy()
        <= out.decision_date.iloc[1:].to_numpy()
    ).all():
        raise ValueError("Overlapping holding periods")

    return out


def weekly_summary(frame):
    returns = frame.hl_5d.to_numpy(float)
    std = returns.std(ddof=1)

    sharpe = (
        float(returns.mean() / std * np.sqrt(252 / 5))
        if std > 0
        else float("nan")
    )

    return {
        "rebalances": len(returns),
        "mean_5d_hl": float(returns.mean()),
        "annualized_sharpe_gross": sharpe,
        "d1_mean": float(frame.d1.mean()),
        "d10_mean": float(frame.d10.mean()),
    }


def display(title, frame, columns):
    print(f"\n{title}", flush=True)

    print(
        frame[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cnn",
        type=Path,
        default=Path("outputs/exp200_factorial"),
    )
    p.add_argument(
        "--vit",
        type=Path,
        default=Path("outputs/exp200_day_vit"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/exp200/universe200.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/exp200_ensemble_paper_eval"),
    )
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021],
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    a = p.parse_args()

    if not set(a.years).issubset({2019, 2020, 2021}):
        p.error("Evaluation years must be 2019-2021")

    if len(set(a.seeds)) != len(a.seeds) or len(a.seeds) < 2:
        p.error("Provide at least two distinct seeds")

    manifest = pd.read_csv(
        a.manifest,
        dtype={"ticker": str, "group": str},
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
        raise ValueError("Invalid locked universe")

    metrics = []
    portfolio = []
    weekly_records = []

    for year in a.years:
        reference_path = (
            a.cnn
            / "u200_start2010"
            / f"test{year}"
            / f"seed{a.seeds[0]}"
            / "predictions.csv"
        )

        reference = read_one(reference_path, year)

        if set(reference.ticker.unique()) != set(manifest.ticker):
            raise ValueError(f"Universe mismatch in {year}")

        groups = dict(zip(manifest.ticker, manifest.group))

        if not reference.group.equals(
            reference.ticker.map(groups)
        ):
            raise ValueError("Universe group mismatch")

        for model, kind in MODELS.items():
            root = a.cnn if kind == "cnn" else a.vit

            ensemble, singles = make_ensemble(
                root,
                year,
                a.seeds,
                reference,
            )

            for evaluation in ("common100", "all200"):
                if evaluation == "common100":
                    mask = reference.ticker.isin(legacy).to_numpy()
                else:
                    mask = np.ones(len(reference), dtype=bool)

                predictions = [
                    (f"seed{seed}", scores)
                    for seed, scores in singles.items()
                ]
                predictions.append(("ensemble", ensemble))

                for prediction_name, scores in predictions:
                    frame = reference.loc[mask].copy()
                    frame["prob_up"] = scores[mask]

                    stats = daily_metrics(frame)

                    metrics.append({
                        "year": year,
                        "model": model,
                        "evaluation": evaluation,
                        "prediction": prediction_name,
                        **stats,
                    })

                frame = reference.loc[mask].copy()
                frame["prob_up"] = ensemble[mask]

                for offset in range(5):
                    weekly = weekly_deciles(
                        frame,
                        year,
                        model,
                        evaluation,
                        offset,
                    )

                    weekly_records.append(weekly)

                    portfolio.append({
                        "year": year,
                        "model": model,
                        "evaluation": evaluation,
                        "offset": offset,
                        **weekly_summary(weekly),
                    })

    metrics_df = pd.DataFrame(metrics)
    portfolio_df = pd.DataFrame(portfolio)
    weekly_df = pd.concat(weekly_records, ignore_index=True)

    a.out.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(
        a.out / "per_seed_and_ensemble_metrics.csv",
        index=False,
    )

    portfolio_df.to_csv(
        a.out / "weekly_portfolio_summary.csv",
        index=False,
    )

    weekly_df.to_csv(
        a.out / "weekly_decile_returns.csv",
        index=False,
    )

    common = metrics_df[
        (metrics_df.evaluation == "common100")
        & (metrics_df.prediction == "ensemble")
    ].copy()

    metric_cols = [
        "auc",
        "rank_ic",
        "top10_excess_5d",
        "top10_bottom10_spread_5d",
    ]

    display(
        "[ENSEMBLE] Common100, 3-Seed averaged predictions",
        common,
        ["year", "model"] + metric_cols,
    )

    averages = (
        common.groupby("model", as_index=False)[metric_cols]
        .mean()
    )

    display(
        "[3-YEAR MEAN] Ensemble performance",
        averages,
        ["model"] + metric_cols,
    )

    pair = metrics_df.pivot(
        index=["year", "evaluation", "prediction"],
        columns="model",
        values=metric_cols,
    )

    delta = pd.DataFrame(index=pair.index)

    for metric in metric_cols:
        delta[f"delta_{metric}"] = (
            pair[(metric, "DayAligned_ViT")]
            - pair[(metric, "JKX_CNN")]
        )

    delta = delta.reset_index()

    delta.to_csv(
        a.out / "paired_model_deltas.csv",
        index=False,
    )

    ens_delta = delta[
        (delta.evaluation == "common100")
        & (delta.prediction == "ensemble")
    ]

    display(
        "[ENSEMBLE DELTA] ViT minus JKX CNN",
        ens_delta,
        ["year"] + [
            c for c in ens_delta.columns
            if c.startswith("delta_")
        ],
    )

    fixed = portfolio_df[
        (portfolio_df.evaluation == "common100")
        & (portfolio_df.offset == 0)
    ]

    display(
        "[5D PORTFOLIO] Fixed offset=0, gross H-L",
        fixed,
        [
            "year",
            "model",
            "rebalances",
            "mean_5d_hl",
            "annualized_sharpe_gross",
            "d1_mean",
            "d10_mean",
        ],
    )

    sensitivity = (
        portfolio_df[
            portfolio_df.evaluation == "common100"
        ]
        .pivot(
            index=["year", "model"],
            columns="offset",
            values="mean_5d_hl",
        )
        .reset_index()
    )

    display(
        "[OFFSET SENSITIVITY] Five rebalance schedules",
        sensitivity,
        ["year", "model", 0, 1, 2, 3, 4],
    )

    print(f"\n[SAVED] {a.out}", flush=True)
    print(
        "[LIMITATIONS] Survivor-only universe; gross "
        "5-day returns; no trading costs or MVO."
    )
    print("[HOLDOUT] 2022-2025 untouched")


if __name__ == "__main__":
    main()
