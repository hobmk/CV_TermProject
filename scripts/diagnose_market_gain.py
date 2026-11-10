
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.compare_day_market_context import (
    load_dev_split,
    build_peer_context,
)


def load_predictions(path, reference):
    predictions = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "decision_date": str,
        },
    )

    required = {
        "ticker", "decision_date", "prob_up"
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"{path}: missing {sorted(missing)}"
        )

    if predictions.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError(
            f"Duplicated predictions: {path}"
        )

    merged = reference.merge(
        predictions,
        on=["ticker", "decision_date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )

    if (
        len(merged) != len(reference)
        or merged["prob_up"].isna().any()
    ):
        raise ValueError(
            f"Missing predictions: {path}"
        )

    if "label_prediction" in merged.columns:
        if not np.array_equal(
            merged["label"].to_numpy(),
            merged["label_prediction"].to_numpy(),
        ):
            raise ValueError(
                f"Label mismatch: {path}"
            )

    return merged[
        [
            "ticker", "decision_date",
            "label", "return_5d", "prob_up"
        ]
    ].copy()


def analyze(name, data):
    data = data.copy()

    data["date_mean_prob"] = (
        data.groupby("decision_date")[
            "prob_up"
        ].transform("mean")
    )

    pooled_auc = roc_auc_score(
        data["label"],
        data["prob_up"],
    )

    # Each stock gets the date's mean prediction.
    # Measures how much date-level variation
    # alone can discriminate up/down outcomes.
    date_only_auc = roc_auc_score(
        data["label"],
        data["date_mean_prob"],
    )

    daily_rows = []

    for date, group in data.groupby(
        "decision_date", sort=True
    ):
        daily_auc = float("nan")
        rank_ic = float("nan")

        if group["label"].nunique() == 2:
            daily_auc = roc_auc_score(
                group["label"],
                group["prob_up"],
            )

        if (
            group["prob_up"].nunique() > 1
            and group["return_5d"].nunique() > 1
        ):
            rank_ic = group["prob_up"].corr(
                group["return_5d"],
                method="spearman",
            )

        daily_rows.append({
            "model": name,
            "date": date,
            "daily_auc": daily_auc,
            "rank_ic": rank_ic,
            "up_rate": group["label"].mean(),
            "mean_prob": group["prob_up"].mean(),
            "within_day_prob_std": (
                group["prob_up"].std(ddof=0)
            ),
        })

    daily = pd.DataFrame(daily_rows)

    summary = {
        "model": name,
        "pooled_auc": float(pooled_auc),
        "date_only_auc": float(date_only_auc),
        "mean_daily_auc": float(
            daily["daily_auc"].mean()
        ),
        "mean_rank_ic": float(
            daily["rank_ic"].mean()
        ),
        "date_up_corr": float(
            daily["mean_prob"].corr(
                daily["up_rate"]
            )
        ),
        "median_within_day_std": float(
            daily["within_day_prob_std"].median()
        ),
        "valid_daily_auc": int(
            daily["daily_auc"].notna().sum()
        ),
    }

    return summary, daily


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics"),
    )

    args = parser.parse_args()

    train, dev = load_dev_split(
        Path("data/processed/samples.csv")
    )

    reference = dev[
        [
            "ticker", "decision_date",
            "label", "return_5d"
        ]
    ].copy()

    summary_rows = []
    daily_frames = []

    for variant in ("baseline", "market"):
        path = args.metrics_dir / (
            f"day_market_{variant}_"
            f"seed{args.seed}_dev2021_"
            f"dev_predictions.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        data = load_predictions(
            path, reference
        )

        summary, daily = analyze(
            f"ViT_{variant}", data
        )

        summary_rows.append(summary)
        daily_frames.append(daily)

    # Train a lightweight market-context-only model.
    # Same features and temporal split as the ViT.
    print("\n[CONTEXT-ONLY LOGISTIC REGRESSION]")

    x_train, x_dev, _ = build_peer_context(
        train,
        dev,
        Path("data/raw"),
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
    )

    model.fit(
        x_train,
        train["label"].to_numpy(),
    )

    train_probability = model.predict_proba(
        x_train
    )[:, 1]

    dev_probability = model.predict_proba(
        x_dev
    )[:, 1]

    print(
        "[CONTEXT ONLY] Train AUROC="
        f"{roc_auc_score(train['label'], train_probability):.4f}"
    )

    context_data = reference.copy()
    context_data["prob_up"] = dev_probability

    summary, daily = analyze(
        "Context_Only_LogReg",
        context_data,
    )

    summary_rows.append(summary)
    daily_frames.append(daily)

    result = pd.DataFrame(summary_rows)

    print("\n[MARKET CONTEXT DIAGNOSTIC]")

    print(
        result.to_string(
            index=False,
            float_format=lambda v: f"{v:+.5f}",
        )
    )

    args.metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    summary_path = (
        args.metrics_dir
        / "market_gain_diagnostic_summary.csv"
    )

    daily_path = (
        args.metrics_dir
        / "market_gain_diagnostic_daily.csv"
    )

    result.to_csv(
        summary_path,
        index=False,
    )

    pd.concat(
        daily_frames,
        ignore_index=True,
    ).to_csv(
        daily_path,
        index=False,
    )

    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {daily_path}")
    print(
        "[NOTE] 2021 Dev only. "
        "No 2022-2025 evaluation."
    )


if __name__ == "__main__":
    main()
