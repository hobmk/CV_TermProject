
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("outputs/metrics")

MODELS = {
    "CNN": "cnn_30stocks_seed42_val_predictions.csv",
    "Standard_ViT": "vit_p32_seed42_val_predictions.csv",
    "Day_Optuna": (
        "optuna_day_aligned_vit_e15_seed42_"
        "trial3_val_predictions.csv"
    ),
    "Linear_Best": (
        "day_tokenizer_linear_center_seed42_best_"
        "val_predictions.csv"
    ),
    "LocalCNN_Best": (
        "day_tokenizer_local_cnn_center_seed42_best_"
        "val_predictions.csv"
    ),
    "Linear_Last": (
        "day_tokenizer_linear_center_seed42_last_"
        "val_predictions.csv"
    ),
    "LocalCNN_Last": (
        "day_tokenizer_local_cnn_center_seed42_last_"
        "val_predictions.csv"
    ),
    "LogReg": "numeric_logreg_val_predictions.csv",
    "HGB_Medium": "hgb_medium_val_predictions.csv",
}


def main():
    samples = pd.read_csv(
        "data/processed/samples.csv",
        dtype={
            "ticker": str,
            "split": str,
            "decision_date": str,
        },
    )

    reference = samples.loc[
        samples["split"] == "val",
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ],
    ].copy()

    date_rows = []
    summary_rows = []

    for model_name, filename in MODELS.items():
        path = ROOT / filename

        if not path.exists():
            print(f"[SKIP] {path}")
            continue

        predictions = pd.read_csv(
            path,
            dtype={
                "ticker": str,
                "decision_date": str,
            },
        )

        predictions = predictions.merge(
            reference[
                ["ticker", "decision_date", "return_5d"]
            ],
            on=["ticker", "decision_date"],
            how="inner",
            validate="one_to_one",
        )

        if len(predictions) != len(reference):
            raise ValueError(
                f"{model_name}: missing validation predictions"
            )

        if not np.array_equal(
            predictions["label"].to_numpy(),
            (predictions["return_5d"] > 0)
            .astype(int).to_numpy(),
        ):
            raise ValueError(
                f"{model_name}: label/return mismatch"
            )

        predictions["year"] = (
            predictions["decision_date"].str[:4]
        )

        daily = []

        for date, part in predictions.groupby(
            "decision_date", sort=True
        ):
            y = part["label"].to_numpy()
            p = part["prob_up"].to_numpy()

            returns = part["return_5d"]

            ic = np.nan
            daily_auc = np.nan

            if (
                part["prob_up"].nunique() > 1
                and returns.nunique() > 1
            ):
                ic = part["prob_up"].corr(
                    returns, method="spearman"
                )

            if len(np.unique(y)) == 2:
                daily_auc = roc_auc_score(y, p)

            daily.append({
                "model": model_name,
                "date": date,
                "year": date[:4],
                "n": len(part),
                "rank_ic": ic,
                "daily_auc": daily_auc,
                "prob_spread": float(
                    p.max() - p.min()
                ),
            })

        daily_df = pd.DataFrame(daily)
        date_rows.append(daily_df)

        for year in ["ALL", "2022", "2023"]:
            if year == "ALL":
                group = predictions
                dg = daily_df
            else:
                group = predictions[
                    predictions["year"] == year
                ]
                dg = daily_df[
                    daily_df["year"] == year
                ]

            if group.empty:
                continue

            summary_rows.append({
                "model": model_name,
                "year": year,
                "pooled_auc": roc_auc_score(
                    group["label"],
                    group["prob_up"],
                ),
                "mean_daily_auc": (
                    dg["daily_auc"].mean()
                ),
                "mean_rank_ic": (
                    dg["rank_ic"].mean()
                ),
                "median_rank_ic": (
                    dg["rank_ic"].median()
                ),
                "positive_ic_ratio": (
                    (dg.loc[
                        dg["rank_ic"].notna(),
                        "rank_ic",
                    ] > 0).mean()
                ),
                "median_daily_prob_spread": (
                    dg["prob_spread"].median()
                ),
                "valid_ic_days": int(
                    dg["rank_ic"].notna().sum()
                ),
            })

    if not summary_rows:
        raise RuntimeError(
            "No prediction files were found"
        )

    summary = pd.DataFrame(summary_rows)

    daily_output = (
        ROOT / "daily_signal_by_date.csv"
    )

    summary_output = (
        ROOT / "daily_signal_summary.csv"
    )

    pd.concat(
        date_rows, ignore_index=True
    ).to_csv(daily_output, index=False)

    summary.to_csv(
        summary_output, index=False
    )

    print("\n[DAILY SIGNAL SUMMARY]")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print(f"\n[SAVED] {summary_output}")
    print(f"[SAVED] {daily_output}")
    print("[NOTE] No model training or test evaluation")


if __name__ == "__main__":
    main()
