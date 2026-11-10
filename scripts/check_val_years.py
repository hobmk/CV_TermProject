
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


FILES = {
    "Logistic": "numeric_logreg_val_predictions.csv",
    "HGB Shallow": "hgb_shallow_val_predictions.csv",
    "HGB Medium": "hgb_medium_val_predictions.csv",
    "CNN": "cnn_30stocks_seed42_val_predictions.csv",
    "Standard ViT": "vit_p32_seed42_val_predictions.csv",
    "Day ViT Optuna": (
        "optuna_day_aligned_vit_e15_seed42_"
        "trial3_val_predictions.csv"
    ),
}


def main():
    root = Path("outputs/metrics")

    samples = pd.read_csv(
        "data/processed/samples.csv"
    )

    val = samples[samples["split"] == "val"].copy()
    val["year"] = pd.to_datetime(
        val["decision_date"]
    ).dt.year

    print("\n[VALIDATION LABEL DISTRIBUTION]")

    print(
        val.groupby("year")["label"]
        .agg(["size", "mean"])
        .to_string()
    )

    rows = []

    for model_name, filename in FILES.items():
        path = root / filename

        if not path.exists():
            print(f"[SKIPPED] {path}")
            continue

        df = pd.read_csv(path)

        df["year"] = pd.to_datetime(
            df["decision_date"]
        ).dt.year

        for year, part in df.groupby("year"):
            y = part["label"].to_numpy()
            p = part["prob_up"].to_numpy()
            pred = (p >= 0.5).astype(int)

            auc = (
                roc_auc_score(y, p)
                if len(set(y)) == 2
                else float("nan")
            )

            rows.append({
                "model": model_name,
                "year": year,
                "n": len(part),
                "positive_rate": y.mean(),
                "accuracy": accuracy_score(y, pred),
                "balanced_accuracy": (
                    balanced_accuracy_score(y, pred)
                ),
                "macro_f1": f1_score(
                    y, pred, average="macro",
                    zero_division=0,
                ),
                "auroc": auc,
                "pred_up_ratio": pred.mean(),
            })

    result = pd.DataFrame(rows)

    print("\n[MODEL PERFORMANCE BY YEAR]")

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output = root / "validation_year_comparison.csv"
    result.to_csv(output, index=False)

    print(f"\n[SAVED] {output}")


if __name__ == "__main__":
    main()
