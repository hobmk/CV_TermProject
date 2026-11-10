
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scripts.numeric_baseline import (
    make_features,
    compute_metrics,
)


def main():
    samples = pd.read_csv(
        "data/processed/samples.csv",
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
        },
    )

    # No test data
    samples = samples[
        samples["split"].isin(["train", "val"])
    ].reset_index(drop=True)

    print("[STEP 1] Extracting features...")

    features = make_features(
        samples,
        Path("data/raw"),
    )

    train_mask = (samples["split"] == "train").to_numpy()
    val_mask = (samples["split"] == "val").to_numpy()

    x_train = features.loc[train_mask]
    x_val = features.loc[val_mask]

    y_train = samples.loc[
        train_mask, "label"
    ].to_numpy(dtype=int)

    y_val = samples.loc[
        val_mask, "label"
    ].to_numpy(dtype=int)

    print(f"[TRAIN] {len(y_train):,}")
    print(f"[VAL] {len(y_val):,}")

    configurations = {
        "hgb_shallow": {
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 300,
            "l2_regularization": 10.0,
        },
        "hgb_medium": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 150,
            "l2_regularization": 5.0,
        },
    }

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, config in configurations.items():
        print(f"\n[TRAINING] {name}", flush=True)

        model = HistGradientBoostingClassifier(
            **config,
            early_stopping=False,
            random_state=42,
        )

        model.fit(x_train, y_train)

        train_probs = model.predict_proba(
            x_train
        )[:, 1]

        val_probs = model.predict_proba(
            x_val
        )[:, 1]

        train_metrics = compute_metrics(
            y_train, train_probs
        )

        val_metrics = compute_metrics(
            y_val, val_probs
        )

        print(
            f"Train AUROC : {train_metrics['auroc']:.4f}"
        )
        print(
            f"Val AUROC   : {val_metrics['auroc']:.4f}"
        )
        print(
            f"Val Bal Acc : {val_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )
        print(
            f"Pred Up Rate: {val_metrics['predicted_up_ratio']:.4f}"
        )
        print(
            f"Confusion   : {val_metrics['confusion_matrix']}"
        )

        predictions = samples.loc[
            val_mask,
            ["ticker", "decision_date", "label"],
        ].copy()

        predictions["prob_up"] = val_probs
        predictions["predicted_label"] = (
            val_probs >= 0.5
        ).astype(int)

        predictions.to_csv(
            output_dir / f"{name}_val_predictions.csv",
            index=False,
        )

        ticker_aucs = {}

        for ticker, part in predictions.groupby("ticker"):
            if part["label"].nunique() == 2:
                ticker_aucs[ticker] = float(
                    roc_auc_score(
                        part["label"],
                        part["prob_up"],
                    )
                )

        print(
            "[Per-Ticker AUROC] "
            f"Median={np.median(list(ticker_aucs.values())):.4f}"
        )

        results[name] = {
            "config": config,
            "train": train_metrics,
            "val": val_metrics,
            "ticker_auroc": ticker_aucs,
        }

    output = output_dir / "numeric_nonlinear_summary.json"

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\n[SAVED] {output}")
    print("[NOTE] Test set remains untouched")


if __name__ == "__main__":
    main()
