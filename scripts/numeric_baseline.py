
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WINDOW = 20

FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_19d",
    "log_volatility_19d",
    "price_range_20d",
    "last_candle_body",
    "last_daily_range",
    "close_location_20d",
    "close_vs_sma20",
    "up_day_fraction",
    "last_volume_ratio",
    "volume_5_to_20_ratio",
]


def load_ticker(path):
    df = pd.read_csv(
        path, dtype={"Date": str}
    ).sort_values("Date").reset_index(drop=True)

    required = {
        "Date", "Open", "High",
        "Low", "Close", "Volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    if df["Date"].duplicated().any():
        raise ValueError(f"{path}: duplicate dates")

    prices = df[
        ["Open", "High", "Low", "Close"]
    ].to_numpy(dtype=np.float64)

    volumes = df["Volume"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{path}: invalid prices")

    if not np.isfinite(volumes).all() or (volumes < 0).any():
        raise ValueError(f"{path}: invalid volume")

    return {
        "prices": prices,
        "volumes": volumes,
        "dates": df["Date"].to_numpy(),
        "index": {
            date: i
            for i, date in enumerate(df["Date"])
        },
    }


def extract_features(prices, volumes):
    o, h, l, c = prices.T
    v = volumes

    daily_returns = c[1:] / c[:-1] - 1.0
    log_returns = np.diff(np.log(c))

    minimum_low = l.min()
    maximum_high = h.max()

    avg_volume = max(v.mean(), 1.0)

    return [
        c[-1] / c[-2] - 1.0,
        c[-1] / c[-6] - 1.0,
        c[-1] / c[-11] - 1.0,
        c[-1] / c[0] - 1.0,

        log_returns.std(ddof=1),

        (maximum_high - minimum_low) / c[0],

        (c[-1] - o[-1]) / o[-1],
        (h[-1] - l[-1]) / o[-1],

        (c[-1] - minimum_low)
        / max(maximum_high - minimum_low, 1e-12),

        c[-1] / c.mean() - 1.0,

        (daily_returns > 0).mean(),

        v[-1] / avg_volume,
        v[-5:].mean() / avg_volume,
    ]


def make_features(samples, raw_dir):
    cache = {}
    rows = []

    for sample in samples.itertuples(index=False):
        ticker = sample.ticker

        if ticker not in cache:
            path = raw_dir / f"{ticker}.csv"

            if not path.is_file():
                raise FileNotFoundError(path)

            cache[ticker] = load_ticker(path)

        stock = cache[ticker]

        idx = stock["index"].get(
            sample.decision_date
        )

        if idx is None or idx < WINDOW - 1:
            raise ValueError(
                f"{ticker}: invalid decision date "
                f"{sample.decision_date}"
            )

        start = idx - WINDOW + 1

        if stock["dates"][start] != sample.window_start:
            raise ValueError(
                f"{ticker}: window date mismatch "
                f"at {sample.decision_date}"
            )

        prices = stock["prices"][start:idx + 1]
        volumes = stock["volumes"][start:idx + 1]

        rows.append(
            extract_features(prices, volumes)
        )

    features = pd.DataFrame(
        rows, columns=FEATURE_NAMES
    )

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "Non-finite numerical features found"
        )

    return features


def compute_metrics(labels, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels, predictions
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(labels, probabilities)
        ),
        "predicted_up_ratio": float(
            predictions.mean()
        ),
        "probability_std": float(
            probabilities.std()
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.c <= 0:
        parser.error("--c must be positive")

    samples = pd.read_csv(
        args.samples,
        dtype={
            "ticker": str,
            "split": str,
            "decision_date": str,
            "window_start": str,
        },
    )

    required = {
        "ticker", "split", "decision_date",
        "window_start", "label",
    }

    missing = required - set(samples.columns)

    if missing:
        raise ValueError(
            f"Missing sample columns: {missing}"
        )

    # Only Train and Validation are used.
    samples = samples[
        samples["split"].isin(["train", "val"])
    ].reset_index(drop=True)

    print("[STEP 1] Extracting OHLCV features...")

    features = make_features(
        samples, args.raw_dir
    )

    train_mask = (
        samples["split"] == "train"
    ).to_numpy()

    val_mask = (
        samples["split"] == "val"
    ).to_numpy()

    labels = samples["label"].to_numpy(
        dtype=np.int64
    )

    x_train = features.loc[train_mask]
    x_val = features.loc[val_mask]

    y_train = labels[train_mask]
    y_val = labels[val_mask]

    if (
        len(y_train) == 0
        or len(y_val) == 0
        or len(np.unique(y_train)) != 2
        or len(np.unique(y_val)) != 2
    ):
        raise ValueError(
            "Train and Val must contain both classes"
        )

    print(
        f"[TRAIN] N={len(y_train):,} "
        f"| Positive={y_train.mean():.2%}"
    )

    print(
        f"[VAL] N={len(y_val):,} "
        f"| Positive={y_val.mean():.2%}"
    )

    print(
        f"[FEATURES] {len(FEATURE_NAMES)}"
    )

    print(
        "[CHECK] Historical features only; "
        "scaler fit on Train only"
    )

    args.out_dir.mkdir(
        parents=True, exist_ok=True
    )

    # --------------------------------------------------
    # Majority-class baseline
    # --------------------------------------------------
    baseline_probs = np.full(
        len(y_val),
        y_train.mean(),
        dtype=np.float64,
    )

    results = {
        "always_majority": compute_metrics(
            y_val, baseline_probs
        )
    }

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------
    settings = [
        ("logreg", None),
        ("logreg_balanced", "balanced"),
    ]

    print("\n[STEP 2] Training Logistic Regression...")

    for name, class_weight in settings:
        pipeline = Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=args.c,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=args.seed,
                ),
            ),
        ])

        pipeline.fit(
            x_train,
            y_train,
        )

        train_probs = pipeline.predict_proba(
            x_train
        )[:, 1]

        val_probs = pipeline.predict_proba(
            x_val
        )[:, 1]

        train_metrics = compute_metrics(
            y_train, train_probs
        )

        val_metrics = compute_metrics(
            y_val, val_probs
        )

        results[name] = {
            "train": train_metrics,
            "val": val_metrics,
        }

        predictions_df = samples.loc[
            val_mask,
            ["ticker", "decision_date", "label"],
        ].copy()

        predictions_df["prob_up"] = val_probs
        predictions_df["predicted_label"] = (
            val_probs >= 0.5
        ).astype(int)

        output_path = (
            args.out_dir
            / f"numeric_{name}_val_predictions.csv"
        )

        predictions_df.to_csv(
            output_path,
            index=False,
        )

        print(f"[SAVED] {output_path}")

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("[NUMERICAL BASELINE RESULTS]")
    print("=" * 80)

    for name, entry in results.items():
        val_result = (
            entry["val"]
            if "val" in entry
            else entry
        )

        print(
            f"\n[{name}]"
            f"\n  Val Accuracy : {val_result['accuracy']:.4f}"
            f"\n  Val Bal Acc  : {val_result['balanced_accuracy']:.4f}"
            f"\n  Val Macro F1 : {val_result['macro_f1']:.4f}"
            f"\n  Val AUROC    : {val_result['auroc']:.4f}"
            f"\n  Pred Up Ratio: {val_result['predicted_up_ratio']:.4f}"
            f"\n  Prob Std     : {val_result['probability_std']:.6f}"
        )

        print(
            f"  Confusion    : "
            f"{val_result['confusion_matrix']}"
        )

        if "train" in entry:
            print(
                f"  Train AUROC  : "
                f"{entry['train']['auroc']:.4f}"
            )

    summary_path = (
        args.out_dir / "numeric_baseline_summary.json"
    )

    summary = {
        "features": FEATURE_NAMES,
        "regularization_C": args.c,
        "results": results,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n[SAVED] {summary_path}")
    print("[NOTE] Test set was not evaluated")


if __name__ == "__main__":
    main()
