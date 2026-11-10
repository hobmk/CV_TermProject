
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.numeric_sequence_baseline import (
    NumericMLP,
    create_sequences,
    evaluate,
    set_seed,
)


def load_pool(path):
    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    required = {
        "ticker", "split", "window_start",
        "decision_date", "target_date",
        "label", "return_5d",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    pool = df.loc[
        df["split"].eq("train")
    ].copy().reset_index(drop=True)

    if (
        pool.empty
        or pool.duplicated(
            ["ticker", "decision_date"]
        ).any()
    ):
        raise ValueError(
            "Empty or duplicated original train samples"
        )

    pool["row_id"] = np.arange(len(pool))

    return pool


def split_indices(pool, year, policy):
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    is_train = (
        (pool["decision_date"] < start)
        & (pool["target_date"] < start)
    )

    if policy == "rolling2":
        is_train &= (
            pool["window_start"]
            >= f"{year - 2}-01-01"
        )

    is_dev = (
        (pool["window_start"] >= start)
        & (pool["decision_date"] <= end)
        & (pool["target_date"] <= end)
    )

    train = pool.loc[is_train]
    dev = pool.loc[is_dev]

    if train.empty or dev.empty:
        raise ValueError(
            f"Empty split: {year}, {policy}"
        )

    if (
        train["target_date"].max()
        >= dev["window_start"].min()
    ):
        raise AssertionError(
            "Development input overlaps training targets"
        )

    return train, dev


def standardize_extra_features(x_train, x_dev):
    x_train = x_train.copy()
    x_dev = x_dev.copy()

    mean = x_train[:, 0, 5:].mean(
        axis=0, dtype=np.float64
    )

    std = x_train[:, 0, 5:].std(
        axis=0, dtype=np.float64
    )

    std = np.maximum(std, 1e-8)

    x_train[:, :, 5:] = np.clip(
        (x_train[:, :, 5:] - mean) / std,
        -5, 5,
    )

    x_dev[:, :, 5:] = np.clip(
        (x_dev[:, :, 5:] - mean) / std,
        -5, 5,
    )

    scaler = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    return x_train, x_dev, scaler


def fit_one(
    train,
    dev,
    full_features,
    args,
    year,
    policy,
    seed,
    device,
):
    set_seed(seed)

    train_idx = train["row_id"].to_numpy(
        dtype=int
    )

    dev_idx = dev["row_id"].to_numpy(
        dtype=int
    )

    x_train, x_dev, scaler = (
        standardize_extra_features(
            full_features[train_idx],
            full_features[dev_idx],
        )
    )

    y_train = train["label"].to_numpy(
        dtype=np.int64
    )

    y_dev = dev["label"].to_numpy(
        dtype=np.int64
    )

    if (
        len(np.unique(y_train)) != 2
        or len(np.unique(y_dev)) != 2
    ):
        raise ValueError(
            "Both classes must appear"
        )

    train_ds = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(x_train)
        ),
        torch.from_numpy(y_train),
    )

    dev_ds = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(x_dev)
        ),
        torch.from_numpy(y_dev),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(
            seed
        ),
    )

    train_eval = DataLoader(
        train_ds,
        batch_size=1024,
        shuffle=False,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=1024,
        shuffle=False,
    )

    model = NumericMLP(7).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    prefix = (
        f"recency_mlp_aug_dev{year}_"
        f"{policy}_seed{seed}"
    )

    metrics_dir = Path("outputs/metrics")
    checkpoint_dir = Path("outputs/checkpoints")

    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    best_auc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    history = []
    start_time = time.perf_counter()

    print(
        f"\n[RUN] {prefix} "
        f"| train={len(train):,} "
        f"| dev={len(dev):,} "
        f"| train_up={y_train.mean():.3f} "
        f"| dev_up={y_dev.mean():.3f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits, y
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

        train_metrics = evaluate(
            model, train_eval, device
        )

        dev_metrics = evaluate(
            model, dev_loader, device
        )

        history.append({
            "epoch": epoch,
            "train_auc": train_metrics["auroc"],
            "dev_auc": dev_metrics["auroc"],
            "dev_f1": dev_metrics["macro_f1"],
            "dev_bal": dev_metrics[
                "balanced_accuracy"
            ],
            "dev_up_ratio": dev_metrics["up_ratio"],
            "dev_prob_std": dev_metrics["prob_std"],
            "dev_loss": dev_metrics["loss"],
        })

        improved = (
            dev_metrics["auroc"] > best_auc + 1e-5
            or (
                abs(
                    dev_metrics["auroc"] - best_auc
                ) <= 1e-5
                and dev_metrics["loss"] < best_loss
            )
        )

        if improved:
            best_auc = dev_metrics["auroc"]
            best_loss = dev_metrics["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model": "NumericMLP",
                    "model_state": model.state_dict(),
                    "dev_year": year,
                    "train_policy": policy,
                    "seed": seed,
                    "epoch": epoch,
                    "scaler": scaler,
                    "train_metrics": train_metrics,
                    "dev_metrics": dev_metrics,
                },
                checkpoint_dir / f"{prefix}_best.pt",
            )

        else:
            stale += 1

        print(
            f"[{year} {policy} s{seed}] "
            f"epoch={epoch:02d} "
            f"train_auc={train_metrics['auroc']:.4f} "
            f"dev_auc={dev_metrics['auroc']:.4f} "
            f"dev_bal={dev_metrics['balanced_accuracy']:.4f} "
            f"dev_up={dev_metrics['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience}",
            flush=True,
        )

        if stale >= args.patience:
            break

    pd.DataFrame(history).to_csv(
        metrics_dir / f"{prefix}_history.csv",
        index=False,
    )

    saved = torch.load(
        checkpoint_dir / f"{prefix}_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    best_dev = evaluate(
        model, dev_loader, device
    )

    best_train = evaluate(
        model, train_eval, device
    )

    return {
        "dev_year": year,
        "policy": policy,
        "seed": seed,
        "train_n": len(train),
        "dev_n": len(dev),
        "best_epoch": best_epoch,
        "train_auc_best": best_train["auroc"],
        "dev_auc": best_dev["auroc"],
        "dev_bal_acc": best_dev[
            "balanced_accuracy"
        ],
        "dev_macro_f1": best_dev["macro_f1"],
        "dev_up_ratio": best_dev["up_ratio"],
        "seconds": round(
            time.perf_counter() - start_time, 2
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021],
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["expanding", "rolling2"],
        default=["expanding", "rolling2"],
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
    ):
        parser.error(
            "epochs, patience, batch-size "
            "and lr must be positive"
        )

    if any(
        year not in (2019, 2020, 2021)
        for year in args.years
    ):
        parser.error(
            "years must be 2019, 2020 or 2021"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")
    print(
        "[DATA] Original train split only; "
        "2022-2025 never used"
    )

    pool = load_pool(
        "data/processed/samples.csv"
    )

    print("[EXTRACT] OHLCV sequences")

    _, augmented = create_sequences(
        pool,
        Path("data/raw"),
    )

    results = []

    summary_path = Path(
        "outputs/metrics/"
        "recency_walk_forward_summary.csv"
    )

    for year in args.years:
        for policy in args.policies:
            train, dev = split_indices(
                pool, year, policy
            )

            for seed in args.seeds:
                result = fit_one(
                    train,
                    dev,
                    augmented,
                    args,
                    year,
                    policy,
                    seed,
                    device,
                )

                results.append(result)

                pd.DataFrame(results).to_csv(
                    summary_path,
                    index=False,
                )

    summary = pd.DataFrame(results)

    print("\n[SUMMARY]")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n[BY POLICY]")

    print(
        summary.groupby("policy")[
            ["dev_auc", "dev_bal_acc"]
        ].mean().round(4).to_string()
    )

    print(f"\n[SAVED] {summary_path}")
    print(
        "[NOTE] 2022-2025 data untouched. "
        "Development results are exploratory."
    )


if __name__ == "__main__":
    main()
