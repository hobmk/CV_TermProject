
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.build_dataset import ChartDataset
from src.models.day_aligned_vit import DayAlignedViT
from src.train import run_epoch, set_seed


VARIANTS = {
    "base": (False, False),
    "center": (True, False),
    "norm": (False, True),
    "both": (True, True),
}


def run_variant(args, name, train_ds, val_ds, device):
    set_seed(args.seed)

    center, token_norm = VARIANTS[name]

    config = {
        "image_size": 160,
        "num_days": 20,
        "embed_dim": 128,
        "depth": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "num_classes": 2,
        "pooling": "mean",
        "background_center": center,
        "token_norm": token_norm,
    }

    model = DayAlignedViT(**config).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    run_name = f"day_input_{name}_seed{args.seed}"

    checkpoint_path = (
        checkpoint_dir / f"{run_name}.pt"
    )

    history_path = (
        metrics_dir / f"{run_name}_history.csv"
    )

    best_auc = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    history = []
    start = time.perf_counter()

    print("\n" + "=" * 72)
    print(f"[VARIANT] {name}")
    print(f"[CONFIG] {config}")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=1.0,
        )

        val = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "variant": name,
            **{
                f"train_{k}": v
                for k, v in train.items()
            },
            **{
                f"val_{k}": v
                for k, v in val.items()
            },
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        improved = (
            val["auroc"] > best_auc + 1e-6
            or (
                abs(val["auroc"] - best_auc) <= 1e-6
                and val["loss"] < best_loss - 1e-6
            )
        )

        if improved:
            best_auc = val["auroc"]
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_name": "day_aligned_vit",
                    "model_config": config,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "train_metrics": train,
                    "val_metrics": val,
                    "monitor": "auroc",
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "grad_clip": 1.0,
                },
                checkpoint_path,
            )
        else:
            stale += 1

        elapsed = time.perf_counter() - start

        print(
            f"[{name}] "
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train['loss']:.4f} "
            f"| train_auc={train['auroc']:.4f} "
            f"| val_loss={val['loss']:.4f} "
            f"| val_auc={val['auroc']:.4f} "
            f"| val_f1={val['macro_f1']:.4f} "
            f"| up_ratio={val['pred_up_ratio']:.3f} "
            f"| prob_std={val['prob_std']:.6f} "
            f"| best={best_epoch} "
            f"| elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(f"[EARLY STOP] {name}")
            break

    return {
        "variant": name,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auc,
        "last_train_loss": train["loss"],
        "last_train_auroc": train["auroc"],
        "last_val_up_ratio": val["pred_up_ratio"],
        "last_val_prob_std": val["prob_std"],
        "elapsed_sec": round(
            time.perf_counter() - start, 1
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    train_ds = ChartDataset("train")
    val_ds = ChartDataset("val")

    results = []

    for name in args.variants:
        result = run_variant(
            args,
            name,
            train_ds,
            val_ds,
            device,
        )

        results.append(result)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(
        "outputs/metrics/day_input_ablation_summary.csv"
    )

    pd.DataFrame(results).to_csv(
        output,
        index=False,
    )

    print("\n" + "=" * 72)
    print("[ABLATION SUMMARY]")
    print("=" * 72)

    print(
        pd.DataFrame(results).to_string(
            index=False
        )
    )

    print(f"\n[SAVED] {output}")
    print("[NOTE] Test set untouched")


if __name__ == "__main__":
    main()
