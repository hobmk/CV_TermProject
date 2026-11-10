
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


def run_variant(
    args, variant, train_ds, val_ds, device
):
    set_seed(args.seed)

    # Same sampling order across variants
    generator = torch.Generator().manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
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

    config = {
        "image_size": 160,
        "num_days": 20,
        "embed_dim": 128,
        "depth": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "num_classes": 2,
        "pooling": "mean",
        "background_center": True,
        "token_norm": False,
        "tokenizer": variant,
    }

    model = DayAlignedViT(**config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    run_name = (
        f"day_tokenizer_{variant}_"
        f"center_seed{args.seed}"
    )

    checkpoint_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    best_path = (
        checkpoint_dir / f"{run_name}_best.pt"
    )

    last_path = (
        checkpoint_dir / f"{run_name}_last.pt"
    )

    history_path = (
        metrics_dir / f"{run_name}_history.csv"
    )

    best_auc = -float("inf")
    best_epoch = 0
    history = []

    start_time = time.perf_counter()

    print("\n" + "=" * 75)
    print(f"[VARIANT] {variant}")
    print(f"[PARAMETERS] {num_parameters:,}")
    print(f"[CONFIG] {config}")
    print("=" * 75)

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

        elapsed = (
            time.perf_counter() - start_time
        )

        history.append({
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 1),
            **{
                f"train_{k}": v
                for k, v in train.items()
            },
            **{
                f"val_{k}": v
                for k, v in val.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        checkpoint = {
            "model_name": "day_aligned_vit",
            "model_config": config,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "train_metrics": train,
            "val_metrics": val,
            "grad_clip": 1.0,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
        }

        # Always preserve the latest epoch
        torch.save(checkpoint, last_path)

        if val["auroc"] > best_auc + 1e-6:
            best_auc = val["auroc"]
            best_epoch = epoch
            torch.save(checkpoint, best_path)

        print(
            f"[{variant}] "
            f"{epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train['loss']:.4f} "
            f"| train_auc={train['auroc']:.4f} "
            f"| val_auc={val['auroc']:.4f} "
            f"| val_f1={val['macro_f1']:.4f} "
            f"| up_ratio={val['pred_up_ratio']:.3f} "
            f"| prob_std={val['prob_std']:.5f} "
            f"| best_epoch={best_epoch} "
            f"| elapsed={elapsed:.0f}s",
            flush=True,
        )

    return {
        "variant": variant,
        "params": num_parameters,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "last_train_loss": train["loss"],
        "last_train_auc": train["auroc"],
        "last_val_auc": val["auroc"],
        "last_up_ratio": val["pred_up_ratio"],
        "last_prob_std": val["prob_std"],
        "elapsed_sec": round(
            time.perf_counter() - start_time, 1
        ),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["linear", "local_cnn"],
        default=["linear", "local_cnn"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.lr <= 0
    ):
        parser.error(
            "epochs, batch-size and lr must be positive"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(
            f"[GPU] {torch.cuda.get_device_name(0)}"
        )

    train_ds = ChartDataset("train")
    val_ds = ChartDataset("val")

    summaries = []

    summary_path = Path(
        "outputs/metrics/"
        "day_tokenizer_comparison.csv"
    )

    for variant in args.variants:
        result = run_variant(
            args,
            variant,
            train_ds,
            val_ds,
            device,
        )

        summaries.append(result)

        pd.DataFrame(summaries).to_csv(
            summary_path,
            index=False,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print("[SUMMARY]")
    print("=" * 75)

    print(
        pd.DataFrame(summaries).to_string(
            index=False
        )
    )

    print(f"\n[SAVED] {summary_path}")
    print("[NOTE] Test set untouched")


if __name__ == "__main__":
    main()
