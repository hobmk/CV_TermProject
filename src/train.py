
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.build_dataset import ChartDataset
from src.evaluate import compute_metrics
from src.models.cnn import CNNBaseline
from src.models.standard_vit import StandardViT
from src.models.day_aligned_vit import DayAlignedViT


MODEL_NAMES = ("cnn", "standard_vit", "day_aligned_vit")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name):
    if model_name == "cnn":
        config = {}
        model = CNNBaseline()

    elif model_name == "standard_vit":
        config = {
            "image_size": 160,
            "patch_size": 32,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "num_classes": 2,
        }
        model = StandardViT(**config)

    elif model_name == "day_aligned_vit":
        config = {
            "image_size": 160,
            "num_days": 20,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "num_classes": 2,
        }
        model = DayAlignedViT(**config)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, config


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    all_labels = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_count += batch_size

        probs = torch.softmax(
            logits.detach(), dim=1
        )[:, 1]

        all_labels.extend(
            targets.detach().cpu().numpy().tolist()
        )
        all_probs.extend(
            probs.cpu().numpy().tolist()
        )

    metrics = compute_metrics(all_labels, all_probs)

    return {
        "loss": total_loss / total_count,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=MODEL_NAMES, default="cnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.patience < 1:
        parser.error("--patience must be >= 1")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")

    if args.lr <= 0:
        parser.error("--lr must be > 0")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------
    train_dataset = ChartDataset("train")
    val_dataset = ChartDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model, model_config = build_model(args.model)
    model = model.to(device)

    num_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"[MODEL] {args.model}")
    print(f"[PARAMETERS] {num_params:,}")
    print(f"[CONFIG] {model_config}")

    # --------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------
    run_name = (
        args.run_name
        if args.run_name
        else f"{args.model}_seed{args.seed}"
    )

    checkpoint_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    history_path = metrics_dir / f"{run_name}_history.csv"

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []
    start_time = time.perf_counter()

    print("\n[TRAINING START]")
    print(
        f"Model={args.model} "
        f"| Epochs={args.epochs} "
        f"| Batch={args.batch_size} "
        f"| LR={args.lr} "
        f"| WeightDecay={args.weight_decay} "
        f"| Patience={args.patience} "
        f"| Seed={args.seed}"
    )
    print("-" * 110)

    for epoch in range(1, args.epochs + 1):

        # --------------------------------------------------
        # Train
        # --------------------------------------------------
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        elapsed = time.perf_counter() - start_time

        # --------------------------------------------------
        # Save epoch history
        # --------------------------------------------------
        row = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
        }

        row.update({
            f"train_{key}": value
            for key, value in train_metrics.items()
        })

        row.update({
            f"val_{key}": value
            for key, value in val_metrics.items()
        })

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        # --------------------------------------------------
        # Best model selection
        # Primary: Validation Macro F1
        # Tie-breaker: Validation Loss
        # --------------------------------------------------
        current_f1 = val_metrics["macro_f1"]
        current_loss = val_metrics["loss"]

        improved = (
            current_f1 > best_f1 + 1e-6
            or (
                abs(current_f1 - best_f1) <= 1e-6
                and current_loss < best_loss - 1e-6
            )
        )

        if improved:
            best_f1 = current_f1
            best_loss = current_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "model_name": args.model,
                "model_config": model_config,
                "model_state": model.state_dict(),
                "seed": args.seed,
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "normalization": "window_minmax",
                "window": 20,
                "horizon": 5,
                "stride": 1,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "num_parameters": num_params,
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

        else:
            patience_counter += 1

        # --------------------------------------------------
        # Terminal output
        # --------------------------------------------------
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"| train_acc={train_metrics['accuracy']:.4f} "
            f"| train_f1={train_metrics['macro_f1']:.4f} "
            f"| val_acc={val_metrics['accuracy']:.4f} "
            f"| val_bal={val_metrics['balanced_accuracy']:.4f} "
            f"| val_f1={val_metrics['macro_f1']:.4f} "
            f"| val_auc={val_metrics['auroc']:.4f} "
            f"| best={best_epoch} "
            f"| patience={patience_counter}/{args.patience} "
            f"| elapsed={elapsed:.0f}s",
            flush=True,
        )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------
        if patience_counter >= args.patience:
            print(
                f"\n[EARLY STOPPING] "
                f"No improvement for {args.patience} epochs."
            )
            break

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print("[TRAINING COMPLETED]")
    print(f"Model       : {args.model}")
    print(f"Seed        : {args.seed}")
    print(f"Best Epoch  : {best_epoch}")
    print(f"Best Val F1 : {best_f1:.4f}")
    print(f"Total Time  : {total_time:.1f}s")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"History     : {history_path}")
    print("=" * 80)

    print("[NOTE] Test set remains untouched.")


if __name__ == "__main__":
    main()
