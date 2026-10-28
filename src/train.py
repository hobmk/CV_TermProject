
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    losses, count = 0.0, 0
    labels, probs = [], []

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

        bs = targets.shape[0]
        losses += loss.item() * bs
        count += bs

        labels.extend(
            targets.detach().cpu().numpy().tolist()
        )
        probs.extend(
            logits.detach()
            .softmax(dim=1)[:, 1]
            .cpu().numpy().tolist()
        )

    return {
        "loss": losses / count,
        **compute_metrics(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    train_ds = ChartDataset("train")
    val_ds = ChartDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CNNBaseline().to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"[MODEL] CNNBaseline | Params={params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    name = args.run_name or f"cnn_seed{args.seed}"

    checkpoint_path = (
        Path("outputs/checkpoints") / f"{name}.pt"
    )
    history_path = (
        Path("outputs/metrics") / f"{name}_history.csv"
    )

    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    history_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val = run_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
        }
        row.update({
            f"train_{k}": v for k, v in train.items()
        })
        row.update({
            f"val_{k}": v for k, v in val.items()
        })

        history.append(row)
        pd.DataFrame(history).to_csv(
            history_path, index=False
        )

        improved = (
            val["macro_f1"] > best_f1 + 1e-6
            or (
                abs(val["macro_f1"] - best_f1) <= 1e-6
                and val["loss"] < best_loss - 1e-6
            )
        )

        if improved:
            best_f1 = val["macro_f1"]
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_name": "cnn",
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_metrics": val,
                    "train_metrics": train,
                    "normalization": "window_minmax",
                    "window": 20,
                    "horizon": 5,
                },
                checkpoint_path,
            )
        else:
            stale += 1

        print(
            f"Epoch {epoch:03d} "
            f"| train_loss={train['loss']:.4f} "
            f"| val_loss={val['loss']:.4f} "
            f"| val_acc={val['accuracy']:.4f} "
            f"| val_bal={val['balanced_accuracy']:.4f} "
            f"| val_f1={val['macro_f1']:.4f} "
            f"| val_auc={val['auroc']:.4f} "
            f"| best_epoch={best_epoch} "
            f"| elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] No improvement "
                f"for {args.patience} epochs"
            )
            break

    print(
        f"[DONE] Best epoch={best_epoch}, "
        f"Val Macro F1={best_f1:.4f}"
    )
    print(f"[SAVED] {checkpoint_path}")
    print(f"[SAVED] {history_path}")
    print("[NOTE] Keep Test untouched until model selection ends")


if __name__ == "__main__":
    main()
