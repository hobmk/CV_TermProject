
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name, pooling="cls"):
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
            "depth": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "num_classes": 2,
            "pooling": pooling,
        }
        model = DayAlignedViT(**config)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, config


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    grad_clip=0.0,
):
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

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                    )

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

    probs_array = np.asarray(all_probs)

    return {
        "loss": total_loss / total_count,
        **compute_metrics(all_labels, all_probs),
        "pred_up_ratio": float(
            (probs_array >= 0.5).mean()
        ),
        "prob_std": float(probs_array.std()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["cnn", "standard_vit", "day_aligned_vit"], default="cnn")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--monitor", choices=["macro_f1", "auroc"], default="macro_f1")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    if args.patience < 1 or args.num_workers < 0:
        parser.error("Invalid patience or num-workers")

    if args.lr <= 0 or args.grad_clip < 0:
        parser.error("Invalid lr or grad-clip")

    if args.model != "day_aligned_vit" and args.pooling != "cls":
        parser.error("--pooling mean currently supports day_aligned_vit only")

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

    model, config = build_model(
        args.model,
        pooling=args.pooling,
    )

    model = model.to(device)

    num_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"[MODEL] {args.model}")
    print(f"[CONFIG] {config}")
    print(f"[PARAMETERS] {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_name = (
        args.run_name
        or f"{args.model}_{args.pooling}_seed{args.seed}"
    )

    ckpt_dir = Path("outputs/checkpoints")
    metric_dir = Path("outputs/metrics")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{run_name}.pt"
    history_path = metric_dir / f"{run_name}_history.csv"

    best_score = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    stale = 0
    history = []

    start_time = time.perf_counter()

    print("\n[TRAINING START]")
    print(
        f"Model={args.model} | Pooling={args.pooling} "
        f"| Monitor={args.monitor} "
        f"| GradClip={args.grad_clip} "
        f"| Patience={args.patience}"
    )

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        elapsed = time.perf_counter() - start_time

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
            history_path,
            index=False,
        )

        score = val[args.monitor]

        improved = (
            score > best_score + 1e-6
            or (
                abs(score - best_score) <= 1e-6
                and val["loss"] < best_loss - 1e-6
            )
        )

        if improved:
            best_score = score
            best_loss = val["loss"]
            best_epoch = epoch
            best_metrics = dict(val)
            stale = 0

            torch.save(
                {
                    "model_name": args.model,
                    "model_config": config,
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "train_metrics": train,
                    "val_metrics": val,
                    "monitor": args.monitor,
                    "grad_clip": args.grad_clip,
                    "normalization": "window_minmax",
                    "window": 20,
                    "horizon": 5,
                    "stride": 1,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                    "num_parameters": num_params,
                },
                ckpt_path,
            )
        else:
            stale += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={train['loss']:.4f} "
            f"| train_f1={train['macro_f1']:.4f} "
            f"| val_loss={val['loss']:.4f} "
            f"| val_acc={val['accuracy']:.4f} "
            f"| val_bal={val['balanced_accuracy']:.4f} "
            f"| val_f1={val['macro_f1']:.4f} "
            f"| val_auc={val['auroc']:.4f} "
            f"| up_ratio={val['pred_up_ratio']:.3f} "
            f"| prob_std={val['prob_std']:.6f} "
            f"| best={best_epoch} "
            f"| patience={stale}/{args.patience} "
            f"| elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] No {args.monitor} improvement "
                f"for {args.patience} epochs"
            )
            break

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("[TRAINING COMPLETED]")
    print(f"Model         : {args.model}")
    print(f"Pooling       : {args.pooling}")
    print(f"Best Epoch    : {best_epoch}")
    print(f"Best Val {args.monitor}: {best_score:.4f}")
    print(f"Best Val F1   : {best_metrics['macro_f1']:.4f}")
    print(f"Best Val AUROC: {best_metrics['auroc']:.4f}")
    print(f"Total Time    : {total_time:.1f}s")
    print(f"Checkpoint    : {ckpt_path}")
    print(f"History       : {history_path}")
    print("=" * 70)
    print("[NOTE] Test set remains untouched.")


if __name__ == "__main__":
    main()
