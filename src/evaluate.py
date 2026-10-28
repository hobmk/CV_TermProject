
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from src.data.build_dataset import ChartDataset
from src.models.cnn import CNNBaseline
from src.models.standard_vit import StandardViT
from src.models.day_aligned_vit import DayAlignedViT


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    predicted = (probs >= 0.5).astype(np.int64)

    accuracy = accuracy_score(
        labels,
        predicted,
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predicted,
    )

    macro_f1 = f1_score(
        labels,
        predicted,
        average="macro",
        zero_division=0,
    )

    if len(np.unique(labels)) == 2:
        auroc = roc_auc_score(
            labels,
            probs,
        )
    else:
        auroc = float("nan")

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc),
    }


def build_model_from_checkpoint(checkpoint):
    model_name = checkpoint.get("model_name")
    model_config = checkpoint.get("model_config", {}) or {}

    if model_name == "cnn":
        model = CNNBaseline(**model_config)

    elif model_name == "standard_vit":
        model = StandardViT(**model_config)

    elif model_name == "day_aligned_vit":
        model = DayAlignedViT(**model_config)

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    return model, model_name


def predict(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.inference_mode():

        for images, targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            all_labels.extend(
                targets.cpu().numpy().tolist()
            )

            all_probs.extend(
                probs.cpu().numpy().tolist()
            )

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probs, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model, model_name = build_model_from_checkpoint(
        checkpoint
    )

    model = model.to(device)

    print(f"[MODEL] {model_name}")
    print(f"[CHECKPOINT] {checkpoint_path}")
    print(f"[BEST EPOCH] {checkpoint.get('epoch', 'Unknown')}")

    # --------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------
    dataset = ChartDataset(args.split)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    labels, probabilities = predict(
        model=model,
        loader=loader,
        device=device,
    )

    metrics = compute_metrics(
        labels,
        probabilities,
    )

    predicted = (
        probabilities >= 0.5
    ).astype(np.int64)

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
    )

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[EVALUATION: {args.split.upper()}]")
    print(f"Model             : {model_name}")
    print(f"Samples           : {len(labels):,}")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"AUROC             : {metrics['auroc']:.4f}")

    print("\n[CONFUSION MATRIX]")
    print("Rows = Actual / Columns = Predicted")
    print(cm)

    print("=" * 70)

    # --------------------------------------------------
    # Save Predictions
    # --------------------------------------------------
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_name = checkpoint_path.stem
    prefix = output_dir / f"{run_name}_{args.split}"

    predictions_df = dataset.samples[
        ["ticker", "decision_date", "label"]
    ].copy()

    predictions_df["prob_up"] = probabilities
    predictions_df["predicted_label"] = predicted

    predictions_path = Path(
        f"{prefix}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------
    # Save Evaluation Metrics
    # --------------------------------------------------
    metrics_data = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "best_epoch": checkpoint.get("epoch"),
        "seed": checkpoint.get("seed"),
        "num_samples": int(len(labels)),
        **metrics,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = Path(
        f"{prefix}_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_data,
            f,
            indent=2,
        )

    print(f"\n[SAVED] {predictions_path}")
    print(f"[SAVED] {metrics_path}")


if __name__ == "__main__":
    main()
