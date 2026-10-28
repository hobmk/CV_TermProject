
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


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    predicted = (probs >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "macro_f1": float(
            f1_score(
                labels, predicted,
                average="macro", zero_division=0
            )
        ),
        "auroc": (
            float(roc_auc_score(labels, probs))
            if len(np.unique(labels)) == 2
            else float("nan")
        ),
    }


def predict(model, loader, device):
    model.eval()
    labels, probs = [], []

    with torch.inference_mode():
        for images, y in loader:
            logits = model(images.to(device, non_blocking=True))
            p = logits.softmax(dim=1)[:, 1]

            labels.extend(y.numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())

    return np.asarray(labels), np.asarray(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/cnn_seed42.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )

    if checkpoint.get("model_name") != "cnn":
        raise ValueError("Currently supports CNN checkpoints only")

    model = CNNBaseline().to(device)
    model.load_state_dict(checkpoint["model_state"])

    dataset = ChartDataset(args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    labels, probs = predict(model, loader, device)
    results = compute_metrics(labels, probs)
    predicted = (probs >= 0.5).astype(int)

    print(
        f"[{args.split.upper()}] "
        + " | ".join(
            f"{k}={v:.4f}" for k, v in results.items()
        )
    )
    print("Confusion matrix (rows=true, columns=pred):")
    print(confusion_matrix(labels, predicted, labels=[0, 1]))

    name = Path(args.checkpoint).stem
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{name}_{args.split}"

    result_df = dataset.samples[
        ["ticker", "decision_date", "label"]
    ].copy()
    result_df["prob_up"] = probs
    result_df["predicted_label"] = predicted

    result_df.to_csv(
        f"{prefix}_predictions.csv", index=False
    )

    with open(
        f"{prefix}_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] {prefix}_predictions.csv")


if __name__ == "__main__":
    main()
