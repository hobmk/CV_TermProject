
import argparse
import random

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.data.build_dataset import ChartDataset
from src.models.day_aligned_vit import DayAlignedViT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = ChartDataset("train")
    labels = dataset.samples["label"].to_numpy()

    rng = np.random.default_rng(args.seed)

    selected = []
    for cls in (0, 1):
        candidates = np.flatnonzero(labels == cls)
        chosen = rng.choice(
            candidates,
            size=args.per_class,
            replace=False,
        )
        selected.extend(chosen.tolist())

    rng.shuffle(selected)

    # Cache the tiny subset in GPU memory
    images = torch.stack([
        dataset[i][0] for i in selected
    ]).to(device)

    targets = torch.tensor(
        labels[selected],
        dtype=torch.long,
        device=device,
    )

    model = DayAlignedViT(
        image_size=160,
        num_days=20,
        embed_dim=128,
        depth=2,
        num_heads=4,
        dropout=0.0,
        num_classes=2,
        pooling=args.pooling,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    criterion = torch.nn.CrossEntropyLoss()

    print(f"[DEVICE] {device}")
    print(f"[SAMPLES] {len(targets)} | Balanced 0/1")
    print(f"[IMAGE STD] {images.std().item():.6f}")
    print("[START] Tiny-subset overfitting test")

    for epoch in range(1, args.epochs + 1):
        model.train()

        order = torch.randperm(
            len(targets), device=device
        )

        for start in range(
            0, len(targets), args.batch_size
        ):
            indices = order[
                start:start + args.batch_size
            ]

            logits = model(images[indices])
            loss = criterion(logits, targets[indices])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            model.eval()

            with torch.inference_mode():
                logits = model(images)
                loss = criterion(logits, targets)
                probs = logits.softmax(dim=1)[:, 1]

            y = targets.cpu().numpy()
            p = probs.cpu().numpy()
            pred = (p >= 0.5).astype(int)

            acc = (pred == y).mean()
            bal = balanced_accuracy_score(y, pred)
            auc = roc_auc_score(y, p)

            print(
                f"Epoch {epoch:03d} "
                f"| loss={loss.item():.4f} "
                f"| acc={acc:.4f} "
                f"| bal={bal:.4f} "
                f"| auc={auc:.4f} "
                f"| prob_std={p.std():.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
