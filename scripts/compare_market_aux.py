
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.numeric_sequence_baseline import (
    create_sequences,
    load_metadata,
    normalize_augmented,
    set_seed,
)


class MarketAuxMLP(nn.Module):
    def __init__(self, input_channels=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * input_channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.direction_head = nn.Linear(64, 2)
        self.relative_head = nn.Linear(64, 2)

    def forward(self, x):
        features = self.features(x)

        return (
            self.direction_head(features),
            self.relative_head(features),
        )


def add_relative_label(samples):
    samples = samples.copy().reset_index(drop=True)

    if "return_5d" not in samples:
        raise ValueError(
            "samples.csv requires return_5d"
        )

    if samples.duplicated(
        ["decision_date", "ticker"]
    ).any():
        raise ValueError(
            "Duplicate ticker/date"
        )

    grouped = samples.groupby(
        "decision_date",
        sort=False,
    )

    counts = grouped["ticker"].transform("count")

    if counts.min() < 10:
        raise ValueError(
            "Some dates have fewer than 10 stocks"
        )

    market_return = grouped[
        "return_5d"
    ].transform("mean")

    samples["relative_return"] = (
        samples["return_5d"] - market_return
    )

    samples["relative_label"] = (
        samples["relative_return"] > 0
    ).astype("int64")

    return samples


def make_loader(
    x, frame, batch_size,
    shuffle=False, seed=42
):
    x = np.ascontiguousarray(
        x, dtype=np.float32
    )

    y = frame["label"].to_numpy(
        np.int64, copy=True
    )

    y_relative = frame[
        "relative_label"
    ].to_numpy(np.int64, copy=True)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(y_relative),
    )

    generator = (
        torch.Generator().manual_seed(seed)
        if shuffle else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def safe_auc(y, probabilities):
    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(y, probabilities)
    )


def evaluate(
    model, loader, metadata, device,
    include_predictions=False,
    calculate_ic=True,
):
    model.eval()

    labels = []
    relative_labels = []
    main_probs = []
    relative_probs = []

    total_loss = 0.0

    with torch.inference_mode():
        for x, y, y_relative in loader:
            logits, relative_logits = model(
                x.to(device)
            )

            total_loss += (
                nn.functional.cross_entropy(
                    logits,
                    y.to(device),
                    reduction="sum",
                ).item()
            )

            labels.append(y.numpy())
            relative_labels.append(
                y_relative.numpy()
            )

            main_probs.append(
                torch.softmax(
                    logits, dim=-1
                )[:, 1].cpu().numpy()
            )

            relative_probs.append(
                torch.softmax(
                    relative_logits, dim=-1
                )[:, 1].cpu().numpy()
            )

    y = np.concatenate(labels)
    yr = np.concatenate(relative_labels)
    p = np.concatenate(main_probs)
    q = np.concatenate(relative_probs)

    if len(y) != len(metadata):
        raise AssertionError(
            "Evaluation and metadata mismatch"
        )

    predicted = (
        p >= 0.5
    ).astype(np.int64)

    main_ic = []
    relative_ic = []

    if calculate_ic:
        table = pd.DataFrame({
            "date": metadata[
                "decision_date"
            ].to_numpy(),

            "future_return": metadata[
                "return_5d"
            ].to_numpy(float),

            "p_main": p,
            "p_relative": q,
        })

        for _, group in table.groupby(
            "date", sort=False
        ):
            if (
                len(group) < 10
                or group[
                    "future_return"
                ].nunique() < 2
            ):
                continue

            for key, collector in (
                ("p_main", main_ic),
                ("p_relative", relative_ic),
            ):
                if group[key].nunique() > 1:
                    ic = group[key].corr(
                        group["future_return"],
                        method="spearman",
                    )

                    collector.append(ic)

    metrics = {
        "loss": float(
            total_loss / len(y)
        ),

        "auroc": safe_auc(y, p),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y, predicted
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),

        "up_ratio": float(
            predicted.mean()
        ),

        "prob_std": float(
            p.std()
        ),

        "relative_auc": safe_auc(
            yr, q
        ),

        "main_rank_ic": (
            float(np.mean(main_ic))
            if main_ic
            else float("nan")
        ),

        "relative_rank_ic": (
            float(np.mean(relative_ic))
            if relative_ic
            else float("nan")
        ),
    }

    if not include_predictions:
        return metrics

    predictions = metadata[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
            "relative_label",
            "relative_return",
        ]
    ].copy()

    predictions["prob_up"] = p
    predictions["prob_relative_up"] = q

    return metrics, predictions


def fit_one(
    lam,
    args,
    train_x,
    dev_x,
    train_meta,
    dev_meta,
    scaler,
    device,
):
    set_seed(args.seed)

    model = MarketAuxMLP(7).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        train_x,
        train_meta,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    train_eval_loader = make_loader(
        train_x,
        train_meta,
        1024,
    )

    dev_loader = make_loader(
        dev_x,
        dev_meta,
        1024,
    )

    safe_lambda = str(lam).replace(
        ".", "p"
    )

    prefix = (
        f"market_aux_lambda{safe_lambda}_"
        f"seed{args.seed}_dev2021"
    )

    metrics_dir = Path("outputs/metrics")
    checkpoint_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    best_path = (
        checkpoint_dir
        / f"{prefix}_best.pt"
    )

    best_auc = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    start = time.perf_counter()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[RUN] lambda_rel={lam} "
        f"params={params:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        for x, y, yr in train_loader:
            x = x.to(device)
            y = y.to(device)
            yr = yr.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, relative_logits = model(x)

            loss = (
                nn.functional.cross_entropy(
                    logits, y
                )
            )

            if lam > 0:
                relative_loss = (
                    nn.functional.cross_entropy(
                        relative_logits, yr
                    )
                )

                loss = (
                    loss + lam * relative_loss
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        train = evaluate(
            model,
            train_eval_loader,
            train_meta,
            device,
            calculate_ic=False,
        )

        dev = evaluate(
            model,
            dev_loader,
            dev_meta,
            device,
        )

        history.append({
            "epoch": epoch,
            "lambda_rel": lam,

            **{
                f"train_{k}": v
                for k, v in train.items()
            },

            **{
                f"dev_{k}": v
                for k, v in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            metrics_dir
            / f"{prefix}_history.csv",
            index=False,
        )

        improved = (
            dev["auroc"] > best_auc + 1e-5
            or (
                abs(
                    dev["auroc"] - best_auc
                ) <= 1e-5
                and dev["loss"] < best_loss
            )
        )

        if improved:
            best_auc = dev["auroc"]
            best_loss = dev["loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": "market_aux_mlp",
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "lambda_rel": lam,
                    "scaler": scaler,
                    "train_metrics": train,
                    "dev_metrics": dev,
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                },
                best_path,
            )

        else:
            patience_counter += 1

        print(
            f"[lambda={lam:g}] "
            f"ep={epoch:02d} "
            f"train_auc={train['auroc']:.4f} "
            f"dev_auc={dev['auroc']:.4f} "
            f"rel_auc={dev['relative_auc']:.4f} "
            f"dev_ic={dev['main_rank_ic']:+.4f} "
            f"dev_up={dev['up_ratio']:.3f} "
            f"pstd={dev['prob_std']:.4f} "
            f"best={best_epoch} "
            f"patience={patience_counter}/{args.patience}",
            flush=True,
        )

        if patience_counter >= args.patience:
            print(
                f"[EARLY STOP] lambda={lam}"
            )
            break

    saved = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    best_dev, predictions = evaluate(
        model,
        dev_loader,
        dev_meta,
        device,
        include_predictions=True,
    )

    predictions.to_csv(
        metrics_dir
        / f"{prefix}_dev_predictions.csv",
        index=False,
    )

    return {
        "lambda_rel": lam,
        "best_epoch": best_epoch,
        "best_dev_auc": best_dev["auroc"],
        "best_dev_bal": (
            best_dev["balanced_accuracy"]
        ),
        "best_dev_macro_f1": (
            best_dev["macro_f1"]
        ),
        "best_dev_up_ratio": (
            best_dev["up_ratio"]
        ),
        "best_dev_prob_std": (
            best_dev["prob_std"]
        ),
        "best_dev_rank_ic": (
            best_dev["main_rank_ic"]
        ),
        "best_dev_relative_auc": (
            best_dev["relative_auc"]
            if lam > 0
            else float("nan")
        ),
        "best_dev_relative_rank_ic": (
            best_dev["relative_rank_ic"]
            if lam > 0
            else float("nan")
        ),
        "elapsed_sec": round(
            time.perf_counter() - start, 1
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lambda-rel",
        type=float,
        nargs="+",
        default=[0.0, 0.5],
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or any(
            x < 0
            for x in args.lambda_rel
        )
    ):
        parser.error(
            "Invalid training arguments"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] Train 2015-2020 "
        "| Dev 2021 "
        "| 2022-2025 untouched"
    )

    train_meta, dev_meta = (
        load_metadata(
            Path(
                "data/processed/samples.csv"
            )
        )
    )

    train_meta = add_relative_label(
        train_meta
    )

    dev_meta = add_relative_label(
        dev_meta
    )

    print(
        "[RELATIVE LABEL RATE] "
        f"train="
        f"{train_meta['relative_label'].mean():.4f} "
        f"dev="
        f"{dev_meta['relative_label'].mean():.4f}"
    )

    print("[EXTRACT] Train...")

    _, train_aug = create_sequences(
        train_meta,
        Path("data/raw"),
    )

    print("[EXTRACT] Dev...")

    _, dev_aug = create_sequences(
        dev_meta,
        Path("data/raw"),
    )

    train_aug, dev_aug, scaler = (
        normalize_augmented(
            train_aug,
            dev_aug,
        )
    )

    results = []

    for lam in args.lambda_rel:
        result = fit_one(
            lam,
            args,
            train_aug,
            dev_aug,
            train_meta,
            dev_meta,
            scaler,
            device,
        )

        results.append(result)

    summary = pd.DataFrame(results)

    summary_path = Path(
        "outputs/metrics/"
        "market_aux_dev2021_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n[SUMMARY]")

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\n[SAVED] {summary_path}"
    )

    print(
        "[NOTE] Future returns are used "
        "only to define supervised targets, "
        "never as input features."
    )


if __name__ == "__main__":
    main()
