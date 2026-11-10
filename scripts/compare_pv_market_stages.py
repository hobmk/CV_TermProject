
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.compare_day_market_context import (
    ChartMarketDataset,
    DayViTWithMarket,
    build_peer_context,
    load_dev_split,
)


MODES = (
    "context_only",
    "original_only",
    "pv_only",
    "original_market",
    "pv_market",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContextOnlyDataset(Dataset):
    def __init__(self, metadata, features):
        self.context = torch.from_numpy(
            np.asarray(
                features,
                dtype=np.float32,
            ).copy()
        )

        self.y = torch.tensor(
            metadata["label"].to_numpy(
                dtype=np.int64,
                copy=True,
            )
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # No chart image for context-only learning
        return (
            torch.empty(0),
            self.context[i],
            self.y[i],
        )


class ContextOnlyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.context_encoder = nn.Sequential(
            nn.Linear(6, 32, bias=False),
            nn.GELU(),

            nn.Linear(32, 32, bias=False),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2),
        )

    def forward(self, images, context):
        return self.head(
            self.context_encoder(context)
        )


class PriceVolumeDayViT(nn.Module):
    """
    Separate price and volume areas.

    Input: [B,3,160,160]

    Price:  y=4..117   (114 rows)
    Volume: y=126..155 (30 rows)

    Output:
        20 trading-day tokens, dim=128
    """

    def __init__(self, use_market):
        super().__init__()

        self.use_market = bool(use_market)

        # Price: 96-dimensional day embedding
        self.price_embed = nn.Conv2d(
            3,
            96,
            kernel_size=(114, 8),
            stride=(114, 8),
        )

        # Volume: 32-dimensional day embedding
        self.volume_embed = nn.Conv2d(
            3,
            32,
            kernel_size=(30, 8),
            stride=(30, 8),
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 20, 128)
        )

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])

        self.norm = nn.LayerNorm(128)

        self.context_encoder = nn.Sequential(
            nn.Linear(6, 32, bias=False),
            nn.GELU(),

            nn.Linear(32, 32, bias=False),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2),
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

    def forward(self, images, context):
        if (
            images.ndim != 4
            or tuple(images.shape[1:])
            != (3, 160, 160)
        ):
            raise ValueError(
                f"Expected [B,3,160,160], "
                f"got {tuple(images.shape)}"
            )

        # Remove white background
        price = self.price_embed(
            images[:, :, 4:118, :] - 1.0
        )

        volume = self.volume_embed(
            images[:, :, 126:156, :] - 1.0
        )

        # [B,96,1,20] + [B,32,1,20]
        # -> [B,128,1,20]
        tokens = torch.cat(
            (price, volume),
            dim=1,
        )

        # -> [B,20,128]
        tokens = (
            tokens.flatten(2).transpose(1, 2)
        )

        tokens = self.dropout(
            tokens + self.pos_embed
        )

        for layer in self.blocks:
            tokens = layer(tokens)

        image_features = self.norm(
            tokens.mean(dim=1)
        )

        if not self.use_market:
            context = torch.zeros_like(
                context
            )

        context_features = (
            self.context_encoder(context)
        )

        fused = torch.cat(
            (image_features, context_features),
            dim=1,
        )

        return self.head(fused)


def build_model(mode):
    if mode == "context_only":
        return ContextOnlyMLP()

    if mode in (
        "original_only",
        "original_market",
    ):
        return DayViTWithMarket(
            use_market=(
                mode == "original_market"
            )
        )

    if mode in (
        "pv_only",
        "pv_market",
    ):
        return PriceVolumeDayViT(
            use_market=(
                mode == "pv_market"
            )
        )

    raise ValueError(mode)


def metrics_for(y, p, metadata):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    predicted = (
        p >= 0.5
    ).astype(np.int64)

    table = metadata[
        ["decision_date", "return_5d"]
    ].copy()

    table["prob_up"] = p
    table["label"] = y

    ic_values = []
    daily_auc_values = []

    for _, part in table.groupby(
        "decision_date",
        sort=False,
    ):
        if part["label"].nunique() == 2:
            daily_auc_values.append(
                roc_auc_score(
                    part["label"],
                    part["prob_up"],
                )
            )

        if (
            part["prob_up"].nunique() > 1
            and part["return_5d"].nunique() > 1
        ):
            ic_values.append(
                part["prob_up"].corr(
                    part["return_5d"],
                    method="spearman",
                )
            )

    table["date_avg_prob"] = (
        table.groupby("decision_date")[
            "prob_up"
        ].transform("mean")
    )

    return {
        "auc": float(
            roc_auc_score(y, p)
        ),

        "date_auc": float(
            roc_auc_score(
                y,
                table["date_avg_prob"],
            )
        ),

        "daily_auc": float(
            np.mean(daily_auc_values)
        ),

        "rank_ic": (
            float(np.mean(ic_values))
            if ic_values
            else float("nan")
        ),

        "acc": float(
            accuracy_score(y, predicted)
        ),

        "bal": float(
            balanced_accuracy_score(
                y, predicted
            )
        ),

        "f1": float(
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
    }


def evaluate(
    model,
    loader,
    metadata,
    device,
    return_prob=False,
):
    model.eval()

    all_y = []
    all_p = []
    loss_total = 0.0

    with torch.inference_mode():
        for images, context, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            context = context.to(
                device,
                non_blocking=True,
            )

            labels_device = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images,
                context,
            )

            loss_total += (
                nn.functional.cross_entropy(
                    logits,
                    labels_device,
                    reduction="sum",
                ).item()
            )

            all_y.append(
                labels.numpy()
            )

            all_p.append(
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    expected_y = metadata[
        "label"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        y, expected_y
    ):
        raise RuntimeError(
            "Labels and metadata are out of order"
        )

    metrics = metrics_for(
        y, p, metadata
    )

    metrics["loss"] = (
        loss_total / len(y)
    )

    if return_prob:
        return metrics, p

    return metrics


def run_one(
    mode,
    args,
    train_data,
    dev_data,
    train_rows,
    dev_rows,
    scaler,
    device,
):
    seed_everything(args.seed)

    model = build_model(mode).to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=(
            torch.Generator().manual_seed(
                args.seed
            )
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    dev_loader = DataLoader(
        dev_data,
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

    prefix = (
        f"pv_context_{mode}_"
        f"seed{args.seed}_dev2021"
    )

    checkpoint = (
        Path("outputs/checkpoints")
        / f"{prefix}_best.pt"
    )

    history_file = (
        Path("outputs/metrics")
        / f"{prefix}_history.csv"
    )

    prediction_file = (
        Path("outputs/metrics")
        / f"{prefix}_dev_predictions.csv"
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_auc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    start = time.perf_counter()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[VARIANT] {mode} "
        f"| params={total_params:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        tr_y = []
        tr_p = []

        for images, context, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            context = context.to(
                device,
                non_blocking=True,
            )

            labels_device = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images,
                context,
            )

            loss = (
                nn.functional.cross_entropy(
                    logits,
                    labels_device,
                )
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            tr_y.append(
                labels.numpy()
            )

            tr_p.append(
                torch.softmax(
                    logits.detach(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

        train_auc = roc_auc_score(
            np.concatenate(tr_y),
            np.concatenate(tr_p),
        )

        dev = evaluate(
            model,
            dev_loader,
            dev_rows,
            device,
        )

        improved = (
            dev["auc"] > best_auc + 1e-5
            or (
                abs(
                    dev["auc"] - best_auc
                ) <= 1e-5
                and dev["loss"] < best_loss
            )
        )

        if improved:
            best_auc = dev["auc"]
            best_loss = dev["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_name": "pv_context_ablation",
                    "mode": mode,
                    "model_state": model.state_dict(),
                    "seed": args.seed,
                    "epoch": epoch,
                    "scaler": scaler,
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                    "dev_metrics": dev,
                },
                checkpoint,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_auc_training_mode": float(
                train_auc
            ),
            **{
                f"dev_{k}": v
                for k, v in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_file,
            index=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"[{mode}] "
            f"ep={epoch:02d}/{args.epochs} "
            f"train_auc={train_auc:.4f} "
            f"dev_auc={dev['auc']:.4f} "
            f"date_auc={dev['date_auc']:.4f} "
            f"daily_auc={dev['daily_auc']:.4f} "
            f"IC={dev['rank_ic']:+.4f} "
            f"bal={dev['bal']:.4f} "
            f"up={dev['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {mode}"
            )
            break

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    best, prob = evaluate(
        model,
        dev_loader,
        dev_rows,
        device,
        return_prob=True,
    )

    predictions = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    predictions["prob_up"] = prob

    predictions.to_csv(
        prediction_file,
        index=False,
    )

    return {
        "mode": mode,
        "params": total_params,
        "best_epoch": best_epoch,
        "dev_auc": best["auc"],
        "dev_date_auc": best["date_auc"],
        "dev_daily_auc": best["daily_auc"],
        "dev_rank_ic": best["rank_ic"],
        "dev_bal": best["bal"],
        "dev_macro_f1": best["f1"],
        "dev_up_ratio": best["up_ratio"],
        "dev_prob_std": best["prob_std"],
        "elapsed_sec": round(
            time.perf_counter() - start,
            1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.num_workers < 0
        or args.weight_decay < 0
    ):
        parser.error(
            "Invalid training arguments"
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] Train=2015-2020 "
        "| Dev=2021 "
        "| No 2022-2025 evaluation"
    )

    train_rows, dev_rows = load_dev_split(
        Path("data/processed/samples.csv")
    )

    train_context, dev_context, scaler = (
        build_peer_context(
            train_rows,
            dev_rows,
            Path("data/raw"),
        )
    )

    context_train = ContextOnlyDataset(
        train_rows,
        train_context,
    )

    context_dev = ContextOnlyDataset(
        dev_rows,
        dev_context,
    )

    use_images = any(
        mode != "context_only"
        for mode in args.variants
    )

    if use_images:
        image_train = ChartMarketDataset(
            train_rows,
            train_context,
            Path("data/charts"),
        )

        image_dev = ChartMarketDataset(
            dev_rows,
            dev_context,
            Path("data/charts"),
        )

    rows = []

    summary_path = Path(
        "outputs/metrics/"
        "pv_context_stages_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode in args.variants:
        if mode == "context_only":
            ds_train = context_train
            ds_dev = context_dev
        else:
            ds_train = image_train
            ds_dev = image_dev

        result = run_one(
            mode,
            args,
            ds_train,
            ds_dev,
            train_rows,
            dev_rows,
            scaler,
            device,
        )

        rows.append(result)

        pd.DataFrame(rows).to_csv(
            summary_path,
            index=False,
        )

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(rows).to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print(f"[SAVED] {summary_path}")

    print(
        "[NOTE] Exploratory 2021 Dev only; "
        "2022-2025 untouched"
    )


if __name__ == "__main__":
    main()
