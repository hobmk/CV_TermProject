
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from scripts.compare_day_market_context import load_dev_split


VARIANTS = ("resnet", "cnn_tcn", "cnn_transformer")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChartOnlyDataset(Dataset):
    def __init__(self, rows, charts_dir):
        self.rows = rows.reset_index(drop=True)

        self.labels = self.rows[
            "label"
        ].to_numpy(dtype=np.int64, copy=True)

        self.paths = [
            charts_dir
            / "train"
            / row.ticker
            / f"{row.decision_date}.png"
            for row in self.rows.itertuples(index=False)
        ]

        missing = next(
            (p for p in self.paths if not p.is_file()),
            None,
        )

        if missing is not None:
            raise FileNotFoundError(
                f"Missing chart: {missing}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            if img.size != (160, 160):
                raise ValueError(
                    f"Invalid image size: "
                    f"{self.paths[index]} {img.size}"
                )

            image = to_tensor(
                img.convert("RGB")
            )

        return image, int(self.labels[index])


class Residual2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=(1, 1),
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if (
            in_channels != out_channels
            or stride != (1, 1)
        ):
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        z = F.gelu(
            self.bn1(self.conv1(x))
        )

        z = self.bn2(self.conv2(z))

        return F.gelu(
            z + self.skip(x)
        )


class SharedChartCNN(nn.Module):
    """
    Full-image 2D CNN.

    Input:  [B, 3, 160, 160]
    Output: [B, 128, 20]

    Horizontal dimension is reduced
    to exactly 20 trading-day positions.
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                24,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            # 80 x 80
            Residual2D(
                24, 24
            ),

            # 40 x 40
            Residual2D(
                24, 48, (2, 2)
            ),

            # 20 x 20
            Residual2D(
                48, 96, (2, 2)
            ),

            # 10 x 20
            Residual2D(
                96, 128, (2, 1)
            ),
        )

    def forward(self, images):
        expected = (3, 160, 160)

        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != expected
        ):
            raise ValueError(
                f"Expected [B,3,160,160], "
                f"got {tuple(images.shape)}"
            )

        # White background -> zero
        x = images - 1.0

        features = self.blocks(
            self.stem(x)
        )

        if features.shape[-1] != 20:
            raise RuntimeError(
                "Expected 20 day columns, "
                f"got {features.shape[-1]}"
            )

        # Pool vertical axis only.
        # Keep all 20 trading-day positions.
        return features.mean(dim=2)


class TemporalCausalBlock(nn.Module):
    def __init__(
        self,
        channels=128,
        dilation=1,
    ):
        super().__init__()

        self.pad = 2 * dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )

        self.norm1 = nn.LayerNorm(
            channels
        )

        self.norm2 = nn.LayerNorm(
            channels
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Left padding for causal convolution
        z = self.conv1(
            F.pad(
                x,
                (self.pad, 0)
            )
        )

        z = self.norm1(
            z.transpose(1, 2)
        ).transpose(1, 2)

        z = self.dropout(
            F.gelu(z)
        )

        z = self.conv2(
            F.pad(
                z,
                (self.pad, 0)
            )
        )

        z = self.norm2(
            z.transpose(1, 2)
        ).transpose(1, 2)

        return F.gelu(
            x + self.dropout(z)
        )


class ChartTemporalClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()

        if variant not in VARIANTS:
            raise ValueError(variant)

        self.variant = variant

        # Identical image backbone
        # for all three variants
        self.encoder = SharedChartCNN()

        if variant == "cnn_tcn":
            self.temporal = nn.Sequential(
                TemporalCausalBlock(
                    128, dilation=1
                ),
                TemporalCausalBlock(
                    128, dilation=2
                ),
                TemporalCausalBlock(
                    128, dilation=4
                ),
            )

        elif variant == "cnn_transformer":
            self.positional = nn.Parameter(
                torch.zeros(1, 20, 128)
            )

            nn.init.trunc_normal_(
                self.positional,
                std=0.02,
            )

            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.temporal = nn.TransformerEncoder(
                layer,
                num_layers=2,
            )

        self.norm = nn.LayerNorm(128)

        self.head = nn.Sequential(
            nn.Linear(
                128, 64
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                64, 2
            ),
        )

    def forward(self, images):
        # [B, 128, 20]
        days = self.encoder(images)

        if self.variant == "resnet":
            pooled = days.mean(dim=-1)

        elif self.variant == "cnn_tcn":
            # Temporal 1D convolutions
            temporal_features = (
                self.temporal(days)
            )

            pooled = temporal_features.mean(
                dim=-1
            )

        else:
            # [B, 20, 128]
            sequence = (
                days.transpose(1, 2)
                + self.positional
            )

            sequence = self.temporal(
                sequence
            )

            pooled = sequence.mean(
                dim=1
            )

        return self.head(
            self.norm(pooled)
        )


def score_metrics(rows, probabilities):
    y = rows[
        "label"
    ].to_numpy(dtype=np.int64)

    p = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predicted = (
        p >= 0.5
    ).astype(np.int64)

    table = rows[
        ["decision_date", "return_5d"]
    ].copy()

    table["label"] = y
    table["p"] = p

    table["date_p"] = (
        table.groupby("decision_date")[
            "p"
        ].transform("mean")
    )

    daily_aucs = []
    rank_ics = []

    for _, group in table.groupby(
        "decision_date",
        sort=False,
    ):
        if group["label"].nunique() == 2:
            daily_aucs.append(
                roc_auc_score(
                    group["label"],
                    group["p"],
                )
            )

        if (
            group["p"].nunique() > 1
            and group["return_5d"].nunique() > 1
        ):
            rank_ics.append(
                group["p"].corr(
                    group["return_5d"],
                    method="spearman",
                )
            )

    return {
        "auc": float(
            roc_auc_score(y, p)
        ),

        "date_auc": float(
            roc_auc_score(
                y,
                table["date_p"],
            )
        ),

        "daily_auc": float(
            np.nanmean(daily_aucs)
        ),

        "rank_ic": float(
            np.nanmean(rank_ics)
        ),

        "accuracy": float(
            accuracy_score(
                y,
                predicted,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                predicted,
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
    }


def evaluate(
    model,
    loader,
    rows,
    device,
    return_predictions=False,
):
    model.eval()

    labels = []
    probabilities = []

    loss_sum = 0.0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss_sum += (
                F.cross_entropy(
                    logits,
                    targets.to(
                        device,
                        non_blocking=True,
                    ),
                    reduction="sum",
                ).item()
            )

            labels.append(
                targets.numpy()
            )

            probabilities.append(
                torch.softmax(
                    logits.float(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probabilities)

    if not np.array_equal(
        y,
        rows["label"].to_numpy(
            np.int64
        ),
    ):
        raise RuntimeError(
            "Evaluation labels and "
            "metadata order mismatch"
        )

    result = score_metrics(
        rows, p
    )

    result["loss"] = float(
        loss_sum / len(y)
    )

    if return_predictions:
        return result, p

    return result


def train_variant(
    variant,
    args,
    train_data,
    dev_data,
    train_rows,
    dev_rows,
    device,
):
    set_seed(args.seed)

    model = ChartTemporalClassifier(
        variant
    ).to(device)

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
        f"cnn_temporal_{variant}_"
        f"seed{args.seed}_dev2021"
    )

    metrics_dir = Path(
        "outputs/metrics"
    )

    checkpoint_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{prefix}_best.pt"
    )

    history_path = (
        metrics_dir
        / f"{prefix}_history.csv"
    )

    best_auc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    history = []

    start = time.perf_counter()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[MODEL] {variant} "
        f"| params={parameters:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        train_y = []
        train_p = []
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            y_device = targets.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = F.cross_entropy(
                logits,
                y_device,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_loss += (
                loss.item() * len(targets)
            )

            train_y.append(
                targets.numpy()
            )

            train_p.append(
                torch.softmax(
                    logits.detach().float(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

        online_train_auc = roc_auc_score(
            np.concatenate(train_y),
            np.concatenate(train_p),
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
                    "model_name": (
                        "cnn_temporal_comparison"
                    ),
                    "variant": variant,
                    "model_state": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {
                        "variant": variant,
                    },
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                    "dev_metrics": dev,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,

            "train_auc_online": float(
                online_train_auc
            ),

            "train_loss_online": float(
                train_loss / len(train_data)
            ),

            **{
                f"dev_{key}": value
                for key, value in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"[{variant}] "
            f"ep={epoch:02d}/{args.epochs} "
            f"train_auc={online_train_auc:.4f} "
            f"dev_auc={dev['auc']:.4f} "
            f"daily_auc={dev['daily_auc']:.4f} "
            f"IC={dev['rank_ic']:+.4f} "
            f"bal={dev['balanced_accuracy']:.4f} "
            f"up={dev['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {variant}",
                flush=True,
            )
            break

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    best, p = evaluate(
        model,
        dev_loader,
        dev_rows,
        device,
        return_predictions=True,
    )

    predictions = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    predictions["prob_up"] = p

    predictions.to_csv(
        metrics_dir
        / f"{prefix}_dev_predictions.csv",
        index=False,
    )

    return {
        "variant": variant,
        "params": parameters,
        "best_epoch": best_epoch,

        **{
            f"dev_{key}": value
            for key, value in best.items()
        },

        "seconds": round(
            time.perf_counter() - start,
            1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
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

    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("data/charts"),
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.weight_decay < 0
        or args.num_workers < 0
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

    if device.type == "cuda":
        print(
            f"[GPU] "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "[PROTOCOL] Train 2015-2020 "
        "| Dev 2021 "
        "| 2022-2025 untouched"
    )

    train_rows, dev_rows = load_dev_split(
        Path("data/processed/samples.csv")
    )

    train_data = ChartOnlyDataset(
        train_rows,
        args.charts_dir,
    )

    dev_data = ChartOnlyDataset(
        dev_rows,
        args.charts_dir,
    )

    results = []

    summary_path = Path(
        "outputs/metrics/"
        "cnn_temporal_dev2021_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for variant in args.variants:
        result = train_variant(
            variant,
            args,
            train_data,
            dev_data,
            train_rows,
            dev_rows,
            device,
        )

        results.append(result)

        pd.DataFrame(results).to_csv(
            summary_path,
            index=False,
        )

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(results).to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print(f"[SAVED] {summary_path}")

    print(
        "[NOTE] Original chart images only; "
        "no Market Context or 2022-2025 data."
    )


if __name__ == "__main__":
    main()
