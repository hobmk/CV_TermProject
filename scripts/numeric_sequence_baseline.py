
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


WINDOW = 20
HORIZON = 5

TRAIN_END = "2020-12-31"
DEV_START = "2021-01-01"
DEV_END = "2021-12-31"

VARIANTS = (
    "mlp_chart",
    "cnn_chart",
    "mlp_aug",
    "cnn_aug",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(path):
    df = pd.read_csv(
        path,
        dtype={
            c: str
            for c in (
                "ticker",
                "split",
                "window_start",
                "decision_date",
                "target_date",
            )
        },
    )

    required = {
        "ticker",
        "split",
        "window_start",
        "decision_date",
        "target_date",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"samples.csv missing columns: {sorted(missing)}"
        )

    # Original train split contains 2015-2021.
    # Repartition without modifying samples.csv.
    original_train = df["split"].eq("train")

    train = df[
        original_train
        & df["decision_date"].le(TRAIN_END)
        & df["target_date"].le(TRAIN_END)
    ].copy()

    dev = df[
        original_train
        & df["window_start"].ge(DEV_START)
        & df["decision_date"].le(DEV_END)
        & df["target_date"].le(DEV_END)
    ].copy()

    if train.empty or dev.empty:
        raise RuntimeError(
            "No train/dev examples; verify 2021 "
            "samples exist in original train split."
        )

    if train["target_date"].max() >= dev["window_start"].min():
        raise AssertionError(
            "Train targets overlap development input window"
        )

    for name, part in (
        ("train", train),
        ("dev", dev),
    ):
        if not part["label"].isin((0, 1)).all():
            raise ValueError(
                f"Invalid labels in {name}"
            )

        if part.duplicated(
            ["ticker", "decision_date"]
        ).any():
            raise ValueError(
                f"Duplicate samples in {name}"
            )

        print(
            f"[{name.upper()}] "
            f"N={len(part):,} "
            f"up={part['label'].mean():.4f} "
            f"dates={part['decision_date'].min()}"
            f"..{part['decision_date'].max()}"
        )

    return (
        train.reset_index(drop=True),
        dev.reset_index(drop=True),
    )


def create_sequences(samples, raw_dir):
    """
    Build:
      chart: [N,20,5]
      augmented: [N,20,7]
    """

    n = len(samples)

    x_chart = np.empty(
        (n, WINDOW, 5),
        dtype=np.float32,
    )

    x_aug = np.empty(
        (n, WINDOW, 7),
        dtype=np.float32,
    )

    indexed = samples.copy()
    indexed["row_number"] = np.arange(n)

    for ticker, part in indexed.groupby(
        "ticker", sort=True
    ):
        raw_path = raw_dir / f"{ticker}.csv"

        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        raw = pd.read_csv(raw_path)

        required = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        if not required.issubset(raw.columns):
            raise ValueError(
                f"Bad OHLCV columns: {raw_path}"
            )

        raw["Date"] = (
            pd.to_datetime(raw["Date"])
            .dt.strftime("%Y-%m-%d")
        )

        raw = (
            raw.sort_values("Date")
            .reset_index(drop=True)
        )

        if raw["Date"].duplicated().any():
            raise ValueError(
                f"Duplicate price dates: {ticker}"
            )

        date_to_idx = {
            d: i
            for i, d in enumerate(
                raw["Date"].tolist()
            )
        }

        values = raw[
            ["Open", "High", "Low", "Close", "Volume"]
        ].to_numpy(np.float64)

        if (
            not np.isfinite(values).all()
            or (values[:, :4] <= 0).any()
        ):
            raise ValueError(
                f"Invalid prices: {ticker}"
            )

        if (values[:, 4] < 0).any():
            raise ValueError(
                f"Negative volume: {ticker}"
            )

        for row in part.itertuples(index=False):
            end = date_to_idx.get(
                row.decision_date
            )

            if (
                end is None
                or end < WINDOW - 1
                or end + HORIZON >= len(raw)
            ):
                raise ValueError(
                    f"Missing price window/target: "
                    f"{ticker} {row.decision_date}"
                )

            actual_start = raw.at[
                end - WINDOW + 1, "Date"
            ]

            actual_target = raw.at[
                end + HORIZON, "Date"
            ]

            if (
                actual_start != row.window_start
                or actual_target != row.target_date
            ):
                raise ValueError(
                    f"Misaligned window/target: "
                    f"{ticker} {row.decision_date}"
                )

            w = values[
                end - WINDOW + 1:end + 1
            ]

            # Window-level price normalization
            p_low = w[:, 2].min()
            p_high = w[:, 1].max()

            price_span = max(
                float(p_high - p_low),
                1e-8,
            )

            low_with_margin = (
                p_low - 0.05 * price_span
            )

            full_span = 1.10 * price_span

            norm_price = (
                (w[:, :4] - low_with_margin)
                / full_span
            )

            # Relative volume normalization
            max_volume = max(
                float(w[:, 4].max()),
                1.0,
            )

            norm_volume = (
                w[:, 4:5] / max_volume
            )

            chart = np.concatenate(
                (norm_price, norm_volume),
                axis=1,
            )

            # Two additional scale features
            # computed only from input window
            log_range = np.log(
                p_high / p_low
            )

            log_close = np.log(
                w[:, 3]
            )

            daily_vol = np.diff(
                log_close
            ).std(ddof=0)

            scale = np.repeat(
                [[log_range, daily_vol]],
                WINDOW,
                axis=0,
            )

            augmented = np.concatenate(
                (chart, scale),
                axis=1,
            )

            x_chart[row.row_number] = chart
            x_aug[row.row_number] = augmented

            future_up = int(
                values[end + HORIZON, 3]
                > values[end, 3]
            )

            if future_up != int(row.label):
                raise ValueError(
                    f"Label mismatch: "
                    f"{ticker} {row.decision_date}"
                )

        print(
            f"  [EXTRACTED] {ticker}: "
            f"{len(part):,}",
            flush=True,
        )

    return x_chart, x_aug


def normalize_augmented(train_x, dev_x):
    """
    Standardize extra features only.
    Statistics come from Train.
    """

    train_x = train_x.copy()
    dev_x = dev_x.copy()

    mean = train_x[
        :, 0, 5:
    ].mean(
        axis=0,
        dtype=np.float64,
    )

    std = train_x[
        :, 0, 5:
    ].std(
        axis=0,
        dtype=np.float64,
    )

    std = np.maximum(std, 1e-8)

    train_x[:, :, 5:] = np.clip(
        (train_x[:, :, 5:] - mean) / std,
        -5,
        5,
    )

    dev_x[:, :, 5:] = np.clip(
        (dev_x[:, :, 5:] - mean) / std,
        -5,
        5,
    )

    scaler = {
        "extra_mean": mean.tolist(),
        "extra_std": std.tolist(),
    }

    return train_x, dev_x, scaler


class NumericMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                WINDOW * channels, 128
            ),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class NumericCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                channels, 32, 3, padding=1
            ),
            nn.GELU(),

            nn.Conv1d(
                32, 64, 3, padding=1
            ),
            nn.GELU(),

            nn.Conv1d(
                64, 64, 3, padding=1
            ),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * WINDOW, 64
            ),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        # [B,20,C] -> [B,C,20]
        x = x.transpose(1, 2)

        x = self.conv(x)

        return self.head(x)


def evaluate(model, loader, device):
    model.eval()

    labels = []
    probs = []
    loss_sum = 0.0

    with torch.inference_mode():
        for x, y in loader:
            logits = model(
                x.to(device)
            )

            loss_sum += (
                nn.functional.cross_entropy(
                    logits,
                    y.to(device),
                    reduction="sum",
                ).item()
            )

            labels.append(
                y.numpy()
            )

            probs.append(
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probs)

    pred = (
        p >= 0.5
    ).astype(np.int64)

    return {
        "loss": loss_sum / len(y),

        "auroc": float(
            roc_auc_score(y, p)
        ),

        "accuracy": float(
            accuracy_score(y, pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),

        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),

        "up_ratio": float(
            pred.mean()
        ),

        "prob_std": float(
            p.std()
        ),

        "confusion_matrix": (
            confusion_matrix(
                y, pred, labels=[0, 1]
            ).tolist()
        ),
    }


def train_variant(
    name,
    args,
    train_data,
    dev_data,
    train_meta,
    dev_meta,
    device,
    scaler,
):
    set_seed(args.seed)

    x_train = torch.from_numpy(
        np.ascontiguousarray(train_data)
    )

    x_dev = torch.from_numpy(
        np.ascontiguousarray(dev_data)
    )

    y_train = torch.tensor(
        train_meta["label"].to_numpy(
            np.int64
        )
    )

    y_dev = torch.tensor(
        dev_meta["label"].to_numpy(
            np.int64
        )
    )

    train_set = TensorDataset(
        x_train, y_train
    )

    dev_set = TensorDataset(
        x_dev, y_dev
    )

    shuffle_generator = (
        torch.Generator().manual_seed(
            args.seed
        )
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    train_eval = DataLoader(
        train_set,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    channels = x_train.shape[-1]

    if name.startswith("mlp_"):
        model = NumericMLP(channels)
    else:
        model = NumericCNN(channels)

    model = model.to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_epoch = 0
    best_dev_loss = float("inf")
    stale = 0

    history = []

    metrics_dir = Path(
        "outputs/metrics"
    )

    checkpoints_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"numeric_seq_{name}_"
        f"seed{args.seed}_dev2021"
    )

    history_path = (
        metrics_dir /
        f"{prefix}_history.csv"
    )

    checkpoint_path = (
        checkpoints_dir /
        f"{prefix}_best.pt"
    )

    begin = time.perf_counter()

    print(
        f"\n[MODEL] {name} "
        f"| channels={channels} "
        f"| params={params:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        for x, y in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = (
                nn.functional.cross_entropy(
                    logits, y
                )
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        tr = evaluate(
            model,
            train_eval,
            device,
        )

        dv = evaluate(
            model,
            dev_loader,
            device,
        )

        row = {
            "epoch": epoch,

            **{
                f"train_{k}": v
                for k, v in tr.items()
            },

            **{
                f"dev_{k}": v
                for k, v in dv.items()
            },
        }

        history.append(row)

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        improved = (
            dv["auroc"] > best_auc + 1e-5
            or (
                abs(
                    dv["auroc"] - best_auc
                ) <= 1e-5
                and dv["loss"] < best_dev_loss
            )
        )

        if improved:
            best_auc = dv["auroc"]
            best_dev_loss = dv["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model": name,
                    "channels": channels,
                    "state_dict": (
                        model.state_dict()
                    ),
                    "seed": args.seed,
                    "epoch": epoch,
                    "train_metrics": tr,
                    "dev_metrics": dv,
                    "train_end": TRAIN_END,
                    "dev_year": 2021,
                    "scaler": scaler,
                    "window": WINDOW,
                    "horizon": HORIZON,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        elapsed = (
            time.perf_counter()
            - begin
        )

        print(
            f"[{name}] "
            f"{epoch:02d}/{args.epochs:02d} "
            f"train_auc={tr['auroc']:.4f} "
            f"train_loss={tr['loss']:.4f} "
            f"dev_auc={dv['auroc']:.4f} "
            f"dev_bal={dv['balanced_accuracy']:.4f} "
            f"dev_f1={dv['macro_f1']:.4f} "
            f"dev_up={dv['up_ratio']:.3f} "
            f"prob_std={dv['prob_std']:.4f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {name} "
                f"after {epoch} epochs"
            )
            break

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if name.startswith("mlp_"):
        best_model = NumericMLP(channels)
    else:
        best_model = NumericCNN(channels)

    best_model = best_model.to(device)

    best_model.load_state_dict(
        saved["state_dict"]
    )

    best_dev = evaluate(
        best_model,
        dev_loader,
        device,
    )

    print(
        f"[BEST] {name} "
        f"epoch={best_epoch} "
        f"dev_auc={best_dev['auroc']:.4f} "
        f"dev_macro_f1="
        f"{best_dev['macro_f1']:.4f} "
        f"up_ratio="
        f"{best_dev['up_ratio']:.3f}"
    )

    return {
        "variant": name,
        "params": params,
        "best_epoch": best_epoch,
        "best_dev_auc": (
            best_dev["auroc"]
        ),
        "best_dev_balanced_acc": (
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
        "last_train_auc": (
            tr["auroc"]
        ),
        "best_checkpoint": str(
            checkpoint_path
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=[
            "mlp_chart",
            "cnn_chart",
        ],
    )

    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(
            "data/processed/samples.csv"
        ),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
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
    ):
        parser.error(
            "epochs, patience, batch-size "
            "and lr must be positive"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] "
        "Train: 2015-2020 | "
        "Dev: 2021 | "
        "No 2022-2025 evaluation"
    )

    train_meta, dev_meta = (
        load_metadata(args.samples)
    )

    print(
        "[EXTRACT] Train sequences..."
    )

    chart_train, aug_train = (
        create_sequences(
            train_meta,
            args.raw_dir,
        )
    )

    print(
        "[EXTRACT] Dev sequences..."
    )

    chart_dev, aug_dev = (
        create_sequences(
            dev_meta,
            args.raw_dir,
        )
    )

    aug_train, aug_dev, scaler = (
        normalize_augmented(
            aug_train,
            aug_dev,
        )
    )

    print(
        "[FEATURES] chart: "
        "OHLC min-max + relative volume "
        "[20,5]"
    )

    print(
        "[FEATURES] augmented: "
        "chart + log high/low range + "
        "realized log-return volatility "
        "[20,7]"
    )

    selected = []

    summary_path = Path(
        "outputs/metrics/"
        "numeric_seq_dev2021_"
        + "_".join(args.variants)
        + "_summary.csv"
    )

    for name in args.variants:
        is_aug = name.endswith(
            "_aug"
        )

        if is_aug:
            tr_x = aug_train
            dev_x = aug_dev
        else:
            tr_x = chart_train
            dev_x = chart_dev

        result = train_variant(
            name,
            args,
            tr_x,
            dev_x,
            train_meta,
            dev_meta,
            device,
            scaler if is_aug else None,
        )

        selected.append(result)

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            selected
        ).to_csv(
            summary_path,
            index=False,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(
            selected
        ).to_string(
            index=False
        )
    )

    print(
        f"[SAVED] {summary_path}"
    )

    print(
        "[NOTE] Old 2022-23 validation "
        "and 2024-25 test remain untouched"
    )


if __name__ == "__main__":
    main()
