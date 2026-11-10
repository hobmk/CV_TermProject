
"""Jiang, Kelly & Xiu (2023) I20/R5 reproduction.

Adapted to the 100-stock CV TermProject dataset.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageDraw
from numpy.lib.format import open_memmap

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import Dataset, DataLoader


HEIGHT = 64
WIDTH = 60
WINDOW = 20
HORIZON = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prices(path):
    df = pd.read_csv(
        path,
        dtype={"Date": str},
    ).sort_values("Date").reset_index(drop=True)

    required = {
        "Date", "Open", "High", "Low",
        "Close", "Adj Close", "Volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing {missing}"
        )

    df = df.loc[
        (df.Date >= "2015-01-01")
        & (df.Date < "2022-01-01")
    ].copy()

    if df.empty or df.Date.duplicated().any():
        raise ValueError(
            f"{path}: invalid dates"
        )

    cols = sorted(required - {"Date"})

    for col in cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    values = df[cols].to_numpy(float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{path}: NaN or infinite values"
        )

    price_cols = [
        "Open", "High", "Low",
        "Close", "Adj Close",
    ]

    if (df[price_cols] <= 0).any().any():
        raise ValueError(
            f"{path}: nonpositive prices"
        )

    if (df.Volume < 0).any():
        raise ValueError(
            f"{path}: negative volume"
        )

    # Dividend-adjusted Yahoo price representation.
    # Approximation to CRSP-adjusted price series.
    factor = (
        df["Adj Close"] / df["Close"]
    ).to_numpy(float)

    for col in (
        "Open", "High", "Low", "Close"
    ):
        df["A_" + col] = (
            df[col].to_numpy(float)
            * factor
        )

    # Historical moving average:
    # only current/past closes, no future data.
    df["MA20"] = (
        df["A_Close"]
        .rolling(20, min_periods=20)
        .mean()
    )

    return df.reset_index(drop=True)


def render_i20(window):
    if len(window) != 20:
        raise ValueError(
            "Image requires 20 trading days"
        )

    columns = [
        "A_Open", "A_High",
        "A_Low", "A_Close", "MA20",
    ]

    values = window[
        columns
    ].to_numpy(float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Incomplete OHLC/MA values"
        )

    lo = float(values.min())
    hi = float(values.max())

    span = max(hi - lo, 1e-12)

    # Price panel: rows 0..50
    y = np.rint(
        (hi - values) / span * 50
    ).astype(int).clip(0, 50)

    volume = window[
        "Volume"
    ].to_numpy(float)

    max_volume = max(
        float(volume.max()),
        1.0,
    )

    image = Image.new(
        "L",
        (WIDTH, HEIGHT),
        0,
    )

    draw = ImageDraw.Draw(image)

    # Moving average
    ma_points = [
        (3 * i + 1, int(y[i, 4]))
        for i in range(20)
    ]

    draw.line(
        ma_points,
        fill=255,
        width=1,
    )

    for i in range(20):
        x = 3 * i + 1

        op, high, low, cl = [
            int(v)
            for v in y[i, :4]
        ]

        # OHLC vertical high-low bar
        draw.line(
            (x, high, x, low),
            fill=255,
            width=1,
        )

        # Open left, close right
        draw.point(
            (x - 1, op),
            fill=255,
        )

        draw.point(
            (x + 1, cl),
            fill=255,
        )

        # Volume: rows 52..63
        n = (
            int(np.ceil(
                12 * volume[i] / max_volume
            ))
            if volume[i] > 0
            else 0
        )

        if n > 0:
            draw.line(
                (x, 63, x, 64 - n),
                fill=255,
                width=1,
            )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def prepare(args):
    source = args.source
    output = args.output

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        source / "processed" / "samples.csv",
        dtype={
            "ticker": str,
            "decision_date": str,
            "window_start": str,
            "target_date": str,
        },
    )

    required = {
        "ticker", "decision_date",
        "window_start", "target_date",
        "return_5d", "label",
    }

    if not required.issubset(metadata.columns):
        raise ValueError(
            "Missing metadata columns"
        )

    tickers = sorted(
        metadata.ticker.unique()
    )

    if len(tickers) != 100:
        raise ValueError(
            f"Expected 100 stocks, "
            f"found {len(tickers)}"
        )

    if metadata.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError(
            "Duplicate ticker/date"
        )

    masks = {
        "train": (
            (metadata.decision_date < "2021-01-01")
            & (metadata.target_date < "2021-01-01")
        ),
        "dev": (
            (metadata.window_start >= "2021-01-01")
            & (metadata.decision_date < "2022-01-01")
            & (metadata.target_date < "2022-01-01")
        ),
    }

    frames = {}

    for split, mask in masks.items():
        frames[split] = (
            metadata.loc[mask]
            .sort_values(
                ["decision_date", "ticker"]
            )
            .reset_index(drop=True)
        )

    if (
        frames["train"].target_date.max()
        >= frames["dev"].window_start.min()
    ):
        raise ValueError(
            "Train/Dev temporal overlap"
        )

    # 20-day MA for each of 20 input days:
    # require 19 extra historical days.
    first_valid = {}

    for ticker in tickers:
        df = load_prices(
            source / "raw" / f"{ticker}.csv"
        )

        if len(df) < 44:
            raise ValueError(
                f"{ticker}: insufficient data"
            )

        first_valid[ticker] = (
            df.Date.iloc[38]
        )

    for split in ("train", "dev"):
        frame = frames[split]

        frame = frame.loc[
            frame.decision_date
            >= frame.ticker.map(first_valid)
        ].copy()

        frame = (
            frame.sort_values(
                ["decision_date", "ticker"]
            )
            .reset_index(drop=True)
        )

        if frame.empty:
            raise ValueError(
                f"{split}: empty dataset"
            )

        frames[split] = frame

        print(
            f"[SPLIT] {split} "
            f"N={len(frame):,} "
            f"days={frame.decision_date.nunique()} "
            f"tickers={frame.ticker.nunique()}",
            flush=True,
        )

    for split, frame in frames.items():
        image_path = (
            output / f"{split}_images.npy"
        )

        meta_path = (
            output / f"{split}_meta.csv"
        )

        if (
            image_path.exists()
            or meta_path.exists()
        ) and not args.overwrite:
            raise FileExistsError(
                f"{split}: output already exists. "
                "Use --overwrite to rebuild."
            )

        images = open_memmap(
            image_path,
            mode="w+",
            dtype="uint8",
            shape=(
                len(frame), 1, 64, 60
            ),
        )

        labels = np.zeros(
            len(frame),
            dtype=np.int64,
        )

        returns = np.zeros(
            len(frame),
            dtype=np.float64,
        )

        started = time.perf_counter()

        print(
            f"[RENDER] {split}",
            flush=True,
        )

        for k, (ticker, group) in enumerate(
            frame.groupby(
                "ticker",
                sort=True,
            ),
            start=1,
        ):
            df = load_prices(
                source / "raw" / f"{ticker}.csv"
            )

            date_to_idx = {
                date: i
                for i, date in enumerate(df.Date)
            }

            adjusted_close = (
                df["A_Close"].to_numpy(float)
            )

            for row in group.itertuples():
                i = date_to_idx.get(
                    row.decision_date
                )

                j = date_to_idx.get(
                    row.target_date
                )

                if (
                    i is None
                    or j is None
                    or i < 38
                    or j != i + 5
                ):
                    raise ValueError(
                        f"{ticker} "
                        f"{row.decision_date}: "
                        "target mismatch"
                    )

                if (
                    df.Date.iloc[i - 19]
                    != row.window_start
                ):
                    raise ValueError(
                        f"{ticker}: window mismatch"
                    )

                window = df.iloc[
                    i - 19:i + 1
                ]

                images[row.Index, 0] = (
                    render_i20(window)
                )

                future_return = (
                    adjusted_close[j]
                    / adjusted_close[i]
                    - 1.0
                )

                returns[row.Index] = (
                    future_return
                )

                labels[row.Index] = int(
                    future_return > 0
                )

            if k % 10 == 0:
                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    f"[{split}] "
                    f"{k}/100 tickers "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

        images.flush()
        del images

        result = frame.copy()

        result["unadjusted_return_5d"] = (
            result["return_5d"]
        )

        result["return_5d"] = returns
        result["label"] = labels

        result.to_csv(
            meta_path,
            index=False,
        )

        counts = result.groupby(
            "decision_date"
        ).ticker.nunique()

        print(
            f"[SAVED] {split} "
            f"N={len(result):,} "
            f"Up={labels.mean():.2%} "
            f"stocks/day="
            f"{counts.min()}..{counts.max()}",
            flush=True,
        )

    preview_data = np.load(
        output / "train_images.npy",
        mmap_mode="r",
    )

    preview = np.array(
        preview_data[0, 0]
    )

    Image.fromarray(preview).save(
        output / "preview_i20.png"
    )

    print(
        "[READY]",
        output,
    )

    print(
        "[NOTE] All labels were recomputed "
        "using adjusted close prices."
    )


class Jkx20(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 64,
                kernel_size=(5, 3),
                stride=(3, 1),
                dilation=(2, 1),
                padding=(3, 1),
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(
                64, 128,
                kernel_size=(5, 3),
                padding=(3, 1),
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(
                128, 256,
                kernel_size=(5, 3),
                padding=(2, 1),
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 1)),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(46080, 2),
        )

        self.apply(self.initialize)

        params = sum(
            p.numel()
            for p in self.parameters()
        )

        if params != 708866:
            raise AssertionError(
                f"Parameter mismatch: {params}"
            )

    @staticmethod
    def initialize(module):
        if isinstance(
            module,
            (nn.Linear, nn.Conv2d),
        ):
            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

    def forward(self, x):
        x = self.features(x)

        if tuple(x.shape[1:]) != (
            256, 3, 60
        ):
            raise RuntimeError(
                f"Invalid feature shape: "
                f"{tuple(x.shape)}"
            )

        # Raw logits for CrossEntropyLoss.
        return self.head(
            x.flatten(1)
        )


class MemmapCharts(Dataset):
    def __init__(
        self,
        image_path,
        metadata_path,
        indices=None,
    ):
        self.images = np.load(
            image_path,
            mmap_mode="r",
        )

        self.meta = pd.read_csv(
            metadata_path,
            dtype={
                "ticker": str,
                "decision_date": str,
            },
        )

        if len(self.images) != len(self.meta):
            raise ValueError(
                "Image/metadata length mismatch"
            )

        self.indices = (
            np.arange(len(self.meta))
            if indices is None
            else np.asarray(indices)
        )

        self.labels = (
            self.meta.label.to_numpy(
                np.int64
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        i = int(self.indices[pos])

        pixels = torch.from_numpy(
            np.array(
                self.images[i],
                copy=True,
            )
        ).float().div_(255.0)

        return (
            pixels,
            int(self.labels[i]),
        )


def evaluate(model, loader, device):
    model.eval()

    labels = []
    probabilities = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    with torch.inference_mode():
        for images, y in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = y.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            total_loss += criterion(
                logits,
                targets,
            ).item()

            probabilities.append(
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            labels.append(
                y.numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probabilities)

    metrics = {
        "loss": float(
            total_loss / len(y)
        ),
        "auc": float(
            roc_auc_score(y, p)
        ),
        "accuracy": float(
            accuracy_score(y, p >= 0.5)
        ),
        "balanced_acc": float(
            balanced_accuracy_score(
                y, p >= 0.5
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                p >= 0.5,
                average="macro",
                zero_division=0,
            )
        ),
        "up_ratio": float(
            (p >= 0.5).mean()
        ),
        "prob_std": float(
            p.std()
        ),
    }

    return metrics, p


def cross_section_metrics(meta, probs):
    df = meta.copy()
    df["prob_up"] = probs

    daily_auc = []
    daily_ic = []
    top_excess = []
    spreads = []

    for _, group in df.groupby(
        "decision_date"
    ):
        if group.label.nunique() == 2:
            daily_auc.append(
                roc_auc_score(
                    group.label,
                    group.prob_up,
                )
            )

        if (
            group.prob_up.nunique() > 1
            and group.return_5d.nunique() > 1
        ):
            daily_ic.append(
                group.prob_up.corr(
                    group.return_5d,
                    method="spearman",
                )
            )

        if len(group) >= 20:
            ranked = group.sort_values(
                ["prob_up", "ticker"]
            )

            top = (
                ranked.tail(10)
                .return_5d.mean()
            )

            bottom = (
                ranked.head(10)
                .return_5d.mean()
            )

            market = (
                ranked.return_5d.mean()
            )

            top_excess.append(
                top - market
            )

            spreads.append(
                top - bottom
            )

    return {
        "daily_auc": float(
            np.mean(daily_auc)
        ),
        "rank_ic": float(
            np.nanmean(daily_ic)
        ),
        "top10_excess_5d": float(
            np.mean(top_excess)
        ),
        "top10_bottom10_spread_5d": float(
            np.mean(spreads)
        ),
    }


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = args.output

    train_images = (
        root / "train_images.npy"
    )

    train_metadata = (
        root / "train_meta.csv"
    )

    dev_images = (
        root / "dev_images.npy"
    )

    dev_metadata = (
        root / "dev_meta.csv"
    )

    for path in (
        train_images,
        train_metadata,
        dev_images,
        dev_metadata,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. "
                "Run --stage prepare first."
            )

    meta = pd.read_csv(
        train_metadata,
        dtype={
            "ticker": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    indices = np.arange(len(meta))

    if args.val_policy == "paper_random":
        train_idx, val_idx = (
            train_test_split(
                indices,
                test_size=0.30,
                random_state=args.seed,
                stratify=meta.label.to_numpy(
                    int
                ),
            )
        )

        print(
            "[VALIDATION] Paper-style "
            "random 70/30 split"
        )

        print(
            "[WARNING] Adjacent overlapping "
            "windows may enter both subsets."
        )

    else:
        train_mask = (
            meta.target_date < "2020-01-01"
        )

        val_mask = (
            (meta.decision_date >= "2020-01-01")
            & (meta.target_date < "2021-01-01")
        )

        train_idx = indices[
            train_mask.to_numpy()
        ]

        val_idx = indices[
            val_mask.to_numpy()
        ]

        if (
            meta.iloc[train_idx].target_date.max()
            >= meta.iloc[val_idx].decision_date.min()
        ):
            raise ValueError(
                "Chronological split overlap"
            )

        print(
            "[VALIDATION] Chronological "
            "2015-2019 / 2020"
        )

    train_ds = MemmapCharts(
        train_images,
        train_metadata,
        train_idx,
    )

    val_ds = MemmapCharts(
        train_images,
        train_metadata,
        val_idx,
    )

    dev_ds = MemmapCharts(
        dev_images,
        dev_metadata,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=(
            torch.Generator().manual_seed(
                args.seed
            )
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = Jkx20().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    output_dir = Path(
        "outputs/jkx_i20_100"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"{args.val_policy}_seed{args.seed}"
    )

    checkpoint_path = (
        output_dir
        / f"{prefix}_best.pt"
    )

    history_path = (
        output_dir
        / f"{prefix}_history.csv"
    )

    if (
        checkpoint_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{checkpoint_path} exists. "
            "Use --overwrite only if intended."
        )

    print(
        f"[DEVICE] {device}"
    )

    print(
        f"[MODEL] Jkx20 "
        f"params=708866 "
        f"train={len(train_ds):,} "
        f"val={len(val_ds):,} "
        f"dev2021={len(dev_ds):,}"
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    history = []
    started = time.perf_counter()

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        n = 0
        loss_sum = 0.0

        train_y = []
        train_p = []

        for step, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = nn.functional.cross_entropy(
                logits,
                targets,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            n += len(labels)

            loss_sum += (
                loss.item() * len(labels)
            )

            train_y.append(
                labels.numpy()
            )

            train_p.append(
                torch.softmax(
                    logits.detach(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            if step % args.log_steps == 0:
                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    f"[TRAIN] "
                    f"epoch={epoch} "
                    f"batch={step}/"
                    f"{len(train_loader)} "
                    f"loss={loss_sum / n:.5f} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

        train_auc = roc_auc_score(
            np.concatenate(train_y),
            np.concatenate(train_p),
        )

        val, _ = evaluate(
            model,
            val_loader,
            device,
        )

        if (
            val["loss"]
            < best_loss - 1e-6
        ):
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_policy": args.val_policy,
                    "model_state": (
                        model.state_dict()
                    ),
                    "val_metrics": val,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_loss_online": (
                loss_sum / n
            ),
            "train_auc_online": (
                float(train_auc)
            ),
            **{
                f"val_{key}": value
                for key, value in val.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"[EPOCH] {epoch:02d} "
            f"train_loss={loss_sum / n:.5f} "
            f"train_auc={train_auc:.4f} "
            f"val_loss={val['loss']:.5f} "
            f"val_auc={val['auc']:.4f} "
            f"val_bal={val['balanced_acc']:.4f} "
            f"val_up={val['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                "[EARLY STOP] Validation Loss",
                flush=True,
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    # 2021 is evaluated only after selecting
    # the best validation checkpoint.
    dev, probs = evaluate(
        model,
        dev_loader,
        device,
    )

    dev_frame = dev_ds.meta.copy()

    dev_frame["prob_up"] = probs

    dev_frame.to_csv(
        output_dir
        / f"{prefix}_dev2021_predictions.csv",
        index=False,
    )

    cross = cross_section_metrics(
        dev_ds.meta,
        probs,
    )

    summary = {
        "best_epoch": best_epoch,
        "seed": args.seed,
        "val_policy": args.val_policy,
        **{
            f"dev_{key}": value
            for key, value in dev.items()
        },
        **{
            f"dev_{key}": value
            for key, value in cross.items()
        },
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        / f"{prefix}_summary.csv",
        index=False,
    )

    print("\n[FINAL DEV2021]")

    for key, value in summary.items():
        print(
            f"{key}: {value}",
            flush=True,
        )

    print(
        "[NOTE] 2021 Dev was not used "
        "for early stopping."
    )

    print(
        "[NOTE] 2022-2025 not accessed."
    )


def self_test():
    rng = np.random.default_rng(7)

    close = 100 * np.exp(
        np.cumsum(
            rng.normal(0, 0.02, 50)
        )
    )

    df = pd.DataFrame({
        "A_Open": (
            close * rng.uniform(
                0.997, 1.003, 50
            )
        ),
        "A_High": close * 1.012,
        "A_Low": close * 0.985,
        "A_Close": close,
        "MA20": (
            pd.Series(close)
            .rolling(20).mean()
        ),
        "Volume": rng.integers(
            1000, 5000, 50
        ),
    })

    image = render_i20(
        df.iloc[25:45]
    )

    assert image.shape == (64, 60)

    assert set(
        np.unique(image)
    ).issubset({0, 255})

    assert image.sum() > 0

    model = Jkx20()

    x = torch.from_numpy(
        image.copy()
    ).reshape(
        1, 1, 64, 60
    ).float() / 255.0

    logits = model(x)

    assert logits.shape == (1, 2)

    loss = nn.functional.cross_entropy(
        logits,
        torch.tensor([1]),
    )

    loss.backward()

    assert (
        model.features[0].weight.grad
        is not None
    )

    print(
        "[SELF-TEST PASS] "
        "64x60 binary image | "
        "708866 parameters | "
        "logits [1,2] | "
        "backward OK"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=[
            "selftest",
            "prepare",
            "train",
        ],
        default="selftest",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/exp100"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exp100_jkx"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--val-policy",
        choices=[
            "paper_random",
            "chronological",
        ],
        default="paper_random",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.workers < 0
        or args.log_steps < 1
    ):
        parser.error(
            "Invalid numeric argument"
        )

    if args.stage == "selftest":
        self_test()

    elif args.stage == "prepare":
        prepare(args)

    else:
        train(args)


if __name__ == "__main__":
    main()
