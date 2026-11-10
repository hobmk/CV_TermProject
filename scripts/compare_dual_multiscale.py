
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from scripts.compare_day_market_context import (
    DayViTWithMarket,
    load_dev_split,
)


MODES = ("single", "dual", "multi")
DAYS = 20
SIDE = 160
DAY_WIDTH = 8


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw(raw_dir, tickers):
    stocks = {}

    for ticker in tickers:
        df = pd.read_csv(
            raw_dir / f"{ticker}.csv",
            dtype={"Date": str},
        )

        df = (
            df.loc[df["Date"] <= "2021-12-31"]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if df["Date"].duplicated().any():
            raise ValueError(
                f"{ticker}: duplicated date"
            )

        close = df["Close"].to_numpy(
            np.float64
        )

        if (
            (close <= 0).any()
            or not np.isfinite(close).all()
        ):
            raise ValueError(
                f"{ticker}: invalid Close"
            )

        stocks[ticker] = (
            np.log(close),
            {
                date: i
                for i, date in enumerate(df["Date"])
            },
            df["Date"].to_numpy(),
        )

    return stocks


def window_log_returns(row, stocks):
    logs, index, dates = stocks[row.ticker]

    i = index.get(row.decision_date)

    if (
        i is None
        or i < DAYS - 1
        or dates[i - DAYS + 1] != row.window_start
    ):
        raise ValueError(
            f"Bad window: "
            f"{row.ticker} / {row.decision_date}"
        )

    window = logs[i - DAYS + 1:i + 1]

    cumulative = window - window[0]

    daily = np.empty(
        DAYS,
        dtype=np.float64,
    )

    daily[0] = 0.0
    daily[1:] = np.diff(window)

    return cumulative, daily


def train_scaler(rows, stocks):
    cumulative = []
    daily = []

    for row in rows.itertuples(index=False):
        c, d = window_log_returns(
            row, stocks
        )

        cumulative.append(c)
        daily.append(d[1:])

    cum = np.abs(
        np.concatenate(cumulative)
    )

    day = np.abs(
        np.concatenate(daily)
    )

    result = {}

    for key, values in (
        ("cum", cum),
        ("day", day),
    ):
        positive = values[
            values > 1e-10
        ]

        result[f"{key}_scale"] = float(
            max(
                np.median(positive),
                1e-5,
            )
        )

        result[f"{key}_clip"] = float(
            max(
                np.quantile(values, 0.995),
                1e-4,
            )
        )

    return result


def compress(x, scale, clip):
    clamped = np.clip(
        x,
        -clip,
        clip,
    )

    return (
        np.arcsinh(clamped / scale)
        / np.arcsinh(clip / scale)
    )


def make_return_image(cum, day, scaler):
    image = Image.new(
        "L",
        (SIDE, SIDE),
        255,
    )

    draw = ImageDraw.Draw(image)

    # Upper panel: cumulative log return
    draw.line(
        (0, 51, SIDE - 1, 51),
        fill=220,
        width=1,
    )

    # Lower panel: daily log return
    draw.line(
        (0, 128, SIDE - 1, 128),
        fill=220,
        width=1,
    )

    c = compress(
        cum,
        scaler["cum_scale"],
        scaler["cum_clip"],
    )

    d = compress(
        day,
        scaler["day_scale"],
        scaler["day_clip"],
    )

    points = []

    for j in range(DAYS):
        x = j * DAY_WIDTH + 4

        points.append(
            (
                x,
                int(round(51 - 43 * c[j])),
            )
        )

        if j:
            y = int(
                round(128 - 23 * d[j])
            )

            draw.line(
                (x, 128, x, y),
                fill=35,
                width=3,
            )

    draw.line(
        points,
        fill=0,
        width=2,
    )

    return image


def prepare_return_images(
    train,
    dev,
    raw_dir,
    aux_dir,
    overwrite=False,
):
    stocks = load_raw(
        raw_dir,
        sorted(
            set(train.ticker)
            | set(dev.ticker)
        ),
    )

    # Fit image scaling using Train only
    scaler = train_scaler(
        train,
        stocks,
    )

    aux_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scale_path = (
        aux_dir
        / "train2015_2020_scaler.json"
    )

    if scale_path.exists():
        old = json.loads(
            scale_path.read_text(
                encoding="utf-8"
            )
        )

        if any(
            not np.isclose(
                scaler[key],
                old[key],
                atol=1e-10,
                rtol=1e-8,
            )
            for key in scaler
        ):
            raise RuntimeError(
                "Training scaler changed. "
                "Use a fresh --aux-dir."
            )

    else:
        if any(aux_dir.rglob("*.png")):
            raise RuntimeError(
                "Cached images without scaler "
                "metadata. Use a fresh --aux-dir."
            )

        scale_path.write_text(
            json.dumps(
                scaler,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        "[SCALE]",
        json.dumps(scaler),
        flush=True,
    )

    generated = 0
    skipped = 0

    for frame in (train, dev):
        for row in frame.itertuples(
            index=False
        ):
            output = (
                aux_dir
                / "train"
                / row.ticker
                / f"{row.decision_date}.png"
            )

            if (
                output.exists()
                and not overwrite
            ):
                skipped += 1
                continue

            cumulative, daily = (
                window_log_returns(
                    row,
                    stocks,
                )
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            make_return_image(
                cumulative,
                daily,
                scaler,
            ).save(
                output,
                optimize=True,
            )

            generated += 1

    print(
        f"[RETURN IMAGES] "
        f"generated={generated:,} "
        f"skipped={skipped:,}",
        flush=True,
    )

    return scaler


class DualDataset(Dataset):
    def __init__(
        self,
        frame,
        chart_dir,
        aux_dir,
    ):
        self.frame = frame.reset_index(
            drop=True
        )

        self.y = self.frame[
            "label"
        ].to_numpy(np.int64)

        self.original = []
        self.returns = []

        for row in self.frame.itertuples(
            index=False
        ):
            old = (
                chart_dir
                / "train"
                / row.ticker
                / f"{row.decision_date}.png"
            )

            new = (
                aux_dir
                / "train"
                / row.ticker
                / f"{row.decision_date}.png"
            )

            if (
                not old.exists()
                or not new.exists()
            ):
                raise FileNotFoundError(
                    f"Missing image: "
                    f"{old} or {new}"
                )

            self.original.append(old)
            self.returns.append(new)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        with Image.open(
            self.original[idx]
        ) as image:
            original = to_tensor(
                image.convert("RGB")
            )

        with Image.open(
            self.returns[idx]
        ) as image:
            returns = to_tensor(
                image.convert("L")
            )

        return (
            original,
            returns,
            int(self.y[idx]),
        )


class DayModel(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        # Existing 1-day-aligned ViT backbone
        self.base = DayViTWithMarket(
            use_market=False
        )

        if mode != "single":
            # Second view: one token per day
            self.return_patch = nn.Conv2d(
                1,
                128,
                kernel_size=(160, 8),
                stride=(160, 8),
            )

            self.return_gain = (
                nn.Parameter(
                    torch.tensor(1.0)
                )
            )

        if mode == "multi":
            # 2-day and 5-day tokens
            self.pos_2 = nn.Parameter(
                torch.zeros(1, 10, 128)
            )

            self.pos_5 = nn.Parameter(
                torch.zeros(1, 4, 128)
            )

            self.scale_1 = nn.Parameter(
                torch.zeros(1, 1, 128)
            )

            self.scale_2 = nn.Parameter(
                torch.zeros(1, 1, 128)
            )

            self.scale_5 = nn.Parameter(
                torch.zeros(1, 1, 128)
            )

            nn.init.trunc_normal_(
                self.pos_2,
                std=0.02,
            )

            nn.init.trunc_normal_(
                self.pos_5,
                std=0.02,
            )

    def forward(self, image, returns):
        batch = image.size(0)

        zero_context = image.new_zeros(
            (batch, 6)
        )

        if self.mode == "single":
            return self.base(
                image,
                zero_context,
            )

        backbone = self.base

        # View 1: original OHLCV image
        day = (
            backbone.patch_embed(
                image - 1.0
            )
            .flatten(2)
            .transpose(1, 2)
        )

        # View 2: return-strength image
        return_day = (
            self.return_patch(
                returns - 1.0
            )
            .flatten(2)
            .transpose(1, 2)
        )

        # Day-aligned fusion
        day = (
            day
            + self.return_gain * return_day
        )

        if self.mode == "multi":
            # 20 x 1-day tokens
            # 10 x 2-day tokens
            #  4 x 5-day tokens

            two = (
                day.reshape(
                    batch, 10, 2, 128
                )
                .mean(dim=2)
            )

            five = (
                day.reshape(
                    batch, 4, 5, 128
                )
                .mean(dim=2)
            )

            tokens = torch.cat(
                [
                    day
                    + backbone.pos_embed
                    + self.scale_1,

                    two
                    + self.pos_2
                    + self.scale_2,

                    five
                    + self.pos_5
                    + self.scale_5,
                ],
                dim=1,
            )

        else:
            tokens = (
                day + backbone.pos_embed
            )

        tokens = backbone.dropout(
            tokens
        )

        for block in backbone.blocks:
            tokens = block(tokens)

        image_features = backbone.norm(
            tokens.mean(dim=1)
        )

        # Same classifier dimensions as
        # the existing Day-Aligned ViT.
        # Market features are NOT used.
        context_features = (
            backbone.context_encoder(
                zero_context
            )
        )

        return backbone.head(
            torch.cat(
                [
                    image_features,
                    context_features,
                ],
                dim=1,
            )
        )


def evaluate(
    model,
    loader,
    rows,
    device,
    include_probs=False,
):
    model.eval()

    labels = []
    probs = []

    with torch.inference_mode():
        for image, returns, y in loader:
            logits = model(
                image.to(device),
                returns.to(device),
            )

            probs.append(
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            labels.append(
                y.numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probs)

    if not np.array_equal(
        y,
        rows.label.to_numpy(np.int64),
    ):
        raise RuntimeError(
            "Metadata order mismatch"
        )

    table = rows[
        ["decision_date", "return_5d"]
    ].copy()

    table["prob"] = p
    table["label"] = y

    daily_aucs = []
    daily_ics = []

    for _, group in table.groupby(
        "decision_date",
        sort=False,
    ):
        if group.label.nunique() == 2:
            daily_aucs.append(
                roc_auc_score(
                    group.label,
                    group.prob,
                )
            )

        if (
            group.prob.nunique() > 1
            and group.return_5d.nunique() > 1
        ):
            daily_ics.append(
                group.prob.corr(
                    group.return_5d,
                    method="spearman",
                )
            )

    table["date_mean"] = (
        table.groupby(
            "decision_date"
        ).prob.transform("mean")
    )

    predicted = (
        p >= 0.5
    ).astype(np.int64)

    metrics = {
        "auc": float(
            roc_auc_score(y, p)
        ),

        "date_auc": float(
            roc_auc_score(
                y,
                table.date_mean,
            )
        ),

        "daily_auc": float(
            np.nanmean(daily_aucs)
        ),

        "rank_ic": float(
            np.nanmean(daily_ics)
        ),

        "bal": float(
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

    if include_probs:
        return metrics, p

    return metrics


def run_one(
    mode,
    args,
    train_ds,
    dev_ds,
    train_rows,
    dev_rows,
    scaler,
    device,
):
    seed_all(args.seed)

    model = DayModel(mode).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=(
            torch.Generator().manual_seed(
                args.seed
            )
        ),
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    prefix = (
        f"dual_multiscale_{mode}_"
        f"seed{args.seed}_dev2021"
    )

    out = Path("outputs/metrics")

    checkpoint = (
        Path("outputs/checkpoints")
        / f"{prefix}_best.pt"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_auc = -1.0
    best_epoch = 0
    stale = 0
    history = []

    start = time.perf_counter()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[MODEL] {mode} "
        f"params={params:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        train_y = []
        train_p = []

        for image, returns, y in train_loader:
            image = image.to(device)
            returns = returns.to(device)
            target = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                image,
                returns,
            )

            loss = (
                nn.functional.cross_entropy(
                    logits,
                    target,
                )
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_y.append(
                y.numpy()
            )

            train_p.append(
                torch.softmax(
                    logits.detach(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

        train_auc = roc_auc_score(
            np.concatenate(train_y),
            np.concatenate(train_p),
        )

        dev = evaluate(
            model,
            dev_loader,
            dev_rows,
            device,
        )

        if dev["auc"] > best_auc + 1e-5:
            best_auc = dev["auc"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "mode": mode,
                    "epoch": epoch,
                    "seed": args.seed,
                    "scaler": scaler,
                    "dev_metrics": dev,
                },
                checkpoint,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_auc_training_mode": (
                float(train_auc)
            ),
            **{
                "dev_" + key: value
                for key, value
                in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            out / f"{prefix}_history.csv",
            index=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"[{mode}] "
            f"epoch={epoch:02d} "
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
                f"[EARLY STOP] {mode}",
                flush=True,
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

    best, probabilities = evaluate(
        model,
        dev_loader,
        dev_rows,
        device,
        include_probs=True,
    )

    predictions = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    predictions["prob_up"] = (
        probabilities
    )

    predictions.to_csv(
        out
        / f"{prefix}_dev_predictions.csv",
        index=False,
    )

    return {
        "mode": mode,
        "params": params,
        "best_epoch": best_epoch,

        **{
            "dev_" + key: value
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
        choices=MODES,
        default=["dual", "multi"],
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
        "--seed",
        type=int,
        default=42,
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
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--aux-dir",
        type=Path,
        default=Path(
            "data/charts_dual_train2020"
        ),
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.num_workers < 0
    ):
        parser.error(
            "Invalid numeric argument"
        )

    seed_all(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] Train 2015-2020 "
        "| Dev 2021 "
        "| No 2022-2025 data"
    )

    train_rows, dev_rows = load_dev_split(
        Path("data/processed/samples.csv")
    )

    scaler = prepare_return_images(
        train_rows,
        dev_rows,
        Path("data/raw"),
        args.aux_dir,
    )

    train_ds = DualDataset(
        train_rows,
        Path("data/charts"),
        args.aux_dir,
    )

    dev_ds = DualDataset(
        dev_rows,
        Path("data/charts"),
        args.aux_dir,
    )

    summary_path = Path(
        "outputs/metrics/"
        "dual_multiscale_dev2021_summary.csv"
    )

    results = []

    for mode in args.variants:
        result = run_one(
            mode,
            args,
            train_ds,
            dev_ds,
            train_rows,
            dev_rows,
            scaler,
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
        "[NOTE] Original single-view "
        "baseline from prior run: "
        "AUROC 0.51009 "
        "(2021 Dev, seed 42)."
    )

    print(
        "[NOTE] 2022-2025 untouched."
    )


if __name__ == "__main__":
    main()
