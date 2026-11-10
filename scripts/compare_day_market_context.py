
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


IMAGE_SIZE = 160
NUM_DAYS = 20

FEATURE_NAMES = (
    "peer_return_5",
    "peer_return_10",
    "peer_return_19",
    "peer_volatility_19",
    "peer_breadth_5",
    "peer_dispersion_5",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dev_split(path):
    samples = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    needed = {
        "ticker",
        "split",
        "window_start",
        "decision_date",
        "target_date",
        "label",
    }

    missing = needed - set(samples.columns)

    if missing:
        raise ValueError(
            f"Missing sample columns: {sorted(missing)}"
        )

    base = samples.loc[
        samples["split"] == "train"
    ].copy()

    train = base.loc[
        (base["target_date"] <= "2020-12-31")
        & (base["decision_date"] <= "2020-12-31")
    ].copy()

    dev = base.loc[
        (base["window_start"] >= "2021-01-01")
        & (base["target_date"] <= "2021-12-31")
        & (base["decision_date"] <= "2021-12-31")
    ].copy()

    if train.empty or dev.empty:
        raise ValueError("Empty train or dev split")

    for name, df in (
        ("TRAIN", train),
        ("DEV", dev),
    ):
        if df.duplicated(
            ["ticker", "decision_date"]
        ).any():
            raise ValueError(
                f"Duplicated samples in {name}"
            )

        if not df["label"].isin([0, 1]).all():
            raise ValueError(
                f"Non-binary labels in {name}"
            )

        print(
            f"[{name}] N={len(df):,} "
            f"Up={df['label'].mean():.2%} "
            f"Dates={df['decision_date'].min()}"
            f"..{df['decision_date'].max()}"
        )

    return (
        train.reset_index(drop=True),
        dev.reset_index(drop=True),
    )


def build_peer_context(train, dev, raw_dir):
    tickers = sorted(
        set(train["ticker"]) | set(dev["ticker"])
    )

    close_arrays = []
    common_dates = None

    for ticker in tickers:
        path = raw_dir / f"{ticker}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if not {"Date", "Close"}.issubset(df.columns):
            raise ValueError(
                f"{ticker}: need Date, Close"
            )

        df["Date"] = (
            pd.to_datetime(df["Date"])
            .dt.strftime("%Y-%m-%d")
        )

        df = df.sort_values(
            "Date"
        ).reset_index(drop=True)

        if df["Date"].duplicated().any():
            raise ValueError(
                f"{ticker}: duplicate dates"
            )

        dates = df["Date"].to_numpy()

        if common_dates is None:
            common_dates = dates

        elif not np.array_equal(
            common_dates, dates
        ):
            raise ValueError(
                "Trading dates differ between tickers. "
                "Calendar alignment is required."
            )

        close = df["Close"].to_numpy(
            dtype=np.float64
        )

        if (
            not np.isfinite(close).all()
            or (close <= 0).any()
        ):
            raise ValueError(
                f"{ticker}: invalid Close"
            )

        close_arrays.append(close)

    if len(tickers) < 3:
        raise ValueError(
            "At least three stocks required"
        )

    log_close = np.log(
        np.stack(close_arrays, axis=1)
    )

    date_to_idx = {
        date: idx
        for idx, date in enumerate(common_dates)
    }

    ticker_to_idx = {
        ticker: idx
        for idx, ticker in enumerate(tickers)
    }

    def extract(rows):
        features = np.empty(
            (len(rows), len(FEATURE_NAMES)),
            dtype=np.float32,
        )

        for k, row in enumerate(
            rows.itertuples(index=False)
        ):
            idx = date_to_idx.get(
                row.decision_date
            )

            if (
                idx is None
                or idx < NUM_DAYS - 1
            ):
                raise ValueError(
                    f"Insufficient history: "
                    f"{row.ticker} {row.decision_date}"
                )

            if (
                common_dates[idx - NUM_DAYS + 1]
                != row.window_start
            ):
                raise ValueError(
                    f"Window mismatch: "
                    f"{row.ticker} {row.decision_date}"
                )

            # Exclude the stock being predicted
            peer_mask = np.ones(
                len(tickers),
                dtype=bool,
            )

            peer_mask[
                ticker_to_idx[row.ticker]
            ] = False

            # Exactly 20 historical closing prices
            peer_logs = log_close[
                idx - NUM_DAYS + 1:idx + 1,
                peer_mask,
            ]

            # 19 daily return observations
            peer_returns = np.diff(
                peer_logs,
                axis=0,
            )

            daily_market_return = (
                peer_returns.mean(axis=1)
            )

            features[k] = [
                daily_market_return[-5:].sum(),
                daily_market_return[-10:].sum(),
                daily_market_return.sum(),
                daily_market_return.std(),
                (peer_returns[-5:] > 0).mean(),
                peer_returns[-5:]
                .std(axis=1).mean(),
            ]

        return features

    train_features = extract(train)
    dev_features = extract(dev)

    # Standardization: training statistics only
    mean = train_features.mean(
        axis=0,
        dtype=np.float64,
    )

    std = train_features.std(
        axis=0,
        dtype=np.float64,
    )

    std = np.maximum(std, 1e-8)

    train_features = np.clip(
        (train_features - mean) / std,
        -5,
        5,
    ).astype(np.float32)

    dev_features = np.clip(
        (dev_features - mean) / std,
        -5,
        5,
    ).astype(np.float32)

    scaler = {
        "features": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    print(
        f"[CONTEXT] {len(tickers)} stocks "
        f"| Peers per sample={len(tickers) - 1} "
        "| Train-only standardization"
    )

    return train_features, dev_features, scaler


class ChartMarketDataset(Dataset):
    def __init__(self, rows, context, chart_dir):
        self.rows = rows.reset_index(drop=True)

        self.context = torch.from_numpy(
            np.ascontiguousarray(
                context,
                dtype=np.float32,
            )
        )

        self.labels = torch.tensor(
            self.rows["label"].to_numpy(
                dtype=np.int64,
                copy=True,
            )
        )

        if len(self.context) != len(self.rows):
            raise ValueError(
                "Context count does not match samples"
            )

        # Original 2015-2021 images are all in train/
        self.paths = [
            chart_dir
            / "train"
            / row.ticker
            / f"{row.decision_date}.png"
            for row in self.rows.itertuples(
                index=False
            )
        ]

        missing = next(
            (str(p) for p in self.paths if not p.exists()),
            None,
        )

        if missing:
            raise FileNotFoundError(
                f"Missing image: {missing}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            if img.size != (
                IMAGE_SIZE, IMAGE_SIZE
            ):
                raise ValueError(
                    f"Invalid image: {self.paths[idx]}"
                )

            image = to_tensor(
                img.convert("RGB")
            )

        return (
            image,
            self.context[idx],
            self.labels[idx],
        )


class DayViTWithMarket(nn.Module):
    def __init__(self, use_market):
        super().__init__()

        self.use_market = bool(use_market)

        embed_dim = 128

        # One patch = exactly one trading day
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=(160, 8),
            stride=(160, 8),
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, NUM_DAYS, embed_dim
            )
        )

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Zero context maps exactly to zero features
        self.context_encoder = nn.Sequential(
            nn.Linear(
                len(FEATURE_NAMES),
                32,
                bias=False,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                32,
                bias=False,
            ),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(
                embed_dim + 32,
                64,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

    def forward(self, images, context):
        expected = (3, 160, 160)

        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != expected
        ):
            raise ValueError(
                f"Unexpected image shape: "
                f"{tuple(images.shape)}"
            )

        # White background -> 0
        tokens = self.patch_embed(
            images - 1.0
        )

        tokens = (
            tokens.flatten(2).transpose(1, 2)
        )

        tokens = self.dropout(
            tokens + self.pos_embed
        )

        for block in self.blocks:
            tokens = block(tokens)

        image_features = self.norm(
            tokens.mean(dim=1)
        )

        if not self.use_market:
            context = torch.zeros_like(
                context
            )

        peer_features = (
            self.context_encoder(context)
        )

        fused = torch.cat(
            [image_features, peer_features],
            dim=1,
        )

        return self.head(fused)


def get_metrics(labels, scores):
    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    p = np.asarray(
        scores,
        dtype=np.float64,
    )

    pred = (p >= 0.5).astype(
        np.int64
    )

    return {
        "auc": float(
            roc_auc_score(y, p)
        ),
        "acc": float(
            accuracy_score(y, pred)
        ),
        "bal": float(
            balanced_accuracy_score(y, pred)
        ),
        "f1": float(
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
    }


def validate(
    model,
    loader,
    device,
    metadata,
    return_scores=False,
):
    model.eval()

    labels = []
    probs = []
    loss_sum = 0.0

    with torch.inference_mode():
        for images, context, targets in loader:
            logits = model(
                images.to(
                    device,
                    non_blocking=True,
                ),
                context.to(
                    device,
                    non_blocking=True,
                ),
            )

            loss_sum += (
                nn.functional.cross_entropy(
                    logits,
                    targets.to(device),
                    reduction="sum",
                ).item()
            )

            probs.append(
                torch.softmax(
                    logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            labels.append(
                targets.numpy()
            )

    y = np.concatenate(labels)
    p = np.concatenate(probs)

    if not np.array_equal(
        y,
        metadata["label"].to_numpy()
    ):
        raise RuntimeError(
            "Evaluation order does not match metadata"
        )

    result = get_metrics(y, p)

    result["loss"] = (
        loss_sum / len(y)
    )

    # Daily cross-sectional Spearman IC
    if "return_5d" in metadata.columns:
        table = metadata[
            ["decision_date", "return_5d"]
        ].copy()

        table["score"] = p

        daily_ic = []

        for _, part in table.groupby(
            "decision_date",
            sort=False,
        ):
            if (
                part["score"].nunique() > 1
                and part["return_5d"].nunique() > 1
            ):
                ic = part["score"].corr(
                    part["return_5d"],
                    method="spearman",
                )

                daily_ic.append(ic)

        result["rank_ic"] = (
            float(np.nanmean(daily_ic))
            if daily_ic
            else float("nan")
        )

    if return_scores:
        return result, p

    return result


def train_variant(
    variant,
    args,
    train_set,
    dev_set,
    train_rows,
    dev_rows,
    scaler,
    device,
):
    set_seed(args.seed)

    use_market = variant == "market"

    model = DayViTWithMarket(
        use_market
    ).to(device)

    train_loader = DataLoader(
        train_set,
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
        dev_set,
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

    best_auc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    checkpoint_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir = Path(
        "outputs/metrics"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"day_market_{variant}_"
        f"seed{args.seed}_dev2021"
    )

    best_path = (
        checkpoint_dir
        / f"{prefix}_best.pt"
    )

    history = []
    start = time.perf_counter()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[VARIANT] {variant} "
        f"| params={params:,} "
        f"| market={use_market}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        total_loss = 0.0
        count = 0
        all_y = []
        all_p = []

        for images, context, targets in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            context = context.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
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
                    targets,
                )
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            count += len(targets)

            total_loss += (
                loss.item() * len(targets)
            )

            all_y.append(
                targets.detach()
                .cpu().numpy()
            )

            all_p.append(
                torch.softmax(
                    logits.detach(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

        train_auc = float(
            roc_auc_score(
                np.concatenate(all_y),
                np.concatenate(all_p),
            )
        )

        dev_metrics = validate(
            model,
            dev_loader,
            device,
            dev_rows,
        )

        improved = (
            dev_metrics["auc"] > best_auc + 1e-5
            or (
                abs(
                    dev_metrics["auc"] - best_auc
                ) <= 1e-5
                and dev_metrics["loss"] < best_loss
            )
        )

        if improved:
            best_auc = dev_metrics["auc"]
            best_loss = dev_metrics["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_name": "day_market_context",
                    "model_state": model.state_dict(),
                    "model_config": {
                        "use_market": use_market,
                    },
                    "epoch": epoch,
                    "seed": args.seed,
                    "context_scaler": scaler,
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                    "dev_metrics": dev_metrics,
                },
                best_path,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / count,
            "train_auc_training_mode": train_auc,
            **{
                f"dev_{k}": v
                for k, v in dev_metrics.items()
            },
        })

        pd.DataFrame(history).to_csv(
            metrics_dir
            / f"{prefix}_history.csv",
            index=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"[{variant}] "
            f"ep={epoch:02d}/{args.epochs} "
            f"train_auc={train_auc:.4f} "
            f"dev_auc={dev_metrics['auc']:.4f} "
            f"dev_bal={dev_metrics['bal']:.4f} "
            f"dev_f1={dev_metrics['f1']:.4f} "
            f"dev_ic={dev_metrics['rank_ic']:+.4f} "
            f"up={dev_metrics['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {variant}"
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

    best_metrics, probabilities = validate(
        model,
        dev_loader,
        device,
        dev_rows,
        return_scores=True,
    )

    prediction = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    prediction["prob_up"] = (
        probabilities
    )

    prediction.to_csv(
        metrics_dir
        / f"{prefix}_dev_predictions.csv",
        index=False,
    )

    return {
        "variant": variant,
        "best_epoch": best_epoch,
        "params": params,
        "best_dev_auc": best_metrics["auc"],
        "best_dev_bal": best_metrics["bal"],
        "best_dev_f1": best_metrics["f1"],
        "best_dev_ic": best_metrics["rank_ic"],
        "best_dev_up_ratio": (
            best_metrics["up_ratio"]
        ),
        "best_dev_prob_std": (
            best_metrics["prob_std"]
        ),
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
        choices=["baseline", "market"],
        default=["baseline", "market"],
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
            f"[GPU] {torch.cuda.get_device_name(0)}"
        )

    print(
        "[PROTOCOL] Train=2015-2020 "
        "| Dev=2021 | 2022-2025 untouched"
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

    train_set = ChartMarketDataset(
        train_rows,
        train_context,
        Path("data/charts"),
    )

    dev_set = ChartMarketDataset(
        dev_rows,
        dev_context,
        Path("data/charts"),
    )

    results = []

    for variant in args.variants:
        result = train_variant(
            variant,
            args,
            train_set,
            dev_set,
            train_rows,
            dev_rows,
            scaler,
            device,
        )

        results.append(result)

    output_path = Path(
        "outputs/metrics/"
        "day_market_context_comparison.csv"
    )

    pd.DataFrame(results).to_csv(
        output_path,
        index=False,
    )

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(results).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"[SAVED] {output_path}")
    print(
        "[NOTE] No 2022-2025 evaluation"
    )


if __name__ == "__main__":
    main()
