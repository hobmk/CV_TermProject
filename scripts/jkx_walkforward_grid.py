"""Walk-forward, 3-variant, multi-seed I20/R5 study.

Uses the data and functions already created by scripts.jkx_i20_reproduce.
This is an extension/ablation study, NOT an exact reproduction of JKX.
The original data/exp100_jkx, 2022-2025 holdout are read-only/untouched.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
from torch import nn
from torch.utils.data import DataLoader

from scripts.jkx_i20_reproduce import (
    Jkx20, MemmapCharts, cross_section_metrics, evaluate, set_seed
)

VARIANTS = ("jkx", "day_pool", "recency_attn")


class DayAlignedCNN(nn.Module):
    """JKX convolutional feature extractor + day-aligned head.

    Feature width 60 = 20 trading days * 3 columns/day.
    Variant 'day_pool' is the necessary control for attention.
    """
    def __init__(self, attention=False):
        super().__init__()
        self.features = Jkx20().features
        self.attention_enabled = attention
        if attention:
            self.attention = nn.Sequential(
                nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1)
            )
            self.position_bias = nn.Parameter(torch.zeros(20))
            self.register_buffer("recency_prior", torch.linspace(-1, 1, 20))
            for layer in self.attention.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 2))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        features = self.features(x)
        if features.shape[1:] != (256, 3, 60):
            raise RuntimeError(f"Unexpected feature shape: {features.shape}")
        b = x.shape[0]
        days = features.reshape(b, 256, 3, 20, 3).mean(dim=(2, 4))
        days = days.transpose(1, 2)  # B x 20 days x 256 channels
        if self.attention_enabled:
            logits = self.attention(days).squeeze(-1)
            logits = logits + self.position_bias + 0.5 * self.recency_prior
            weights = torch.softmax(logits, dim=1)
            pooled = (days * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = days.mean(dim=1)
        return self.head(pooled)


def model_for(variant):
    if variant == "jkx":
        return Jkx20()
    if variant == "day_pool":
        return DayAlignedCNN(attention=False)
    if variant == "recency_attn":
        return DayAlignedCNN(attention=True)
    raise ValueError(variant)


def read_metadata(root):
    dtype = {c: str for c in ("ticker", "decision_date", "target_date", "window_start")}
    train = pd.read_csv(root / "train_meta.csv", dtype=dtype)
    dev = pd.read_csv(root / "dev_meta.csv", dtype=dtype)
    for name, frame, image_file in (
        ("train", train, "train_images.npy"),
        ("dev", dev, "dev_images.npy"),
    ):
        required = {"ticker", "decision_date", "target_date", "window_start", "return_5d", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}: missing columns: {required - set(frame.columns)}")
        if frame.duplicated(["ticker", "decision_date"]).any():
            raise ValueError(f"{name}: duplicate stock-date rows")
        data = np.load(root / image_file, mmap_mode="r")
        if len(data) != len(frame) or tuple(data.shape[1:]) != (1, 64, 60):
            raise ValueError(f"{name}: image metadata mismatch: {data.shape}")
        if not np.isfinite(frame.return_5d.to_numpy(float)).all():
            raise ValueError(f"{name}: nonfinite return")
        if not np.array_equal(frame.label.to_numpy(int), (frame.return_5d.to_numpy(float) > 0).astype(int)):
            raise ValueError(f"{name}: labels inconsistent with return_5d")
        del data
    if train.target_date.max() >= dev.window_start.min():
        # train_meta contains late-2020 examples whose 5-day target may
        # nevertheless be before 2021 starts. They must not overlap dev.
        raise ValueError("Train metadata horizons overlap 2021 Dev images")
    return train, dev


def split_indices(train, dev, test_year):
    """Example: test 2021, train<=2019 with known labels, val=2020."""
    val_year = test_year - 1
    val_start = f"{val_year}-01-01"
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year + 1}-01-01"

    tr_idx = np.flatnonzero((train.target_date < val_start).to_numpy())
    va_idx = np.flatnonzero((
        (train.window_start >= val_start) & (train.target_date < test_start)
    ).to_numpy())
    test_source = dev if test_year == 2021 else train
    te_idx = np.flatnonzero((
        (test_source.window_start >= test_start) &
        (test_source.target_date < test_end)
    ).to_numpy())
    if min(len(tr_idx), len(va_idx), len(te_idx)) < 1000:
        raise ValueError(f"Insufficient data for test year {test_year}")
    if train.iloc[tr_idx].target_date.max() >= train.iloc[va_idx].window_start.min():
        raise ValueError("Train target overlaps Val input")
    if train.iloc[va_idx].target_date.max() >= test_source.iloc[te_idx].window_start.min():
        raise ValueError("Val target overlaps Test input")
    test_frame = test_source.iloc[te_idx].reset_index(drop=True)
    group_sizes = test_frame.groupby("decision_date").ticker.nunique()
    if group_sizes.min() != 100 or group_sizes.max() != 100:
        raise ValueError(f"test {test_year} date groups aren't all 100 stocks")
    return tr_idx, va_idx, te_idx, test_source, test_frame


def make_loader(root, split, indices, batch_size, shuffle, workers, seed):
    dataset = MemmapCharts(
        root / f"{split}_images.npy",
        root / f"{split}_meta.csv",
        indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=(workers > 0),
    )


def train_run(args, device, metadata, year, variant, seed):
    train_meta, dev_meta = metadata
    tr_idx, va_idx, te_idx, test_source, test_frame = split_indices(train_meta, dev_meta, year)
    run_dir = args.out / f"test{year}" / variant / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "predictions.csv"
    config = {
        "version": 1, "test_year": year, "variant": variant, "seed": seed,
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "batch_size": args.batch_size, "source": str(args.root.resolve()),
    }
    if summary_path.exists() and predictions_path.exists() and not args.force:
        with summary_path.open(encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config") != config:
            raise ValueError(f"Existing run uses different settings: {run_dir}. Choose a new --out or --force")
        print(f"[SKIP COMPLETED] {year} {variant} seed{seed}", flush=True)
        return old["metrics"]

    set_seed(seed)
    train_loader = make_loader(args.root, "train", tr_idx, args.batch_size, True, args.workers, seed)
    val_loader = make_loader(args.root, "train", va_idx, args.batch_size, False, args.workers, seed)
    test_loader = make_loader(args.root, "dev" if year == 2021 else "train", te_idx, args.batch_size, False, args.workers, seed)
    model = model_for(variant).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    best_epoch = 0
    patience_count = 0
    records = []
    started = time.perf_counter()
    print(
        f"\n[RUN] test={year} val={year-1} model={variant} seed={seed} "
        f"N={len(tr_idx):,}/{len(va_idx):,}/{len(te_idx):,} params={params:,}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(label)
            total_count += len(label)
        val, _ = evaluate(model, val_loader, device)
        improved = val["loss"] < best - 1e-6
        if improved:
            best, best_epoch, patience_count = val["loss"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "config": config, "val_metrics": val},
                run_dir / "best.pt",
            )
        else:
            patience_count += 1
        record = {"epoch": epoch, "train_loss": total_loss / total_count, **{f"val_{k}": v for k, v in val.items()}}
        records.append(record)
        pd.DataFrame.from_records(records).to_csv(run_dir / "history.csv", index=False)
        print(
            f"[EPOCH] {year} {variant} seed{seed} ep={epoch:02d} "
            f"train_loss={record['train_loss']:.4f} val_loss={val['loss']:.4f} "
            f"val_auc={val['auc']:.4f} best={best_epoch} "
            f"stale={patience_count}/{args.patience} elapsed={time.perf_counter()-started:.0f}s",
            flush=True,
        )
        if patience_count >= args.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test, probability = evaluate(model, test_loader, device)
    cross = cross_section_metrics(test_frame, probability)
    output_frame = test_frame[["ticker", "decision_date", "target_date", "return_5d", "label"]].copy()
    output_frame["prob_up"] = probability
    output_frame.to_csv(predictions_path, index=False)
    metrics = {
        "test_year": year, "val_year": year - 1, "variant": variant, "seed": seed,
        "train_n": len(tr_idx), "val_n": len(va_idx), "test_n": len(te_idx),
        "params": params, "best_epoch": best_epoch,
        **{f"test_{k}": float(v) for k, v in {**test, **cross}.items()},
        "seconds": round(time.perf_counter() - started, 1),
    }
    summary_path.write_text(json.dumps({"config": config, "metrics": metrics}, indent=2), encoding="utf-8")
    print(
        f"[RESULT] year={year} {variant} seed{seed} "
        f"AUC={metrics['test_auc']:.4f} IC={metrics['test_rank_ic']:+.4f} "
        f"top10={metrics['test_top10_excess_5d']:+.3%} "
        f"spread={metrics['test_top10_bottom10_spread_5d']:+.3%}",
        flush=True,
    )
    return metrics


def summarize(args):
    rows = []
    for path in sorted(args.out.glob("test*/*/seed*/summary.json")):
        with path.open(encoding="utf-8") as f:
            rows.append(json.load(f)["metrics"])
    if not rows:
        return
    runs = pd.DataFrame.from_records(rows)
    runs.sort_values(["test_year", "variant", "seed"]).to_csv(args.out / "all_runs.csv", index=False)

    ensemble_rows = []
    for (year, variant), group in runs.groupby(["test_year", "variant"]):
        frames = []
        for seed in sorted(group.seed):
            file = args.out / f"test{year}" / variant / f"seed{seed}" / "predictions.csv"
            frames.append(pd.read_csv(file, dtype={"ticker": str, "decision_date": str, "target_date": str}))
        base = frames[0]
        for frame in frames[1:]:
            if not base[["ticker", "decision_date", "target_date", "label"]].equals(frame[["ticker", "decision_date", "target_date", "label"]]):
                raise ValueError(f"Seed prediction alignment mismatch for {year} {variant}")
            if not np.allclose(base.return_5d, frame.return_5d, rtol=0, atol=1e-12):
                raise ValueError(f"Seed return mismatch for {year} {variant}")
        ensemble_prob = np.mean(np.stack([f.prob_up.to_numpy(float) for f in frames]), axis=0)
        ens = base.copy()
        ens["prob_up"] = ensemble_prob
        ens.to_csv(args.out / f"ensemble_{year}_{variant}.csv", index=False)
        y = ens.label.to_numpy(int)
        p = ens.prob_up.to_numpy(float)
        pred = p >= 0.5
        metrics = {
            "test_year": year, "variant": variant, "seed_count": len(frames),
            "test_auc": float(roc_auc_score(y, p)),
            "test_accuracy": float(accuracy_score(y, pred)),
            "test_balanced_acc": float(balanced_accuracy_score(y, pred)),
            "test_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "test_up_ratio": float(np.mean(pred)),
            **{f"test_{k}": float(v) for k, v in cross_section_metrics(ens, p).items()},
        }
        ensemble_rows.append(metrics)
    ensemble = pd.DataFrame.from_records(ensemble_rows)
    ensemble.sort_values(["test_year", "variant"]).to_csv(args.out / "ensemble_summary.csv", index=False)
    print("\n[ENSEMBLE RESULTS] (2022-2025 final test untouched)", flush=True)
    cols = ["test_year", "variant", "seed_count", "test_auc", "test_rank_ic", "test_top10_excess_5d", "test_top10_bottom10_spread_5d"]
    print(ensemble[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print(f"[SAVED] {args.out / 'all_runs.csv'} | {args.out / 'ensemble_summary.csv'}", flush=True)
    print("[LIMITATION] 5-day return summaries overlap; not a transaction-cost-adjusted backtest.", flush=True)


def main():
    p = argparse.ArgumentParser(description="JKX CNN vs Day Pool vs Recency Attention, chronological walk forward")
    p.add_argument("--root", type=Path, default=Path("data/exp100_jkx"))
    p.add_argument("--out", type=Path, default=Path("outputs/jkx_walkforward_grid"))
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=3.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if (args.epochs < 1 or args.patience < 1 or args.lr <= 0 or
            args.batch_size < 1 or args.workers < 0 or args.max_hours <= 0):
        p.error("Invalid numeric settings")
    if not set(args.years).issubset({2019, 2020, 2021}):
        p.error("Only test years 2019/2020/2021 are supported by this dataset")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} | budget={args.max_hours:g}h", flush=True)
    metadata = read_metadata(args.root)
    planned = [(year, variant, seed) for year in args.years for variant in args.variants for seed in args.seeds]
    print(f"[PLAN] {len(planned)} independent trainings, chronological validation, final 2022-2025 untouched", flush=True)
    started = time.perf_counter()
    try:
        for i, (year, variant, seed) in enumerate(planned, start=1):
            if time.perf_counter() - started >= args.max_hours * 3600:
                print(f"[TIME BUDGET] stop before run {i}/{len(planned)}. Re-run same command to continue.", flush=True)
                break
            print(f"[PROGRESS] {i}/{len(planned)}", flush=True)
            train_run(args, device, metadata, year, variant, seed)
    finally:
        summarize(args)


if __name__ == "__main__":
    main()
