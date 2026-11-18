"""64x60 I20 Day-Aligned ViT: shared local-CNN tokenization vs saved linear baseline.

Run from the CV_TermProject repository root:
    python -u -m scripts.compare_exp200_tokenizer --years 2019 2020 2021 --seed 42

This script does not train the original linear model or overwrite existing outputs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.day_aligned_vit_exp200 import DayAlignedViT
from scripts.jkx_i20_reproduce import evaluate, set_seed
from scripts.run_exp200_factorial import loader, prepare_data, split, subset_scores


class LocalCNNDayAlignedViT(DayAlignedViT):
    """Same attention encoder/CLS head; per-day (1,64,3) shared CNN tokenizer."""

    def __init__(self, dim=128, depth=2, heads=4, dropout=0.1):
        super().__init__(dim=dim, depth=depth, heads=heads, dropout=dropout)
        # Replace the linear tokenizer; all Transformer layers remain unchanged.
        self.patch_embed = nn.Identity()
        self.local_tokenizer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 3)),  # preserve open/close left/right pixels
            nn.Flatten(),
            nn.Linear(32 * 4 * 3, dim),
            nn.LayerNorm(dim),
        )
        # Explicit initialization; do not change the Transformer initialization.
        for layer in self.local_tokenizer.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (1, 64, 60):
            raise ValueError(f"Expected (B,1,64,60), got {tuple(image.shape)}")
        batch = image.shape[0]
        # (B,1,64,60) -> (B,20,1,64,3) -> (B*20,1,64,3)
        days = (
            image.reshape(batch, 1, 64, 20, 3)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch * 20, 1, 64, 3)
        )
        tokens = self.local_tokenizer(days).reshape(batch, 20, -1)
        cls = self.cls.expand(batch, -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.pos)
        return self.head(self.norm(encoded[:, 0]))


def saved_linear_predictions(root, year, seed, frame, fingerprint):
    path = root / "u200_start2010" / f"test{year}" / f"seed{seed}"
    info_path = path / "summary.json"
    pred_path = path / "predictions.csv"
    if not (info_path.is_file() and pred_path.is_file()):
        raise FileNotFoundError(f"Missing saved linear ViT run: {path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cfg = info.get("config", {})
    for k, expected in {
        "data_fingerprint": fingerprint,
        "universe": 200,
        "start_year": 2010,
        "test_year": year,
        "seed": seed,
    }.items():
        if cfg.get(k) != expected:
            raise ValueError(f"Saved linear ViT config mismatch ({k}): {path}")
    pred = pd.read_csv(
        pred_path,
        dtype={"ticker": str, "group": str, "decision_date": str, "target_date": str},
    )
    keys = ["ticker", "group", "decision_date", "target_date", "label"]
    if not pred[keys].equals(frame[keys]):
        raise ValueError(f"Saved linear ViT sample alignment mismatch: {pred_path}")
    if not np.allclose(pred.return_5d, frame.return_5d, rtol=0, atol=1e-10):
        raise ValueError(f"Saved linear ViT target mismatch: {pred_path}")
    return pred


def train_one(args, device, meta, legacy, fingerprint, year):
    tag = f"u200_start2010/test{year}/seed{args.seed}"
    folder = args.out / tag
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / "summary.json"
    predictions_path = folder / "predictions.csv"
    config = {
        "architecture": "local_cnn_day_aligned_vit_i20_gray",
        "tokenizer": "local_cnn_3conv_per_day",
        "data_fingerprint": fingerprint,
        "start_year": 2010,
        "universe": 200,
        "test_year": year,
        "validation_year": year - 1,
        "seed": args.seed,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    tr, va, te, test_frame = split(meta, legacy, year, 2010, 200)
    # Validate reference BEFORE training; old linear checkpoints remain untouched.
    saved_linear_predictions(args.linear, year, args.seed, test_frame, fingerprint)
    if summary_path.exists() or predictions_path.exists():
        if not (summary_path.exists() and predictions_path.exists()):
            raise RuntimeError(f"Incomplete run in {folder}; inspect manually")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError(f"Config mismatch: {folder}. Use a new --out, not overwrite.")
        print(f"[SKIP] Already complete: {tag}", flush=True)
        return

    set_seed(args.seed)
    train_loader = loader(args.root, tr, args.batch_size, True, args.workers, args.seed)
    val_loader = loader(args.root, va, args.batch_size, False, args.workers, args.seed)
    test_loader = loader(args.root, te, args.batch_size, False, args.workers, args.seed)
    model = LocalCNNDayAlignedViT(
        dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"[RUN] {tag} train={len(tr):,} val={len(va):,} "
        f"test={len(te):,} params={param_count:,}",
        flush=True,
    )
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            n += len(labels)
        val, _ = evaluate(model, val_loader, device)
        if val["loss"] < best_loss - 1e-6:
            best_loss, best_epoch, stale = val["loss"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "config": config, "best_epoch": epoch},
                folder / "best.pt",
            )
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_loss": total / n,
            **{f"val_{k}": v for k, v in val.items()},
        })
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(
            f"[EPOCH] {tag} {epoch:02d} train={total/n:.5f} "
            f"valloss={val['loss']:.5f} valauc={val['auc']:.5f} "
            f"best={best_epoch} patience={stale}/{args.patience} "
            f"elapsed={time.perf_counter()-start:.0f}s",
            flush=True,
        )
        if stale >= args.patience:
            break

    checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, probabilities = evaluate(model, test_loader, device)
    groups = subset_scores(test_frame, probabilities, legacy)
    pred = test_frame[
        ["ticker", "group", "decision_date", "target_date", "return_5d", "label"]
    ].copy()
    pred["prob_up"] = probabilities
    pred.to_csv(predictions_path, index=False)
    summary_path.write_text(
        json.dumps({
            "config": config, "params": param_count, "best_epoch": best_epoch,
            "seconds": round(time.perf_counter() - start, 1), "groups": groups,
        }, indent=2),
        encoding="utf-8",
    )
    print(
        f"[RESULT] {tag} common100 AUC={groups['common100']['auc']:.5f} "
        f"IC={groups['common100']['rank_ic']:+.5f} "
        f"Top10={groups['common100']['top10_excess_5d']:+.3%}",
        flush=True,
    )


def make_report(args, meta, legacy, fingerprint):
    rows = []
    for year in args.years:
        tag = f"u200_start2010/test{year}/seed{args.seed}"
        folder = args.out / tag
        pred_path = folder / "predictions.csv"
        info_path = folder / "summary.json"
        if not (pred_path.exists() and info_path.exists()):
            print(f"[INCOMPLETE] {tag}", flush=True)
            continue
        _, _, _, frame = split(meta, legacy, year, 2010, 200)
        linear = saved_linear_predictions(args.linear, year, args.seed, frame, fingerprint)
        conv = pd.read_csv(
            pred_path,
            dtype={"ticker": str, "group": str, "decision_date": str, "target_date": str},
        )
        ids = ["ticker", "group", "decision_date", "target_date", "label"]
        if not conv[ids].equals(linear[ids]):
            raise ValueError(f"CNN tokenizer sample alignment mismatch: {pred_path}")
        if not np.allclose(conv.return_5d, linear.return_5d, rtol=0, atol=1e-10):
            raise ValueError(f"CNN tokenizer realized returns mismatch: {pred_path}")
        cfg = json.loads(info_path.read_text(encoding="utf-8")).get("config", {})
        if cfg.get("data_fingerprint") != fingerprint or cfg.get("seed") != args.seed:
            raise ValueError(f"CNN tokenizer config mismatch: {folder}")
        for name, predictions in (("Linear", linear), ("LocalCNN", conv)):
            metrics = subset_scores(frame, predictions.prob_up.to_numpy(float), legacy)
            for evaluation in ("common100", "all"):
                s = metrics[evaluation]
                rows.append({
                    "year": year, "seed": args.seed, "tokenizer": name,
                    "evaluation": evaluation, "auc": s["auc"],
                    "rank_ic": s["rank_ic"],
                    "top10_excess_5d": s["top10_excess_5d"],
                    "top_bottom_spread_5d": s["top10_bottom10_spread_5d"],
                    "up_ratio": s["up_ratio"], "prob_std": float(predictions.prob_up.std()),
                })
    if not rows:
        print("[REPORT] No completed local-CNN runs", flush=True)
        return
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"seed{args.seed}_comparison.csv", index=False)
    common = table[table.evaluation == "common100"].copy()
    cols = ["auc", "rank_ic", "top10_excess_5d", "top_bottom_spread_5d"]
    print("\n[COMMON100] Existing Linear vs new LocalCNN", flush=True)
    print(common[["year", "tokenizer"] + cols + ["up_ratio", "prob_std"]].to_string(
        index=False, float_format=lambda x: f"{x:+.5f}"
    ), flush=True)
    if common.year.nunique() == len(args.years):
        means = common.groupby("tokenizer")[cols].mean()
        print("\n[MEAN] Equal-weight average across test years", flush=True)
        print(means.to_string(float_format=lambda x: f"{x:+.5f}"), flush=True)
        delta = means.loc["LocalCNN"] - means.loc["Linear"]
        print("\n[MEAN DELTA] LocalCNN - Linear", flush=True)
        print(delta.to_string(float_format=lambda x: f"{x:+.5f}"), flush=True)
    print(f"[SAVED] {args.out / f'seed{args.seed}_comparison.csv'}", flush=True)
    print("[HOLDOUT] 2022-2025 untouched", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/exp200_jkx"))
    p.add_argument("--raw-root", type=Path, default=Path("data/exp200"))
    p.add_argument("--linear", type=Path, default=Path("outputs/exp200_day_vit"))
    p.add_argument("--out", type=Path, default=Path("outputs/exp200_day_vit_localcnn"))
    p.add_argument("--stage", choices=["train", "report", "smoke"], default="train")
    p.add_argument("--years", nargs="+", type=int, default=[2019, 2020, 2021])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.000253)
    p.add_argument("--weight-decay", type=float, default=0.000325)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    if not set(args.years).issubset({2019, 2020, 2021}):
        p.error("Only 2019-2021 development years are allowed")
    if (
        args.epochs < 1 or args.patience < 1 or args.batch_size < 1
        or args.lr <= 0 or args.weight_decay < 0 or args.dim % args.heads
        or args.depth < 1 or not (0 <= args.dropout < 1) or args.workers < 0
    ):
        p.error("Invalid training settings")
    if args.linear.resolve() == args.out.resolve():
        p.error("--out must differ from existing linear outputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)
    if args.stage == "smoke":
        set_seed(args.seed)
        model = LocalCNNDayAlignedViT(args.dim, args.depth, args.heads, args.dropout).to(device)
        x = torch.rand(2, 1, 64, 60, device=device)
        y = model(x)
        y.square().mean().backward()
        grad = model.local_tokenizer[0].weight.grad
        print(f"[SMOKE] logits={tuple(y.shape)} params={sum(p.numel() for p in model.parameters()):,} conv_grad={grad.norm().item():.6f}", flush=True)
        return
    meta, legacy, fingerprint = prepare_data(args.root, args.raw_root)
    if args.stage == "train":
        for year in args.years:
            train_one(args, device, meta, legacy, fingerprint, year)
    make_report(args, meta, legacy, fingerprint)


if __name__ == "__main__":
    main()
