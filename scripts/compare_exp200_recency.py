
"""Recency-weighted Expanding Window experiment.

Keep the original 64x60 Linear Day-Aligned ViT.
Only reweight training Cross-Entropy by sample age.
Use existing 2019-2021 walk-forward splits and baseline predictions.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.day_aligned_vit_exp200 import DayAlignedViT
from scripts.jkx_i20_reproduce import MemmapCharts, evaluate, set_seed
from scripts.run_exp200_factorial import (
    loader,
    prepare_data,
    split,
    subset_scores,
)


class WeightedCharts(Dataset):
    def __init__(self, root, indices, weights):
        self.base = MemmapCharts(
            root / "images.npy",
            root / "meta.csv",
            indices,
        )
        self.weights = np.asarray(weights, dtype=np.float32)

        if len(self.base) != len(self.weights):
            raise ValueError("Indices/weights length mismatch")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, float(self.weights[index])


def make_weights(train_dates, half_life):
    dates = pd.to_datetime(pd.Series(train_dates))
    reference = dates.max()

    ages = (
        (reference - dates).dt.days.to_numpy(dtype=float)
        / 365.25
    )

    if (ages < 0).any():
        raise ValueError("Negative sample age")

    raw = np.exp2(-ages / half_life)

    # Preserve approximately the original loss scale.
    weights = (raw / raw.mean()).astype(np.float32)

    # Weight-only approximation; ignores sample correlations.
    w64 = weights.astype(np.float64)
    effective_n = float(
        w64.sum() ** 2 / np.square(w64).sum()
    )

    return weights, str(reference.date()), effective_n


def baseline_predictions(a, frame, fingerprint, year):
    folder = (
        a.baseline
        / "u200_start2010"
        / f"test{year}"
        / f"seed{a.seed}"
    )

    summary_path = folder / "summary.json"
    pred_path = folder / "predictions.csv"

    if not summary_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError(
            f"Original Linear ViT result missing: {folder}"
        )

    info = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    cfg = info["config"]

    expected = {
        "data_fingerprint": fingerprint,
        "universe": 200,
        "start_year": 2010,
        "test_year": year,
        "seed": a.seed,
        "depth": 2,
        "dim": 128,
        "heads": 4,
        "dropout": 0.1,
        "lr": a.lr,
        "weight_decay": a.weight_decay,
        "batch_size": a.batch_size,
        "patience": a.patience,
        "epochs": a.epochs,
    }

    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Baseline config mismatch: {key}. "
                f"Saved={cfg.get(key)}, expected={value}"
            )

    pred = pd.read_csv(
        pred_path,
        dtype={
            "ticker": str,
            "group": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    ids = [
        "ticker", "group", "decision_date",
        "target_date", "label",
    ]

    if not pred[ids].equals(frame[ids]):
        raise ValueError("Baseline sample alignment mismatch")

    if not np.allclose(
        pred.return_5d,
        frame.return_5d,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Baseline return mismatch")

    return pred


def train_one(a, device, meta, legacy, fingerprint, year):
    tr, va, te, test_frame = split(
        meta, legacy, year, 2010, 200
    )

    # Verify the saved baseline before starting training.
    baseline_predictions(
        a, test_frame, fingerprint, year
    )

    folder = (
        a.out
        / "u200_start2010"
        / f"test{year}"
        / f"seed{a.seed}"
    )

    summary_path = folder / "summary.json"
    pred_path = folder / "predictions.csv"

    config = {
        "data_fingerprint": fingerprint,
        "model": "day_aligned_vit_i20_grayscale",
        "universe": 200,
        "start_year": 2010,
        "test_year": year,
        "validation_year": year - 1,
        "seed": a.seed,
        "depth": 2,
        "dim": 128,
        "heads": 4,
        "dropout": 0.1,
        "lr": a.lr,
        "weight_decay": a.weight_decay,
        "epochs": a.epochs,
        "patience": a.patience,
        "batch_size": a.batch_size,
        "train_objective": "recency_weighted_CE",
        "half_life_years": a.half_life,
        "normalization": "global_train_mean_one",
    }

    if summary_path.exists() or pred_path.exists():
        if not (
            summary_path.exists() and pred_path.exists()
        ):
            raise RuntimeError(
                f"Incomplete saved experiment: {folder}"
            )

        old = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        if old["config"] != config:
            raise ValueError(
                f"Saved configuration mismatch: {folder}"
            )

        print(f"[SKIP] {year} seed{a.seed}", flush=True)
        return

    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(
            f"Partial experiment exists: {folder}"
        )

    folder.mkdir(parents=True, exist_ok=True)

    train_dates = (
        meta.iloc[tr].decision_date.to_numpy()
    )

    weights, reference, effective_n = make_weights(
        train_dates, a.half_life
    )

    set_seed(a.seed)

    train_loader = DataLoader(
        WeightedCharts(a.root, tr, weights),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
        generator=torch.Generator().manual_seed(a.seed),
    )

    val_loader = loader(
        a.root, va, a.batch_size,
        False, a.workers, a.seed
    )

    test_loader = loader(
        a.root, te, a.batch_size,
        False, a.workers, a.seed
    )

    model = DayAlignedViT(
        dim=128,
        depth=2,
        heads=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()

    print(
        f"[RUN] test{year} seed{a.seed} "
        f"train={len(tr):,} "
        f"val={len(va):,} "
        f"test={len(te):,} "
        f"half_life={a.half_life:g}",
        flush=True,
    )

    print(
        f"[WEIGHTS] latest_train_date={reference} "
        f"min={weights.min():.4f} "
        f"max={weights.max():.4f} "
        f"mean={weights.mean():.4f} "
        f"effective_n={effective_n:,.0f}",
        flush=True,
    )

    for epoch in range(1, a.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for images, labels, batch_weights in train_loader:
            x = images.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            w = batch_weights.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            # The only changed part of the training objective.
            per_sample_ce = F.cross_entropy(
                logits, y, reduction="none"
            )

            loss = (per_sample_ce * w).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(y)
            n += len(y)

        # Validation remains completely unweighted.
        val, _ = evaluate(
            model, val_loader, device
        )

        if not np.isfinite(val["loss"]):
            raise RuntimeError("Nonfinite validation loss")

        if val["loss"] < best_loss - 1e-6:
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "val_metrics": val,
                },
                folder / "best.pt",
            )
        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_weighted_loss": total_loss / n,
            **{
                f"val_{key}": value
                for key, value in val.items()
            },
        })

        pd.DataFrame(history).to_csv(
            folder / "history.csv",
            index=False,
        )

        elapsed = time.perf_counter() - started

        print(
            f"[EPOCH] {year} seed{a.seed} "
            f"ep={epoch:02d} "
            f"train={total_loss/n:.5f} "
            f"val={val['loss']:.5f} "
            f"val_auc={val['auc']:.5f} "
            f"best={best_epoch} "
            f"patience={stale}/{a.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= a.patience:
            break

    checkpoint = torch.load(
        folder / "best.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    _, probs = evaluate(
        model, test_loader, device
    )

    prediction = test_frame[
        [
            "ticker", "group", "decision_date",
            "target_date", "return_5d", "label",
        ]
    ].copy()

    prediction["prob_up"] = probs
    prediction.to_csv(pred_path, index=False)

    groups = subset_scores(
        test_frame, probs, legacy
    )

    summary_path.write_text(
        json.dumps(
            {
                "config": config,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "effective_sample_n_weight_only": effective_n,
                "elapsed_seconds": round(
                    time.perf_counter() - started, 1
                ),
                "groups": groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    common = groups["common100"]

    print(
        f"[RESULT] {year} "
        f"AUC={common['auc']:.5f} "
        f"RankIC={common['rank_ic']:+.5f} "
        f"Top10={common['top10_excess_5d']:+.3%}",
        flush=True,
    )


def report(a, meta, legacy, fingerprint):
    rows = []

    for year in a.years:
        _, _, _, test_frame = split(
            meta, legacy, year, 2010, 200
        )

        baseline = baseline_predictions(
            a, test_frame, fingerprint, year
        )

        folder = (
            a.out
            / "u200_start2010"
            / f"test{year}"
            / f"seed{a.seed}"
        )

        summary_path = folder / "summary.json"
        pred_path = folder / "predictions.csv"

        if not (
            summary_path.exists() and pred_path.exists()
        ):
            print(f"[INCOMPLETE] {year}", flush=True)
            continue

        info = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        cfg = info["config"]

        if (
            cfg["data_fingerprint"] != fingerprint
            or cfg["half_life_years"] != a.half_life
            or cfg["seed"] != a.seed
            or cfg["test_year"] != year
        ):
            raise ValueError(
                f"Weighted output config mismatch: {folder}"
            )

        weighted = pd.read_csv(
            pred_path,
            dtype={
                "ticker": str,
                "group": str,
                "decision_date": str,
                "target_date": str,
            },
        )

        ids = [
            "ticker", "group", "decision_date",
            "target_date", "label",
        ]

        if not weighted[ids].equals(baseline[ids]):
            raise ValueError(
                "Baseline and weighted samples differ"
            )

        if not np.allclose(
            weighted.return_5d,
            baseline.return_5d,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                "Baseline and weighted returns differ"
            )

        for method, pred in (
            ("Expanding", baseline),
            ("RecencyWeighted", weighted),
        ):
            scores = subset_scores(
                test_frame,
                pred.prob_up.to_numpy(float),
                legacy,
            )

            common = scores["common100"]

            rows.append({
                "year": year,
                "method": method,
                "auc": common["auc"],
                "rank_ic": common["rank_ic"],
                "top10_excess_5d":
                    common["top10_excess_5d"],
                "top_bottom_spread_5d":
                    common["top10_bottom10_spread_5d"],
                "up_ratio": common["up_ratio"],
            })

    if not rows:
        print("[REPORT] No completed experiments")
        return

    table = pd.DataFrame(rows)

    a.out.mkdir(parents=True, exist_ok=True)

    table.to_csv(
        a.out / f"seed{a.seed}_comparison.csv",
        index=False,
    )

    print(
        "\n[COMMON100] Expanding vs Recency-Weighted",
        flush=True,
    )

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    columns = [
        "auc",
        "rank_ic",
        "top10_excess_5d",
        "top_bottom_spread_5d",
    ]

    means = table.groupby("method")[columns].mean()

    print(
        "\n[MEAN] Equal-weight average across years",
        flush=True,
    )

    print(
        means.to_string(
            float_format=lambda x: f"{x:+.5f}"
        ),
        flush=True,
    )

    if {
        "Expanding",
        "RecencyWeighted",
    }.issubset(means.index):

        delta = (
            means.loc["RecencyWeighted"]
            - means.loc["Expanding"]
        )

        print(
            "\n[MEAN DELTA] Weighted - Original",
            flush=True,
        )

        print(
            delta.to_string(
                float_format=lambda x: f"{x:+.5f}"
            ),
            flush=True,
        )

    print(
        f"[SAVED] {a.out / f'seed{a.seed}_comparison.csv'}",
        flush=True,
    )

    print("[HOLDOUT] 2022-2025 untouched", flush=True)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root", type=Path,
        default=Path("data/exp200_jkx"),
    )
    p.add_argument(
        "--raw-root", type=Path,
        default=Path("data/exp200"),
    )
    p.add_argument(
        "--baseline", type=Path,
        default=Path("outputs/exp200_day_vit"),
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("outputs/exp200_recency_h4"),
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=[2019, 2020, 2021],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--half-life", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.000253)
    p.add_argument(
        "--weight-decay", type=float,
        default=0.000325,
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--stage",
        choices=["train", "report"],
        default="train",
    )

    a = p.parse_args()

    if (
        any(y not in (2019, 2020, 2021) for y in a.years)
        or a.half_life <= 0
        or a.epochs < 1
        or a.patience < 1
        or a.batch_size < 1
        or a.workers < 0
        or a.lr <= 0
    ):
        p.error("Invalid experiment arguments")

    meta, legacy, fingerprint = prepare_data(
        a.root, a.raw_root
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"[DEVICE] {device} "
        f"| [PLAN] {len(a.years)} weighted runs "
        f"| half_life={a.half_life:g} years",
        flush=True,
    )

    if a.stage == "train":
        for year in a.years:
            train_one(
                a, device, meta, legacy,
                fingerprint, year
            )

    report(a, meta, legacy, fingerprint)


if __name__ == "__main__":
    main()
