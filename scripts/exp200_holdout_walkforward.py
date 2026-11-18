
"""Frozen 2022-2025 walk-forward: JKX CNN vs Day-Aligned ViT, 10 seeds.

Run from repository root:
  python -u -m scripts.exp200_holdout_walkforward --stage check
  python -u -m scripts.exp200_holdout_walkforward --stage train
  python -u -m scripts.exp200_holdout_walkforward --stage report

2010-2021 images/outputs remain unchanged.
Existing model architectures are reused.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from scripts.jkx_i20_reproduce import (
    Jkx20,
    evaluate,
    set_seed,
)

from scripts.day_aligned_vit_exp200 import DayAlignedViT

from scripts.run_exp200_factorial import subset_scores


MODELS = {
    "cnn": "JKX_CNN",
    "vit": "DayAligned_ViT",
}

KEYS = [
    "ticker",
    "group",
    "decision_date",
    "target_date",
    "label",
]


def fingerprint(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1 << 20), b""
        ):
            h.update(block)

    return h.hexdigest()


class JointImages(Dataset):
    """Read historical/future images without copying large arrays."""

    def __init__(
        self,
        old_images,
        future_images,
        labels,
        ids,
        old_count,
    ):
        self.old = old_images
        self.future = future_images
        self.y = labels
        self.ids = np.asarray(ids, dtype=np.int64)
        self.old_count = old_count

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, pos):
        i = int(self.ids[pos])

        if i < self.old_count:
            image = self.old[i]
        else:
            image = self.future[i - self.old_count]

        x = (
            torch.from_numpy(
                np.array(image, copy=True)
            )
            .float()
            .div_(255.0)
        )

        return x, int(self.y[i])


def load_data(args):
    manifest = pd.read_csv(
        args.manifest,
        dtype={
            "ticker": str,
            "group": str,
        },
    )

    legacy = set(
        manifest.loc[
            manifest.group == "legacy",
            "ticker",
        ]
    )

    if (
        len(manifest) != 200
        or manifest.ticker.nunique() != 200
        or len(legacy) != 100
    ):
        raise ValueError(
            "Frozen 200-stock manifest is invalid"
        )

    tables = []
    images = []
    hashes = []

    sources = (
        (args.old, "historical"),
        (args.future, "future"),
    )

    for source, label in sources:
        meta_path = source / "meta.csv"
        img_path = source / "images.npy"
        ready_path = source / "READY.json"

        for path in (
            meta_path,
            img_path,
            ready_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing required file: {path}"
                )

        h = fingerprint(meta_path)

        ready = json.loads(
            ready_path.read_text(
                encoding="utf-8"
            )
        )

        if ready.get("fingerprint") != h:
            raise ValueError(
                f"Fingerprint mismatch: {source}"
            )

        frame = pd.read_csv(
            meta_path,
            dtype={
                "ticker": str,
                "group": str,
                "window_start": str,
                "decision_date": str,
                "target_date": str,
            },
        )

        array = np.load(
            img_path,
            mmap_mode="r",
        )

        if (
            array.shape != (len(frame), 1, 64, 60)
            or array.dtype != np.uint8
        ):
            raise ValueError(
                f"Image shape/dtype mismatch: {img_path}"
            )

        if frame.duplicated(
            ["ticker", "decision_date"]
        ).any():
            raise ValueError(
                f"Duplicate ticker/date: {source}"
            )

        if frame.ticker.nunique() != 200:
            raise ValueError(
                f"Expected 200 tickers: {source}"
            )

        expected_labels = (
            frame.return_5d.to_numpy(float) > 0
        ).astype(int)

        if not np.array_equal(
            expected_labels,
            frame.label.to_numpy(int),
        ):
            raise ValueError(
                f"Invalid labels: {source}"
            )

        if not np.isfinite(
            frame.return_5d.to_numpy(float)
        ).all():
            raise ValueError(
                f"Invalid future returns: {source}"
            )

        group_map = dict(
            zip(
                manifest.ticker,
                manifest.group,
            )
        )

        expected_groups = frame.ticker.map(
            group_map
        )

        if not expected_groups.equals(
            frame.group
        ):
            raise ValueError(
                f"Membership mismatch: {source}"
            )

        tables.append(frame)
        images.append(array)
        hashes.append(h)

        print(
            f"[DATA] {label}={len(frame):,} "
            f"samples | SHA256={h[:12]}",
            flush=True,
        )

    old, future = tables

    if (
        old.decision_date.max() >= "2022-01-01"
        or future.decision_date.min() < "2022-01-01"
    ):
        raise ValueError(
            "Historical/future date overlap"
        )

    if future.target_date.max() >= "2026-01-01":
        raise ValueError(
            "Future labels extend past 2025"
        )

    if set(old.ticker) != set(future.ticker):
        raise ValueError(
            "Stock universes differ"
        )

    combined = pd.concat(
        tables,
        ignore_index=True,
    )

    if combined.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError(
            "Duplicate observations across datasets"
        )

    labels = combined.label.to_numpy(
        np.int64
    )

    return (
        combined,
        images,
        len(old),
        labels,
        legacy,
        hashes,
    )


def split_year(meta, year):
    val_start = f"{year - 1}-01-01"
    test_start = f"{year}-01-01"
    test_end = f"{year + 1}-01-01"

    begins = meta.window_start.to_numpy(
        dtype=str
    )

    targets = meta.target_date.to_numpy(
        dtype=str
    )

    train_idx = np.flatnonzero(
        (begins >= "2010-01-01")
        & (targets < val_start)
    )

    val_idx = np.flatnonzero(
        (begins >= val_start)
        & (targets < test_start)
    )

    test_idx = np.flatnonzero(
        (begins >= test_start)
        & (targets < test_end)
    )

    if min(
        len(train_idx),
        len(val_idx),
        len(test_idx),
    ) < 1000:
        raise ValueError(
            f"Insufficient samples: {year}"
        )

    train_frame = meta.iloc[train_idx]
    val_frame = meta.iloc[val_idx]
    test_frame = meta.iloc[test_idx]

    # Prevent overlapping input/target windows.
    if (
        train_frame.target_date.max()
        >= val_frame.window_start.min()
    ):
        raise ValueError(
            f"Train/validation purge violated: {year}"
        )

    if (
        val_frame.target_date.max()
        >= test_frame.window_start.min()
    ):
        raise ValueError(
            f"Validation/test purge violated: {year}"
        )

    if test_frame.ticker.nunique() != 200:
        raise ValueError(
            f"Incomplete test universe: {year}"
        )

    if test_frame.decision_date.str[:4].ne(
        str(year)
    ).any():
        raise ValueError(
            f"Wrong test year: {year}"
        )

    if test_frame.target_date.str[:4].ne(
        str(year)
    ).any():
        raise ValueError(
            f"Target escapes test year: {year}"
        )

    return (
        train_idx,
        val_idx,
        test_idx,
        test_frame.reset_index(drop=True),
    )


def make_loader(
    images,
    labels,
    ids,
    old_count,
    args,
    seed,
    shuffle,
):
    dataset = JointImages(
        images[0],
        images[1],
        labels,
        ids,
        old_count,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        generator=(
            torch.Generator().manual_seed(seed)
        ),
    )


def folder_for(args, model, year, seed):
    root = (
        args.cnn_out
        if model == "cnn"
        else args.vit_out
    )

    return (
        root
        / "u200_start2010"
        / f"test{year}"
        / f"seed{seed}"
    )


def run_one(
    args,
    device,
    meta,
    images,
    old_count,
    labels,
    legacy,
    hashes,
    model_name,
    year,
    seed,
):
    folder = folder_for(
        args,
        model_name,
        year,
        seed,
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = folder / "summary.json"
    pred_file = folder / "predictions.csv"

    config = {
        "experiment": "exp200_holdout_walkforward_v1",
        "old_fingerprint": hashes[0],
        "future_fingerprint": hashes[1],
        "model": model_name,
        "train_universe": 200,
        "start_year": 2010,
        "year": year,
        "val_year": year - 1,
        "seed": seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "optimizer": (
            "Adam"
            if model_name == "cnn"
            else "AdamW"
        ),
        "lr": (
            1e-5
            if model_name == "cnn"
            else 0.000253
        ),
        "weight_decay": (
            0.0
            if model_name == "cnn"
            else 0.000325
        ),
        "dim": (
            None
            if model_name == "cnn"
            else 128
        ),
        "depth": (
            None
            if model_name == "cnn"
            else 2
        ),
        "heads": (
            None
            if model_name == "cnn"
            else 4
        ),
        "dropout": (
            None
            if model_name == "cnn"
            else 0.1
        ),
    }

    # Resume: skip only fully completed runs.
    if (
        summary_file.exists()
        and pred_file.exists()
    ):
        old = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        if old.get("config") != config:
            raise ValueError(
                f"Completed config differs: {folder}"
            )

        frame = pd.read_csv(
            pred_file,
            usecols=[
                "ticker",
                "decision_date",
                "prob_up",
            ],
        )

        _, _, expected_idx, _ = split_year(
            meta, year
        )

        if (
            len(frame) != len(expected_idx)
            or frame.prob_up.isna().any()
        ):
            raise ValueError(
                f"Invalid completed predictions: {folder}"
            )

        print(
            f"[SKIP DONE] "
            f"{model_name} {year} seed{seed}",
            flush=True,
        )
        return

    train_idx, val_idx, test_idx, test_frame = (
        split_year(meta, year)
    )

    set_seed(seed)

    train_loader = make_loader(
        images,
        labels,
        train_idx,
        old_count,
        args,
        seed,
        True,
    )

    val_loader = make_loader(
        images,
        labels,
        val_idx,
        old_count,
        args,
        seed,
        False,
    )

    test_loader = make_loader(
        images,
        labels,
        test_idx,
        old_count,
        args,
        seed,
        False,
    )

    if model_name == "cnn":
        model = Jkx20().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
        )

    else:
        model = DayAlignedViT(
            dim=128,
            depth=2,
            heads=4,
            dropout=0.1,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.000253,
            weight_decay=0.000325,
        )

    started = time.perf_counter()

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[RUN] {model_name} "
        f"year={year} seed={seed} "
        f"train={len(train_idx):,} "
        f"val={len(val_idx):,} "
        f"test={len(test_idx):,} "
        f"params={n_params:,}",
        flush=True,
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()

        total_loss = 0.0
        n = 0

        for step, (x, y) in enumerate(
            train_loader, 1
        ):
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

            loss = nn.functional.cross_entropy(
                logits, y
            )

            loss.backward()

            if model_name == "vit":
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

            optimizer.step()

            total_loss += (
                float(loss.item()) * len(y)
            )

            n += len(y)

            if (
                args.log_steps
                and step % args.log_steps == 0
            ):
                print(
                    f"[BATCH] {model_name} "
                    f"{year} seed{seed} "
                    f"ep={epoch} "
                    f"{step}/{len(train_loader)} "
                    f"loss={total_loss/n:.5f}",
                    flush=True,
                )

        val, _ = evaluate(
            model,
            val_loader,
            device,
        )

        improved = (
            val["loss"] < best_loss - 1e-6
        )

        if improved:
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": epoch,
                    "config": config,
                    "validation": val,
                },
                folder / "best.pt",
            )
        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / n,
            **{
                f"val_{k}": v
                for k, v in val.items()
            },
        })

        pd.DataFrame(history).to_csv(
            folder / "history.csv",
            index=False,
        )

        elapsed = (
            time.perf_counter() - started
        )

        print(
            f"[EPOCH] {model_name} "
            f"{year} seed{seed} "
            f"ep={epoch:02d} "
            f"train={total_loss/n:.5f} "
            f"val={val['loss']:.5f} "
            f"val_auc={val['auc']:.4f} "
            f"best={best_epoch} "
            f"stale={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] "
                f"{model_name} {year} seed{seed}",
                flush=True,
            )
            break

    checkpoint = torch.load(
        folder / "best.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    test_metrics, probabilities = evaluate(
        model,
        test_loader,
        device,
    )

    prediction = test_frame[
        [
            "ticker",
            "group",
            "decision_date",
            "target_date",
            "return_5d",
            "label",
        ]
    ].copy()

    prediction["prob_up"] = probabilities

    prediction.to_csv(
        pred_file,
        index=False,
    )

    groups = subset_scores(
        test_frame,
        probabilities,
        legacy,
    )

    elapsed = (
        time.perf_counter() - started
    )

    doc = {
        "config": config,
        "best_epoch": best_epoch,
        "seconds": round(elapsed, 2),
        "test": test_metrics,
        "groups": groups,
    }

    summary_file.write_text(
        json.dumps(doc, indent=2),
        encoding="utf-8",
    )

    print(
        f"[DONE] {model_name} "
        f"year={year} seed={seed} "
        f"best_epoch={best_epoch} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )


def report(args, legacy):
    rows = []
    seed_rows = []

    for year in args.years:
        year_reference = None

        for model_name, printable in MODELS.items():
            frames = []

            for seed in args.seeds:
                path = (
                    folder_for(
                        args,
                        model_name,
                        year,
                        seed,
                    )
                    / "predictions.csv"
                )

                if not path.is_file():
                    raise FileNotFoundError(
                        f"Not yet trained: {path}"
                    )

                frame = pd.read_csv(
                    path,
                    dtype={
                        "ticker": str,
                        "group": str,
                        "decision_date": str,
                        "target_date": str,
                    },
                )

                frames.append(frame)

                doc_path = (
                    path.parent / "summary.json"
                )

                if not doc_path.is_file():
                    raise FileNotFoundError(
                        f"Missing summary: {doc_path}"
                    )

                doc = json.loads(
                    doc_path.read_text(
                        encoding="utf-8"
                    )
                )

                common = doc["groups"]["common100"]

                seed_rows.append({
                    "year": year,
                    "model": printable,
                    "seed": seed,
                    "auc": common["auc"],
                    "rank_ic": common["rank_ic"],
                    "top10_excess_5d": (
                        common["top10_excess_5d"]
                    ),
                })

            first = frames[0]

            if (
                first.ticker.nunique() != 200
                or first.decision_date.str[:4].ne(
                    str(year)
                ).any()
            ):
                raise ValueError(
                    f"Invalid test universe: "
                    f"{printable} {year}"
                )

            if year_reference is None:
                year_reference = first

            else:
                if not year_reference[KEYS].equals(
                    first[KEYS]
                ):
                    raise ValueError(
                        f"CNN/ViT alignment mismatch: {year}"
                    )

                if not np.allclose(
                    year_reference.return_5d,
                    first.return_5d,
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"CNN/ViT return mismatch: {year}"
                    )

            for other in frames[1:]:
                if not first[KEYS].equals(
                    other[KEYS]
                ):
                    raise ValueError(
                        f"Seed alignment mismatch: "
                        f"{printable} {year}"
                    )

                if not np.allclose(
                    first.return_5d,
                    other.return_5d,
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"Seed return mismatch: "
                        f"{printable} {year}"
                    )

            probabilities = np.stack([
                f.prob_up.to_numpy(float)
                for f in frames
            ]).mean(axis=0)

            ensemble = first.copy()
            ensemble["prob_up"] = probabilities

            out_dir = (
                args.report_out
                / "ensemble_predictions"
                / f"test{year}"
            )

            out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            ensemble.to_csv(
                out_dir / f"{printable}.csv",
                index=False,
            )

            evaluations = (
                (
                    "common100",
                    ensemble.ticker.isin(
                        legacy
                    ).to_numpy(),
                ),
                (
                    "all200",
                    np.ones(
                        len(ensemble),
                        dtype=bool,
                    ),
                ),
            )

            for evaluation, mask in evaluations:
                selected = (
                    ensemble.loc[mask]
                    .reset_index(drop=True)
                )

                stats = subset_scores(
                    selected,
                    probabilities[mask],
                    legacy,
                )

                chosen = stats["all"]

                rows.append({
                    "year": year,
                    "model": printable,
                    "evaluation": evaluation,
                    "seed_count": len(args.seeds),
                    "auc": chosen["auc"],
                    "rank_ic": chosen["rank_ic"],
                    "top10_excess_5d": (
                        chosen["top10_excess_5d"]
                    ),
                    "top10_bottom10_spread_5d": (
                        chosen[
                            "top10_bottom10_spread_5d"
                        ]
                    ),
                    "balanced_acc": (
                        chosen["balanced_acc"]
                    ),
                })

    results = pd.DataFrame(rows)

    args.report_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.report_out / "ensemble_summary.csv",
        index=False,
    )

    pd.DataFrame(seed_rows).to_csv(
        args.report_out / "individual_seed_summary.csv",
        index=False,
    )

    common = results.loc[
        results.evaluation == "common100"
    ]

    print(
        f"\n[HOLDOUT ENSEMBLE] "
        f"Common100 | {len(args.seeds)} seeds",
        flush=True,
    )

    print(
        common.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    cols = [
        "auc",
        "rank_ic",
        "top10_excess_5d",
        "top10_bottom10_spread_5d",
    ]

    print(
        "\n[4-YEAR MEAN]",
        flush=True,
    )

    print(
        common.groupby("model")[cols]
        .mean()
        .to_string(
            float_format=lambda x: f"{x:+.5f}"
        ),
        flush=True,
    )

    variability = (
        pd.DataFrame(seed_rows)
        .groupby("model")[
            ["auc", "rank_ic"]
        ]
        .agg(["mean", "std"])
    )

    print(
        "\n[INDIVIDUAL SEED MEAN / STD]",
        flush=True,
    )

    print(
        variability.to_string(
            float_format=lambda x: f"{x:+.5f}"
        ),
        flush=True,
    )

    print(
        f"[SAVED] {args.report_out}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage",
        choices=["check", "train", "report"],
        default="check",
    )

    p.add_argument(
        "--old",
        type=Path,
        default=Path("data/exp200_jkx"),
    )

    p.add_argument(
        "--future",
        type=Path,
        default=Path("data/exp200_holdout_jkx"),
    )

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
        ),
    )

    p.add_argument(
        "--cnn-out",
        type=Path,
        default=Path(
            "outputs/exp200_holdout_cnn"
        ),
    )

    p.add_argument(
        "--vit-out",
        type=Path,
        default=Path(
            "outputs/exp200_holdout_vit"
        ),
    )

    p.add_argument(
        "--report-out",
        type=Path,
        default=Path(
            "outputs/exp200_holdout_10seed_eval"
        ),
    )

    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
    )

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
    )

    p.add_argument(
        "--models",
        nargs="+",
        choices=["cnn", "vit"],
        default=["cnn", "vit"],
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--log-steps",
        type=int,
        default=0,
    )

    args = p.parse_args()

    invalid = (
        not args.years
        or len(set(args.years)) != len(args.years)
        or not set(args.years).issubset(
            {2022, 2023, 2024, 2025}
        )
        or not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.log_steps < 0
    )

    if invalid:
        p.error(
            "Invalid experiment parameters"
        )

    (
        meta,
        images,
        old_count,
        labels,
        legacy,
        hashes,
    ) = load_data(args)

    if args.stage == "check":
        for year in args.years:
            tr, va, te, _ = split_year(
                meta, year
            )

            print(
                f"[SPLIT OK] {year} "
                f"train={len(tr):,} "
                f"val={len(va):,} "
                f"test={len(te):,} "
                f"| 200 tickers | purge verified",
                flush=True,
            )

        print(
            "[READY TO TRAIN] "
            "Historical and holdout data unchanged",
            flush=True,
        )
        return

    if args.stage == "report":
        report(args, legacy)
        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    jobs = [
        (year, model, seed)
        for year in args.years
        for model in args.models
        for seed in args.seeds
    ]

    print(
        f"[DEVICE] {device} "
        f"| [PLAN] {len(jobs)} runs "
        f"| resume by completed predictions",
        flush=True,
    )

    for n, (year, model, seed) in enumerate(
        jobs, 1
    ):
        print(
            f"\n[PROGRESS] {n}/{len(jobs)}",
            flush=True,
        )

        run_one(
            args,
            device,
            meta,
            images,
            old_count,
            labels,
            legacy,
            hashes,
            model,
            year,
            seed,
        )

    print(
        "[TRAIN COMPLETE] "
        "Use --stage report to aggregate ensembles",
        flush=True,
    )


if __name__ == "__main__":
    main()
