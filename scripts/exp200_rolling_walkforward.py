
"""Rolling 5/7-year walk-forward diagnostic.

Run:
  python -u -m scripts.exp200_rolling_walkforward --stage check --window-years 5
  python -u -m scripts.exp200_rolling_walkforward --stage train --window-years 5
  python -u -m scripts.exp200_rolling_walkforward --stage report --window-years 5

Existing expanding-window models and results remain unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts import exp200_holdout_walkforward as base


def split_rolling(meta, year, window_years):
    train_start = f"{year - 1 - window_years}-01-01"
    val_start = f"{year - 1}-01-01"
    test_start = f"{year}-01-01"
    test_end = f"{year + 1}-01-01"

    begins = meta.window_start.to_numpy(dtype=str)
    targets = meta.target_date.to_numpy(dtype=str)

    tr = np.flatnonzero(
        (begins >= train_start)
        & (targets < val_start)
    )

    va = np.flatnonzero(
        (begins >= val_start)
        & (targets < test_start)
    )

    te = np.flatnonzero(
        (begins >= test_start)
        & (targets < test_end)
    )

    if min(map(len, (tr, va, te))) < 1000:
        raise ValueError(
            f"Too few observations: "
            f"year={year}, rolling={window_years}"
        )

    a = meta.iloc[tr]
    b = meta.iloc[va]
    c = meta.iloc[te]

    if a.target_date.max() >= b.window_start.min():
        raise ValueError(
            f"Train/validation leakage: {year}"
        )

    if b.target_date.max() >= c.window_start.min():
        raise ValueError(
            f"Validation/test leakage: {year}"
        )

    if (
        b.ticker.nunique() != 200
        or c.ticker.nunique() != 200
    ):
        raise ValueError(
            f"Missing validation/test stocks: {year}"
        )

    if (
        (c.decision_date.str[:4] != str(year)).any()
        or (c.target_date.str[:4] != str(year)).any()
    ):
        raise ValueError(
            f"Incorrect test dates: {year}"
        )

    return tr, va, te, c.reset_index(drop=True)


def train_one(
    a,
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
    folder = base.folder_for(
        a, model_name, year, seed
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = folder / "summary.json"
    pred_file = folder / "predictions.csv"
    ckpt_file = folder / "best.pt"

    tr, va, te, frame = split_rolling(
        meta, year, a.window_years
    )

    lr = (
        1e-5 if model_name == "cnn"
        else 0.000253
    )

    wd = (
        0.0 if model_name == "cnn"
        else 0.000325
    )

    config = {
        "experiment": "exp200_rolling_window_diagnostic_v1",
        "old_fingerprint": hashes[0],
        "future_fingerprint": hashes[1],
        "train_universe": 200,
        "evaluation_universe": "common100",
        "model": model_name,
        "test_year": year,
        "validation_year": year - 1,
        "rolling_train_years": a.window_years,
        "train_start_year": (
            year - 1 - a.window_years
        ),
        "train_end_year": year - 2,
        "seed": seed,
        "epochs": a.epochs,
        "patience": a.patience,
        "batch_size": a.batch_size,
        "lr": lr,
        "weight_decay": wd,
        "optimizer": (
            "Adam" if model_name == "cnn"
            else "AdamW"
        ),
        "dim": (
            None if model_name == "cnn"
            else 128
        ),
        "depth": (
            None if model_name == "cnn"
            else 2
        ),
        "heads": (
            None if model_name == "cnn"
            else 4
        ),
        "dropout": (
            None if model_name == "cnn"
            else 0.1
        ),
    }

    # Skip only completed matching experiments.
    if summary_file.exists() and pred_file.exists():
        previous = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        if previous.get("config") != config:
            raise ValueError(
                f"Existing config differs: {folder}"
            )

        pred = pd.read_csv(
            pred_file,
            dtype={
                "ticker": str,
                "decision_date": str,
                "target_date": str,
            },
        )

        check_cols = [
            "ticker",
            "decision_date",
            "target_date",
            "label",
        ]

        if (
            len(pred) != len(frame)
            or not pred[check_cols].equals(
                frame[check_cols]
            )
            or not np.isfinite(
                pred.prob_up.to_numpy(float)
            ).all()
        ):
            raise ValueError(
                f"Invalid completed predictions: {pred_file}"
            )

        print(
            f"[SKIP DONE] rolling{a.window_years} "
            f"{model_name} {year} seed{seed}",
            flush=True,
        )
        return

    base.set_seed(seed)

    train_loader = base.make_loader(
        images, labels, tr, old_count,
        a, seed, True
    )

    val_loader = base.make_loader(
        images, labels, va, old_count,
        a, seed, False
    )

    test_loader = base.make_loader(
        images, labels, te, old_count,
        a, seed, False
    )

    if model_name == "cnn":
        model = base.Jkx20().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

    else:
        model = base.DayAlignedViT(
            dim=128,
            depth=2,
            heads=4,
            dropout=0.1,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    print(
        f"[RUN] rolling{a.window_years} "
        f"{model_name} year={year} seed={seed} "
        f"train={len(tr):,} "
        f"val={len(va):,} "
        f"test={len(te):,}",
        flush=True,
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    started = time.perf_counter()

    for epoch in range(1, a.epochs + 1):
        model.train()

        loss_sum = 0.0
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

            loss_sum += (
                float(loss.item()) * len(y)
            )

            n += len(y)

            if (
                a.log_steps
                and step % a.log_steps == 0
            ):
                print(
                    f"[BATCH] rolling{a.window_years} "
                    f"{model_name} {year} seed{seed} "
                    f"ep={epoch} "
                    f"batch={step}/{len(train_loader)} "
                    f"loss={loss_sum/n:.5f}",
                    flush=True,
                )

        val, _ = base.evaluate(
            model,
            val_loader,
            device,
        )

        if val["loss"] < best_loss - 1e-6:
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
                ckpt_file,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / n,
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
            f"[EPOCH] rolling{a.window_years} "
            f"{model_name} {year} seed{seed} "
            f"ep={epoch} "
            f"train={loss_sum/n:.5f} "
            f"val={val['loss']:.5f} "
            f"val_auc={val['auc']:.4f} "
            f"best={best_epoch} "
            f"stale={stale}/{a.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= a.patience:
            print(
                "[EARLY STOP] validation loss",
                flush=True,
            )
            break

    checkpoint = torch.load(
        ckpt_file,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    test_metrics, probabilities = (
        base.evaluate(
            model,
            test_loader,
            device,
        )
    )

    prediction = frame[
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

    groups = base.subset_scores(
        frame,
        probabilities,
        legacy,
    )

    summary = {
        "config": config,
        "best_epoch": best_epoch,
        "seconds": round(
            time.perf_counter() - started, 2
        ),
        "train_n": len(tr),
        "val_n": len(va),
        "test_n": len(te),
        "test": test_metrics,
        "groups": groups,
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[DONE] rolling{a.window_years} "
        f"{model_name} {year} seed{seed} "
        f"common100_AUC="
        f"{groups['common100']['auc']:.5f} "
        f"RankIC="
        f"{groups['common100']['rank_ic']:+.5f}",
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Post-hoc rolling-window 5/7y analysis"
        )
    )

    p.add_argument(
        "--stage",
        choices=["check", "train", "report"],
        default="check",
    )

    p.add_argument(
        "--window-years",
        type=int,
        choices=[5, 7],
        required=True,
    )

    p.add_argument(
        "--old",
        type=Path,
        default=Path("data/exp200_jkx"),
    )

    p.add_argument(
        "--future",
        type=Path,
        default=Path(
            "data/exp200_holdout_jkx"
        ),
    )

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/exp200/universe200.csv"
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
        choices=["cnn", "vit"],
        nargs="+",
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

    a = p.parse_args()

    invalid = (
        not a.years
        or len(set(a.years)) != len(a.years)
        or not set(a.years).issubset(
            {2022, 2023, 2024, 2025}
        )
        or not a.seeds
        or len(set(a.seeds)) != len(a.seeds)
        or a.epochs < 1
        or a.patience < 1
        or a.batch_size < 1
        or a.workers < 0
        or a.log_steps < 0
    )

    if invalid:
        p.error(
            "Invalid experiment parameters"
        )

    prefix = (
        f"exp200_rolling{a.window_years}"
    )

    a.cnn_out = (
        Path("outputs") / f"{prefix}_cnn"
    )

    a.vit_out = (
        Path("outputs") / f"{prefix}_vit"
    )

    a.report_out = (
        Path("outputs")
        / f"{prefix}_10seed_eval"
    )

    return a


def main():
    a = parse_args()

    (
        meta,
        images,
        old_count,
        labels,
        legacy,
        hashes,
    ) = base.load_data(a)

    for year in a.years:
        tr, va, te, _ = split_rolling(
            meta,
            year,
            a.window_years,
        )

        print(
            f"[SPLIT OK] rolling{a.window_years} "
            f"test={year} "
            f"train={year-1-a.window_years}"
            f"..{year-2} "
            f"val={year-1} "
            f"test={year} "
            f"N={len(tr):,}/{len(va):,}/{len(te):,} "
            f"| purge checked",
            flush=True,
        )

    if a.stage == "check":
        print(
            "[READY] Historical and holdout "
            "datasets untouched",
            flush=True,
        )
        return

    if a.stage == "report":
        base.report(a, legacy)
        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    jobs = [
        (year, model, seed)
        for year in a.years
        for model in a.models
        for seed in a.seeds
    ]

    print(
        f"[PLAN] rolling{a.window_years}: "
        f"{len(jobs)} runs | device={device}",
        flush=True,
    )

    for i, (year, model, seed) in enumerate(
        jobs, 1
    ):
        print(
            f"\n[PROGRESS] {i}/{len(jobs)}",
            flush=True,
        )

        train_one(
            a,
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
        "[TRAIN COMPLETE] Run --stage report",
        flush=True,
    )


if __name__ == "__main__":
    main()
