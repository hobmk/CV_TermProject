
"""Compare absolute vs relative return labels using JKX CNN.

Reuses existing Exp200 images and absolute-direction predictions.
Trains 6 additional relative-label models by default.
No downloads and no access to 2022+ data.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
)

from torch import nn
from torch.utils.data import DataLoader

from scripts.jkx_i20_reproduce import (
    Jkx20,
    MemmapCharts,
    evaluate,
    cross_section_metrics,
    set_seed,
)

from scripts.run_exp200_factorial import (
    prepare_data,
    split,
)


class RelativeCharts(MemmapCharts):
    def __init__(self, root, indices, labels):
        super().__init__(
            root / "images.npy",
            root / "meta.csv",
            indices,
        )

        if len(labels) != len(self.meta):
            raise ValueError(
                "Relative label length mismatch"
            )

        self.labels = np.asarray(
            labels,
            dtype=np.int64,
        )


def relative_labels(meta, legacy, universe):
    if universe == 100:
        eligible = meta.ticker.isin(legacy)
    else:
        eligible = pd.Series(
            True,
            index=meta.index,
        )

    selected = meta.loc[eligible]

    daily_mean = (
        selected.groupby("decision_date")[
            "return_5d"
        ].transform("mean")
    )

    labels = np.zeros(
        len(meta),
        dtype=np.int64,
    )

    indices = np.flatnonzero(
        eligible.to_numpy()
    )

    labels[indices] = (
        selected.return_5d.to_numpy(float)
        > daily_mean.to_numpy(float)
    ).astype(np.int64)

    if (
        len(indices) == 0
        or not np.isfinite(
            daily_mean.to_numpy(float)
        ).all()
    ):
        raise ValueError(
            "Invalid relative labels"
        )

    return labels


def verify_baseline(
    args,
    meta,
    fingerprint,
    universe,
    year,
    seed,
    test_indices,
):
    folder = (
        args.baseline
        / f"u{universe}_start{args.start_year}"
        / f"test{year}"
        / f"seed{seed}"
    )

    summary_path = folder / "summary.json"
    prediction_path = (
        folder / "predictions.csv"
    )

    if (
        not summary_path.is_file()
        or not prediction_path.is_file()
    ):
        raise FileNotFoundError(
            f"Missing existing baseline: {folder}"
        )

    config = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )["config"]

    required = {
        "data_fingerprint": fingerprint,
        "universe": universe,
        "start_year": args.start_year,
        "test_year": year,
        "seed": seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }

    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Baseline config mismatch: "
                f"{key} in {folder}"
            )

    if not np.isclose(
        float(config["lr"]),
        args.lr,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Baseline learning rate mismatch"
        )

    predictions = pd.read_csv(
        prediction_path,
        dtype={
            "ticker": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    frame = (
        meta.iloc[test_indices]
        .reset_index(drop=True)
    )

    columns = [
        "ticker",
        "decision_date",
        "target_date",
        "group",
        "label",
    ]

    if not frame[columns].equals(
        predictions[columns]
    ):
        raise ValueError(
            "Baseline sample alignment mismatch"
        )

    if not np.allclose(
        frame.return_5d,
        predictions.return_5d,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "Baseline return mismatch"
        )

    return predictions.prob_up.to_numpy(
        float
    )


def calculate_group_metrics(
    frame,
    probabilities,
    legacy,
):
    masks = {
        "common100": (
            frame.ticker.isin(legacy).to_numpy()
        ),
        "all": np.ones(
            len(frame),
            dtype=bool,
        ),
    }

    results = {}

    for name, mask in masks.items():
        subset = (
            frame.loc[mask]
            .reset_index(drop=True)
        )

        scores = np.asarray(
            probabilities,
            dtype=float,
        )[mask]

        absolute_y = (
            subset.label.to_numpy(int)
        )

        daily_mean = (
            subset.groupby("decision_date")[
                "return_5d"
            ].transform("mean")
        )

        relative_y = (
            subset.return_5d.to_numpy(float)
            > daily_mean.to_numpy(float)
        ).astype(int)

        if (
            len(np.unique(absolute_y)) != 2
            or len(np.unique(relative_y)) != 2
        ):
            raise ValueError(
                "One-class evaluation set"
            )

        metrics = cross_section_metrics(
            subset,
            scores,
        )

        metrics.update({
            "absolute_auc": float(
                roc_auc_score(
                    absolute_y,
                    scores,
                )
            ),
            "relative_auc": float(
                roc_auc_score(
                    relative_y,
                    scores,
                )
            ),
            "relative_balanced_accuracy": float(
                balanced_accuracy_score(
                    relative_y,
                    scores >= 0.5,
                )
            ),
            "relative_positive_rate": float(
                relative_y.mean()
            ),
            "predicted_positive_rate": float(
                (scores >= 0.5).mean()
            ),
            "stocks": int(
                subset.ticker.nunique()
            ),
            "samples": len(subset),
        })

        results[name] = metrics

    return results


def make_loader(
    dataset,
    args,
    device,
    shuffle,
    seed,
):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.workers > 0
        ),
        generator=(
            torch.Generator().manual_seed(seed)
            if shuffle else None
        ),
    )


def train_relative(
    args,
    device,
    meta,
    legacy,
    fingerprint,
    universe,
    year,
    seed,
):
    tag = (
        f"u{universe}_start{args.start_year}"
        f"/test{year}/seed{seed}"
    )

    folder = args.out / tag

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        folder / "summary.json"
    )

    prediction_path = (
        folder / "predictions.csv"
    )

    config = {
        "data_fingerprint": fingerprint,
        "target": "relative_5d_return",
        "universe": universe,
        "start_year": args.start_year,
        "test_year": year,
        "seed": seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    if (
        summary_path.is_file()
        and prediction_path.is_file()
        and not args.force
    ):
        old = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        if old.get("config") != config:
            raise ValueError(
                f"Configuration mismatch: {folder}"
            )

        print(
            f"[SKIP DONE] {tag}",
            flush=True,
        )

        return

    tr, va, te, test_frame = split(
        meta,
        legacy,
        year,
        args.start_year,
        universe,
    )

    labels = relative_labels(
        meta,
        legacy,
        universe,
    )

    set_seed(seed)

    train_ds = RelativeCharts(
        args.root,
        tr,
        labels,
    )

    val_ds = RelativeCharts(
        args.root,
        va,
        labels,
    )

    test_ds = RelativeCharts(
        args.root,
        te,
        labels,
    )

    train_loader = make_loader(
        train_ds,
        args,
        device,
        True,
        seed,
    )

    val_loader = make_loader(
        val_ds,
        args,
        device,
        False,
        seed,
    )

    test_loader = make_loader(
        test_ds,
        args,
        device,
        False,
        seed,
    )

    model = Jkx20().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    started = time.perf_counter()

    print(
        f"[RUN] {tag} "
        f"train={len(tr):,} "
        f"val={len(va):,} "
        f"test={len(te):,} "
        f"relative_up={labels[tr].mean():.3f}",
        flush=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        seen = 0

        for batch, (
            images,
            targets,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(
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

            logits = model(images)

            loss = nn.functional.cross_entropy(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(targets)
            )

            seen += len(targets)

            if (
                args.log_steps
                and batch % args.log_steps == 0
            ):
                print(
                    f"[BATCH] {tag} "
                    f"ep={epoch} "
                    f"{batch}/{len(train_loader)} "
                    f"loss={total_loss/seen:.5f}",
                    flush=True,
                )

        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
        )

        improved = (
            val_metrics["loss"]
            < best_loss - 1e-6
        )

        if improved:
            best_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "config": config,
                },
                folder / "best.pt",
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_loss": (
                total_loss / seen
            ),
            "val_loss": (
                val_metrics["loss"]
            ),
            "val_relative_auc": (
                val_metrics["auc"]
            ),
            "val_positive_ratio": (
                val_metrics["up_ratio"]
            ),
        })

        pd.DataFrame(history).to_csv(
            folder / "history.csv",
            index=False,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"[EPOCH] {tag} "
            f"ep={epoch:02d} "
            f"train={total_loss/seen:.5f} "
            f"val={val_metrics['loss']:.5f} "
            f"val_rel_auc={val_metrics['auc']:.4f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {tag}",
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

    _, probabilities = evaluate(
        model,
        test_loader,
        device,
    )

    predictions = test_frame[
        [
            "ticker",
            "group",
            "decision_date",
            "target_date",
            "return_5d",
            "label",
        ]
    ].copy()

    predictions[
        "prob_relative_positive"
    ] = probabilities

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            {
                "config": config,
                "best_epoch": best_epoch,
                "seconds": (
                    time.perf_counter() - started
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[DONE] {tag} "
        f"best_epoch={best_epoch}",
        flush=True,
    )


def report(
    args,
    meta,
    legacy,
    fingerprint,
):
    rows = []

    for year in args.years:
        for universe in args.universes:
            for seed in args.seeds:

                _, _, te, frame = split(
                    meta,
                    legacy,
                    year,
                    args.start_year,
                    universe,
                )

                absolute_scores = verify_baseline(
                    args,
                    meta,
                    fingerprint,
                    universe,
                    year,
                    seed,
                    te,
                )

                folder = (
                    args.out
                    / f"u{universe}_start{args.start_year}"
                    / f"test{year}"
                    / f"seed{seed}"
                )

                path = (
                    folder / "predictions.csv"
                )

                if not path.is_file():
                    print(
                        f"[MISSING] {folder}",
                        flush=True,
                    )
                    continue

                relative_predictions = pd.read_csv(
                    path,
                    dtype={
                        "ticker": str,
                        "decision_date": str,
                        "target_date": str,
                    },
                )

                columns = [
                    "ticker",
                    "decision_date",
                    "target_date",
                    "label",
                ]

                if not frame[columns].equals(
                    relative_predictions[columns]
                ):
                    raise ValueError(
                        "Relative prediction alignment mismatch"
                    )

                if not np.allclose(
                    frame.return_5d,
                    relative_predictions.return_5d,
                    atol=1e-10,
                    rtol=0,
                ):
                    raise ValueError(
                        "Relative return mismatch"
                    )

                config = json.loads(
                    (
                        folder / "summary.json"
                    ).read_text(
                        encoding="utf-8"
                    )
                )["config"]

                expected = {
                    "data_fingerprint": fingerprint,
                    "universe": universe,
                    "start_year": args.start_year,
                    "test_year": year,
                    "seed": seed,
                }

                for key, value in expected.items():
                    if config.get(key) != value:
                        raise ValueError(
                            f"Relative config mismatch: {key}"
                        )

                relative_scores = (
                    relative_predictions[
                        "prob_relative_positive"
                    ].to_numpy(float)
                )

                for target, scores in (
                    ("absolute", absolute_scores),
                    ("relative", relative_scores),
                ):
                    groups = calculate_group_metrics(
                        frame,
                        scores,
                        legacy,
                    )

                    for group_name, stats in groups.items():
                        rows.append({
                            "year": year,
                            "universe": universe,
                            "start_year": args.start_year,
                            "seed": seed,
                            "target": target,
                            "eval_group": group_name,
                            **stats,
                        })

    if not rows:
        print(
            "[REPORT] No completed pairs",
            flush=True,
        )
        return

    df = pd.DataFrame(rows)

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.out / "target_comparison.csv",
        index=False,
    )

    keys = [
        "year",
        "universe",
        "start_year",
        "seed",
        "eval_group",
    ]

    metrics = [
        "absolute_auc",
        "relative_auc",
        "rank_ic",
        "top10_excess_5d",
        "top10_bottom10_spread_5d",
    ]

    paired = df.pivot(
        index=keys,
        columns="target",
        values=metrics,
    )

    paired.columns = [
        f"{metric}_{target}"
        for metric, target
        in paired.columns
    ]

    paired = (
        paired.dropna()
        .reset_index()
    )

    for metric in metrics:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_relative"]
            - paired[f"{metric}_absolute"]
        )

    paired.to_csv(
        args.out / "paired_deltas.csv",
        index=False,
    )

    common = df[
        df.eval_group == "common100"
    ].copy()

    print(
        "\n[COMMON100] "
        "Absolute vs Relative",
        flush=True,
    )

    print(
        common[
            [
                "year",
                "universe",
                "target",
                "absolute_auc",
                "relative_auc",
                "rank_ic",
                "top10_excess_5d",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    delta_columns = [
        "delta_absolute_auc",
        "delta_relative_auc",
        "delta_rank_ic",
        "delta_top10_excess_5d",
    ]

    avg = (
        paired[
            paired.eval_group == "common100"
        ]
        .groupby("universe")[delta_columns]
        .mean()
    )

    print(
        "\n[AVERAGE DELTA] "
        "Relative minus Absolute",
        flush=True,
    )

    print(
        avg.to_string(
            float_format=lambda x: f"{x:+.5f}"
        ),
        flush=True,
    )

    print(
        f"[SAVED] "
        f"{args.out / 'target_comparison.csv'}",
        flush=True,
    )

    print(
        f"[SAVED] "
        f"{args.out / 'paired_deltas.csv'}",
        flush=True,
    )

    print(
        "[HOLDOUT] 2022-2025 not accessed"
    )

    print(
        "[NOTE] Overlapping 5d returns; "
        "not a tradable backtest"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/exp200_jkx"),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/exp200"),
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/exp200_factorial"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/exp200_relative_target"),
    )

    parser.add_argument(
        "--stage",
        choices=["train", "report"],
        default="train",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        choices=[2010, 2015],
        default=2010,
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021],
    )

    parser.add_argument(
        "--universes",
        nargs="+",
        type=int,
        choices=[100, 200],
        default=[100, 200],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
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
        default=500,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.workers < 0
        or args.log_steps < 0
    ):
        parser.error(
            "Invalid training configuration"
        )

    meta, legacy, fingerprint = prepare_data(
        args.root,
        args.raw_root,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.stage == "train":
        total = (
            len(args.years)
            * len(args.universes)
            * len(args.seeds)
        )

        print(
            f"[DEVICE] {device} "
            f"| [PLAN] {total} "
            f"relative-target trainings",
            flush=True,
        )

        for year in args.years:
            for universe in args.universes:
                for seed in args.seeds:

                    _, _, te, _ = split(
                        meta,
                        legacy,
                        year,
                        args.start_year,
                        universe,
                    )

                    verify_baseline(
                        args,
                        meta,
                        fingerprint,
                        universe,
                        year,
                        seed,
                        te,
                    )

                    train_relative(
                        args,
                        device,
                        meta,
                        legacy,
                        fingerprint,
                        universe,
                        year,
                        seed,
                    )

    report(
        args,
        meta,
        legacy,
        fingerprint,
    )


if __name__ == "__main__":
    main()
