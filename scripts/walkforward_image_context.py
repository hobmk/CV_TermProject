
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from scripts.compare_cnn_temporal import (
    ChartOnlyDataset,
    ChartTemporalClassifier,
    score_metrics,
    set_seed,
)
from scripts.compare_stride_regularization import build_five_buckets
from scripts.compare_pv_market_stages import ContextOnlyDataset, ContextOnlyMLP
from scripts.compare_day_market_context import build_peer_context


VARIANTS = ("context_only", "cnn_tcn", "cnn_transformer")

POLICIES = {
    "context_only": "standard_shuffle",
    "cnn_tcn": "random5",
    "cnn_transformer": "stride5",
}


def load_year_split(path, year):
    metadata = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    required = {
        "ticker", "split", "window_start", "decision_date",
        "target_date", "label", "return_5d",
    }

    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    original_train = metadata.loc[
        metadata["split"] == "train"
    ].copy()

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    train = original_train.loc[
        (original_train["target_date"] < start)
        & (original_train["decision_date"] < start)
    ].copy()

    dev = original_train.loc[
        (original_train["window_start"] >= start)
        & (original_train["decision_date"] < end)
        & (original_train["target_date"] < end)
    ].copy()

    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)

    if train.empty or dev.empty:
        raise ValueError(
            f"{year}: Empty train/dev split"
        )

    for name, frame in (
        ("train", train),
        ("dev", dev),
    ):
        if frame.duplicated(
            ["ticker", "decision_date"]
        ).any():
            raise ValueError(
                f"{year}: Duplicate {name} sample"
            )

        if not frame["label"].isin([0, 1]).all():
            raise ValueError(
                f"{year}: Invalid {name} label"
            )

        if frame["label"].nunique() != 2:
            raise ValueError(
                f"{year}: Only one label in {name}"
            )

    if (
        train["target_date"].max()
        >= dev["window_start"].min()
    ):
        raise RuntimeError(
            "Train label horizon overlaps "
            "dev input period"
        )

    print(
        f"[SPLIT {year}] "
        f"Train={len(train):,} "
        f"({train['label'].mean():.2%} Up) "
        f"Dev={len(dev):,} "
        f"({dev['label'].mean():.2%} Up)",
        flush=True,
    )

    print(
        f"[DATES] train targets through "
        f"{train['target_date'].max()} | "
        f"dev inputs from "
        f"{dev['window_start'].min()} | "
        f"dev targets through "
        f"{dev['target_date'].max()}",
        flush=True,
    )

    return train, dev


def make_loader(
    dataset,
    batch_size,
    shuffle=False,
    seed=None,
    workers=0,
    cuda=False,
):
    generator = (
        torch.Generator().manual_seed(seed)
        if seed is not None
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=cuda,
    )


def forward_batch(model, batch, variant, device):
    if variant == "context_only":
        _, context, labels = batch

        logits = model(
            None,
            context.to(
                device,
                non_blocking=True,
            ),
        )

    else:
        images, labels = batch

        logits = model(
            images.to(
                device,
                non_blocking=True,
            )
        )

    return logits, labels


def evaluate_model(
    model,
    loader,
    rows,
    variant,
    device,
    with_probs=False,
):
    model.eval()

    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            logits, labels = forward_batch(
                model,
                batch,
                variant,
                device,
            )

            total_loss += F.cross_entropy(
                logits,
                labels.to(
                    device,
                    non_blocking=True,
                ),
                reduction="sum",
            ).item()

            all_labels.append(
                labels.numpy()
            )

            all_probs.append(
                torch.softmax(
                    logits.float(),
                    dim=1,
                )[:, 1].cpu().numpy()
            )

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probs)

    if not np.array_equal(
        labels,
        rows["label"].to_numpy(
            np.int64
        ),
    ):
        raise RuntimeError(
            "Dev label order mismatch"
        )

    metrics = score_metrics(
        rows,
        probabilities,
    )

    metrics["loss"] = float(
        total_loss / len(labels)
    )

    if with_probs:
        return metrics, probabilities

    return metrics


def train_one(
    year,
    variant,
    args,
    train_ds,
    dev_ds,
    train_rows,
    dev_rows,
    device,
):
    set_seed(args.seed)

    policy = POLICIES[variant]

    model = (
        ContextOnlyMLP()
        if variant == "context_only"
        else ChartTemporalClassifier(variant)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dev_loader = make_loader(
        dev_ds,
        args.batch_size,
        workers=args.num_workers,
        cuda=(device.type == "cuda"),
    )

    if variant == "context_only":
        context_train_loader = make_loader(
            train_ds,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            workers=args.num_workers,
            cuda=(device.type == "cuda"),
        )

    prefix = (
        f"wf_dev{year}_{variant}_"
        f"{policy}_seed{args.seed}"
    )

    ckpt_dir = Path(
        "outputs/checkpoints"
    )

    metric_dir = Path(
        "outputs/metrics"
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt = (
        ckpt_dir
        / f"{prefix}_best.pt"
    )

    history_path = (
        metric_dir
        / f"{prefix}_history.csv"
    )

    prediction_path = (
        metric_dir
        / f"{prefix}_dev_predictions.csv"
    )

    best_auc = -1.0
    best_loss = float("inf")
    best_cycle = 0
    stale = 0

    history = []
    started = time.perf_counter()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[RUN] dev={year} "
        f"variant={variant} "
        f"policy={policy} "
        f"params={params:,}",
        flush=True,
    )

    for cycle in range(
        1,
        args.cycles + 1,
    ):
        model.train()

        train_labels = []
        train_probs = []

        train_loss_sum = 0.0
        seen = 0

        if variant == "context_only":
            loaders = [
                context_train_loader
            ]

        else:
            buckets = build_five_buckets(
                train_rows,
                policy,
                args.seed,
                cycle,
            )

            if cycle == 1:
                print(
                    "[BUCKETS]",
                    [len(b) for b in buckets],
                    flush=True,
                )

            loaders = [
                make_loader(
                    Subset(
                        train_ds,
                        bucket,
                    ),
                    args.batch_size,
                    shuffle=True,
                    seed=(
                        args.seed
                        + 5003 * cycle
                        + phase
                    ),
                    workers=args.num_workers,
                    cuda=(device.type == "cuda"),
                )
                for phase, bucket
                in enumerate(buckets)
            ]

        for loader in loaders:
            for batch in loader:
                logits, labels = forward_batch(
                    model,
                    batch,
                    variant,
                    device,
                )

                targets = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss = F.cross_entropy(
                    logits,
                    targets,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                n = len(labels)

                seen += n

                train_loss_sum += (
                    float(loss.item()) * n
                )

                train_labels.append(
                    labels.numpy()
                )

                train_probs.append(
                    torch.softmax(
                        logits.detach().float(),
                        dim=1,
                    )[:, 1].cpu().numpy()
                )

        if seen != len(train_ds):
            raise AssertionError(
                f"Trained on {seen} "
                f"of {len(train_ds)} samples"
            )

        train_auc = float(
            roc_auc_score(
                np.concatenate(
                    train_labels
                ),
                np.concatenate(
                    train_probs
                ),
            )
        )

        dev = evaluate_model(
            model,
            dev_loader,
            dev_rows,
            variant,
            device,
        )

        improved = (
            dev["auc"] > best_auc + 1e-5
            or (
                abs(
                    dev["auc"] - best_auc
                ) <= 1e-5
                and dev["loss"] < best_loss
            )
        )

        if improved:
            best_auc = dev["auc"]
            best_loss = dev["loss"]
            best_cycle = cycle
            stale = 0

            torch.save(
                {
                    "model_name": (
                        "walkforward_cnn_context"
                    ),
                    "variant": variant,
                    "policy": policy,
                    "model_state": (
                        model.state_dict()
                    ),
                    "year": year,
                    "seed": args.seed,
                    "best_cycle": cycle,
                    "dev_metrics": dev,
                    "train_end": (
                        f"{year - 1}-12-31"
                    ),
                },
                ckpt,
            )

        else:
            stale += 1

        history.append({
            "cycle": cycle,
            "train_auc_online": (
                train_auc
            ),
            "train_loss_online": (
                train_loss_sum / seen
            ),
            **{
                f"dev_{key}": val
                for key, val in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        print(
            f"[{year} {variant}] "
            f"cycle={cycle:02d}/{args.cycles} "
            f"train_auc={train_auc:.4f} "
            f"dev_auc={dev['auc']:.4f} "
            f"daily_auc={dev['daily_auc']:.4f} "
            f"IC={dev['rank_ic']:+.4f} "
            f"bal={dev['balanced_accuracy']:.4f} "
            f"up={dev['up_ratio']:.3f} "
            f"dev_loss={dev['loss']:.4f} "
            f"best={best_cycle} "
            f"patience={stale}/{args.patience}",
            flush=True,
        )

        min_cycles = (
            1
            if variant == "context_only"
            else args.min_cycles
        )

        if (
            cycle >= min_cycles
            and stale >= args.patience
        ):
            print(
                f"[EARLY STOP] "
                f"{year} {variant}",
                flush=True,
            )
            break

    checkpoint = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    best, probs = evaluate_model(
        model,
        dev_loader,
        dev_rows,
        variant,
        device,
        with_probs=True,
    )

    prediction = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    prediction["prob_up"] = probs

    prediction.to_csv(
        prediction_path,
        index=False,
    )

    return {
        "dev_year": year,
        "variant": variant,
        "policy": policy,
        "params": params,
        "train_n": len(train_rows),
        "dev_n": len(dev_rows),
        "best_cycle": best_cycle,
        **{
            f"dev_{key}": val
            for key, val in best.items()
        },
        "seconds": round(
            time.perf_counter() - started,
            1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020],
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-cycles",
        type=int,
        default=8,
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

    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("data/charts"),
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )

    args = parser.parse_args()

    if (
        not args.years
        or any(
            y not in (2019, 2020)
            for y in args.years
        )
        or args.cycles < 1
        or not (
            1 <= args.min_cycles <= args.cycles
        )
        or args.patience < 1
        or args.batch_size < 1
        or args.lr <= 0
        or args.weight_decay < 0
        or args.num_workers < 0
    ):
        parser.error(
            "Invalid experiment configuration"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] Original train metadata "
        "only; 2022-2025 untouched"
    )

    print(
        "[POLICY] TCN=random5; "
        "Transformer=stride5; "
        "Context=standard_shuffle"
    )

    print(
        "[CHECK] 2021 existing runs "
        "will not be retrained"
    )

    path = Path(
        "outputs/metrics/"
        "walkforward_cnn_context_2019_2020.csv"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for year in args.years:
        train_rows, dev_rows = load_year_split(
            Path("data/processed/samples.csv"),
            year,
        )

        need_images = any(
            v != "context_only"
            for v in args.variants
        )

        need_context = (
            "context_only" in args.variants
        )

        if need_images:
            image_train = ChartOnlyDataset(
                train_rows,
                args.charts_dir,
            )

            image_dev = ChartOnlyDataset(
                dev_rows,
                args.charts_dir,
            )

        if need_context:
            x_train, x_dev, _ = (
                build_peer_context(
                    train_rows,
                    dev_rows,
                    args.raw_dir,
                )
            )

            context_train = ContextOnlyDataset(
                train_rows,
                x_train,
            )

            context_dev = ContextOnlyDataset(
                dev_rows,
                x_dev,
            )

        for variant in args.variants:
            if variant == "context_only":
                train_ds = context_train
                dev_ds = context_dev

            else:
                train_ds = image_train
                dev_ds = image_dev

            row = train_one(
                year,
                variant,
                args,
                train_ds,
                dev_ds,
                train_rows,
                dev_rows,
                device,
            )

            results.append(row)

            pd.DataFrame(results).to_csv(
                path,
                index=False,
            )

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(results).to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print(f"[SAVED] {path}")

    print(
        "[NOTE] 2021 results can be "
        "compared from previous runs."
    )

    print(
        "[NOTE] 2022-2025 untouched; "
        "years 2019/2020 are exploratory."
    )


if __name__ == "__main__":
    main()
