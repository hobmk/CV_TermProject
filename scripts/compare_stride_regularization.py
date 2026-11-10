
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from scripts.compare_cnn_temporal import (
    ChartOnlyDataset,
    ChartTemporalClassifier,
    evaluate,
    set_seed,
)

from scripts.compare_day_market_context import (
    load_dev_split,
)


VARIANTS = (
    "cnn_tcn",
    "cnn_transformer",
)

POLICIES = (
    "random5",
    "stride5",
    "stride5_reg",
)


def build_five_buckets(
    rows, policy, seed, cycle
):
    """
    Divide samples into 5 disjoint groups.

    Each sample appears exactly once
    during each full training cycle.
    """

    buckets = [[] for _ in range(5)]

    rng = np.random.default_rng(
        seed + 1009 * cycle
    )

    for _, group in rows.groupby(
        "ticker", sort=True
    ):
        indices = group.sort_values(
            "decision_date"
        ).index.to_numpy()

        if policy == "random5":
            indices = rng.permutation(
                indices
            )

        for phase in range(5):
            buckets[phase].extend(
                indices[phase::5].tolist()
            )

    combined = np.concatenate([
        np.asarray(b)
        for b in buckets
    ])

    if (
        len(combined) != len(rows)
        or not np.array_equal(
            np.sort(combined),
            np.arange(len(rows))
        )
    ):
        raise AssertionError(
            "Buckets must cover every "
            "training sample exactly once"
        )

    return buckets


def apply_regularization(model, policy):
    if policy != "stride5_reg":
        return 0.0, 0.01

    for module in model.modules():
        if isinstance(
            module, nn.Dropout
        ):
            module.p = 0.30

    # Label smoothing, weight decay
    return 0.05, 0.05


def train_one(
    variant,
    policy,
    args,
    train_data,
    dev_loader,
    train_rows,
    dev_rows,
    device,
):
    set_seed(args.seed)

    model = ChartTemporalClassifier(
        variant
    ).to(device)

    smoothing, weight_decay = (
        apply_regularization(
            model, policy
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=weight_decay,
    )

    label = (
        f"stride_reg_{variant}_{policy}_"
        f"seed{args.seed}_dev2021"
    )

    metrics_dir = Path(
        "outputs/metrics"
    )

    weights_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir.mkdir(
        parents=True, exist_ok=True
    )

    weights_dir.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint_path = (
        weights_dir / f"{label}_best.pt"
    )

    history_path = (
        metrics_dir / f"{label}_history.csv"
    )

    prediction_path = (
        metrics_dir
        / f"{label}_dev_predictions.csv"
    )

    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[RUN] {variant} | {policy} "
        f"| params={param_count:,} "
        f"| dropout="
        f"{'0.30' if policy == 'stride5_reg' else 'default'} "
        f"| label_smoothing={smoothing:.2f} "
        f"| wd={weight_decay:.2f}",
        flush=True,
    )

    best_auc = -1.0
    best_loss = float("inf")
    best_cycle = 0
    stale = 0

    history = []
    started = time.perf_counter()

    for cycle in range(
        1, args.cycles + 1
    ):
        buckets = build_five_buckets(
            train_rows,
            policy,
            args.seed,
            cycle,
        )

        if cycle == 1:
            sizes = [
                len(b) for b in buckets
            ]

            print(
                f"[BUCKETS] "
                f"sizes={sizes} "
                f"| total={sum(sizes):,} "
                "| each sample once",
                flush=True,
            )

        model.train()

        train_labels = []
        train_probs = []

        train_loss_sum = 0.0
        seen = 0

        # Five micro-epochs form one cycle
        for phase, bucket in enumerate(
            buckets
        ):
            loader = DataLoader(
                Subset(
                    train_data,
                    bucket,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                generator=(
                    torch.Generator()
                    .manual_seed(
                        args.seed
                        + 5003 * cycle
                        + phase
                    )
                ),
                num_workers=args.num_workers,
                pin_memory=(
                    device.type == "cuda"
                ),
            )

            for images, labels in loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                targets = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(images)

                loss = F.cross_entropy(
                    logits,
                    targets,
                    label_smoothing=smoothing,
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                batch_size = len(labels)

                seen += batch_size

                train_loss_sum += (
                    loss.item() * batch_size
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

        if seen != len(train_data):
            raise AssertionError(
                f"Seen {seen} of "
                f"{len(train_data)} samples"
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

        dev = evaluate(
            model,
            dev_loader,
            dev_rows,
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
                        "cnn_temporal_stride_experiment"
                    ),
                    "variant": variant,
                    "policy": policy,
                    "model_state": (
                        model.state_dict()
                    ),
                    "cycle": cycle,
                    "seed": args.seed,
                    "dropout": (
                        0.30
                        if policy == "stride5_reg"
                        else "original"
                    ),
                    "weight_decay": weight_decay,
                    "label_smoothing": smoothing,
                    "dev_metrics": dev,
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        row = {
            "variant": variant,
            "policy": policy,
            "cycle": cycle,
            "train_auc_online": train_auc,
            "train_loss_online": float(
                train_loss_sum / seen
            ),
            **{
                f"dev_{key}": value
                for key, value in dev.items()
            },
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        elapsed = (
            time.perf_counter() - started
        )

        print(
            f"[{variant} {policy}] "
            f"cycle={cycle:02d}/{args.cycles} "
            f"train_auc={train_auc:.4f} "
            f"dev_auc={dev['auc']:.4f} "
            f"daily_auc={dev['daily_auc']:.4f} "
            f"IC={dev['rank_ic']:+.4f} "
            f"bal={dev['balanced_accuracy']:.4f} "
            f"up={dev['up_ratio']:.3f} "
            f"dev_loss={dev['loss']:.4f} "
            f"best={best_cycle} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if (
            cycle >= args.min_cycles
            and stale >= args.patience
        ):
            print(
                f"[EARLY STOP] "
                f"{variant} {policy}",
                flush=True,
            )
            break

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    best, probs = evaluate(
        model,
        dev_loader,
        dev_rows,
        device,
        return_predictions=True,
    )

    predictions = dev_rows[
        [
            "ticker",
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    predictions["prob_up"] = probs

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    return {
        "variant": variant,
        "policy": policy,
        "params": param_count,
        "best_cycle": best_cycle,
        "train_n": len(train_data),
        "dev_n": len(dev_rows),

        **{
            f"dev_{key}": value
            for key, value in best.items()
        },

        "seconds": round(
            time.perf_counter() - started,
            1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
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

    args = parser.parse_args()

    if (
        args.cycles < 1
        or args.min_cycles < 1
        or args.min_cycles > args.cycles
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

    print(
        "[PROTOCOL] Train 2015-2020 "
        "| Dev 2021 "
        "| 2022-2025 untouched"
    )

    print(
        "[DESIGN] Five disjoint "
        "micro-epochs = one cycle"
    )

    train_rows, dev_rows = (
        load_dev_split(
            Path(
                "data/processed/samples.csv"
            )
        )
    )

    train_data = ChartOnlyDataset(
        train_rows,
        args.charts_dir,
    )

    dev_data = ChartOnlyDataset(
        dev_rows,
        args.charts_dir,
    )

    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    output = Path(
        "outputs/metrics/"
        "stride_regularization_dev2021_summary.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for variant in args.variants:
        for policy in args.policies:
            result = train_one(
                variant,
                policy,
                args,
                train_data,
                dev_loader,
                train_rows,
                dev_rows,
                device,
            )

            results.append(result)

            pd.DataFrame(results).to_csv(
                output,
                index=False,
            )

    print("\n[SUMMARY]")

    print(
        pd.DataFrame(results).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print(f"[SAVED] {output}")

    print(
        "[REFERENCE] Old full-shuffle "
        "seed42: cnn_tcn AUROC=0.51224; "
        "cnn_transformer AUROC=0.51801"
    )

    print(
        "[NOTE] All training samples "
        "are used once per cycle; "
        "no final test data used."
    )


if __name__ == "__main__":
    main()
