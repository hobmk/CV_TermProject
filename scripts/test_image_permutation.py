import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from scripts.compare_day_market_context import (
    ChartMarketDataset,
    DayViTWithMarket,
    build_peer_context,
    load_dev_split,
)


def predict(model, loader, device):
    model.eval()
    probabilities = []

    with torch.inference_mode():
        for images, context, _ in loader:
            logits = model(
                images.to(device),
                context.to(device),
            )

            p = torch.softmax(
                logits, dim=1
            )[:, 1]

            probabilities.append(
                p.cpu().numpy()
            )

    return np.concatenate(probabilities)


def compute_metrics(metadata, probabilities):
    table = metadata[
        [
            "decision_date",
            "label",
            "return_5d",
        ]
    ].copy()

    table["prob_up"] = probabilities

    auc = roc_auc_score(
        table["label"],
        table["prob_up"],
    )

    table["date_mean_prob"] = (
        table.groupby("decision_date")[
            "prob_up"
        ].transform("mean")
    )

    date_auc = roc_auc_score(
        table["label"],
        table["date_mean_prob"],
    )

    daily_auc = []
    daily_ic = []

    for _, group in table.groupby(
        "decision_date", sort=True
    ):
        if group["label"].nunique() == 2:
            daily_auc.append(
                roc_auc_score(
                    group["label"],
                    group["prob_up"],
                )
            )

        if (
            group["prob_up"].nunique() > 1
            and group["return_5d"].nunique() > 1
        ):
            daily_ic.append(
                group["prob_up"].corr(
                    group["return_5d"],
                    method="spearman",
                )
            )

    return {
        "pooled_auc": float(auc),
        "date_only_auc": float(date_auc),
        "mean_daily_auc": float(
            np.mean(daily_auc)
        ),
        "mean_rank_ic": float(
            np.mean(daily_ic)
        ),
        "prob_std": float(
            probabilities.std()
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
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    print(
        "[PROTOCOL] 2021 Dev only | "
        "No retraining"
    )

    train_rows, dev_rows = load_dev_split(
        Path("data/processed/samples.csv")
    )

    _, dev_context, _ = build_peer_context(
        train_rows,
        dev_rows,
        Path("data/raw"),
    )

    dataset = ChartMarketDataset(
        dev_rows,
        dev_context,
        Path("data/charts"),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    original_paths = list(dataset.paths)

    groups = list(
        dev_rows.groupby(
            "decision_date",
            sort=False,
        ).indices.values()
    )

    results = []

    for variant in args.variants:
        prefix = (
            f"day_market_{variant}_"
            f"seed{args.seed}_dev2021"
        )

        checkpoint_path = (
            Path("outputs/checkpoints")
            / f"{prefix}_best.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model = DayViTWithMarket(
            use_market=(variant == "market")
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        dataset.paths = list(original_paths)

        original_prob = predict(
            model, loader, device
        )

        original = compute_metrics(
            dev_rows,
            original_prob,
        )

        print(
            f"\n[{variant.upper()}] ORIGINAL"
        )

        print(
            f"AUROC={original['pooled_auc']:.5f} "
            f"DailyAUC="
            f"{original['mean_daily_auc']:.5f} "
            f"IC={original['mean_rank_ic']:+.5f}"
        )

        results.append({
            "variant": variant,
            "repeat": 0,
            "condition": "original",
            **original,
        })

        for repeat in range(
            1, args.repeats + 1
        ):
            rng = np.random.default_rng(
                args.shuffle_seed + repeat
            )

            source_indices = np.arange(
                len(dev_rows)
            )

            # Shuffle images only within each date.
            # Labels and context remain unchanged.
            for idx in groups:
                if len(idx) > 1:
                    source_indices[idx] = (
                        rng.permutation(idx)
                    )

            dataset.paths = [
                original_paths[j]
                for j in source_indices
            ]

            shuffled_prob = predict(
                model, loader, device
            )

            shuffled = compute_metrics(
                dev_rows,
                shuffled_prob,
            )

            print(
                f"[{variant}] "
                f"shuffle={repeat} "
                f"AUROC={shuffled['pooled_auc']:.5f} "
                f"DailyAUC="
                f"{shuffled['mean_daily_auc']:.5f} "
                f"IC={shuffled['mean_rank_ic']:+.5f} "
                f"delta_auc="
                f"{shuffled['pooled_auc'] - original['pooled_auc']:+.5f}",
                flush=True,
            )

            results.append({
                "variant": variant,
                "repeat": repeat,
                "condition": "shuffled",
                **shuffled,
            })

        dataset.paths = list(
            original_paths
        )

    results = pd.DataFrame(results)

    summary_rows = []

    for variant in args.variants:
        part = results[
            results["variant"] == variant
        ]

        orig = part[
            part["condition"] == "original"
        ].iloc[0]

        shuffled = part[
            part["condition"] == "shuffled"
        ]

        summary_rows.append({
            "variant": variant,
            "original_auc": orig["pooled_auc"],
            "shuffled_mean_auc": (
                shuffled["pooled_auc"].mean()
            ),
            "auc_drop": (
                orig["pooled_auc"]
                - shuffled["pooled_auc"].mean()
            ),
            "original_daily_auc": (
                orig["mean_daily_auc"]
            ),
            "shuffled_mean_daily_auc": (
                shuffled["mean_daily_auc"].mean()
            ),
            "original_ic": orig["mean_rank_ic"],
            "shuffled_mean_ic": (
                shuffled["mean_rank_ic"].mean()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    output_dir = Path(
        "outputs/metrics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir
        / "image_permutation_trials.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "image_permutation_summary.csv",
        index=False,
    )

    print("\n[PERMUTATION SUMMARY]")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        )
    )

    print(
        "\n[NOTE] Date, labels and market "
        "context remain unchanged."
    )

    print(
        "[NOTE] 2022-2025 untouched."
    )


if __name__ == "__main__":
    main()