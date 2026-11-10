
"""Day-Aligned ViT vs existing JKX CNN, 2010-2021.

Uses existing Exp200 grayscale I20 images.
No re-downloads, no re-rendering, no 2022+ data.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.jkx_i20_reproduce import (
    evaluate,
    set_seed,
)

from scripts.run_exp200_factorial import (
    prepare_data,
    split,
    loader,
    subset_scores,
)


class DayAlignedViT(nn.Module):
    """One token per trading day (64 rows x 3 pixels)."""

    def __init__(
        self,
        dim=128,
        depth=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        if dim % heads:
            raise ValueError(
                "dim must be divisible by heads"
            )

        self.patch_embed = nn.Sequential(
            nn.Linear(64 * 3, dim),
            nn.LayerNorm(dim),
        )

        self.cls = nn.Parameter(
            torch.zeros(1, 1, dim)
        )

        self.pos = nn.Parameter(
            torch.zeros(1, 21, dim)
        )

        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            block,
            num_layers=depth,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

        nn.init.trunc_normal_(
            self.cls, std=0.02
        )

        nn.init.trunc_normal_(
            self.pos, std=0.02
        )

        nn.init.trunc_normal_(
            self.patch_embed[0].weight,
            std=0.02,
        )

        nn.init.zeros_(
            self.patch_embed[0].bias
        )

        nn.init.trunc_normal_(
            self.head[-1].weight,
            std=0.02,
        )

        nn.init.zeros_(
            self.head[-1].bias
        )

    def forward(self, image):
        if (
            image.ndim != 4
            or tuple(image.shape[1:])
            != (1, 64, 60)
        ):
            raise ValueError(
                f"Expected (B,1,64,60), "
                f"got {tuple(image.shape)}"
            )

        b = image.shape[0]

        # Width 60 = 20 days x 3 columns.
        # Preserve left-to-right trading-day order.
        tokens = image.reshape(
            b, 1, 64, 20, 3
        )

        tokens = (
            tokens.permute(0, 3, 1, 2, 4)
            .reshape(b, 20, 192)
        )

        tokens = self.patch_embed(tokens)

        cls = self.cls.expand(
            b, -1, -1
        )

        tokens = torch.cat(
            (cls, tokens),
            dim=1,
        ) + self.pos

        encoded = self.encoder(tokens)

        return self.head(
            self.norm(encoded[:, 0])
        )


def baseline_file(
    base,
    universe,
    start,
    year,
    seed,
):
    return (
        base
        / f"u{universe}_start{start}"
        / f"test{year}"
        / f"seed{seed}"
    )


def check_baseline(
    folder,
    frame,
    fingerprint,
    universe,
    start,
    year,
    seed,
):
    info_path = folder / "summary.json"
    pred_path = folder / "predictions.csv"

    if (
        not info_path.exists()
        or not pred_path.exists()
    ):
        raise FileNotFoundError(
            f"JKX baseline missing: {folder}"
        )

    info = json.loads(
        info_path.read_text(
            encoding="utf-8"
        )
    )

    config = info.get("config", {})

    required = {
        "data_fingerprint": fingerprint,
        "universe": universe,
        "start_year": start,
        "test_year": year,
        "seed": seed,
    }

    for key, wanted in required.items():
        if config.get(key) != wanted:
            raise ValueError(
                f"JKX baseline mismatch: "
                f"{folder} {key}"
            )

    old = pd.read_csv(
        pred_path,
        dtype={
            "ticker": str,
            "group": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    ids = [
        "ticker",
        "group",
        "decision_date",
        "target_date",
        "label",
    ]

    if not old[ids].equals(frame[ids]):
        raise ValueError(
            f"JKX sample alignment mismatch: "
            f"{folder}"
        )

    if not np.allclose(
        old.return_5d,
        frame.return_5d,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "JKX return mismatch"
        )

    return old


def run_one(
    a,
    device,
    meta,
    legacy,
    fingerprint,
    year,
    universe,
):
    tag = (
        f"u{universe}_start{a.start_year}"
        f"/test{year}/seed{a.seed}"
    )

    out = a.out / tag

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = out / "summary.json"
    pred_file = out / "predictions.csv"

    config = {
        "data_fingerprint": fingerprint,
        "model": "day_aligned_vit_i20_grayscale",
        "universe": universe,
        "start_year": a.start_year,
        "test_year": year,
        "validation_year": year - 1,
        "seed": a.seed,
        "depth": a.depth,
        "dim": a.dim,
        "heads": a.heads,
        "dropout": a.dropout,
        "lr": a.lr,
        "weight_decay": a.weight_decay,
        "epochs": a.epochs,
        "patience": a.patience,
        "batch_size": a.batch_size,
    }

    tr, va, te, frame = split(
        meta,
        legacy,
        year,
        a.start_year,
        universe,
    )

    check_baseline(
        baseline_file(
            a.baseline,
            universe,
            a.start_year,
            year,
            a.seed,
        ),
        frame,
        fingerprint,
        universe,
        a.start_year,
        year,
        a.seed,
    )

    if (
        summary_file.exists()
        and pred_file.exists()
        and not a.force
    ):
        old = json.loads(
            summary_file.read_text(
                encoding="utf-8"
            )
        )

        if old.get("config") != config:
            raise ValueError(
                f"Configuration mismatch: {out}"
            )

        print(
            f"[SKIP] {tag}",
            flush=True,
        )

        return

    set_seed(a.seed)

    train_loader = loader(
        a.root,
        tr,
        a.batch_size,
        True,
        a.workers,
        a.seed,
    )

    val_loader = loader(
        a.root,
        va,
        a.batch_size,
        False,
        a.workers,
        a.seed,
    )

    test_loader = loader(
        a.root,
        te,
        a.batch_size,
        False,
        a.workers,
        a.seed,
    )

    model = DayAlignedViT(
        a.dim,
        a.depth,
        a.heads,
        a.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
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
        f"params={parameters:,}",
        flush=True,
    )

    for epoch in range(
        1, a.epochs + 1
    ):
        model.train()

        total_loss = 0.0
        n = 0

        for step, (
            images,
            labels,
        ) in enumerate(
            train_loader,
            1,
        ):
            x = images.to(
                device,
                non_blocking=True,
            )

            y = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits,
                y,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += (
                float(loss.item())
                * len(y)
            )

            n += len(y)

            if (
                a.log_steps
                and step % a.log_steps == 0
            ):
                print(
                    f"[BATCH] {tag} "
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

        if val["loss"] < best_loss - 1e-6:
            best_loss = val["loss"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "config": config,
                    "best_epoch": epoch,
                    "val_metrics": val,
                },
                out / "best.pt",
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
            out / "history.csv",
            index=False,
        )

        print(
            f"[EPOCH] {tag} "
            f"ep={epoch:02d} "
            f"train={total_loss/n:.5f} "
            f"val={val['loss']:.5f} "
            f"val_auc={val['auc']:.4f} "
            f"best={best_epoch} "
            f"patience={stale}/{a.patience} "
            f"elapsed={time.perf_counter()-started:.0f}s",
            flush=True,
        )

        if stale >= a.patience:
            print(
                f"[EARLY STOP] {tag}",
                flush=True,
            )
            break

    saved = torch.load(
        out / "best.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        saved["model_state"]
    )

    _, probs = evaluate(
        model,
        test_loader,
        device,
    )

    predictions = frame[
        [
            "ticker",
            "group",
            "decision_date",
            "target_date",
            "return_5d",
            "label",
        ]
    ].copy()

    predictions["prob_up"] = probs

    predictions.to_csv(
        pred_file,
        index=False,
    )

    groups = subset_scores(
        frame,
        probs,
        legacy,
    )

    summary_file.write_text(
        json.dumps(
            {
                "config": config,
                "params": parameters,
                "best_epoch": best_epoch,
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    2,
                ),
                "groups": groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    common = groups["common100"]

    print(
        f"[RESULT] {tag} "
        f"common100_auc={common['auc']:.4f} "
        f"IC={common['rank_ic']:+.4f} "
        f"top10={common['top10_excess_5d']:+.3%}",
        flush=True,
    )


def report(
    a,
    meta,
    legacy,
    fingerprint,
):
    rows = []

    for year in a.years:
        for universe in a.universes:
            tag = (
                f"u{universe}_start{a.start_year}"
                f"/test{year}/seed{a.seed}"
            )

            folder = a.out / tag

            if not (
                folder / "predictions.csv"
            ).exists():
                print(
                    f"[INCOMPLETE] {tag}"
                )
                continue

            info = json.loads(
                (
                    folder / "summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            _, _, _, frame = split(
                meta,
                legacy,
                year,
                a.start_year,
                universe,
            )

            if (
                info.get("config", {}).get(
                    "data_fingerprint"
                )
                != fingerprint
            ):
                raise ValueError(
                    f"ViT fingerprint mismatch: {tag}"
                )

            baseline = check_baseline(
                baseline_file(
                    a.baseline,
                    universe,
                    a.start_year,
                    year,
                    a.seed,
                ),
                frame,
                fingerprint,
                universe,
                a.start_year,
                year,
                a.seed,
            )

            vit = pd.read_csv(
                folder / "predictions.csv",
                dtype={
                    "ticker": str,
                    "group": str,
                    "decision_date": str,
                    "target_date": str,
                },
            )

            ids = [
                "ticker",
                "group",
                "decision_date",
                "target_date",
                "label",
            ]

            if not vit[ids].equals(
                baseline[ids]
            ):
                raise ValueError(
                    "ViT/JKX sample mismatch"
                )

            if not np.allclose(
                vit.return_5d,
                baseline.return_5d,
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    "ViT/JKX return mismatch"
                )

            for name, pred in (
                ("JKX_CNN", baseline),
                ("DayAligned_ViT", vit),
            ):
                scores = pred.prob_up.to_numpy(
                    float
                )

                results = subset_scores(
                    frame,
                    scores,
                    legacy,
                )

                for group in (
                    "common100",
                    "all",
                ):
                    s = results[group]

                    rows.append({
                        "year": year,
                        "train_universe": universe,
                        "train_start": a.start_year,
                        "seed": a.seed,
                        "model": name,
                        "evaluation": group,
                        "auc": s["auc"],
                        "rank_ic": s["rank_ic"],
                        "top10_excess_5d": (
                            s["top10_excess_5d"]
                        ),
                        "top_bottom_spread_5d": (
                            s[
                                "top10_bottom10_spread_5d"
                            ]
                        ),
                        "daily_auc": (
                            s["daily_auc"]
                        ),
                        "balanced_acc": (
                            s["balanced_acc"]
                        ),
                        "up_ratio": (
                            s["up_ratio"]
                        ),
                    })

    if not rows:
        print(
            "[REPORT] No finished comparisons"
        )
        return

    table = pd.DataFrame(rows)

    a.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        a.out / "comparison.csv",
        index=False,
    )

    common = table[
        table.evaluation == "common100"
    ].copy()

    print(
        "\n[COMMON100] "
        "Identical stocks, dates, returns",
        flush=True,
    )

    print(
        common[
            [
                "year",
                "train_universe",
                "model",
                "auc",
                "rank_ic",
                "top10_excess_5d",
                "top_bottom_spread_5d",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    pair = common.pivot(
        index=[
            "year",
            "train_universe",
        ],
        columns="model",
        values=[
            "auc",
            "rank_ic",
            "top10_excess_5d",
            "top_bottom_spread_5d",
        ],
    )

    model_names = set(
        pair.columns.get_level_values(1)
    )

    if {
        "JKX_CNN",
        "DayAligned_ViT",
    }.issubset(model_names):

        delta = pd.DataFrame(
            index=pair.index
        )

        metrics = [
            "auc",
            "rank_ic",
            "top10_excess_5d",
            "top_bottom_spread_5d",
        ]

        for metric in metrics:
            delta[
                "delta_" + metric
            ] = (
                pair[
                    (metric, "DayAligned_ViT")
                ]
                - pair[
                    (metric, "JKX_CNN")
                ]
            )

        delta = delta.reset_index()

        delta.to_csv(
            a.out / "paired_deltas.csv",
            index=False,
        )

        print(
            "\n[DELTA] ViT - JKX CNN",
            flush=True,
        )

        print(
            delta.to_string(
                index=False,
                float_format=lambda x: f"{x:+.5f}",
            ),
            flush=True,
        )

        print(
            "\n[MEAN DELTA across 2019-2021]",
            flush=True,
        )

        print(
            delta.groupby(
                "train_universe"
            ).mean(
                numeric_only=True
            ).to_string(
                float_format=lambda x: f"{x:+.5f}",
            ),
            flush=True,
        )

    print(
        f"[SAVED] "
        f"{a.out / 'comparison.csv'}",
        flush=True,
    )

    print(
        "[HOLDOUT] 2022-2025 untouched",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/exp200_jkx"),
    )

    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/exp200"),
    )

    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "outputs/exp200_factorial"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/exp200_day_vit"
        ),
    )

    p.add_argument(
        "--stage",
        choices=["train", "report"],
        default="train",
    )

    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021],
    )

    p.add_argument(
        "--universes",
        nargs="+",
        type=int,
        choices=[100, 200],
        default=[100, 200],
    )

    p.add_argument(
        "--start-year",
        type=int,
        choices=[2010, 2015],
        default=2010,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--lr",
        type=float,
        default=0.000253,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.000325,
    )

    p.add_argument(
        "--dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    p.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--log-steps",
        type=int,
        default=300,
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    a = p.parse_args()

    if (
        a.epochs < 1
        or a.patience < 1
        or a.batch_size < 1
        or a.lr <= 0
        or a.workers < 0
        or a.dim < 1
        or a.depth < 1
        or a.heads < 1
        or a.dim % a.heads
        or not (0 <= a.dropout < 1)
    ):
        p.error(
            "Invalid training arguments"
        )

    meta, legacy, fingerprint = (
        prepare_data(
            a.root,
            a.raw_root,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DEVICE] {device} "
        f"| [PLAN] "
        f"{len(a.years)*len(a.universes)} "
        f"ViT runs",
        flush=True,
    )

    try:
        if a.stage == "train":
            for year in a.years:
                for universe in a.universes:
                    run_one(
                        a,
                        device,
                        meta,
                        legacy,
                        fingerprint,
                        year,
                        universe,
                    )

    finally:
        report(
            a,
            meta,
            legacy,
            fingerprint,
        )


if __name__ == "__main__":
    main()
