

"""CE versus CE+pairwise ranking on 64x60 LocalCNN Day-Aligned ViT."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from scripts.compare_exp200_tokenizer import LocalCNNDayAlignedViT
from scripts.jkx_i20_reproduce import MemmapCharts, evaluate, set_seed
from scripts.run_exp200_factorial import loader, prepare_data, split, subset_scores


class WithReturns(Dataset):
    def __init__(self, base, returns):
        self.base = base
        self.returns = np.asarray(returns, dtype=np.float32)
        assert len(base) == len(self.returns)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base[i]
        return image, label, float(self.returns[i])


class DateBatches(Sampler):
    def __init__(self, dates, seed):
        positions = pd.Series(np.arange(len(dates)))
        self.groups = [
            v.to_numpy()
            for _, v in positions.groupby(
                np.asarray(dates), sort=True
            )
        ]
        self.seed = int(seed)
        self.epoch = 0

        if len(self.groups) < 2 or min(
            len(x) for x in self.groups
        ) < 2:
            raise ValueError("Date batches require at least 2 stocks")

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + 1009 * self.epoch
        )
        for k in rng.permutation(len(self.groups)):
            group = self.groups[int(k)].copy()
            rng.shuffle(group)
            yield group.tolist()


def pairwise_ranking_loss(logits, returns, min_diff=0.001):
    scores = logits[:, 1] - logits[:, 0]
    difference = returns[:, None] - returns[None, :]

    upper = torch.triu(
        torch.ones_like(difference, dtype=torch.bool),
        diagonal=1,
    )
    valid = upper & (difference.abs() > min_diff)

    if not bool(valid.any()):
        return scores.sum() * 0.0

    score_diff = scores[:, None] - scores[None, :]

    return F.softplus(
        -difference[valid].sign() * score_diff[valid]
    ).mean()


def validate_predictions(path, frame):
    predictions = pd.read_csv(
        path,
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

    if not predictions[ids].equals(frame[ids]):
        raise ValueError(f"Sample alignment mismatch: {path}")

    if not np.allclose(
        predictions.return_5d,
        frame.return_5d,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"Return alignment mismatch: {path}")

    return predictions


def train_one(a, device, meta, legacy, fingerprint, year, mode):
    tr, va, te, test_frame = split(
        meta, legacy, year, 2010, 200
    )
    val_frame = meta.iloc[va].reset_index(drop=True)

    tag = f"u200_start2010/test{year}/seed{a.seed}/{mode}"
    folder = a.out / tag
    folder.mkdir(parents=True, exist_ok=True)

    summary_path = folder / "summary.json"
    prediction_path = folder / "predictions.csv"

    config = dict(
        mode=mode,
        data_fingerprint=fingerprint,
        year=year,
        seed=a.seed,
        tokenizer="local_cnn",
        batch_policy="whole_decision_date",
        selection="validation_rank_ic_all200",
        epochs=a.epochs,
        patience=a.patience,
        lr=a.lr,
        weight_decay=a.weight_decay,
        lambda_rank=a.lambda_rank,
        min_diff=a.min_diff,
    )

    if summary_path.exists() or prediction_path.exists():
        if not (
            summary_path.exists() and prediction_path.exists()
        ):
            raise RuntimeError(f"Incomplete run: {folder}")

        saved_config = json.loads(
            summary_path.read_text(encoding="utf-8")
        )["config"]

        if saved_config != config:
            raise ValueError(f"Configuration mismatch: {folder}")

        print(f"[SKIP] {tag}", flush=True)
        return

    set_seed(a.seed)

    dataset = WithReturns(
        MemmapCharts(
            a.root / "images.npy",
            a.root / "meta.csv",
            tr,
        ),
        meta.iloc[tr].return_5d.to_numpy(float),
    )

    sampler = DateBatches(
        meta.iloc[tr].decision_date.to_numpy(),
        a.seed,
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = loader(
        a.root, va, 128, False, a.workers, a.seed
    )
    test_loader = loader(
        a.root, te, 128, False, a.workers, a.seed
    )

    model = LocalCNNDayAlignedViT().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )

    best_ic = -float("inf")
    best_epoch = 0
    stale = 0
    history = []

    print(
        f"[RUN] {tag} train={len(tr):,} "
        f"val={len(va):,} date_batches={len(sampler)} "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    for epoch in range(1, a.epochs + 1):
        sampler.epoch = epoch
        model.train()
        total_ce = 0.0
        total_rank = 0.0
        n = 0

        for images, labels, returns in train_loader:
            x = images.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            r = returns.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            ce = F.cross_entropy(logits, y)

            if mode == "ce_rank":
                rank = pairwise_ranking_loss(
                    logits, r, a.min_diff
                )
                loss = ce + a.lambda_rank * rank
            else:
                rank = ce.detach() * 0
                loss = ce

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            batch_size = len(y)
            total_ce += float(ce.item()) * batch_size
            total_rank += float(rank.item()) * batch_size
            n += batch_size

        val_stats, val_prob = evaluate(
            model, val_loader, device
        )

        ic = subset_scores(
            val_frame, val_prob, legacy
        )["all"]["rank_ic"]

        if not np.isfinite(ic):
            raise ValueError("Nonfinite validation Rank IC")

        if ic > best_ic + 1e-6:
            best_ic = float(ic)
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "state": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "val_rank_ic": best_ic,
                },
                folder / "best.pt",
            )
        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_ce": total_ce / n,
            "train_rank": total_rank / n,
            "val_loss": val_stats["loss"],
            "val_auc": val_stats["auc"],
            "val_rank_ic": ic,
        })

        pd.DataFrame(history).to_csv(
            folder / "history.csv", index=False
        )

        print(
            f"[EPOCH] {year} {mode} ep={epoch:02d} "
            f"trainCE={total_ce/n:.5f} "
            f"trainRank={total_rank/n:.5f} "
            f"valAUC={val_stats['auc']:.5f} "
            f"valIC={ic:+.5f} best={best_epoch} "
            f"patience={stale}/{a.patience}",
            flush=True,
        )

        if stale >= a.patience:
            break

    checkpoint = torch.load(
        folder / "best.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["state"])

    _, probs = evaluate(model, test_loader, device)
    scores = subset_scores(test_frame, probs, legacy)

    pred = test_frame[
        [
            "ticker", "group", "decision_date",
            "target_date", "return_5d", "label",
        ]
    ].copy()

    pred["prob_up"] = probs
    pred.to_csv(prediction_path, index=False)

    summary_path.write_text(
        json.dumps(
            {
                "config": config,
                "best_epoch": best_epoch,
                "best_val_rank_ic": best_ic,
                "groups": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    common = scores["common100"]

    print(
        f"[RESULT] {year} {mode} "
        f"AUC={common['auc']:.5f} "
        f"IC={common['rank_ic']:+.5f} "
        f"Top10={common['top10_excess_5d']:+.3%}",
        flush=True,
    )


def report(a, meta, legacy):
    rows = []

    for year in a.years:
        _, _, _, frame = split(
            meta, legacy, year, 2010, 200
        )

        for mode in ("ce", "ce_rank"):
            path = (
                a.out
                / f"u200_start2010/test{year}/seed{a.seed}"
                / mode
                / "predictions.csv"
            )

            if not path.exists():
                continue

            pred = validate_predictions(path, frame)

            stats = subset_scores(
                frame,
                pred.prob_up.to_numpy(float),
                legacy,
            )["common100"]

            rows.append({
                "year": year,
                "mode": mode,
                "auc": stats["auc"],
                "rank_ic": stats["rank_ic"],
                "top10_excess_5d": stats["top10_excess_5d"],
                "top_bottom_spread_5d":
                    stats["top10_bottom10_spread_5d"],
                "up_ratio": stats["up_ratio"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    a.out.mkdir(exist_ok=True, parents=True)

    df.to_csv(
        a.out / f"seed{a.seed}_comparison.csv",
        index=False,
    )

    print(
        "\n[COMMON100] Date-batched CE vs CE+Ranking",
        flush=True,
    )
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:+.5f}",
        ),
        flush=True,
    )

    wide = df.pivot(
        index="year",
        columns="mode",
        values=[
            "auc", "rank_ic",
            "top10_excess_5d",
            "top_bottom_spread_5d",
        ],
    )

    if set(wide.columns.get_level_values(1)) == {
        "ce", "ce_rank"
    }:
        names = [
            "auc", "rank_ic",
            "top10_excess_5d",
            "top_bottom_spread_5d",
        ]

        delta = pd.DataFrame({
            name: wide[(name, "ce_rank")]
            - wide[(name, "ce")]
            for name in names
        })

        print(
            "\n[MEAN DELTA] CE+Ranking minus date-batched CE",
            flush=True,
        )
        print(
            delta.mean().to_string(
                float_format=lambda x: f"{x:+.5f}"
            ),
            flush=True,
        )

    print(f"[SAVED] {a.out}", flush=True)
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
        "--out", type=Path,
        default=Path("outputs/exp200_rankloss_localcnn"),
    )
    p.add_argument(
        "--years", nargs="+", type=int,
        default=[2019, 2020, 2021],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.000253)
    p.add_argument(
        "--weight-decay", type=float,
        default=0.000325,
    )
    p.add_argument(
        "--lambda-rank", type=float,
        default=0.1,
    )
    p.add_argument(
        "--min-diff", type=float,
        default=0.001,
    )
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--stage",
        choices=["train", "report"],
        default="train",
    )

    a = p.parse_args()

    if any(
        y not in (2019, 2020, 2021)
        for y in a.years
    ):
        p.error("Only development years 2019-2021 allowed")

    if (
        a.lambda_rank <= 0
        or a.epochs <= 0
        or a.patience <= 0
        or a.min_diff < 0
    ):
        p.error("Invalid training parameters")

    meta, legacy, fingerprint = prepare_data(
        a.root, a.raw_root
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"[DEVICE] {device} | "
        f"[PLAN] {len(a.years)*2} runs",
        flush=True,
    )

    if a.stage == "train":
        for year in a.years:
            for mode in ("ce", "ce_rank"):
                train_one(
                    a, device, meta, legacy,
                    fingerprint, year, mode
                )

    report(a, meta, legacy)


if __name__ == "__main__":
    main()

