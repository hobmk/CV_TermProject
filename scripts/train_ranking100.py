
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import expit
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
)
from torchvision.transforms.functional import to_tensor

from scripts.compare_cnn_temporal import (
    ChartTemporalClassifier,
    set_seed,
)


MODES = ("ce", "ranking", "combined")


def read_split(root):
    path = root / "processed" / "samples.csv"

    df = pd.read_csv(
        path,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
            "target_date": str,
        },
    )

    needed = {
        "ticker",
        "window_start",
        "decision_date",
        "target_date",
        "label",
        "return_5d",
    }

    missing = needed - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(
        ["ticker", "decision_date"]
    ).any():
        raise ValueError("Duplicate ticker/date")

    if not df.label.isin([0, 1]).all():
        raise ValueError("Invalid labels")

    if not np.isfinite(
        df.return_5d.to_numpy(float)
    ).all():
        raise ValueError("Invalid future returns")

    correct_labels = (
        df.return_5d.to_numpy() > 0
    ).astype(int)

    if not np.array_equal(
        df.label.to_numpy(),
        correct_labels,
    ):
        raise ValueError(
            "Labels do not match returns"
        )

    if not df.decision_date.between(
        "2015-01-01",
        "2021-12-31",
    ).all():
        raise ValueError(
            "Unexpected dates outside 2015-2021"
        )

    train = df.loc[
        (df.decision_date < "2021-01-01")
        & (df.target_date < "2021-01-01")
    ].sort_values(
        ["decision_date", "ticker"]
    ).reset_index(drop=True)

    dev = df.loc[
        (df.window_start >= "2021-01-01")
        & (df.decision_date < "2022-01-01")
        & (df.target_date < "2022-01-01")
    ].sort_values(
        ["decision_date", "ticker"]
    ).reset_index(drop=True)

    if train.empty or dev.empty:
        raise ValueError("Empty Train or Dev")

    if (
        train.target_date.max()
        >= dev.window_start.min()
    ):
        raise ValueError(
            "Train target horizon overlaps "
            "Dev input period"
        )

    if (
        train.ticker.nunique(),
        dev.ticker.nunique(),
    ) != (100, 100):
        raise ValueError(
            "Expected 100 tickers in both splits"
        )

    print(
        f"[TRAIN] N={len(train):,} "
        f"days={train.decision_date.nunique()} "
        f"Up={train.label.mean():.2%}"
    )

    print(
        f"[DEV] N={len(dev):,} "
        f"days={dev.decision_date.nunique()} "
        f"Up={dev.label.mean():.2%}"
    )

    print(
        f"[PURGE] Train target <= "
        f"{train.target_date.max()} "
        f"| Dev image begins "
        f"{dev.window_start.min()}"
    )

    return train, dev


class ChartReturnDataset(Dataset):
    def __init__(self, rows, root):
        self.rows = rows.reset_index(
            drop=True
        )

        self.y = self.rows.label.to_numpy(
            dtype=np.int64
        )

        self.r = self.rows.return_5d.to_numpy(
            dtype=np.float32
        )

        base = root / "charts" / "train"

        self.paths = [
            base
            / row.ticker
            / f"{row.decision_date}.png"
            for row in self.rows.itertuples(
                index=False
            )
        ]

        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(
            self.paths[index]
        ) as image:
            if image.size != (160, 160):
                raise ValueError(
                    f"Invalid image size: "
                    f"{self.paths[index]}"
                )

            x = to_tensor(
                image.convert("RGB")
            )

        return (
            x,
            int(self.y[index]),
            float(self.r[index]),
        )


class DateBatchSampler(Sampler):
    def __init__(
        self,
        rows,
        seed,
        min_stocks=80,
    ):
        self.groups = [
            group.index.to_list()
            for _, group in rows.groupby(
                "decision_date",
                sort=True,
            )
        ]

        self.seed = int(seed)
        self.epoch = 0

        sizes = [
            len(group)
            for group in self.groups
        ]

        if min(sizes) < min_stocks:
            raise ValueError(
                f"Too few stocks on a date: "
                f"{min(sizes)}"
            )

        if sum(sizes) != len(rows):
            raise AssertionError(
                "Incomplete sample coverage"
            )

        print(
            f"[DATE BATCHES] "
            f"{len(self.groups)} dates "
            f"| min={min(sizes)} "
            f"median={int(np.median(sizes))} "
            f"max={max(sizes)}"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed
            + self.epoch * 12347
        )

        for index in rng.permutation(
            len(self.groups)
        ):
            group = self.groups[
                int(index)
            ].copy()

            rng.shuffle(group)

            yield group

    def __len__(self):
        return len(self.groups)


def pairwise_loss(logits, returns):
    scores = (
        logits[:, 1]
        - logits[:, 0]
    )

    differences = (
        returns[:, None]
        - returns[None, :]
    )

    positive = differences > 0.0

    if not bool(positive.any()):
        return scores.sum() * 0.0

    score_diff = (
        scores[:, None]
        - scores[None, :]
    )

    return F.softplus(
        -score_diff[positive]
    ).mean()


def calculate_metrics(
    rows,
    raw_scores,
    top_k=10,
):
    df = rows[
        [
            "decision_date",
            "ticker",
            "label",
            "return_5d",
        ]
    ].copy()

    df["score"] = np.asarray(
        raw_scores,
        dtype=float,
    )

    y = df.label.to_numpy(int)
    scores = df.score.to_numpy(float)

    predicted = (
        scores >= 0.0
    ).astype(int)

    daily_aucs = []
    daily_ics = []
    top_excess = []
    top_bottom = []

    for _, day in df.groupby(
        "decision_date",
        sort=False,
    ):
        if day.label.nunique() == 2:
            daily_aucs.append(
                roc_auc_score(
                    day.label,
                    day.score,
                )
            )

        if (
            day.score.nunique() > 1
            and day.return_5d.nunique() > 1
        ):
            daily_ics.append(
                day.score.corr(
                    day.return_5d,
                    method="spearman",
                )
            )

        if len(day) >= 2 * top_k:
            ordered = day.sort_values(
                "score"
            )

            top = ordered.tail(
                top_k
            ).return_5d.mean()

            bottom = ordered.head(
                top_k
            ).return_5d.mean()

            market = ordered.return_5d.mean()

            top_excess.append(
                top - market
            )

            top_bottom.append(
                top - bottom
            )

    return {
        "auc": float(
            roc_auc_score(y, scores)
        ),

        "daily_auc": float(
            np.mean(daily_aucs)
        ),

        "rank_ic": float(
            np.nanmean(daily_ics)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                predicted,
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),

        "up_ratio": float(
            predicted.mean()
        ),

        "top10_excess_5d": float(
            np.mean(top_excess)
        ),

        "top10_bottom10_spread_5d": float(
            np.mean(top_bottom)
        ),
    }


def evaluate(
    model,
    loader,
    rows,
    device,
    save_predictions=False,
):
    model.eval()

    all_scores = []
    all_labels = []
    all_returns = []

    ce_total = 0.0

    with torch.inference_mode():
        for images, y, returns in loader:
            logits = model(
                images.to(
                    device,
                    non_blocking=True,
                )
            )

            ce_total += F.cross_entropy(
                logits,
                y.to(
                    device,
                    non_blocking=True,
                ),
                reduction="sum",
            ).item()

            scores = (
                logits[:, 1]
                - logits[:, 0]
            )

            all_scores.append(
                scores.float().cpu().numpy()
            )

            all_labels.append(
                y.numpy()
            )

            all_returns.append(
                returns.numpy()
            )

    scores = np.concatenate(
        all_scores
    )

    if not np.array_equal(
        np.concatenate(all_labels),
        rows.label.to_numpy(np.int64),
    ):
        raise AssertionError(
            "Dev label order mismatch"
        )

    if not np.allclose(
        np.concatenate(all_returns),
        rows.return_5d.to_numpy(),
        atol=1e-6,
    ):
        raise AssertionError(
            "Dev return order mismatch"
        )

    metrics = calculate_metrics(
        rows,
        scores,
    )

    metrics["ce_loss"] = float(
        ce_total / len(rows)
    )

    if save_predictions:
        return metrics, scores

    return metrics


def train_mode(
    mode,
    args,
    train_ds,
    dev_ds,
    train_rows,
    dev_rows,
    device,
):
    set_seed(args.seed)

    model = ChartTemporalClassifier(
        "cnn_tcn"
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    date_sampler = DateBatchSampler(
        train_rows,
        seed=args.seed,
        min_stocks=args.min_stocks,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=date_sampler,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    metrics_dir = Path(
        "outputs/metrics"
    )

    checkpoint_dir = Path(
        "outputs/checkpoints"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"exp100_tcn_{mode}_"
        f"seed{args.seed}_dev2021"
    )

    checkpoint_path = (
        checkpoint_dir
        / f"{prefix}_best.pt"
    )

    history_path = (
        metrics_dir
        / f"{prefix}_history.csv"
    )

    prediction_path = (
        metrics_dir
        / f"{prefix}_predictions.csv"
    )

    best_ic = float("-inf")
    best_epoch = 0
    stale = 0
    history = []

    started = time.perf_counter()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\n[MODEL] {mode} "
        f"| params={params:,} "
        f"| rank_weight={args.rank_weight}",
        flush=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        date_sampler.set_epoch(epoch)
        model.train()

        train_labels = []
        train_scores = []

        ce_sum = 0.0
        rank_sum = 0.0
        seen = 0

        for images, y, returns in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            returns = returns.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            ce = F.cross_entropy(
                logits,
                y,
            )

            if mode == "ce":
                rank = torch.zeros(
                    (),
                    device=device,
                )
            else:
                rank = pairwise_loss(
                    logits,
                    returns,
                )

            if mode == "ce":
                loss = ce

            elif mode == "ranking":
                loss = rank

            else:
                loss = (
                    ce
                    + args.rank_weight * rank
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            n = len(y)
            seen += n

            ce_sum += (
                float(ce.detach()) * n
            )

            rank_sum += (
                float(rank.detach()) * n
            )

            train_labels.append(
                y.detach().cpu().numpy()
            )

            scores = (
                logits[:, 1]
                - logits[:, 0]
            )

            train_scores.append(
                scores.detach().float().cpu().numpy()
            )

        if seen != len(train_ds):
            raise AssertionError(
                f"Expected {len(train_ds)} "
                f"training samples, got {seen}"
            )

        train_auc = roc_auc_score(
            np.concatenate(train_labels),
            np.concatenate(train_scores),
        )

        dev = evaluate(
            model,
            dev_loader,
            dev_rows,
            device,
        )

        improved = (
            dev["rank_ic"]
            > best_ic + 1e-5
        )

        if improved:
            best_ic = dev["rank_ic"]
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state": (
                        model.state_dict()
                    ),
                    "mode": mode,
                    "epoch": epoch,
                    "seed": args.seed,
                    "rank_weight": (
                        args.rank_weight
                    ),
                    "dev_metrics": dev,
                    "train_end": "2020-12-31",
                    "dev_year": 2021,
                },
                checkpoint_path,
            )

        else:
            stale += 1

        history.append({
            "epoch": epoch,
            "train_auc_online": (
                float(train_auc)
            ),
            "train_ce_online": (
                ce_sum / seen
            ),
            "train_rank_online": (
                rank_sum / seen
            ),
            **{
                f"dev_{key}": value
                for key, value in dev.items()
            },
        })

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"[{mode}] "
            f"ep={epoch:02d} "
            f"train_auc={train_auc:.4f} "
            f"dev_auc={dev['auc']:.4f} "
            f"IC={dev['rank_ic']:+.4f} "
            f"top10_excess="
            f"{dev['top10_excess_5d']:+.4%} "
            f"spread="
            f"{dev['top10_bottom10_spread_5d']:+.4%} "
            f"bal={dev['balanced_accuracy']:.4f} "
            f"up={dev['up_ratio']:.3f} "
            f"best={best_epoch} "
            f"patience={stale}/{args.patience} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        if stale >= args.patience:
            print(
                f"[EARLY STOP] {mode}",
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

    best, scores = evaluate(
        model,
        dev_loader,
        dev_rows,
        device,
        save_predictions=True,
    )

    predictions = dev_rows[
        [
            "ticker",
            "decision_date",
            "return_5d",
            "label",
        ]
    ].copy()

    predictions["score"] = scores
    predictions["prob_up"] = expit(scores)

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    return {
        "mode": mode,
        "params": params,
        "best_epoch": best_epoch,
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
        "--root",
        type=Path,
        default=Path("data/exp100"),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.5,
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
        "--eval-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--min-stocks",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.patience < 1
        or args.rank_weight < 0
        or args.lr <= 0
        or args.eval_batch_size < 1
        or args.min_stocks < 2
        or args.workers < 0
    ):
        parser.error(
            "Invalid training arguments"
        )

    set_seed(args.seed)

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

    train_rows, dev_rows = read_split(
        args.root
    )

    train_ds = ChartReturnDataset(
        train_rows,
        args.root,
    )

    dev_ds = ChartReturnDataset(
        dev_rows,
        args.root,
    )

    summary_file = Path(
        "outputs/metrics/"
        f"exp100_tcn_ranking_seed{args.seed}_summary.csv"
    )

    if summary_file.exists():
        results = pd.read_csv(
            summary_file
        )
    else:
        results = pd.DataFrame()

    for mode in args.modes:
        row = train_mode(
            mode,
            args,
            train_ds,
            dev_ds,
            train_rows,
            dev_rows,
            device,
        )

        if not results.empty:
            results = results.loc[
                results["mode"] != mode
            ].copy()

        results = pd.concat(
            [
                results,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        results.to_csv(
            summary_file,
            index=False,
        )

    print("\n[SUMMARY]")

    print(
        results.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print(f"[SAVED] {summary_file}")

    print(
        "[NOTE] Top-10 metrics use "
        "overlapping 5-day Close-to-Close returns. "
        "They are NOT a tradable backtest."
    )


if __name__ == "__main__":
    main()
