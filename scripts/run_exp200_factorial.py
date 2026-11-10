"""200-stock / longer-history JKX I20 factorial walk-forward experiment.

Precondition:
  python -m scripts.prepare_exp200
Run (no wall-clock cutoff):
  python -u -m scripts.run_exp200_factorial

2 universes (legacy100, all200) x 2 training starts (2015,2010)
x 3 out-of-time evaluation years (2019-2021) x 3 seeds = 36 trainings.

Each train/test compares SAME fixed target calendar period. All 200-stock
models are ALSO evaluated on the original 100-stock slice, allowing a cleaner
measurement of the training universe effect. No 2022+ data are accessed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from scripts.jkx_i20_reproduce import Jkx20, MemmapCharts, cross_section_metrics, evaluate, set_seed


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_data(root, raw_root):
    ready_file = root / "READY.json"
    meta_file = root / "meta.csv"
    image_file = root / "images.npy"
    manifest_file = raw_root / "universe200.csv"
    for file in (ready_file, meta_file, image_file, manifest_file):
        if not file.is_file():
            raise FileNotFoundError(f"Missing {file}. Run python -m scripts.prepare_exp200 first")
    ready = json.loads(ready_file.read_text(encoding="utf-8"))
    if ready.get("fingerprint") != sha256(meta_file):
        raise ValueError("Dataset fingerprint mismatch")
    manifest = pd.read_csv(manifest_file, dtype={"ticker": str})
    if len(manifest) != 200 or manifest.ticker.nunique() != 200:
        raise ValueError("Expected exact 200-stock locked manifest")
    meta = pd.read_csv(
        meta_file,
        dtype={"ticker": str, "group": str, "window_start": str,
               "decision_date": str, "target_date": str},
    )
    data = np.load(image_file, mmap_mode="r")
    if tuple(data.shape) != (len(meta), 1, 64, 60):
        raise ValueError(f"Metadata/image mismatch: {data.shape}")
    del data
    if meta.ticker.nunique() != 200 or meta.decision_date.max() >= "2022-01-01":
        raise ValueError("Unexpected universe or future data")
    if meta.duplicated(["ticker", "decision_date"]).any():
        raise ValueError("Duplicate ticker-date in metadata")
    if not np.isfinite(meta.return_5d.to_numpy(float)).all():
        raise ValueError("Non-finite adjusted future returns")
    expected = (meta.return_5d.to_numpy(float) > 0).astype(int)
    if not np.array_equal(expected, meta.label.to_numpy(int)):
        raise ValueError("Incorrect binary direction labels")
    groups = meta.ticker.map(dict(zip(manifest.ticker, manifest.group)))
    if not groups.equals(meta.group):
        raise ValueError("Group labels differ from frozen membership")
    legacy = set(manifest.loc[manifest.group == "legacy", "ticker"])
    if len(legacy) != 100:
        raise ValueError("Expected 100 legacy stocks")
    print(f"[DATA] 200 tickers | {len(meta):,} I20/R5 samples | years 2010-2021", flush=True)
    return meta, legacy, ready["fingerprint"]


def split(meta, legacy, year, start_year, universe):
    if year not in (2019, 2020, 2021):
        raise ValueError("Evaluation years must be 2019/2020/2021")
    if start_year not in (2010, 2015):
        raise ValueError("Start years must be 2010/2015")
    if universe not in (100, 200):
        raise ValueError("Universe must be 100 or 200")
    val_year = year - 1
    eligible = (
        meta.ticker.isin(legacy).to_numpy()
        if universe == 100
        else np.ones(len(meta), dtype=bool)
    )
    begins = meta.window_start.to_numpy()
    targets = meta.target_date.to_numpy()
    start_date = f"{start_year}-01-01"
    val_date = f"{val_year}-01-01"
    test_date = f"{year}-01-01"
    test_end = f"{year + 1}-01-01"
    tr = np.flatnonzero(eligible & (begins >= start_date) & (targets < val_date))
    va = np.flatnonzero(eligible & (begins >= val_date) & (targets < test_date))
    te = np.flatnonzero(eligible & (begins >= test_date) & (targets < test_end))
    if min(len(tr), len(va), len(te)) < 1000:
        raise ValueError(f"Not enough data for {start_year}, {universe}, {year}")
    a, b, c = meta.iloc[tr], meta.iloc[va], meta.iloc[te]
    if a.target_date.max() >= b.window_start.min() or b.target_date.max() >= c.window_start.min():
        raise ValueError("Information horizon or input window crosses a split boundary")
    if set(c.ticker.unique()) != (legacy if universe == 100 else set(meta.ticker.unique())):
        raise ValueError("At least one stock missing from the test year")
    return tr, va, te, c.reset_index(drop=True)


def loader(data_root, indices, batch_size, shuffle, workers, seed):
    ds = MemmapCharts(data_root / "images.npy", data_root / "meta.csv", indices)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        generator=torch.Generator().manual_seed(seed),
    )


def subset_scores(frame, probability, legacy):
    """Calculate OOS metrics on all and shared original-100 candidates."""
    results = {}
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "common100": frame.ticker.isin(legacy).to_numpy(),
    }
    if frame.ticker.nunique() > 100:
        masks.update({
            "added100": (~frame.ticker.isin(legacy)).to_numpy(),
            "mid_candidate": (frame.group == "mid_candidate").to_numpy(),
            "small_candidate": (frame.group == "small_candidate").to_numpy(),
        })
    for group_name, mask in masks.items():
        f = frame.loc[mask].reset_index(drop=True)
        p = np.asarray(probability)[mask]
        y = f.label.to_numpy(int)
        if not len(f) or len(np.unique(y)) != 2:
            raise ValueError(f"Invalid subgroup {group_name}")
        preds = p >= 0.5
        stats = cross_section_metrics(f, p)
        stats.update({
            "auc": float(roc_auc_score(y, p)),
            "balanced_acc": float(balanced_accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
            "up_ratio": float(np.mean(preds)),
            "n": int(len(y)),
            "stocks": int(f.ticker.nunique()),
        })
        results[group_name] = stats
    return results


def run_one(a, device, meta, legacy, fingerprint, universe, start_year, year, seed):
    tag = f"u{universe}_start{start_year}"
    run_dir = a.out / tag / f"test{year}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    pred_path = run_dir / "predictions.csv"
    config = {
        "version": 1,
        "data_fingerprint": fingerprint,
        "architecture": "JKX-I20-original-708866-parameters",
        "universe": universe, "start_year": start_year,
        "test_year": year, "validation_year": year - 1,
        "seed": seed, "epochs": a.epochs, "patience": a.patience,
        "lr": a.lr, "batch_size": a.batch_size,
    }
    if summary_path.exists() and pred_path.exists() and not a.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("config") != config:
            raise ValueError(f"Completed experiment config mismatch: {run_dir}; use --force deliberately")
        print(f"[SKIP DONE] {tag} {year} seed{seed}", flush=True)
        return
    tr, va, te, test_frame = split(meta, legacy, year, start_year, universe)
    set_seed(seed)
    train_loader = loader(a.root, tr, a.batch_size, True, a.workers, seed)
    val_loader = loader(a.root, va, a.batch_size, False, a.workers, seed)
    test_loader = loader(a.root, te, a.batch_size, False, a.workers, seed)
    model = Jkx20().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    print(
        f"[RUN] {tag} eval={year} seed={seed} "
        f"N(train/val/test)={len(tr):,}/{len(va):,}/{len(te):,}", flush=True,
    )
    for epoch in range(1, a.epochs + 1):
        model.train()
        n = 0
        train_loss = 0.0
        for batch, (images, targets) in enumerate(train_loader, 1):
            x = images.to(device, non_blocking=True)
            y = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
            n += len(y)
            if a.log_steps and batch % a.log_steps == 0:
                print(
                    f"[BATCH] {tag} {year} seed{seed} ep={epoch} "
                    f"{batch}/{len(train_loader)} loss={train_loss/n:.5f}", flush=True,
                )
        val_metrics, _ = evaluate(model, val_loader, device)
        improved = val_metrics["loss"] < best - 1e-6
        if improved:
            best = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model_state": model.state_dict(), "best_epoch": epoch,
                 "config": config, "validation": val_metrics},
                run_dir / "best.pt",
            )
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss_online": train_loss/n,
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"[EPOCH] {tag} eval={year} seed={seed} {epoch:02d} "
            f"trainloss={train_loss/n:.5f} valloss={val_metrics['loss']:.5f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"best={best_epoch} patience={stale}/{a.patience} "
            f"elapsed={time.perf_counter()-started:.0f}s", flush=True,
        )
        if stale >= a.patience:
            print(f"[EARLY STOP] {tag} eval={year} seed={seed}", flush=True)
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    dev_metrics, probs = evaluate(model, test_loader, device)
    groups = subset_scores(test_frame, probs, legacy)
    prediction = test_frame[["ticker", "group", "decision_date", "target_date", "return_5d", "label"]].copy()
    prediction["prob_up"] = probs
    prediction.to_csv(pred_path, index=False)
    common = groups["common100"]
    stats = {
        "universe": universe, "start_year": start_year, "test_year": year,
        "seed": seed, "train_n": len(tr), "val_n": len(va), "test_n": len(te),
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "seconds": round(time.perf_counter()-started, 2),
        **{f"test_{k}": float(v) for k, v in dev_metrics.items()},
        **{f"test_common100_{k}": float(v) for k, v in common.items()},
        **{f"test_all_{k}": float(v) for k, v in groups["all"].items()},
    }
    summary_path.write_text(json.dumps({"config": config, "metrics": stats, "groups": groups}, indent=2), encoding="utf-8")
    print(
        f"[RESULT] {tag} year={year} seed{seed} "
        f"common100_AUC={common['auc']:.4f} common100_IC={common['rank_ic']:+.4f} "
        f"top10={common['top10_excess_5d']:+.3%} "
        f"all{universe}_IC={groups['all']['rank_ic']:+.4f}", flush=True,
    )


def collect(a):
    rows, g_rows = [], []
    for file in sorted(a.out.glob("u*_start*/test*/seed*/summary.json")):
        doc = json.loads(file.read_text(encoding="utf-8"))
        r = doc["metrics"]
        rows.append(r)
        for name, stats in doc["groups"].items():
            g_rows.append({"universe": r["universe"], "start_year": r["start_year"],
                           "test_year": r["test_year"], "seed": r["seed"],
                           "eval_group": name, **stats})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["universe", "start_year", "test_year", "seed"])
    df.to_csv(a.out / "all_runs.csv", index=False)
    pd.DataFrame(g_rows).sort_values(["universe", "start_year", "test_year", "seed", "eval_group"]).to_csv(
        a.out / "all_group_runs.csv", index=False
    )
    ensemble_results = []
    group_results = []
    for (universe, start_year, year), g in df.groupby(["universe", "start_year", "test_year"]):
        frames = []
        for seed in sorted(g.seed):
            f = a.out / f"u{universe}_start{start_year}" / f"test{year}" / f"seed{seed}" / "predictions.csv"
            frames.append(pd.read_csv(f, dtype={"ticker": str, "decision_date": str, "target_date": str}))
        first = frames[0]
        columns = ["ticker", "group", "decision_date", "target_date", "label"]
        for other in frames[1:]:
            if not first[columns].equals(other[columns]) or not np.allclose(
                first.return_5d.to_numpy(), other.return_5d.to_numpy(), rtol=0, atol=1e-12
            ):
                raise ValueError(f"Seed prediction alignment mismatch: {universe},{start_year},{year}")
        p = np.mean(np.stack([f.prob_up.to_numpy(float) for f in frames]), axis=0)
        ens = first.copy()
        ens["prob_up"] = p
        ens.to_csv(a.out / f"ensemble_u{universe}_start{start_year}_test{year}.csv", index=False)
        legacy_tickers = set(first.loc[first.group == "legacy", "ticker"])
        # For 200-stock model, group='legacy' is in predictions.
        groups = subset_scores(ens, p, legacy_tickers)
        for label, stats in groups.items():
            group_results.append({"universe": universe, "start_year": start_year,
                                  "test_year": year, "seed_count": len(frames),
                                  "eval_group": label, **stats})
        common = groups["common100"]
        overall = groups["all"]
        ensemble_results.append({
            "universe": universe, "start_year": start_year, "test_year": year,
            "seed_count": len(frames),
            **{f"common100_{k}": v for k, v in common.items()},
            **{f"all_{k}": v for k, v in overall.items()},
        })
    results = pd.DataFrame(ensemble_results).sort_values(["test_year", "universe", "start_year"])
    results.to_csv(a.out / "ensemble_summary.csv", index=False)
    pd.DataFrame(group_results).sort_values(["test_year", "universe", "start_year", "eval_group"]).to_csv(
        a.out / "ensemble_group_summary.csv", index=False
    )
    delta_rows = []
    required = {(u, s, year) for year in a.years for u in (100, 200) for s in (2015, 2010)}
    available = set(zip(results.universe, results.start_year, results.test_year))
    for year in a.years:
        if not all((u, s, year) in available for u, s in ((100,2015),(100,2010),(200,2015),(200,2010))):
            continue
        indexed = results.set_index(["universe", "start_year", "test_year"])
        A, B, C, D = (indexed.loc[(u,s,year)] for u,s in ((100,2015),(100,2010),(200,2015),(200,2010)))
        for metric in ("auc", "rank_ic", "top10_excess_5d", "top10_bottom10_spread_5d"):
            key = f"common100_{metric}"
            delta_rows.append({"test_year": year, "metric": metric,
               "A_u100_2015": A[key], "B_u100_2010": B[key],
               "C_u200_2015": C[key], "D_u200_2010": D[key],
               "longer_history_u100_B_minus_A": B[key] - A[key],
               "longer_history_u200_D_minus_C": D[key] - C[key],
               "universe_expansion_2015_C_minus_A": C[key] - A[key],
               "universe_expansion_2010_D_minus_B": D[key] - B[key],
               "interaction_D_minus_C_minus_B_plus_A": D[key]-C[key]-B[key]+A[key],
            })
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(a.out / "contrasts.csv", index=False)
    print("\n[ENSEMBLE] Common100 evaluation (same stocks; out-of-time)", flush=True)
    print(results[["test_year", "universe", "start_year", "seed_count",
                   "common100_auc", "common100_rank_ic", "common100_top10_excess_5d"]]
          .to_string(index=False, float_format=lambda v:f"{v:.5f}"), flush=True)
    print(f"[SAVED] {a.out / 'all_runs.csv'} (up to 36) | {a.out / 'ensemble_summary.csv'} (up to 12)", flush=True)
    print("[LIMITATIONS] Current survivors, unverified historical cap buckets, overlapping 5-day labels; not a tradable backtest.", flush=True)
    print("[HOLDOUT] 2022-2025 never used", flush=True)


def main():
    p = argparse.ArgumentParser(description="JKX I20 factorial: universe 100/200 x history 2015/2010")
    p.add_argument("--root", type=Path, default=Path("data/exp200_jkx"))
    p.add_argument("--raw-root", type=Path, default=Path("data/exp200"))
    p.add_argument("--out", type=Path, default=Path("outputs/exp200_factorial"))
    p.add_argument("--stage", choices=["train", "report"], default="train")
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--universes", type=int, nargs="+", choices=[100, 200], default=[100, 200])
    p.add_argument("--starts", type=int, nargs="+", choices=[2015, 2010], default=[2015, 2010])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-steps", type=int, default=500)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.log_steps < 0:
        p.error("Invalid numeric argument")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        collect(args)
        return
    meta, legacy, fingerprint = prepare_data(args.root, args.raw_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planned = [(year, u, start, seed) for year in args.years for u in args.universes for start in args.starts for seed in args.seeds]
    print(f"[DEVICE] {device} | [PLAN] {len(planned)} trainings | no wall-clock cap", flush=True)
    try:
        for i, (year, u, start, seed) in enumerate(planned, start=1):
            print(f"[PROGRESS] {i}/{len(planned)}", flush=True)
            run_one(args, device, meta, legacy, fingerprint, u, start, year, seed)
    finally:
        collect(args)


if __name__ == "__main__":
    main()
