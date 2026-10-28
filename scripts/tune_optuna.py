
import argparse
import json
import time
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.build_dataset import ChartDataset
from src.models.day_aligned_vit import DayAlignedViT
from src.models.standard_vit import StandardViT
from src.train import run_epoch, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["day_aligned_vit", "standard_vit"], default="day_aligned_vit")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if min(args.trials, args.epochs, args.batch_size, args.patience) < 1:
        parser.error("trials, epochs, batch-size, patience must be positive")

    output_dir = Path("outputs/metrics")
    ckpt_dir = Path("outputs/checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"optuna_{args.model}_e{args.epochs}_seed{args.seed}"
    db_file = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{db_file.as_posix()}"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[DEVICE] {device}", flush=True)

    # Load datasets only once
    train_ds = ChartDataset("train")
    val_ds = ChartDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # One Optuna trial = one model training experiment
    # --------------------------------------------------
    def objective(trial):
        set_seed(args.seed)

        config = {
            "image_size": 160,
            "embed_dim": trial.suggest_categorical(
                "embed_dim", [64, 128]
            ),
            "depth": trial.suggest_int(
                "depth", 2, 4
            ),
            "num_heads": 4,
            "dropout": trial.suggest_categorical(
                "dropout", [0.0, 0.1, 0.2]
            ),
            "num_classes": 2,
        }

        lr = trial.suggest_float(
            "lr", 3e-5, 1e-3, log=True
        )

        weight_decay = trial.suggest_float(
            "weight_decay", 1e-4, 1e-1, log=True
        )

        if args.model == "day_aligned_vit":
            config["num_days"] = 20
            model = DayAlignedViT(**config)
        else:
            config["patch_size"] = 32
            model = StandardViT(**config)

        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        checkpoint_path = (
            ckpt_dir
            / f"{study_name}_trial{trial.number}.pt"
        )

        best_auc = -1.0
        best_epoch = 0
        best_f1 = 0.0
        stale = 0

        started = time.perf_counter()

        # ----------------------------------------------
        # Training epochs
        # ----------------------------------------------
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
            )

            val = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            auc = val["auroc"]

            if auc > best_auc + 1e-6:
                best_auc = auc
                best_epoch = epoch
                best_f1 = val["macro_f1"]
                stale = 0

                torch.save(
                    {
                        "model_name": args.model,
                        "model_config": config,
                        "model_state": model.state_dict(),
                        "seed": args.seed,
                        "epoch": epoch,
                        "train_metrics": train,
                        "val_metrics": val,
                        "normalization": "window_minmax",
                        "window": 20,
                        "horizon": 5,
                        "stride": 1,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1

            elapsed = time.perf_counter() - started

            print(
                f"Trial {trial.number:03d} "
                f"| Epoch {epoch:02d} "
                f"| train_loss={train['loss']:.4f} "
                f"| val_loss={val['loss']:.4f} "
                f"| val_auc={auc:.4f} "
                f"| val_f1={val['macro_f1']:.4f} "
                f"| best_auc={best_auc:.4f} "
                f"| elapsed={elapsed:.0f}s",
                flush=True,
            )

            # Report intermediate result to Optuna
            trial.report(best_auc, step=epoch)

            # Stop unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping within this trial
            if stale >= args.patience:
                break

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_val_macro_f1", best_f1)
        trial.set_user_attr("checkpoint", str(checkpoint_path))
        trial.set_user_attr("model_config", config)

        return best_auc

    # --------------------------------------------------
    # Optuna study
    # --------------------------------------------------
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=4,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,
            n_warmup_steps=6,
            interval_steps=2,
        ),
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
    )

    # --------------------------------------------------
    # Save trial history
    # --------------------------------------------------
    trials_path = output_dir / f"{study_name}_trials.csv"

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    # --------------------------------------------------
    # Save best configuration
    # --------------------------------------------------
    winner = study.best_trial

    summary = {
        "model": args.model,
        "best_auroc": winner.value,
        "best_epoch": winner.user_attrs["best_epoch"],
        "best_val_macro_f1": winner.user_attrs[
            "best_val_macro_f1"
        ],
        "params": winner.params,
        "model_config": winner.user_attrs["model_config"],
        "checkpoint": winner.user_attrs["checkpoint"],
    }

    summary_file = (
        output_dir / f"{study_name}_best.json"
    )

    summary_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("[OPTUNA COMPLETE]")
    print(f"Best Trial : {winner.number}")
    print(f"Best AUROC : {winner.value:.4f}")
    print(f"Best Epoch : {summary['best_epoch']}")
    print(f"Best Params: {winner.params}")
    print(f"Checkpoint : {summary['checkpoint']}")
    print(f"Study CSV  : {trials_path}")
    print(f"Best JSON  : {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
