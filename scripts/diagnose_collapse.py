
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data.build_dataset import ChartDataset
from src.evaluate import build_model_from_checkpoint


def choose_indices(dataset, per_class, seed):
    labels = dataset.samples["label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    selected = []

    for cls in (0, 1):
        candidates = np.flatnonzero(labels == cls)

        if len(candidates) < per_class:
            raise ValueError(
                f"Only {len(candidates)} examples in class {cls}"
            )

        selected.extend(
            rng.choice(
                candidates,
                per_class,
                replace=False,
            )
        )

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)

    return selected.tolist()


def stage_stats(tensor, pool_method):
    x = tensor.float()

    if x.ndim == 3:
        token_between_rms = (
            x.var(dim=0, unbiased=False)
            .mean()
            .sqrt()
            .item()
        )

        if pool_method == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

    else:
        token_between_rms = None
        pooled = x

    centered = pooled - pooled.mean(
        dim=0,
        keepdim=True,
    )

    between_rms = (
        centered.square()
        .mean()
        .sqrt()
        .item()
    )

    magnitude_rms = (
        pooled.square()
        .mean()
        .sqrt()
        .item()
    )

    return {
        "token_between_rms": token_between_rms,
        "pooled_between_rms": between_rms,
        "pooled_magnitude_rms": magnitude_rms,
        "relative_variation": (
            between_rms / max(magnitude_rms, 1e-12)
        ),
    }


def diagnose_one(
    model,
    model_name,
    pooling,
    loader,
    device,
    split,
):
    captured = {
        "patch": [],
        "block_first": [],
        "block_last": [],
        "final_feature": [],
    }

    handles = []

    def hook_for(name):
        def hook(_module, _inputs, output):
            x = output.detach()

            if x.ndim == 4:
                x = x.flatten(2).transpose(1, 2)

            captured[name].append(x.cpu())

        return hook

    handles.append(
        model.patch_embed.register_forward_hook(
            hook_for("patch")
        )
    )

    handles.append(
        model.blocks[0].register_forward_hook(
            hook_for("block_first")
        )
    )

    handles.append(
        model.blocks[-1].register_forward_hook(
            hook_for("block_last")
        )
    )

    handles.append(
        model.norm.register_forward_hook(
            hook_for("final_feature")
        )
    )

    labels_parts = []
    probs_parts = []
    gap_parts = []
    image_parts = []

    model.eval()

    try:
        with torch.inference_mode():
            for images, labels in loader:
                image_parts.append(images)

                logits = model(
                    images.to(
                        device,
                        non_blocking=True,
                    )
                )

                p_up = torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]

                gap = (
                    logits[:, 1] - logits[:, 0]
                )

                labels_parts.append(labels.numpy())
                probs_parts.append(
                    p_up.cpu().numpy()
                )
                gap_parts.append(
                    gap.cpu().numpy()
                )

    finally:
        for handle in handles:
            handle.remove()

    y = np.concatenate(labels_parts)
    p = np.concatenate(probs_parts)
    gaps = np.concatenate(gap_parts)

    images = torch.cat(
        image_parts,
        dim=0,
    )

    image_between_rms = (
        images.var(dim=0, unbiased=False)
        .mean()
        .sqrt()
        .item()
    )

    stage_results = {}

    for name, pieces in captured.items():
        tensor = torch.cat(
            pieces,
            dim=0,
        )

        method = (
            "mean" if name == "patch"
            else pooling
        )

        stage_results[name] = stage_stats(
            tensor,
            method,
        )

    final_features = torch.cat(
        captured["final_feature"],
        dim=0,
    ).numpy()

    weight_diff = (
        model.head.weight[1]
        - model.head.weight[0]
    ).detach().cpu().numpy()

    if model.head.bias is not None:
        bias_diff = (
            model.head.bias[1]
            - model.head.bias[0]
        ).detach().cpu().item()
    else:
        bias_diff = 0.0

    feature_signal = (
        final_features @ weight_diff
    )

    return {
        "split": split,
        "model_name": model_name,
        "pooling": pooling,
        "sample_count": int(len(y)),
        "sample_positive_ratio": float(y.mean()),
        "image_between_rms": image_between_rms,
        "auroc": float(roc_auc_score(y, p)),
        "pred_up_ratio": float((p >= 0.5).mean()),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
        "logit_gap_mean": float(gaps.mean()),
        "logit_gap_std": float(gaps.std()),
        "head_signal_std": float(
            feature_signal.std()
        ),
        "head_weight_difference_norm": float(
            np.linalg.norm(weight_diff)
        ),
        "head_bias_difference": float(
            bias_diff
        ),
        "stages": stage_results,
    }


def print_report(result):
    print("\n" + "=" * 75)

    print(
        f"[{result['model_name']} | "
        f"{result['pooling']} | "
        f"{result['split']}]"
    )

    print(
        f"N={result['sample_count']} "
        f"| labels_up="
        f"{result['sample_positive_ratio']:.3f} "
        f"| AUROC={result['auroc']:.4f} "
        f"| predicted_up="
        f"{result['pred_up_ratio']:.3f}"
    )

    print(
        "Input image between-sample RMS: "
        f"{result['image_between_rms']:.6e}"
    )

    print(
        f"{'stage':<15} "
        f"{'token_var':>13} "
        f"{'pooled_var':>13} "
        f"{'relative':>13}"
    )

    for name, stats in result["stages"].items():
        if stats["token_between_rms"] is None:
            token = "-"
        else:
            token = (
                f"{stats['token_between_rms']:.3e}"
            )

        print(
            f"{name:<15} "
            f"{token:>13} "
            f"{stats['pooled_between_rms']:>13.3e} "
            f"{stats['relative_variation']:>13.3e}"
        )

    print(
        f"P(up): mean={result['prob_mean']:.6f} "
        f"std={result['prob_std']:.6e}"
    )

    print(
        "Logit(up-down): "
        f"mean={result['logit_gap_mean']:.6f} "
        f"std={result['logit_gap_std']:.6e}"
    )

    print(
        f"Head: feature_signal_std="
        f"{result['head_signal_std']:.6e}, "
        f"Wdiff_norm="
        f"{result['head_weight_difference_norm']:.6e}, "
        f"Biasdiff="
        f"{result['head_bias_difference']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--per-class", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/metrics/representation_diagnostics.json"
        ),
    )

    args = parser.parse_args()

    if args.per_class < 1 or args.batch_size < 1:
        parser.error(
            "per-class and batch-size must be positive"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[DEVICE] {device}")

    datasets = {
        name: ChartDataset(name)
        for name in ("train", "val")
    }

    chosen = {
        name: choose_indices(
            dataset,
            args.per_class,
            args.seed,
        )
        for name, dataset in datasets.items()
    }

    results = []

    for path_string in args.checkpoints:
        path = Path(path_string)

        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model, model_name = (
            build_model_from_checkpoint(ckpt)
        )

        if model_name != "day_aligned_vit":
            raise ValueError(
                f"Only day_aligned_vit supported: {path}"
            )

        pooling = ckpt.get(
            "model_config", {}
        ).get("pooling", "cls")

        model = model.to(device)

        print(
            f"\n[CHECKPOINT] {path} "
            f"| best_epoch={ckpt.get('epoch')} "
            f"| pooling={pooling}"
        )

        for split in ("train", "val"):
            loader = DataLoader(
                Subset(
                    datasets[split],
                    chosen[split],
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            report = diagnose_one(
                model,
                model_name,
                pooling,
                loader,
                device,
                split,
            )

            report["checkpoint"] = str(path)

            print_report(report)
            results.append(report)

        del model

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[SAVED] {args.output}")
    print(
        "[NOTE] Balanced diagnostic subsamples; "
        "not official validation metrics. "
        "Test untouched."
    )


if __name__ == "__main__":
    main()
