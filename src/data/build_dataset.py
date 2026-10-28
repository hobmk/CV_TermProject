
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor


class ChartDataset(Dataset):
    def __init__(
        self,
        split: str,
        samples_path: str = "data/processed/samples.csv",
        charts_dir: str = "data/charts",
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        self.split = split

        df = pd.read_csv(
            samples_path,
            dtype={
                "ticker": str,
                "split": str,
                "decision_date": str,
            },
        )

        required = {"ticker", "split", "decision_date", "label"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.samples = df[df["split"] == split].reset_index(drop=True)

        if self.samples.empty:
            raise RuntimeError(f"No samples in split: {split}")

        if not self.samples["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary (0 or 1)")

        root = Path(charts_dir)
        self.paths = [
            root / row.split / row.ticker / f"{row.decision_date}.png"
            for row in self.samples.itertuples(index=False)
        ]

        missing_files = [p for p in self.paths if not p.is_file()]

        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} images missing. "
                f"Example: {missing_files[0]}"
            )

        print(
            f"[DATASET] {split:5s} "
            f"| N={len(self.samples):,} "
            f"| Positive={self.samples['label'].mean():.2%}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            if img.size != (160, 160):
                raise ValueError(
                    f"Invalid size: {self.paths[idx]} ({img.size})"
                )

            image = to_tensor(img.convert("RGB"))

        label = int(self.samples.iloc[idx]["label"])

        return image, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        dataset = ChartDataset(split)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )

        images, labels = next(iter(loader))

        print(
            f"[BATCH] {split:5s} "
            f"| X={tuple(images.shape)} "
            f"| y={tuple(labels.shape)} "
            f"| range=({images.min():.3f}, {images.max():.3f})"
        )


if __name__ == "__main__":
    main()
