
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


WINDOW = 20
DAY_WIDTH = 8
SIZE = 160

PRICE_TOP = 4
PRICE_BOTTOM = 117
VOLUME_TOP = 126
VOLUME_BOTTOM = 155

UP_COLOR = (30, 143, 99)
DOWN_COLOR = (205, 77, 77)


def render_window(df: pd.DataFrame) -> Image.Image:
    if len(df) != WINDOW:
        raise ValueError(f"Expected {WINDOW} days, got {len(df)}")

    image = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(image)

    # Window-specific Min-Max normalization
    low = float(df["Low"].min())
    high = float(df["High"].max())

    span = max(high - low, 1e-8)

    # 5% vertical margin
    low -= span * 0.05
    high += span * 0.05

    def price_y(price):
        ratio = (high - price) / (high - low)
        y = PRICE_TOP + ratio * (PRICE_BOTTOM - PRICE_TOP)
        return max(PRICE_TOP, min(PRICE_BOTTOM, round(y)))

    # Volume normalization within each window
    max_volume = max(1.0, float(df["Volume"].max()))

    # Price / volume separator
    draw.line(
        (0, 121, SIZE - 1, 121),
        fill=(220, 225, 231),
    )

    for i, row in enumerate(df.itertuples(index=False)):
        x = i * DAY_WIDTH + DAY_WIDTH // 2

        color = (
            UP_COLOR
            if row.Close >= row.Open
            else DOWN_COLOR
        )

        # Candlestick wick
        draw.line(
            (x, price_y(row.High), x, price_y(row.Low)),
            fill=color,
            width=1,
        )

        # Candlestick body
        y_open = price_y(row.Open)
        y_close = price_y(row.Close)

        draw.rectangle(
            (
                x - 2,
                min(y_open, y_close),
                x + 2,
                max(y_open, y_close),
            ),
            fill=color,
        )

        # Volume bar
        volume_height = round(
            max(0.0, float(row.Volume))
            / max_volume
            * (VOLUME_BOTTOM - VOLUME_TOP)
        )

        if volume_height > 0:
            draw.rectangle(
                (
                    x - 2,
                    VOLUME_BOTTOM - volume_height,
                    x + 2,
                    VOLUME_BOTTOM,
                ),
                fill=color,
            )

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("data/processed/samples.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/charts"))
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    samples = pd.read_csv(
        args.samples,
        dtype={
            "ticker": str,
            "split": str,
            "window_start": str,
            "decision_date": str,
        },
    )

    if args.split != "all":
        samples = samples[samples["split"] == args.split]

    if args.limit:
        samples = samples.head(args.limit)

    if samples.empty:
        raise RuntimeError("No samples selected")

    generated = 0
    skipped = 0

    for ticker, part in samples.groupby("ticker"):
        path = args.raw_dir / f"{ticker}.csv"

        df = pd.read_csv(path, dtype={"Date": str})
        df = df.sort_values("Date").reset_index(drop=True)

        date_to_idx = {
            date: i for i, date in enumerate(df["Date"])
        }

        if len(date_to_idx) != len(df):
            raise ValueError(f"{ticker}: Duplicate dates")

        for sample in part.itertuples(index=False):
            idx = date_to_idx.get(sample.decision_date)

            if idx is None or idx < WINDOW - 1:
                raise ValueError(
                    f"{ticker}: Invalid date {sample.decision_date}"
                )

            window = df.iloc[idx - WINDOW + 1:idx + 1]

            if window["Date"].iloc[0] != sample.window_start:
                raise AssertionError("Window date mismatch")

            output = (
                args.output_dir
                / sample.split
                / ticker
                / f"{sample.decision_date}.png"
            )

            if output.exists() and not args.overwrite:
                skipped += 1
                continue

            output.parent.mkdir(parents=True, exist_ok=True)

            image = render_window(window)
            image.save(output, optimize=True)

            generated += 1

    print("\n[CHART GENERATION COMPLETE]")
    print("  Normalization: Window Min-Max")
    print(f"  Window: {WINDOW} trading days")
    print(f"  Image size: {SIZE}x{SIZE}")
    print(f"  Generated: {generated:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
