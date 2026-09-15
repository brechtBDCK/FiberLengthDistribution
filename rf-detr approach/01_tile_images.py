#!/usr/bin/env python3
"""Tile large microscopy images into square images for RF-DETR training.

Splits are assigned at SOURCE-IMAGE level to avoid leakage between train/valid/test.
Tiles are saved directly inside dataset/{train,valid,test}/, which matches RF-DETR's
COCO directory layout once annotations are added.

Example:
    python 01_tile_images.py \
        --input raw_images \
        --output fiber_dataset \
        --tile-size 512 \
        --overlap 64 \
        --train 0.8 --valid 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def positions(length: int, tile: int, overlap: int) -> list[int]:
    """Tile starts that cover the full dimension without tiny edge tiles."""
    if length <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile-size")
    out = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if out[-1] != last:
        out.append(last)
    return out


def assign_splits(files: list[Path], train: float, valid: float, test: float, seed: int) -> dict[Path, str]:
    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+valid+test must equal 1.0, got {total}")

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train))
    n_valid = int(round(n * valid))

    # Keep counts sane after rounding.
    n_train = min(n_train, n)
    n_valid = min(n_valid, n - n_train)

    split_map: dict[Path, str] = {}
    for i, p in enumerate(shuffled):
        if i < n_train:
            split_map[p] = "train"
        elif i < n_train + n_valid:
            split_map[p] = "valid"
        else:
            split_map[p] = "test"
    return split_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Image file or folder of source images")
    ap.add_argument("--output", required=True, type=Path, help="Output RF-DETR dataset root")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    args = ap.parse_args()

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    if not files:
        raise SystemExit("No input images found.")

    # Important: split by original source image, not by tile.
    split_map = assign_splits(files, args.train, args.valid, args.test, args.seed)

    for split in ("train", "valid", "test"):
        (args.output / split).mkdir(parents=True, exist_ok=True)

    metadata = {
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "source_level_split": True,
        "tiles": [],
    }

    counts = {"train": 0, "valid": 0, "test": 0}

    for src in files:
        split = split_map[src]
        image = Image.open(src).convert("RGB")
        w, h = image.size
        xs = positions(w, args.tile_size, args.overlap)
        ys = positions(h, args.tile_size, args.overlap)

        # Short hash prevents collisions when different directories contain same filename.
        source_key = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:8]

        for y in ys:
            for x in xs:
                x2 = min(x + args.tile_size, w)
                y2 = min(y + args.tile_size, h)
                crop = image.crop((x, y, x2, y2))

                # Only needed if the source is smaller than the requested tile.
                if crop.size != (args.tile_size, args.tile_size):
                    canvas = Image.new("RGB", (args.tile_size, args.tile_size), (0, 0, 0))
                    canvas.paste(crop, (0, 0))
                    crop = canvas

                name = f"{src.stem}_{source_key}_x{x:05d}_y{y:05d}.{args.format}"
                dst = args.output / split / name
                if args.format == "jpg":
                    crop.save(dst, quality=95, subsampling=0)
                else:
                    crop.save(dst)

                metadata["tiles"].append(
                    {
                        "tile": str(Path(split) / name),
                        "source": str(src),
                        "split": split,
                        "x": x,
                        "y": y,
                        "valid_width": x2 - x,
                        "valid_height": y2 - y,
                        "source_width": w,
                        "source_height": h,
                    }
                )
                counts[split] += 1

    with open(args.output / "tiles.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done: {sum(counts.values())} tiles -> {args.output}")
    print("Split tile counts:", counts)
    if len(files) < 3:
        print("WARNING: very few source images; source-level train/valid/test splitting may leave a split empty.")


if __name__ == "__main__":
    main()
