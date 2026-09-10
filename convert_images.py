#!/usr/bin/env python3
"""Resize every readable image to 2 MP max and save lossless WebP."""

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

INPUT_FOLDER = Path("/home/bdck/PROJECTS_WSL/FiberLengthDistribution/data")
OUTPUT_FOLDER = Path("/home/bdck/PROJECTS_WSL/FiberLengthDistribution/data_webp")
MAX_PIXELS = 20_000_000

# Input images are trusted; some legitimately exceed Pillow's safety limit.
Image.MAX_IMAGE_PIXELS = None


def convert() -> int:
    count = 0
    for path in INPUT_FOLDER.rglob("*"):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                output = OUTPUT_FOLDER / path.relative_to(INPUT_FOLDER).with_suffix(".webp")
                output.parent.mkdir(parents=True, exist_ok=True)
                scale = min(1, (MAX_PIXELS / (image.width * image.height)) ** 0.5)
                image.thumbnail(
                    (int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS
                )
                ImageOps.exif_transpose(image).convert("RGB").save(
                    output, "WEBP", lossless=True, method=6
                )
                count += 1
        except (UnidentifiedImageError, OSError):
            pass
    return count


if __name__ == "__main__":
    print(f"Converted {convert()} image(s) to {OUTPUT_FOLDER}")
