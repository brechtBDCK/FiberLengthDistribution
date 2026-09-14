"""Run SynthMT's published SAM3Text+HPO configuration on one image.

Set INPUT_PATH, then run: uv run python sam3text_detect.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from cellpose.transforms import smooth_sharpen_img
from PIL import Image
from transformers import Sam3Model, Sam3Processor


# -----------------------------------------------------------------------------
# Change only these values.
# -----------------------------------------------------------------------------
INPUT_PATH = Path("/home/bdck/PROJECTS_WSL/FiberLengthDistribution/data_webp/Glasvezels 1- 4 mm/beker_1/250429-Z50-B1-P1-F1`.webp")
OUTPUT_DIR = INPUT_PATH.parent / f"{INPUT_PATH.stem}_sam3text_hpo"
HF_TOKEN = None  # Or Hugging Face access token. facebook/sam3 access required.

# Published SynthMT SAM3Text HPO configuration.
TEXT_PROMPT = "thin line"
THRESHOLD = 0.1
MASK_THRESHOLD = 0.3
PERCENTILE_MIN = 3.506993161284297
PERCENTILE_MAX = 99.46537030219284
SHARPEN_RADIUS = 0.38717865927737405
SMOOTH_RADIUS = 3.410555347410367

MODEL_ID = "facebook/sam3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(image: np.ndarray) -> np.ndarray:
    """Match SynthMT's selected grayscale, Cellpose, percentile preprocessing."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    image = smooth_sharpen_img(
        image, sharpen_radius=SHARPEN_RADIUS, smooth_radius=SMOOTH_RADIUS
    )
    for channel in range(image.shape[-1]):
        plane = image[..., channel]
        lo, hi = np.percentile(plane, (PERCENTILE_MIN, PERCENTILE_MAX))
        if hi > lo:
            image[..., channel] = np.clip((plane - lo) / (hi - lo), 0, 1)
        else:
            image[..., channel] = 0
    return (image * 255).astype(np.uint8)


def save_results(original: np.ndarray, masks: np.ndarray) -> None:
    """Save every instance, non-overlapping ID TIFF, and readable overlay."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(original.shape[:2], dtype=np.uint16)
    overlay = original.astype(np.float32).copy()
    colors = np.random.default_rng(0).integers(40, 256, (len(masks), 3))

    for index, (mask, color) in enumerate(zip(masks, colors), start=1):
        mask = np.asarray(mask, dtype=bool)
        labels[mask] = index
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            OUTPUT_DIR / f"instance_{index:03d}.png"
        )

    Image.fromarray(labels).save(OUTPUT_DIR / "instances.tiff")
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        OUTPUT_DIR / "overlay.png"
    )


def as_numpy_masks(mask_list: object, image_shape: tuple[int, int]) -> np.ndarray:
    masks = [
        np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask, dtype=bool)
        for mask in mask_list
    ]
    return (
        np.stack(masks)
        if masks
        else np.empty((0, *image_shape), dtype=bool)
    )


def self_check() -> None:
    assert as_numpy_masks([], (2, 3)).shape == (0, 2, 3)


def main() -> None:
    self_check()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Set INPUT_PATH to an existing image: {INPUT_PATH}")

    original = np.asarray(Image.open(INPUT_PATH).convert("RGB"))
    image = preprocess(original)
    # AutoProcessor now selects SAM3's video wrapper. SynthMT uses image SAM3.
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = Sam3Model.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE).eval()

    inputs = processor(
        images=Image.fromarray(image), text=TEXT_PROMPT, return_tensors="pt"
    ).to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    masks = as_numpy_masks(result.get("masks", []), original.shape[:2])
    save_results(original, masks)
    print(f"{len(masks)} instances -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
