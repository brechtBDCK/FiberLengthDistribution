from pathlib import Path

import numpy as np
from PIL import Image

from . import detectors
from .measure import measure
from .report import write_result
from .unet import unet_detect


def _gray(path: Path, max_pixels: int | None, edge_crop: float) -> tuple[np.ndarray, float, tuple[float, float]]:
    with Image.open(path) as image:
        scale = 1.0
        if max_pixels and image.width * image.height > max_pixels:
            scale = (max_pixels / (image.width * image.height)) ** .5
            image.thumbnail((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        crop_y, crop_x = round(image.height * edge_crop), round(image.width * edge_crop)
        if crop_y or crop_x:
            image = image.crop((crop_x, crop_y, image.width - crop_x, image.height - crop_y))
        image = image.convert("L")
        return np.asarray(image, dtype=np.float32) / 255, scale, (crop_y / scale, crop_x / scale)


def analyze_image(path: Path, output: Path, methods: list[str], cfg: dict, units_per_pixel: float | None, unit_name: str, debug: bool) -> dict:
    gray, scale, origin = _gray(path, cfg.get("max_analysis_pixels"), cfg.get("edge_crop_fraction", .01)); result = {}
    available = {"adaptive_skeleton": detectors.adaptive_skeleton, "multiscale_ridge": detectors.multiscale_ridge, "line_segments": detectors.line_segments}
    for method in methods:
        print(f"{path.name}: {method}...", flush=True)
        if method == "unet":
            detection, reason = unet_detect(gray, cfg)
            if detection is None: result[method] = {"skipped": reason}; continue
        elif method in available: detection = available[method](gray, cfg)
        else: result[method] = {"skipped": "unknown method"}; continue
        fibers, skeleton, graph_debug = measure(detection.mask, detection.response, cfg, units_per_pixel, 1 / scale, origin)
        result[method] = write_result(output, path.stem, method, fibers, {**detection.debug, "mask": detection.mask, "skeleton": skeleton, **graph_debug}, debug, unit_name)
    return result


def images(path: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
    return [p for p in path.rglob("*") if p.suffix.lower() in suffixes] if path.is_dir() else [path]
