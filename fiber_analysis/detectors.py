from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.filters import apply_hysteresis_threshold, sato


@dataclass
class Detection:
    mask: np.ndarray
    response: np.ndarray
    debug: dict[str, np.ndarray]


def _clean(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = morphology.closing(mask, morphology.disk(1))
    return morphology.remove_small_objects(mask, max_size=min_area - 1)


def adaptive_skeleton(gray: np.ndarray, cfg: dict) -> Detection:
    c = cfg["adaptive"]
    background = gaussian_filter(gray, c["background_sigma"])
    residual = gray - background
    local_std = np.sqrt(gaussian_filter(residual**2, c["contrast_sigma"])) + 1e-4
    z = residual / local_std
    mask = _clean((z > c["z_threshold"]) | (z < -c["z_threshold"]), cfg["postprocess"]["min_component_area"])
    return Detection(mask, np.abs(z), {"background": background, "local_z": z})


def multiscale_ridge(gray: np.ndarray, cfg: dict) -> Detection:
    c = cfg["ridge"]
    sigmas = c["sigmas"]
    bright = sato(gray, sigmas=sigmas, black_ridges=False)
    dark = sato(gray, sigmas=sigmas, black_ridges=True)
    response = np.maximum(bright, dark)
    lo, hi = np.percentile(response, [c["low_percentile"], c["high_percentile"]])
    mask = _clean(apply_hysteresis_threshold(response, lo, hi), cfg["postprocess"]["min_component_area"])
    return Detection(mask, response, {"bright_ridge": bright, "dark_ridge": dark})


def _segment_angle(s: np.ndarray) -> float:
    return np.arctan2(s[3] - s[1], s[2] - s[0]) % np.pi


def _merge_segments(lines: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """Merge only near-collinear nearby LSD segments; preserves parallel fibers."""
    c = cfg["lines"]
    merged: list[np.ndarray] = []
    for line in sorted(lines, key=lambda x: -np.hypot(x[2] - x[0], x[3] - x[1])):
        angle = _segment_angle(line)
        for i, old in enumerate(merged):
            delta = abs((angle - _segment_angle(old) + np.pi / 2) % np.pi - np.pi / 2)
            endpoints = np.array([[line[0], line[1]], [line[2], line[3]]])
            old_endpoints = np.array([[old[0], old[1]], [old[2], old[3]]])
            gap = np.min(np.linalg.norm(endpoints[:, None] - old_endpoints[None, :], axis=2))
            mid_distance = abs(np.cross(old_endpoints[1] - old_endpoints[0], endpoints.mean(0) - old_endpoints[0])) / (np.linalg.norm(old_endpoints[1] - old_endpoints[0]) + 1e-6)
            if np.degrees(delta) <= c["angle_degrees"] and gap <= c["gap"] and mid_distance <= c["distance"]:
                points = np.vstack([endpoints, old_endpoints])
                direction = np.array([np.cos(_segment_angle(old)), np.sin(_segment_angle(old))])
                projection = points @ direction
                merged[i] = np.r_[points[projection.argmin()], points[projection.argmax()]]
                break
        else:
            merged.append(line)
    return merged


def line_segments(gray: np.ndarray, cfg: dict) -> Detection:
    image = np.uint8(np.clip(gray * 255, 0, 255))
    found = cv2.createLineSegmentDetector().detect(image)[0]
    raw = [] if found is None else np.asarray(found).reshape(-1, 4)
    lines = [x for x in raw if np.hypot(x[2]-x[0], x[3]-x[1]) >= cfg["lines"]["min_length"]]
    lines = _merge_segments(lines, cfg)
    mask = np.zeros_like(image, dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (round(x1), round(y1)), (round(x2), round(y2)), 255, 1, cv2.LINE_AA)
    mask = _clean(mask > 0, cfg["postprocess"]["min_component_area"])
    return Detection(mask, mask.astype(float), {"segments": mask})
