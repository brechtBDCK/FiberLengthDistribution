#!/usr/bin/env python3
"""Tiled full-image RF-DETR instance-segmentation inference for fibers.

Workflow:
  1. Tile a large microscopy image with overlap.
  2. Run RF-DETR instance segmentation on each tile.
  3. Translate masks back to full-image coordinates.
  4. Merge likely duplicate instances from overlapping tiles.
  5. Save an annotated full image, per-fiber CSV, COCO-like polygon JSON,
     and length/orientation histograms.

Length is estimated as the extent of the mask along its PCA major axis. This is
very good for approximately straight fibers. For strongly curved fibers, replace
this later with skeleton/geodesic length.

Example:
    python 06_infer_full_image.py \
        --weights runs/fibers_v2/checkpoint_best_total.pth \
        --image raw_images/example.png \
        --output-dir full_prediction \
        --model medium \
        --tile-size 512 \
        --overlap 64 \
        --threshold 0.20 \
        --queries 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rfdetr import (
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
)

MODELS = {
    "nano": RFDETRSegNano,
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge,
    "xlarge": RFDETRSegXLarge,
    "2xlarge": RFDETRSeg2XLarge,
}


@dataclass
class Instance:
    # mask is stored only inside bbox: [y0:y1, x0:x1]
    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    score: float
    tile_count: int = 1

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


def positions(length: int, tile: int, overlap: int) -> list[int]:
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


def crop_instance(global_x: int, global_y: int, mask: np.ndarray, score: float) -> Instance | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    lx0, lx1 = int(xs.min()), int(xs.max()) + 1
    ly0, ly1 = int(ys.min()), int(ys.max()) + 1
    cropped = mask[ly0:ly1, lx0:lx1].astype(bool, copy=True)
    return Instance(global_x + lx0, global_y + ly0,
                    global_x + lx1, global_y + ly1,
                    cropped, float(score))


def instance_orientation_deg(inst: Instance) -> float:
    ys, xs = np.where(inst.mask)
    if len(xs) < 2:
        return 0.0
    pts = np.column_stack((xs.astype(float), ys.astype(float)))
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    # Image y points downward, so negate y to report conventional CCW angle.
    return float(math.degrees(math.atan2(-v[1], v[0])) % 180.0)


def axial_angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def overlap_stats(a: Instance, b: Instance) -> tuple[int, float, float]:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0, 0.0, 0.0

    am = a.mask[iy0-a.y0:iy1-a.y0, ix0-a.x0:ix1-a.x0]
    bm = b.mask[iy0-b.y0:iy1-b.y0, ix0-b.x0:ix1-b.x0]
    inter = int(np.count_nonzero(am & bm))
    if inter == 0:
        return 0, 0.0, 0.0
    aa, ba = a.area, b.area
    union = aa + ba - inter
    iou = inter / max(1, union)
    ios = inter / max(1, min(aa, ba))  # intersection over smaller instance
    return inter, iou, ios


def should_merge(a: Instance, b: Instance, merge_iou: float,
                 merge_ios: float, max_angle_diff: float, min_intersection: int) -> bool:
    inter, iou, ios = overlap_stats(a, b)
    if inter < min_intersection:
        return False
    if iou >= merge_iou:
        return True
    if ios < merge_ios:
        return False
    return axial_angle_diff(instance_orientation_deg(a), instance_orientation_deg(b)) <= max_angle_diff


def union_instances(group: list[Instance]) -> Instance:
    x0 = min(i.x0 for i in group)
    y0 = min(i.y0 for i in group)
    x1 = max(i.x1 for i in group)
    y1 = max(i.y1 for i in group)
    out = np.zeros((y1-y0, x1-x0), dtype=bool)
    for inst in group:
        out[inst.y0-y0:inst.y1-y0, inst.x0-x0:inst.x1-x0] |= inst.mask
    return Instance(x0, y0, x1, y1, out,
                    score=max(i.score for i in group),
                    tile_count=sum(i.tile_count for i in group))


def merge_duplicates(instances: list[Instance], merge_iou: float,
                     merge_ios: float, max_angle_diff: float,
                     min_intersection: int) -> list[Instance]:
    n = len(instances)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise is fine for the usual few hundred candidate instances per image.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = instances[i], instances[j]
            if a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0:
                continue
            if should_merge(a, b, merge_iou, merge_ios, max_angle_diff, min_intersection):
                union(i, j)

    groups: dict[int, list[Instance]] = {}
    for i, inst in enumerate(instances):
        groups.setdefault(find(i), []).append(inst)
    return [union_instances(g) for g in groups.values()]


def fiber_measurements(inst: Instance, full_w: int, full_h: int,
                       border_margin: int = 1) -> dict:
    ys, xs = np.where(inst.mask)
    gx = xs.astype(float) + inst.x0
    gy = ys.astype(float) + inst.y0
    if len(gx) < 2:
        angle = 0.0
        length = 0.0
        cx = float(gx[0]) if len(gx) else 0.0
        cy = float(gy[0]) if len(gy) else 0.0
    else:
        pts = np.column_stack((gx, gy))
        center = pts.mean(axis=0)
        centered = pts - center
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmax(vals))]
        proj = centered @ v
        length = float(proj.max() - proj.min() + 1.0)
        angle = float(math.degrees(math.atan2(-v[1], v[0])) % 180.0)
        cx, cy = float(center[0]), float(center[1])

    touches = (
        inst.x0 <= border_margin or inst.y0 <= border_margin or
        inst.x1 >= full_w - border_margin or inst.y1 >= full_h - border_margin
    )
    return {
        "length_px": length,
        "orientation_deg": angle,
        "center_x": cx,
        "center_y": cy,
        "area_px": inst.area,
        "touches_image_border": touches,
    }


def instance_polygons(inst: Instance) -> list[list[float]]:
    u8 = inst.mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(float)
        pts[:, 0] += inst.x0
        pts[:, 1] += inst.y0
        flat = pts.reshape(-1).tolist()
        if len(flat) >= 6:
            polys.append(flat)
    return polys


def save_overlay(image_bgr: np.ndarray, instances: list[Instance], path: Path) -> None:
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    for idx, inst in enumerate(instances, start=1):
        # Deterministic per-instance BGR color.
        color = ((37*idx + 80) % 255, (97*idx + 130) % 255, (173*idx + 40) % 255)
        roi = overlay[inst.y0:inst.y1, inst.x0:inst.x1]
        roi[inst.mask] = color
    canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0.0)

    for idx, inst in enumerate(instances, start=1):
        cv2.rectangle(canvas, (inst.x0, inst.y0), (inst.x1-1, inst.y1-1), (255, 255, 255), 1)
        cv2.putText(canvas, str(idx), (inst.x0, max(12, inst.y0 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_histograms(rows: list[dict], output_dir: Path, um_per_px: float | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping histograms (pip install matplotlib)")
        return

    lengths = [r["length_um"] if um_per_px is not None else r["length_px"] for r in rows
               if not r["touches_image_border"]]
    orientations = [r["orientation_deg"] for r in rows]

    if lengths:
        plt.figure(figsize=(7, 4))
        plt.hist(lengths, bins=30)
        plt.xlabel("Fiber length (µm)" if um_per_px is not None else "Fiber length (px)")
        plt.ylabel("Count")
        plt.title("Fiber length distribution (border-touching fibers excluded)")
        plt.tight_layout()
        plt.savefig(output_dir / "length_histogram.png", dpi=160)
        plt.close()

    if orientations:
        plt.figure(figsize=(7, 4))
        plt.hist(orientations, bins=np.arange(0, 185, 5))
        plt.xlabel("Orientation (degrees, 0–180)")
        plt.ylabel("Count")
        plt.title("Fiber orientation distribution")
        plt.xlim(0, 180)
        plt.tight_layout()
        plt.savefig(output_dir / "orientation_histogram.png", dpi=160)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("full_prediction"))
    ap.add_argument("--model", choices=MODELS, default="medium")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--merge-iou", type=float, default=0.15,
                    help="Merge duplicate tile predictions above this mask IoU")
    ap.add_argument("--merge-ios", type=float, default=0.08,
                    help="Or merge if intersection/smaller-area exceeds this and orientation agrees")
    ap.add_argument("--merge-angle", type=float, default=15.0,
                    help="Max axial angle difference (degrees) for low-IoU duplicate merge")
    ap.add_argument("--min-intersection", type=int, default=8,
                    help="Minimum overlapping mask pixels before two instances can merge")
    ap.add_argument("--min-area", type=int, default=8,
                    help="Discard predicted masks smaller than this many pixels")
    ap.add_argument("--microns-per-pixel", type=float, default=None,
                    help="Optional calibration; adds length_um to CSV")
    args = ap.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    h, w = image.shape[:2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = MODELS[args.model](
        pretrain_weights=str(args.weights),
        num_classes=1,
        num_queries=args.queries,
        num_select=args.queries,
    )

    xs = positions(w, args.tile_size, args.overlap)
    ys = positions(h, args.tile_size, args.overlap)
    candidates: list[Instance] = []
    tile_n = 0

    for y in ys:
        for x in xs:
            tile_n += 1
            x2, y2 = min(x + args.tile_size, w), min(y + args.tile_size, h)
            tile_bgr = image[y:y2, x:x2]
            valid_h, valid_w = tile_bgr.shape[:2]

            if (valid_w, valid_h) != (args.tile_size, args.tile_size):
                padded = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.uint8)
                padded[:valid_h, :valid_w] = tile_bgr
                tile_bgr = padded

            # RF-DETR expects numpy-array images in RGB order.
            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            det = model.predict(tile_rgb, threshold=args.threshold)
            masks = det.mask
            scores = det.confidence

            if masks is None:
                print(f"tile {tile_n}/{len(xs)*len(ys)} @ ({x},{y}): 0")
                continue

            kept = 0
            for k in range(len(det)):
                mask = np.asarray(masks[k], dtype=bool)
                mask = mask[:valid_h, :valid_w]
                if np.count_nonzero(mask) < args.min_area:
                    continue
                score = float(scores[k]) if scores is not None else 1.0
                inst = crop_instance(x, y, mask, score)
                if inst is not None:
                    candidates.append(inst)
                    kept += 1
            print(f"tile {tile_n}/{len(xs)*len(ys)} @ ({x},{y}): {kept}")

    print(f"Raw tile instances: {len(candidates)}")
    merged = merge_duplicates(candidates, args.merge_iou, args.merge_ios,
                              args.merge_angle, args.min_intersection)
    print(f"Merged full-image instances: {len(merged)}")

    # Stable order: top-to-bottom, then left-to-right.
    merged.sort(key=lambda i: (i.y0, i.x0))

    rows = []
    annotations = []
    for idx, inst in enumerate(merged, start=1):
        m = fiber_measurements(inst, w, h)
        row = {
            "fiber_id": idx,
            "confidence": inst.score,
            "tile_predictions_merged": inst.tile_count,
            "bbox_x": inst.x0,
            "bbox_y": inst.y0,
            "bbox_width": inst.x1 - inst.x0,
            "bbox_height": inst.y1 - inst.y0,
            **m,
        }
        if args.microns_per_pixel is not None:
            row["length_um"] = m["length_px"] * args.microns_per_pixel
        rows.append(row)
        annotations.append({
            "id": idx,
            "category_id": 1,
            "score": inst.score,
            "bbox": [inst.x0, inst.y0, inst.x1-inst.x0, inst.y1-inst.y0],
            "area": inst.area,
            "segmentation": instance_polygons(inst),
            "length_px": m["length_px"],
            "orientation_deg": m["orientation_deg"],
            "touches_image_border": m["touches_image_border"],
        })

    csv_path = args.output_dir / "fibers.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("fiber_id\n", encoding="utf-8")

    json_path = args.output_dir / "instances.json"
    json_path.write_text(json.dumps({
        "image": str(args.image),
        "width": w,
        "height": h,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "threshold": args.threshold,
        "categories": [{"id": 1, "name": "fiber"}],
        "annotations": annotations,
    }, indent=2), encoding="utf-8")

    overlay_path = args.output_dir / "prediction_full.png"
    save_overlay(image, merged, overlay_path)
    save_histograms(rows, args.output_dir, args.microns_per_pixel)

    print("\nSaved:")
    print(" ", overlay_path)
    print(" ", csv_path)
    print(" ", json_path)
    if args.microns_per_pixel is not None:
        print(f"Calibration: {args.microns_per_pixel} µm/pixel")


if __name__ == "__main__":
    main()
