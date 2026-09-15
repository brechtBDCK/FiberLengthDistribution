#!/usr/bin/env python3
"""Run RF-DETR instance-segmentation inference on one image/tile and save overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import supervision as sv
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--output", default="prediction.png", type=Path)
    ap.add_argument("--model", choices=MODELS, default="medium")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")

    model = MODELS[args.model](
        pretrain_weights=args.weights,
        num_queries=args.queries,
        num_select=args.queries,
    )

    detections = model.predict(image, threshold=args.threshold)

    annotated = sv.MaskAnnotator(opacity=0.45).annotate(image.copy(), detections)
    annotated = sv.BoxAnnotator(thickness=1).annotate(annotated, detections)

    labels = []
    for i, conf in enumerate(detections.confidence if detections.confidence is not None else []):
        labels.append(f"fiber {i+1} {conf:.2f}")
    if labels:
        annotated = sv.LabelAnnotator(text_scale=0.4, text_thickness=1).annotate(annotated, detections, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)
    print(f"instances: {len(detections)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
