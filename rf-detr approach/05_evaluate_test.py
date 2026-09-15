#!/usr/bin/env python3
"""Evaluate a fine-tuned RF-DETR instance-segmentation checkpoint on a held-out split.

This uses RF-DETR's built-in COCO evaluator, so the printed metrics match the
validation metrics you see during training (bbox mAP/mAR + segmentation mAP).

Example:
    python 05_evaluate_test.py \
        --dataset fiber_dataset \
        --weights runs/fibers_v2/checkpoint_best_total.pth \
        --model medium \
        --split test \
        --batch-size 1 \
        --queries 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Dataset root containing train/valid/test")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--model", choices=MODELS, default="medium")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="Use 'test' for final held-out evaluation; 'val' for validation")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--output", type=Path, default=Path("test_metrics.json"))
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.weights.exists():
        raise SystemExit(f"Checkpoint not found: {args.weights}")

    model = MODELS[args.model](
        pretrain_weights=str(args.weights),
        num_classes=1,
        num_queries=args.queries,
        num_select=args.queries,
    )

    metrics = model.evaluate(
        dataset_dir=str(args.dataset),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.workers,
        eval_max_dets=args.queries,
    )

    # Make values JSON serializable (some releases return numpy/torch scalars).
    clean = {}
    for k, v in metrics.items():
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            clean[k] = str(v)  #

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    print("\nSaved metrics to:", args.output)
    for k, v in clean.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
