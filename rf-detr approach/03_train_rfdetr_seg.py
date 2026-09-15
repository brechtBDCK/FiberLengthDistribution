#!/usr/bin/env python3
"""Train an RF-DETR INSTANCE segmentation model on the COCO dataset.

Recommended start for this fiber task:
    python 03_train_rfdetr_seg.py --dataset fiber_dataset --output runs/fibers

RFDETRSegMedium and RFDETRSegLarge natively use 200 queries/select slots in
current RF-DETR configs, so they can produce up to 200 instances per tile.
"""

from __future__ import annotations

import argparse

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


def parse_batch(s: str):
    return "auto" if s.lower() == "auto" else int(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset root containing train/valid/test")
    ap.add_argument("--output", default="runs/rfdetr_fibers")
    ap.add_argument("--model", choices=MODELS, default="medium")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=parse_batch, default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queries", type=int, default=200, help="Max instance slots/predictions per image")
    args = ap.parse_args()

    model_cls = MODELS[args.model]
    model = model_cls(num_queries=args.queries, num_select=args.queries)

    model.train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        eval_max_dets=args.queries,
        use_ema=True,
        run_test=False,
        tensorboard=True,
        seed=42,
    )

    print("Training finished.")
    print(f"Look in {args.output} for the best checkpoint (commonly checkpoint_best_total.pth).")


if __name__ == "__main__":
    main()
