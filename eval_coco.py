#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR Paper Results Verification - COCO Evaluation Script
# ------------------------------------------------------------------------
"""
Evaluate pretrained RF-DETR models on COCO val2017 and compare against
the paper's reported benchmark numbers.

Paper: RF-DETR: Neural Architecture Search for Real-Time Detection Transformers
arXiv: https://arxiv.org/abs/2511.09554

Usage:
    # Quick smoke test (100 images, ~30 seconds):
    python eval_coco.py --model nano --coco-dir E:/coco-2017 --max-images 100

    # Full evaluation - Nano model (5000 images, ~45 min CPU):
    python eval_coco.py --model nano --coco-dir E:/coco-2017

    # Full evaluation - Medium model (5000 images, ~90 min CPU):
    python eval_coco.py --model medium --coco-dir E:/coco-2017

    # All detection models (N/S/M/L):
    python eval_coco.py --model all --coco-dir E:/coco-2017
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paper benchmark results (from Table in README / paper)
# ---------------------------------------------------------------------------
PAPER_BENCHMARKS = {
    "nano":  {"AP50": 67.6, "AP": 48.4, "resolution": 384, "params_m": 30.5},
    "small": {"AP50": 72.1, "AP": 53.0, "resolution": 512, "params_m": 32.1},
    "medium":{"AP50": 73.6, "AP": 54.7, "resolution": 576, "params_m": 33.7},
    "large": {"AP50": 75.1, "AP": 56.5, "resolution": 704, "params_m": 33.9},
}

MODEL_MAP = {
    "nano":   ("RFDETRNano",   384),
    "small":  ("RFDETRSmall",  512),
    "medium": ("RFDETRMedium", 576),
    "large":  ("RFDETRLarge",  704),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_size: str):
    """Load the specified RF-DETR model variant."""
    class_name, _ = MODEL_MAP[model_size]
    import rfdetr
    model_cls = getattr(rfdetr, class_name)
    print(f"Loading {class_name}...")
    return model_cls()


def run_evaluation(model, coco_gt: COCO, coco_dir: str,
                   max_images: int | None = None):
    """Run COCO evaluation on val2017.

    Args:
        model: RFDETR model instance.
        coco_gt: COCO ground truth object.
        coco_dir: Path to COCO dataset root.
        max_images: Limit to first N images for quick testing (None = all).

    Returns:
        dict with AP metrics (fractions 0-1), or None if no detections.
    """
    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    print(f"Evaluating on {len(img_ids)} images...")

    results = []
    inference_times = []
    val_dir = os.path.join(coco_dir, "val2017")

    for img_id in tqdm(img_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"  Warning: {img_path} not found, skipping")
            continue

        image = Image.open(img_path).convert("RGB")

        t_start = time.time()
        try:
            detections = model.predict(image, threshold=0.01)
        except Exception as e:
            print(f"  Error on {img_info['file_name']}: {e}")
            continue
        t_end = time.time()
        inference_times.append(t_end - t_start)

        if detections is not None and len(detections) > 0:
            boxes_xyxy = detections.xyxy
            scores = detections.confidence
            class_ids = detections.class_id

            for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1

                # RF-DETR COCO-pretrained models output raw COCO category IDs
                # (1-90), so use them directly. For fine-tuned models with
                # 0-indexed labels, add 1.
                cat_id = int(class_id)
                if cat_id < 1 or cat_id > 90:
                    cat_id = cat_id + 1  # fallback: 0-indexed -> 1-indexed

                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

    avg_time = np.mean(inference_times) if inference_times else 0
    total_time = sum(inference_times)
    n_images = len(inference_times)

    print(f"\nInference complete:")
    print(f"  Images processed: {n_images}")
    print(f"  Detections: {len(results)}")
    print(f"  Avg time/image: {avg_time:.3f}s")
    print(f"  Total inference: {total_time:.0f}s ({total_time/60:.1f} min)")

    if not results:
        print("No detections found. Check model and dataset.")
        return None

    # Save results to temp file for pycocotools
    results_file = "coco_results_temp.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    coco_dt = coco_gt.loadRes(results_file)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "AP50:95": float(stats[0]),
        "AP50":    float(stats[1]),
        "AP75":    float(stats[2]),
        "AP_small":   float(stats[3]),
        "AP_medium":  float(stats[4]),
        "AP_large":   float(stats[5]),
        "AR_max1":    float(stats[6]),
        "AR_max10":   float(stats[7]),
        "AR_max100":  float(stats[8]),
        "AR_small":   float(stats[9]),
        "AR_medium":  float(stats[10]),
        "AR_large":   float(stats[11]),
        "n_images":   n_images,
        "avg_time_s": avg_time,
    }

    os.remove(results_file)
    return metrics


def print_comparison(model_size: str, metrics: dict):
    """Print side-by-side comparison with paper results."""
    paper = PAPER_BENCHMARKS.get(model_size)
    if paper is None:
        return

    print(f"\n{'=' * 70}")
    print(f"  RF-DETR-{model_size.upper()}  |  Paper vs. Reproduced")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20} {'Paper':>10} {'Ours':>10} {'Diff':>10}")
    print(f"  {'-' * 50}")

    for metric_name, paper_val in [("AP", paper["AP"]), ("AP50", paper["AP50"])]:
        our_val = metrics.get("AP50:95" if metric_name == "AP" else metric_name, -1)
        if our_val < 0:
            continue
        our_val_pct = our_val * 100  # pycocotools returns 0-1 fraction
        diff = our_val_pct - paper_val
        diff_str = f"{diff:+.1f}"
        if abs(diff) < 1.0:
            diff_str += "  <-- match"
        elif abs(diff) < 2.0:
            diff_str += "  (close)"
        print(f"  {metric_name:<20} {paper_val:>9.1f}% {our_val_pct:>9.1f}% {diff_str:>12}")

    print(f"  {'-' * 50}")
    print(f"  Resolution: {paper['resolution']}x{paper['resolution']}")
    print(f"  Parameters: {paper['params_m']}M")
    print(f"  Images evaluated: {metrics.get('n_images', '?')}")
    print(f"  Avg inference: {metrics.get('avg_time_s', 0):.3f}s/image")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RF-DETR on COCO val2017"
    )
    parser.add_argument(
        "--model", type=str, default="medium",
        choices=["nano", "small", "medium", "large", "all"],
        help="Model size to evaluate (default: medium)"
    )
    parser.add_argument(
        "--coco-dir", type=str, default=r"E:\coco-2017",
        help="Path to COCO dataset root (must contain annotations/ and val2017/)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit to first N images (e.g. 100 for quick test; omit for full eval)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON file to save metrics (e.g. results_nano.json)"
    )
    args = parser.parse_args()

    # --- Validate COCO dataset ---
    ann_file = os.path.join(args.coco_dir, "annotations", "instances_val2017.json")
    val_dir  = os.path.join(args.coco_dir, "val2017")

    if not os.path.exists(ann_file):
        print(f"ERROR: COCO annotations not found at {ann_file}")
        print("Expected structure:")
        print(f"  {args.coco_dir}/annotations/instances_val2017.json")
        print(f"  {args.coco_dir}/val2017/*.jpg")
        sys.exit(1)
    if not os.path.isdir(val_dir):
        print(f"ERROR: val2017 directory not found at {val_dir}")
        sys.exit(1)

    print(f"COCO annotations: {ann_file}")
    print(f"COCO val images:  {val_dir}")
    coco_gt = COCO(ann_file)

    # --- Determine which models to evaluate ---
    models_to_run = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    all_metrics = {}

    for model_size in models_to_run:
        print(f"\n{'#' * 60}")
        print(f"#  Evaluating RF-DETR-{model_size.upper()}")
        
        print(f"{'#' * 60}")

        model = load_model(model_size)
        metrics = run_evaluation(model, coco_gt, args.coco_dir,
                                 max_images=args.max_images)

        if metrics:
            all_metrics[model_size] = metrics
            print_comparison(model_size, metrics)

    # --- Save results ---
    if all_metrics and args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "coco_dir": args.coco_dir,
            "max_images": args.max_images,
            "models": all_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
