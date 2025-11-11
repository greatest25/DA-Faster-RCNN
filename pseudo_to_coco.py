#!/usr/bin/env python
"""Convert pseudo label JSON to COCO detection format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

CITYSCAPES_CLASSES = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pseudo label JSON to COCO format")
    parser.add_argument("--pseudo-json", required=True, help="Input pseudo label JSON produced by generate_pseudo_labels.py")
    parser.add_argument("--output", required=True, help="Output COCO json path")
    parser.add_argument("--image-root", default="/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN/datasets/cityscape/VOC2007/JPEGImages", help="Root directory of images; used to create relative paths")
    parser.add_argument("--min-annotations", type=int, default=0, help="Minimum number of annotations to keep an image after filtering")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Drop detections below this score")
    parser.add_argument("--max-per-image", type=int, default=0, help="Keep at most this many detections per image (highest scores first)")
    return parser.parse_args()


def make_categories() -> List[Dict[str, object]]:
    return [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(CITYSCAPES_CLASSES)
    ]


def main() -> None:
    args = parse_args()

    pseudo_path = Path(args.pseudo_json)
    if not pseudo_path.is_file():
        raise FileNotFoundError(f"Pseudo label json not found: {pseudo_path}")

    with pseudo_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []

    next_ann_id = 1
    root = Path(args.image_root)

    for idx, record in enumerate(results):
        anns = record.get("annotations", []) or []
        filtered = []
        for ann in anns:
            score = float(ann.get("score", 0.0))
            if score < args.score_threshold:
                continue
            filtered.append(ann)

        if args.max_per_image and args.max_per_image > 0 and filtered:
            filtered.sort(key=lambda a: float(a.get("score", 0.0)), reverse=True)
            filtered = filtered[: args.max_per_image]

        if len(filtered) < args.min_annotations:
            continue

        img_id = idx + 1
        file_name = record.get("file_name")
        if file_name:
            try:
                rel_path = Path(file_name).resolve().relative_to(root.resolve())
                file_name = str(rel_path).replace(os.sep, "/")
            except Exception:
                file_name = str(file_name)
        else:
            raise ValueError(f"Entry missing file_name: {record}")

        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for ann in filtered:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            if w <= 0 or h <= 0:
                continue

            cat_idx = int(ann.get("category_id", 0))
            category_id = cat_idx + 1
            annotations.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "score": float(ann.get("score", 0.0)),
            })
            next_ann_id += 1

    coco_payload = {
        "info": {
            "description": f"Pseudo labels converted from {pseudo_path.name}",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": make_categories(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco_payload, f)
    print(f"COCO annotation saved to {output_path}. images={len(images)}, annotations={len(annotations)}")


if __name__ == "__main__":
    main()
