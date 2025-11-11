#!/usr/bin/env python
"""Generate pseudo labels on target-domain images using a trained teacher model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import sys

D2_PATH = "/mnt/lyh/DA-FasterCNN/detectron2-main"
if D2_PATH not in sys.path:
    sys.path.insert(0, D2_PATH)

import cv2  # type: ignore
import torch
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog

from register_cityscapes import register_city_datasets

setup_logger()

CITYSCAPES_CLASSES = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def build_cfg(args: argparse.Namespace) -> Any:
    cfg = get_cfg()
    config_file = args.config_file
    if config_file:
        if (not os.path.isabs(config_file)
                and not os.path.exists(config_file)
                and ("/" in config_file or "\\" in config_file)):
            try:
                config_file = model_zoo.get_config_file(config_file)
            except Exception as exc:
                print(f"[warn] model_zoo.get_config_file failed: {exc}; using raw path {args.config_file}")
        cfg.merge_from_file(config_file)
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.DATASETS.TRAIN = ("city_trainS",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 100
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 5000
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CITYSCAPES_CLASSES)
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    default_source_weights = "/mnt/lyh/DA-FasterCNN/weights/COCO-Detection/faster_rcnn_R_50_FPN_1x/model_final_b275ba.pkl"
    if not cfg.MODEL.WEIGHTS or "FPN" not in str(cfg.MODEL.WEIGHTS):
        if os.path.isfile(default_source_weights):
            cfg.MODEL.WEIGHTS = default_source_weights
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    cfg.MODEL.DEVICE = args.device
    return cfg


class SimplePredictor:
    """Lightweight predictor mirroring detectron2 DefaultPredictor with custom weight loading."""

    def __init__(self, cfg: Any, weights: str) -> None:
        if weights and not os.path.isfile(weights):
            raise FileNotFoundError(f"Weights file not found: {weights}")

        self.cfg = cfg.clone()
        self.cfg.freeze()

        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.to(self.cfg.MODEL.DEVICE)

        DetectionCheckpointer(self.model).load(weights)

        self.input_format = self.cfg.INPUT.FORMAT
        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.aug = T.ResizeShortestEdge(min_size, max_size)

    @torch.no_grad()
    def __call__(self, original_image):
        image = original_image
        if self.input_format == "RGB":
            image = image[:, :, ::-1]
        height, width = image.shape[:2]
        image = self.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1), device=self.cfg.MODEL.DEVICE)

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        if isinstance(predictions, dict) and "instances" in predictions:
            predictions = predictions["instances"]
        return predictions


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pseudo labels for target-domain images")
    parser.add_argument("--config-file", default=None, help="Config file (optional). Model Zoo names allowed.")
    parser.add_argument("--weights", required=True, help="Teacher checkpoint (pth/pkl)")
    parser.add_argument("--dataset", default="city_trainT", help="Target dataset name")
    parser.add_argument("--output", default="pseudo_labels/demo_city_trainT.json", help="Output JSON path")
    parser.add_argument("--output-dir", default="./output/pseudo_labeling", help="Auxiliary output dir")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="Score threshold for pseudo labels")
    parser.add_argument("--max-images", type=int, default=50, help="Number of images to process (<=0 for all)")
    return parser


def to_serializable(instances, metadata, score_threshold: float) -> List[Dict[str, Any]]:
    if instances is None:
        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    class_names = metadata.get("thing_classes", CITYSCAPES_CLASSES)

    outputs: List[Dict[str, Any]] = []
    for box, score, cls_idx in zip(boxes, scores, classes):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        outputs.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "category_id": int(cls_idx),
            "category_name": class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx),
            "score": float(score),
        })
    return outputs


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    data_format = register_city_datasets()
    print(f"Registered datasets in {data_format} mode. Generating pseudo labels for {args.dataset}.")

    cfg = build_cfg(args)
    predictor = SimplePredictor(cfg, args.weights)

    dataset_dicts = DatasetCatalog.get(args.dataset)
    metadata = MetadataCatalog.get(args.dataset)

    results: List[Dict[str, Any]] = []
    max_images = args.max_images if args.max_images > 0 else len(dataset_dicts)

    for idx, record in enumerate(dataset_dicts):
        if idx >= max_images:
            break

        image = cv2.imread(record["file_name"])
        if image is None:
            print(f"[warn] cannot read image {record['file_name']}, skip")
            continue

        instances = predictor(image)
        annotations = to_serializable(instances, metadata, args.score_threshold)
        results.append({
            "image_id": record.get("image_id", idx),
            "file_name": record["file_name"],
            "width": int(record.get("width", image.shape[1])),
            "height": int(record.get("height", image.shape[0])),
            "annotations": annotations,
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{max_images} images...")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "score_threshold": args.score_threshold,
            "results": results,
        }, f, indent=2)

    print(f"Pseudo labels saved to {args.output}. Images processed: {len(results)}")


if __name__ == "__main__":
    main()
