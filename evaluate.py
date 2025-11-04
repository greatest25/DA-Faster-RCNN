# evaluate.py
import argparse
import os
from pathlib import Path

import sys
d2_path = '/mnt/lyh/DA-FasterCNN/detectron2-main'
if d2_path not in sys.path:
    sys.path.insert(0, d2_path)

from register_cityscapes import register_city_datasets

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, inference_on_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--config-file", default=None, help="Optional: config file to load (YAML). If omitted, uses COCO R50-FPN default.")
    parser.add_argument("--weights", default=None, help="Path to model weights (pth / pkl). If omitted, uses cfg.MODEL.WEIGHTS or model_zoo default.")
    parser.add_argument("--dataset", default="city_testT", help="Dataset name to evaluate (registered name).")
    parser.add_argument("--output-dir", default="./output_eval", help="Where to save evaluation outputs.")
    parser.add_argument("--device", default="cuda", help="device to run model on (cuda or cpu).")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 注册数据集
    data_format = register_city_datasets()

    # 2) 构造 cfg，参数与训练阶段完全一致
    cfg = get_cfg()
    # 如训练时用 add_hsfpn_config(cfg)，这里也要加（如不用可注释）
    # from hsfpn import add_hsfpn_config
    # add_hsfpn_config(cfg)
    config_file = args.config_file
    if config_file:
        if not os.path.isabs(config_file) and not os.path.exists(config_file) and ("/" in config_file or "\\" in config_file):
            try:
                config_file = model_zoo.get_config_file(config_file)
            except Exception as e:
                print(f"[警告] model_zoo.get_config_file 失败: {e}, 尝试直接用 {args.config_file}")
        cfg.merge_from_file(config_file)
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

    # === 关键自定义参数（与训练阶段保持一致） ===
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.DATASETS.TRAIN = ("city_trainS",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 100
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 5000
    cfg.INPUT.MIN_SIZE_TRAIN = (600,)
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 权重加载设置：保持 cfg.MODEL.WEIGHTS 含有 "FPN" 便于判定 backbone 维度
    default_source_weights = "/mnt/lyh/DA-FasterCNN/weights/COCO-Detection/faster_rcnn_R_50_FPN_1x/model_final_b275ba.pkl"
    eval_weights = args.weights or cfg.MODEL.WEIGHTS or default_source_weights

    if eval_weights and not os.path.isfile(eval_weights):
        raise FileNotFoundError(f'指定的权重文件不存在: {eval_weights}')

    if not cfg.MODEL.WEIGHTS or "FPN" not in str(cfg.MODEL.WEIGHTS):
        if os.path.isfile(default_source_weights):
            cfg.MODEL.WEIGHTS = default_source_weights
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    # 3) 构建模型并加载权重
    model = build_model(cfg)
    model.to(args.device)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(eval_weights)  # 加载评估权重（默认等同于 cfg.MODEL.WEIGHTS ）

    # 4) 选择评估器
    if data_format == "coco":
        evaluator = COCOEvaluator(args.dataset, cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    else:
        evaluator = PascalVOCDetectionEvaluator(args.dataset)

    # 5) 构建 val_loader，并运行评估
    val_loader = build_detection_test_loader(cfg, args.dataset)
    results = inference_on_dataset(model, val_loader, evaluator)

    print("Evaluation results:")
    print(results)

    # 可选：保存到文件
    out_path = Path(cfg.OUTPUT_DIR) / "eval_results.txt"
    with open(out_path, "w") as f:
        f.write(str(results))
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()