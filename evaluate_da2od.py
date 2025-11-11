#!/usr/bin/env python3
"""
评估脚本 - 支持标准模型和 DA2OD 训练的模型
用途: 评估训练好的模型在测试集上的性能，支持 EMA 权重加载
"""

import argparse
import os
from pathlib import Path
import sys

d2_path = '/mnt/lyh/DA-FasterCNN/detectron2-main'
if d2_path not in sys.path:
    sys.path.insert(0, d2_path)

from register_cityscapes import register_city_datasets, register_city_pseudo_dataset

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, inference_on_dataset

# DA2OD 支持（可选）
try:
    from da2od.config import add_da2od_config
    from da2od.checkpoint import CheckpointerWithEMA
    from da2od.ema import EMA
    DA2OD_AVAILABLE = True
except ImportError:
    DA2OD_AVAILABLE = False
    print("⚠ DA2OD modules not available. Only standard evaluation supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint (supports DA2OD)")
    parser.add_argument("--config-file", default=None, help="Config file (YAML). If omitted, uses COCO R50-FPN default.")
    parser.add_argument("--weights", default=None, help="Path to model weights (pth/pkl).")
    parser.add_argument("--dataset", default="city_testT", help="Dataset name to evaluate.")
    parser.add_argument("--output-dir", default="./output_eval", help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights (for DA2OD models).")
    parser.add_argument("--eval-pseudo", default=None, help="Evaluate on pseudo-labeled dataset (JSON path).")
    return parser.parse_args()


def setup_cfg(args):
    """配置模型参数"""
    cfg = get_cfg()
    
    # 添加 DA2OD 配置（如果可用）
    if DA2OD_AVAILABLE:
        add_da2od_config(cfg)
    
    # 加载配置文件
    if args.config_file:
        config_file = args.config_file
        if not os.path.isabs(config_file) and not os.path.exists(config_file):
            try:
                config_file = model_zoo.get_config_file(config_file)
            except Exception as e:
                print(f"⚠ model_zoo.get_config_file failed: {e}")
        cfg.merge_from_file(config_file)
    else:
        # 使用默认 Faster R-CNN R50-FPN 配置
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    
    # 基本配置
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.DEVICE = args.device
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


def load_model(cfg, args):
    """构建并加载模型"""
    model = build_model(cfg)
    model.to(args.device)
    
    # 确定权重路径
    if args.weights:
        eval_weights = args.weights
    elif cfg.MODEL.WEIGHTS:
        eval_weights = cfg.MODEL.WEIGHTS
    else:
        eval_weights = "/mnt/lyh/DA-FasterCNN/weights/COCO-Detection/faster_rcnn_R_50_FPN_1x/model_final_b275ba.pkl"
    
    if not os.path.isfile(eval_weights):
        raise FileNotFoundError(f'权重文件不存在: {eval_weights}')
    
    print(f"✓ Loading weights from: {eval_weights}")
    
    # 根据是否使用 EMA 选择不同的加载方式
    if args.use_ema and DA2OD_AVAILABLE:
        print("✓ Using EMA weights (DA2OD mode)")
        # 创建 EMA 模型
        ema = EMA(build_model(cfg), cfg.EMA.ALPHA if hasattr(cfg, 'EMA') else 0.999)
        
        # 使用 CheckpointerWithEMA 加载
        checkpointer = CheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.add_checkpointable("ema", ema)
        checkpointer.load(eval_weights)
        
        # 用 EMA 模型替换原模型
        model = ema.ema_model
        model.eval()
    else:
        # 标准加载方式
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(eval_weights)
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("模型评估 - DA-Faster-RCNN")
    print("=" * 60)
    
    # 1. 注册数据集
    data_format = register_city_datasets()
    print(f"✓ Registered Cityscapes datasets (format: {data_format})")
    
    # 如果需要评估伪标签数据集
    if args.eval_pseudo:
        if not os.path.isfile(args.eval_pseudo):
            raise FileNotFoundError(f"Pseudo label file not found: {args.eval_pseudo}")
        
        pseudo_dataset_name = "eval_pseudo_labeled"
        image_root = "/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN/datasets/cityscape/train_t"
        register_city_pseudo_dataset(
            json_file=args.eval_pseudo,
            dataset_name=pseudo_dataset_name,
            image_root=image_root
        )
        args.dataset = pseudo_dataset_name
        print(f"✓ Registered pseudo-labeled dataset: {pseudo_dataset_name}")
    
    # 2. 配置模型
    cfg = setup_cfg(args)
    
    # 3. 加载模型
    model = load_model(cfg, args)
    
    # 4. 选择评估器
    print(f"\n✓ Evaluating on dataset: {args.dataset}")
    
    if data_format == "coco" or args.eval_pseudo:
        evaluator = COCOEvaluator(
            args.dataset, 
            cfg, 
            False, 
            output_dir=os.path.join(cfg.OUTPUT_DIR, "inference")
        )
    else:
        evaluator = PascalVOCDetectionEvaluator(args.dataset)
    
    # 5. 构建测试数据加载器
    val_loader = build_detection_test_loader(cfg, args.dataset)
    
    # 6. 运行评估
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    
    results = inference_on_dataset(model, val_loader, evaluator)
    
    # 7. 打印和保存结果
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果到文件
    out_path = Path(cfg.OUTPUT_DIR) / "eval_results.txt"
    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Weights: {args.weights or cfg.MODEL.WEIGHTS}\n")
        f.write(f"EMA: {args.use_ema}\n\n")
        
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
