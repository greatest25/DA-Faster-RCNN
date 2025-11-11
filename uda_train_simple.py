#!/usr/bin/env python
"""
简化版DA2OD训练脚本 - 使用DA2ODTrainer
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 确保导入路径正确
sys.path.insert(0, "/mnt/lyh/DA-FasterCNN/detectron2-main")
sys.path.insert(0, "/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN")

from detectron2.config import get_cfg
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger

# 注册Cityscapes数据集
from register_cityscapes import register_city_datasets, register_city_pseudo_dataset
register_city_datasets()

# DA2OD imports
from da2od.config import add_da2od_config
from da2od.trainer import DA2ODTrainer

setup_logger()
logger = logging.getLogger("detectron2")

# === 伪标签注册逻辑 ===
pseudo_root = Path(__file__).resolve().parent / "pseudo_labels"
pseudo_candidates = [
    (pseudo_root / "city_trainT_r50fpn_full_thr07.json", "city_trainT_pseudo_thr07"),
    (pseudo_root / "city_trainT_full_pseudo_coco.json", "city_trainT_pseudo_full"),
    (pseudo_root / "city_trainT_pseudo_coco.json", "city_trainT_pseudo"),
]

for pseudo_path, dataset_name in pseudo_candidates:
    if pseudo_path.is_file():
        logger.info(f"✓ Found pseudo label file: {pseudo_path}")
        target_image_root = "/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN/datasets/cityscape/train_t"
        register_city_pseudo_dataset(
            json_file=str(pseudo_path),
            dataset_name=dataset_name,
            image_root=target_image_root  
        )
        logger.info(f"✓ Registered pseudo-labeled dataset: {dataset_name}")
        break
else:
    logger.warning(f"⚠ No pseudo label file found in {pseudo_root}")

def setup(args):
    """设置配置"""
    cfg = get_cfg()
    add_da2od_config(cfg)
    
    # 加载配置文件
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    cfg.freeze()
    return cfg

def main(args):
    """主函数"""
    cfg = setup(args)
    
    logger.info(f"✓ 配置加载完成")
    logger.info(f"  - 训练数据集: {cfg.DATASETS.TRAIN}")
    logger.info(f"  - 测试数据集: {cfg.DATASETS.TEST}")
    logger.info(f"  - 最大迭代: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"  - 输出目录: {cfg.OUTPUT_DIR}")
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 创建trainer并训练
    trainer = DA2ODTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA2OD Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DA2OD Training Script")
    print(f"Config: {args.config_file}")
    print(f"Resume: {args.resume}")
    print(f"GPUs: {args.num_gpus}")
    print("=" * 60)
    
    launch(
        main,
        args.num_gpus,
        args=(args,),
    )
