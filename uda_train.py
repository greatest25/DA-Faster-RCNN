from pathlib import Path
import sys
import os
import torch
import torch_dct,einops  # noqa: F401


d2_path = '/mnt/lyh/DA-FasterCNN/detectron2-main'
if d2_path not in sys.path:
    sys.path.insert(0, d2_path)

print(sys.path)

# sys.path.append(str(Path(__file__).parent / "custom_backbones"))
# import hsfpn  # noqa: F401
# from hsfpn import add_hsfpn_config

from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
import logging
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.engine import default_writers

from detectron2.data.datasets import register_coco_instances, register_pascal_voc

sys.path.append("/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN")
from da2od.config import add_da2od_config
from da2od.model import build_da2od  # Build DA2OD model
from da2od.aligner_pred_guided import AlignMixin
from da2od.pseudolabeler import PseudoLabeler
from da2od.trainer import DA2ODTrainer


# #FOR PASCAL VOC ANNOTATIONS
# register_pascal_voc("city_trainS", "drive/My Drive/cityscape/", "train_s", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])
# register_pascal_voc("city_trainT", "drive/My Drive/cityscape/", "train_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

# register_pascal_voc("city_testT", "drive/My Drive/cityscape/", "test_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

# #FOR COCO ANNOTATIONS   
# register_coco_instances("dataset_train_synthetic", {}, "drive/My Drive/Bellomo_Dataset_UDA/synthetic/Object_annotations/Training_annotations.json", "./drive/My Drive/Bellomo_Dataset_UDA/synthetic/images")
# register_coco_instances("dataset_train_real", {}, "drive/My Drive/Bellomo_Dataset_UDA/real_hololens/training/training_set.json", "./drive/My Drive/Bellomo_Dataset_UDA/real_hololens/training")

# register_coco_instances("dataset_test_real", {}, "drive/My Drive/Bellomo_Dataset_UDA/real_hololens/test/test_set.json", "./drive/My Drive/Bellomo_Dataset_UDA/real_hololens/test")

from register_cityscapes import register_city_datasets, register_city_pseudo_dataset

# 添加命令行参数支持
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", default="", help="DA2OD config file")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()


data_format = register_city_datasets()      

logger = logging.getLogger("detectron2")

# === 伪标签加载逻辑 ===
pseudo_root = Path(__file__).resolve().parent / "pseudo_labels"
os.makedirs(pseudo_root, exist_ok=True)  # 确保目录存在

pseudo_candidates = [
    (pseudo_root / "city_trainT_full_pseudo_thr07_coco.json", "city_trainT_pseudo_thr07"),
    (pseudo_root / "city_trainT_full_pseudo_coco.json", "city_trainT_pseudo_full"),
    (pseudo_root / "city_trainT_pseudo_coco.json", "city_trainT_pseudo"),
]

target_train_dataset = ("city_trainT",)  # 默认使用无标签目标域
for pseudo_path, dataset_name in pseudo_candidates:
    if pseudo_path.is_file():
        logger.info(f"✓ Found pseudo label file: {pseudo_path}")
        target_image_root = "/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN/datasets/cityscape/train_t"
        register_city_pseudo_dataset(
            json_file=str(pseudo_path),
            dataset_name=dataset_name,
            image_root=target_image_root  
        )
        target_train_dataset = (dataset_name,)
        logger.info(f"✓ Using pseudo-labeled dataset: {dataset_name}")
        break
else:
    logger.warning(f"⚠ No pseudo label file found in {pseudo_root}")
    logger.warning(f"   Expected one of: {[str(p) for p, _ in pseudo_candidates]}")
    logger.info("   Using unlabeled target domain: city_trainT")
    logger.info("   → Run generate_pseudo_labels.py first to create pseudo labels!")
    
def do_train(cfg_source, cfg_target, model, resume = False):
    
    model.train()
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg_source.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    # Only load weights if file exists
    if cfg_source.MODEL.WEIGHTS and os.path.exists(cfg_source.MODEL.WEIGHTS):
        start_iter = (checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
    else:
        start_iter = 0
    max_iter = cfg_source.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg_source.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg_source.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader_source = build_detection_train_loader(cfg_source)
    data_loader_target = build_detection_train_loader(cfg_target) 
    logger.info("Starting training from iteration {}".format(start_iter))

    lambda_hyper = 0.1

    with EventStorage(start_iter) as storage:
        for data_source, data_target, iteration in zip(data_loader_source, data_loader_target, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data_source, False, 1)
            loss_dict_target = model(data_target, True, 1)
            
            loss_dict["loss_image_d"] += loss_dict_target["loss_image_d"]
            loss_dict["loss_instance_d"] += loss_dict_target["loss_instance_d"]
            loss_dict["loss_consistency_d"] += loss_dict_target["loss_consistency_d"]

            loss_dict["loss_image_d"] *= ( 0.5 * lambda_hyper)
            loss_dict["loss_instance_d"] *= ( 0.5 * lambda_hyper)
            loss_dict["loss_consistency_d"] *= ( 0.5 * lambda_hyper)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

cfg_source = get_cfg()
# add_hsfpn_config(cfg_source)
# cfg_source.MODEL.BACKBONE.NAME = "build_resnet_hsfpn_backbone"
# cfg_source.MODEL.HSFPN.ENABLED = True
add_da2od_config(cfg_source)
cfg_source.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
# Merge from command line config
if args.config_file:
    cfg_source.merge_from_file(args.config_file)


# 只需覆盖路径相关的本地设置
# cfg_source.MODEL.WEIGHTS = "/mnt/lyh/DA-FasterCNN/weights/COCO-Detection/faster_rcnn_R_50_FPN_1x/model_final_b275ba.pkl"
cfg_source.DATASETS.TRAIN = ("city_trainS",)
cfg_source.OUTPUT_DIR = "./output/da2od_baseline/"
os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)

model = build_da2od(cfg_source)

cfg_target = get_cfg()
# add_hsfpn_config(cfg_target)
# cfg_target.MODEL.BACKBONE.NAME = "build_resnet_hsfpn_backbone"
# cfg_target.MODEL.HSFPN.ENABLED = True
add_da2od_config(cfg_target)
cfg_target.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
# Merge from command line config
if args.config_file:
    cfg_target.merge_from_file(args.config_file)

cfg_target.OUTPUT_DIR = "./output/da2od_target/"
cfg_target.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg_target.DATASETS.TRAIN = target_train_dataset
cfg_target.DATALOADER.NUM_WORKERS = 0
os.makedirs(cfg_target.OUTPUT_DIR, exist_ok=True)


last_ckpt = os.path.join(cfg_source.OUTPUT_DIR, "last_checkpoint")
resume_flag = os.path.isfile(last_ckpt)
if resume_flag:
    logger.info(f"Resuming from {last_ckpt}")
do_train(cfg_source, cfg_target, model, resume=resume_flag)
