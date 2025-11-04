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

# #FOR PASCAL VOC ANNOTATIONS
# register_pascal_voc("city_trainS", "drive/My Drive/cityscape/", "train_s", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])
# register_pascal_voc("city_trainT", "drive/My Drive/cityscape/", "train_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

# register_pascal_voc("city_testT", "drive/My Drive/cityscape/", "test_t", 2007, ['car','person','rider','truck','bus','train','motorcycle','bicycle'])

# #FOR COCO ANNOTATIONS   
# register_coco_instances("dataset_train_synthetic", {}, "drive/My Drive/Bellomo_Dataset_UDA/synthetic/Object_annotations/Training_annotations.json", "./drive/My Drive/Bellomo_Dataset_UDA/synthetic/images")
# register_coco_instances("dataset_train_real", {}, "drive/My Drive/Bellomo_Dataset_UDA/real_hololens/training/training_set.json", "./drive/My Drive/Bellomo_Dataset_UDA/real_hololens/training")

# register_coco_instances("dataset_test_real", {}, "drive/My Drive/Bellomo_Dataset_UDA/real_hololens/test/test_set.json", "./drive/My Drive/Bellomo_Dataset_UDA/real_hololens/test")

from register_cityscapes import register_city_datasets

data_format = register_city_datasets()      

logger = logging.getLogger("detectron2")

def do_train(cfg_source, cfg_target, model, resume = False):
    
    model.train()
    optimizer = build_optimizer(cfg_source, model)
    scheduler = build_lr_scheduler(cfg_source, optimizer)
    checkpointer = DetectionCheckpointer(model, cfg_source.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    start_iter = (checkpointer.resume_or_load(cfg_source.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
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
cfg_source.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg_source.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
cfg_source.OUTPUT_DIR = "./output/resnet50_fpn_baseline/"
cfg_source.DATASETS.TRAIN = ("city_trainS",)
cfg_source.DATALOADER.NUM_WORKERS = 2
# cfg_source.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg_source.MODEL.WEIGHTS = "/mnt/lyh/DA-FasterCNN/weights/COCO-Detection/faster_rcnn_R_50_FPN_1x/model_final_b275ba.pkl"
cfg_source.SOLVER.IMS_PER_BATCH = 4
cfg_source.SOLVER.BASE_LR = 0.0005
cfg_source.SOLVER.WARMUP_FACTOR = 1.0 / 100
cfg_source.SOLVER.WARMUP_ITERS = 1000
cfg_source.SOLVER.MAX_ITER = 5000
cfg_source.INPUT.MIN_SIZE_TRAIN = (600,)
cfg_source.INPUT.MIN_SIZE_TEST = 0
os.makedirs(cfg_source.OUTPUT_DIR, exist_ok=True)
cfg_source.MODEL.ROI_HEADS.NUM_CLASSES = 8
model = build_model(cfg_source)

cfg_target = get_cfg()
# add_hsfpn_config(cfg_target)
# cfg_target.MODEL.BACKBONE.NAME = "build_resnet_hsfpn_backbone"
# cfg_target.MODEL.HSFPN.ENABLED = True
cfg_target.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg_target.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
cfg_target.OUTPUT_DIR = "./output/resnet50_fpn_uda_cityscape_pascalvoc/"
cfg_target.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg_target.DATASETS.TRAIN = ("city_trainT",)
cfg_target.INPUT.MIN_SIZE_TRAIN = (600,)
cfg_target.DATALOADER.NUM_WORKERS = 0
cfg_target.SOLVER.IMS_PER_BATCH = 4

last_ckpt = os.path.join(cfg_source.OUTPUT_DIR, "last_checkpoint")
resume_flag = os.path.isfile(last_ckpt)
if resume_flag:
    logger.info(f"Resuming from {last_ckpt}")
do_train(cfg_source, cfg_target, model, resume=resume_flag)


