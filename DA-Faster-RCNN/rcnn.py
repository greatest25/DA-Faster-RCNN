# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from ..da_modules.image_level_discriminators import *
from ..da_modules.consistency_regularization_loss import *

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    生成RCNN的通用架构。
    1. 图像级特征提取（又名骨干）
    2. 区域建议生成
    3. 每区域特征提取和预测(又名ROI头)
    该模块支持端到端训练和推理。    
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        model_weights: str
    ):
        """
        参数说明：
        backbone: 一个骨干模块，必须遵循detectron2的骨干接口
        proposal_generator: 一个使用骨干特征生成建议的模块
        roi_heads: 一个执行每区域计算的ROI头
        pixel_mean, pixel_std: 列表或元组，包含＃channels元素，表示
            用于规范化输入图像的每通道均值和标准差
        input_format: 描述输入通道的含义。可视化时需要
        vis_period: 运行可视化的周期。设置为0以禁用。
        model_weights: 预训练模型权重路径

        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        # 设置图像级域分类器,dim_in_feature_discriminator表示输入特征维度
        if self.training:
            if "FPN" in model_weights:
                self.backbone_name = "FPN"
                self.dim_in_feature_discriminator = 256
            elif "DC5" in model_weights:
                self.backbone_name = "DC5"
                self.dim_in_feature_discriminator = 2048
            elif "C4" in model_weights:
                self.backbone_name = "C4"
                self.dim_in_feature_discriminator = 1024
            self.discriminator = ImageDomainDiscriminator(self.dim_in_feature_discriminator)
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        print(cfg.MODEL.BACKBONE.NAME)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "model_weights": cfg.MODEL.WEIGHTS
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        主要用于可视化图像和建议的函数。它显示原始图像上的真实边界框和最多20个
        原始图像上的高分预测对象建议。用户可以为不同的模型实现不同的可视化函数。
        参数说明：
        batched_inputs: 一个包含输入图像和其他信息的字典列表
        proposals: 一个包含建议的Instances列表
        该函数没有返回值，但会将可视化图像存储在事件存储中以供后续查看。

        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], target_domain = False, alpha=0.1):
        """
        参数说明：
        batched_inputs: 一个列表，批量输出：class：`DatasetMapper` 。
            列表中的每个项目都包含一个图像的输入。
            目前，列表中的每个项目都是一个包含以下内容的字典：
            * image: Tensor，图像格式为（C，H，W）。
            * instances（可选）：groundtruth：class：`Instances`
            * proposals（可选）：class：`Instances`，预计算建议。
            包含在原始字典中的其他信息，例如：
            *“height”，“width”（int）：模型的输出分辨率，用于推理
                详情请参见：meth：`postprocess` 。
        返回值：
        list [dict]：
            每个字典都是一个输入图像的输出。
            该字典包含一个键“instances”，其值为：class：`Instances` 。
            ：class：`Instances` 对象具有以下键：
            “pred_boxes”，“pred_classes”，“scores”，“pred_masks”，“pred_keypoints”
        """        
        if not self.training:
            return self.inference(batched_inputs)

        # 前向传播，返回损失
        images = self.preprocess_image(batched_inputs)
        # 获取groundtruth实例
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]# *
        else:
            gt_instances = None
        # 提取图像特征
        features = self.backbone(images.tensor)

        # 图像级域分类器前向传播，选择不同骨干对应的特征层
        if self.backbone_name == "FPN":
            disc_out =  self.discriminator(features["p5"], target_domain, alpha)
        elif self.backbone_name == "C4":
            disc_out =  self.discriminator(features["res4"], target_domain, alpha)
        elif self.backbone_name == "DC5":
            disc_out =  self.discriminator(features["res5"], target_domain, alpha)
        else:
            raise ValueError(f"Unknown backbone name: {self.backbone_name}")

        # 获取图像级域分类器损失和logits
        loss_image_d = disc_out["loss_image_d"]
        img_level_logits = disc_out["logits"]  # shape: [B, 1, H, W]

        # 获取建议和建议损失，为后续的ROI头做准备
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, target_domain)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # 通过ROI头获取检测器损失
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, target_domain, alpha)
        if self.vis_period > 0:
            storage = get_event_storage() # 获取事件存储
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # 获取实例级域分类器损失和logits
        # 从检测器损失中弹出实例级域分类器损失和logits，避免重复计算
        loss_instance_d = detector_losses.pop("loss_instance_d")
        inst_level_logits = detector_losses.pop("logits")  

        # 计算一致性正则化损失
        loss_consistency = consistency_regularization_loss(img_level_logits, inst_level_logits)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses["loss_image_d"] = loss_image_d
        losses["loss_instance_d"] = loss_instance_d
        losses["loss_consistency_d"] = loss_consistency
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        参数说明：
        batched_inputs: 与：meth：`forward`相同
        detected_instances（可选）：如果提供，则跳过建议生成和ROI头，并使用 
            提供的检测实例进行推理。
        do_postprocess：是否对结果进行后处理，以匹配输入图像的原始尺寸。

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        # 检查是否为推理模式
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        # 如果未提供检测实例，则生成建议并通过ROI头获取结果
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        对输入图像进行预处理，包括归一化和批处理。包括：
        1. 将图像移动到当前设备
        2. 使用注册的均值和标准差对图像进行归一化
        3. 使用ImageList将图像打包成批次
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        重新调整实例预测以匹配输入图像的原始尺寸。
        """
        # 将每个图像的结果调整为原始尺寸
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    仅用于生成建议的RCNN架构。
    1. 图像级特征提取（又名骨干）
    2. 区域建议生成
    该模块支持端到端训练和推理。
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        参数说明：
        backbone: 一个骨干模块，必须遵循detectron2的骨干接口
        proposal_generator: 一个使用骨干特征生成建议的模块
        pixel_mean, pixel_std: 列表或元组，包含＃channels元素，表示
            用于规范化输入图像的每通道均值和标准差
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        参数说明：
        batched_inputs: 一个列表，批量输出：class：`DatasetMapper` 。
            列表中的每个项目都包含一个图像的输入。
            目前，列表中的每个项目都是一个包含以下内容的字典：
            * image: Tensor，图像格式为（C，H，W）。
            * instances（可选）：groundtruth：class：`Instances`
            * proposals（可选）：class：`Instances`，预计算建议。
            包含在原始字典中的其他信息，例如：
            *“height”，“width”（int）：模型的输出分辨率，用于推理
        
        返回值：
            list [dict]：
                每个字典都是一个输入图像的输出。
                该字典包含一个键“proposals”，其值为：class：`Instances` 。
                ：class：`Instances` 对象具有以下键：
                “proposal_boxes”，“objectness_logits”
        """

        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # During training, return the losses and the proposals.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
