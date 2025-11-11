# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, ResNet
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head
from ..da_modules.instance_level_discriminators import *
ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)
# [FPN] FPN-configured models dispatch to StandardROIHeads; the Res5/C4 blocks
# below serve non-FPN backbones such as C4/DC5 variants.
# [FPN] Comments tagged [FPN-DA-HOOK] highlight instance-level DA hooks on
# top of multi-scale ROI features.


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    # [FPN] BACKBONE 名含 "FPN" 时，此名称应解析为 StandardROIHeads，
    # 以启用 p2–p5 的多尺度池化；其他名称对应 C4/DC5 等单尺度头。
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    用于选择前景建议的辅助函数。
    参数说明：
        Proposal: 一个包含N个实例的列表，其中N是批次中图像的数量。
        bg_label: 背景类的标签索引。
    返回值说明：
        list[Instances]: N个实例，每个实例仅包含所选的前景实例。
        list[Tensor]: N个布尔向量，对应于每个实例对象的选择掩码。True表示所选实例。
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    # 遍历每个图像的建议
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes # 获取每个建议的真实类别
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)   # 创建前景选择掩码
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)    # 获取前景索引
        fg_proposals.append(proposals_per_image[fg_idxs])   # 选择前景建议
        fg_selection_masks.append(fg_selection_mask)    # 存储前景选择掩码
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    用于选择具有可见关键点的建议的辅助函数。
    参数说明：
        proposals: 一个包含N个实例的列表，其中N是批次中图像的数量。
    返回值：
        Proposals: 仅包含至少有一个可见关键点的建议。   
    !需要注意的是，这仍然与Detectron略有不同。
    在Detectron中，用于训练关键点头的建议是从·所有IOU>threshold且>=1个可见关键点的建议中重新采样的。
    在这里，建议首先从所有IOU>threshold的建议中采样，然后过滤掉没有可见关键点的建议。
    这种策略似乎对Detectron没有影响，并且更容易实现。
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # 检查图像中是否有建议，如果没有则跳过，否则继续处理
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1   # 提取可见掩码
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]   # 提取关键点坐标
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)   # 选择至少有一个可见关键点的建议
        selection_idxs = nonzero_tuple(selection)[0]    # 获取选择的索引
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])  

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads模块在R-CNN中执行所有每区域计算。
    通常包含以下逻辑：
    1.（仅在训练中）将建议与真实值匹配并对其进行采样
    2. 使用建议裁剪区域并提取每区域特征
    3. 使用不同的头进行每区域预测
    它可以有许多变体，作为此类的子类实现。
    该基类包含匹配/采样建议的逻辑。
    但是如果不需要采样逻辑，则不必继承此类。
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
    ):
        """
        NOTE: 该接口是实验性的。
        参数说明：
            num_classes (int): 前景类的数量（即不包括背景）
            batch_size_per_image (int): 用于训练的建议数量
            positive_fraction (float): 用于训练的正（前景）建议的比例。
                用于训练。
            proposal_matcher (Matcher): 用于匹配建议和真实值的匹配器
            proposal_append_gt (bool): 是否也将真实值包括为建议
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于N个建议和M个真实值之间的匹配，
        对建议进行采样并设置其分类标签。

        参数说明：
            matched_idxs (Tensor): 长度为N的向量，每个元素是每个建议在[0, M)中的最佳匹配真实值索引。
            matched_labels (Tensor): 长度为N的向量，表示每个建议的匹配器标签。
                （cfg.MODEL.ROI_HEADS.IOU_LABELS之一）。
            gt_classes (Tensor): 长度为M的向量。
        返回值：
            Tensor: 长度为N的向量，表示采样建议的索引。每个索引在[0, N)范围内。
            Tensor: 长度相同的向量，表示每个采样建议的分类标签。每个样本被标记为[0, num_classes)中的类别或背景（num_classes）。
        """
        has_gt = gt_classes.numel() > 0 # 是否有真实值
        # 为每个建议分配分类标签
        if has_gt:
            gt_classes = gt_classes[matched_idxs]   # 根据匹配索引获取对应的真实类别
            # 标签背景建议（0标签）
            gt_classes[matched_labels == 0] = self.num_classes  
            # 忽略建议（-1标签）
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes  # 全部标记为背景

        # 采样建议以获得正负平衡的批次
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)  # 合并前景和背景索引
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        为了训练ROI头，准备一些建议。
        它在`proposals`和`targets`之间执行框匹配，并为建议分配训练标签。
        它从建议和真实框中返回``self.batch_size_per_image``个随机样本，
        正样本的比例不大于``self.positive_fraction``。
        参数说明：
            见：meth:`ROIHeads.forward`
        返回值：
            list[Instances]:
                长度为N的`Instances`列表，包含为训练采样的建议。
                每个`Instances`具有以下字段：
                - proposal_boxes: 建议框
                - gt_boxes: 分配给建议的真实框
                （仅当建议的标签> 0时才有意义；如果标签= 0则真实框是随机的）
                其他字段，如“gt_classes”，“gt_masks”，包含在`targets`中。
        """
        """
        添加真实值作为建议
        在训练的早期阶段，区域建议网络（RPN）生成的初始建议可能质量较差。
        因此，可能没有足够的建议与真实对象有足够的重叠，可以用作第二阶段组件（框头、cls头、掩码头）的正面示例。
        将真实框添加到建议集中确保第二阶段组件从训练开始时就有一些正面示例。
        对于RPN，这种增强可以改善收敛性，并在COCO上实证提高了约0.5点的框AP（在一种测试配置下）。
        """
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)   # 将真实值添加为建议

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )   # 计算真实框和建议框之间的IOU矩阵
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)  # 匹配建议和真实框
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )   # 采样建议并获取对应的分类标签

            # 设定采样建议的属性
            proposals_per_image = proposals_per_image[sampled_idxs]  # 选择采样的建议
            proposals_per_image.gt_classes = gt_classes # 设置分类标签

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # 我们检索所有"gt_"前缀且并未在建议中设置的字段，并将它们添加到建议中。
                # 需要注意的是，此处的索引会浪费一些计算资源，
                # 因为像掩码、关键点等头部会再次过滤建议（通过前景/背景，或图像中的关键点数量等），
                # 因此我们实际上对数据进行了两次索引。
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # 如果图像中没有GT，我们不知道虚拟GT值可以是什么。
            # 因此返回的建议将没有任何gt_*字段，除了一个充满背景标签的gt_classes。
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())    # 计算背景样本数量
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])  # 计算前景样本数量
            proposals_with_gt.append(proposals_per_image)   # 添加处理后的建议

        # 记录前景和背景样本的统计信息
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        参数说明：
            ImageList:
            features (dict[str,Tensor]): 输入数据作为从特征图名称到张量的映射。
                轴0表示输入数据中的图像数量`N`；轴1-3是通道、高度和宽度，
                不同特征图之间可能有所不同（例如，如果使用特征金字塔）。
            proposals (list[Instances]): 长度为`N`的`Instances`列表。第i个
                `Instances`包含第i个输入图像的对象建议，具有字段“proposal_boxes”和“objectness_logits”。
            targets (list[Instances], optional): 长度为`N`的`Instances`列表。第i个
                `Instances`包含第i个输入图像的每实例真实注释。仅在训练期间指定`targets`。
                它可能具有以下字段：
                - gt_boxes: 每个实例的边界框。
                - gt_classes: 每个实例的标签，类别范围在[0，＃class]内。
                - gt_masks: PolygonMasks或BitMasks，每个实例的真实掩码。
                - gt_keypoints: NxKx3，每个实例的真实关键点。
        返回值说明：
            list[Instances]: 长度为`N`的`Instances`列表，包含检测到的实例。
            仅在推理期间返回；在训练期间可能为空。
            dict[str->Tensor]:
            从命名损失到存储损失的张量的映射。仅在训练期间使用。        
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    [非FPN] 该实现仅在 C4/DC5 类单尺度骨干下使用；FPN 配置不会走到这里。

    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.discriminatorProposalDC5 = DiscriminatorProposalDC5(2048)
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        target_domain = False,
        alpha = 1
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        loss_discriminatorProposal = self.discriminatorProposalDC5(box_features, target_domain, alpha)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        
        if self.training:
            del features
            if target_domain:
                return [], loss_discriminatorProposal
            else:
                losses = self.box_predictor.losses(predictions, proposals)
                losses.update(loss_discriminatorProposal)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(feature_list, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        参数说明：
            box_in_features(list[str]): 用于box head的特征名称列表。
            box_pooler(ROIPooler): 用于box head的区域特征池化器
            box_head(nn.Module): 转换特征以进行box预测
            box_predictor(nn.Module): 从特征中进行box预测。
                应具有与:class:`FastRCNNOutputLayers`相同的接口。
            mask_in_features(list[str]): 用于mask池化器或mask head的特征名称列表。
                如果不使用mask head，则为None。
            mask_pooler(ROIPooler): 用于从图像特征中提取区域特征的池化器。
                然后mask head将采用区域特征进行预测。
                如果为None，则mask head将直接采用由`mask_in_features`定义的图像特征字典
            mask_head(nn.Module): 转换特征以进行mask预测
            keypoint_in_features, keypoint_pooler, keypoint_head: 类似于``mask_*``。
            train_on_pred_boxes(bool): 是否使用box head的预测框来训练其他head，而不是建议框。
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        # [FPN] FPN 会传入多尺度特征名（如 p2–p5）；下方 ROIPooler 按 stride
        # 做多尺度 ROIAlign，供判别器与预测头共享特征。
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.discriminatorProposal = DiscriminatorProposal(1024)

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None # 是否启用关键点头
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    # 覆盖基类的 from_config 方法以处理多个头部的初始化
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclassses 并未更新为使用 from_config 风格的构造函数
        # 可能覆盖了 _init_*_head 方法。在这种情况下，这些被覆盖的方法
        # 将不是类方法，我们需要避免在这里尝试调用它们。
        # 我们使用 ismethod 进行测试，它仅对 cls 的绑定方法返回 True。
        # 这样的子类将需要处理调用它们覆盖的 _init_*_head 方法。

        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))    
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))   
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))   
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        # [FPN] 对 FPN 而言，上述 pooler_scales 记录了各金字塔层的 stride，
        # 供 ROIPooler 将 proposal 映射到正确的特征层。

        # 如果标准 ROI 头有多个输入特征，那么我们共享相同的预测器，因此通道数必须相同
        in_channels = [input_shape[f].channels for f in in_features]
        # 检查通道数是否一致
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        # [FPN] 多尺度层需保持通道一致以共享下游预测头参数。

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        target_domain = False,
        alpha = 1
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # 通常情况下，box head 的原始建议被 mask、keypoint head 使用。
            #但是当 `self.train_on_pred_boxes is True` 时，proposals 包含 box head 预测的框。
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # 在推理期间使用级联预测：mask 和 keypoints 头仅应用于得分最高的框检测。
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        该方法在推理期间使用。
        使用实例中的给定框来生成其他（非框）每ROI输出。
        对于下游任务非常有用，其中已知一个框，但需要获得其他属性（其他头的输出）。
        测试时增强也使用这个方法。
        参数说明：
            features:同`forward()`中的参数
            instances (list[Instances]): 用于预测其他输出的实例。期望存在键
                "pred_boxes"和"pred_classes"。
        返回值：
            list[Instances]:
                相同的`Instances`对象，具有额外的字段，如`pred_masks`或`pred_keypoints`。        
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        # [FPN] 推理分支一般不回传域判别损失；如需调试，可在此条件化调用判别器。
        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], target_domain = False, alpha = 1):
        """
        该函数实现了box head的前向传播逻辑。
        如果`self.train_on_pred_boxes is True`，则该函数将预测的框放在`proposals`参数的`proposal_boxes`字段中。
        参数说明：
            features (dict[str, Tensor]): 从特征图名称到张量的映射。
                同: meth：`ROIHeads.forward`。
            proposals (list[Instances]): 带有匹配真实值的每个图像的对象建议。
                每个实例具有字段“proposal_boxes”和“objectness_logits”。
        返回值：
            在训练中，返回损失的字典。
            在推理中，返回`Instances`列表，即预测的实例。
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        # [FPN-DA-HOOK] 多尺度 ROI 特征已在此融合，可在此处接入实例级域判别器。
        loss_discriminator = self.discriminatorProposal(box_features, target_domain, alpha)
        predictions = self.box_predictor(box_features)
        
        del box_features    # 释放内存

        if self.training:
            if target_domain:
                return loss_discriminator
            else:
                losses = self.box_predictor.losses(predictions, proposals)
                # [FPN] 判别器需提供 loss_instance_d 与 logits，rcnn.py 将 pop 这两项
                # 做一致性正则；这里 update 后即可写入 detector_losses。
                losses.update(loss_discriminator)
            # !由于建议会被覆盖，所以损失必须先计算。            
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)  # 替换为预测框
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        前向掩码预测分支的逻辑。
        参数说明：
            features (dict[str, Tensor]): 用于从特征图名称到张量的映射。同: meth：`ROIHeads.forward`。
            instances (list[Instances]): 用于训练/预测掩码的每个图像实例。
                在训练中，它们可以是建议。
                在推理中，它们可以是R-CNN框头预测的框。     
        返回值：
            在训练中，返回损失的字典。
            在推理中，使用新字段“pred_masks”更新`instances`并返回它。        
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            # [FPN] 若实例级判别器只关注前景，可在此过滤后的 ROI 上提特征。
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        关键点预测分支的前向逻辑。
        参数说明：
            features (dict[str, Tensor]): 从特征图名称到张量的映射。
                同: meth：`ROIHeads.forward`。
            instances (list[Instances]): 用于训练/预测关键点的每个图像实例。
                在训练中，它们可以是建议。
                在推理中，它们可以是R-CNN框头预测的框。     
        返回值：
            在训练中，返回损失的字典。
            在推理中，使用新字段“pred_keypoints”更新`instances`并返回它。        
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            # [FPN] 可选：仅用带可见关键点的前景 ROI 参与实例级判别。
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)
