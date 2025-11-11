# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
模型输入/输出张量维度约定:

    N: minibatch中的图像数量
    L:运行RPN的每个图像的特征图数量
    A:单元锚点的数量（所有特征图必须相同）
    Hi, Wi:第i个特征图的高度和宽度
    B:盒子参数化的大小
    box_dim:盒子的维度（4或5）

命名约定：
    objectness:指锚点作为对象与非对象的二进制分类。
    deltas:指参数化box2box变换的4d（dx，dy，dw，dh）增量，或旋转框的5d。
    pred_objectness_logits:预测的对象性分数在[-inf，+inf]中;使用
        sigmoid（pred_objectness_logits）来估计P（对象）。
    gt_labels:对象性的地面真实二进制分类标签
    pred_anchor_deltas:预测的box2box变换增量
    gt_anchor_deltas:真实box2box变换增量
"""

# RPN主要作用是为后续的ROI头生成高质量的区域建议。
# RPN头负责从特征图中预测每个锚点的对象性分数和边界框回归增量。
# RPN头的实现可以通过注册不同的类来扩展，这些类可以通过配置进行选择。

def build_rpn_head(cfg, input_shape):
    """
    通过配置构建RPN头。
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    标准RPN分类和回归头，如:paper:`Faster R-CNN`中所述。
    使用3x3 conv生成共享隐藏状态，从中一个1x1 conv预测每个锚点的对象性logits，
    第二个1x1 conv预测边界框增量，指定如何将每个锚点变形为对象建议。
    该实现支持多个3x3 conv层以生成隐藏表示。
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        """
        NOTE: this interface is experimental.

        参数说明：
            in_channels (int): 输入特征通道的数量。当使用多个输入特征时，它们必须具有相同的通道数。
            num_anchors (int): 每个空间位置上要预测的锚点数量。
                每个特征图的总锚点数将为`num_anchors * H * W`。
            box_dim (int): 盒子的维度，这也是每个锚点要进行的边界框回归预测的数量。
                轴对齐盒子的box_dim=4，而旋转盒子的box_dim=5。
            conv_dims (list[int]): 一个整数列表，表示N个conv层的输出通道数。
                将其设置为-1以使用与输入通道相同的输出通道数。
        """
        super().__init__()
        cur_channels = in_channels
        # 使旧的变量名和结构保持向后兼容性。否则旧的检查点将无法加载。
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # 标准RPN头假设所有输入特征具有相同的通道数。
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead需要知道每个空间位置的锚点数量和盒子维度。
        # NOTE: 这里假设所有输入特征具有相同的锚点数量和盒子维度。
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        """
        参数说明：
            features (list[Tensor]): 特征图列表
        返回值：
            list[Tensor]: 包含L个元素的列表。
                元素i是形状为(N, A, Hi, Wi)的张量，表示
                所有锚点的预测对象性logits。A是单元锚点的数量。
            list[Tensor]: 包含L个元素的列表。元素i是形状为
                (N, A*box_dim, Hi, Wi)的张量，表示用于将锚点
                转换为建议的预测“增量”。
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
    ):
        """
        NOTE: this interface is experimental.

        参数说明：
            in_features (list[str]): 要使用的输入特征名称列表
            head (nn.Module): 一个模块，从每个级别的每级特征中预测logits和回归增量的列表
            anchor_generator (nn.Module): 一个从特征列表创建锚点的模块。通常是:class:`AnchorGenerator`的实例
            anchor_matcher (Matcher): 通过将锚点与地面真实值匹配来标记锚点。
            box2box_transform (Box2BoxTransform): 定义从锚点盒到实例盒的变换
            batch_size_per_image (int): 每个图像用于训练的锚点数量
            positive_fraction (float): 用于训练的前景锚点的比例
            pre_nms_topk (tuple[float]): (train, test)表示在NMS之前选择的前k个建议的数量，训练和测试中。
            post_nms_topk (tuple[float]): (train, test)表示
                在NMS之后选择的前k个建议的数量，训练和测试中。
            nms_thresh (float): 用于去重预测建议的NMS阈值
            min_box_size (float): 删除任何边小于此阈值的建议框，
                以输入图像像素为单位
            anchor_boundary_thresh (float): 传统选项
            loss_weight (float|dict): 用于损失的权重。可以是用于加权
                所有rpn损失的单个浮点数，或单个权重的字典。有效的字典键是：
                    "loss_rpn_cls" - 应用于分类损失
                    "loss_rpn_loc" - 应用于盒回归损失
            box_reg_loss_type (str): 使用的损失类型。支持的损失："smooth_l1"、"giou"。
            smooth_l1_beta (float): 平滑L1回归损失的beta参数。默认为
                使用L1损失。仅当`box_reg_loss_type`为"smooth_l1"时使用        
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label):
        """
        随机采样正负样本子集，并将标签向量中未包含在样本中的所有元素覆盖为忽略值（-1）。
        参数说明：
            label (Tensor): 一个-1、0、1的向量。将被就地修改并返回。
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # 填充所有标签为-1，然后将采样的正负索引设置为1和0
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        参数说明：
            anchors (list[Boxes]): 每个特征图的锚点。
            gt_instances: 每个图像的真实实例。
        返回值：
            list[Tensor]:
                包含#img张量的列表。第i个元素是一个标签向量，其长度为
                所有特征图R = sum(Hi * Wi * A)的锚点总数。
                标签值在{-1，0，1}中，含义为：-1 = 忽略; 0 = 负类;
                1 = 正类。
            list[Tensor]:
                第i个元素是一个Rx4张量。值是每个锚点匹配的gt盒子。
                对于未标记为1的锚点，值未定义。
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            # 计算锚点和真实盒子之间的匹配质量矩阵
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors) 
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # 匹配的索引：对于每个锚点，返回匹配的gt盒子的索引
            # gt_labels_i：对于每个锚点，返回匹配的gt盒子的标签（1=正类，0=负类，-1=忽略）
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # 通过仅考虑完全在图像内的锚点来过滤锚点
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)   # 对标签进行采样

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)  # 如果没有gt盒子，则创建一个全零张量
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor    # 获取每个锚点匹配的gt盒子

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        返回RPN以及与其相关的损失。

        参数说明：
            achors (list[Boxes]): 每个特征图的锚点，每个锚点形状为(Hi*Wi*A, B)，其中B是盒子维度（4或5）。
            pred_objectness_logits (list[Tensor]): 包含L个元素的列表。
                元素i是形状为(N, Hi*Wi*A)的张量，表示
                所有锚点的预测对象性logits。
            gt_labels (list[Tensor]): :meth:`label_and_sample_anchors`的输出。
            pred_anchor_deltas (list[Tensor]): 包含L个元素的列表。元素i是形状为
                (N, Hi*Wi*A, 4或5)的张量，表示用于将锚点转换为建议的预测“增量”。
            gt_boxes (list[Tensor]): :meth:`label_and_sample_anchors`的输出。

        返回值：
            dict[loss name -> loss value]: 一个字典，将损失名称映射到损失值。
                损失名称为：`loss_rpn_cls`表示对象性分类，`loss_rpn_loc`表示建议定位。
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # 记录用于训练的正负锚点数量
        pos_mask = gt_labels == 1   # 正样本掩码，用于筛选正样本
        num_pos_anchors = pos_mask.sum().item() # 统计正样本数量
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        # 计算定位损失和对象性分类损失
        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0 # 仅考虑未忽略的锚点
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images # 标准化因子，用于归一化损失
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # 原始论文使用的标准化因子略微不同，但效果相似
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}# 应用损失权重
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        target_domain = False
    ):
        """
        参数说明：
            images (ImageList): 长度为`N`的输入图像
            features (dict[str, Tensor]): 输入数据作为从特征图名称到张量的映射。
                轴0表示输入数据中的图像数量`N`;轴1-3是通道、高度和宽度，
                特征图之间可能有所不同（例如，如果使用特征金字塔）。    
            gt_instances (list[Instances], optional): 长度为`N`的`Instances`列表。
                每个`Instances`存储对应图像的地面真实实例。
        返回值：
            proposals: list[Instances]: 包含字段“proposal_boxes”、“objectness_logits”
            loss: dict[Tensor] or None: RPN损失。如果在训练模式下返回，否则为None。
        """
        features = [features[f] for f in self.in_features]  # 提取所需的输入特征
         # 为每个特征图生成锚点
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)    # 通过RPN头获取预测
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            if target_domain:
                # If we are using the target domain, we do not have ground truth instances
                # and we skip the loss computation.
                losses = {}
            else:
                assert gt_instances is not None, "RPN requires gt_instances in training!"
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                losses = self.losses(
                    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
                )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        通过解码所有预测的盒子回归增量为建议来生成建议。再通过应用NMS并删除过小的盒子来找到顶级建议框。

        参数说明：
            anchors (list[Boxes]): 每个特征图的锚点。
            pred_objectness_logits (list[Tensor]): 包含L个元素的列表。
                元素i是形状为(N, Hi*Wi*A)的张量，表示
                所有锚点的预测对象性logits。
            pred_anchor_deltas (list[Tensor]): 包含L个元素的列表。元素i是形状为
                (N, Hi*Wi*A, 4或5)的张量，表示用于将锚点转换为建议的预测“增量”。
            image_sizes (list[tuple]): 每个图像的（高度，宽度）。

        返回值：    
            proposals (list[Instances]): N个Instances的列表。第i个Instances
                存储图像i的post_nms_topk对象建议，这些建议按其对象性分数降序排序。
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        将锚点通过应用预测的锚点增量转换为建议。

        参数说明：
            anchors (list[Boxes]): 每个特征图的锚点。
            pred_anchor_deltas (list[Tensor]): 包含L个元素的列表。元素i是形状为
                (N, Hi*Wi*A, 4或5)的张量，表示用于将锚点转换为建议的预测“增量”。
        返回值：
            proposals (list[Tensor]): 包含L个元素的列表。元素i是形状为
                (N, Hi*Wi*A, B)的张量，表示预测的建议盒子。
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1) 
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
