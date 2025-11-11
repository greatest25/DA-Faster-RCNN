import torch
import torch.nn.functional as F
from typing import List

def consistency_regularization_loss(
    img_logits: torch.Tensor,               # shape: [B, 1, H, W]
    ins_logits_list: List[torch.Tensor]     # list of B tensors, each of shape [Nᵢ, 1]
) -> torch.Tensor:
    """
    计算 image-level 和 instance-level 域分类器输出之间的一致性正则化损失。
    Args:
        img_logits (Tensor): 来自图像级域分类器的 logits，
                             形状为 [B, 1, H, W]。
        ins_logits_list (List[Tensor]): 来自实例级域分类器的张量列表，
                                        每个批次中的图像对应一个张量。
                                        每个张量的形状为 [Nᵢ, 1]，其中 Nᵢ 是
                                        图像 i 的 ROI 提议数量。
    Returns:
        Tensor: 标量张量，表示批次中的平均一致性损失。
    计算公式：
        对于每个图像 i，计算图像级域概率 p̄_i（通过 sigmoid 后的平均值）和
        每个实例 j 的实例级域概率 p_{i,j}（通过 sigmoid 后的值）。
        一致性损失定义为 L2 距离的平均值：  
        L_i = (1 / Nᵢ) * Σ_j (p̄_i - p_{i,j})²  
        最终损失为所有图像的平均值：
        L = (1 / B) * Σ_i L_i
    其中 B 是批次大小。
    """
    B = img_logits.size(0)
    total_loss = 0.0
    valid_samples = 0

    for i in range(B):
        # Compute average image-level domain probability (after sigmoid)
        img_prob = torch.sigmoid(img_logits[i, 0]).mean()  # scalar ∈ [0,1]

        if i >= len(ins_logits_list):
            continue

        ins_logits = ins_logits_list[i]  # shape [Nᵢ, 1]
        if ins_logits.numel() == 0:
            continue

        ins_probs = torch.sigmoid(ins_logits.view(-1))  # shape [Nᵢ]

        # Compute L2 distance between image-level and each instance-level probability
        loss = F.mse_loss(ins_probs, img_prob.expand_as(ins_probs))  # sum_j (p̄_i - p_{i,j})²
        total_loss += loss
        valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, requires_grad=True, device=img_logits.device)

    return total_loss / valid_samples