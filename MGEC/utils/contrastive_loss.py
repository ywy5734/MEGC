import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable=True):
        """
        temperature: 初始温度系数，控制 logits 的缩放范围
        learnable: 是否让温度系数作为可学习参数 (CLIP的标准做法是 True)
        """
        super().__init__()
        if learnable:
            # 初始化为 log(1/0.07) ≈ 2.65
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.register_buffer('logit_scale', torch.tensor(np.log(1 / temperature)))

    def forward(self, image_features, text_features, labels):
        """
        image_features: [Batch_Size, Dim] (ECG 特征)
        text_features: [Num_Classes, Dim] (文本特征/Prompt特征)
        labels: [Batch_Size, Num_Classes] (0或1的标签矩阵)
        """
        # 1. 特征归一化 (L2 Normalize)
        # 这是对比学习的关键，确保计算的是余弦相似度
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 2. 计算相似度矩阵
        # [B, D] @ [Num_Classes, D]^T -> [B, Num_Classes]
        logits = torch.matmul(image_features, text_features.t())
        
        # 3. 温度缩放
        # 限制 logit_scale 最大值为 100 (防止溢出)，exp 后通常在 10~100 之间
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        scaled_logits = logits * logit_scale
        
        # 4. 计算损失
        # 对于多标签分类，使用 BCEWithLogitsLoss 是最标准的做法 (即 SigLIP)
        # 它会让正样本对的相似度尽可能高，负样本对的相似度尽可能低
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
        
        return loss