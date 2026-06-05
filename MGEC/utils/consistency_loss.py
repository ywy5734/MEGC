import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ConsistencyLoss, self).__init__()
        self.margin = margin

    def forward(self, ecg_emb, text_emb, labels):
        """
        ecg_emb: (Batch_Size, Embed_Dim)
        text_emb: (Num_Classes, Embed_Dim)
        labels: (Batch_Size, Num_Classes) - 0 或 1
        """
        # 计算相似度矩阵 (Batch_Size, Num_Classes)
        # 假设输入已经 normalize 过
        # 1. 强制进行 L2 归一化，确保计算的是余弦相似度
        ecg_emb = F.normalize(ecg_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        # 2. 计算相似度矩阵
    
        similarity_matrix = torch.matmul(ecg_emb, text_emb.T)
        
        # 将标签转换为 DALR 的逻辑
        # 正样本 (y=1): 损失为 1 - cos
        # 负样本 (y=0): 损失为 max(0, cos - margin)
        
        # # 正样本掩码
        # pos_mask = (labels == 1).float()
        # # 负样本掩码
        # neg_mask = (labels == 0).float()
        # 建议修改为 (更安全)
        pos_mask = (labels > 0.5).float()
        neg_mask = (labels <= 0.5).float()
        
        # 正样本损失: 1 - sim
        pos_loss = (1 - similarity_matrix) * pos_mask
        
        # 负样本损失: max(0, sim - margin)
        neg_loss = torch.clamp(similarity_matrix - self.margin, min=0) * neg_mask
        
        # 计算平均损失 (除以总元素数量或非零元素数量)
        loss = (pos_loss.sum() + neg_loss.sum()) / (labels.numel() + 1e-8)
        
        return loss