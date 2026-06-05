import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import random
from models.vit1d import TransformerBlock, PreNorm, Attention, FeedForward


class CrossChannelMultiGranularityPatchEmbedding(nn.Module):
    """
    跨通道多粒度 Patching
    将所有导联在同一时间窗口下的数据视为一个整体 Patch
    """
    def __init__(self, num_leads, patch_sizes, embed_dim, seq_len):
        super().__init__()
        self.num_leads = num_leads
        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # 为每个粒度创建独立的 patch embedding 层
        self.patch_embeddings = nn.ModuleList()
        self.num_patches_list = []
        self.padded_seq_lens = []  # 存储每个粒度需要的填充后序列长度
        
        for patch_size in patch_sizes:
            # 计算需要的 patch 数量和填充后的序列长度
            num_patches = (seq_len + patch_size - 1) // patch_size  # 向上取整
            padded_seq_len = num_patches * patch_size
            self.num_patches_list.append(num_patches)
            self.padded_seq_lens.append(padded_seq_len)
            
            # 跨通道 Patching: 输入 [B, C, T] -> [B, P, C*patch_size] -> [B, P, embed_dim]
            # 其中 C=num_leads, P=num_patches
            patch_embed = nn.Sequential(
                Rearrange('b c (n p) -> b n (p c)', p=patch_size),
                nn.LayerNorm(patch_size * num_leads),
                nn.Linear(patch_size * num_leads, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            self.patch_embeddings.append(patch_embed)
        
        # 位置嵌入：每个粒度独立的位置嵌入
        self.pos_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_patches, embed_dim) / embed_dim ** 0.5)
            for num_patches in self.num_patches_list
        ])
    
    def forward(self, x):
        """
        x: [B, num_leads, seq_len]
        返回: List of [B, num_patches, embed_dim] for each granularity
        """
        B, C, T = x.shape
        assert C == self.num_leads, f'导联数不匹配: 期望 {self.num_leads}, 得到 {C}'
        
        patch_sequences = []
        for i, (patch_embed, pos_emb, padded_seq_len) in enumerate(
            zip(self.patch_embeddings, self.pos_embeddings, self.padded_seq_lens)
        ):
            # 如果序列长度不能被 patch_size 整除，进行填充或截断
            if T < padded_seq_len:
                # 使用零填充到需要的长度
                padding_size = padded_seq_len - T
                x_padded = torch.nn.functional.pad(x, (0, padding_size), mode='constant', value=0)
            elif T > padded_seq_len:
                # 如果序列长度超过需要的长度，截断
                x_padded = x[:, :, :padded_seq_len]
            else:
                x_padded = x
            
            # 跨通道 Patching: 所有导联一起切片
            patches = patch_embed(x_padded)  # [B, num_patches, embed_dim]
            # 添加位置嵌入
            patches = patches + pos_emb
            patch_sequences.append(patches)
        
        return patch_sequences


class PatchWiseAugmentation(nn.Module):
    """
    Patch-wise 数据增强：在 Embedding 层之后进行 Masking 或 Jittering
    """
    def __init__(self, mask_prob=0.15, jitter_std=0.01, enable_augmentation=True):
        super().__init__()
        self.mask_prob = mask_prob
        self.jitter_std = jitter_std
        self.enable_augmentation = enable_augmentation
    
    def forward(self, x):
        """
        x: List of [B, num_patches, embed_dim]
        返回: 增强后的相同形状的列表
        """
        if not self.training or not self.enable_augmentation:
            return x
        
        augmented_sequences = []
        for patches in x:
            B, N, D = patches.shape
            
            # Patch-wise Masking: 随机将某些 patch 置零
            mask = torch.rand(B, N, 1, device=patches.device) > self.mask_prob
            masked_patches = patches * mask
            
            # Jittering: 添加小的随机噪声
            if self.jitter_std > 0:
                jitter = torch.randn_like(masked_patches) * self.jitter_std
                masked_patches = masked_patches + jitter
            
            augmented_sequences.append(masked_patches)
        
        return augmented_sequences


class CrossAttention(nn.Module):
    """
    交叉注意力：用于将粒度内的 tokens 聚合到 Router Token
    Router Token 作为 query，粒度内的 tokens 作为 key/value
    """
    def __init__(self, embed_dim, heads=8, dim_head=64, qkv_bias=True, 
                 drop_out_rate=0., attn_drop_out_rate=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        
        # Query 来自 Router Token
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=qkv_bias)
        # Key 和 Value 来自粒度内的 tokens
        self.to_kv = nn.Linear(embed_dim, inner_dim * 2, bias=qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(drop_out_rate)
        )
    
    def forward(self, router_token, patch_tokens):
        """
        router_token: [B, 1, embed_dim] - Router Token
        patch_tokens: [B, num_patches, embed_dim] - 粒度内的 tokens
        返回: [B, 1, embed_dim] - 聚合后的 Router Token
        """
        q = self.to_q(router_token)  # [B, 1, inner_dim]
        kv = self.to_kv(patch_tokens)  # [B, num_patches, inner_dim * 2]
        k, v = kv.chunk(2, dim=-1)  # 每个 [B, num_patches, inner_dim]
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, heads, 1, num_patches]
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, heads, 1, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, 1, inner_dim]
        out = self.to_out(out)  # [B, 1, embed_dim]
        
        return out


class RouterToken(nn.Module):
    """
    Router Token: 用于聚合每个粒度的全局信息
    类似 CLS token，但用于粒度级别的特征聚合
    """
    def __init__(self, embed_dim, num_granularities):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_granularities = num_granularities
        
        # 为每个粒度创建一个 Router Token
        self.router_tokens = nn.Parameter(torch.randn(1, num_granularities, embed_dim) / embed_dim ** 0.5)
    
    def forward(self, batch_size):
        """
        返回: [B, num_granularities, embed_dim]
        """
        return self.router_tokens.expand(batch_size, -1, -1)


class IntraGranularityAttention(nn.Module):
    """
    粒度内注意力：对每个粒度生成的 Token 序列独立进行 Self-Attention
    支持多层级特征融合（Beginning/Middle/Ending layers）
    """
    def __init__(self, embed_dim, depth, mlp_dim, heads, dim_head, 
                 drop_out_rate=0., attn_drop_out_rate=0., drop_path_rate=0.,
                 use_multilayer_fusion=False, fusion_method='concat'):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_multilayer_fusion = use_multilayer_fusion
        self.fusion_method = fusion_method
        
        # 为每个粒度创建独立的 Transformer Blocks
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                input_dim=embed_dim,
                output_dim=embed_dim,
                hidden_dim=mlp_dim,
                heads=heads,
                dim_head=dim_head,
                qkv_bias=True,
                drop_out_rate=drop_out_rate,
                attn_drop_out_rate=attn_drop_out_rate,
                drop_path_rate=drop_path_rate_list[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 多层级特征融合相关配置
        if self.use_multilayer_fusion:
            # 根据 Lin et al. 的建议，选择 Beginning, Middle, Ending 层
            # 对于 depth=12 的模型，选择第 1 层（Beginning，索引0）、第 6 层（Middle，索引5）、第 12 层（Ending，索引11）
            # 对于其他深度，按比例选择
            if depth >= 12:
                # Beginning (第1层), Middle (第6层), Ending (第12层)
                self.selected_layers = [0, depth // 2 - 1, depth - 1]  # [0, 5, 11] for depth=12
            elif depth >= 6:
                # Beginning (第1层), Middle (中间层), Ending (最后一层)
                self.selected_layers = [0, depth // 2, depth - 1]
            else:
                # 对于浅层网络，选择第1层、中间层、最后一层
                self.selected_layers = [0, depth // 2, depth - 1] if depth > 2 else [0, depth - 1]
            
            # 融合方式：'concat' (拼接) 或 'add' (相加)
            if self.fusion_method == 'concat':
                # 外部直接融合：拼接多层特征后投影回原始维度
                # 输入维度: embed_dim * len(selected_layers), 输出维度: embed_dim
                self.fusion_projection = nn.Linear(embed_dim * len(self.selected_layers), embed_dim)
            elif self.fusion_method == 'add':
                # 相加融合：多层特征相加后可选投影
                # 相加后维度仍为 embed_dim，可选投影层用于特征调整
                self.fusion_projection = nn.Linear(embed_dim, embed_dim)
            else:
                raise ValueError(f"Unsupported fusion_method: {self.fusion_method}. Must be 'concat' or 'add'.")
        else:
            self.selected_layers = []
            self.fusion_projection = None
    
    def forward(self, patch_sequences):
        """
        patch_sequences: List of [B, num_patches, embed_dim]
        返回: List of [B, num_patches, embed_dim] (经过粒度内注意力处理)
        """
        processed_sequences = []
        for patches in patch_sequences:
            x = patches
            
            if self.use_multilayer_fusion:
                # 多层级特征融合：保存选定层的特征
                selected_features = []
                for i, block in enumerate(self.blocks):
                    x = block(x)
                    # 保存选定层的特征（在应用 norm 之前）
                    if i in self.selected_layers:
                        selected_features.append(x)
                
                # 融合方式选择：拼接或相加
                if self.fusion_method == 'concat':
                    # 外部直接融合：在通道维度拼接多层特征
                    # selected_features: [feat1, feat2, feat3], 每个形状为 [B, num_patches, embed_dim]
                    # 拼接后: [B, num_patches, embed_dim * len(selected_layers)]
                    fused_features = torch.cat(selected_features, dim=-1)
                    # 投影回原始维度: [B, num_patches, embed_dim]
                    x = self.fusion_projection(fused_features)
                elif self.fusion_method == 'add':
                    # 相加融合：多层特征直接相加
                    # selected_features: [feat1, feat2, feat3], 每个形状为 [B, num_patches, embed_dim]
                    # 相加后: [B, num_patches, embed_dim]
                    fused_features = sum(selected_features)  # 或使用 torch.stack + sum
                    # 可选投影层用于特征调整
                    x = self.fusion_projection(fused_features)
                
                # 应用 LayerNorm
                x = self.norm(x)
            else:
                # 原始方式：只使用最后一层
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
            
            processed_sequences.append(x)
        
        return processed_sequences


class InterGranularityAttention(nn.Module):
    """
    粒度间注意力：在不同 Router Tokens 之间进行 Self-Attention
    允许不同尺度的特征进行交互
    """
    def __init__(self, embed_dim, num_granularities, heads=8, dim_head=64, 
                 drop_out_rate=0., attn_drop_out_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_granularities = num_granularities
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(drop_out_rate)
        )
    
    def forward(self, router_tokens):
        """
        router_tokens: [B, num_granularities, embed_dim]
        返回: [B, num_granularities, embed_dim] (经过粒度间注意力处理)
        """
        qkv = self.to_qkv(router_tokens).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out


class MMLFormerViT(nn.Module):
    """
    基于 MMLFormer 方法的 ViT 编码器
    实现跨通道多粒度 Patching 和两阶段多粒度自注意力机制
    """
    def __init__(self,
                 num_leads: int,
                 seq_len: int,
                 patch_sizes: list = [32, 64, 128],  # 多粒度 patch sizes
                 width: int = 192,
                 depth: int = 12,
                 mlp_dim: int = 768,
                 heads: int = 3,
                 dim_head: int = 64,
                 drop_out_rate: float = 0.1,
                 attn_drop_out_rate: float = 0.1,
                 drop_path_rate: float = 0.1,
                 enable_augmentation: bool = True,
                 mask_prob: float = 0.15,
                 jitter_std: float = 0.01,
                 use_multilayer_fusion: bool = False,  # 是否启用多层级特征融合
                 fusion_method: str = 'concat',  # 融合方式：'concat' (拼接) 或 'add' (相加)
                 **kwargs):
        super().__init__()
        
        self.num_leads = num_leads
        self.seq_len = seq_len
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)
        self.width = width
        
        # 1. 跨通道多粒度 Patch Embedding
        self.patch_embedding = CrossChannelMultiGranularityPatchEmbedding(
            num_leads=num_leads,
            patch_sizes=patch_sizes,
            embed_dim=width,
            seq_len=seq_len
        )
        
        # 2. Patch-wise 数据增强
        self.augmentation = PatchWiseAugmentation(
            mask_prob=mask_prob,
            jitter_std=jitter_std,
            enable_augmentation=enable_augmentation
        )
        
        # 3. Router Token
        self.router_token = RouterToken(
            embed_dim=width,
            num_granularities=self.num_granularities
        )
        
        # 4. Stage 1: 粒度内注意力 (Intra-Granularity Attention)
        # 支持多层级特征融合（Beginning/Middle/Ending layers）
        self.intra_attention = IntraGranularityAttention(
            embed_dim=width,
            depth=depth,
            mlp_dim=mlp_dim,
            heads=heads,
            dim_head=dim_head,
            drop_out_rate=drop_out_rate,
            attn_drop_out_rate=attn_drop_out_rate,
            drop_path_rate=drop_path_rate,
            use_multilayer_fusion=use_multilayer_fusion,
            fusion_method=fusion_method
        )
        
        # 5. Stage 2: 粒度间注意力 (Inter-Granularity Attention)
        self.inter_attention = InterGranularityAttention(
            embed_dim=width,
            num_granularities=self.num_granularities,
            heads=heads,
            dim_head=dim_head,
            drop_out_rate=drop_out_rate,
            attn_drop_out_rate=attn_drop_out_rate
        )
        
        # 6. 最终融合层：将多粒度特征融合
        self.fusion_norm = nn.LayerNorm(width)
        self.fusion_dropout = nn.Dropout(drop_out_rate)
        
        # 用于聚合每个粒度的特征到 Router Token（使用交叉注意力）
        # Router Token 作为 query，粒度内的 tokens 作为 key/value
        self.aggregation_attn = nn.ModuleList([
            CrossAttention(
                embed_dim=width,
                heads=heads,
                dim_head=dim_head,
                qkv_bias=True,
                drop_out_rate=drop_out_rate,
                attn_drop_out_rate=attn_drop_out_rate
            ) for _ in range(self.num_granularities)
        ])
    
    def forward(self, series, pool='mean'):
        """
        series: [B, num_leads, seq_len]
        返回: [B, total_patches, width] 或 [B, num_granularities, width] (取决于 pool 方式)
        """
        B, C, T = series.shape
        
        # Stage 1: 跨通道多粒度 Patching
        patch_sequences = self.patch_embedding(series)  # List of [B, num_patches, width]
        
        # 数据增强
        patch_sequences = self.augmentation(patch_sequences)
        
        # Stage 2: 粒度内注意力
        processed_sequences = self.intra_attention(patch_sequences)  # List of [B, num_patches, width]
        
        # Stage 3: 聚合每个粒度的特征到 Router Token
        router_tokens = self.router_token(B)  # [B, num_granularities, width]
        
        aggregated_routers = []
        for i, (processed_seq, cross_attn) in enumerate(zip(processed_sequences, self.aggregation_attn)):
            # 使用 Router Token 作为 query，粒度内的所有 tokens 作为 key/value
            router_q = router_tokens[:, i:i+1, :]  # [B, 1, width]
            # 通过交叉注意力聚合粒度内的信息到 Router Token
            aggregated_router = cross_attn(router_q, processed_seq)  # [B, 1, width]
            aggregated_routers.append(aggregated_router.squeeze(1))  # [B, width]
        
        router_tokens = torch.stack(aggregated_routers, dim=1)  # [B, num_granularities, width]
        
        # Stage 4: 粒度间注意力
        fused_routers = self.inter_attention(router_tokens)  # [B, num_granularities, width]
        fused_routers = self.fusion_norm(fused_routers)
        fused_routers = self.fusion_dropout(fused_routers)
        
        # 根据 pool 方式返回结果
        if pool == 'none':
            # 返回所有粒度的所有 patches
            all_patches = torch.cat(processed_sequences, dim=1)  # [B, total_patches, width]
            return all_patches
        elif pool == 'router':
            # 返回融合后的 Router Tokens
            return fused_routers  # [B, num_granularities, width]
        elif pool == 'mean':
            # 返回所有 Router Tokens 的平均值
            return fused_routers.mean(dim=1, keepdim=True)  # [B, 1, width]
        elif pool == 'concat':
            # 拼接所有 Router Tokens
            return fused_routers.view(B, -1)  # [B, num_granularities * width]
        else:
            # 默认返回融合后的 Router Tokens
            return fused_routers


def mmlformer_vit_tiny(num_leads, num_classes=1, seq_len=5000, patch_sizes=[32, 64, 128], **kwargs):
    model_args = dict(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_sizes=patch_sizes,
        width=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs
    )
    return MMLFormerViT(**model_args)


def mmlformer_vit_small(num_leads, num_classes=1, seq_len=5000, patch_sizes=[32, 64, 128], **kwargs):
    model_args = dict(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_sizes=patch_sizes,
        width=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs
    )
    return MMLFormerViT(**model_args)


def mmlformer_vit_middle(num_leads, num_classes=1, seq_len=5000, patch_sizes=[32, 64, 128], **kwargs):
    model_args = dict(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_sizes=patch_sizes,
        width=512,
        depth=12,
        heads=8,
        mlp_dim=2048,
        **kwargs
    )
    return MMLFormerViT(**model_args)


def mmlformer_vit_base(num_leads, num_classes=1, seq_len=5000, patch_sizes=[32, 64, 128], **kwargs):
    model_args = dict(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=seq_len,
        patch_sizes=patch_sizes,
        width=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs
    )
    return MMLFormerViT(**model_args)

