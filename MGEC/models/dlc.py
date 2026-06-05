import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DualLocalCrossNetwork(nn.Module):
    def __init__(self, embed_dim: int, class_num: int):
        super().__init__()
        self.d_model = embed_dim
        self.class_num = class_num

        self.local_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.local_norm = nn.LayerNorm(embed_dim)

        decoder_layer = DLCDecoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
        )

        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = DLCDecoderStack(
            decoder_layer,
            num_layers=4,
            norm=self.decoder_norm,
            return_intermediate=False,
        )
        self.dropout_feats = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(self.d_model, self.class_num))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, image_features, text_features, return_atten=False, return_feats=False):
        batch_size = image_features.shape[0]

        local_feats = self.local_conv(image_features.transpose(1, 2)).transpose(1, 2)
        image_features = self.local_norm(image_features + local_feats)

        image_features = image_features.transpose(0, 1)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)

        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)

        features, atten_map = self.decoder(
            text_features,
            image_features,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )

        features = self.dropout_feats(features).transpose(0, 1)
        out = self.mlp_head(features)

        outputs = [out]
        if return_feats:
            outputs.append(features)
        if return_atten:
            outputs.append(atten_map)

        if len(outputs) > 1:
            return tuple(outputs)
        return out


class DLCDecoderStack(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _clone_layers(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output, attn_weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                residual=True,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, attn_weights


class DLCDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _activation_fn(activation)
        self.normalize_before = normalize_before

        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=False,
        )
        self.local_proj = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Parameter(torch.zeros(1))

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        residual=True,
    ):
        query = key = self.with_pos_embed(tgt, query_pos)
        _, attn_weights = self.self_attn(
            query,
            key,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt)
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        query = key = self.with_pos_embed(tgt2, query_pos)
        tgt2, _ = self.self_attn(
            query,
            key,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2_global, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
        )

        local_memory = self.local_conv(memory.permute(1, 2, 0)).permute(0, 2, 1)
        local_out = torch.bmm(attn_weights, local_memory).transpose(0, 1)
        local_out = self.local_proj(local_out)
        tgt2_fused = tgt2_global + self.fusion_gate * local_out

        tgt = tgt + self.dropout2(tgt2_fused)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, attn_weights

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        residual=True,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            residual,
        )


def _clone_layers(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


TransformerQueryNetwork = DualLocalCrossNetwork
