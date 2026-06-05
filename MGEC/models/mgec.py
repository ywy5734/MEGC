import os
import sys

import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlc import TransformerQueryNetwork
from models.mmlformer_vit import (
    mmlformer_vit_base,
    mmlformer_vit_middle,
    mmlformer_vit_small,
    mmlformer_vit_tiny,
)
from models.vit1d import vit_base, vit_middle, vit_small, vit_tiny


MMLFORMER_BUILDERS = {
    "tiny": mmlformer_vit_tiny,
    "small": mmlformer_vit_small,
    "middle": mmlformer_vit_middle,
    "base": mmlformer_vit_base,
}

VIT_BUILDERS = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_middle": vit_middle,
    "vit_base": vit_base,
}


def _mmlformer_size(model_name):
    for size_name in MMLFORMER_BUILDERS:
        if size_name in model_name:
            return size_name
    raise ValueError(f"Unsupported MMLFormer encoder: {model_name}")


def _build_mmlformer_encoder(model_name, config, num_leads, num_classes, patch_sizes):
    builder = MMLFORMER_BUILDERS[_mmlformer_size(model_name)]
    return builder(
        num_leads=num_leads,
        num_classes=num_classes,
        seq_len=config.get("seq_len", 5000),
        patch_sizes=patch_sizes,
        enable_augmentation=config.get("enable_augmentation", True),
        mask_prob=config.get("mask_prob", 0.15),
        jitter_std=config.get("jitter_std", 0.01),
        use_multilayer_fusion=config.get("use_multilayer_fusion", False),
        fusion_method=config.get("mmlformer_fusion_method", "concat"),
    )


def _build_vit_encoder(model_name, config, num_leads, num_classes, patch_size):
    if model_name not in VIT_BUILDERS:
        raise ValueError(f"Unsupported ViT encoder: {model_name}")

    return VIT_BUILDERS[model_name](
        num_leads=num_leads,
        num_classes=num_classes,
        patch_size=patch_size,
        use_multilayer_selection=config.get("use_multilayer_selection", False),
        fusion_method=config.get("fusion_method", "concat"),
    )


def _projection_stack(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _word_embedding_layer(language_model):
    if hasattr(language_model, "embeddings"):
        return language_model.embeddings.word_embeddings
    if hasattr(language_model, "word_embeddings"):
        return language_model.word_embeddings
    raise ValueError("Cannot locate the text encoder word embedding layer.")


def _mask_for_embeds(inputs_embeds, attention_mask):
    if attention_mask is not None:
        return attention_mask

    batch_size, seq_len = inputs_embeds.shape[:2]
    return torch.ones(
        batch_size,
        seq_len,
        device=inputs_embeds.device,
        dtype=torch.long,
    )


class MGEC(torch.nn.Module):
    def __init__(self, device, network_config):
        super(MGEC, self).__init__()

        self.ecg_model = network_config["ecg_model"]
        self.num_leads = network_config["num_leads"]
        self.patch_size = network_config.get("patch_size", 125)
        self.num_classes = network_config["num_classes"]
        self.dropout_rate = network_config["dropout"]
        self.freeze_layers = network_config["freeze_layers"]
        self.device = device

        self.proj_hidden = network_config["projection_head"]["mlp_hidden_size"]
        self.proj_out = network_config["projection_head"]["projection_size"]

        self.use_mmlformer = network_config.get("use_mmlformer", False)
        self.patch_sizes = network_config.get("patch_sizes", [32, 64, 128])
        self.mmlformer_pool = network_config.get("mmlformer_pool", "none")

        use_mmlformer_encoder = self.use_mmlformer or "mmlformer" in self.ecg_model
        if use_mmlformer_encoder:
            model = _build_mmlformer_encoder(
                self.ecg_model,
                network_config,
                self.num_leads,
                self.num_classes,
                self.patch_sizes,
            )
            self.proj_e_input = model.width
            if self.mmlformer_pool == "concat":
                self.proj_e_input = model.width * len(self.patch_sizes)
        else:
            model = _build_vit_encoder(
                self.ecg_model,
                network_config,
                self.num_leads,
                self.num_classes,
                self.patch_size,
            )
            self.proj_e_input = model.width

        self.proj_e = _projection_stack(
            self.proj_e_input,
            self.proj_hidden,
            self.proj_out,
        )
        self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
        self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)

        self.ecg_encoder = model
        print(
            f"[MGEC] ECG encoder config name: {self.ecg_model}, "
            f"actual encoder class: {self.ecg_encoder.__class__.__name__}"
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        text_model_url = network_config["text_model"]
        self.lm_model = AutoModel.from_pretrained(
            text_model_url,
            trust_remote_code=True,
            revision="main",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_url,
            trust_remote_code=True,
            revision="main",
        )

        self.use_coop = network_config.get("use_coop", True)
        self.coop_context_length = network_config.get("coop_context_length", 16)
        self.freeze_class_embedding = network_config.get("freeze_class_embedding", True)
        try:
            self.word_embed_dim = _word_embedding_layer(self.lm_model).embedding_dim
        except ValueError:
            self.word_embed_dim = 768

        if self.use_coop:
            self.context_vectors = nn.Parameter(self._init_context_vectors())
        else:
            self.context_vectors = None

        if self.freeze_layers is not None:
            for layer in self.lm_model.encoder.layer[: int(self.freeze_layers)]:
                for param in layer.parameters():
                    param.requires_grad = False

        if self.use_coop and self.freeze_class_embedding:
            for param in _word_embedding_layer(self.lm_model).parameters():
                param.requires_grad = False

        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )

        self.tqn = TransformerQueryNetwork(embed_dim=self.proj_out, class_num=self.num_classes)

    def _init_context_vectors(self):
        init_prompt = "Standard clinical definition of"
        with torch.no_grad():
            init_tokenized = self.tokenizer(
                init_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            init_embeddings = _word_embedding_layer(self.lm_model)(
                init_tokenized["input_ids"]
            )
            init_seq_len = init_embeddings.shape[1]

            if init_seq_len >= self.coop_context_length:
                init_embeddings = init_embeddings[:, : self.coop_context_length, :]
            else:
                last_token = init_embeddings[:, -1:, :]
                padding = last_token.repeat(
                    1,
                    self.coop_context_length - init_seq_len,
                    1,
                )
                init_embeddings = torch.cat([init_embeddings, padding], dim=1)

        return init_embeddings.squeeze(0)

    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )

    def _project_mmlformer_features(self, ecg_emb):
        if self.mmlformer_pool == "concat":
            return self.proj_e(ecg_emb).unsqueeze(1)

        if self.mmlformer_pool == "mean":
            return self.proj_e(ecg_emb.squeeze(1)).unsqueeze(1)

        if self.mmlformer_pool == "router":
            batch_size, granularity_count, width = ecg_emb.shape
            projected = self.proj_e(ecg_emb.view(batch_size * granularity_count, width))
            return projected.view(batch_size, granularity_count, -1)

        batch_size, patch_count, width = ecg_emb.shape
        projected = self.proj_e(ecg_emb.view(batch_size * patch_count, width))
        return projected.view(batch_size, patch_count, -1)

    def ext_ecg_emb(self, ecg):
        if self.use_mmlformer or "mmlformer" in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg, pool=self.mmlformer_pool)
            return self._project_mmlformer_features(ecg_emb)

        ecg_emb = self.ecg_encoder(ecg)
        return self.proj_e(ecg_emb)

    def get_text_emb(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            attention_mask = _mask_for_embeds(inputs_embeds, attention_mask)
            if self.freeze_layers == 12:
                with torch.no_grad():
                    outputs = self.lm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                    )
                    return outputs.pooler_output

            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            return outputs.pooler_output

        if self.freeze_layers == 12:
            with torch.no_grad():
                return self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).pooler_output

        return self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

    def _coop_text_features(self, text):
        class_count = len(text)
        tokenized_output = self._tokenize(text)
        class_input_ids = tokenized_output.input_ids.to(self.device).contiguous()
        class_attention_mask = tokenized_output.attention_mask.to(self.device).contiguous()

        class_embeddings = _word_embedding_layer(self.lm_model)(class_input_ids)
        context_vectors = self.context_vectors.unsqueeze(0).expand(class_count, -1, -1)
        prompts = torch.cat([context_vectors, class_embeddings], dim=1)

        context_mask = torch.ones(
            class_count,
            self.coop_context_length,
            device=self.device,
            dtype=torch.long,
        )
        prompt_attention_mask = torch.cat([context_mask, class_attention_mask], dim=1)
        return self.get_text_emb(
            inputs_embeds=prompts,
            attention_mask=prompt_attention_mask,
        )

    def _token_text_features(self, text):
        tokenized_output = self._tokenize(text)
        input_ids = tokenized_output.input_ids.to(self.device).contiguous()
        attention_mask = tokenized_output.attention_mask.to(self.device).contiguous()
        return self.get_text_emb(input_ids=input_ids, attention_mask=attention_mask)

    def forward(
        self,
        ecg,
        text,
        return_feats=False,
        return_atten_map=False,
        return_dlc_feats=False,
    ):
        proj_ecg_emb = normalize(self.ext_ecg_emb(ecg), dim=-1)
        proj_ecg_emb = self.dropout1(proj_ecg_emb)

        if self.use_coop:
            pattern_emb = self._coop_text_features(text)
        else:
            pattern_emb = self._token_text_features(text)

        proj_text_emb = self.proj_t(pattern_emb)
        proj_text_emb = normalize(proj_text_emb, dim=-1)
        proj_text_emb = self.dropout2(proj_text_emb)

        dlc_feats = None
        atten_map = None
        if return_dlc_feats and return_atten_map:
            logits, dlc_feats, atten_map = self.tqn(
                proj_ecg_emb,
                proj_text_emb,
                return_atten=True,
                return_feats=True,
            )
        elif return_dlc_feats:
            logits, dlc_feats = self.tqn(
                proj_ecg_emb,
                proj_text_emb,
                return_feats=True,
            )
        elif return_atten_map:
            logits, atten_map = self.tqn(
                proj_ecg_emb,
                proj_text_emb,
                return_atten=True,
            )
        else:
            logits = self.tqn(proj_ecg_emb, proj_text_emb)

        logits = logits.mean(dim=-1)

        if return_feats and return_dlc_feats and return_atten_map:
            return logits, proj_ecg_emb, proj_text_emb, dlc_feats, atten_map
        if return_feats and return_dlc_feats:
            return logits, proj_ecg_emb, proj_text_emb, dlc_feats
        if return_dlc_feats and return_atten_map:
            return logits, dlc_feats, atten_map
        if return_dlc_feats:
            return logits, dlc_feats
        if return_feats and return_atten_map:
            return logits, proj_ecg_emb, proj_text_emb, atten_map
        if return_feats:
            return logits, proj_ecg_emb, proj_text_emb
        if return_atten_map:
            return logits, atten_map

        return logits
