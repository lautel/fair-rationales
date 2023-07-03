# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# Modified from source:
# https://github.com/AmeenAli/XAI_Transformers/blob/main/utils.py

import dataclasses
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict


@dataclass
class Config:
    def __init__(
        self,
        config,
        max_length,
        n_labels,
        device,
        output_attn=False,
    ):
        self.device = device
        self.max_length = max_length
        self.n_classes = n_labels
        self.num_labels = n_labels  # compatibility HF
        if n_labels in [1, 2]:
            self.problem_type = "single_label_classification"
        else:
            self.problem_type = "multi_label_classification"

        self.output_attn = output_attn
        self.output_hidden_states = output_attn

        if config.model_type in ["distilbert", "distilroberta"]:
            dim = config.dim
            attn_heads = config.n_heads  # 12
            n_layers = config.n_layers  # 6
            layer_norm_eps = 1e-12
            intermediate_size = config.hidden_dim  # 3072
            layer_norm_mode = "distillnorm"
        else:
            dim = config.hidden_size  # 768
            attn_heads = config.num_attention_heads  # 12
            n_layers = config.num_hidden_layers
            layer_norm_eps = config.layer_norm_eps  # 1e-12
            intermediate_size = config.intermediate_size  # 3072 base, 4096 large
            layer_norm_mode = "nowb"

        self.model_type = config.model_type
        try:
            self.base_model_name = config.base_model_name
        except AttributeError:
            self.base_model_name = config.model_type
        self.hidden_size = dim
        self.intermediate_size = intermediate_size
        self.num_attention_heads = attn_heads
        self.num_layers = n_layers
        self.layer_norm_eps = layer_norm_eps
        self.lnv = layer_norm_mode

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        try:
            self.embedding_size = config.embedding_size
            self.num_hidden_groups = config.num_hidden_groups  # used in albert
        except AttributeError:
            self.embedding_size = 0

        self.detach_layernorm = True
        self.train_mode = False
        self.detach_mean = True

        if self.model_type.startswith("deberta"):
            self.relative_attention = config.relative_attention
            self.share_att_key = config.share_att_key
            self.pos_att_type = config.pos_att_type
            self.position_buckets = config.position_buckets
            self.max_relative_positions = config.max_relative_positions
            self.max_position_embeddings = config.max_position_embeddings
            self.norm_rel_ebd = config.norm_rel_ebd

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return Config(**config_dict)


def rename_model_params(model, model_type):
    renamed_state_dict = OrderedDict()
    key_map = {}
    intermediate_dense = False
    for k, v in model.named_parameters():
        k_new = k.replace(f"{model_type}.", "")
        k_new = k_new.replace(f"base_model.", "")
        k_new = k_new.replace(f"model.", "")
        if "transformer.layer" in k_new:
            # distilbert
            k_new = k_new.replace(".attention.", ".")
            k_new = k_new.replace("transformer.layer", "attention_layers")
        elif "encoder.layer" in k_new:
            k_new = k_new.replace("encoder.layer", "attention_layers")
            if ".attention." in k_new:
                k_new = k_new.replace(".attention.", ".")
        elif "encoder." in k_new and model_type.startswith("albert"):
            k_new = k_new.replace("encoder.", "")

        # Renaming of Attention Block
        if "query_proj" in k_new:
            k_new = k_new.replace("self.query_proj", "query")
        elif "key_proj" in k_new:
            k_new = k_new.replace("self.key_proj", "key")
        elif "value_proj" in k_new:
            k_new = k_new.replace("self.value_proj", "value")
        elif "query" in k_new:
            k_new = k_new.replace("self.query", "query")
        elif "key" in k_new:
            k_new = k_new.replace("self.key", "key")
        elif "value" in k_new:
            k_new = k_new.replace("self.value", "value")
        elif "q_lin" in k_new:
            k_new = k_new.replace("q_lin", "query")
        elif "k_lin" in k_new:
            k_new = k_new.replace("k_lin", "key")
        elif "v_lin" in k_new:
            k_new = k_new.replace("v_lin", "value")

        if "output.dense" in k_new:
            if intermediate_dense:
                k_new = k_new.replace("output.dense", "output.ffn_output")
        elif "output.LayerNorm" in k_new:
            if intermediate_dense:
                k_new = k_new.replace("output.LayerNorm", "output.output_layer_norm")
                if "bias" in k_new:
                    intermediate_dense = False
        elif "intermediate.dense" in k_new:
            k_new = k_new.replace("intermediate.dense", "output.ffn")
            intermediate_dense = True

        # Renaming of output classification layers
        if model_type.startswith("distilbert"):
            if "pre_classifier." in k_new:
                k_new = k_new.replace("pre_classifier.", "classifier.dense.")
            elif "classifier." in k_new:
                k_new = k_new.replace("classifier.", "classifier.out_proj.")
            elif ".sa_layer_norm." in k_new:
                k_new = k_new.replace(".sa_layer_norm.", ".output.LayerNorm.")
            elif ".out_lin." in k_new:
                k_new = k_new.replace(".out_lin.", ".output.dense.")
            elif ".ffn.lin1." in k_new:
                k_new = k_new.replace(".ffn.lin1.", ".output.ffn.")
            elif ".ffn.lin2." in k_new:
                k_new = k_new.replace(".ffn.lin2.", ".output.ffn_output.")
            elif ".output_layer_norm." in k_new:
                k_new = k_new.replace(
                    ".output_layer_norm.", ".output.output_layer_norm."
                )

        if k_new.startswith("pooler") and model_type.startswith("albert"):
            k_new = k_new.replace("pooler", "pooler.dense")

        renamed_state_dict[k_new] = v
        key_map[k] = k_new

    return renamed_state_dict, key_map
