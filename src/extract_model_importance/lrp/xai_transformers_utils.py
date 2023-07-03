# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import copy
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def make_p_layer(layer, gamma):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight + gamma * layer.weight.clamp(min=0))
    player.bias = torch.nn.Parameter(layer.bias + gamma * layer.bias.clamp(min=0))
    return player


def detach_ln(config):
    if config.detach_layernorm:
        assert not config.train_mode
        if not config.detach_mean:
            # print('Detach LayerNorm only Norm')
            largs = LNargsDetachNotMean()
        else:
            # print('Detach LayerNorm Mean+Norm')
            largs = LNargsDetach()
    else:
        largs = LNargs()
    return largs


# Deberta
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid)
            / torch.log(torch.tensor((max_position - 1) / mid))
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = torch.where(
        abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign
    )
    return bucket_pos


def build_relative_position(
    query_size, key_size, bucket_size=-1, max_position=-1, device=None
):
    """
    Build relative position according to the query and key
    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
        device (`torch.device`): the device on which tensors will be created.
    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """

    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


class LNargs(object):
    def __init__(self, lnv="nowb"):
        self.lnv = lnv
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False


class LNargsDetach(object):
    def __init__(self, lnv="nowb"):
        self.lnv = lnv
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = True
        self.std_detach = True
        if lnv == "distillnorm":
            self.elementwise_affine = True


class LNargsDetachNotMean(object):
    def __init__(self, lnv="nowb"):
        self.lnv = lnv
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True


class LayerNormImpl(nn.Module):
    __constants__ = ["weight", "bias", "eps"]

    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == "no_norm":
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == "no_norm":
            return input
        elif self.mode == "topk":
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps
            )
        elif self.mode == "adanorm":
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm * self.adanorm_scale
        elif self.mode == "nowb":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == "distillnorm":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)

            input_norm = input_norm * self.weight + self.bias

            return input_norm

        elif self.mode == "gradnorm":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output


def LayerNorm(
    normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None
):
    if args is not None:
        if args.lnv != "origin":
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class AttentionBlock(nn.Module):
    def __init__(self, config, self_output):
        super().__init__()

        self.config = config

        # attention
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.attn_gradients = None

        self.output = self_output

        self.detach = True  # config.detach_kq
        if self.config.train_mode:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        if self.detach:
            assert not self.config.train_mode
            # print('Detach K-Q-softmax branch')

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])
        # print(x.size())
        new_x_shape = x.size()[:-1] + (
            self.config.num_attention_heads,
            self.config.attention_head_size,
        )
        x = x.view(*new_x_shape)
        # print(x.size())
        X = x.permute(0, 2, 1, 3)
        return X

    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0, 1, 2, 3)
        return x.reshape(old_shape)

    @staticmethod
    def pproc(layer, player, x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data

    def forward(self, hidden_states, gamma=0, method=None):
        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        n_nodes = hidden_states.shape[1]

        if self.config.train_mode:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([1, 12, 10, 10])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))

        # if torch.isnan(attention_scores).any():
        #    import pdb;pdb.set_trace()

        # if self.detach:
        #     assert not self.config.train_mode
        #     attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        # else:
        #     attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # Assume that we detach
        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)
        if method == "GAE":
            attention_probs.register_hook(self.save_attn_gradients)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        old_context_layer_shape = context_layer.shape
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.config.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            # print('Out gamma', gamma)
            output = self.output.pforward(context_layer, hidden_states, gamma=gamma)

        return (
            output,
            attention_probs,
        )  # , (attention_scores, hidden_states) #, query_t, key_t, val_t)


class AttentionBlockAlbert(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.config = config

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.num_attention_heads * self.attention_head_size
        # print("hidden_size", config.hidden_size)  # albert-base 768
        # print("num_attn_heads", config.num_attention_heads)  # albert-base 12
        # print("attention_head_size", self.attention_head_size)  # albert-base 64
        # print("all_head_size", self.all_head_size)  # albert-base 768

        # attention
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_gradients = None

        # output
        largs = detach_ln(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, args=largs
        )

        self.detach = True  # config.detach_kq
        if self.config.train_mode:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        if self.detach:
            assert not self.config.train_mode
            # print('Detach K-Q-softmax branch')

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.config.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        X = x.permute(0, 2, 1, 3)
        return X

    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0, 1, 2, 3)
        return x.reshape(old_shape)

    @staticmethod
    def pproc(layer, player, x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data

    def output_forward(self, context_layer, hidden_states, gamma):
        pdense = make_p_layer(self.dense, gamma)
        hidden_states_cl = pdense(context_layer)
        if self.config.train_mode:
            hidden_states_cl = self.dropout(hidden_states_cl)
        return self.LayerNorm(hidden_states_cl + hidden_states)

    def forward(self, hidden_states, gamma=0, method=None):
        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        n_nodes = hidden_states.shape[1]

        if self.config.train_mode:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([1, 12, 10, 10])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)
        if method == "GAE":
            attention_probs.register_hook(self.save_attn_gradients)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.transpose(2, 1).flatten(2)

        # projected_context_layer = self.output.dense(context_layer)

        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            # print('Out gamma', gamma)
            output = self.output_forward(context_layer, hidden_states, gamma=gamma)

        return (
            output,
            attention_probs,
        )  # , (attention_scores, hidden_states) #, query_t, key_t, val_t)


class AttentionBlockDeberta(nn.Module):
    def __init__(self, config, self_output):
        super().__init__()

        self.config = config

        # attention
        self.query = nn.Linear(config.hidden_size, config.all_head_size, bias=True).to(
            config.device
        )
        self.key = nn.Linear(config.hidden_size, config.all_head_size, bias=True).to(
            config.device
        )
        self.value = nn.Linear(config.hidden_size, config.all_head_size, bias=True).to(
            config.device
        )

        self.output = self_output

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = (
            config.pos_att_type if config.pos_att_type is not None else []
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            # self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        config.hidden_size, self.all_head_size, bias=True
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(
                        config.hidden_size, self.all_head_size
                    )

        self.detach = True  # config.detach_kq
        if self.config.train_mode:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        if self.detach:
            assert not self.config.train_mode
            # print('Detach K-Q-softmax branch')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    @staticmethod
    def pproc(layer, player, x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data

    def forward(self, hidden_states, rel_embeddings, gamma=0, method=None):
        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        n_nodes = hidden_states.shape[1]

        if self.config.train_mode:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(
            torch.tensor(query_t.size(-1), dtype=torch.float) * scale_factor
        )
        attention_scores = torch.bmm(query_t, key_t.transpose(-1, -2)) / scale.to(
            dtype=query_t.dtype
        )
        if self.relative_attention:
            # rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_t, key_t, None, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.view(
            -1,
            self.config.num_attention_heads,
            attention_scores.size(-2),
            attention_scores.size(-1),
        )

        # attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))

        # attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.bmm(
            attention_probs.view(
                -1, attention_probs.size(-2), attention_probs.size(-1)
            ),
            val_t,
        )
        context_layer = (
            context_layer.view(
                -1,
                self.config.num_attention_heads,
                context_layer.size(-2),
                context_layer.size(-1),
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            # print('Out gamma', gamma)
            output = self.output.pforward(context_layer, hidden_states, gamma=gamma)

        return (
            output,
            attention_probs,
        )  # , (attention_scores, hidden_states) #, query_t, key_t, val_t)

    def disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long()  # .to(query_layer.device)

        rel_embeddings = (
            rel_embeddings[0 : att_span * 2, :].unsqueeze(0).to(query_layer.device)
        )
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query(rel_embeddings)
            ).repeat(query_layer.size(0) // self.config.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key(rel_embeddings)).repeat(
                query_layer.size(0) // self.config.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings)
                ).repeat(
                    query_layer.size(0) // self.config.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings)
                ).repeat(
                    query_layer.size(0) // self.config.num_attention_heads, 1, 1
                )  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(
                torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor
            )
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
                ),
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(
                torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor
            )
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(
                    key_layer.size(-2),
                    key_layer.size(-2),
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                    device=query_layer.device,
                )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).expand(
                    [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
                ),
            ).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score
