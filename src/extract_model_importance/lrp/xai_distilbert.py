# Code adapted from https://github.com/AmeenAli/XAI_Transformers/blob/main/xai_transformer.py
import math
import torch
from torch import nn
from extract_model_importance.lrp.xai_transformers_utils import (
    LayerNorm,
    LNargsDetach,
    make_p_layer,
    LNargsDetachNotMean,
    LNargs,
)


class DistilSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

        largs = LNargsDetach(lnv="distillnorm")

        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, args=largs
        )
        # FFN
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, args=largs
        )

        if self.config.train_mode:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.config.train_mode:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def pforward(self, hidden_states, input_tensor, gamma):
        pdense = make_p_layer(self.dense, gamma)
        hidden_states = pdense(hidden_states)
        # hidden_states = self.dense(hidden_states)
        if self.config.train_mode:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # FFN
        x = self.ffn(hidden_states)
        x = nn.functional.gelu(x)
        ffn_output = self.ffn_output(x)
        ffn_output = self.output_layer_norm(ffn_output + hidden_states)
        return ffn_output


class DistilAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        ## attention
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)

        self.output = DistilSelfOutput(config)
        self.detach = False

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        new_x_shape = x.size()[:-1] + (
            self.config.num_attention_heads,
            self.config.attention_head_size,
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

    def forward(self, hidden_states, mask=None, gamma=0):
        def shape(x):
            """separate heads"""
            return x.view(1, -1, 12, 64).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(1, -1, 12 * 64)

        bs = hidden_states.shape[0]

        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        n_nodes = hidden_states.shape[1]

        query = key = value = hidden_states
        q = shape(pquery(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(pkey(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(pvalue(value))

        q = q / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(2, 3))

        k_length = key.size(1)
        mask_reshp = (bs, 1, 1, k_length)

        # Assume that we detach
        weights = nn.Softmax(dim=-1)(
            scores
        ).detach()  # (bs, n_heads, q_length, k_length)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)

        ffn_output = self.output.pforward(context, hidden_states, gamma)

        return ffn_output, weights


class DistilBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.n_classes, bias=True)

    def forward(self, features, **kwargs):
        x = features
        # x = self.dropout(x)
        x = self.dense(x)
        x = nn.ReLU()(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DistilBertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        self.config = config
        self.embeddings = embeddings
        self.n_blocks = config.num_layers

        self.attention_layers = torch.nn.Sequential(
            *[DistilAttentionBlock(config) for i in range(self.n_blocks)]
        )
        self.classifier = DistilBertClassificationHead(self.config)
        self.attention_probs = {i: [] for i in range(self.n_blocks)}

    def forward_and_explain(
        self,
        input_ids,
        cl,
        attention_mask=None,
        token_type_ids=None,
        # position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
        gammas=None,
        problem_type=None,
    ):
        # Forward
        A = {}
        hidden_states = self.embeddings(input_ids=input_ids)  # .cuda()
        A["hidden_states"] = hidden_states
        attn_input = hidden_states

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)
            A["attn_input_{}_data".format(i)] = attn_inputdata
            A["attn_input_{}".format(i)] = attn_input

            gamma = 0.0 if gammas is None else gammas[i]

            output, attention_probs = block(
                A["attn_input_{}_data".format(i)], gamma=gamma
            )
            self.attention_probs[i] = attention_probs
            attn_input = output

        # (1, 12, 768) -> (1x768)
        output = output[:, 0]

        outputdata = output.data
        outputdata.requires_grad_(True)

        # VARIES WITH THE MODEL
        # pooled = nn.ReLU()(self.pre_classifier(outputdata))  # (bs, dim)

        # (1x768) -> (1,nclasses)
        # pooleddata = pooled.data
        # pooleddata.requires_grad_(True)

        logits = self.classifier(outputdata)
        A["logits"] = logits

        #### Backward ####

        # Through clf layer
        Rout = A["logits"][:, cl]

        try:
            Rout.backward()
        except RuntimeError:
            print("RuntimeError. Attempt .sum()")
            Rout.sum().backward()
        # ((pooleddata.grad) * pooled).sum().backward()

        Rpool = (outputdata.grad) * output
        Rpool.sum().backward(retain_graph=True)  # !

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()
            R_grad = A["attn_input_{}_data".format(i)].grad
            R_attn_ = (R_grad) * A["attn_input_{}".format(i)]
            R_ = R_attn_
        R = R_.sum(2).detach().cpu().numpy()

        loss = None
        if labels is not None:
            if (
                problem_type is not None
                and problem_type == "single_label_classification"
            ):
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
            elif (
                problem_type is not None
                and problem_type == "multi_label_classification"
            ):
                loss = torch.nn.BCEWithLogitsLoss()(logits, labels)

        return {"loss": loss, "logits": logits, "R": R}
