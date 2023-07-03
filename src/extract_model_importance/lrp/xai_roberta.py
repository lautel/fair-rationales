# Code adapted from https://github.com/AmeenAli/XAI_Transformers/blob/main/xai_transformer.py
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from extract_model_importance.lrp.xai_transformers_utils import (
    AttentionBlock,
    LayerNorm,
    LNargsDetachNotMean,
    LNargsDetach,
    LNargs,
    make_p_layer,
)


class RobertaPooler(nn.Module):
    # Copied from transformers.models.bert.modeling_bert.BertPooler
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

        self.detach = config.detach_layernorm
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
        if self.config.train_mode:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # FFN
        x = self.ffn(hidden_states)
        x = nn.functional.gelu(x)
        ffn_output = self.ffn_output(x)
        ffn_output = self.output_layer_norm(ffn_output + hidden_states)
        return ffn_output


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.n_classes)

    def forward(self, features, **kwargs):
        x = features
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config.num_layers
        self.n_blocks = n_blocks
        self.embeddings = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(
            *[
                AttentionBlock(config, RobertaSelfOutput(config))
                for i in range(n_blocks)
            ]
        )

        self.classifier = RobertaClassificationHead(self.config)
        self.device = config.device

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
    ):
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=self.embeddings.token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        ).to(self.config.device)

        attn_input = hidden_states
        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs
            attn_input = output

        # pooled = self.pooler(output)
        logits = self.classifier(output)

        # self.output_ = output
        # self.pooled_ = pooled
        self.logits_ = logits

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {"loss": loss, "logits": logits}

    def forward_and_explain(
        self,
        input_ids,
        cl,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
        gammas=None,
        method=None,
        problem_type=None,
    ):
        # Forward
        A = {}
        hidden_states = self.embeddings(
            input_ids=input_ids,
            # token_type_ids=self.embeddings.token_type_ids
        ).to(self.config.device)

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
                A["attn_input_{}_data".format(i)], gamma=gamma, method=method
            )

            self.attention_probs[i] = attention_probs
            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        # pooled = self.pooler(outputdata)  # A['attn_output'] )
        # (1x768) -> (1,nclasses)
        # pooleddata = pooled.data
        # pooleddata.requires_grad_(True)

        logits = self.classifier(outputdata)

        A["logits"] = logits

        # Through clf layer
        Rout = A["logits"][:, cl]

        # self.R0 = Rout.detach().cpu().numpy()

        Rout.sum().backward()
        # ((pooleddata.grad) * pooled).sum().backward()

        Rpool = (outputdata.grad) * output

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()
            R_grad = A["attn_input_{}_data".format(i)].grad
            R_attn = (R_grad) * A["attn_input_{}".format(i)]
            if method == "GAE":
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

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
