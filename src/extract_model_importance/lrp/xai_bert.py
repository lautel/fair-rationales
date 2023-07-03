# Code adapted from https://github.com/AmeenAli/XAI_Transformers/blob/main/xai_transformer.py
from torch import nn
import torch
from extract_model_importance.lrp.xai_transformers_utils import (
    AttentionBlock,
    LayerNorm,
    LNargsDetachNotMean,
    LNargsDetach,
    LNargs,
    make_p_layer,
)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output1 = self.dense(first_token_tensor)
        pooled_output2 = self.activation(pooled_output1)
        return pooled_output2


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
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

    # def forward(self, hidden_states, input_tensor):
    #     hidden_states = self.dense(hidden_states)
    #     if self.config.train_mode:
    #         hidden_states = self.dropout(hidden_states)
    #     hidden_states = self.LayerNorm(hidden_states + input_tensor)
    #     return hidden_states

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


class BertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        self.n_blocks = config.num_layers
        self.embeddings = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(
            *[
                AttentionBlock(config, BertSelfOutput(config))
                for _ in range(self.n_blocks)
            ]
        )

        # self.output = BertSelfOutput(config)

        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(
            in_features=config.hidden_size, out_features=config.n_classes, bias=True
        )
        self.device = config.device

        self.attention_probs = {i: [] for i in range(self.n_blocks)}
        self.attention_debug = {i: [] for i in range(self.n_blocks)}
        self.attention_gradients = {i: [] for i in range(self.n_blocks)}
        self.attention_cams = {i: [] for i in range(self.n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(self.n_blocks)}

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
    ):
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        ).to(self.config.device)

        attn_input = hidden_states
        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs
            attn_input = output

        pooled = self.pooler(output)
        logits = self.classifier(pooled)

        self.output_ = output
        self.pooled_ = pooled
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
        token_type_ids=None,
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
        hidden_states = self.embeddings(input_ids=input_ids).to(self.config.device)

        if self.config.base_model_name.startswith("albert"):
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)

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

        pooled = self.pooler(outputdata)  # A['attn_output'] )

        # (1x768) -> (1,nclasses)
        pooleddata = pooled.data
        pooleddata.requires_grad_(True)

        logits = self.classifier(pooleddata)

        A["logits"] = logits

        # Through clf layer
        Rout = A["logits"][:, cl]

        # self.R0 = Rout.detach().cpu().numpy()

        Rout.backward()
        ((pooleddata.grad) * pooled).sum().backward()

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
