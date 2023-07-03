# Code adapted from https://github.com/AmeenAli/XAI_Transformers/blob/main/xai_transformer.py
import torch
from torch import nn
from transformers.activations import NewGELUActivation
from transformers.modeling_utils import apply_chunking_to_forward
from extract_model_importance.lrp.xai_transformers_utils import (
    AttentionBlockAlbert,
    LayerNorm,
    detach_ln,
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


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.seq_len_dim = 1
        largs = detach_ln(config)
        self.full_layer_layer_norm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, args=largs
        )
        self.attention = AttentionBlockAlbert(config)

        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = NewGELUActivation()

        # self.store_intermediate_outputs = []

    def forward(self, hidden_states: torch.Tensor, gamma, method):
        attention_output, attention_probs = self.attention(hidden_states, gamma, method)
        attention_output.requires_grad_(True)
        attention_output.retain_grad()
        # ffn_output = apply_chunking_to_forward(
        #     self.ff_chunk,
        #     self.chunk_size_feed_forward,
        #     self.seq_len_dim,
        #     attention_output,
        # )
        ffn_output = self.ff_chunk(attention_output)
        hidden_states_output = self.full_layer_layer_norm(ffn_output + attention_output)
        hidden_states_output.requires_grad_(True)
        hidden_states_output.retain_grad()
        # self.store_intermediate_outputs.append(hidden_states_output)
        return hidden_states_output, attention_probs

    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(1)])

    def forward(self, hidden_states: torch.Tensor, gamma, method):
        attention_probs = None
        for albert_layer in self.albert_layers:
            hidden_states, attention_probs = albert_layer(hidden_states, gamma, method)
            hidden_states.requires_grad_(True)
            hidden_states.retain_grad()
        return (
            hidden_states,
            attention_probs,
        )


class AlbertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()
        self.n_blocks = config.num_layers
        self.embeddings = embeddings
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        self.albert_layer_groups = nn.ModuleList(
            [AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )

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
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        A["hidden_states"] = hidden_states

        attn_input = hidden_states

        for i in range(self.n_blocks):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)
            attn_inputdata.retain_grad()

            A["attn_input_{}_data".format(i)] = attn_inputdata
            A["attn_input_{}".format(i)] = attn_input

            gamma = 0.0 if gammas is None else gammas[i]

            # Index of the hidden group
            group_idx = int(i / (self.n_blocks / self.config.num_hidden_groups))

            output, attention_probs = self.albert_layer_groups[group_idx](
                A["attn_input_{}_data".format(i)], gamma=gamma, method=method
            )
            output.requires_grad_(True)
            output.retain_grad()

            self.attention_probs[i] = attention_probs
            # self.attention_lrp_gradients[i] = output
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
        # Rpool.sum().backward(retain_graph=True)  # !
        # Rpool.sum().backward()

        R_ = Rpool
        for i in range(self.n_blocks)[::-1]:
            R_.sum().backward()
            R_grad = A["attn_input_{}_data".format(i)].grad
            if R_grad is not None:
                R_attn = (R_grad) * A["attn_input_{}".format(i)]
            else:
                print(f">>> R_grad is None")
                R_attn = A["attn_input_{}".format(i)]
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
