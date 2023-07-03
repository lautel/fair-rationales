import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from data.dataset_loader import SupportedDatasets, dataset_loader
from extract_model_importance.attention_flows import compute_joint_attention
from extract_model_importance.tokenization_utils import (
    merge_subwords,
    merge_hyphens,
    merge_symbols,
)
from extract_model_importance.lrp.xai_config import Config
from models.seq_classification_model import BaseModelForSeqClassification
from utils.training_utils import summary_parameters


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

CASE = "rollout_2"


def main(
    modelname: str,
    dataset: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    simplified: bool,
) -> None:
    """Main method to compute Attention Rollout on a model prediction.
    Results are stored in a CSV file.

    Args:
        modelname (str): name of the model to be loaded (following huggingface naming)
        dataset (str): name of the dataset used
        model_path (str): path to the root folder with the model checkpoint
        data_dir (str): path to the directory containing the annotations
        output_dir (str): path to the output directory
        simplified (bool): whether to use the simplified version of Cos-E. Only applies if dataset=='cose'
    """

    # Define variables
    eval_batch_size = 1
    max_seq_len = 128
    add_residual_connections = True

    # Load the configuration and the model
    print(f"Loading model from {model_path}")
    config = Config(
        AutoConfig.from_pretrained(modelname),
        max_seq_len,
        len(ID2SENTIMENT),
        device,
        output_attn=True,
    )
    config.base_model_name = modelname
    config.model_path = model_path
    if "distilroberta" in modelname:
        config.num_hidden_layers = 6
    model = BaseModelForSeqClassification.from_pretrained(config)
    model.zero_grad()
    model.to(device)

    ### Print parameters
    summary_parameters(model)

    # Load the data
    eval_dataset = dataset_loader(
        dataset,
        model.tokenizer,
        max_seq_len,
        split="",
        datadir=data_dir,
        simplified=simplified,
        attention=True,  # Load only examples with annotations (from the corresponding splits)
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=2
    )
    tqdm_dataloader = tqdm(eval_dataloader)

    attn_results = []
    idx_layer = int(CASE.replace("rollout_", ""))
    for batch in tqdm_dataloader:
        b_input_ids = batch["inputs"].to(device)
        b_input_mask = batch["attn_masks"].to(device)
        b_sentence_ids = batch["originaldata_id"]
        _, _, probs, _, all_attentions = model(
            b_input_ids, token_type_ids=None, attention_mask=b_input_mask
        )

        sentiment_id = torch.argmax(probs.detach().cpu()).item()

        # Compute attention rollout
        _attns = [att.detach().cpu().numpy() for att in all_attentions]
        attentions_mat = np.stack(_attns, axis=0).squeeze()
        # print("attn shape:", attentions_mat.shape)
        res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
        joint_attentions = compute_joint_attention(
            res_att_mat, add_residual=add_residual_connections
        )

        attention_tokens = joint_attentions[idx_layer].sum(0)

        # Merge tokens if necessary
        b_input_tokens = model.tokenizer.decode(
            b_input_ids[0], skip_special_tokens=True
        )
        b_input_tokens = b_input_tokens.split()
        b_input_tokens_merged, merged_attention = merge_symbols(
            b_input_tokens, attention_tokens[1 : len(b_input_tokens) + 1]
        )
        b_input_tokens_merged, merged_attention = merge_subwords(
            b_input_tokens_merged, merged_attention
        )
        b_input_tokens_merged, merged_attention = merge_hyphens(
            b_input_tokens_merged, merged_attention
        )

        original_data_id = b_sentence_ids.cpu().detach().tolist()[0]
        if dataset == SupportedDatasets.DYNASENT:
            original_data_id = "r2-00" + str(original_data_id)
        elif dataset == SupportedDatasets.COSE:
            original_data_id = eval_dataset.sentence_id_enc.inverse_transform(
                [original_data_id]
            )[0]

        attn_results.append(
            {
                "originaldata_id": original_data_id,
                "tokens_input_ids": b_input_ids.cpu().detach().tolist()[0],
                "tokens_input": b_input_tokens,
                "tokens_merged": b_input_tokens_merged,
                "attention": merged_attention,
                "label": ID2SENTIMENT[sentiment_id],
            }
        )

    # SAVE RESULTS
    modelname = modelname.replace("/", "_")
    df = pd.DataFrame(
        attn_results,
        columns=[
            "originaldata_id",
            "tokens_input_ids",
            "tokens_input",
            "tokens_merged",
            "attention",
            "label",
        ],
    )
    out_file = os.path.join(
        output_dir, f"attn_rollout_{modelname}_{dataset}_attention.csv"
    )
    df.to_csv(out_file)
    print(out_file, " saved")


if __name__ == "__main__":
    # READ INPUT PARAMETERS
    dataset = sys.argv[1]
    modelname = sys.argv[2]
    simplified = sys.argv[3] == "simplified"
    print("ARGS:", dataset, modelname, simplified)

    # LOAD PROJECT CONFIGURATION
    project_config = json.load(open("config/config.json", "r"))

    # DEFINE VARIABLES
    if dataset == SupportedDatasets.SST2:
        ID2SENTIMENT = {0: "negative", 1: "positive"}
        model_path = f"{project_config['model_checkpoint']}/{dataset}/{modelname}/checkpoint-best"

    elif dataset == SupportedDatasets.DYNASENT:
        ID2SENTIMENT = {0: "positive", 1: "negative", 2: "neutral"}
        model_path = f"{project_config['model_checkpoint']}/{dataset}/{modelname}/checkpoint-best"
    else:
        if simplified:
            ID2SENTIMENT = {0: "false", 1: "true"}
            model_path = f"{project_config['model_checkpoint']}/{dataset}/simplified/{modelname}/checkpoint-best"
        else:
            ID2SENTIMENT = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
            model_path = f"{project_config['model_checkpoint']}/{dataset}/{modelname}/checkpoint-best"

    # DEFINE OUTPUT DIR
    output_dir = os.path.join(project_config["model_importance"], "attn_rollout")
    os.makedirs(output_dir, exist_ok=True)

    # FIX RANDOM SEED AND MOVE TO DEVICE
    RANDOM_SEED = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on device {device}")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    main(
        modelname,
        dataset,
        model_path,
        project_config["data_dir"],
        output_dir,
        simplified,
    )
