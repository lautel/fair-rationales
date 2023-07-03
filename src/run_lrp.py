import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset_loader import dataset_loader, SupportedDatasets
from models.seq_classification_model import BaseModelForSeqClassification
from extract_model_importance.lrp.xai_roberta import RobertaAttention
from extract_model_importance.lrp.xai_bert import BertAttention
from extract_model_importance.lrp.xai_distilbert import DistilBertAttention
from extract_model_importance.lrp.xai_albert import AlbertAttention
from extract_model_importance.lrp.xai_deberta import DebertaV2Attention
from extract_model_importance.lrp.xai_config import Config, rename_model_params
from extract_model_importance.tokenization_utils import (
    merge_subwords,
    merge_hyphens,
    merge_symbols,
)
from utils.training_utils import summary_parameters
from transformers import AutoTokenizer, AutoConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    modelname: str,
    dataset: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    simplified: bool,
) -> None:
    """Main method to compute LRP on a model prediction.
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
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = BaseModelForSeqClassification.from_pretrained(config)
    model.zero_grad()
    model.to(device)

    ### Print parameters
    summary_parameters(model)

    renamed_state_dict, key_map = rename_model_params(model, config.model_type)
    # print(key_map)

    # Initialize the explainable model
    print("Loading XAI model")
    if modelname.startswith("distil"):
        xmodel = DistilBertAttention(config, model.base_model.embeddings)
    elif (
        modelname.startswith("bert")
        or modelname.startswith("prajjwal1")
        or "MiniLM-" in modelname
    ):
        xmodel = BertAttention(config, model.base_model.embeddings)
    elif modelname.startswith("albert"):
        xmodel = AlbertAttention(config, model.base_model.embeddings)
    elif modelname.startswith("roberta") or modelname.startswith("facebook"):
        xmodel = RobertaAttention(config, model.base_model.embeddings)
    elif modelname.startswith("microsoft/deberta"):
        xmodel = DebertaV2Attention(config, model.base_model.embeddings)
    else:
        raise (f"Model name {modelname} not supported")

    xmodel.load_state_dict(renamed_state_dict, strict=False)  # ! strict=False
    xmodel.to(device)
    xmodel.eval()

    # Load the data
    print("Loading dataset")
    eval_dataset = dataset_loader(
        dataset,
        tokenizer,
        max_seq_len,
        split="",
        datadir=data_dir,
        simplified=simplified,
        attention=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=2
    )
    tqdm_dataloader = tqdm(eval_dataloader)

    lrp_results = []
    for batch in tqdm_dataloader:
        b_input_ids = batch["inputs"].to(device)
        b_input_mask = batch["attn_masks"].to(device)
        b_sentence_ids = batch["originaldata_id"]

        # Pass it to our finetuned model to get the output
        # print("b_input_ids", b_input_ids.shape)
        # print("b_input_mask", b_input_mask.shape)
        _, _, probs, all_hidden_states, all_attentions = model(
            b_input_ids, attention_mask=b_input_mask
        )
        y_pred = torch.argmax(probs.detach().cpu()).item()

        # Let's explain!
        output_xai = xmodel.forward_and_explain(
            input_ids=b_input_ids,
            cl=y_pred,  # y_true
            labels=torch.tensor([y_pred] * len(b_input_ids)).long().to(device),
            gammas=[0.01] * config.num_layers,
        )
        y_pred_xai = np.argmax(output_xai["logits"].detach().cpu().numpy().squeeze())
        # print("Output xai:", output_xai["logits"].cpu())

        x = output_xai["R"].squeeze() / np.sum(np.abs(output_xai["R"].squeeze()))

        b_input_tokens = tokenizer.decode(b_input_ids[0], skip_special_tokens=True)
        b_input_tokens = b_input_tokens.split()
        b_input_tokens_merged, merged_x = merge_symbols(
            b_input_tokens, x[1 : len(b_input_tokens) + 1]
        )
        b_input_tokens_merged, merged_x = merge_subwords(
            b_input_tokens_merged, merged_x
        )
        b_input_tokens_merged, merged_x = merge_hyphens(b_input_tokens_merged, merged_x)

        original_data_id = b_sentence_ids.cpu().detach().tolist()[0]
        if dataset == SupportedDatasets.DYNASENT:
            original_data_id = "r2-00" + str(original_data_id)
        elif dataset == SupportedDatasets.COSE:
            original_data_id = eval_dataset.sentence_id_enc.inverse_transform(
                [original_data_id]
            )[0]

        lrp_results.append(
            {
                "originaldata_id": original_data_id,
                "tokens_input_ids": b_input_ids.cpu().detach().tolist()[0],
                "tokens_input": b_input_tokens,
                "tokens_merged": b_input_tokens_merged,
                "lrp": merged_x,
                "label": ID2SENTIMENT[y_pred],
            }
        )

    # SAVE RESULTS
    modelname = modelname.replace("/", "_")
    df = pd.DataFrame(
        lrp_results,
        columns=[
            "originaldata_id",  # sst2_id
            "tokens_input_ids",
            "tokens_input",
            "tokens_merged",
            "lrp",
            "label",
        ],
    )
    out_file = os.path.join(output_dir, f"lrp_{modelname}_{dataset}_attention.csv")
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
