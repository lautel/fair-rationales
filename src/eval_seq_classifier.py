import os
import sys
import json
import torch
import numpy as np
from transformers import AutoConfig
from torch.utils.data import DataLoader
from data.dataset_loader import SupportedDatasets, dataset_loader
from models.seq_classification_model import BaseModelForSeqClassification
from extract_model_importance.lrp.xai_config import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main(
    modelname: str, dataset: str, split: str, model_path: str, data_dir: str
) -> None:
    """Code to evaluate a trained text classifier

    Args:
        modelname (str): Name of the model type as in huggigface
        dataset (str): dataset to evaluate on
        split (str): split to use for evaluation
        model_path (str): path to the folder with the model checkpoint
        data_dir (str): path to the directory containing the annotations
    """

    num_labels = len(ID2SENTIMENT)
    # Load the model and send to device
    modelname = modelname.split("_")[0]
    config = Config(
        AutoConfig.from_pretrained(modelname),
        max_seq_len,
        num_labels,
        device,
        output_attn=True,
    )
    config.base_model_name = modelname
    config.model_path = model_path
    model = BaseModelForSeqClassification.from_pretrained(config)
    model.zero_grad()
    model.to(device)

    # Load the data
    eval_dataset = dataset_loader(
        dataset,
        model.tokenizer,
        max_seq_len,
        split=split,
        datadir=data_dir,
        simplified=simplified,
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=2
    )

    # Evaluate
    ok = 0
    ko = 0
    for batch in eval_dataloader:
        # print("batch size", len(batch["inputs"]))
        b_input_ids = batch["inputs"].to(device)
        b_input_mask = batch["attn_masks"].to(device)
        b_sentence_ids = batch["originaldata_id"]
        _, _, probs, _, _ = model(
            b_input_ids, token_type_ids=None, attention_mask=b_input_mask
        )
        true_label = ""
        if "labels" in batch:
            true_label = batch["labels"].cpu().item()
        if num_labels == 1:
            sentiment_id = probs.detach().cpu().flatten() > 0.5
            if sentiment_id.item():
                sentiment_id = 1
            else:
                sentiment_id = 0
        else:
            sentiment_id = torch.argmax(probs.detach().cpu().flatten())
            sentiment_id = sentiment_id.item()

        # merge word pieces if necessary
        b_input_tokens = model.tokenizer.decode(
            b_input_ids[0], skip_special_tokens=True
        )
        b_input_tokens = b_input_tokens.split()

        original_data_id = b_sentence_ids.cpu().detach().tolist()[0]
        if dataset == SupportedDatasets.DYNASENT:
            original_data_id = "r2-00" + str(original_data_id)
        elif dataset == SupportedDatasets.COSE:
            original_data_id = eval_dataset.sentence_id_enc.inverse_transform(
                [original_data_id]
            )[0]

        # Uncomment to inspect predictions
        # print(
        #     f"Id:{original_data_id}, Input:{b_input_tokens}, Pred:{sentiment_id}|True:{true_label}, probs:{probs}"
        # )
        if sentiment_id == true_label:
            ok += 1
        else:
            ko += 1
    print(f"Accuracy: {100*(ok/(ok+ko))}%")


if __name__ == "__main__":
    # Read input parameters
    dataset = sys.argv[1]
    split = sys.argv[2]
    modelname = sys.argv[3]
    simplified = sys.argv[4] == "simplified"
    print(dataset, split, modelname, simplified)

    # LOAD PROJECT CONFIGURATION
    project_config = json.load(open("config/config.json", "r"))

    # Define fix variables

    eval_batch_size = 1
    max_seq_len = 128

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

    # Set random seed
    RANDOM_SEED = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Working on device {device}")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    main(modelname, dataset, split, model_path, project_config["data_dir"])
