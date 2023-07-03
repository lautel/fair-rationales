import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any


class DynasentDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        split: str,
        datadir: str,
        attention: bool,
    ):
        """Dataset with entries from Dynasent

        Args:
            tokenizer (AutoTokenizer): instance of a tokenizer
            max_seq_len (int): maximum sequence length
            split (str): split to be loaded
            datadir (str): path to the root folder containing dynasent-v1.1/
            attention (bool): whether the data entries are loaded to compute xai
        """

        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if attention:
            self.samples = self.prepare_dataset_to_compute_attn(datadir)
        else:
            self.samples = self.prepare_dataset(datadir)

    def __len__(self) -> int:
        return len(self.samples)

    def prepare_dataset(
        self, data_directory: str
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, np.int]]:
        if self.split == "validation":
            split_ = "dev"
        else:
            split_ = self.split
        filename = os.path.join(
            data_directory,
            "dynasent-v1.1",
            "dynasent-v1.1-round02-dynabench-{}.jsonl".format(split_),
        )
        label2integer = {"positive": 0, "negative": 1, "neutral": 2}
        x = []
        y = []
        sentences_id = []
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if d["gold_label"] in label2integer.keys():
                    x.append(d["sentence"])
                    y.append(d["gold_label"])
                    sentences_id.append(int(d["text_id"][len("r2-") :]))

        # Convert string labels to integers matching those in our re-annotated data
        y_int = [label2integer[i] for i in y]

        input_ids = []
        attention_masks = []
        for sent in x:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_seq_len,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        attention_masks = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        labels = torch.tensor(y_int).type(torch.LongTensor)
        sentence_id = torch.tensor(sentences_id)

        assert len(input_ids) == len(attention_masks) == len(labels) == len(sentence_id)
        samples = tuple(zip(input_ids, attention_masks, labels, sentence_id))
        return samples

    def prepare_dataset_to_compute_attn(
        self, directory: str
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, np.int]]:
        label2integer = {"positive": 0, "negative": 1, "neutral": 2}

        # All annotated sentences belong to the test split (480)
        split_ = "test"

        # Open annotations from 1 group to gather the id of sentences annotated
        df = pd.read_csv(
            os.path.join(
                directory, "processed_annotations", "dynasent", "BO_processed.csv"
            )
        )
        sentences_id_str = list(set(df["originaldata_id"]))

        filename = os.path.join(
            directory,
            "dynasent-v1.1",
            "dynasent-v1.1-round02-dynabench-{}.jsonl".format(split_),
        )
        x = []
        y = []
        sentences_id = []
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if (
                    d["text_id"] in sentences_id_str
                    and d["gold_label"] in label2integer.keys()
                ):
                    x.append(d["sentence"])
                    y.append(d["gold_label"])
                    sentences_id.append(int(d["text_id"][len("r2-") :]))

        # Convert string labels to integers matching those in our re-annotated data
        y_int = [label2integer[i] for i in y]

        input_ids = []
        attention_masks = []
        for sent in x:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_seq_len,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        attention_masks = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        labels = torch.tensor(y_int).type(torch.LongTensor)
        sentence_id = torch.tensor(sentences_id)

        assert len(input_ids) == len(attention_masks) == len(labels)
        samples = tuple(zip(input_ids, attention_masks, labels, sentence_id))
        return samples

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            Dictionary with keys: (inputs, attn_masks, labels)
            where the input_ids is the tokenized sentence,
            attention_masks the tokenizer att mask,
            labels are the target (sentiment of the sentence)
        """
        input_ids, attention_masks, labels, sentence_ids = self.samples[idx]
        return {
            "inputs": input_ids,
            "attn_masks": attention_masks,
            "labels": labels,
            "originaldata_id": sentence_ids,
        }
