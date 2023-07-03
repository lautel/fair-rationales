import os
import json
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Tuple, List, Dict, Any
from sklearn import preprocessing
from extract_model_importance.tokenization_utils import (
    merge_cose_whitespaces_sentence,
)


class CosEDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        split: str,
        datadir: str,
        simplified: bool,
        attention: bool,
    ):
        super().__init__()

        # Make sure max. length is larger than our max. sequence
        assert max_seq_len > 94

        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.one_hot_map = {
            "A": [1, 0, 0, 0, 0],
            "B": [0, 1, 0, 0, 0],
            "C": [0, 0, 1, 0, 0],
            "D": [0, 0, 0, 1, 0],
            "E": [0, 0, 0, 0, 1],
        }
        if attention:
            self.samples = self.prepare_dataset_to_compute_attn(datadir, simplified)
        else:
            self.samples = self.make_dataset(datadir, simplified)

        self.sentence_id_enc = self._fit_encode_sentence_ids()

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _label2int(y: List[str], simplified: bool) -> List[np.int]:
        if simplified:
            label2integer = {"false": 0, "true": 1}
        else:
            label2integer = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        return [label2integer[i] for i in y]

    def _fit_encode_sentence_ids(self) -> preprocessing.LabelEncoder:
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit([item[-1] for item in self.samples])
        return label_enc

    def make_dataset(
        self, data_directory: str, simplified: bool
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, str]]:
        # Using Eraser Benchmark data files
        if self.split == "validation":
            split_ = "val"
        else:
            split_ = self.split

        folder_name = "cose"
        if simplified:
            folder_name += "_simplified"
        filename_queries = os.path.join(
            data_directory, folder_name, "{}.jsonl".format(split_)
        )
        filename_questions = os.path.join(data_directory, folder_name, "docs.jsonl")

        # [sep] in the queries should be replaced with the corresponding seperator token
        sep_token = self.tokenizer.sep_token

        ids_queries = []
        data = defaultdict(defaultdict)
        # Add query, label and evidence
        with open(filename_queries) as f:
            for line in f:
                d = json.loads(line)
                id = d["annotation_id"]
                data[id]["query"] = d["query"].replace("[sep]", sep_token)
                data[id]["label"] = d["classification"]
                data[id]["evidence"] = d["evidences"][0][0]["text"]
                ids_queries.append(id)
        # Add id and question
        with open(filename_questions) as f:
            for line in f:
                d = json.loads(line)
                id_ = d["docid"]
                if id_ in ids_queries:
                    # removing the whitepaces left after spacy tokenization
                    sent, _, _ = merge_cose_whitespaces_sentence(d["document"])
                    data[id_]["question"] = sent

        if simplified:
            x, y_int = zip(
                *[
                    (
                        data[i]["question"] + sep_token + data[i]["query"],
                        self._label2int([data[i]["label"]], simplified)[0],
                    )
                    for i in ids_queries
                ]
            )

            # Downsample negative examples
            y_true = np.where(np.asarray(y_int) == 1)[0]
            y_false = np.where(np.asarray(y_int) == 0)[0][::4]
            y_all = np.concatenate([y_true, y_false])
            y_all.sort()
            x = np.asarray(x)[y_all]
            y_int = np.asarray(y_int)[y_all]
            ids_queries_final = np.asarray(ids_queries)[y_all]

        else:
            x, y_int = zip(
                *[
                    (
                        data[i]["question"]
                        + sep_token
                        + data[i]["evidence"]
                        + sep_token
                        + data[i]["query"],
                        self.one_hot_map[data[i]["label"]],
                    )
                    for i in ids_queries
                ]
            )
            ids_queries_final = np.asarray(ids_queries)

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
        labels = np.asarray(y_int)

        assert (
            len(input_ids)
            == len(attention_masks)
            == len(labels)
            == len(ids_queries_final)
        )
        samples = tuple(zip(input_ids, attention_masks, labels, ids_queries_final))
        return samples

    def prepare_dataset_to_compute_attn(
        self, data_directory: str, simplified: bool
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, str]]:
        split_ = "test"  # All annotated sentences belong to the test split (500)
        folder_name = "cose"
        if simplified:
            folder_name += "_simplified"
        filename_queries = os.path.join(
            data_directory, folder_name, "{}.jsonl".format(split_)
        )
        filename_questions = os.path.join(data_directory, folder_name, "docs.jsonl")

        # [sep] in the queries should be replaced with the corresponding seperator token
        sep_token = self.tokenizer.sep_token

        ids_queries = []
        data = defaultdict(defaultdict)
        # Add query, label and evidence
        with open(filename_queries) as f:
            for line in f:
                d = json.loads(line)
                id = d["annotation_id"]
                data[id]["query"] = d["query"].replace("[sep]", sep_token)
                data[id]["label"] = d["classification"]
                ids_queries.append(id)
        # Add id and question
        with open(filename_questions) as f:
            for line in f:
                d = json.loads(line)
                id_ = d["docid"]
                if id_ in ids_queries:
                    # Remove the whitepaces left after from spacy tokenization
                    sent, _, _ = merge_cose_whitespaces_sentence(d["document"])
                    data[id_]["question"] = sent

        if simplified:
            X, y = zip(
                *[
                    (
                        data[i]["question"] + sep_token + data[i]["query"],
                        data[i]["label"],
                    )
                    for i in ids_queries
                ]
            )
            y_int = self._label2int(y, simplified)
        else:
            X, y_int = zip(
                *[
                    (
                        data[i]["question"]
                        + sep_token
                        + data[i]["evidence"]
                        + sep_token
                        + data[i]["query"],
                        self.one_hot_map[data[i]["label"]],
                    )
                    for i in ids_queries
                ]
            )

        input_ids = []
        attention_masks = []
        for sent in X:
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
        labels = np.asarray(y_int)

        assert len(input_ids) == len(attention_masks) == len(labels) == len(ids_queries)
        samples = tuple(zip(input_ids, attention_masks, labels, ids_queries))
        return samples

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            Dictionary with keys: (inputs, attn_masks, labels)
            where the input_ids is the tokenized sentence,
            attention_masks the tokenizer att mask,
            labels are the target
        """
        input_ids, attention_masks, labels, sentence_ids = self.samples[idx]

        return {
            "inputs": input_ids,
            "attn_masks": attention_masks,
            "labels": labels,
            "originaldata_id": self.sentence_id_enc.transform([sentence_ids]),
        }
