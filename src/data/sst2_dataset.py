import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Tuple, Dict, Any


class Sst2Dataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        split: str,
        datadir: str,
        attention: bool,
    ):
        """
        Dataset with entries from SST-2

        :param tokenizer: instance of a tokenizer
        :param max_seq_len: maximum sequence length
        :param split: split to be loaded
        :param datadir: path to the root folder containing SST/
        :param attention: whether the data entries are loaded to compute xai
        """
        super().__init__()

        self.split = split
        self.tokenizer = tokenizer
        # Load dataset from huggingface
        self.dataset = load_dataset("sst2")
        self.datadir = datadir
        if attention:
            self.samples = self.prepare_dataset_to_compute_attn(max_seq_len)
        else:
            self.samples = self.prepare_dataset(max_seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def prepare_dataset(
        self, max_seq_len: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, np.int]]:
        """Load entries and tokenize them.

        Args:
            max_seq_len (int): Maximum sequence length

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, np.int]]: a tuple of tuples.
            Each inner tuple refers to an entry of the dataset.
            Each inner tuple contains 4 items:
            input_ids: encoded sentence after the tokenizer
            attention_masks: attention mask
            labels: ground-truth class label
            sentence_id: sentence id from the original dataset
        """
        input_ids = []
        attention_masks = []
        data = self.dataset[self.split].to_pandas()
        if self.split == "train":
            data_from_annotations = pd.read_csv(
                os.path.join(self.datadir, "SST", "sst2_trainset_indexes.csv")
            )
            data = data[~data.idx.isin(data_from_annotations["sst2_idxs"])].reset_index(
                drop=True
            )

        for sent in data["sentence"].values:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_seq_len,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the List type
        input_ids = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        attention_masks = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        labels = np.asarray(data["label"].values)
        sentence_id = np.asarray(data["idx"].values)

        assert len(input_ids) == len(attention_masks) == len(labels) == len(sentence_id)
        samples = tuple(zip(input_ids, attention_masks, labels, sentence_id))
        return samples

    def prepare_dataset_to_compute_attn(
        self, max_seq_len: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, np.int, np.int]]:
        """
        Load entries and tokenize them.
        This method loads all the splits necessary
        to compute the attribution scores.

        :param max_seq_len: Maximum sequence length
        :return: a tuple of tuples. Each inner tuple refers to an entry of the dataset.
        Each inner tuple contains 4 items:
            input_ids: encoded sentence after the tokenizer
            attention_masks: attention mask
            labels: ground-truth class label
            sentence_id: sentence id from the original dataset
        """
        input_ids = []
        attention_masks = []
        dfs = []
        for _split in ["train", "test", "validation"]:
            data = self.dataset[_split].to_pandas()
            data_from_annotations = pd.read_csv(
                os.path.join(self.datadir, "SST", f"sst2_{_split}set_indexes.csv")
            )
            dfs.append(
                data[data.idx.isin(data_from_annotations["sst2_idxs"])].reset_index(
                    drop=True
                )
            )

        data = pd.concat(dfs)
        for sent in data["sentence"].values:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_seq_len,  # Pad & truncate all sentences.
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
        labels = np.asarray(data["label"].values)
        sentence_id = np.asarray(data["idx"].values)

        assert len(input_ids) == len(attention_masks) == len(labels) == len(sentence_id)
        samples = tuple(zip(input_ids, attention_masks, labels, sentence_id))
        return samples

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            Dictionary with keys: (inputs, attn_masks, labels, originaldata_id)
            where the input_ids is the tokenized sentence,
            attention_masks the tokenizer att mask,
            labels are the target (sentiment of the sentence),
            originaldata_id is the sentence Id from the original dataset
        """
        input_ids, attention_masks, labels, sentence_ids = self.samples[idx]
        return {
            "inputs": input_ids,
            "attn_masks": attention_masks,
            "labels": labels,
            "originaldata_id": sentence_ids,
        }
