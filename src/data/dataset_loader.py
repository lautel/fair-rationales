from enum import Enum
from data import Sst2Dataset, CosEDataset, DynasentDataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class SupportedDatasets(str, Enum):
    SST2 = "sst2"
    DYNASENT = "dynasent"
    COSE = "cose"

    def __str__(self):
        return self.value


def dataset_loader(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    split: str = "train",
    datadir: str = None,
    simplified: bool = False,
    attention: bool = False,
) -> Dataset:
    """
    Generic method to load one of the supported datasets.

    Args:
        dataset (str): Name of the dataset to load. Need to match the naming as in SupportedDatasets
        tokenizer (AutoTokenizer): An instance of a tokenizer class
        max_seq_len (int):  Maximum sequence length
        split (str, optional): Split to be loaded (train, validation or test). Defaults to "train".
        datadir (str, optional): Path to the data directory. Defaults to None.
        simplified (bool, optional): Whether we load the Cos-E simplified version (only used if dataset=='cose'). Defaults to False.
        attention (bool, optional): Whether we load the dataset to compute explanations.
        If True, all the needed entries are loaded,
        i.e., those for which we have human annotations (input param 'split' is not used). Defaults to False.

    Raises:
        NotImplementedError: _description_

    Returns:
        Dataset: An instance of the corresponding Dataset class
    """
    if dataset == SupportedDatasets.SST2:
        return Sst2Dataset(tokenizer, max_seq_len, split, datadir, attention)
    elif dataset == SupportedDatasets.DYNASENT:
        return DynasentDataset(tokenizer, max_seq_len, split, datadir, attention)
    elif dataset == SupportedDatasets.COSE:
        return CosEDataset(
            tokenizer, max_seq_len, split, datadir, simplified, attention
        )
    else:
        raise NotImplementedError(dataset)
