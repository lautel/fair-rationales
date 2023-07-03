import torch
import numpy as np
from torchmetrics import Accuracy


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flat_accuracy(preds: np.array, labels: np.array) -> float:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def accuracy(preds: torch.Tensor, target: torch.Tensor, n_labels: int) -> float:
    preds = torch.max(preds, axis=1).indices
    if n_labels in [1, 2]:
        task = "binary"
    else:
        task = "multiclass"
    accuracy = Accuracy(task=task, num_classes=n_labels)
    acc = accuracy(preds, target)
    return acc.item()


def print_and_log(string: str, logger=None) -> None:
    if logger is None:
        print(string)
    else:
        logger.info(string)


def summary_parameters(model, logger=None, notebook: bool = False) -> None:
    """Print summary with the parameters of the model"""
    if not notebook:
        print_and_log(">> Parameters:", logger)
        parameters = [
            (
                str(n),
                str(v.dtype),
                str(tuple(v.shape)),
                str(v.numel()),
                str(v.requires_grad),
            )
            for n, v in model.named_parameters()
        ]
        max_lens = [max([len(item) + 4 for item in col]) for col in zip(*parameters)]
        raw_format = (
            "|" + "|".join(["{{:{}s}}".format(max_len) for max_len in max_lens]) + "|"
        )
        raw_split = "-" * (sum(max_lens) + len(max_lens) + 1)
        print_and_log(raw_split, logger)
        print_and_log(
            raw_format.format("Name", "Dtype", "Shape", "#Params", "Trainable"), logger
        )
        print_and_log(raw_split, logger)

        for name, dtype, shape, number, grad in parameters:
            print_and_log(raw_format.format(name, dtype, shape, number, grad), logger)
            print_and_log(raw_split, logger)

        num_trainable_params = sum(
            [v.numel() for v in model.parameters() if v.requires_grad]
        )
        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        print_and_log(
            ">> {:25s}\t{:.2f}\tM".format(
                "# TrainableParams:", num_trainable_params / (1.0 * 10**6)
            ),
            logger,
        )
        print_and_log(
            ">> {:25s}\t{:.2f}\tM".format(
                "# NonTrainableParams:", non_trainable_params / (1.0 * 10**6)
            ),
            logger,
        )
        print_and_log(
            ">> {:25s}\t{:.2f}\tM".format(
                "# TotalParams:", total_params / (1.0 * 10**6)
            ),
            logger,
        )
    else:
        parameters = [
            (str(n), str(tuple(v.shape)), str(v.numel()))
            for n, v in model.named_parameters()
        ]
        max_lens = [max([len(item) + 4 for item in col]) for col in zip(*parameters)]
        raw_format = (
            "|" + "|".join(["{{:{}s}}".format(max_len) for max_len in max_lens]) + "|"
        )
        raw_split = "-" * (sum(max_lens) + len(max_lens) + 1)
        print(raw_split)
        print(raw_format.format("Name", "Shape", "#Params"))
        print(raw_split)

        for name, shape, number in parameters:
            print(raw_format.format(name, shape, number))
            print(raw_split)

        num_trainable_params = sum(
            [v.numel() for v in model.parameters() if v.requires_grad]
        )
        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        print(
            ">> {:25s}\t{:.2f}\tM".format(
                "# TrainableParams:", num_trainable_params / (1.0 * 10**6)
            )
        )
        print(
            ">> {:25s}\t{:.2f}\tM".format(
                "# NonTrainableParams:", non_trainable_params / (1.0 * 10**6)
            )
        )
        print(
            ">> {:25s}\t{:.2f}\tM".format(
                "# TotalParams:", total_params / (1.0 * 10**6)
            )
        )
