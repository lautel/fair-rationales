import os
import json
import torch
import torch.nn as nn
import logging as log
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extract_model_importance.lrp.xai_config import Config


class BaseModelForSeqClassification(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        Initialize the corresponding model for sequence classification with
        a custom forward pass.

        :param config: Extended model configuration
        """
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name, config=config, num_labels=config.num_labels
        )
        self.base_model = self.model.base_model
        for param in self.model.parameters():
            param.requires_grad = True  # Every parameter requires gradient
        lowercase = "uncased" in config.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name, do_lower_case=lowercase
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        if self.config.num_labels == 1:
            probabilities = nn.functional.sigmoid(outputs.logits)
        else:
            probabilities = nn.functional.softmax(outputs.logits, dim=-1)
        return (
            outputs.loss,
            outputs.logits,
            probabilities,
            outputs.hidden_states,
            outputs.attentions,
        )

    def save(self, output_dir):
        # Save model weights
        weights = self.state_dict()
        torch.save(weights, os.path.join(output_dir, "pytorch_model.bin"))

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                self.config.as_dict(),
                f,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
                indent=4,
            )

    @classmethod
    def from_pretrained(cls, config: Config):
        """Instantiates one of the model classes of the library
        -with the architecture used for pretraining this model–
        from a pretrained model configuration.
        """
        log.info(f"Loading pretrained model from {config.model_path}")

        weights_path = os.path.join(config.model_path, "pytorch_model.bin")
        w_state_dict = torch.load(weights_path, map_location=config.device)

        model = BaseModelForSeqClassification(config)
        model.load_state_dict(state_dict=w_state_dict, strict=False)
        log.info(f"Loaded weights from {weights_path}")
        return model


class EvaluationTracker:
    """Compute and return predictions and prediction errors for sentiment analysis"""

    def __init__(self):
        self.error_dict = {}
        self.preds_dict = {}

    def update_errors(
        self, b_inputs: List[str], output_prob: np.array, labels: np.array
    ) -> None:
        predictions = np.argmax(output_prob, axis=1).flatten()
        errors_mask = np.array([False] * len(labels))
        error = np.array(predictions != labels)
        errors_mask = np.logical_or(errors_mask, error)

        preds_str = [int(p_) for p_ in predictions[errors_mask]]
        target_str = [int(l_) for l_ in labels[errors_mask]]
        input_sents = [str(s_) for s_ in np.asarray(b_inputs)[errors_mask]]
        for s_, p_, l_ in zip(input_sents, preds_str, target_str):
            self.error_dict[s_] = (p_, l_)

    def update_predictions(self, b_inputs: List[str], output_prob: np.array) -> None:
        predictions = np.argmax(output_prob.cpu(), axis=1).flatten()

        assert len(b_inputs) == len(predictions)
        for s_, p_ in zip(b_inputs, predictions):
            self.preds_dict[s_] = int(p_)

    def reset(self) -> None:
        self.error_dict = {}

    def get_error_results(self) -> Dict[str, Tuple]:
        results = self.error_dict
        self.reset()
        return results

    def get_prediction_results(self) -> Dict[str, Tuple]:
        return self.preds_dict
