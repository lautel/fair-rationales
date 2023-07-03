"""Definition of the trainer for the sequence classification model.
It takes care of the training and eval steps."""
import os
import torch
import logging as log
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from utils.training_utils import flat_accuracy, accuracy
from models.seq_classification_model import (
    BaseModelForSeqClassification,
    EvaluationTracker,
)

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SeqClassificationModelTrainer:
    def __init__(
        self,
        training_args,
        accelerator: Accelerator,
        model: BaseModelForSeqClassification,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
    ) -> None:
        """
        Initialize trainer for a sequence classification model.

        :param training_args: Input  training arguments defined in src/utils/parser_utils.
        :param accelerator: Instance of an accelerator for distributed training.
        :param model: Instance of a pretrained model ready for sequence classification.
        :param train_dataloader: dataloader for the train data.
        :param eval_dataloader: [Optional] dataloader for the evaluation data.
        :return: None
        """

        self.training_args = training_args
        self.accelerator = accelerator
        self.model = model
        self.train_dataloader = train_dataloader
        if eval_dataloader is not None:
            self.eval_dataloader = eval_dataloader
            self.eval_tracker = EvaluationTracker()

        # Define other training variables
        self.best_perf_so_far = 0
        self.training_steps = 0
        self.num_train_optimization_steps = (
            int(len(self.train_dataloader) / self.training_args.train_batch_size)
            * self.training_args.num_epochs
        )
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
        self.checkpoint_dir = self.training_args.checkpoint_dir
        self.wandb = has_wandb and self.training_args.use_wandb

    def create_optimizer(self) -> AdamW:
        param_optimizer = self.model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "lr": self.training_args.learning_rate,
        }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        return optimizer

    def create_lr_scheduler(self) -> LambdaLR:
        warmup_steps = (
            self.training_args.warmup_steps
            or self.training_args.warmup_proportion * self.num_train_optimization_steps
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )
        # When the monitored value is not improving,
        # the network performance could be improved by reducing the learning rate.
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.75, patience=0)
        return lr_scheduler

    def log(self, metrics: Dict) -> None:
        if self.wandb and self.accelerator.is_main_process:
            wandb.log(metrics)
        else:
            for k, v in metrics.items():
                self.accelerator.print(f"{k}: {v}")

    def train(self) -> None:
        """Define training loop"""

        # Prepare for devices
        log.info(f"Starting training in {self.accelerator.device}")
        self.model.to(self.accelerator.device)
        (self.model, self.optimizer, self.train_dataloader) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        # Initialize variables
        epochs_trained = 0
        steps_in_epoch_trained = 0
        self.best_perf_so_far = 0

        tqdm_dataloader = tqdm(
            self.train_dataloader, disable=not self.accelerator.is_main_process
        )
        for epoch in range(epochs_trained, self.training_args.num_epochs):
            # Reset the total loss for this epoch
            total_train_loss = 0.0
            total_train_acc = 0.0

            self.model.train()
            for batch_i, (batch) in enumerate(tqdm_dataloader, 1):
                # Skip batches already seen this epoch
                # when resuming from checkpoint
                if epoch == epochs_trained and batch_i < steps_in_epoch_trained:
                    continue

                b_input_ids = batch["inputs"]
                b_input_mask = batch["attn_masks"]
                b_labels = batch["labels"]

                self.optimizer.zero_grad()
                loss, logits, prob, _, _ = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                total_train_loss += loss.item()
                total_train_acc += accuracy(
                    prob.detach().cpu(), b_labels.to("cpu"), self.training_args.n_labels
                )

                self.accelerator.backward(loss)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_args.max_gradient_norm
                )
                self.optimizer.step()
                self.training_steps += 1

            # Calculate the average loss over all of the batches
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            avg_train_acc = total_train_acc / len(self.train_dataloader)

            evaluation_metrics = {}
            if (
                self.training_args.do_eval
                and epoch % self.training_args.eval_every_epoch == 0
            ):
                evaluation_metrics = self.evaluate()
                if self.training_args.store_model_with_best is not None:
                    self.maybe_update_curr_best_model(
                        evaluation_metrics[self.training_args.store_model_with_best],
                        epoch,
                        batch_i,
                    )

            self.lr_scheduler.step(evaluation_metrics["val_loss"])

            # Wait for all processes to finish
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(epoch, batch_i)

            self.log(
                {
                    "epoch": epoch,
                    "step": self.training_steps,
                    "train_loss": avg_train_loss,
                    "train_acc": avg_train_acc,
                    **evaluation_metrics,
                }
            )

    def evaluate(self):
        log.info("Starting evaluation")
        # Prepare for devices
        self.model.to(self.accelerator.device)
        (self.model, self.eval_dataloader) = self.accelerator.prepare(
            self.model, self.eval_dataloader
        )

        self.model.eval()

        # Tracking variables
        val_metrics = {"val_acc": 0.0, "val_loss": 0.0}
        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating batch",
            disable=not self.accelerator.is_main_process,
        ):
            b_input_ids = batch["inputs"]
            b_input_mask = batch["attn_masks"]
            b_labels = batch["labels"]

            with torch.no_grad():
                loss, logits, prob, _, _ = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
            self.eval_tracker.update_predictions(
                self.model.tokenizer.batch_decode(
                    b_input_ids.to("cpu"), skip_special_tokens=True
                ),
                prob,
            )
            if loss and b_labels is not None:
                # Update validation loss only if not testing
                val_metrics["val_loss"] += loss.item()
                prob = prob.detach().cpu()
                b_labels_np = b_labels.to("cpu")

                val_metrics["val_acc"] += accuracy(
                    prob, b_labels_np, self.training_args.n_labels
                )
                self.eval_tracker.update_errors(
                    self.model.tokenizer.batch_decode(
                        b_input_ids.to("cpu"), skip_special_tokens=True
                    ),
                    prob,
                    b_labels_np,
                )

        val_metrics["val_loss"] = val_metrics["val_loss"] / len(self.eval_dataloader)
        val_metrics["val_acc"] = val_metrics["val_acc"] / len(self.eval_dataloader)
        return val_metrics

    def get_full_errors(self) -> Dict[str, Tuple]:
        return self.eval_tracker.get_error_results()

    def get_predictions(self) -> Dict[str, Tuple]:
        return self.eval_tracker.get_prediction_results()

    def maybe_update_curr_best_model(
        self, curr_metric_value: float, epoch: int, step: int
    ) -> None:
        self.accelerator.wait_for_everyone()
        if curr_metric_value > self.best_perf_so_far:
            if self.accelerator.is_main_process:
                self.best_perf_so_far = curr_metric_value
            # Overwrite the best model each time.
            self.save_checkpoint(epoch, step, "best")

    def resume_from_checkpoint(self) -> Tuple[Any, Any]:
        log.info(
            f"Loading optimizer, scheduler, and training states from "
            "{self.checkpoint_dir}"
        )
        self.optimizer.load_state_dict(
            torch.load(
                os.path.join(self.checkpoint_dir, "optimizer.pt"),
                map_location=self.accelerator.device,
            )
        )
        self.lr_scheduler.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, "scheduler.pt"))
        )
        training_state = torch.load(
            os.path.join(self.checkpoint_dir, "training_state.pt")
        )
        return training_state["current_epoch"], training_state["current_step"]

    def save_checkpoint(self, epoch: int, step: int, name_ckpt: str = None) -> None:
        # Output directory
        if name_ckpt is None:
            name_ckpt = epoch
        checkpoint_folder = f"checkpoint-{name_ckpt}"
        run_dir = self.training_args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        self._save_model(output_dir)

        # TODO. This could be improved if saving everything with the accelerator
        if self.accelerator.is_main_process:
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
            )
            torch.save(
                self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
            )
            training_state = {"current_epoch": epoch, "current_step": step + 1}
            torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

    def _save_model(self, output_dir: Optional[str] = None) -> None:
        # Output directory
        if output_dir is None:
            output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.accelerator.print(f"Saving model checkpoint to {output_dir}")

        # Save model config and weights
        self.accelerator.unwrap_model(self.model).save(output_dir)

        if self.accelerator.is_main_process:
            # Save training args as well
            torch.save(
                self.training_args, os.path.join(output_dir, "training_args.bin")
            )
