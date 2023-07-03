import os
import json
import torch
import logging as log
from datetime import datetime
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig
from data.dataset_loader import dataset_loader
from models.seq_classification_model import BaseModelForSeqClassification
from models.trainer import SeqClassificationModelTrainer
from extract_model_importance.lrp.xai_config import Config
from utils.training_utils import summary_parameters, set_seed
from utils.parser_utils import create_parser

try:
    import wandb

    has_wandb = True
    os.environ["WANDB_MODE"] = "online"
except ImportError:
    has_wandb = False


def main():
    """Main code to train (fine-tune) a text classifier"""
    parser = create_parser()
    args = parser.parse_args()

    # LOAD PROJECT CONFIGURATION
    project_config = json.load(open("config/config.json", "r"))

    # Double-check arguments
    if args.simplified:
        assert args.n_labels == 2

    # Set seed and log level
    set_seed(args.seed)
    numeric_level = getattr(log, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log)
    log.basicConfig(level=numeric_level)
    log.debug(f"Logging level set to {args.log}")

    # Create output dir if it does not exist yet
    output_dir = os.path.join(
        project_config["model_checkpoint"], args.dataset, args.base_model
    )
    if args.add_timestamp_to_output_dir:
        timestamp = "_{:%d%h_%H%M}".format(datetime.today())
        output_dir = output_dir + timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(fp16=args.fp16, cpu=device.type == "cpu")
    log.info(f"Working on device: {device}")

    # Set wandb config
    if args.use_wandb:
        if has_wandb:
            wandb.init(
                project=args.wandb_project,
                entity="lcp",
                name=args.wandb_run_name,
                config=args,
            )
            wandb.run.define_metric("val_acc", summary="max")
            log.info(f"Using wandb in mode {os.environ['WANDB_MODE']}")
        else:
            log.warning(
                "You've requested to log metrics to wandb but package not "
                "found. Metrics not being logged to wandb, try `pip install "
                "wandb`"
            )

    # Initialize model and dataloader
    log.info("Creating new BaseModelForSeqClassification from config")
    config = Config(
        AutoConfig.from_pretrained(args.base_model),
        args.max_seq_len,
        args.n_labels,
        device,
    )
    config.base_model_name = args.base_model

    model = BaseModelForSeqClassification(config=config)

    train_dataset = dataset_loader(
        args.dataset,
        model.tokenizer,
        args.max_seq_len,
        split="train",
        datadir=project_config["data_dir"],
        simplified=args.simplified,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2
    )

    if args.do_eval:
        eval_dataset = dataset_loader(
            args.dataset,
            model.tokenizer,
            args.max_seq_len,
            split="validation",
            datadir=project_config["data_dir"],
            simplified=args.simplified,
        )
        eval_dataloader = DataLoader(
            eval_dataset, shuffle=True, batch_size=args.eval_batch_size, num_workers=2
        )
    else:
        eval_dataloader = None

    trainer = SeqClassificationModelTrainer(
        training_args=args,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    summary_parameters(model, log)
    log.info("***** Running training *****")
    log.info("  Num examples = %d", len(train_dataset))
    log.info("  Batch size = %d", args.train_batch_size)
    log.info("  Num steps = %d", trainer.num_train_optimization_steps)

    # Train
    trainer.train()
    accelerator.print("Training complete. Saving model for future use")
    # model.save(output_dir=output_dir)
    accelerator.unwrap_model(model).save(output_dir)

    if args.do_eval:
        eval_results = trainer.evaluate()
        eval_output_filepath = os.path.join(output_dir, "eval_results.txt")

        if accelerator.is_main_process:
            log.info(f"Writing evaluation results to {eval_output_filepath}")
            with open(eval_output_filepath, "w", encoding="utf-8") as f:
                for k, v in eval_results.items():
                    f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
