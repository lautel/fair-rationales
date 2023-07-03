import argparse
from data.dataset_loader import SupportedDatasets


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Args config for training script.")

    # Model
    parser.add_argument(
        "--base_model", type=str, required=True, help="Tranformers model name"
    )
    parser.add_argument("--n_labels", type=int, default=2, help="Number of classes")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Length of the max input sequence"
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=SupportedDatasets,
        required=True,
        choices=list(SupportedDatasets),
    )
    parser.add_argument("--add_timestamp_to_output_dir", action="store_true")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing training checkpoint to resume from",
    )
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Only applies to Cos-E data. Whether to load the simplified version.",
    )

    # Training
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to evaluate the model."
    )
    parser.add_argument(
        "--store_model_with_best",
        type=str,
        default=None,
        help="It should be set to the name of an evaluation metric. "
        "If set the checkpoint with the best such evaluation "
        "metric will be in the 'best' folder.",
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--eval_every_epoch",
        type=int,
        default=1,
        help="Evaluation interval in training epochs.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-2, help="Learning rate"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If passed, will use FP16 training."
    )

    # Seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    # Optimizer
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW"  # 0.0
    )
    parser.add_argument(
        "--max_gradient_norm",
        type=float,
        default=10.0,
        help="Max. norm for gradient norm clipping",
    )

    # Scheduler
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=None,
        type=float,
        help="Number of training steps to perform linear learning rate warmup for. "
        "It overwrites --warmup_proportion.",
    )

    # Logging
    parser.add_argument("--log", type=str, default="DEBUG", help="Logging level")
    parser.add_argument(
        "--use_wandb", action="store_true", help="If passed, will log to wandb."
    )
    parser.add_argument("--wandb_project", type=str, default="fair-rationales")
    parser.add_argument(
        "--wandb_run_name", type=str, default="", help="Name of the wandb run."
    )

    return parser
