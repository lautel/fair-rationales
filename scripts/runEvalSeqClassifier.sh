#!/bin/bash
#SBATCH --job-name=freval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem 40GB
#SBATCH --time=0-10:00:00

# Example usage:
# sbatch run/runEvalSeqClassifier.sh


# Remember to export your wandb API key when using the --use_wandb flag!
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
export CUDA_HOME=/usr/local/cuda-11.0

# DATASET="sst2"
# DATASET="dynasent"
DATASET="cose"
BASE_MODEL="roberta-base" 
SPLIT="validation"

python src/eval_seq_classifier.py ${DATASET} ${SPLIT} ${BASE_MODEL} "simplified"