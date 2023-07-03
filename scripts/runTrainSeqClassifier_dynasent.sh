#!/bin/bash
#SBATCH --job-name=dyna-tr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00

# Example usage:
# sbatch run/runTrainSeqClassifier_dynasent.sh


# Remember to export your wandb API key when using the --use_wandb flag!
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
export PYTHONWARNINGS="ignore"

# Activate conda environment
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate fair


DATASET="dynasent"

## declare an array variable
declare -a models=(
"albert-base-v2"
"albert-large-v2"
"nreimers/MiniLM-L6-H384-uncased"
"microsoft/MiniLM-L12-H384-uncased"
"albert-xlarge-v2"
"distilbert-base-uncased"
"distilroberta-base"
"bert-base-uncased"
"roberta-base"
"facebook/muppet-roberta-base"
"microsoft/deberta-v3-base"
"albert-xxlarge-v2"
"bert-large-uncased"
"roberta-large"
"facebook/muppet-roberta-large"
"microsoft/deberta-v3-large"
)

for BASE_MODEL in "${models[@]}"
do
  echo ">>> "${BASE_MODEL}
  python src/train_seq_classifier.py \
    --base_model=${BASE_MODEL} \
    --store_model_with_best="val_acc" \
    --dataset=${DATASET} \
    --n_labels=3 \
    --do_eval \
    --eval_every_epoch=1 \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --num_epochs=3 \
    --learning_rate=0.00002 \
    --warmup_proportion 0.1 \
    --use_wandb \
    --wandb_project="fair-rationales" \
    --wandb_run_name="train_${BASE_MODEL}_${DATASET}"
done

conda deactivate