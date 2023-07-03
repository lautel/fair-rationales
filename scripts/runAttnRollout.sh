#!/bin/bash
#SBATCH --job-name=attr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=0-02:00:00

# Example usage:
# sbatch run/runAttnRollout.sh


# Remember to export your wandb API key when using the --use_wandb flag!
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
export PYTHONPATH=$(builtin cd ..; pwd)
export PYTHONWARNINGS="ignore"

# Activate conda environment
. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate fair

DATASET="sst2" 
#DATASET="dynasent" 
#DATASET="cose" 

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
 echo ${BASE_MODEL}
 echo "running attention rollout..."
 python src/run_attention_rollout.py ${DATASET} ${BASE_MODEL} ""
 echo "running topkd rationale to binarize attribution..."
 python src/topkd_rationale.py "attn_rollout" ${DATASET} ${BASE_MODEL}
done


echo "END"
conda deactivate