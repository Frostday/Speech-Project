#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=h100-80:1
#SBATCH -n 2
#SBATCH --output=output_pred.log
#SBATCH -e output_pred.err

export HF_HOME=/ocean/projects/cis250086p/dgarg2/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

. /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python -u /ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_ctc_finetuning/predict.py

# . /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python -u /ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_ctc_finetuning/predict.py > output_pred.log 2> output_pred.err
