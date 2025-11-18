#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=output.log
#SBATCH -n 2
#SBATCH -e output.err

export HF_HOME=/ocean/projects/cis250086p/dgarg2/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

. /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python /ocean/projects/cis250187p/dgarg2/project/baseline.py
