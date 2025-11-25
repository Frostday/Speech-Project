#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -n 2
#SBATCH --output=output.log
#SBATCH -e output.err

/ocean/projects/cis240137p/dgarg2/miniconda3/envs/genai/bin/python /ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/olmo_finetuning.py
