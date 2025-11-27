#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -n 2
#SBATCH --output=output_pred_2.log
#SBATCH -e output_pred_2.err

/ocean/projects/cis240137p/dgarg2/miniconda3/envs/genai/bin/python -u /ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/predict_2.py
