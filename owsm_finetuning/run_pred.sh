#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=h100-80:1
#SBATCH -n 2
#SBATCH --output=output_pred.log
#SBATCH -e output_pred.err

. /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python -u /ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/predict.py
