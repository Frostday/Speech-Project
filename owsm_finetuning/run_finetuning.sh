#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=h100-80:1
#SBATCH -n 2
#SBATCH --output=output.log
#SBATCH -e output.err

. /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python /ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/espnet_owsm_finetuning.py

# . /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python /ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/espnet_owsm_finetuning.py > logs.txt 2> errors.txt
