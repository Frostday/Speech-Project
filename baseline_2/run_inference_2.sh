#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=h100-32:1
#SBATCH --output=output.log
#SBATCH -n 2
#SBATCH -e output.err

. /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python /ocean/projects/cis250187p/dgarg2/Speech-Project/baseline_2/baseline_2.py

# . /ocean/projects/cis250187p/dgarg2/espnet/tools/activate_python.sh && python /ocean/projects/cis250187p/dgarg2/Speech-Project/baseline_2/baseline_2.py > output.log 2> output.err
