#!/bin/bash

#SBATCH --job-name diffusion_batch
#SBATCH --time 8:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 32gb

module add anaconda3
source activate torch_env
cd ~/graphics/diffusion-gradients/Python
python batch2.py "$@"
