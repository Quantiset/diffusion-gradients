#!/bin/bash

#SBATCH --job-name diffusion_batch
#SBATCH --time 2:00:00
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 32gb

module add anaconda3
source activate torch_env
cd graphics/diffusion-gradients/Python
python batch.py 2000 1.273 5 512 0.25 0.0625 1 1.0