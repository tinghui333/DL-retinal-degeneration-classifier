#!/bin/bash
#SBATCH -p zhangz2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --job-name=OCT_large

source /common/wut4/src/mambaforge/etc/profile.d/conda.sh

conda activate oct

python train.py \
        --input-size 1000 \
        -v