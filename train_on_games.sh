#!/bin/bash
#SBATCH --partition=a100

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=0-30:00 
#SBATCH --gres=gpu:1 
#SBATCH --output=log/train_out
#SBATCH --error=log/train_err


source ~/.bashrc 

# This is to make render on MuJoCo work
conda activate rl
echo "Starting job name: $SLURM_JOB_NAME"

python train_on_games.py
 
