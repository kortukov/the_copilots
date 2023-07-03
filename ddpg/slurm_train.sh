#!/bin/bash
#SBATCH --partition=gpu-2080ti

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G 
#SBATCH --time=0-14:00 
#SBATCH --gres=gpu:1 
#SBATCH --output=log/%x_out 
#SBATCH --error=log/%x_err 


 source ~/.bashrc 
 conda activate rl 
 echo "Starting job name: $SLURM_JOB_NAME"
 python  train.py -e HalfCheetah-v4 -m 10000 --agent TD3 --results-dir results_$SLURM_JOB_NAME

