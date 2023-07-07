#!/bin/bash
#SBATCH --partition=a100-preemptable

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G 
#SBATCH --time=0-10:00 
#SBATCH --gres=gpu:1 
#SBATCH --output=log/%x_out 
#SBATCH --error=log/%x_err 


 source ~/.bashrc 
 conda activate rl 
 echo "Starting job name: $SLURM_JOB_NAME"
#  python  train.py -e HalfCheetah-v4 -m 10000 --agent TD3 --model results_3/TD3_HalfCheetah-v4_500-eps0.1-t32-l0.0001-s42.pth --results-dir results_$SLURM_JOB_NAME
 python  train.py -e HockeyTrainShooting -m 25000 --agent TD3 --model results_0/TD3_HockeyTrainDefense_1000-eps0.1-t32-l0.0001-s42.pth --results-dir results_$SLURM_JOB_NAME
