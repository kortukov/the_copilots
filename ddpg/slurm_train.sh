#!/bin/bash
#SBATCH --partition=a100

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G 
#SBATCH --time=0-20:00 
#SBATCH --gres=gpu:1 
#SBATCH --output=log/%x_out 
#SBATCH --error=log/%x_err 


source ~/.bashrc 

# This is to make render on MuJoCo work
conda activate rl
echo "Starting job name: $SLURM_JOB_NAME"

 python  train.py -e HockeyWeak -m 60000 --agent TD3 --model results_weak_cont/TD3_HockeyWeak_30000-eps0.1-t32-l0.0001-s42.pth --results-dir results_$SLURM_JOB_NAME 
#  python  train.py -e HockeyNormal -m 25000 --agent TD3 --model results/TD3_Def_Shoot.pth --results-dir results_$SLURM_JOB_NAME
# python  train.py -e HockeyTrainShooting -m 20000 --agent TD3 --model results_defense/TD3_HockeyTrainDefense_20000-eps0.1-t32-l0.0001-s42.pth  --results-dir results_$SLURM_JOB_NAME 
 