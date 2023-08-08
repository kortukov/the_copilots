#!/bin/bash
#SBATCH --partition=a100

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=0-24:00 
#SBATCH --gres=gpu:1 
#SBATCH --output=log/%x_out 
#SBATCH --error=log/%x_err 


source ~/.bashrc 

# This is to make render on MuJoCo work
conda activate rl
echo "Starting job name: $SLURM_JOB_NAME"

python  train.py -e SelfPlayVar -m 150000 --agent TD3 --model resulting_models/win_strong_td3.pth --results-dir results/$SLURM_JOB_NAME 
# python  train.py -e SelfPlay -m 100000 --agent TD3 --model resulting_models/self_td3.pth --results-dir results/$SLURM_JOB_NAME 
# python  train.py -e HockeyTrainShooting -m 20000 --agent TD3 --model results_defense/TD3_HockeyTrainDefense_20000-eps0.1-t32-l0.0001-s42.pth  --results-dir results_$SLURM_JOB_NAME 

# python  train.py -e Pendulum-v1 -m 5000 --agent TD3  --results-dir results_$SLURM_JOB_NAME --eval-every 250
# python  train.py -e HalfCheetah-v4 -m 20000 --agent DDPG  --results-dir results_$SLURM_JOB_NAME --eval-every 250 

 