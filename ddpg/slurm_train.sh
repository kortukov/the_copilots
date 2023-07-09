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

# This is to make render on MuJoCo work
conda activate rl
echo "Starting job name: $SLURM_JOB_NAME"

#  python  train.py -e HockeyWeak -m 25000 --agent TD3 --model results/TD3_Shoot_Def.pth --results-dir results_$SLURM_JOB_NAME
#  python  train.py -e HockeyWeak -m 25000 --agent TD3 --modhel results/TD3_Def_Shoot.pth --results-dir results_$SLURM_JOB_NAME
#  python  train.py -e HockeyNormal -m 25000 --agent TD3 --model results/TD3_Def_Shoot.pth --results-dir results_$SLURM_JOB_NAME
python  train.py -e HalfCheetah-v4 -m 25000 --agent TD3  --results-dir results_$SLURM_JOB_NAME --prioritize