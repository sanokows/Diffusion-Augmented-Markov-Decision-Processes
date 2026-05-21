#!/bin/bash -l
#SBATCH --job-name Karolina
#SBATCH --account EU-25-100
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

cd /home/it4i-sanokows/code/DIMEReppo
conda activate REPPO

SWEEP_SCRIPT="./READMEs/Sweeps/DMERL/vanilla/linear_schedule/run_sweep_dmerl_env_time_fish_hopper.sh"
chmod +x "$SWEEP_SCRIPT"
bash "$SWEEP_SCRIPT"
