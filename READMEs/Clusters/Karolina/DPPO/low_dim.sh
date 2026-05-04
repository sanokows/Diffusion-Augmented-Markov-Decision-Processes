#!/bin/bash -l
#SBATCH --job-name Karolina
#SBATCH --account EU-25-100 
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
# (optional) load needed modules

cd /home/it4i-sanokows/code/DIMEReppo
conda activate REPPO
# run your code
chmod +x ./READMEs/Sweeps/DMERL/PPO/DPPO/run_sweep_dmerl_PPO_low_dim.sh
sh ./READMEs/Sweeps/DMERL/PPO/DPPO/run_sweep_dmerl_PPO_low_dim.sh