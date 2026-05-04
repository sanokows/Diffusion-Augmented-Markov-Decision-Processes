#!/bin/bash -l
#SBATCH --job-name Karolina
#SBATCH --account EU-25-100 
#SBATCH --partition qgpu
#SBATCH --time 42:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# (optional) load needed modules

cd /home/it4i-sanokows/code/DIMEReppo
conda activate REPPO
# run your code
chmod +x ./READMEs/Sweeps/DMERL/vanilla/linear_schedule/run_sweep_dmerl_asp_rerun.sh
sh ./READMEs/Sweeps/DMERL/vanilla/linear_schedule/run_sweep_dmerl_asp_rerun.sh