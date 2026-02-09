#!/bin/bash -l
#SBATCH --job-name Karolina
#SBATCH --account EU-25-100 
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
# (optional) load needed modules

cd /home/it4i-sanokows/code/DIMEReppo
conda activate REPPO
# run your code
chmod +x ./READMEs/Sweeps/DMERL/vanilla/more_steps/run_sweep_dmerl.sh
sh ./READMEs/Sweeps/DMERL/vanilla/more_steps/run_sweep_dmerl.sh