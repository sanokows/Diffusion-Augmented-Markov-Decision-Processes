#!/bin/bash -l
#SBATCH --job-name Karolina
#SBATCH --account EU-25-100 
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=0.3
#SBATCH --cpus-per-task=32
# (optional) load needed modules

conda activate humanoid_ppo
# run your code
chmod +x ./sweeps/Humanoid/FinalRuns/start_sweeps_DiffPPO.sh
sh ./sweeps/Humanoid/FinalRuns/start_sweeps_DiffPPO.sh