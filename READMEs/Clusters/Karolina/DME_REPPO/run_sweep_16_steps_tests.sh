#!/bin/bash -l
#SBATCH --job-name dmerl_16_steps
#SBATCH --account EU-25-100
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

cd /home/it4i-sanokows/code/DIMEReppo
conda activate REPPO

SWEEP_SCRIPT="./READMEs/Sweeps/DMERL/vanilla/more_steps/Final_Runs/run_sweep_16_steps_tests.sh"
chmod +x "$SWEEP_SCRIPT"
bash "$SWEEP_SCRIPT"
