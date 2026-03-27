#!/bin/bash
#SBATCH --job-name="dppo_entropy_sweep"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=94GB
#SBATCH --account=d2023d12-053-users
#SBATCH --signal=INT@60

# cd /ceph/hpc/home/eusebastians/code/DMERL
# conda activate REPPO

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/DMERL/PPO/DPPO/run_sweep_reppo_PPO_entropy_coef.sh
bash ./READMEs/Sweeps/DMERL/PPO/DPPO/run_sweep_reppo_PPO_entropy_coef.sh
