#!/bin/bash
#SBATCH --job-name="dmerl_low_dim"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16GB
#SBATCH --account=d2023d12-053-users
#SBATCH --signal=INT@60

# conda activate REPPO
# cd /ceph/hpc/home/eusebastians/code/DMERL

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/run_sweep_dmerl_low_dim.sh
sh ./READMEs/Sweeps/run_sweep_dmerl_low_dim.sh