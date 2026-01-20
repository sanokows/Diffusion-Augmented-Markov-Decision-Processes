#!/bin/bash
#SBATCH --job-name="WPO_hard_envs"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16GB
#SBATCH --account=d2025d09-019-users
#SBATCH --signal=INT@60

# cd /ceph/hpc/home/eusebastians/code/DMERL
# conda activate REPPO

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/DMERL/WPO/run_sweep_dmerl_WPO_low_dim.sh
sh ./READMEs/Sweeps/DMERL/WPO/run_sweep_dmerl_WPO_low_dim.sh