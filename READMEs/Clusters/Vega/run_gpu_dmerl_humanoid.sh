#!/bin/bash
#SBATCH --job-name="dmerl_control"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=16GB
#SBATCH --account=d2025d09-019-users
#SBATCH --signal=INT@60

# conda activate REPPO
# cd /ceph/hpc/home/eusebastians/code/DMERL

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/DMERL/vanilla/linear_schedule/run_sweep_dmerl_humanoid.sh
sh ./READMEs/Sweeps/DMERL/vanilla/linear_schedule/run_sweep_dmerl_humanoid.sh