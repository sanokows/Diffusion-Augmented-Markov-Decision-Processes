#!/bin/bash
#SBATCH --job-name="dime_control"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem=4GB
#SBATCH --account=d2023d12-053-users
#SBATCH --signal=INT@60

conda activate REPPO
cd /ceph/hpc/home/eusebastians/code/DMERL

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/run_sweep_dime_control.sh
sh ./READMEs/Sweeps/run_sweep_dime_control.sh