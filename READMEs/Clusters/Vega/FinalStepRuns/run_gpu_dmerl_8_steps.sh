#!/bin/bash
#SBATCH --job-name="dmerl_8_steps"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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
chmod +x ./READMEs/Sweeps/DMERL/vanilla/more_steps/Final_Runs/run_sweep_8_steps_tests.sh
bash ./READMEs/Sweeps/DMERL/vanilla/more_steps/Final_Runs/run_sweep_8_steps_tests.sh