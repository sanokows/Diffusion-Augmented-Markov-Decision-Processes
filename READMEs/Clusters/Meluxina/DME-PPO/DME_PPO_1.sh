#!/bin/bash -l
#SBATCH --job-name=my_job_name             # optional: name of the job
#SBATCH --nodes=1                          # request 1 node
#SBATCH --ntasks=1                         # 8 tasks total
#SBATCH --ntasks-per-node=1                # 8 tasks per node
#SBATCH --gpus-per-node=4                  # request 4 GPUs on this node
#SBATCH --cpus-per-task=8                 # 32 CPU cores for the task
#SBATCH --time=24:00:00                    # walltime limit (HH:MM:SS)
#SBATCH --partition=gpu                    # GPU partition
#SBATCH --account=p201037                  # project account
#SBATCH --qos=default                      # quality of service

# (optional) load needed modules
conda activate REPPO

which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__file__)"
# run your code
chmod +x ./READMEs/Sweeps/DMERL/PPO/run_sweep_dmerl_PPO_control_1.sh
sh ./READMEs/Sweeps/DMERL/PPO/run_sweep_dmerl_PPO_control_1.sh