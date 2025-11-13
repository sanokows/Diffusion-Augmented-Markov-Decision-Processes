#!/bin/bash

# Job configuration
CLUSTER="batch_a100"      # This is a valid queue according to the document
TIME="10:00"              # 10-hour wall time, within the 168-hour limit for the queue
USER=$(whoami)
GPUS=1
CPUS=16                   # Number of CPU cores for your job (matching SLURM config)

# Correctly define memory per CPU core in MB.
# Example: For a total of 64GB memory with 32 CPUs, this would be 2000 (64000 / 32).
MEMORY_PER_CPU_MB=2000

PWD=$(pwd)

# Environment name passed as variable (with defaults)
VIRTUAL_ENV=${VIRTUAL_ENV:-${PWD}/env_hb/bin/activate}
ENV=${ENV:-humanoid_bench}
OVERRIDE=${OVERRIDE:-humanoid_bench}
ENV_NAME=${ENV_NAME:-h1hand-pole-v0}
SEED=${SEED:-1000}
CONFIG=${CONFIG:-reppo_dime_humanoid_bench}
ENT_TARGET_MULT=${ENT_TARGET_MULT:-0.5}
OVERRIDES=${OVERRIDES:-humanoid_bench}
WANDB_ENTITY=${WANDB_ENTITY:-bh3136-karlsruhe-institute-of-technology}
WANDB_PROJECT=${WANDB_PROJECT:-HUMANOID_BENCH}
VMIN=${VMIN:--250}
VMAX=${VMAX:-250}
KL_BOUND=${KL_BOUND:-0.1}
ENT_START=${ENT_START:-1.0}
KL_ACTION_REP=${KL_ACTION_REP:-4}
TOTAL_TIME_STEPS=${TOTAL_TIME_STEPS:-20000000}
EVAL_ONLY=${EVAL_ONLY:-false}
EVAL_EPISODES=${EVAL_EPISODES:-10}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}
RENDER_INTERVAL=${RENDER_INTERVAL:-0}

# Define the method for the experiment
METHOD="reppo_dime_humanoid_bench"

ENV_LIST=(
    h1hand-pole-v0 
    h1hand-hurdle-v0 
    h1hand-walk-v0 
    h1hand-run-v0 
    h1hand-stand-v0 
)
SEEDS=(1000 1001 1002 1003 1004)
CONFIGS=("reppo_dime_humanoid_bench")

# Loop through seeds and environments
for ENV_NAME in "${ENV_LIST[@]}"; do
    for USING_SEED in "${SEEDS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
    eval "bsub <<EOF
#!/bin/bash -l
#BSUB -J ${METHOD}_${ENV_NAME}_${USING_SEED}
#BSUB -o ${PWD}/jobs/out/out.%J.stdout
#BSUB -e ${PWD}/jobs/error/err.%J.stderr
#BSUB -q ${CLUSTER}
#BSUB -W ${TIME}
#BSUB -M ${MEMORY_PER_CPU_MB}    # Request memory per CPU core
#BSUB -n ${CPUS}                 # Request a total of ${CPUS} job slots (cores)
#BSUB -gpu \"num=${GPUS}\"
#BSUB -R \"span[hosts=1]\"       # Ensure the job runs on a single host

# Load modules & source venv
source /home/${USER}/.bashrc
source ${VIRTUAL_ENV}

echo \"Job ID: \$LSB_JOBID\"
echo \"Node: \$LSB_HOSTS\"
echo \"GPU: \$CUDA_VISIBLE_DEVICES\"

echo \"Starting REPPO PYTORCH run with $ENV \$ENV_NAME...\"

# Mujoco Render
export MUJOCO_GL=egl

echo \"Environment: ${ENV_NAME}\"
echo \"Seed: ${USING_SEED}\"

# Run the experiment
cd ${PWD}

python src/torchrl/reppo_dime.py \\
    -cn ${CONFIG} \\
    env=${ENV} \\
    env.name=${ENV_NAME} \\
    env.vmin=${VMIN} \\
    env.vmax=${VMAX} \\
    experiment_overrides=${OVERRIDES} \\
    hyperparameters.ent_start=${ENT_START} \\
    hyperparameters.kl_bound=${KL_BOUND} \\
    hyperparameters.kl_action_rep=${KL_ACTION_REP} \\
    hyperparameters.ent_target_mult=${ENT_TARGET_MULT} \\
    hyperparameters.total_time_steps=${TOTAL_TIME_STEPS} \\
    hyperparameters.render_interval=${RENDER_INTERVAL} \\
    eval_only=${EVAL_ONLY} \\
    eval_episodes=${EVAL_EPISODES} \\
    checkpoint_path=\"${CHECKPOINT_PATH}\" \\
    seed=${USING_SEED} \\
    wandb.mode=online \\
    wandb.entity=${WANDB_ENTITY} \\
    wandb.project=${WANDB_PROJECT}

echo \"Training completed!\"
EOF"
        done
    done
done