#!/bin/bash
#SBATCH --job-name=reppo_dmc_prod
# #SBATCH --account=hk-project-p0024023
#SBATCH --account=hk-project-p0022253
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --exclude=hkn0635,hkn0417,hkn0612,hkn0735,hkn0634,hkn0632,hkn0424
#SBATCH --output=logs/reppo_dmc_prod_%j.out
#SBATCH --error=logs/reppo_dmc_prod_%j.err

# Load required modules
module load devel/cuda/12.4

# Activate virtual environment
source $VIRTUAL_ENV

# Run experiment
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Environment name passed as variable
ENV=${ENV:-maniskill}
OVERRIDE=${OVERRIDE:-maniskill}
ENV_NAME=${ENV_NAME:-PickCube-v1}
SEED=${SEED:-0}
CONFIG=${CONFIG:-reppo_dime_maniskill}
ENT_TARGET_MULT=${ENT_TARGET_MULT:-4.0}
OVERRIDES=${OVERRIDES:-default}
WANDB_ENTITY=${WANDB_ENTITY:-bh3136-karlsruhe-institute-of-technology}
WANDB_PROJECT=${WANDB_PROJECT:-MANISKILL}
VMIN=${VMIN:--15}
VMAX=${VMAX:-15}
KL_BOUND=${KL_BOUND:-0.1}
ENT_START=${ENT_START:-1}
KL_ACTION_REP=${KL_ACTION_REP:-4}
TOTAL_TIME_STEPS=${TOTAL_TIME_STEPS:-50000000}

echo "Starting REPPO PYTORCH run with $ENV $ENV_NAME..."

# Mujoco Render
export MUJOCO_GL=egl

echo "Environment: $ENV_NAME"
echo "Seed: $SEED"

# Run the experiment with full 50M steps
python src/torchrl/reppo.py \
    -cn $CONFIG \
    env=$ENV \
    env.name=$ENV_NAME \
    env.vmin=$VMIN \
    env.vmax=$VMAX \
    experiment_overrides=$OVERRIDES \
    hyperparameters.ent_start=$ENT_START \
    hyperparameters.kl_bound=$KL_BOUND \
    hyperparameters.kl_action_rep=$KL_ACTION_REP \
    hyperparameters.ent_target_mult=$ENT_TARGET_MULT \
    hyperparameters.total_time_steps=$TOTAL_TIME_STEPS \
    seed=$SEED \
    wandb.mode=online \
    wandb.entity=$WANDB_ENTITY \
    wandb.project=$WANDB_PROJECT

echo "Training completed!"