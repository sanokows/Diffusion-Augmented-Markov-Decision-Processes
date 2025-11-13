#!/bin/bash
#SBATCH --job-name=reppo_dmc_prod
# #SBATCH --account=hk-project-p0022253
#SBATCH --account=hk-project-p0024023
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --mem=128G
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
ENV_NAME=${ENV_NAME:-PickCube-v1}
SEED=${SEED:-0}
CONFIG=${CONFIG:-reppo_dime_maniskill}
ENT_TARGET_MULT=${ENT_TARGET_MULT:-4.0}
OVERRIDES=${OVERRIDES:-humanoid_bench}
WANDB_ENTITY=${WANDB_ENTITY:-bh3136-karlsruhe-institute-of-technology}
WANDB_PROJECT=${WANDB_PROJECT:-MANISKILL}
VMIN=${VMIN:--250}
VMAX=${VMAX:-250}
KL_BOUND=${KL_BOUND:-0.1}
ENT_START=${ENT_START:-1}
KL_ACTION_REP=${KL_ACTION_REP:-4}
DIFF_STEPS=${DIFF_STEPS:-8}
TOTAL_TIME_STEPS=${TOTAL_TIME_STEPS:-50000000}
RENDER_INTERVAL=${RENDER_INTERVAL:-0}
SAVE_CHECKPOINT_INTERVAL=${SAVE_CHECKPOINT_INTERVAL:-0}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-null}
RENDER_DIR=${RENDER_DIR:-null}
SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-false}

# print out all env variable here for debug
echo "=== Environment Variables Debug ==="
echo "ENV: $ENV"
echo "ENV_NAME: $ENV_NAME"
echo "SEED: $SEED"
echo "CONFIG: $CONFIG"
echo "ENT_TARGET_MULT: $ENT_TARGET_MULT"
echo "OVERRIDES: $OVERRIDES"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "VMIN: $VMIN"
echo "VMAX: $VMAX"
echo "KL_BOUND: $KL_BOUND"
echo "ENT_START: $ENT_START"
echo "KL_ACTION_REP: $KL_ACTION_REP"
echo "DIFF_STEPS: $DIFF_STEPS"
echo "TOTAL_TIME_STEPS: $TOTAL_TIME_STEPS"
echo "RENDER_INTERVAL: $RENDER_INTERVAL"
echo "SAVE_CHECKPOINT_INTERVAL: $SAVE_CHECKPOINT_INTERVAL"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "RENDER_DIR: $RENDER_DIR"
echo "SAVE_FINAL_CHECKPOINT: $SAVE_FINAL_CHECKPOINT"
echo "=================================="


echo "Starting REPPO DIME PYTORCH run with $ENV $ENV_NAME..."

# Mujoco Render
# export MUJOCO_GL=egl

echo "Environment: $ENV_NAME"
echo "Seed: $SEED"

# Run the experiment with full 50M steps
python src/torchrl/reppo_dime.py \
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
    hyperparameters.diffusion.diff_steps=$DIFF_STEPS \
    hyperparameters.total_time_steps=$TOTAL_TIME_STEPS \
    hyperparameters.render_interval=$RENDER_INTERVAL \
    save_checkpoint_interval=$SAVE_CHECKPOINT_INTERVAL \
    checkpoint_dir=$CHECKPOINT_DIR \
    render_dir=$RENDER_DIR \
    save_final_checkpoint=$SAVE_FINAL_CHECKPOINT \
    seed=$SEED \
    wandb.mode=online \
    wandb.entity=$WANDB_ENTITY \
    wandb.project=$WANDB_PROJECT

echo "Training completed!"