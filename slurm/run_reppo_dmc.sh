#!/bin/bash
#SBATCH --job-name=reppo_dmc
# #SBATCH --account=hk-project-p0022253
#SBATCH --account=hk-project-p0024023
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --exclude=hkn0635,hkn0417,hkn0612,hkn0735,hkn0634,hkn0632
#SBATCH --output=logs/reppo_dmc_%j.out
#SBATCH --error=logs/reppo_dmc_%j.err

# =============================================================================
# Environment Setup
# =============================================================================

# Load required modules
module load devel/cuda/12.4

# Activate virtual environment (passed from submission script)
source ${VIRTUAL_ENV:-.venv/bin/activate}

# =============================================================================
# Job Information
# =============================================================================

echo "================================================================================"
echo "REPPO DMC Production Run"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# =============================================================================
# Configuration Variables (with defaults)
# =============================================================================

# Main script and environment
MAIN_FILE=${MAIN_FILE:-src/jaxrl/reppo.py}
ENV=${ENV:-mjx_dmc}
ENV_NAME=${ENV_NAME:-CartpoleBalance}
SEED=${SEED:-0}

# Config files
CONFIG=${CONFIG:-reppo}
OVERRIDES=${OVERRIDES:-mjx_dmc_small_data}

# Hyperparameters
ENT_TARGET_MULT=${ENT_TARGET_MULT:-0.5}
ENT_START=${ENT_START:-0.01}
KL_BOUND=${KL_BOUND:-0.1}

# Value ranges
VMIN=${VMIN:-0}
VMAX=${VMAX:-150}

# W&B configuration
WANDB_ENTITY=${WANDB_ENTITY:-huylb314}
WANDB_PROJECT=${WANDB_PROJECT:-REPPO_DMC}

# =============================================================================
# JAX and Mujoco Environment Variables
# =============================================================================

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME=gpu
export MUJOCO_GL=egl

# =============================================================================
# Debug Output
# =============================================================================

echo "Configuration:"
echo "  Main File: $MAIN_FILE"
echo "  Environment: $ENV"
echo "  Task: $ENV_NAME"
echo "  Seed: $SEED"
echo "  Config: $CONFIG"
echo "  Overrides: $OVERRIDES"
echo ""
echo "Hyperparameters:"
echo "  Entropy Start: $ENT_START"
echo "  Entropy Target Mult: $ENT_TARGET_MULT"
echo "  KL Bound: $KL_BOUND"
echo ""
echo "Value Range:"
echo "  V Min: $VMIN"
echo "  V Max: $VMAX"
echo ""
echo "W&B:"
echo "  Entity: $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "================================================================================"

# =============================================================================
# Run Experiment
# =============================================================================

python $MAIN_FILE \
    -cn $CONFIG \
    env=$ENV \
    env.name=$ENV_NAME \
    env.vmin=$VMIN \
    env.vmax=$VMAX \
    experiment_overrides=$OVERRIDES \
    hyperparameters.ent_start=$ENT_START \
    hyperparameters.kl_bound=$KL_BOUND \
    hyperparameters.ent_target_mult=$ENT_TARGET_MULT \
    seed=$SEED \
    wandb.mode=online \
    wandb.entity=$WANDB_ENTITY \
    wandb.project=$WANDB_PROJECT

echo "================================================================================"
echo "Training completed!"
echo "================================================================================"
