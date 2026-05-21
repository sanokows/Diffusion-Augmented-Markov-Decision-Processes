#!/bin/bash

# Step 1: Fixed settings
ENV_NAME=AcrobotSwingupSparse
AUX_LOSS_MULT_VALUES=(
    0.025
    0.05
)

# Step 2: Define seed sweep values
SEEDS=(
    0
    1
    2
    3
    5
    8
    13
    21
)

# Step 3: Define GPU pool and round-robin scheduling
NUM_GPUS=2
GPU_INDEX=0

# Step 4: Shared helpers for GPU polling and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../lib/gpu_sweep_helpers.sh"

# Step 5: Launch one run per aux_loss_mult x seed
declare -a GPU_PIDS
for AUX_LOSS_MULT in "${AUX_LOSS_MULT_VALUES[@]}"; do
    AUX_LOSS_MULT="${AUX_LOSS_MULT%,}"
    for SEED in "${SEEDS[@]}"; do
        SEED="${SEED%,}"
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME seed=$SEED aux_loss_mult=$AUX_LOSS_MULT on GPU $GPU_ID..."
        build_run_paths "$GPU_ID" "$ENV_NAME" "$SEED"
        echo "Run directory: $RUN_DIR"
        WANDB_DIR="$WANDB_RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.jaxrl.DA_MDP_REPPO \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_FR_REPPO_20_04" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=50000000 \
            hyperparameters.diffusion.diff_steps=8 \
            hyperparameters.aux_loss_mult="$AUX_LOSS_MULT" \
            env=mjx_dmc \
            seed="$SEED" \
            num_trials=1 \
            hydra.run.dir="$RUN_DIR" \
            experiment_overrides=DA_MDP_REPPO/mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

# Step 6: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
