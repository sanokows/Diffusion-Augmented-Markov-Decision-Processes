#!/bin/bash

ENV_NAMES=(
    FishSwim
    HopperHop
)

SEEDS=(
    0
)

NUM_GPUS=2
GPU_INDEX=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../lib/gpu_sweep_helpers.sh"

declare -a GPU_PIDS
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for SEED in "${SEEDS[@]}"; do
        SEED="${SEED%,}"
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME seed=$SEED on GPU $GPU_ID..."
        build_run_paths "$GPU_ID" "$ENV_NAME" "$SEED"
        echo "Run directory: $RUN_DIR"
        WANDB_DIR="$WANDB_RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_FR_REPPO_ENV_TIME" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=50000000 \
            hyperparameters.diffusion.diff_steps=8 \
            env=mjx_dmc \
            seed="$SEED" \
            num_trials=1 \
            hydra.run.dir="$RUN_DIR" \
            experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_env_time &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
