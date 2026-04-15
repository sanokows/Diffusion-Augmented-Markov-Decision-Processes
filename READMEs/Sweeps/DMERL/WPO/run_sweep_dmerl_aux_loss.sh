#!/bin/bash

# Step 1: Define explicit runs (env, aux_loss_mult, aux_loss_alpha)
# Format: "ENV_NAME|AUX_LOSS_MULT|AUX_LOSS_ALPHA"
RUNS=(
    "AcrobotSwingupSparse|0.02|0.98"
    "AcrobotSwingupSparse|0.04|0.98"
    "AcrobotSwingupSparse|0.06|0.98"
    "AcrobotSwingupSparse|0.01|0.98"
    # Add more runs here

)

# Step 4: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 5: Wait until a GPU is free (no active compute processes)
wait_for_gpu() {
    local GPU_ID=$1
    while true; do
        # nvidia-smi returns empty output when no compute processes are running
        if command -v rg >/dev/null 2>&1; then
            BUSY_CHECK_CMD="rg -q '\\S'"
        else
            BUSY_CHECK_CMD="grep -q '[^[:space:]]'"
        fi
        if ! nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader | eval "$BUSY_CHECK_CMD"; then
            break
        fi
        echo "GPU $GPU_ID busy, waiting..."
        sleep 30
    done
}

# Step 6: Launch runs (one per GPU at a time)
declare -a GPU_PIDS
for RUN in "${RUNS[@]}"; do
    IFS='|' read -r ENV_NAME AUX_LOSS_MULT AUX_LOSS_ALPHA <<< "$RUN"
    GPU_ID=$((GPU_INDEX % NUM_GPUS))
    wait_for_gpu "$GPU_ID"
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
        wait "${GPU_PIDS[$GPU_ID]}"
    fi
    echo "Starting env.name=$ENV_NAME aux_loss_mult=$AUX_LOSS_MULT aux_loss_alpha=$AUX_LOSS_ALPHA on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
        env.name="$ENV_NAME" \
        wandb.project_suffix="_aux_loss_WPO" \
        hyperparameters.num_eval=50 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps=8 \
        hyperparameters.aux_loss_mult="$AUX_LOSS_MULT" \
        hyperparameters.aux_loss_alpha="$AUX_LOSS_ALPHA" \
        env=mjx_dmc \
        num_trials=3 \
        experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_aux_loss &
    GPU_PIDS[$GPU_ID]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
done

# Step 7: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
