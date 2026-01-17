#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    CheetahRun
    FishSwim
    HopperHop
    HopperStand
    WalkerRun
    WalkerStand
    WalkerWalk
    FingerSpin
)


DEFAULT_LMDA=0.98
DEFAULT_LR=3e-4


LRs=(
    2e-3
    6e-4
)

# Step 2: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 3: Wait until a GPU is free (no active compute processes)
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

# Step 4: Launch one run per env.name (one per GPU at a time)
declare -a GPU_PIDS
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"

    for LR in "${LRs[@]}"; do
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME (lr=$LR) on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_WPO_hyper" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=50000000 \
            hyperparameters.diffusion.diff_steps=8 \
            hyperparameters.lr="$LR" \
            hyperparameters.train_mode=WPO \
            env=mjx_dmc \
            num_trials=1 \
            seed=0 \
            experiment_overrides=mjx_dmc_large_data_dmerl_WPO_control &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done

done

# Step 5: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
