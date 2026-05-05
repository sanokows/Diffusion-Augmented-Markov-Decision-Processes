#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    WalkerRun
    WalkerStand
    WalkerWalk
    FingerSpin
    # Add more env names here
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
    GPU_ID=$((GPU_INDEX % NUM_GPUS))
    wait_for_gpu "$GPU_ID"
    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
        echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
        wait "${GPU_PIDS[$GPU_ID]}"
    fi
    echo "Starting env.name=$ENV_NAME on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
        env.name="$ENV_NAME" \
        wandb.project_suffix="_FR_REPPO_05_05" \
        hyperparameters.num_eval=50 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps=8 \
        env=mjx_dmc \
        seed=0 \
        num_trials=8 \
        experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule &
    GPU_PIDS[$GPU_ID]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
done

# Step 5: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=dime/mjx_dmc_large_data