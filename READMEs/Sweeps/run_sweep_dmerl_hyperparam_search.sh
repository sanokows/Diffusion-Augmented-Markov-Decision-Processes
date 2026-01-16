#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    AcrobotSwingup
    BallInCup
    AcrobotSwingupSparse
    PendulumSwingup
    # Add more env names here
)

DEFAULT_GAMMA=0.999
DEFAULT_LMDA=0.98
DEFAULT_LR=3e-4

GAMMAS=(
    0.9992
    0.999
    0.9985
)

LMDAS=(
    0.99
    0.96
)

LRs=(
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
            wandb.project_suffix="_HYPERPARAMS" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=25000000 \
            hyperparameters.diffusion.diff_steps=8 \
            hyperparameters.lr="$LR" \
            hyperparameters.temperature_lr="$LR" \
            hyperparameters.lagrangian_lr="$LR" \
            hyperparameters.ent_target_mult=2.5 \
            hyperparameters.gamma="$DEFAULT_GAMMA" \
            hyperparameters.lmbda="$DEFAULT_LMDA" \
            hyperparameters.diffusion.learn_friction=true \
            hyperparameters.diffusion.learn_dt=true \
            hyperparameters.diffusion.per_step_dt=true \
            env=mjx_dmc \
            num_trials=3 \
            experiment_overrides=mjx_dmc_large_data_dmerl &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done

    for GAMMA in "${GAMMAS[@]}"; do
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME (gamma=$GAMMA) on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_HYPERPARAMS" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=25000000 \
            hyperparameters.diffusion.diff_steps=8 \
            hyperparameters.lr="$DEFAULT_LR" \
            hyperparameters.temperature_lr="$DEFAULT_LR" \
            hyperparameters.lagrangian_lr="$DEFAULT_LR" \
            hyperparameters.ent_target_mult=3 \
            hyperparameters.gamma="$GAMMA" \
            hyperparameters.lmbda="$DEFAULT_LMDA" \
            hyperparameters.diffusion.learn_friction=true \
            hyperparameters.diffusion.learn_dt=true \
            hyperparameters.diffusion.per_step_dt=true \
            env=mjx_dmc \
            num_trials=3 \
            experiment_overrides=mjx_dmc_large_data_dmerl &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done

    for LMDA in "${LMDAS[@]}"; do
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME (lmbda=$LMDA) on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_HYPERPARAMS" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=25000000 \
            hyperparameters.diffusion.diff_steps=8 \
            hyperparameters.lr="$DEFAULT_LR" \
            hyperparameters.temperature_lr="$DEFAULT_LR" \
            hyperparameters.lagrangian_lr="$DEFAULT_LR" \
            hyperparameters.ent_target_mult=3 \
            hyperparameters.gamma="$DEFAULT_GAMMA" \
            hyperparameters.lmbda="$LMDA" \
            hyperparameters.diffusion.learn_friction=true \
            hyperparameters.diffusion.learn_dt=true \
            hyperparameters.diffusion.per_step_dt=true \
            env=mjx_dmc \
            num_trials=3 \
            experiment_overrides=mjx_dmc_large_data_dmerl &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

# Step 5: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
