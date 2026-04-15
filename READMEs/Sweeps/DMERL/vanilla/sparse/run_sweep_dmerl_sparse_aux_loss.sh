#!/bin/bash

# Step 1: Fixed settings (matches the provided baseline command)
ENV_NAME=AcrobotSwingupSparse
NUM_EVAL=50
TOTAL_TIME_STEPS=50000000
DIFF_STEPS=8
EXPERIMENT_OVERRIDES=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp
ENT_TARGET_MULT=4
USE_FINAL_STEP_REWARD_TARGET=true

# Step 2: Define sweep values
SEEDS=(
    0
    1
)

AUX_LOSS_MULT_VALUES=(
    1.
    0.5
    2.
)

TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#AUX_LOSS_MULT_VALUES[@]} ))
echo "Planned runs: $TOTAL_RUNS (seeds=${#SEEDS[@]} aux_loss_mult=${#AUX_LOSS_MULT_VALUES[@]})"

# Step 3: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 4: Wait until a GPU is free (no active compute processes)
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

# Step 5: Launch one run per seed x aux_loss_mult combo
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
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_FR_sparse_test" \
            hyperparameters.num_eval="$NUM_EVAL" \
            hyperparameters.total_time_steps="$TOTAL_TIME_STEPS" \
            hyperparameters.diffusion.diff_steps="$DIFF_STEPS" \
            env=mjx_dmc \
            experiment_overrides="$EXPERIMENT_OVERRIDES" \
            seed="$SEED" \
            hyperparameters.ent_target_mult="$ENT_TARGET_MULT" \
            hyperparameters.use_final_step_reward_target="$USE_FINAL_STEP_REWARD_TARGET" \
            hyperparameters.aux_loss_mult="$AUX_LOSS_MULT" &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

# Step 6: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
