#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    HumanoidWalk
    # Add more env names here
)

# Step 2: Define hyperparameters.diffusion.diff_steps values to loop over
DIFF_STEPS=(
    6
    8
    10
    # Add more diff_steps values here
)

# Step 2a: Per-diff-step hyperparameters (string keys must match DIFF_STEPS)
declare -A NUM_MINI_BATCHES_BY_STEP=(
    ["6"]="4"
    ["10"]="4"
    ["14"]="4"
)

# Step 2a: Per-diff-step hyperparameters (string keys must match DIFF_STEPS)
declare -A NUM_MINI_BATCHES_BY_STEP=(
    ["6"]="4"
    ["8"]="4"
    ["10"]="4"
)

# Step 2b: Per-diff-step hyperparameters (string keys must match DIFF_STEPS)
declare -A FRICTION_BY_STEP=(
    ["6"]="0.3"
    ["8"]="0.4"
    ["10"]="0.5"
)
declare -A LR_BY_STEP=(
    ["6"]="1e-3"
    ["8"]="1.5e-3"
    ["10"]="2e-3"
)
declare -A TEMPERATURE_LR_BY_STEP=(
    ["6"]="5e-4"
    ["8"]="5e-4"
    ["10"]="5e-4"
)
declare -A LAGRANGIAN_LR_BY_STEP=(
    ["6"]="5e-4"
    ["8"]="5e-4"
    ["10"]="5e-4"
)
declare -A GAMMA_BY_STEP=(
    ["6"]="0.998"
    ["8"]="0.999"
    ["10"]="0.9992"
)
declare -A LMBDA_BY_STEP=(
    ["6"]="0.978"
    ["8"]="0.98"
    ["10"]="0.982"
)


# Step 3: Define GPU pool and round-robin scheduling
NUM_GPUS=3
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

# Step 5: Launch runs for each env.name x diff_steps combo (one per GPU at a time)
declare -a GPU_PIDS
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for DIFF_STEP in "${DIFF_STEPS[@]}"; do
        DIFF_STEP="${DIFF_STEP%,}"
        NUM_MINI_BATCHES="${NUM_MINI_BATCHES_BY_STEP[$DIFF_STEP]}"
        FRICTION="${FRICTION_BY_STEP[$DIFF_STEP]}"
        LR="${LR_BY_STEP[$DIFF_STEP]}"
        TEMPERATURE_LR="${TEMPERATURE_LR_BY_STEP[$DIFF_STEP]}"
        LAGRANGIAN_LR="${LAGRANGIAN_LR_BY_STEP[$DIFF_STEP]}"
        GAMMA="${GAMMA_BY_STEP[$DIFF_STEP]}"
        LMBDA="${LMBDA_BY_STEP[$DIFF_STEP]}"
        if [ -z "$NUM_MINI_BATCHES" ] || [ -z "$FRICTION" ] || [ -z "$LR" ] || [ -z "$TEMPERATURE_LR" ] || [ -z "$LAGRANGIAN_LR" ] || [ -z "$GAMMA" ] || [ -z "$LMBDA" ]; then
            echo "Missing hyperparameters for diff_steps=$DIFF_STEP. Please fill in *_BY_STEP maps."
            exit 1
        fi
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME diff_steps=$DIFF_STEP on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DMERL_new \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_FR_more_steps" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=80000000 \
            hyperparameters.diffusion.diff_steps="$DIFF_STEP" \
            hyperparameters.num_mini_batches="$NUM_MINI_BATCHES" \
            hyperparameters.diffusion.friction="$FRICTION" \
            hyperparameters.lr="$LR" \
            hyperparameters.temperature_lr="$TEMPERATURE_LR" \
            hyperparameters.lagrangian_lr="$LAGRANGIAN_LR" \
            hyperparameters.gamma="$GAMMA" \
            hyperparameters.diffusion.learn_friction=false \
            hyperparameters.diffusion.learn_dt=false \
            env=mjx_dmc \
            num_trials=2 \
            seed=0 \
            experiment_overrides=mjx_humanoid_large_data_dmerl_linear_schedule &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

# Step 6: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=mjx_dmc_large_data
