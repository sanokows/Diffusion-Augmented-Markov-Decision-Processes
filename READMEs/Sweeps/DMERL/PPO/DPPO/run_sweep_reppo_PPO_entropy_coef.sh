#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    PendulumSwingup
    #HopperStand
    # Add more env names here
)

# Step 2: Define entropy coefficient values to sweep
ENTROPY_COEFS=(
    0
    1e-5
    1e-4
)

# Step 3: Configure vmapped seeds (parallel seeds in one run on one GPU)
NUM_SEEDS=1
BASE_SEED=0
TRIALS=5

# Step 4: Defaults aligned with run_sweep_dmerl_PPO_control_1.sh
WANDB_PROJECT_SUFFIX="_FR_PPO_temp_abl"
TOTAL_TIME_STEPS=50000000
NUM_EVAL=50
NUM_MINI_BATCHES=8
LR=2e-4
TEMPERATURE_LR=1e-4
UPDATE_ENTROPY_LAGRANGIAN=false
NUM_ENVS=1024
DIFFUSION_LEARN_FRICTION=true
DIFFUSION_LEARN_DT=true
DIFFUSION_PER_STEP_DT=true
USE_CATEGORICAL_VALUE=true
NORMALIZE_ADVANTAGES=false
ENT_TARGET_MULT=8.
NUM_COLLECTION_STEP_FACTOR=0.5
VMIN=-100
VMAX=200
NUM_BINS=301
NUM_EPOCHS=8
DIFFUSION_FRICTION=0.25
DIFFUSION_INIT_STD=3.

# Step 5: Define GPU pool and round-robin scheduling.
# Prefer the SLURM-provided visibility mask when present.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t GPU_DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
fi
if [ "${#GPU_DEVICES[@]}" -eq 0 ]; then
    echo "No GPUs detected from CUDA_VISIBLE_DEVICES or nvidia-smi."
    exit 1
fi
NUM_GPUS=${#GPU_DEVICES[@]}
GPU_INDEX=0
TOTAL_RUNS=$(( ${#ENV_NAMES[@]} * ${#ENTROPY_COEFS[@]} ))
echo "Planned runs: $TOTAL_RUNS (env=${#ENV_NAMES[@]} entropy_coefs=${#ENTROPY_COEFS[@]})"
echo "Visible GPU devices: ${GPU_DEVICES[*]} (count=$NUM_GPUS)"
echo "Each run uses vmapped seeds: num_seeds=$NUM_SEEDS base_seed=$BASE_SEED"

# Step 6: Wait until a GPU is free (no active compute processes)
wait_for_gpu() {
    local GPU_ID=$1
    while true; do
        # Treat GPU as busy only if at least one numeric PID is present.
        local APP_PID_OUTPUT
        local HAS_NUMERIC_PID=0
        APP_PID_OUTPUT="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
        while IFS= read -r LINE; do
            LINE="${LINE//[[:space:]]/}"
            if [[ "$LINE" =~ ^[0-9]+$ ]]; then
                HAS_NUMERIC_PID=1
                break
            fi
        done <<< "$APP_PID_OUTPUT"
        if [ "$HAS_NUMERIC_PID" -eq 0 ]; then
            break
        fi
        echo "GPU $GPU_ID busy, waiting..."
        sleep 30
    done
}

# Step 7: Launch one run per env.name x entropy_coef combo
declare -a GPU_PIDS
launch_run() {
    local ENV_NAME="$1"
    local ENTROPY_COEF="$2"

    local GPU_SLOT=$((GPU_INDEX % NUM_GPUS))
    local GPU_DEVICE="${GPU_DEVICES[$GPU_SLOT]}"
    wait_for_gpu "$GPU_DEVICE"
    if [ -n "${GPU_PIDS[$GPU_SLOT]}" ]; then
        echo "Waiting for previous run on GPU slot $GPU_SLOT (pid ${GPU_PIDS[$GPU_SLOT]})..."
        wait "${GPU_PIDS[$GPU_SLOT]}"
    fi

    echo "Starting env.name=$ENV_NAME entropy_coef=$ENTROPY_COEF num_seeds=$NUM_SEEDS on GPU slot $GPU_SLOT (device $GPU_DEVICE)..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.reppo_DiffPPO \
        env.name="$ENV_NAME" \
        wandb.project_suffix="$WANDB_PROJECT_SUFFIX" \
        hyperparameters.num_eval="$NUM_EVAL" \
        hyperparameters.total_time_steps="$TOTAL_TIME_STEPS" \
        hyperparameters.num_mini_batches="$NUM_MINI_BATCHES" \
        hyperparameters.lr="$LR" \
        hyperparameters.temperature_lr="$TEMPERATURE_LR" \
        hyperparameters.update_entropy_lagrangian="$UPDATE_ENTROPY_LAGRANGIAN" \
        hyperparameters.num_envs="$NUM_ENVS" \
        hyperparameters.diffusion.learn_friction="$DIFFUSION_LEARN_FRICTION" \
        hyperparameters.diffusion.learn_dt="$DIFFUSION_LEARN_DT" \
        hyperparameters.diffusion.per_step_dt="$DIFFUSION_PER_STEP_DT" \
        hyperparameters.use_categorical_value="$USE_CATEGORICAL_VALUE" \
        hyperparameters.normalize_advantages="$NORMALIZE_ADVANTAGES" \
        hyperparameters.ent_target_mult="$ENT_TARGET_MULT" \
        hyperparameters.num_collection_step_factor="$NUM_COLLECTION_STEP_FACTOR" \
        hyperparameters.vmin="$VMIN" \
        hyperparameters.vmax="$VMAX" \
        hyperparameters.num_bins="$NUM_BINS" \
        hyperparameters.num_epochs="$NUM_EPOCHS" \
        hyperparameters.diffusion.friction="$DIFFUSION_FRICTION" \
        hyperparameters.diffusion.init_std="$DIFFUSION_INIT_STD" \
        hyperparameters.entropy_coef="$ENTROPY_COEF" \
        env=mjx_dmc \
        seed="$BASE_SEED" \
        num_seeds="$NUM_SEEDS" \
        trials="$TRIALS" &
    GPU_PIDS[$GPU_SLOT]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
}

for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for ENTROPY_COEF in "${ENTROPY_COEFS[@]}"; do
        ENTROPY_COEF="${ENTROPY_COEF%,}"
        launch_run "$ENV_NAME" "$ENTROPY_COEF"
    done
done

# Step 8: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
