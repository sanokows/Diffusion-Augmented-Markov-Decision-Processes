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

# Step 4: Sweep-specific defaults. Core PPO settings come from overrides=default.
WANDB_PROJECT_SUFFIX="_FR_PPO_temp_abl"

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
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.DA_MDP_PPO \
        env.name="$ENV_NAME" \
        overrides=default \
        env=mjx_dmc \
        wandb.project_suffix="$WANDB_PROJECT_SUFFIX" \
        hyperparameters.entropy_coef="$ENTROPY_COEF" \
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
