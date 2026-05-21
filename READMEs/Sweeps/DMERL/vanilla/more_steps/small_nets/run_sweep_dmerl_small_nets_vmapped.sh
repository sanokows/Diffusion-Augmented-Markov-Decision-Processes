#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    HopperHop
    HopperStand
)

# Step 2: Define hyperparameters.diffusion.diff_steps values to loop over
DIFF_STEPS=(
    2
    4
    8
    16
)

# Step 3: Configure vmapped seeds (single process, multiple seeds on one GPU)
NUM_SEEDS=1
BASE_SEED=0
NUM_TRIALS=8

# Step 4: Optional run-level defaults
WANDB_PROJECT_SUFFIX="_FR_more_steps_small_nets_vmapped"
TOTAL_TIME_STEPS=50000000
NUM_EVAL=50

# Step 5: Gamma base and per-diff-step lambda
# Gamma is computed automatically as: GAMMA_BASE^(1 / diff_steps)
GAMMA_BASE=0.99
declare -A LMBDA_BY_STEP=(
    ["2"]="0.970"
    ["4"]="0.972"
    ["8"]="0.98"
    ["16"]="0.983"
)
declare -A ENT_TARGET_MULT_BY_STEP=(
    ["2"]="3"
    ["4"]="3.5"
    ["8"]="4."
    ["16"]="4.5"
)
declare -A NUM_MINI_BATCHES_BY_STEP=(
    ["2"]="32"
    ["4"]="16"
    ["8"]="8"
    ["16"]="4"
)

# Step 6: Define GPU pool and round-robin scheduling.
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
TOTAL_RUNS=$(( ${#ENV_NAMES[@]} * ${#DIFF_STEPS[@]} ))
echo "Planned runs: $TOTAL_RUNS (env=${#ENV_NAMES[@]} diff_steps=${#DIFF_STEPS[@]})"
echo "Visible GPU devices: ${GPU_DEVICES[*]} (count=$NUM_GPUS)"
echo "Each run uses vmapped seeds: num_seeds=$NUM_SEEDS base_seed=$BASE_SEED"

# Step 7: Wait until a GPU is free (no active compute processes)
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

# Step 8: Launch a single run
declare -a GPU_PIDS
launch_run() {
    local ENV_NAME="$1"
    local DIFF_STEP="$2"
    local GAMMA="$3"
    local LMBDA="$4"
    local ENT_TARGET_MULT="$5"
    local NUM_MINI_BATCHES="$6"

    local GPU_SLOT=$((GPU_INDEX % NUM_GPUS))
    local GPU_DEVICE="${GPU_DEVICES[$GPU_SLOT]}"
    wait_for_gpu "$GPU_DEVICE"
    if [ -n "${GPU_PIDS[$GPU_SLOT]}" ]; then
        echo "Waiting for previous run on GPU slot $GPU_SLOT (pid ${GPU_PIDS[$GPU_SLOT]})..."
        wait "${GPU_PIDS[$GPU_SLOT]}"
    fi

    echo "Starting env.name=$ENV_NAME diff_steps=$DIFF_STEP gamma=$GAMMA lmbda=$LMBDA ent_target_mult=$ENT_TARGET_MULT num_mini_batches=$NUM_MINI_BATCHES num_seeds=$NUM_SEEDS on GPU slot $GPU_SLOT (device $GPU_DEVICE)..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.DA_MDP_REPPO \
        env.name="$ENV_NAME" \
        wandb.project_suffix="$WANDB_PROJECT_SUFFIX" \
        hyperparameters.num_eval="$NUM_EVAL" \
        hyperparameters.total_time_steps="$TOTAL_TIME_STEPS" \
        hyperparameters.num_mini_batches="$NUM_MINI_BATCHES" \
        hyperparameters.diffusion.diff_steps="$DIFF_STEP" \
        hyperparameters.gamma="$GAMMA" \
        hyperparameters.lmbda="$LMBDA" \
        hyperparameters.ent_target_mult="$ENT_TARGET_MULT" \
        env=mjx_dmc \
        num_trials="$NUM_TRIALS" \
        num_seeds="$NUM_SEEDS" \
        seed="$BASE_SEED" \
        experiment_overrides=DA_MDP_REPPO/mjx_dmc_large_data_DA_MDP_REPPO_small_nets &
    GPU_PIDS[$GPU_SLOT]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
}

# Step 9: Sweep loop
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for DIFF_STEP in "${DIFF_STEPS[@]}"; do
        DIFF_STEP="${DIFF_STEP%,}"
        GAMMA="$(python -c "import math; print(f'{math.pow(${GAMMA_BASE}, 1.0/${DIFF_STEP}):.10f}')")"
        LMBDA="${LMBDA_BY_STEP[$DIFF_STEP]}"
        ENT_TARGET_MULT="${ENT_TARGET_MULT_BY_STEP[$DIFF_STEP]}"
        NUM_MINI_BATCHES="${NUM_MINI_BATCHES_BY_STEP[$DIFF_STEP]}"
        if [ -z "$GAMMA" ] || [ -z "$LMBDA" ] || [ -z "$ENT_TARGET_MULT" ] || [ -z "$NUM_MINI_BATCHES" ]; then
            echo "Missing gamma/lmbda/ent_target_mult/num_mini_batches for diff_steps=$DIFF_STEP. Please fill *_BY_STEP maps and check GAMMA_BASE."
            exit 1
        fi
        launch_run "$ENV_NAME" "$DIFF_STEP" "$GAMMA" "$LMBDA" "$ENT_TARGET_MULT" "$NUM_MINI_BATCHES"
    done
done

# Step 10: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
