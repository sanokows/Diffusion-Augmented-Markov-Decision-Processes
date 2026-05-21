#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    G1JoystickFlatTerrain
    # Add more env names here
)

# Step 2: Fixed defaults and explicit tuple sweep.
DIFF_STEP=16
NUM_MINI_BATCHES=2
GAMMA=0.99
LMBDA=0.95
NUM_BINS=151
WANDB_PROJECT_SUFFIX="_FR_16step_no_reward_norm_vbounds_lr"

REWARD_NORMALIZATION_MODES=(
    none
)

REWARD_NORMALIZATION_WINDOW_ROLLOUTS=(
    0
)

# Step 2a: Sweep value bounds and learning-rate tuples.
# Format: "vmin|vmax"
VMIN_VMAX_TUPLES=(
    "-12|5"
    "-10|10"
    "-5|5"
)

# Format: "lr|temperature_lr|lagrangian_lr"
LR_TUPLES=(
    "1e-3|6e-4|6e-4"
    "1e-3|1e-3|1e-3"
    "1e-3|3e-4|3e-4"
)

ADAPTIVE_HL_GAUSS_BOUNDS_VALUES=(
    False
)

# Step 3: Define GPU pool and round-robin scheduling.
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
TOTAL_RUNS=$(( ${#ENV_NAMES[@]} * ${#REWARD_NORMALIZATION_MODES[@]} * ${#REWARD_NORMALIZATION_WINDOW_ROLLOUTS[@]} * ${#VMIN_VMAX_TUPLES[@]} * ${#LR_TUPLES[@]} * ${#ADAPTIVE_HL_GAUSS_BOUNDS_VALUES[@]} ))
echo "Planned runs: $TOTAL_RUNS (env=${#ENV_NAMES[@]} reward_modes=${#REWARD_NORMALIZATION_MODES[@]} window_rollouts=${#REWARD_NORMALIZATION_WINDOW_ROLLOUTS[@]} vmin_vmax_tuples=${#VMIN_VMAX_TUPLES[@]} lr_tuples=${#LR_TUPLES[@]} adaptive_hl_gauss_bounds=${#ADAPTIVE_HL_GAUSS_BOUNDS_VALUES[@]})"
echo "Fixed hyperparameters: gamma=$GAMMA lmbda=$LMBDA num_bins=$NUM_BINS"
echo "Visible GPU devices: ${GPU_DEVICES[*]} (count=$NUM_GPUS)"

# Step 4: Wait until a GPU is free (no active compute processes)
wait_for_gpu() {
    local GPU_ID=$1
    while true; do
        # Some nvidia-smi versions print non-PID text (e.g., "No running processes found").
        # Treat the GPU as busy only if at least one numeric PID is present.
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

# Step 5: Launch a single run with explicit hyperparameters (one per GPU slot at a time)
declare -a GPU_PIDS
declare -A LAUNCHED_CONFIGS
launch_run() {
    local ENV_NAME="$1"
    local SWEEP_AXIS="$2"
    local VMIN="$3"
    local VMAX="$4"
    local LR="$5"
    local TEMPERATURE_LR="$6"
    local LAGRANGIAN_LR="$7"
    local ADAPTIVE_HL_GAUSS_BOUNDS="$8"
    local REWARD_NORMALIZATION_MODE="$9"
    local REWARD_NORMALIZATION_WINDOW_ROLLOUT="${10}"
    local CONFIG_KEY="${ENV_NAME}|${DIFF_STEP}|${GAMMA}|${LMBDA}|${REWARD_NORMALIZATION_MODE}|${REWARD_NORMALIZATION_WINDOW_ROLLOUT}|${VMIN}|${VMAX}|${LR}|${TEMPERATURE_LR}|${LAGRANGIAN_LR}|${NUM_BINS}|${ADAPTIVE_HL_GAUSS_BOUNDS}"

    if [ -n "${LAUNCHED_CONFIGS[$CONFIG_KEY]+x}" ]; then
        echo "Skipping duplicate config from axis=$SWEEP_AXIS: env.name=$ENV_NAME diff_steps=$DIFF_STEP num_mini_batches=$NUM_MINI_BATCHES gamma=$GAMMA lmbda=$LMBDA reward_normalization_mode=$REWARD_NORMALIZATION_MODE reward_normalization_window_rollouts=$REWARD_NORMALIZATION_WINDOW_ROLLOUT vmin=$VMIN vmax=$VMAX lr=$LR temperature_lr=$TEMPERATURE_LR lagrangian_lr=$LAGRANGIAN_LR num_bins=$NUM_BINS adaptive_hl_gauss_bounds=$ADAPTIVE_HL_GAUSS_BOUNDS"
        return
    fi
    LAUNCHED_CONFIGS["$CONFIG_KEY"]=1

    local GPU_SLOT=$((GPU_INDEX % NUM_GPUS))
    local GPU_DEVICE="${GPU_DEVICES[$GPU_SLOT]}"
    wait_for_gpu "$GPU_DEVICE"
    if [ -n "${GPU_PIDS[$GPU_SLOT]}" ]; then
        echo "Waiting for previous run on GPU slot $GPU_SLOT (pid ${GPU_PIDS[$GPU_SLOT]})..."
        wait "${GPU_PIDS[$GPU_SLOT]}"
    fi
    echo "Starting axis=$SWEEP_AXIS env.name=$ENV_NAME diff_steps=$DIFF_STEP num_mini_batches=$NUM_MINI_BATCHES gamma=$GAMMA lmbda=$LMBDA reward_normalization_mode=$REWARD_NORMALIZATION_MODE reward_normalization_window_rollouts=$REWARD_NORMALIZATION_WINDOW_ROLLOUT vmin=$VMIN vmax=$VMAX lr=$LR temperature_lr=$TEMPERATURE_LR lagrangian_lr=$LAGRANGIAN_LR num_bins=$NUM_BINS adaptive_hl_gauss_bounds=$ADAPTIVE_HL_GAUSS_BOUNDS on GPU slot $GPU_SLOT (device $GPU_DEVICE)..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.reppo_DMERL_new \
        env.name="$ENV_NAME" \
        wandb.project_suffix="$WANDB_PROJECT_SUFFIX" \
        hyperparameters.diffusion.diff_steps="$DIFF_STEP" \
        hyperparameters.num_mini_batches="$NUM_MINI_BATCHES" \
        hyperparameters.gamma="$GAMMA" \
        hyperparameters.lmbda="$LMBDA" \
        hyperparameters.reward_normalization_mode="$REWARD_NORMALIZATION_MODE" \
        hyperparameters.reward_normalization_window_rollouts="$REWARD_NORMALIZATION_WINDOW_ROLLOUT" \
        hyperparameters.lr="$LR" \
        hyperparameters.temperature_lr="$TEMPERATURE_LR" \
        hyperparameters.lagrangian_lr="$LAGRANGIAN_LR" \
        hyperparameters.vmin="$VMIN" \
        hyperparameters.vmax="$VMAX" \
        hyperparameters.num_bins="$NUM_BINS" \
        hyperparameters.adaptive_hl_gauss_bounds="$ADAPTIVE_HL_GAUSS_BOUNDS" \
        env=mjx_humanoid_dime \
        num_trials=1 \
        experiment_overrides=dmerl/mjx_humanoid_large_data_DMERL &
    GPU_PIDS[$GPU_SLOT]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
}

# Step 6: Tuple sweep loop
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"

    for VMIN_VMAX_TUPLE in "${VMIN_VMAX_TUPLES[@]}"; do
        IFS='|' read -r VMIN VMAX <<< "$VMIN_VMAX_TUPLE"
        for REWARD_NORMALIZATION_MODE in "${REWARD_NORMALIZATION_MODES[@]}"; do
            for REWARD_NORMALIZATION_WINDOW_ROLLOUT in "${REWARD_NORMALIZATION_WINDOW_ROLLOUTS[@]}"; do
                for LR_TUPLE in "${LR_TUPLES[@]}"; do
                    IFS='|' read -r LR TEMPERATURE_LR LAGRANGIAN_LR <<< "$LR_TUPLE"
                    for ADAPTIVE_HL_GAUSS_BOUNDS in "${ADAPTIVE_HL_GAUSS_BOUNDS_VALUES[@]}"; do
                        launch_run "$ENV_NAME" "value_bounds_lr_tuple" \
                            "$VMIN" "$VMAX" "$LR" "$TEMPERATURE_LR" "$LAGRANGIAN_LR" "$ADAPTIVE_HL_GAUSS_BOUNDS" "$REWARD_NORMALIZATION_MODE" "$REWARD_NORMALIZATION_WINDOW_ROLLOUT"
                    done
                done
            done
        done
    done
done

# Step 7: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
