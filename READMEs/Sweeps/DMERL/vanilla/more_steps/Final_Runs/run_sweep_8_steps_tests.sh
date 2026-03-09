#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    G1JoystickFlatTerrain
    # Add more env names here
)

# Step 2: Fixed defaults and explicit tuple sweep.
DIFF_STEP=8

# Step 2a: Sweep tuples.
# Format: "v_value|lr|temperature_lr|lagrangian_lr|aux_loss_mult|gamma|lmbda|num_mini_batches|ent_target_mult|num_collection_step_factor"
SWEEP_TUPLES=(
    "10|1e-3|3e-4|3e-4|0.25|0.9952|0.96|2|3|0.5"
    "10|1e-3|3e-4|3e-4|0.25|0.9952|0.96|2|4|0.5" ### TOD vary num bins?
    "10|1e-3|3e-4|3e-4|0.25|0.9952|0.96|4|4|0.5"
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
TOTAL_RUNS=$(( ${#ENV_NAMES[@]} * ${#SWEEP_TUPLES[@]} ))
echo "Planned runs: $TOTAL_RUNS (env=${#ENV_NAMES[@]} sweep_tuples=${#SWEEP_TUPLES[@]})"
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
    local V_VALUE="$3"
    local LR="$4"
    local TEMPERATURE_LR="$5"
    local LAGRANGIAN_LR="$6"
    local AUX_LOSS_MULT="$7"
    local GAMMA="$8"
    local LMBDA="$9"
    local NUM_MINI_BATCHES="${10}"
    local ENT_TARGET_MULT="${11}"
    local NUM_COLLECTION_STEP_FACTOR="${12}"
    local VMIN="-$V_VALUE"
    local VMAX="$V_VALUE"
    local CONFIG_KEY="${ENV_NAME}|${DIFF_STEP}|${V_VALUE}|${LR}|${TEMPERATURE_LR}|${LAGRANGIAN_LR}|${AUX_LOSS_MULT}|${GAMMA}|${LMBDA}|${NUM_MINI_BATCHES}|${ENT_TARGET_MULT}|${NUM_COLLECTION_STEP_FACTOR}"

    if [ -n "${LAUNCHED_CONFIGS[$CONFIG_KEY]+x}" ]; then
        echo "Skipping duplicate config from axis=$SWEEP_AXIS: env.name=$ENV_NAME diff_steps=$DIFF_STEP v_value=$V_VALUE lr=$LR temperature_lr=$TEMPERATURE_LR lagrangian_lr=$LAGRANGIAN_LR aux_loss_mult=$AUX_LOSS_MULT gamma=$GAMMA lmbda=$LMBDA"
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
    echo "Starting axis=$SWEEP_AXIS env.name=$ENV_NAME diff_steps=$DIFF_STEP v_value=$V_VALUE lr=$LR temperature_lr=$TEMPERATURE_LR lagrangian_lr=$LAGRANGIAN_LR aux_loss_mult=$AUX_LOSS_MULT gamma=$GAMMA lmbda=$LMBDA vmin=$VMIN vmax=$VMAX on GPU slot $GPU_SLOT (device $GPU_DEVICE)..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.reppo_DMERL_new \
        env.name="$ENV_NAME" \
        wandb.project_suffix="_FR_old_branch" \
        hyperparameters.num_eval=50 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps="$DIFF_STEP" \
        hyperparameters.num_mini_batches="$NUM_MINI_BATCHES" \
        hyperparameters.lr="$LR" \
        hyperparameters.temperature_lr="$TEMPERATURE_LR" \
        hyperparameters.lagrangian_lr="$LAGRANGIAN_LR" \
        hyperparameters.gamma="$GAMMA" \
        hyperparameters.lmbda="$LMBDA" \
        hyperparameters.vmin="$VMIN" \
        hyperparameters.vmax="$VMAX" \
        hyperparameters.num_bins=301 \
        hyperparameters.aux_loss_mult="$AUX_LOSS_MULT" \
        hyperparameters.ent_target_mult="$ENT_TARGET_MULT" \
        hyperparameters.num_collection_step_factor="$NUM_COLLECTION_STEP_FACTOR" \
        hyperparameters.ent_start=0.001 \
        env=mjx_humanoid_dime \
        num_trials=1 \
        seed=0 \
        experiment_overrides=mjx_humanoid_large_data_DMERL &
    GPU_PIDS[$GPU_SLOT]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
}

# Step 6: Tuple sweep loop
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"

    for SWEEP_TUPLE in "${SWEEP_TUPLES[@]}"; do
        IFS='|' read -r V_VALUE LR TEMPERATURE_LR LAGRANGIAN_LR AUX_LOSS_MULT GAMMA LMBDA NUM_MINI_BATCHES ENT_TARGET_MULT NUM_COLLECTION_STEP_FACTOR <<< "$SWEEP_TUPLE"
        launch_run "$ENV_NAME" "sweep_tuple" \
            "$V_VALUE" "$LR" "$TEMPERATURE_LR" "$LAGRANGIAN_LR" "$AUX_LOSS_MULT" \
            "$GAMMA" "$LMBDA" "$NUM_MINI_BATCHES" "$ENT_TARGET_MULT" "$NUM_COLLECTION_STEP_FACTOR"
    done
done

# Step 7: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=mjx_dmc_large_data
