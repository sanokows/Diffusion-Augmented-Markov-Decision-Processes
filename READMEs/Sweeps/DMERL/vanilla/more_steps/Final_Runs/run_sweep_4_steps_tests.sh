#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    G1JoystickFlatTerrain
    # Add more env names here
)

# Step 2: Fixed defaults and explicit tuple sweep.
DIFF_STEP=4

# Step 2a: Sweep tuples.
# Format: "vmin|vmax|lr|temperature_lr|lagrangian_lr|aux_loss_mult|gamma|lmbda|num_mini_batches|ent_target_mult|num_collection_step_factor|num_bins|friction|seed"
SWEEP_TUPLES=(
    "-20|12|2e-3|3e-4|3e-4|0.25|0.9924|0.96|3|2|0.5|151|0.25|0"
    "-25|10|1e-3|3e-4|3e-4|0.25|0.9924|0.96|3|2|0.5|151|0.25|0"
    "-20|12|1e-3|3e-4|3e-4|0.25|0.9924|0.96|3|2|0.5|151|0.25|0"
    "-25|20|1e-3|3e-4|3e-4|0.25|0.9924|0.96|3|2|0.5|151|0.25|0"
    # "-10|10|1e-3|3e-4|3e-4|0.15|0.9908|0.96|4|4|0.5|151|0.25|1"
    # "-10|10|1e-3|3e-4|3e-4|0.15|0.9908|0.96|4|4|0.5|151|0.25|2"
    # "-10|10|1e-3|3e-4|3e-4|0.15|0.9908|0.96|4|4|0.5|151|0.25|3"
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
launch_run() {
    local ENV_NAME="$1"
    local SWEEP_AXIS="$2"
    local VMIN="$3"
    local VMAX="$4"
    local LR="$5"
    local TEMPERATURE_LR="$6"
    local LAGRANGIAN_LR="$7"
    local AUX_LOSS_MULT="$8"
    local GAMMA="$9"
    local LMBDA="${10}"
    local NUM_MINI_BATCHES="${11}"
    local ENT_TARGET_MULT="${12}"
    local NUM_COLLECTION_STEP_FACTOR="${13}"
    local NUM_BINS="${14}"
    local FRICTION="${15}"
    local SEED="${16:-0}"

    local GPU_SLOT=$((GPU_INDEX % NUM_GPUS))
    local GPU_DEVICE="${GPU_DEVICES[$GPU_SLOT]}"
    wait_for_gpu "$GPU_DEVICE"
    if [ -n "${GPU_PIDS[$GPU_SLOT]}" ]; then
        echo "Waiting for previous run on GPU slot $GPU_SLOT (pid ${GPU_PIDS[$GPU_SLOT]})..."
        wait "${GPU_PIDS[$GPU_SLOT]}"
    fi
    echo "Starting axis=$SWEEP_AXIS env.name=$ENV_NAME diff_steps=$DIFF_STEP vmin=$VMIN vmax=$VMAX lr=$LR temperature_lr=$TEMPERATURE_LR lagrangian_lr=$LAGRANGIAN_LR aux_loss_mult=$AUX_LOSS_MULT gamma=$GAMMA lmbda=$LMBDA num_bins=$NUM_BINS friction=$FRICTION seed=$SEED on GPU slot $GPU_SLOT (device $GPU_DEVICE)..."
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m src.jaxrl.reppo_DMERL_new \
        env.name="$ENV_NAME" \
        wandb.project_suffix="_FR_more_steps_final" \
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
        hyperparameters.num_bins="$NUM_BINS" \
        hyperparameters.aux_loss_mult="$AUX_LOSS_MULT" \
        hyperparameters.ent_target_mult="$ENT_TARGET_MULT" \
        hyperparameters.num_collection_step_factor="$NUM_COLLECTION_STEP_FACTOR" \
        hyperparameters.ent_start=0.01 \
        hyperparameters.diffusion.friction="$FRICTION" \
        hyperparameters.normalize_reward=false \
        env=mjx_humanoid_dime \
        num_trials=1 \
        seed="$SEED" \
        experiment_overrides=mjx_humanoid_large_data_DMERL &
    GPU_PIDS[$GPU_SLOT]=$!
    GPU_INDEX=$((GPU_INDEX + 1))
}

# Step 6: Tuple sweep loop
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"

    for SWEEP_TUPLE in "${SWEEP_TUPLES[@]}"; do
        IFS='|' read -r VMIN VMAX LR TEMPERATURE_LR LAGRANGIAN_LR AUX_LOSS_MULT GAMMA LMBDA NUM_MINI_BATCHES ENT_TARGET_MULT NUM_COLLECTION_STEP_FACTOR NUM_BINS FRICTION SEED <<< "$SWEEP_TUPLE"
        launch_run "$ENV_NAME" "sweep_tuple" \
            "$VMIN" "$VMAX" "$LR" "$TEMPERATURE_LR" "$LAGRANGIAN_LR" "$AUX_LOSS_MULT" \
            "$GAMMA" "$LMBDA" "$NUM_MINI_BATCHES" "$ENT_TARGET_MULT" "$NUM_COLLECTION_STEP_FACTOR" "$NUM_BINS" "$FRICTION" "$SEED"
    done
done

# Step 7: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=mjx_dmc_large_data
