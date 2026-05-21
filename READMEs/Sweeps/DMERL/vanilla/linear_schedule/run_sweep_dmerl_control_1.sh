#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    #WalkerRun #
    #WalkerStand #/ running
    WalkerWalk
    FingerSpin 
    # Add more env names here
)

# Step 2: Define seed sweep values
SEEDS=(
    0
    1
    2
    3
    5
    8
    13
    21
)

# Step 3: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 4: Shared helpers for GPU polling and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../lib/gpu_sweep_helpers.sh"

# Step 5: Launch one run per env.name x seed (one per GPU at a time)
declare -a GPU_PIDS
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for SEED in "${SEEDS[@]}"; do
        SEED="${SEED%,}"
        GPU_ID=$((GPU_INDEX % NUM_GPUS))
        wait_for_gpu "$GPU_ID"
        if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
            echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
            wait "${GPU_PIDS[$GPU_ID]}"
        fi
        echo "Starting env.name=$ENV_NAME seed=$SEED on GPU $GPU_ID..."
        build_run_paths "$GPU_ID" "$ENV_NAME" "$SEED"
        echo "Run directory: $RUN_DIR"
        WANDB_DIR="$WANDB_RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.jaxrl.DA_MDP_REPPO \
            env.name="$ENV_NAME" \
            wandb.project_suffix="_FR_REPPO_05_05" \
            hyperparameters.num_eval=50 \
            hyperparameters.total_time_steps=50000000 \
            hyperparameters.diffusion.diff_steps=8 \
            env=mjx_dmc \
            seed="$SEED" \
            num_trials=1 \
            hydra.run.dir="$RUN_DIR" \
            experiment_overrides=DA_MDP_REPPO/mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule &
        GPU_PIDS[$GPU_ID]=$!
        GPU_INDEX=$((GPU_INDEX + 1))
    done
done

# Step 6: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=dime/mjx_dmc_large_data
