#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    AcrobotSwingup
    BallInCup
    AcrobotSwingupSparse
    PendulumSwingup
    # Add more env names here
)

# Step 2: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 3: Launch one run per env.name
for ENV_NAME in "${ENV_NAMES[@]}"; do
    GPU_ID=$((GPU_INDEX % NUM_GPUS))
    echo "Starting env.name=$ENV_NAME on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_dime \
        env.name="$ENV_NAME" \
        wandb.project_suffix="_FinalRuns" \
        hyperparameters.num_eval=100 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps=8 \
        hyperparameters.kl_action_rep=1 \
        hyperparameters.reverse_kl=false \
        hyperparameters.actor_kl_clip_mode=clipped \
        hyperparameters.ent_start=0.01 \
        hyperparameters.vmin=-100 \
        hyperparameters.vmax=200 \
        hyperparameters.num_bins=301 \
        hyperparameters.diffusion.learn_friction=true \
        hyperparameters.diffusion.learn_dt=true \
        hyperparameters.diffusion.per_step_dt=true \
        hyperparameters.lr=3e-4 \
        env=mjx_dmc \
        num_trials=3 \
        experiment_overrides=mjx_dmc_large_data &
    GPU_INDEX=$((GPU_INDEX + 1))
done

# Step 4: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
