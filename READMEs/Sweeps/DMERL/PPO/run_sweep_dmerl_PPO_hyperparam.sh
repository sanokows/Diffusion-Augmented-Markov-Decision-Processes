#!/bin/bash

# Step 1: Define env.name values to loop over
ENV_NAMES=(
    PendulumSwingup
    #HopperStand
    # Add more env names here
)

# Step 2: Define hyperparameter values to sweep
LRS=(
    2e-3
    1e-3
    5e-4
)

ENTROPY_COEFS=(
    1e-4
    1e-5
    1e-6
)

NUM_ENVS_VALUES=(
    2024
    4048
)

NUM_EPOCHS_VALUES=(
    12
    #16
)

TOTAL_RUNS=$(( ${#ENV_NAMES[@]} * ${#LRS[@]} * ${#ENTROPY_COEFS[@]} * ${#NUM_ENVS_VALUES[@]} * ${#NUM_EPOCHS_VALUES[@]} ))
echo "Planned runs: $TOTAL_RUNS (env=${#ENV_NAMES[@]} lrs=${#LRS[@]} entropy_coefs=${#ENTROPY_COEFS[@]} num_envs=${#NUM_ENVS_VALUES[@]} num_epochs=${#NUM_EPOCHS_VALUES[@]})"

# Step 3: Define GPU pool and round-robin scheduling
NUM_GPUS=4
GPU_INDEX=0

# Step 4: Wait until a GPU is free (no active compute processes)
wait_for_gpu() {
    local GPU_ID=$1
    while true; do
        # nvidia-smi returns empty output when no compute processes are running
        if command -v rg >/dev/null 2>&1; then
            if ! nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader | rg -q '\S'; then
                break
            fi
        else
            if ! nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader | grep -q '[^[:space:]]'; then
                break
            fi
        fi
        echo "GPU $GPU_ID busy, waiting..."
        sleep 30
    done
}

# Step 5: Launch one run per env.name x lr x entropy_coef x num_envs x num_epochs combo
declare -a GPU_PIDS
for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAME%,}"
    for LR in "${LRS[@]}"; do
        LR="${LR%,}"
        for ENTROPY_COEF in "${ENTROPY_COEFS[@]}"; do
            ENTROPY_COEF="${ENTROPY_COEF%,}"
            for NUM_ENVS in "${NUM_ENVS_VALUES[@]}"; do
                NUM_ENVS="${NUM_ENVS%,}"
                for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
                    NUM_EPOCHS="${NUM_EPOCHS%,}"
                    GPU_ID=$((GPU_INDEX % NUM_GPUS))
                    wait_for_gpu "$GPU_ID"
                    if [ -n "${GPU_PIDS[$GPU_ID]}" ]; then
                        echo "Waiting for previous run on GPU $GPU_ID (pid ${GPU_PIDS[$GPU_ID]})..."
                        wait "${GPU_PIDS[$GPU_ID]}"
                    fi
                    echo "Starting env.name=$ENV_NAME lr=$LR entropy_coef=$ENTROPY_COEF num_envs=$NUM_ENVS num_epochs=$NUM_EPOCHS on GPU $GPU_ID..."
                    CUDA_VISIBLE_DEVICES=$GPU_ID python -m src.jaxrl.reppo_DiffPPO \
                        env.name="$ENV_NAME" \
                        DiffPPO_overrides=default \
                        env=mjx_dmc \
                        DiffPPO_overrides.hyperparameters.total_time_steps=25000000 \
                        wandb.project_suffix="_FR_13_04" \
                        DiffPPO_overrides.hyperparameters.lr="$LR" \
                        DiffPPO_overrides.hyperparameters.entropy_coef="$ENTROPY_COEF" \
                        DiffPPO_overrides.hyperparameters.num_envs="$NUM_ENVS" \
                        DiffPPO_overrides.hyperparameters.num_epochs="$NUM_EPOCHS" \
                        seed=0 \
                        trials=1 &
                    GPU_PIDS[$GPU_ID]=$!
                    GPU_INDEX=$((GPU_INDEX + 1))
                done
            done
        done
    done
done

# Step 6: Wait for all background runs to finish
echo "All runs started. Waiting for them to finish..."
wait
echo "All runs have finished."
# CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_dime env.name=CartpoleSwingup wandb.project_suffix=_FinalRuns hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-20 hyperparameters.vmax=170 hyperparameters.num_bins=191 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc num_trials=5 experiment_overrides=dime/mjx_dmc_large_data
