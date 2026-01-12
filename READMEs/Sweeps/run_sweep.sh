#!/bin/bash

# Step 1: Initialize the sweep and retrieve the sweep command
SWEEP_OUTPUT=$(wandb sweep ./READMEs/Sweeps/dime_reppo_env_sweeps.yaml 2>&1)
AGENT_COMMAND=$(echo "$SWEEP_OUTPUT" | grep -oP 'Run sweep agent with: \K.*')
echo "Command to start agents: $AGENT_COMMAND" 

# Check if AGENT_COMMAND is valid
if [ -z "$AGENT_COMMAND" ]; then
    echo "Error: AGENT_COMMAND could not be retrieved. Please check the output below:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

# Step 2: Define the number of agents per GPU
AGENTS_PER_GPU=1
NUM_GPUS=4

# Function to start wandb agents on a specific GPU in the background
start_agents() {
    GPU_ID=$1
    NUM_AGENTS=$2

    for (( i=0; i<NUM_AGENTS; i++ )); do
        echo "Starting agent $i on GPU $GPU_ID..."
        # Start the agent in the background
        CUDA_VISIBLE_DEVICES=$GPU_ID $AGENT_COMMAND &
    done
}

# Step 3: Launch agents on each GPU
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    start_agents $GPU_ID $AGENTS_PER_GPU
done

# Step 4: Wait for all background agents to finish
echo "All agents started. Waiting for them to finish..."
wait
echo "All WandB sweep agents have finished."