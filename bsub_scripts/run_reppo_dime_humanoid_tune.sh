#!/bin/bash

# Job configuration
CLUSTER="batch_a100"      # This is a valid queue according to the document
TIME="6:00"              # 6-hour wall time, within the 168-hour limit for the queue
USER=$(whoami)
GPUS=1
CPUS=4                   # Number of CPU cores for your job


# Correctly define memory per CPU core in MB.
# Example: For a total of 32GB memory with 16 CPUs, this would be 2000 (32000 / 16).
MEMORY_PER_CPU_MB=3000

PWD=$(pwd)

ENTITY="huylb314"  # Set your Weights & Biases entity here

# Define the method for the experiment
METHOD="reppo_dime"   # Previously undefined, this is now set.

ENV_LIST=(G1JoystickFlatTerrain G1JoystickRoughTerrain T1JoystickFlatTerrain T1JoystickRoughTerrain)
# SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
SEEDS=(1000 1001 1002 1003 1004)
# SEEDS=(1000)
# SEEDS=(1005 1006 1007 1008 1009)
CONFIGS=("reppo_dime")       # Configuration files to use
# VMINMAX=("-80 80" "-50 50" "-20 20")
VMINMAX=("-10 10")
ENT_START=0.01
DIFF_STEPS=16

# MJX_DMC_DATALIST=(mjx_humanoid_small_data mjx_humanoid_large_data)
MJX_DMC_DATALIST=(mjx_humanoid_small_data)

# Loop through seeds and environments
for i in "${!VMINMAX[@]}"; do
    # Extract VMIN and VMAX from the space-separated string
    VMIN_VMAX="${VMINMAX[$i]}"
    VMIN=$(echo "$VMIN_VMAX" | cut -d' ' -f1)
    VMAX=$(echo "$VMIN_VMAX" | cut -d' ' -f2)
    
    for MJX_DMC_DATA in "${MJX_DMC_DATALIST[@]}"; do
        for ENV_NAME in "${ENV_LIST[@]}"; do
            for USING_SEED in "${SEEDS[@]}"; do
                for CONFIG in "${CONFIGS[@]}"; do
        eval "bsub <<EOF
#!/bin/bash -l
#BSUB -J ${METHOD}_${USING_SEED}
#BSUB -o ${PWD}/jobs/out/out.%J.stdout
#BSUB -e ${PWD}/jobs/error/err.%J.stderr
#BSUB -q ${CLUSTER}
#BSUB -W ${TIME}
#BSUB -M ${MEMORY_PER_CPU_MB}    # Corrected: Request memory per CPU core
#BSUB -n ${CPUS}                 # Corrected: Request a total of ${CPUS} job slots (cores)
#BSUB -gpu \"num=${GPUS}\"
#BSUB -R \"span[hosts=1]\"       # Ensure the job runs on a single host

# Load modules & source venv
source /home/${USER}/.bashrc
# source /fs/applications/p4s-access/2.0/ActivateP4S.sh -a  # This might not be needed depending on your environment
source ${PWD}/.venv/bin/activate

# Run the experiment
cd ${PWD}

python src/jaxrl/reppo_dime.py -cn ${CONFIG} env=mjx_humanoid_dime env.name=${ENV_NAME} seed=${USING_SEED} experiment_overrides=${MJX_DMC_DATA} wandb.entity=${ENTITY} env.vmin=${VMIN} env.vmax=${VMAX} hyperparameters.ent_start=${ENT_START} diffusion_type.diff_steps=${DIFF_STEPS}
EOF"
                done
            done
        done
    done
done