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
METHOD="reppo"   # Previously undefined, this is now set.

ENV_LIST=(
    HumanoidWalk 
    HumanoidRun 
    HumanoidStand
)
SEEDS=(1000 1001 1002 1003 1004)
CONFIGS=("reppo")                # Configuration files to use

MJX_DMC_DATALIST=(mjx_dmc_small_data)

# Loop through seeds and environments
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

python src/jaxrl/reppo.py -cn ${CONFIG} env=mjx_dmc env.name=${ENV_NAME} seed=${USING_SEED} experiment_overrides=${MJX_DMC_DATA} wandb.entity=${ENTITY}
EOF"
            done
        done
    done
done