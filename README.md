# Diffusion-Augmented Markov Decision Processes for Maximum Entropy Reinforcement Learning

## Code for DA-MDP maximum-entropy reinforcement learning experiments

This repository contains the experimental code for the paper **Diffusion-Augmented Markov Decision Processes for Maximum Entropy Reinforcement Learning**.

The repository implements the DA-MDP variants used in the paper:

- `DA_MDP_REPPO`
- `DA_MDP_WPO`
- `DA_MDP_PPO`

This codebase is based on the paper code for **REPPO - Relative Entropy Pathwise Policy Optimization** [arXiv paper link](https://arxiv.org/abs/2507.11019). The DA-MDP entrypoints, configs, sweep scripts, and evaluation utilities in this repository extend that REPPO implementation for diffusion-augmented maximum-entropy RL experiments.

The repository provides:

- JAX implementations of the DA-MDP algorithms used in the paper.
- Hydra configs and sweep scripts for MJX DeepMind Control experiments.
- Checkpoint export and evaluation utilities for the DA-MDP method families.
- Installation through `uv` or a standard editable Python package install.

## Installation

We strongly recommend using the [uv tool](https://docs.astral.sh/uv/getting-started/installation/) for python dependency management.

With uv installed, you can install the project and all dependencies in a local virtual environment under `.venv` with one single command:
```bash 
uv sync
```

Our installation requires a GPU with CUDA 12 compatible drivers.

If you use other dependency management tools such as conda, create a new environment with `Python 3.12` and install our package with
```bash
pip install -e .
```

> [!Note]
> Several mujoco_playground environments, such as the Humanoid tasks, are currently unstable. If environments result in nans, we have simply rerun our experiments manually. As soon as these issues are solved upstream, we will update our dependencies.

> [!NOTE]
>  To provide a level comparison with prior work, we depend on the FastTD3 for of mujoco_playground. As soon as proper terminal state observation handling is merged into the main repository, we will update our dependencies.


## Run DA-MDP Experiments

The current DA-MDP experiments use the JAX entrypoints below. Configurations are handled with [Hydra](https://hydra.cc/), so any setting can be overridden from the command line with `key=value`.

| Method | Python module | Config root | Checkpoint method name | Main override |
| --- | --- | --- | --- | --- |
| `DA_MDP_REPPO` | `src.jaxrl.DA_MDP_REPPO` | `config/DA_MDP_REPPO.yaml` | `DA_MDP_REPPO` | `experiment_overrides=DA_MDP_REPPO/...` |
| `DA_MDP_WPO` | `src.jaxrl.DA_MDP_REPPO` | `config/DA_MDP_REPPO.yaml` | `DA_MDP_WPO` | `hyperparameters.train_mode=WPO` plus `experiment_overrides=DA_MDP_WPO/...` |
| `DA_MDP_PPO` | `src.jaxrl.DA_MDP_PPO` | `config/DA_MDP_PPO/` | `DA_MDP_PPO` | `overrides=default` or another `config/DA_MDP_PPO/overrides/` file |

Use `env.name` for the task, `env=mjx_dmc` for DeepMind Control MJX tasks, and `seed` for the random seed. `DA_MDP_REPPO` and `DA_MDP_WPO` use `num_trials`; `DA_MDP_PPO` uses `trials`. `hydra.run.dir` controls Hydra output location, and `WANDB_DIR` controls where W&B writes local run files.

The examples below are single-run commands. The sweep scripts append `&` to launch multiple background jobs; omit it when running one job interactively.

### DA_MDP_REPPO

```bash
export ENV_NAME=CheetahRun
export SEED=0
export GPU_ID=0
export RUN_DIR="outputs/DA_MDP_REPPO/${ENV_NAME}/seed_${SEED}"
export WANDB_RUN_DIR="${RUN_DIR}/wandb"
mkdir -p "$RUN_DIR" "$WANDB_RUN_DIR"

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
    experiment_overrides=DA_MDP_REPPO/mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule
```

### DA_MDP_WPO

`DA_MDP_WPO` uses the same Python module as `DA_MDP_REPPO`. The public method/checkpoint name becomes `DA_MDP_WPO` when `hyperparameters.train_mode=WPO` is set.

```bash
export ENV_NAME=CheetahRun
export SEED=0
export GPU_ID=0
export RUN_DIR="outputs/DA_MDP_WPO/${ENV_NAME}/seed_${SEED}"
export WANDB_RUN_DIR="${RUN_DIR}/wandb"
mkdir -p "$RUN_DIR" "$WANDB_RUN_DIR"

WANDB_DIR="$WANDB_RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.jaxrl.DA_MDP_REPPO \
    env.name="$ENV_NAME" \
    wandb.project_suffix="_FR_WPO_07_05_ent" \
    hyperparameters.num_eval=50 \
    hyperparameters.total_time_steps=50000000 \
    hyperparameters.diffusion.diff_steps=8 \
    hyperparameters.train_mode=WPO \
    env=mjx_dmc \
    seed="$SEED" \
    num_trials=1 \
    hydra.run.dir="$RUN_DIR" \
    experiment_overrides=DA_MDP_WPO/mjx_dmc_large_data_DA_MDP_WPO_linear_schedule
```

### DA_MDP_PPO

```bash
export ENV_NAME=CheetahRun
export SEED=0
export GPU_ID=0
export RUN_DIR="outputs/DA_MDP_PPO/${ENV_NAME}/seed_${SEED}"
export WANDB_RUN_DIR="${RUN_DIR}/wandb"
mkdir -p "$RUN_DIR" "$WANDB_RUN_DIR"

WANDB_DIR="$WANDB_RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.jaxrl.DA_MDP_PPO \
    env.name="$ENV_NAME" \
    env=mjx_dmc \
    overrides=default \
    wandb.project_suffix="_FR_PPO_27_04" \
    hyperparameters.num_eval=50 \
    hyperparameters.total_time_steps=50000000 \
    hyperparameters.diffusion.diff_steps=8 \
    seed="$SEED" \
    trials=1 \
    hydra.run.dir="$RUN_DIR"
```

### Sweep Scripts

The active sweep scripts are runnable references for launching many environments and seeds:

- `READMEs/Sweeps/DMERL/vanilla/linear_schedule/` for `DA_MDP_REPPO`
- `READMEs/Sweeps/DMERL/WPO/` for `DA_MDP_WPO`
- `READMEs/Sweeps/DMERL/PPO/` for `DA_MDP_PPO`

### Outputs and Checkpoints

Hydra writes run outputs under `hydra.run.dir`. W&B local files can be redirected with `WANDB_DIR`. When final checkpoint export is enabled, checkpoint filenames use the method prefix, for example `DA_MDP_REPPO__...`, `DA_MDP_WPO__...`, or `DA_MDP_PPO__...`.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## Citation

Please cite **Diffusion-Augmented Markov Decision Processes for Maximum Entropy Reinforcement Learning** when using this repository. 
```
