# Turning GMM Visualization Script

This document explains how to run:

- `src/env_utils/visualization/test_turning_gmm_env.py`

The script does two things:

1. Contains unit tests for `TurningGMMEnv`.
2. Generates visualization artifacts for the GMM reward landscapes and rollouts.

## Run As a Visualization Script

From the repository root:

```bash
python src/env_utils/visualization/test_turning_gmm_env.py
```

Artifacts are written to:

- `artifacts/turning_gmm/`

By default this includes:

- `reward_landscape_all_action_parts_A_space.png`
- `reward_landscape_all_action_parts_A_space_detail.png`
- `density_all_action_parts_A_space.png`
- `parallel_rollout_component_mean_summary.png`
- `parallel_rollout_component_mean.gif` (if `imageio` is installed)

## Useful CLI Options

```bash
python src/env_utils/visualization/test_turning_gmm_env.py \
  --output-dir artifacts/turning_gmm \
  --num-action-state 8 \
  --num-gmm-components 5 \
  --gmm-mean-a-margin-d 5.0 \
  --gmm-std 0.05 \
  --gmm-seed 0 \
  --point-symmetric-mode \
  --horizon 120 \
  --batch-size 32 \
  --seed 0
```

Notes:

- Default `--gmm-seed` is `0` (deterministic means).
- The script prints all resolved env hyperparameters before creating the env.
- Rollout policy is component-mean sampling:
  - For each agent state, determine the current action part from heading.
  - Sample one Gaussian component uniformly from that part's GMM.
  - Use the sampled component mean directly as the action `a`.
- `--point-symmetric-mode` enables a specific symmetry setting:
  - Per action part, means are symmetric around `a=0`.
  - If `num_gmm_components` is odd, one mean is fixed at `0`.
  - Remaining means are sampled in `[d*sigma, 1-d*sigma]` and mirrored.
  - Opposite action parts (180 degrees apart) share the same GMM means.
  - This mode requires an even `num_action_state`.

## Run As Tests

To run only this file with `pytest`:

```bash
pytest src/env_utils/visualization/test_turning_gmm_env.py -q
```

To run one test:

```bash
pytest src/env_utils/visualization/test_turning_gmm_env.py::test_step_reward_uses_current_heading_action_part -q
```
