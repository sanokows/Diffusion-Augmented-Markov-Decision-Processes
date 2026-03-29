# Turning Envs Overview

This file documents the two turning-style custom envs:
- `TurningDoubleWellEnv` (`src/env_utils/turning_double_well_env.py`)
- `TurningMultiWellEnv` (`src/env_utils/turning_multi_well_env.py`)

Both envs:
- use a 1-D normalized action in `[-1, 1]` interpreted as a relative turn command
- advance the agent in 2-D by `step_size` along the new heading
- return reward based on the relative turn delta (not absolute heading)

## TurningDoubleWellEnv

Purpose:
- baseline two-mode turning task with two preferred turns at `-well_angle_deg` and `+well_angle_deg`

Reward shape:
- double-well style profile over relative turn angle
- `well_height` controls the center/barrier reward at turn `0`
- rewards decay toward edges near `+-90` deg

Main configuration:
- `well_angle_deg`: location of the two maxima
- `well_height`: center/barrier reward level in `[0, 1]`
- `max_turn_deg`: action-to-turn scaling
- `snap_action_to_optimal`: snap transition turn to `+-well_angle_deg`
- `randomize_initial_heading`: sampled reset heading

Hydra env config:
- `config/env/mjx_double_well.yaml`

## TurningMultiWellEnv

Purpose:
- harder n-minima variant with configurable number of preferred turns

Minima construction:
- for `n = num_minima`, minima are:
  - `theta_i = 90 - 90/n - 2*90/n * i`, `i = 0, ..., n-1`
- examples:
  - `n=2`: `[-45, 45]`
  - `n=4`: `[-67.5, -22.5, 22.5, 67.5]`

Reward shape:
- cosine-modulated interior wells across minima
- `well_height` controls barrier level between neighboring wells
- polynomial tails toward `+-90` controlled by `multi_minima_tail_power`

Main configuration:
- `num_minima`: number of preferred turn locations (`>=2`)
- `well_height`: barrier reward level in `[0, 1]`
- `multi_minima_tail_power`: tail exponent (`>0`)
- `include_opposite_headings`: include `+180` heading copies in snapped reset support
- `snap_action_to_optimal`: snap transition turn to nearest configured minimum

Hydra env config:
- `config/env/mjx_multi_well.yaml`

## Choosing between them

Use `TurningDoubleWellEnv` when:
- you want the original task used in previous experiments
- you need strict backward compatibility with older runs

Use `TurningMultiWellEnv` when:
- you want to scale task difficulty via `num_minima`
- you want more multimodal turning preferences than the double-well baseline

## Scripts and CLI Usage

Run from repository root:
`/home/it4i-sanokows/code/DIMEReppo`

### Plot script

Command:
`python src/env_utils/plot_turning_double_well_reward_configs.py`

Example:
```bash
python src/env_utils/plot_turning_double_well_reward_configs.py \
  --output-dir artifacts/_tmp_turning_plots \
  --num-points 1001 \
  --double-well-angle-deg 45 \
  --multi-num-minima 4
```

Notes:
- It writes one figure for `TurningDoubleWellEnv` and one for `TurningMultiWellEnv`.
- X-axis is relative turn angle with respect to the current agent orientation.
- Each subplot corresponds to one unique start heading.

Arguments:
- `--output-dir`: output directory for figures.
- `--num-points`: sampling resolution over action/turn axis.
- `--global-max-atol`: tolerance for selecting global maxima from reward landscape.
- `--subplot-cols`: subplot grid columns; auto if omitted.
- `--max-turn-deg`: action-to-angle scale (`action * max_turn_deg`).
- `--double-well-angle-deg`: double-well peak location in degrees.
- `--double-well-height`: double-well barrier/center reward level.
- `--multi-num-minima`: number of multi-well maxima.
- `--multi-well-height`: multi-well barrier level between adjacent maxima.
- `--multi-tail-power`: multi-well tail exponent near `+-90` deg.
- `--multi-include-opposite-headings` / `--no-multi-include-opposite-headings`: include or exclude opposite reset headings for multi-well.

### Simulation script

Command:
`python src/env_utils/test_turning_double_well_env.py`

Example:
```bash
python src/env_utils/test_turning_double_well_env.py \
  --env both \
  --output-dir artifacts/_tmp_turning_tests \
  --horizon 80 \
  --num-trajectories 16 \
  --save-gif
```

Notes:
- It computes global reward maxima in each chosen env.
- Agents randomly sample one of those maxima actions at each step.
- It writes landscape, rollout summary, and optional GIF artifacts.

Arguments:
- `--env`: run `double`, `multi`, or `both`.
- `--output-dir`: root directory for artifacts.
- `--horizon`: rollout length in steps.
- `--num-trajectories`: number of parallel agents.
- `--seed`: base random seed.
- `--max-turn-deg`: max relative turn for normalized action `+-1`.
- `--transition-noise-deg`: uniform turn-noise half-range in degrees.
- `--landscape-points`: reward-landscape resolution used for maxima extraction.
- `--global-max-atol`: maxima extraction tolerance.
- `--gif-fps`: GIF frame rate.
- `--save-gif` / `--no-save-gif`: enable or disable GIF export.
- `--double-well-angle-deg`: double-well maxima at `+-angle`.
- `--double-well-height`: double-well barrier reward level.
- `--multi-num-minima`: number of multi-well preferred turn angles.
- `--multi-well-height`: multi-well barrier level.
- `--multi-tail-power`: multi-well tail exponent.
- `--multi-include-opposite-headings` / `--no-multi-include-opposite-headings`: include or exclude opposite reset headings.

Detailed parameter tables are also available in:
- `src/env_utils/README_turning_double_well.md`
