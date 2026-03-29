# Turning Envs Scripts Quick Guide

This guide covers the two runnable scripts in `src/env_utils`:
- `plot_turning_double_well_reward_configs.py`
- `test_turning_double_well_env.py`

Run commands from the repository root:
`/home/it4i-sanokows/code/DIMEReppo`

## 1) Plot per-start-heading reward landscapes

Script:
`python src/env_utils/plot_turning_double_well_reward_configs.py`

Example:
```bash
python src/env_utils/plot_turning_double_well_reward_configs.py \
  --output-dir artifacts/_tmp_turning_plots \
  --num-points 1001 \
  --double-well-angle-deg 45 \
  --multi-num-minima 4
```

What it writes:
- `double_well_reward_landscape_by_start_heading.png`
- `multi_well_reward_landscape_by_start_heading.png`

Important plotting behavior:
- X-axis is always the relative turn angle with respect to the agent orientation.
- Each subplot corresponds to one unique start heading from the env reset support.

### Argparse reference (`plot_turning_double_well_reward_configs.py`)

| Argument | Default | Meaning |
|---|---:|---|
| `--output-dir` | `artifacts/turning_reward_landscapes` | Output directory for generated figures. |
| `--num-points` | `1001` | Number of samples along action/turn axis. |
| `--global-max-atol` | `1e-4` | Absolute tolerance used to detect global maxima in reward landscape. |
| `--subplot-cols` | auto | Number of columns in subplot grid. If omitted, an automatic square-ish grid is used. |
| `--max-turn-deg` | `90.0` | Maps normalized action `[-1,1]` to turn angle `[-max_turn_deg, max_turn_deg]`. |
| `--double-well-angle-deg` | `45.0` | Location of double-well reward peaks at `+-angle`. |
| `--double-well-height` | `0.3` | Barrier/center reward level for double-well env. |
| `--multi-num-minima` | `4` | Number of reward maxima for multi-well env. |
| `--multi-well-height` | `0.3` | Barrier level between neighboring multi-well maxima. |
| `--multi-tail-power` | `2.0` | Tail sharpness for multi-well reward near `+-90` deg. |
| `--multi-include-opposite-headings` / `--no-multi-include-opposite-headings` | `True` | Include or exclude `+180` heading copies in snapped reset support for multi-well env. |

## 2) Simulate random-global-maxima agents

Script:
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

Policy behavior:
- The script computes global reward maxima from each env's reward landscape.
- At each step, each agent randomly samples one of those maxima and applies the corresponding action.

Artifacts per env directory (`double_well/` or `multi_well/`):
- `reward_landscape_with_global_maxima.png`
- `random_global_maxima_policy_summary.png`
- `random_global_maxima_policy.gif` (when `--save-gif`)

### Argparse reference (`test_turning_double_well_env.py`)

| Argument | Default | Meaning |
|---|---:|---|
| `--env` | `both` | Which env(s) to run: `double`, `multi`, or `both`. |
| `--output-dir` | `artifacts/turning_minima_sampling` | Root output directory for run artifacts. |
| `--horizon` | `80` | Episode length in steps. |
| `--num-trajectories` | `16` | Number of parallel agents/rollouts. |
| `--seed` | `0` | Base RNG seed for rollouts. |
| `--max-turn-deg` | `90.0` | Max relative turn for normalized action `+-1`. |
| `--transition-noise-deg` | `0.0` | Uniform transition noise half-range in degrees. |
| `--landscape-points` | `4001` | Resolution used to estimate global maxima from reward landscape. |
| `--global-max-atol` | `1e-4` | Tolerance when grouping/selecting maxima samples. |
| `--gif-fps` | `8` | Frame rate for saved GIF. |
| `--save-gif` / `--no-save-gif` | `True` | Enable or disable GIF export. |
| `--double-well-angle-deg` | `45.0` | Double-well maxima at `+-angle` deg. |
| `--double-well-height` | `0.3` | Double-well center/barrier reward level. |
| `--multi-num-minima` | `4` | Number of preferred turn maxima in multi-well env. |
| `--multi-well-height` | `0.3` | Multi-well barrier level between neighboring maxima. |
| `--multi-tail-power` | `2.0` | Multi-well tail polynomial exponent. |
| `--multi-include-opposite-headings` / `--no-multi-include-opposite-headings` | `True` | Include or exclude opposite-heading reset support for multi-well env. |

## 3) Use env modules directly from Python

- `src/env_utils/turning_double_well_env.py`
- `src/env_utils/turning_multi_well_env.py`

Both modules expose `reset`, `step`, `reward_from_angle`, and plotting/render helpers.
