# TurningDoubleWellEnv Quick README

This short guide covers:
- `src/env_utils/turning_double_well_env.py`
- `src/env_utils/test_turning_double_well_env.py`

Run commands from the repository root:
`/home/it4i-sanokows/code/DIMEReppo`

## 1) Run the test + visualization script

```bash
python src/env_utils/test_turning_double_well_env.py \
  --output-dir artifacts/turning_double_well_demo \
  --num-trajectories 10 \
  --horizon 80 \
  --well-height 0.85 \
  --well-angle-deg 35 \
  --transition-noise-deg 5
```

This script:
- runs a smoke test
- renders policy comparison images/GIF
- plots reward/potential views

Artifacts are written to the `--output-dir`.

## 2) Plot reward sweeps for multiple configurations

```bash
python src/env_utils/plot_turning_double_well_reward_configs.py \
  --output-path artifacts/turning_double_well/reward_configs_over_action.png \
  --well-heights 0.2 \
  --peak-degs 45
```

This writes one figure with multiple reward curves over action (and corresponding degrees on the top axis).

## 3) Use `TurningDoubleWellEnv` directly

`turning_double_well_env.py` is a module (not a standalone CLI script).
Use it from Python:

```bash
python - <<'PY'
import jax
import jax.numpy as jnp
from src.env_utils.turning_double_well_env import TurningDoubleWellEnv

env = TurningDoubleWellEnv(
    horizon=100,
    well_height=0.3,
    well_angle_deg=45.0,
    transition_noise_deg=2.0,
    randomize_initial_heading=False,  # set True to sample a random starting orientation
    mask_state_in_observation=False,  # set True to return zero observations
)

state = env.reset(jax.random.PRNGKey(0))
action = jnp.array([-0.5], dtype=jnp.float32)  # -45 deg relative turn (if max_turn_deg=90)
next_state = env.step(state, action)
print("reward:", float(next_state.reward))
print("done:", bool(next_state.done))
PY
```
