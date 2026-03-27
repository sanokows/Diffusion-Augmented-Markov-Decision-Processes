# Model Evaluation

This folder contains scripts for evaluating checkpoints saved under `saved_models/`.

## Evaluate A Saved Checkpoint

The main entry point is:

```bash
python eval_models/eval_saved_model.py --checkpoint saved_models/<checkpoint>.pkl
```

Common options:

- `--horizon <int>`: override `env.max_episode_steps` (and `hyperparameters.max_episode_steps` when present).
- `--seed-idx <int>`: pick which trained seed index to evaluate when the checkpoint contains multiple seeds.
- `--num-envs <int>`: override `hyperparameters.num_envs`.
  This is only safe when `hyperparameters.normalize_env=false` (the script will ignore it otherwise).
- `--env-config-override key=value` (repeatable): override entries in `cfg.env.config` loaded from the checkpoint.
  For `TurningDoubleWellEnv` this maps to `TurningDoubleWellEnv(**kwargs)` (e.g. `randomize_initial_heading=true`).

### Overriding Environment Flags

`eval_saved_model.py` reconstructs the environment from the checkpoint’s saved `cfg`. To override environment flags at evaluation time, use `--env-config-override` one or more times:

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=false \
  --env-config-override well_angle_deg=22.5
```

Notes:

- Values are type-parsed: `true/false`, ints, floats, `None`, and Python literals like `[1, 2]` / `{"k": 1}` are supported; otherwise the value is treated as a string.
- For `TurningDoubleWellEnv`, common overrides include `horizon`, `max_turn_deg`, `randomize_initial_heading`, `snap_action_to_optimal`, `well_angle_deg`, `well_height`, and `transition_noise_deg`.

Example (evaluate for horizon 200):

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T102744.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --render-width 1200 --render-height 800
```

### action clipping and all directions reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084001

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T105557.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --render-width 1200 --render-height 800
```

### high temp reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260325T210624
### even higher temp reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260325T215952
### more directions reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084336

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_dime__TurningDoubleWellEnv__seed0__trial0__ts20260326T110236.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --render-width 1200 --render-height 800
```

### dime high temp reppo_dime__TurningDoubleWellEnv__seed0__trial0__ts20260325T220431

The script auto-detects whether the checkpoint is for `reppo` vs `reppo_DMERL_new` via `checkpoint["method_name"]` (and falls back to checking whether `cfg.hyperparameters.diffusion` exists).

## TurningDoubleWellEnv Trajectory Rendering

If `cfg.env.name == "TurningDoubleWellEnv"`, you can also render a batch of trajectories (X agents in parallel), similar to `src/env_utils/test_turning_double_well_env.py` (works for both `reppo` and `reppo_DMERL_new` checkpoints):

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --horizon 100 \
  --render \
  --render-num-envs 10
```

This produces a GIF (and also a PNG snapshot of the last frame) in the repo's `artifacts/` folder. By default, the GIF name is prefixed with `REPPO__...`, `DME-REPPO__...`, or `REPPO-DIME__...` depending on the checkpoint/method.

Rendering options:

- `--render-out <name>`: GIF basename (still written into `artifacts/`).
- By default all agents are overlaid into a single plot; use `--render-grid` to render one subplot per agent.
- `--render-width <int>` / `--render-height <int>`: output resolution in pixels (defaults: 960x720).
- `--render-fps <int>`: GIF FPS (default: 20).
- `--render-seed <int>`: PRNG seed used for the rendered rollout.

Tip: if you want to *compare* ODE vs SDE visually, run twice with `--diffusion-sampler ode` and `--diffusion-sampler sde`. The renderer writes to `artifacts/` and now includes `__ode__` / `__sde__` in the filename when you set `--diffusion-sampler`, so the outputs won't overwrite each other.

## TurningDoubleWellEnv Action / Q Analysis

If `cfg.env.name == "TurningDoubleWellEnv"`, you can also generate an analysis plot that:

- Enumerates all discrete starting headings (multiples of `well_angle_deg`, assuming `randomize_initial_heading=true` and `snap_action_to_optimal=true`)
- For each heading, samples many actions from the loaded policy and plots a histogram
- Overlays the environment’s one-step reward curve as a function of action
- Optionally plots `Q(s,a)` over the action grid in a separate PNG (disabled by default; enable with `--tdw-action-analysis-q`)

Example:

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401
```

To also compute and plot `Q(s,a)` (when the checkpoint includes critic params / the method has a critic):

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --tdw-action-analysis --tdw-action-analysis-q
```

Outputs are written as PNG files into `artifacts/` (see `--tdw-action-analysis-out` to control the basename):

- Histogram + reward plot: `<basename>.png`
- Q-function plot (when `--tdw-action-analysis-q` is set): `<basename>__q.png`
