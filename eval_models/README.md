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

Example (evaluate for horizon 200):

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084001.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
  --render-width 1200 --render-height 800
```

### action clipping and all directions reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084001

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084336.pkl \
  --horizon 100 \
  --render --render-num-envs 10 \
  --diffusion-sampler sde \
  --render-width 1200 --render-height 800
```

### high temp reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260325T210624
### even higher temp reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260325T215952
### more directions reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T084336

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_dime__TurningDoubleWellEnv__seed0__trial0__ts20260325T220431.pkl \
  --horizon 100 \
  --render --render-num-envs 10 \
  --diffusion-sampler sde \
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

This produces a GIF (and also a PNG snapshot of the last frame) in the repo's `artifacts/` folder. By default, the GIF name is prefixed with `reppo__...` or `DMERL__...` depending on the checkpoint.

Rendering options:

- `--render-out <name>`: GIF basename (still written into `artifacts/`).
- By default all agents are overlaid into a single plot; use `--render-grid` to render one subplot per agent.
- `--render-width <int>` / `--render-height <int>`: output resolution in pixels (defaults: 960x720).
- `--render-fps <int>`: GIF FPS (default: 20).
- `--render-seed <int>`: PRNG seed used for the rendered rollout.

Tip: if you want to *compare* ODE vs SDE visually, run twice with `--diffusion-sampler ode` and `--diffusion-sampler sde`. The renderer writes to `artifacts/` and now includes `__ode__` / `__sde__` in the filename when you set `--diffusion-sampler`, so the outputs won't overwrite each other.
