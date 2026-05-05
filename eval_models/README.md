# Model Evaluation

This folder contains scripts for evaluating checkpoints saved under `saved_models/`.

## Evaluate A Saved Checkpoint

The main entry point is:

```bash
python eval_models/eval_saved_model.py --checkpoint saved_models/<checkpoint>.pkl
```

## Partition Sum / `log Z` Estimation (`reppo`, `reppo_dime`, `reppo_DMERL_new`)

Use `compute_partition_sum.py` to estimate partition terms from saved `reppo`, `reppo_dime`, or `reppo_DMERL_new` checkpoints:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/reppo__FishSwim__trainmodereparam__seed0__trial0__ts20260329T131447.pkl
```

What it computes:

- Roll out `X = num_envs` environments in parallel for `Y = horizon` steps.
- For `reppo`:
  - At each step, store reward `r_t` and policy log-probability `log q(a_t|s_t)`.
  - For each trajectory:
    - `R = sum_t r_t`
    - `log q = sum_t log q(a_t|s_t)`
    - `log w = (1 / T) * R - log q`
- For `reppo_dime`:
  - At each environment step, also compute diffusion-path terms:
    - `sum(log p - log q)` over diffusion steps
    - `log_prior` from the sampled diffusion prior variable
  - For each trajectory:
    - `R = sum_t r_t`
    - `log w = (1 / T) * R + sum_t(sum_diffusion_steps(log p - log q)) - sum_t(log_prior)`
- For `reppo_DMERL_new`:
  - At each diffusion step, store:
    - `gen_log_prob` (forward/generation step log-probability)
    - `dest_log_prob` (destination/backward step log-probability)
  - At diffusion-chain final steps, apply the tanh-Jacobian correction
    `log|det d(tanh(x))/dx|` so weights are in the transformed action space.
    (If `tanh_transform=true` in the actor, this is already inside `gen_log_prob` and is not double-counted.)
  - `log_prior` is evaluated at diffusion-chain starts (`diff_time_step == 0`) and subtracted.
  - For each trajectory:
    - `R = sum_t r_t`
    - `log w = (1 / T) * R + sum_t(gen_log_prob - dest_log_prob - final_step_tanh_log_det) - sum_t(log_prior)`
- Partition estimates:
  - `log_Z_sum = log(sum_i exp(log w_i))`
  - `log_Z_mean = log((1/N) * sum_i exp(log w_i))`

### More Data With Repeats (`K`)

To collect more trajectories, repeat the rollout loop `K` times:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/<reppo_checkpoint>.pkl \
  --rollout-repeats 8
```

Total sampled trajectories become:

- `N = num_envs * rollout_repeats`

### `log Z` vs Number of Samples

The script also computes `log Z` over varying sample counts and writes a plot.

Each plotted point uses an independent random subsample of trajectories (without replacement) for that sample count.

Default curve points:

- one point every `num_envs` trajectories
- total number of points = `rollout_repeats`
- sample counts are: `num_envs, 2*num_envs, ..., rollout_repeats*num_envs`

Example:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/<reppo_checkpoint>.pkl \
  --rollout-repeats 16
```

- Explicit sample counts:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/<reppo_checkpoint>.pkl \
  --rollout-repeats 16 \
  --sample-counts 64 128 256 512 1024 2048 4096
```

- Linear x-axis example:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/reppo__FishSwim__trainmodereparam__seed0__trial0__ts20260329T131447.pkl \
  --num-envs 4024 \
  --horizon 1000 \
  --temperature 0.1 \
  --rollout-repeats 100 \
  --plot-xscale linear
```

- `reppo_dime` example:

```bash
python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/reppo_dime__FishSwim__seed0__trial0__ts20260329T143011.pkl \
  --num-envs 2024 \
  --horizon 1000 \
  --temperature 0.1 \
  --rollout-repeats 100 \
  --plot-xscale linear
```

- `reppo_DMERL_new` example:

```bash
exec python eval_models/compute_partition_sum.py \
  --checkpoint saved_models/reppo_DMERL_new__FishSwim__trainmodereparam__seed0__trial0__ts20260326T153220.pkl \
  --num-envs 2024 \
  --horizon 1000 \
  --temperature 0.1 \
  --rollout-repeats 100 \
  --plot-xscale linear
```

By default, the x-axis mode is `--plot-xscale auto`, which switches to log-scale when the maximum sample count is greater than `50`.

### Important Options

- `--temperature <float>`: sets `T` in `exp((1/T) * R)`.
- `--rollout-repeats <int>`: number of repeated rollouts (`K`).
- `--no-progress`: disable the rollout progress bar.
- `--horizon <int>`: rollout length override.
- `--num-envs <int>`: parallel env override.
- `--sample-counts ...`: explicit sample counts for the curve.
- `--plot-xscale {auto,log,linear}`: x-axis scale of the output plot.
- `--curve-seed <int>`: seed for independent trajectory subsampling at each curve point.

### Outputs

By default outputs are written under `artifacts/partition_sum/`:

- `*.json`: summary metrics (`log_Z_sum`, `log_Z_mean`, ESS, curve arrays, etc.)
- `*.npz`: stored arrays:
  - per-step: `rewards`, `log_q_steps`, `log_p_minus_log_q_steps`, `log_prior_steps` with shape `[K, horizon, num_envs]`
  - per-trajectory: `returns`, `log_q_traj`, `log_p_traj`, `log_w_traj`, `log_p_minus_log_q_traj`, `log_prior_traj` with shape `[K, num_envs]`
  - flattened arrays and sample-count curve arrays
- `*.png`: plot of `log Z_mean` estimate vs number of samples

## DMERL Alpha Sweep Evaluation

For checkpoints trained with `src/jaxrl/reppo_DMERL_new.py`, use:

```bash
exec python eval_models/eval_reppo_DMERL_alpha_sweep.py \
  --checkpoint saved_models/reppo_DMERL_new__FishSwim__trainmodereparam__seed0__trial0__ts20260329T150527.pkl \
  --alpha-start 0.0 \
  --alpha-end 1. \
  --alpha-steps 20 \
  --guidance-temperature 0.03 \
  --guidance-dts 0.1 0.25 0.5 0.75 0.01 0.05   \
  --guidance-temperature 0.0007 \
   --num-envs 8000 \
   --diffusion-sampler ode \
      --q-grad-clip 100\
     --q-guidance-last-percent 75
```

```bash
exec python eval_models/eval_reppo_DMERL_alpha_sweep.py \
  --checkpoint saved_models/reppo_DMERL_new__AcrobotSwingup__trainmodereparam__seed0__trial0__ts20260330T140419.pkl \
  --alpha-start 0.0 \
  --alpha-end 1. \
  --alpha-steps 20 \
  --guidance-temperature 0.03 \
  --guidance-dts 1. 1.5 2. 4 3.   \
  --guidance-temperature 0.04 \
   --num-envs 8000 \
   --diffusion-sampler ode \
      --q-grad-clip 100\
     --q-guidance-last-percent 75
```

```bash
exec python eval_models/eval_reppo_DMERL_alpha_sweep.py \
  --checkpoint saved_models/reppo_DMERL_new__HopperHop__trainmodereparam__seed0__trial0__ts20260330T145700.pkl \
  --alpha-start 0.0 \
  --alpha-end 1. \
  --alpha-steps 20 \
  --guidance-temperature 0.03 \
  --guidance-dts 0.25 0.5 1. 0.75   \
  --guidance-temperature 0.003 \
   --num-envs 16000 \
   --diffusion-sampler ode \
      --q-grad-clip 100\
     --q-guidance-last-percent 75
```

This evaluates using a guided score:

- `guided_score = (1 - alpha) * score + alpha * dt * (grad_log_p + (1 / T) * grad_a Q(s, a))`

where `grad_log_p` is the gradient of forward-process log probability w.r.t. action.
`T` is set with `--guidance-temperature` (default `1.0`).
If `--guidance-dts` is not provided, `dt=1.0` is used.

and reports rollout metrics (including episode return) for every alpha.
By default it evaluates both samplers (`ode` and `sde`) in one run.
If you pass multiple `dt` values, it runs one alpha-sweep per `dt` and adds one line per `(sampler, dt)` to the final plot.
It also writes a final `episode_return` vs `alpha` plot (PNG) unless you pass `--plot-out`.
Default outputs are saved to `artifacts/guidance/`:

- `<checkpoint>__alpha_sweep_episode_return.png`
- `<checkpoint>__alpha_sweep_results.json`

Notes:

- `--diffusion-sampler both` (default) evaluates both `ode` and `sde`.
- `--diffusion-sampler auto` matches training mode:
  - `WPO -> sde`
  - otherwise `ode`
- Plot uncertainty bands are **standard error** (`SEM = std / sqrt(num_episodes)`), not standard deviation.
- You can pass an explicit alpha list instead of a range:
  - `--alphas 0.0 0.1 0.25 0.5 0.75 1.0`
- You can sweep guidance dt values:
  - `--guidance-dts 0.25 0.5 1.0`
- You can set the Q-gradient temperature scale with:
  - `--guidance-temperature 0.03`
- Use `--num-envs <int>` to override the number of parallel evaluation envs.
- `--normalizer-stats-mode` controls normalization behavior during eval:
  - `fixed` (default): use checkpoint normalization stats and keep them fixed.
  - `online`: update stats online during rollout.
  - `off`: disable normalization.
- Use `--seed-idx` for multi-seed checkpoints.
- Use `--output-json <path>` to override the default JSON path in `artifacts/guidance`.
- Use `--plot-out <path>` to control where the return-vs-alpha PNG is written.
- Use `--q-guidance-last-percent <float>` to apply Q guidance only in the last X% of diffusion steps.
  Example: `--q-guidance-last-percent 25` applies Q-guidance only in the final quarter of steps.
  `0` disables Q-guidance, `100` applies it at all steps.
- By default, no clipping is applied to `(grad_log_p + grad_a Q)`.
  Use `--q-grad-clip <float>` to enable clipping.
  Choose mode with `--q-grad-clip-mode {l2,elementwise}`.

## DMERL Guidance Ratio-By-Step Plot

To inspect how strong the base score is relative to the guidance vector across diffusion steps, run:

```bash
python eval_models/eval_reppo_DMERL_guidance_ratio_by_step.py \
  --checkpoint saved_models/reppo_DMERL_new__AcrobotSwingup__trainmodereparam__seed0__trial0__ts20260330T140419.pkl \
  --diffusion-sampler both \
  --num-envs 8000 \
  --normalizer-stats-mode fixed \
  --q-grad-clip 100 \
  --q-grad-clip-mode l2
```

This computes, for each diffusion step:

- `ratio = ||score||_2 / ||(grad_log_p + grad_a Q)||_2`

and reports mean and SEM over all valid rollout samples that hit each diffusion step.

Default outputs are saved to `artifacts/guidance/`:

- `<checkpoint>__guidance_ratio_by_step.png`
- `<checkpoint>__guidance_ratio_by_step.json`

Common options:

- `--horizon <int>`: override `env.max_episode_steps` (and `hyperparameters.max_episode_steps` when present).
- `--seed-idx <int>`: pick which trained seed index to evaluate when the checkpoint contains multiple seeds.
- `--num-envs <int>`: override `hyperparameters.num_envs`.
  This is only safe when `hyperparameters.normalize_env=false` (the script will ignore it otherwise).
- `--env-config-override key=value` (repeatable): override entries in `cfg.env.config` loaded from the checkpoint.
  For `TurningDoubleWellEnv` this maps to `TurningDoubleWellEnv(**kwargs)` (e.g. `randomize_initial_heading=true`).
- `--collect-trajectories`: save state trajectories for offline analysis.
- `--traj-num-envs <int>` / `--traj-repeats <int>`: run `X` parallel envs repeated `Y` times (`X*Y` trajectories total).
- `--traj-knn-mode {both,trajectories,states}`: compute kNN entropy over either:
  1. full trajectories as points (one trajectory = one point),
  2. pooled states as points (one state = one point), or
  3. both (default).

Trajectory data is saved to `eval_models/saved_trajectories/<run>/trajectories.npz` with:

- `state_trajectories`: shape `[repeats, horizon+1, num_envs, state_dim]`

and metadata in `metadata.json`.

Example:

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --collect-trajectories \
  --traj-num-envs 8000 \
  --traj-repeats 4 \
  --traj-knn-mode both
```


```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T102744.pkl \
  --collect-trajectories \
  --traj-num-envs 1000 \
  --traj-repeats 4 \
  --traj-knn-mode both
```

```bash
exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T102744.pkl \
  --collect-trajectories \
  --traj-num-envs 4000 \
  --traj-repeats 4 \
  --traj-knn-mode both\
    --traj-knn-max-samples 20000 \
  --traj-knn-k 5
```
```bash
exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T105557.pkl \
  --collect-trajectories \
  --traj-num-envs 4000 \
  --traj-repeats 4 \
  --traj-knn-mode both \
  --traj-knn-max-samples 20000 \
  --traj-knn-k 5
```

```bash
exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_dime__TurningDoubleWellEnv__seed0__trial0__ts20260326T110236.pkl \
  --collect-trajectories \
  --traj-num-envs 4000 \
  --traj-repeats 4 \
  --traj-knn-mode both \
  --traj-knn-max-samples 20000 \
  --traj-knn-k 5
```


### Horizon And `k`

- Set rollout horizon with `--horizon <int>`. If omitted, the checkpoint `env.max_episode_steps` is used.
- Saved trajectory length is always `horizon + 1` states (initial state + one state per env step).
- Set kNN neighborhood size with `--traj-knn-k <int>`.
- Practical default is `k=5`; compare with `k=3` and `k=10` for sensitivity.
- For trajectory-point entropy (`--traj-knn-mode trajectories`), sample count is `X*Y` so keep `k` relatively small unless `X*Y` is large.
- For state-point entropy (`--traj-knn-mode states`), sample count is `(horizon+1)*X*Y`, so larger `k` is usually stable.
- The evaluator automatically caps `k` to be `< N` in each mode.

### State Definition For Diffusion Methods

- For `reppo_DMERL_new` and `reppo_DiffPPO`, trajectory collection records states only when the underlying environment advances (i.e. at the last diffusion step of each diffusion cycle), plus the initial reset state.
- Stored states are taken from the diffusion wrapper's raw underlying environment observation (`MjxDiffEnvState.obs`), not the normalized observation tensor used by the policy wrapper.

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

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DiffPPO__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260327T175708.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --diffusion-sampler sde \
  --render-width 1200 --render-height 800
```


```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DiffPPO__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260327T175700.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --diffusion-sampler sde \
  --render-width 1200 --render-height 800
```

### WPO
```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodeWPO__seed0__trial0__ts20260328T123545.pkl \
  --horizon 100 \
  --render --render-num-envs 20 \
    --tdw-action-analysis \
  --tdw-action-analysis-samples 5000 \
  --tdw-action-analysis-grid 401 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --diffusion-sampler sde \
  --render-width 1200 --render-height 800
```



The script auto-detects whether the checkpoint is for `reppo`, `reppo_DMERL_new`, `reppo_DiffPPO`, or `reppo_dime` via `checkpoint["method_name"]` (with a config-based fallback for older checkpoints).

## MJX Rollout-Grid Rendering (Eval-Only)

For MJX checkpoints (for example `FishSwim`) across `reppo`, `reppo_DMERL_new`, `reppo_DiffPPO`, and `reppo_dime`, `--render` records tiled rollout grids (e.g., 20 envs in one frame) during evaluation only:



```bash
exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__HopperHop__trainmodereparam__seed0__trial0__ts20260326T152835.pkl \
  --render \
  --render-format mp4 \
  --render-num-envs 16 \
  --render-cell-width 180 \
  --render-cell-height 180 \
  --render-frame-stride 1 \
  --render-max-frames 1000 \
  --render-camera cam0 \
  --render-follow-agent \
  --render-follow-body torso
  ```


exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo_DMERL_new__HopperHop__trainmodereparam__seed0__trial0__ts20260326T152835.pkl \
  --render \
  --render-format mp4 \
  --render-num-envs 16 \
  --render-cell-width 180 \
  --render-cell-height 180 \
  --render-frame-stride 3 \
  --render-max-frames 1000 \
  --render-camera cam0


```bash
exec python eval_models/eval_saved_model.py \
  --checkpoint saved_models/reppo__FishSwim__trainmodereparam__seed0__trial0__ts20260329T131447.pkl \
  --render \
  --render-num-envs 16 \
  --render-cell-width 180 \
  --render-cell-height 180 \
  --render-frame-stride 1 \
  --render-max-frames 1000
```

Outputs are stored in a dedicated run folder under `artifacts/`, e.g. `artifacts/<run_folder>/` with:

- `*.gif` and/or `*.mp4` tiled rollout animation
- `*.html` autoplay-loop video preview (written when MP4 is enabled)
- `last_frame.png`
- `metadata.json`

Useful MJX render options:

- `--render-camera <name-or-index>`: choose a specific MuJoCo camera.
- `--render-follow-agent`: turn the selected camera into tracking mode (agent-following).
- `--render-follow-body <name-or-index>`: body to track when follow mode is on (default auto body, usually torso/root).
- `--render-format {gif,mp4,both}`: choose GIF, MP4, or both (default: `gif`).

## TurningDoubleWellEnv Trajectory Rendering

If `cfg.env.name == "TurningDoubleWellEnv"`, you can also render a batch of trajectories (X agents in parallel), similar to `src/env_utils/visualization/test_turning_double_well_env.py` (works for `reppo`, `reppo_DMERL_new`, `reppo_DiffPPO`, and `reppo_dime` checkpoints):

```bash
python eval_models/eval_saved_model.py \
  --checkpoint saved_models/<checkpoint>.pkl \
  --horizon 100 \
  --render \
  --render-num-envs 10
```

This produces a GIF (and also a PNG snapshot of the last frame) in a dedicated run folder under `artifacts/`. By default, the GIF name is prefixed with `REPPO__...`, `DME-REPPO__...`, `DME-WPO__...` (for WPO mode), or `REPPO-DIME__...` depending on the checkpoint/method.

Rendering options:

- `--render-out <name>`: output basename (still written into `artifacts/<run_folder>/`).
- By default all agents are overlaid into a single plot; use `--render-grid` to render one subplot per agent.
- `--render-width <int>` / `--render-height <int>`: output resolution in pixels (defaults: 960x720).
- `--render-fps <int>`: GIF FPS (default: 20).
- `--render-format {gif,mp4,both}`: output format (default: `gif`). MP4 also writes a looping autoplay HTML preview file.
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

## TurningDoubleWellEnv Paper Composites (6 Methods + Core4 Variant)

To auto-generate compact paper-ready composites in the fixed method order

`REPPO | DPPO (zero temp) | DME-PPO | DMERL | DMERL-WPO | DIME`

use:

```bash
python eval_models/make_tdw_paper_figures.py \
  --checkpoint-reppo saved_models/reppo__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T102744.pkl \
  --checkpoint-diffppo saved_models/reppo_DiffPPO__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260327T175708.pkl  \
  --checkpoint-diffppo saved_models/reppo_DiffPPO__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260327T175700.pkl \
  --checkpoint-dmerl saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodereparam__seed0__trial0__ts20260326T105557.pkl \
  --checkpoint-dmerl-wpo saved_models/reppo_DMERL_new__TurningDoubleWellEnv__trainmodeWPO__seed0__trial0__ts20260328T123545.pkl \
  --checkpoint-dime saved_models/reppo_dime__TurningDoubleWellEnv__seed0__trial0__ts20260326T110236.pkl \
  --sampler-diffppo sde \
  --sampler-diffppo-zero ode \
  --horizon 100 \
  --analysis-samples 5000 \
  --analysis-grid 401 \
  --analysis-bins 60 \
  --hist-legend-fontsize 18 \
  --hist-axis-label-fontsize 11 \
  --hist-axis-tick-fontsize 14 \
  --env-config-override randomize_initial_heading=true \
  --env-config-override snap_action_to_optimal=true \
  --render-num-envs 20 \
  --render-width 1200 \
  --render-height 800 \
  --trajectory-scale-fontsize 15 \
  --output-dir artifacts/tdw_paper \
  --output-stem tdw_paper_main
```

`--checkpoint-diffppo-zero` is optional. If omitted, the script auto-detects a DiffPPO checkpoint with `entropy_coef` closest to `0` (same env and, when available, same seed/trial).
If you pass `--checkpoint-diffppo` multiple times, zero-temp selection is done among those provided runs.
Use `--hist-legend-fontsize` to control the histogram legend text size, `--hist-axis-label-fontsize`/`--hist-axis-tick-fontsize` to control histogram axis text, and `--trajectory-scale-fontsize` to control the trajectory scale-label text size.

This writes:

- `<output-dir>/<output-stem>__action_hist_row.png`  
  One row (6 columns), each column overlays all state-conditioned action histograms for one method, with a shared state legend and reward curve.
- `<output-dir>/<output-stem>__trajectory_row.png`  
  One row (6 columns) of static trajectory snapshots (not GIFs), one per method, with in-panel scale bars and no x/y ticks.
- `<output-dir>/<output-stem>__trajectory_row_no_title.png`  
  Same as above, but with no per-panel titles at all.
- `<output-dir>/<output-stem>__action_hist_row_core4.png`  
  Additional one-row (4 columns) action-hist composite for `REPPO | DPPO (T=0) | DME-PPO | DMERL`.
- `<output-dir>/<output-stem>__trajectory_row_core4.png`  
  Additional one-row (4 columns) trajectory composite for `REPPO | DPPO (T=0) | DME-PPO | DMERL`.
- `<output-dir>/<output-stem>__trajectory_row_core4_no_title.png`  
  Same core4 trajectory composite with no per-panel titles.
- `<output-dir>/<output-stem>__manifest.json`  
  Metadata with checkpoint paths, resolved samplers, source render artifacts, and grouped figure paths (`full6` and `core4`).
