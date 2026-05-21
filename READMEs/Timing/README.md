# Timing Scripts

## 1) Launch benchmark runs for all methods/envs

`run_timing_benchmark.py` launches a Cartesian product: `methods x env.name`.
By default it adds `wandb.project_suffix=_timing` to every launched run.

Example:

```bash
python READMEs/Timing/run_timing_benchmark.py \
  --env-names WalkerRun,HopperHop,FishSwim \
  --methods reppo,ppo,dime,diffppo,dmerl \
  --method-env reppo:mjx_dmc \
  --method-env ppo:mjx_dmc \
  --method-env dime:mjx_dmc \
  --method-env diffppo:mjx_dmc \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides reppo:mjx_dmc_large_data \
  --method-experiment-overrides ppo:mjx_dmc_large_data \
  --method-experiment-overrides dime:mjx_dmc_large_data \
  --method-experiment-overrides diffppo:mjx_dmc_large_data_DA_MDP_PPO_control \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_no_temp \
  --method-diff-steps dime:8 \
  --method-diff-steps diffppo:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 500000 \
  --extra-override hyperparameters.num_eval=10
```

```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names HopperHop,FishSwim \
  --methods dime,dmerl \
  --method-env dime:mjx_dmc \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dime:mjx_dmc_large_data \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_no_temp \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 100000000  \
  --extra-override hyperparameters.num_eval=50
```

```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names HopperHop,FishSwim \
  --methods dmerl \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_no_temp \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 100000000  \
  --extra-override hyperparameters.num_eval=50
```

```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names HopperHop,FishSwim \
  --methods dmerl \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_time \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 100000000  \
  --extra-override hyperparameters.num_eval=50
```


```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names CheetahRun,WalkerRun \
  --methods dime,dmerl \
  --method-env dime:mjx_dmc \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dime:mjx_dmc_large_data \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_no_temp \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 10000000  \
  --extra-override hyperparameters.num_eval=50
```

```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names CheetahRun,WalkerRun \
  --methods dmerl \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_no_temp \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 10000000  \
  --extra-override hyperparameters.num_eval=50
```

```bash
exec python READMEs/Timing/run_timing_benchmark.py \
  --env-names CheetahRun,WalkerRun \
  --methods dmerl \
  --method-env dmerl:mjx_dmc \
  --method-experiment-overrides dmerl:mjx_dmc_large_data_DA_MDP_REPPO_linear_schedule_time \
  --method-diff-steps dime:8 \
  --method-diff-steps dmerl:8 \
  --total-time-steps 10000000  \
  --extra-override hyperparameters.num_eval=50
```

Tips:
- Use `--dry-run` first to print all generated commands.
- Add method-specific extra overrides with `--method-extra-override method:key=value`.
- Override or disable the suffix with `--project-suffix ...` (or `--project-suffix ""`).

## 2) Load runs and export cleaned timing data

`load_timing_runs.py` loads W&B runs, drops the first `N` timing points (JIT warmup), and exports:
- `timing_series_cleaned.csv`
- `timing_run_summary.csv` (includes per-run mean/median/std and integrated time for rollout/update/total)

Notes:
- If `timing/*` is not logged, the loader falls back to `sps` and estimates `timing/total_seconds` from `_step` deltas (`delta_step / sps`).
- With SPS-only runs, rollout/update splits are unavailable; only total-time estimates are meaningful.

Example:

```bash
python READMEs/Timing/load_timing_runs.py \
  --entity sanokows \
  --env-names WalkerRun,HopperHop,FishSwim \
  --drop-first 1 \
  --out-dir READMEs/Timing/outputs
```

## 3) Plot per-env timing stats by method

`plot_timing_stats.py` uses the same loading pipeline and writes:
- `timing_run_summary.csv`
- `timing_env_method_summary.csv`
- `timing_means_<ENV>.png` for each env.
- `timing_integrated_<ENV>.png` for each env (methods compared on cumulative rollout/update/total time).
- `timing_sps_<ENV>.png` for each env (methods compared by SPS).

Example:

```bash
python READMEs/Timing/plot_timing_stats.py \
  --entity sanokows \
  --env-names WalkerRun,HopperHop,FishSwim \
  --drop-first 1 \
  --out-dir READMEs/Timing/outputs
```
