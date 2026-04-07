"""Benchmark latency of env.step(...) used in scripts/debug_mjx_state.py.

This script measures the runtime of:
    obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(env_subkeys, prev_state, action)

It reports:
- first-call JIT compile+execute time
- steady-state per-call latency stats after warmup
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass

import jax
import numpy as np
from ml_collections import ConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from src.env_utils.jax_wrappers import (
    DiffNormalizeVec,
    LogWrapper,
    MjxDiffEnvWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)


DEFAULT_ENV_NAME = "G1JoystickFlatTerrain"
DEFAULT_EPISODE_LENGTH = 200
DEFAULT_DIFF_STEPS_LIST = (3, 5, 10)


@dataclass(frozen=True)
class BenchmarkSetting:
    name: str
    use_diff_wrapper: bool
    diff_steps: int | None


def create_diffusion_config(diff_steps: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.diff_steps = diff_steps
    cfg.init_std = 3.0
    return cfg


def build_env(
    env_name: str,
    episode_length: int,
    num_envs: int,
    use_diff_wrapper: bool,
    diff_steps: int | None,
):
    base_env = MjxGymnaxWrapper(env_name, episode_length=episode_length)
    if use_diff_wrapper:
        if diff_steps is None:
            raise ValueError("diff_steps must be set when use_diff_wrapper=True")
        diff_cfg = create_diffusion_config(diff_steps)
        env = MjxDiffEnvWrapper(
            base_env,
            num_diff_steps=diff_steps,
            diffusion_config=diff_cfg,
        )
        env = LogWrapper(env, num_envs=num_envs)
        env = DiffNormalizeVec(env, num_diff_steps=diff_steps)
        return env

    env = LogWrapper(base_env, num_envs=num_envs)
    env = NormalizeVec(env)
    return env


def make_settings(include_base: bool, diff_steps_list: list[int]) -> list[BenchmarkSetting]:
    settings: list[BenchmarkSetting] = []
    if include_base:
        settings.append(BenchmarkSetting(name="base", use_diff_wrapper=False, diff_steps=None))
    for ds in diff_steps_list:
        settings.append(
            BenchmarkSetting(
                name=f"diff_{ds}",
                use_diff_wrapper=True,
                diff_steps=ds,
            )
        )
    if not settings:
        raise ValueError("No settings selected. Enable --include-base and/or provide --diff-steps-list.")
    return settings


def summarize_ms(samples_sec: np.ndarray) -> dict[str, float]:
    samples_ms = samples_sec * 1000.0
    return {
        "mean_ms": float(np.mean(samples_ms)),
        "std_ms": float(np.std(samples_ms)),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
    }


def benchmark_setting(
    setting: BenchmarkSetting,
    env_name: str,
    episode_length: int,
    num_envs: int,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float | int | str | bool]:
    env = build_env(
        env_name=env_name,
        episode_length=episode_length,
        num_envs=num_envs,
        use_diff_wrapper=setting.use_diff_wrapper,
        diff_steps=setting.diff_steps,
    )

    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    _, _, state = env.reset(reset_keys)

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]
    action_shape = (num_envs, action_dim)

    def step_once(prev_state, env_subkeys, action):
        return env.step(env_subkeys, prev_state, action)

    step_once_jit = jax.jit(step_once)

    rng, action_key, env_key = jax.random.split(rng, 3)
    action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
    env_subkeys = jax.random.split(env_key, num_envs)

    t0 = time.perf_counter()
    out = step_once_jit(state, env_subkeys, action)
    jax.block_until_ready(out[3])
    compile_and_first_sec = time.perf_counter() - t0
    state = out[2]

    for _ in range(warmup_steps):
        rng, action_key, env_key = jax.random.split(rng, 3)
        action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
        env_subkeys = jax.random.split(env_key, num_envs)
        out = step_once_jit(state, env_subkeys, action)
        jax.block_until_ready(out[3])
        state = out[2]

    samples_sec = np.zeros((timed_steps,), dtype=np.float64)
    for i in range(timed_steps):
        rng, action_key, env_key = jax.random.split(rng, 3)
        action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
        env_subkeys = jax.random.split(env_key, num_envs)

        t1 = time.perf_counter()
        out = step_once_jit(state, env_subkeys, action)
        jax.block_until_ready(out[3])
        samples_sec[i] = time.perf_counter() - t1
        state = out[2]

    stats = summarize_ms(samples_sec)
    stats.update(
        {
            "setting": setting.name,
            "use_diff_wrapper": setting.use_diff_wrapper,
            "diff_steps": -1 if setting.diff_steps is None else int(setting.diff_steps),
            "compile_first_ms": float(compile_and_first_sec * 1000.0),
            "timed_steps": int(timed_steps),
            "num_envs": int(num_envs),
            "env_steps_per_sec": float(num_envs / np.mean(samples_sec)),
        }
    )
    return stats


def print_results(rows: list[dict[str, float | int | str | bool]]) -> None:
    header = (
        f"{'setting':<10} {'diff':<5} {'compile+1st(ms)':>16} {'mean(ms)':>10} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8} {'env_steps/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['setting']):<10} "
            f"{str(r['diff_steps']):<5} "
            f"{r['compile_first_ms']:16.3f} "
            f"{r['mean_ms']:10.3f} "
            f"{r['p50_ms']:8.3f} "
            f"{r['p95_ms']:8.3f} "
            f"{r['p99_ms']:8.3f} "
            f"{r['min_ms']:8.3f} "
            f"{r['max_ms']:8.3f} "
            f"{r['env_steps_per_sec']:12.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MJX env.step call latency.")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="MJX env name.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help="Episode length passed to MjxGymnaxWrapper.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized envs.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup calls after compile before timing.",
    )
    parser.add_argument(
        "--timed-steps",
        type=int,
        default=1000,
        help="Number of measured step calls.",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Include non-diff wrapper setting.",
    )
    parser.add_argument(
        "--diff-steps-list",
        type=int,
        nargs="*",
        default=list(DEFAULT_DIFF_STEPS_LIST),
        help="Diffusion steps settings for MjxDiffEnvWrapper.",
    )
    return parser.parse_args()


def main() -> None:
    # Example:
    # python scripts/benchmark_mjx_step.py --include-base --diff-steps-list 1 3 5 --timed-steps 500
    args = parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.timed_steps <= 0:
        raise ValueError("--timed-steps must be > 0")

    settings = make_settings(args.include_base, args.diff_steps_list)

    print("Benchmarking env.step(...) latency")
    print(
        "Config: "
        f"env={args.env_name}, episode_length={args.episode_length}, num_envs={args.num_envs}, "
        f"warmup_steps={args.warmup_steps}, timed_steps={args.timed_steps}"
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for setting in settings:
        print(
            f"Running setting={setting.name} "
            f"(use_diff_wrapper={setting.use_diff_wrapper}, diff_steps={setting.diff_steps})"
        )
        row = benchmark_setting(
            setting=setting,
            env_name=args.env_name,
            episode_length=args.episode_length,
            num_envs=args.num_envs,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        rows.append(row)

    print_results(rows)


if __name__ == "__main__":
    main()
