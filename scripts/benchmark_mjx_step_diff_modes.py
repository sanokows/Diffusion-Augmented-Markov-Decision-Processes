"""Benchmark env.step(...) latency for fixed diffusion-step modes.

Measures the runtime of this call (from scripts/debug_mjx_state.py):
    obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(env_subkeys, prev_state, action)

Modes:
1) use_diff_wrapper=True, forced diff step = last (num_diff_steps - 1)
2) use_diff_wrapper=True, forced diff step = 0
3) use_diff_wrapper=False

For each mode and env, the script runs multiple repeats, reports per-call mean/std
and percentile stats.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from src.env_utils.jax_wrappers import (  # noqa: E402
    DiffNormalizeVec,
    LogWrapper,
    MjxDiffEnvWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)


DEFAULT_ENV_NAMES = ("PlanarPathEnv", "G1JoystickFlatTerrain")
DEFAULT_EPISODE_LENGTH = 200
DEFAULT_DIFF_STEPS = 5


@dataclass(frozen=True)
class BenchmarkSetting:
    name: str
    use_diff_wrapper: bool
    force_diff_step: int | None


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
    diff_steps: int,
):
    base_env = MjxGymnaxWrapper(env_name, episode_length=episode_length)
    if use_diff_wrapper:
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


def _replace_env_state(state, new_env_state):
    if hasattr(state, "replace"):
        try:
            return state.replace(env_state=new_env_state)
        except TypeError:
            pass
    if hasattr(state, "set_env_state"):
        return state.set_env_state(new_env_state)
    raise TypeError(f"Cannot replace env_state for type {type(state)}")


def force_diff_time_step(state, forced_step: int):
    """Recursively sets diff_time_step on nested wrapper state."""
    if hasattr(state, "diff_time_step"):
        forced = jnp.full_like(state.diff_time_step, forced_step)
        return state.replace(diff_time_step=forced)
    if not hasattr(state, "env_state"):
        raise AttributeError(
            "Could not find diff_time_step in nested state. "
            "This mode requires use_diff_wrapper=True."
        )
    inner = force_diff_time_step(state.env_state, forced_step)
    return _replace_env_state(state, inner)


def summarize_ms(samples_sec: np.ndarray) -> dict[str, float]:
    samples_ms = samples_sec * 1000.0
    return {
        "mean_ms": float(np.mean(samples_ms)),
        "std_ms": float(np.std(samples_ms)),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
    }


def benchmark_setting(
    setting: BenchmarkSetting,
    env_name: str,
    episode_length: int,
    num_envs: int,
    diff_steps: int,
    seed: int,
    warmup_steps: int,
    timed_steps: int,
    repeats: int,
) -> dict[str, float | int | str | bool]:
    env = build_env(
        env_name=env_name,
        episode_length=episode_length,
        num_envs=num_envs,
        use_diff_wrapper=setting.use_diff_wrapper,
        diff_steps=diff_steps,
    )

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]
    action_shape = (num_envs, action_dim)

    def step_once(prev_state, env_subkeys, action):
        return env.step(env_subkeys, prev_state, action)

    step_once_jit = jax.jit(step_once)

    all_samples_sec = np.zeros((repeats, timed_steps), dtype=np.float64)
    repeat_mean_ms = np.zeros((repeats,), dtype=np.float64)
    compile_first_ms = np.zeros((repeats,), dtype=np.float64)

    for rep in range(repeats):
        rep_seed = seed + rep
        rng = jax.random.PRNGKey(rep_seed)
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        _, _, state = env.reset(reset_keys)

        rng, action_key, env_key = jax.random.split(rng, 3)
        action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
        env_subkeys = jax.random.split(env_key, num_envs)
        state_for_call = state
        if setting.force_diff_step is not None:
            state_for_call = force_diff_time_step(state_for_call, setting.force_diff_step)

        t0 = time.perf_counter()
        out = step_once_jit(state_for_call, env_subkeys, action)
        jax.block_until_ready(out[3])
        compile_first_ms[rep] = (time.perf_counter() - t0) * 1000.0
        state = out[2]

        for _ in range(warmup_steps):
            rng, action_key, env_key = jax.random.split(rng, 3)
            action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
            env_subkeys = jax.random.split(env_key, num_envs)
            state_for_call = state
            if setting.force_diff_step is not None:
                state_for_call = force_diff_time_step(
                    state_for_call, setting.force_diff_step
                )
            out = step_once_jit(state_for_call, env_subkeys, action)
            jax.block_until_ready(out[3])
            state = out[2]

        samples_sec = np.zeros((timed_steps,), dtype=np.float64)
        for i in range(timed_steps):
            rng, action_key, env_key = jax.random.split(rng, 3)
            action = jax.random.uniform(action_key, action_shape, minval=-1.0, maxval=1.0)
            env_subkeys = jax.random.split(env_key, num_envs)
            state_for_call = state
            if setting.force_diff_step is not None:
                state_for_call = force_diff_time_step(
                    state_for_call, setting.force_diff_step
                )

            t1 = time.perf_counter()
            out = step_once_jit(state_for_call, env_subkeys, action)
            jax.block_until_ready(out[3])
            samples_sec[i] = time.perf_counter() - t1
            state = out[2]

        all_samples_sec[rep] = samples_sec
        repeat_mean_ms[rep] = float(np.mean(samples_sec) * 1000.0)

    flat = all_samples_sec.reshape(-1)
    stats = summarize_ms(flat)
    stats.update(
        {
            "env_name": env_name,
            "setting": setting.name,
            "use_diff_wrapper": setting.use_diff_wrapper,
            "forced_diff_step": -1
            if setting.force_diff_step is None
            else int(setting.force_diff_step),
            "diff_steps": int(diff_steps) if setting.use_diff_wrapper else -1,
            "compile_first_mean_ms": float(np.mean(compile_first_ms)),
            "compile_first_std_ms": float(np.std(compile_first_ms)),
            "repeat_mean_ms": float(np.mean(repeat_mean_ms)),
            "repeat_std_ms": float(np.std(repeat_mean_ms)),
            "timed_steps_per_repeat": int(timed_steps),
            "repeats": int(repeats),
            "num_envs": int(num_envs),
            "env_steps_per_sec": float(num_envs / np.mean(flat)),
        }
    )
    return stats


def print_results(rows: list[dict[str, float | int | str | bool]]) -> None:
    header = (
        f"{'env':<24} {'setting':<18} {'force':>6} {'mean(ms)':>10} {'std(ms)':>10} "
        f"{'p50':>8} {'p95':>8} {'min':>8} {'max':>8} {'repeat_std':>11} {'env_steps/s':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['env_name']):<24} "
            f"{str(r['setting']):<18} "
            f"{str(r['forced_diff_step']):6} "
            f"{r['mean_ms']:10.3f} "
            f"{r['std_ms']:10.3f} "
            f"{r['p50_ms']:8.3f} "
            f"{r['p95_ms']:8.3f} "
            f"{r['min_ms']:8.3f} "
            f"{r['max_ms']:8.3f} "
            f"{r['repeat_std_ms']:11.3f} "
            f"{r['env_steps_per_sec']:12.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MJX env.step latency for fixed diffusion-step modes."
    )
    parser.add_argument(
        "--env-names",
        nargs="+",
        default=list(DEFAULT_ENV_NAMES),
        help="Environment names to benchmark.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help="Episode length passed to MjxGymnaxWrapper.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized envs.")
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    parser.add_argument(
        "--diff-steps",
        type=int,
        default=DEFAULT_DIFF_STEPS,
        help="num_diff_steps used when diff wrapper is enabled.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Warmup calls (not timed) after initial compile call.",
    )
    parser.add_argument(
        "--timed-steps",
        type=int,
        default=300,
        help="Measured step calls per repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated runs per setting and env.",
    )
    return parser.parse_args()


def main() -> None:
    # Example:
    # python scripts/benchmark_mjx_step_diff_modes.py --env-names PlanarPathEnv G1JoystickFlatTerrain --timed-steps 200 --repeats 5
    args = parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")
    if args.diff_steps <= 0:
        raise ValueError("--diff-steps must be > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.timed_steps <= 0:
        raise ValueError("--timed-steps must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    settings = [
        BenchmarkSetting(
            name="diff_wrapper_last_step",
            use_diff_wrapper=True,
            force_diff_step=args.diff_steps - 1,
        ),
        BenchmarkSetting(
            name="diff_wrapper_step0",
            use_diff_wrapper=True,
            force_diff_step=0,
        ),
        BenchmarkSetting(
            name="no_diff_wrapper",
            use_diff_wrapper=False,
            force_diff_step=None,
        ),
    ]

    print("Benchmarking env.step(...) latency with fixed diffusion-step modes")
    print(
        "Config: "
        f"envs={args.env_names}, episode_length={args.episode_length}, num_envs={args.num_envs}, "
        f"diff_steps={args.diff_steps}, warmup_steps={args.warmup_steps}, "
        f"timed_steps={args.timed_steps}, repeats={args.repeats}"
    )

    if args.diff_steps == 1:
        print(
            "Warning: diff_steps=1 makes forced step=0 and forced step=last equivalent."
        )

    rows: list[dict[str, float | int | str | bool]] = []
    for env_name in args.env_names:
        for setting in settings:
            print(
                f"Running env={env_name}, setting={setting.name} "
                f"(use_diff_wrapper={setting.use_diff_wrapper}, forced_diff_step={setting.force_diff_step})"
            )
            try:
                row = benchmark_setting(
                    setting=setting,
                    env_name=env_name,
                    episode_length=args.episode_length,
                    num_envs=args.num_envs,
                    diff_steps=args.diff_steps,
                    seed=args.seed,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    repeats=args.repeats,
                )
            except Exception as exc:  # pragma: no cover - runtime env failures
                print(f"  Failed: {type(exc).__name__}: {exc}")
                continue
            rows.append(row)

    if not rows:
        print("\nNo successful benchmark runs.")
        return

    print_results(rows)


if __name__ == "__main__":
    main()
