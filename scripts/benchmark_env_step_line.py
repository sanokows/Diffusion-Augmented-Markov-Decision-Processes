"""Benchmark the runtime of:
obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(...)

This script focuses on timing the `env.step(env_subkeys, prev_state, action)` line
inside a jitted `jax.lax.scan`, across a configurable set of settings.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
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
DEFAULT_STEPS_PER_RUN = 256
DEFAULT_WARMUP_RUNS = 1
DEFAULT_TIMED_RUNS = 5


@dataclass(frozen=True)
class BenchmarkSetting:
    name: str
    use_diff_wrapper: bool
    diff_steps: int
    num_envs: int


DEFAULT_SETTINGS = [
    BenchmarkSetting(name="base_n1", use_diff_wrapper=False, diff_steps=1, num_envs=1),
    BenchmarkSetting(name="base_n16", use_diff_wrapper=False, diff_steps=1, num_envs=16),
    BenchmarkSetting(name="diff3_n1", use_diff_wrapper=True, diff_steps=3, num_envs=1),
    BenchmarkSetting(name="diff5_n1", use_diff_wrapper=True, diff_steps=5, num_envs=1),
]


def create_diffusion_config(diff_steps: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.diff_steps = diff_steps
    cfg.init_std = 3.0
    return cfg


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}.")


def parse_settings(settings_json: str | None) -> list[BenchmarkSetting]:
    if settings_json is None:
        return list(DEFAULT_SETTINGS)

    raw = json.loads(settings_json)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("--settings-json must decode to an object or list of objects.")

    settings: list[BenchmarkSetting] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Setting at index {idx} must be an object.")
        name = str(item.get("name", f"setting_{idx}"))
        use_diff_wrapper = _as_bool(item.get("use_diff_wrapper", False))
        diff_steps = int(item.get("diff_steps", 1))
        num_envs = int(item.get("num_envs", 1))
        if diff_steps < 1:
            raise ValueError(f"{name}: diff_steps must be >= 1.")
        if num_envs < 1:
            raise ValueError(f"{name}: num_envs must be >= 1.")
        settings.append(
            BenchmarkSetting(
                name=name,
                use_diff_wrapper=use_diff_wrapper,
                diff_steps=diff_steps,
                num_envs=num_envs,
            )
        )
    return settings


def build_env(setting: BenchmarkSetting, env_name: str, episode_length: int):
    base_env = MjxGymnaxWrapper(env_name, episode_length=episode_length)
    if setting.use_diff_wrapper:
        diff_cfg = create_diffusion_config(setting.diff_steps)
        env = MjxDiffEnvWrapper(
            base_env,
            num_diff_steps=setting.diff_steps,
            diffusion_config=diff_cfg,
        )
        env = LogWrapper(env, num_envs=setting.num_envs)
        env = DiffNormalizeVec(env, num_diff_steps=setting.diff_steps)
        return env

    env = LogWrapper(base_env, num_envs=setting.num_envs)
    env = NormalizeVec(env)
    return env


def _obs_probe(obs: Any) -> jax.Array:
    """Small dependency on obs leaves so XLA does not dead-code-eliminate them."""
    if isinstance(obs, dict):
        probe = jnp.array(0.0, dtype=jnp.float32)
        for value in obs.values():
            flat = jnp.reshape(jnp.asarray(value), (-1,))
            probe = probe + flat[0].astype(jnp.float32)
        return probe

    flat = jnp.reshape(jnp.asarray(obs), (-1,))
    return flat[0].astype(jnp.float32)


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_benchmark_for_setting(
    *,
    setting: BenchmarkSetting,
    env_name: str,
    episode_length: int,
    seed: int,
    steps_per_run: int,
    warmup_runs: int,
    timed_runs: int,
    index: int,
) -> dict[str, Any]:
    env = build_env(setting, env_name=env_name, episode_length=episode_length)

    rng = jax.random.PRNGKey(seed)
    rng = jax.random.fold_in(rng, index)
    reset_keys = jax.random.split(rng, setting.num_envs)
    _, _, state = env.reset(reset_keys)

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]

    rng, env_key, action_key = jax.random.split(rng, 3)
    env_subkeys_seq = jax.random.split(env_key, steps_per_run * setting.num_envs).reshape(
        steps_per_run, setting.num_envs, 2
    )
    actions = jax.random.uniform(
        action_key,
        (steps_per_run, setting.num_envs, action_dim),
        minval=-1.0,
        maxval=1.0,
    )

    def rollout_line_timing(init_state, env_subkeys_steps, action_steps):
        def step_fn(carry, inputs):
            prev_state, probe_acc = carry
            env_subkeys, action = inputs

            # The exact line we want to benchmark.
            obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(
                env_subkeys, prev_state, action
            )

            # Keep dependencies on returned outputs so they are part of compiled work.
            reward_probe = jnp.reshape(jnp.asarray(reward), (-1,))[0].astype(jnp.float32)
            done_probe = jnp.reshape(jnp.asarray(done), (-1,))[0].astype(jnp.float32)
            info_probe = jnp.array(float(len(info)), dtype=jnp.float32)
            probe = (
                _obs_probe(obs_dict)
                + _obs_probe(critic_obs_dict)
                + reward_probe
                + done_probe
                + info_probe
            )
            return (next_state, probe_acc + probe), None

        (final_state, probe_acc), _ = jax.lax.scan(
            step_fn,
            (init_state, jnp.array(0.0, dtype=jnp.float32)),
            (env_subkeys_steps, action_steps),
        )
        return final_state, probe_acc

    rollout_jit = jax.jit(rollout_line_timing)

    t0 = time.perf_counter()
    state, probe = rollout_jit(state, env_subkeys_seq, actions)
    _block_until_ready((state, probe))
    compile_plus_first_run_s = time.perf_counter() - t0

    for _ in range(warmup_runs):
        state, probe = rollout_jit(state, env_subkeys_seq, actions)
        _block_until_ready((state, probe))

    durations_s: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        state, probe = rollout_jit(state, env_subkeys_seq, actions)
        _block_until_ready((state, probe))
        durations_s.append(time.perf_counter() - t0)

    mean_run_s = statistics.fmean(durations_s)
    std_run_s = statistics.pstdev(durations_s) if len(durations_s) > 1 else 0.0

    ms_per_step_call = 1e3 * mean_run_s / steps_per_run
    us_per_env_step = 1e6 * mean_run_s / (steps_per_run * setting.num_envs)

    return {
        "name": setting.name,
        "use_diff_wrapper": setting.use_diff_wrapper,
        "diff_steps": setting.diff_steps,
        "num_envs": setting.num_envs,
        "compile_plus_first_run_s": compile_plus_first_run_s,
        "mean_run_s": mean_run_s,
        "std_run_s": std_run_s,
        "ms_per_step_call": ms_per_step_call,
        "us_per_env_step": us_per_env_step,
    }


def print_results(results: list[dict[str, Any]], steps_per_run: int, timed_runs: int) -> None:
    print("\nBenchmark results")
    print(f"steps_per_run={steps_per_run}, timed_runs={timed_runs}")
    print(
        "Columns: setting, diff, diff_steps, num_envs, compile+first[s], "
        "mean_run[s], std[s], ms/step_call, us/env_step"
    )
    print("-" * 130)
    for r in results:
        print(
            f"{r['name']:<14} "
            f"{str(r['use_diff_wrapper']):<5} "
            f"{r['diff_steps']:<10d} "
            f"{r['num_envs']:<8d} "
            f"{r['compile_plus_first_run_s']:<16.6f} "
            f"{r['mean_run_s']:<11.6f} "
            f"{r['std_run_s']:<8.6f} "
            f"{r['ms_per_step_call']:<13.6f} "
            f"{r['us_per_env_step']:<12.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark env.step line inside jitted scan for multiple settings."
    )
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--steps-per-run", type=int, default=DEFAULT_STEPS_PER_RUN)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--timed-runs", type=int, default=DEFAULT_TIMED_RUNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settings-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object/list overriding settings. "
            'Example: \'[{"name":"base_n1","use_diff_wrapper":false,"diff_steps":1,"num_envs":1}]\''
        ),
    )
    args = parser.parse_args()

    settings = parse_settings(args.settings_json)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Device(s): {[str(d) for d in jax.devices()]}")
    print(f"Benchmarking {len(settings)} setting(s)...")

    results: list[dict[str, Any]] = []
    for idx, setting in enumerate(settings):
        print(
            f"  - Running {setting.name} "
            f"(diff={setting.use_diff_wrapper}, diff_steps={setting.diff_steps}, num_envs={setting.num_envs})"
        )
        result = run_benchmark_for_setting(
            setting=setting,
            env_name=args.env_name,
            episode_length=args.episode_length,
            seed=args.seed,
            steps_per_run=args.steps_per_run,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            index=idx,
        )
        results.append(result)

    print_results(results, steps_per_run=args.steps_per_run, timed_runs=args.timed_runs)


if __name__ == "__main__":
    main()
