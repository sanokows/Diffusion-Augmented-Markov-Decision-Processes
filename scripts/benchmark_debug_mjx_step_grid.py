"""Benchmark the env.step(...) call used in scripts/debug_mjx_state.py line 133.

This script times the exact call pattern:
    obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(env_subkeys, prev_state, action)

Example:
  python scripts/benchmark_debug_mjx_step_grid.py \
    --env-name G1JoystickFlatTerrain \
    --num-envs-list 1,64,256,1024 \
    --diff-steps-list 1,2,4,8 \
    --warmup-iters 20 \
    --measure-iters 200 \
    --csv-out READMEs/Timing/outputs/mjx_step_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Any

import jax
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


DEFAULT_ENV_NAME = "G1JoystickFlatTerrain"
DEFAULT_EPISODE_LENGTH = 200


def create_diffusion_config(diff_steps: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.diff_steps = diff_steps
    cfg.init_std = 3.0
    return cfg


def parse_int_csv(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _block_until_ready(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def build_env(
    *,
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
        env = DiffNormalizeVec(env)
    else:
        env = base_env
        env = LogWrapper(env, num_envs=num_envs)
        env = NormalizeVec(env)

    return env


def benchmark_one_setting(
    *,
    env_name: str,
    episode_length: int,
    seed: int,
    num_envs: int,
    use_diff_wrapper: bool,
    diff_steps: int,
    warmup_iters: int,
    measure_iters: int,
    action_min: float,
    action_max: float,
) -> dict[str, Any]:
    env = build_env(
        env_name=env_name,
        episode_length=episode_length,
        num_envs=num_envs,
        use_diff_wrapper=use_diff_wrapper,
        diff_steps=diff_steps,
    )

    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    _obs_dict, _critic_obs_dict, state = env.reset(reset_keys)

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]
    action_shape = (num_envs, action_dim)

    def step_once(rng_in, state_in, action_in):
        rng_out, env_key = jax.random.split(rng_in)
        env_subkeys = jax.random.split(env_key, num_envs)
        obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(
            env_subkeys, state_in, action_in
        )
        return rng_out, next_state, reward, done, obs_dict, critic_obs_dict, info

    step_once_jit = jax.jit(step_once)

    def sample_action(rng_key):
        return jax.random.uniform(
            rng_key,
            action_shape,
            minval=action_min,
            maxval=action_max,
        )

    rng, action_key = jax.random.split(rng)
    action = sample_action(action_key)
    t0 = time.perf_counter()
    out = step_once_jit(rng, state, action)
    _block_until_ready(out)
    compile_and_first_step_s = time.perf_counter() - t0
    rng, state = out[0], out[1]

    for _ in range(warmup_iters):
        rng, action_key = jax.random.split(rng)
        action = sample_action(action_key)
        out = step_once_jit(rng, state, action)
        _block_until_ready(out)
        rng, state = out[0], out[1]

    times_s = []
    for _ in range(measure_iters):
        rng, action_key = jax.random.split(rng)
        action = sample_action(action_key)

        t_start = time.perf_counter()
        out = step_once_jit(rng, state, action)
        _block_until_ready(out)
        t_end = time.perf_counter()

        times_s.append(t_end - t_start)
        rng, state = out[0], out[1]

    times_ms = np.asarray(times_s, dtype=np.float64) * 1000.0

    return {
        "env_name": env_name,
        "num_envs": num_envs,
        "use_diff_wrapper": use_diff_wrapper,
        "diff_steps": diff_steps if use_diff_wrapper else 0,
        "action_dim": int(action_dim),
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "compile_and_first_step_s": float(compile_and_first_step_s),
        "mean_step_ms": float(times_ms.mean()),
        "median_step_ms": float(np.median(times_ms)),
        "p95_step_ms": float(np.percentile(times_ms, 95.0)),
        "std_step_ms": float(times_ms.std()),
        "min_step_ms": float(times_ms.min()),
        "max_step_ms": float(times_ms.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark env.step(...) timing for a grid of wrapper settings."
    )
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--num-envs-list",
        default="1,64,256,1024",
        help="Comma-separated list of num_envs values.",
    )
    parser.add_argument(
        "--diff-steps-list",
        default="1,2,4,8",
        help="Comma-separated diff_steps used when diffusion wrapper is enabled.",
    )

    parser.add_argument(
        "--include-no-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include baseline wrapper stack without MjxDiffEnvWrapper.",
    )
    parser.add_argument(
        "--include-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include MjxDiffEnvWrapper runs for each diff_steps value.",
    )

    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--action-min", type=float, default=0.0)
    parser.add_argument("--action-max", type=float, default=1.0)

    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV output path for results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.include_no_diff and not args.include_diff:
        raise ValueError("At least one of --include-no-diff or --include-diff must be enabled.")

    num_envs_list = parse_int_csv(args.num_envs_list)
    diff_steps_list = parse_int_csv(args.diff_steps_list)

    settings: list[tuple[int, bool, int]] = []
    for num_envs in num_envs_list:
        if args.include_no_diff:
            settings.append((num_envs, False, 1))
        if args.include_diff:
            for diff_steps in diff_steps_list:
                settings.append((num_envs, True, diff_steps))

    rows: list[dict[str, Any]] = []
    total = len(settings)
    for idx, (num_envs, use_diff_wrapper, diff_steps) in enumerate(settings, start=1):
        print(
            f"[{idx}/{total}] num_envs={num_envs}, "
            f"use_diff_wrapper={use_diff_wrapper}, diff_steps={diff_steps if use_diff_wrapper else 0}"
        )
        row = benchmark_one_setting(
            env_name=args.env_name,
            episode_length=args.episode_length,
            seed=args.seed,
            num_envs=num_envs,
            use_diff_wrapper=use_diff_wrapper,
            diff_steps=diff_steps,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            action_min=args.action_min,
            action_max=args.action_max,
        )
        rows.append(row)

        print(
            "  "
            f"mean={row['mean_step_ms']:.3f} ms | "
            f"median={row['median_step_ms']:.3f} ms | "
            f"p95={row['p95_step_ms']:.3f} ms | "
            f"compile+first={row['compile_and_first_step_s']:.3f} s"
        )

    print("\n=== Summary (sorted by mean_step_ms) ===")
    rows_sorted = sorted(rows, key=lambda r: r["mean_step_ms"])
    for row in rows_sorted:
        print(
            "  "
            f"num_envs={row['num_envs']:>5} | "
            f"diff={int(row['use_diff_wrapper'])} | "
            f"diff_steps={row['diff_steps']:>2} | "
            f"mean={row['mean_step_ms']:.3f} ms | "
            f"p95={row['p95_step_ms']:.3f} ms"
        )

    if args.csv_out:
        out_path = pathlib.Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
