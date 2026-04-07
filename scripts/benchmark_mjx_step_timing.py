"""Benchmark the runtime of `env.step(env_subkeys, prev_state, action)` in MJX wrappers.

Example:
python scripts/benchmark_mjx_step_timing.py \
  --env-name G1JoystickFlatTerrain \
  --num-envs-list 1,64 \
  --diff-steps-list 3,5,8 \
  --rollout-steps 512 \
  --repeats 10
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

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


DEFAULT_ENV_NAME = "G1JoystickFlatTerrain"
DEFAULT_EPISODE_LENGTH = 200


@dataclass(frozen=True)
class BenchmarkSetting:
    use_diff_wrapper: bool
    diff_steps: int | None
    num_envs: int

    @property
    def label(self) -> str:
        if self.use_diff_wrapper:
            return f"diff_wrapper(diff_steps={self.diff_steps}), num_envs={self.num_envs}"
        return f"no_diff_wrapper, num_envs={self.num_envs}"


def create_diffusion_config(diff_steps: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.diff_steps = diff_steps
    cfg.init_std = 3.0
    return cfg


def parse_int_csv(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        return []
    return [int(v) for v in values]


def block_until_ready_tree(tree: Any) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        leaves[0].block_until_ready()


def build_env(
    env_name: str,
    episode_length: int,
    num_envs: int,
    use_diff_wrapper: bool,
    diff_steps: int | None,
):
    base_env = MjxGymnaxWrapper(env_name, episode_length=episode_length)
    if use_diff_wrapper:
        assert diff_steps is not None
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


def build_rollout_tensors(
    rng: jax.Array,
    rollout_steps: int,
    num_envs: int,
    action_dim: int,
) -> tuple[jax.Array, jax.Array]:
    key_for_env, key_for_action = jax.random.split(rng)
    root_keys = jax.random.split(key_for_env, rollout_steps)
    env_subkeys = jax.vmap(lambda k: jax.random.split(k, num_envs))(root_keys)
    actions = jax.random.uniform(
        key_for_action,
        shape=(rollout_steps, num_envs, action_dim),
        minval=-1.0,
        maxval=1.0,
    )
    return env_subkeys, actions


def make_rollout_fn(env):
    def rollout(init_state, all_env_subkeys, all_actions):
        def step_fn(prev_state, inputs):
            env_subkeys, action = inputs
            _, _, next_state, _, _, _ = env.step(env_subkeys, prev_state, action)
            return next_state, ()

        final_state, _ = jax.lax.scan(step_fn, init_state, xs=(all_env_subkeys, all_actions))
        return final_state

    return jax.jit(rollout)


def benchmark_setting(
    setting: BenchmarkSetting,
    env_name: str,
    episode_length: int,
    rollout_steps: int,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    env = build_env(
        env_name=env_name,
        episode_length=episode_length,
        num_envs=setting.num_envs,
        use_diff_wrapper=setting.use_diff_wrapper,
        diff_steps=setting.diff_steps,
    )

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, setting.num_envs)
    _, _, init_state = env.reset(reset_keys)

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]
    all_env_subkeys, all_actions = build_rollout_tensors(
        rollout_rng, rollout_steps, setting.num_envs, action_dim
    )

    rollout_fn = make_rollout_fn(env)

    warmup_out = rollout_fn(init_state, all_env_subkeys, all_actions)
    block_until_ready_tree(warmup_out)

    samples_sec: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_state = rollout_fn(init_state, all_env_subkeys, all_actions)
        block_until_ready_tree(out_state)
        samples_sec.append(time.perf_counter() - t0)

    times = np.asarray(samples_sec, dtype=np.float64)
    mean_total_ms = float(times.mean() * 1e3)
    median_total_ms = float(np.median(times) * 1e3)
    std_total_ms = float(times.std(ddof=0) * 1e3)
    mean_step_us = float(times.mean() / rollout_steps * 1e6)

    return {
        "setting": setting.label,
        "use_diff_wrapper": int(setting.use_diff_wrapper),
        "diff_steps": -1 if setting.diff_steps is None else int(setting.diff_steps),
        "num_envs": setting.num_envs,
        "rollout_steps": rollout_steps,
        "repeats": repeats,
        "mean_total_ms": mean_total_ms,
        "median_total_ms": median_total_ms,
        "std_total_ms": std_total_ms,
        "mean_step_us": mean_step_us,
    }


def build_settings(
    num_envs_list: list[int],
    include_no_diff: bool,
    diff_steps_list: list[int],
) -> list[BenchmarkSetting]:
    settings: list[BenchmarkSetting] = []
    for num_envs in num_envs_list:
        if include_no_diff:
            settings.append(
                BenchmarkSetting(
                    use_diff_wrapper=False,
                    diff_steps=None,
                    num_envs=num_envs,
                )
            )
        for diff_steps in diff_steps_list:
            settings.append(
                BenchmarkSetting(
                    use_diff_wrapper=True,
                    diff_steps=diff_steps,
                    num_envs=num_envs,
                )
            )
    return settings


def write_csv(path: pathlib.Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the runtime of the MJX env.step call across wrapper settings."
    )
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="MJX env to load.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help="Episode length passed to MjxGymnaxWrapper.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="1",
        help="Comma-separated list, e.g. '1,64,256'.",
    )
    parser.add_argument(
        "--diff-steps-list",
        type=str,
        default="3,5,8",
        help="Comma-separated diffusion steps for diff-wrapper settings.",
    )
    parser.add_argument(
        "--skip-no-diff",
        action="store_true",
        help="Skip the non-diff baseline setting.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=512,
        help="Number of env.step calls per timed rollout.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed repeats per setting (after 1 warmup run).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional path to save results as CSV.",
    )
    args = parser.parse_args()

    num_envs_list = parse_int_csv(args.num_envs_list)
    diff_steps_list = parse_int_csv(args.diff_steps_list)
    include_no_diff = not args.skip_no_diff

    if not num_envs_list:
        raise ValueError("--num-envs-list must include at least one positive integer.")
    if any(n <= 0 for n in num_envs_list):
        raise ValueError("--num-envs-list values must be > 0.")
    if any(d <= 0 for d in diff_steps_list):
        raise ValueError("--diff-steps-list values must be > 0.")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be > 0.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0.")
    if not include_no_diff and not diff_steps_list:
        raise ValueError("No settings selected: provide diff steps or include no-diff.")

    settings = build_settings(num_envs_list, include_no_diff, diff_steps_list)
    rows: list[dict[str, float | int | str]] = []

    print(f"Benchmarking {len(settings)} setting(s)")
    print(
        f"env={args.env_name}, episode_length={args.episode_length}, "
        f"rollout_steps={args.rollout_steps}, repeats={args.repeats}"
    )

    for idx, setting in enumerate(settings, start=1):
        print(f"[{idx}/{len(settings)}] {setting.label}")
        row = benchmark_setting(
            setting=setting,
            env_name=args.env_name,
            episode_length=args.episode_length,
            rollout_steps=args.rollout_steps,
            repeats=args.repeats,
            seed=args.seed,
        )
        rows.append(row)
        print(
            "  mean_total_ms={mean_total_ms:.3f}, median_total_ms={median_total_ms:.3f}, "
            "std_total_ms={std_total_ms:.3f}, mean_step_us={mean_step_us:.3f}".format(
                **row
            )
        )

    print("\nSummary")
    for row in rows:
        print(
            "{setting}: mean_step_us={mean_step_us:.3f}, "
            "mean_total_ms={mean_total_ms:.3f}".format(**row)
        )

    if args.csv_out:
        out_path = pathlib.Path(args.csv_out)
        write_csv(out_path, rows)
        print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
