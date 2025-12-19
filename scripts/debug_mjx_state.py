"""Simple helper to inspect MJX observations for many consecutive steps."""

from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from src.env_utils.jax_wrappers import MjxDiffEnvWrapper, MjxGymnaxWrapper, DiffNormalizeVec, LogWrapper, NormalizeVec


DEFAULT_ENV_NAME = "CheetahRun"
DEFAULT_EPISODE_LENGTH = 6
DEFAULT_TOTAL_STEPS = DEFAULT_EPISODE_LENGTH*3


def state_obs_array(state):
    """Returns the observation array stored inside an MJX env state."""
    if hasattr(state, "obs"):
        return state.obs
    if hasattr(state, "env_state") and hasattr(state.env_state, "obs"):
        return state.env_state.obs
    raise AttributeError("State object does not expose an observation field.")


def create_diffusion_config(diff_steps: int) -> ConfigDict:
    cfg = ConfigDict()
    cfg.diff_steps = diff_steps
    cfg.init_std = 3.0
    return cfg


def main() -> None:
    # python scripts/debug_mjx_state.py --use-diff-wrapper --diff-steps 3 
    # python scripts/debug_mjx_state.py --diff-steps 3 
    parser = argparse.ArgumentParser(description="Inspect MJX states over many steps.")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="MJX env to load.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=DEFAULT_EPISODE_LENGTH,
        help="Episode horizon passed to the wrapper.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Total number of transitions to sample.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--use-diff-wrapper",
        action="store_true",
        help="Wrap the env in MjxDiffEnvWrapper before rolling out.",
    )
    parser.add_argument(
        "--diff-steps",
        type=int,
        default=5,
        help="Number of diffusion steps when --use-diff-wrapper is set.",
    )
    args = parser.parse_args()

    base_env = MjxGymnaxWrapper(args.env_name, episode_length=args.episode_length)
    total_steps = args.total_steps
    if args.use_diff_wrapper:
        diff_cfg = create_diffusion_config(args.diff_steps)
        env = MjxDiffEnvWrapper(
            base_env,
            num_diff_steps=args.diff_steps,
            diffusion_config=diff_cfg,
        )
        total_steps *= args.diff_steps
        env = LogWrapper(env, num_envs = 1)  # add logging to the diff env
        env = DiffNormalizeVec(env)
    else:
        env = base_env
        env = LogWrapper(env, num_envs = 1)
        env  = NormalizeVec(env)

    
    rng = jax.random.PRNGKey(args.seed)
    reset_keys = jax.random.split(rng, 1)
    obs_dict, _, state = env.reset(reset_keys)
    step_idx = -1
    jax.debug.print("Step {} done={}", step_idx, obs_dict) 
    num_envs = reset_keys.shape[0]

    action_space_params = getattr(env, "default_params", None)
    action_dim = env.action_space(action_space_params).shape[0]
    action_shape = (num_envs, action_dim)

    def rollout_with_scan_local(rng_in, init_state_in, total_steps, action_shape_in):
        """Runs env.step inside jax.lax.scan for faster rollouts."""

        def step_fn(carry, step_idx):
            key, prev_state = carry
            key, action_key, env_key = jax.random.split(key, 3)
            action = jax.random.uniform(
                action_key, action_shape_in, minval=-1.0, maxval=1.0
            )
            action = jnp.zeros_like(action)  # use zero actions for debugging
            env_subkeys = jax.random.split(env_key, num_envs)
            obs_dict, critic_obs_dict, next_state, reward, done, info = env.step(env_subkeys, prev_state, action)

            done_flag = jnp.reshape(done, (-1,))[0]
            # print the reward and obs dict
            jax.debug.print("Step {} truncated={}", step_idx, state.truncated)
            #jax.debug.print("Step {} state done={}", step_idx, state.done)
            jax.debug.print("Step {} reward={}", step_idx, reward)
            jax.debug.print("Step {} action={}", step_idx, action) 
            jax.debug.print("scan step {} done={}", step_idx, done_flag)
            jax.debug.print("Step {} next_obs={}", step_idx, obs_dict) 
            jax.debug.print("Step {} info={}", step_idx, info) 
 
            return (key, next_state), None

        (_, _), outputs = jax.lax.scan(
            step_fn, (rng_in, init_state_in), xs=jnp.arange(total_steps)
        )
        return outputs

    scan_fn = jax.jit(rollout_with_scan_local, static_argnums=(2, 3))
    obs_stack = scan_fn(rng, state, total_steps, action_shape)

    # obs_stack = jax.device_get(obs_stack)

    # for step in range(args.total_steps):
    #     obs_np = np.asarray(obs_stack[step])
    #     print(f"\nStep {step + 1}/{args.total_steps}")
    #     print("Observation:")
    #     print(obs_np)


if __name__ == "__main__":
    main()
