"""Utility script to render MJX Cheetah and upload a video to wandb."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import os
import pathlib
import sys

# Enable headless Mujoco rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
import wandb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from src.env_utils.jax_wrappers import MjxGymnaxWrapper
from src.jaxrl.camera_configs import get_render_quality_config


def _extract_single_env_state(state, env_index: int = 0):
    """Converts a batched MJX state into a single-environment state."""
    reward_shape = tuple(state.reward.shape)
    batch_size = reward_shape[0] if reward_shape else 1

    def maybe_slice(x):
        arr = np.asarray(x)
        if arr.ndim > 0 and arr.shape[0] == batch_size:
            return arr[env_index]
        return arr

    return jax.tree_util.tree_map(maybe_slice, state)


def collect_random_trajectory(
    env: MjxGymnaxWrapper,
    seed: int,
    num_video_steps: int,
    *,
    manual_reset: bool = True,
    debug: bool = False,
) -> tuple[Sequence, float] | tuple[Sequence, float, dict[str, Any]]:
    """Rolls out a random policy and records all intermediate states."""
    rng = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(rng, 1)
    _, _, init_state = env.reset(reset_keys)

    if num_video_steps <= 0:
        return [_extract_single_env_state(init_state)], 0.0

    action_shape = (1, env.env.action_size)

    def step_fn(carry, step_idx):
        key, state = carry
        jax.debug.print("Step: {}", step_idx)
        key, action_key, env_key, reset_key = jax.random.split(key, 4)
        action = jax.random.uniform(
            action_key,
            action_shape,
            minval=-1.0,
            maxval=1.0,
        )

        _, _, next_state, reward, done_flag, _ = env.step(env_key, state, action)
        reward_scalar = jnp.reshape(reward, (-1,))[0]
        step_done = jnp.reshape(done_flag, (-1,))[0]

        def reset_env(k):
            reset_keys = jnp.expand_dims(k, axis=0)
            _, _, reset_state = env.reset(reset_keys)
            return reset_state

        if manual_reset:
            continued_state = jax.lax.cond(
                step_done,
                reset_env,
                lambda _: next_state,
                operand=reset_key,
            )
        else:
            continued_state = next_state
        return (key, continued_state), (next_state, reward_scalar, step_done)

    init_carry = (rng, init_state)
    (_, _), (state_stack, reward_stack, done_stack) = jax.lax.scan(
        step_fn, init_carry, xs=None, length=num_video_steps
    )

    state_stack = jax.device_get(state_stack)
    reward_stack = np.asarray(jax.device_get(reward_stack))
    done_stack = np.asarray(jax.device_get(done_stack))
    init_state_np = jax.device_get(init_state)

    init_state_expanded = jax.tree_map(lambda x: np.expand_dims(x, axis=0), init_state_np)
    all_states = jax.tree_map(
        lambda first, rest: np.concatenate([first, rest], axis=0),
        init_state_expanded,
        state_stack,
    )

    sample_leaf = jax.tree_leaves(state_stack)[0]
    total_states = sample_leaf.shape[0] + 1
    trajectory = []
    for idx in range(total_states):
        state_i = jax.tree_map(lambda x: x[idx], all_states)
        trajectory.append(_extract_single_env_state(state_i))

    total_reward = float(reward_stack.sum())

    if debug:
        debug_data: dict[str, Any] = {
            "done_stack": done_stack,
            "state_stack": state_stack,
        }
        return trajectory, total_reward, debug_data
    return trajectory, total_reward


def verify_terminal_state_behavior(
    done_stack: np.ndarray, state_stack: Any, manual_reset: bool
) -> None:
    """Checks if MJX keeps returning the terminal state once done is hit."""
    if manual_reset:
        print(
            "Manual reset is enabled; rerun with --disable-manual-reset to inspect raw env behavior."
        )
        return

    done_indices = np.where(done_stack)[0]
    if done_indices.size == 0:
        print("No terminal state encountered within the sampled horizon.")
        return

    first_done = int(done_indices[0])
    terminal_repeat = True
    for leaf in jax.tree_util.tree_leaves(state_stack):
        tail = leaf[first_done:]
        if tail.shape[0] <= 1:
            continue
        reference = np.broadcast_to(tail[0], tail.shape)
        if not np.allclose(tail, reference):
            terminal_repeat = False
            break

    if terminal_repeat:
        print(
            f"MJX reached done at step {first_done} and kept returning the terminal state afterwards."
        )
    else:
        print(
            "MJX returned varying states after reaching done; manual resets are not strictly necessary."
        )


def log_video_to_wandb(
    env: MjxGymnaxWrapper,
    env_name: str,
    trajectory: Sequence,
    project: str,
    entity: str,
    mode: str,
    run_name: str,
    fps: int,
    quality: str,
    episode_return: float,
) -> None:
    """Creates a wandb run and uploads the rendered rollout."""
    render_cfg = get_render_quality_config(env_name, quality)
    frames = env.env.render(
        trajectory,
        height=render_cfg["height"],
        width=render_cfg["width"],
        camera=render_cfg["camera"],
    )
    frames = np.asarray(frames, dtype=np.uint8)
    video_tensor = frames.transpose(0, 3, 1, 2)

    run = wandb.init(
        mode=mode,
        project=project,
        entity=entity or None,
        name=run_name or f"{env_name}-video",
        config={"env_name": env_name, "fps": fps, "quality": quality},
    )
    try:
        video_payload = wandb.Video(video_tensor, fps=fps, format="mp4")
    except wandb.errors.Error as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            "wandb.Video requires moviepy when logging raw frames. "
            "Install wandb[media] or add moviepy to your environment."
        ) from exc
    run.log(
        {
            "render/video": video_payload,
            "episode/return": episode_return,
            "episode/length": len(frames),
        }
    )
    run.finish()


def main() -> None:
    #python scripts/log_cheetah_wandb.py --mode online --fps 30
    parser = argparse.ArgumentParser(
        description="Render a MJX Cheetah rollout and log it to wandb."
    )
    parser.add_argument("--env-name", default="CheetahRun", help="MJX env to render.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Environment episode length before a forced reset.",
    )
    parser.add_argument(
        "--num_video_steps",
        type=int,
        default=1000,
        help="Number of steps to render/log (defaults to --num_steps).",
    )
    parser.add_argument("--project", default="cheetah_video", help="wandb project name.")
    parser.add_argument("--entity", default="sanokows", help="wandb entity (optional).")
    parser.add_argument("--mode", default="online", help="wandb mode.")
    parser.add_argument("--run-name", default="", help="Optional wandb run name.")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS used for the uploaded video.",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "hd"],
        default="medium",
        help="Preset controlling render resolution.",
    )
    parser.add_argument(
        "--disable-manual-reset",
        action="store_true",
        help="Skip the explicit env reset when done is reached (useful for debugging).",
    )
    parser.add_argument(
        "--verify-terminal-state",
        action="store_true",
        help="Prints a diagnostic message verifying MJX's terminal-state behavior.",
    )
    args = parser.parse_args()

    env = MjxGymnaxWrapper(args.env_name, episode_length=args.num_steps)
    num_video_steps = args.num_video_steps if args.num_video_steps is not None else args.num_steps
    manual_reset = not args.disable_manual_reset
    rollout = collect_random_trajectory(
        env,
        args.seed,
        num_video_steps,
        manual_reset=manual_reset,
        debug=args.verify_terminal_state,
    )
    if args.verify_terminal_state:
        trajectory, episode_return, debug_data = rollout
        verify_terminal_state_behavior(
            debug_data["done_stack"], debug_data["state_stack"], manual_reset
        )
    else:
        trajectory, episode_return = rollout

    log_video_to_wandb(
        env=env,
        env_name=args.env_name,
        trajectory=trajectory,
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        run_name=args.run_name,
        fps=args.fps,
        quality=args.quality,
        episode_return=episode_return,
    )


if __name__ == "__main__":
    main()
