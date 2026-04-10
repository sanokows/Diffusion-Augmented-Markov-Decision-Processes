"""Shared utilities for building rollout-aligned auxiliary supervision targets."""

from __future__ import annotations

import jax
from jax import numpy as jnp


def build_rollout_aux_targets(
    next_emb: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    truncated: jax.Array,
    diff_time_step: jax.Array,
    shift_steps: int,
    *,
    mask_next_state_on_episode_end: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build shifted embedding and final-step reward targets for aux losses.

    The function returns:
    1) `next_state_emb`: `next_emb` shifted by `shift_steps` along rollout time.
    2) `next_emb_mask`: mask for invalid shifted supervision near rollout tail; it can
       also exclude windows that cross done/truncated boundaries.
    3) `reward_target`: for each diffusion step, reward from the corresponding final
       diffusion step in the same rollout segment.
    4) `reward_target_mask`: mask for invalid/out-of-range or cross-episode reward
       targets.
    """
    shift_steps = jnp.asarray(shift_steps, dtype=jnp.int32)
    rollout_steps = next_emb.shape[0]
    num_envs = next_emb.shape[1]

    time_idx = jnp.arange(rollout_steps, dtype=jnp.int32)
    shifted_idx = jnp.minimum(time_idx + shift_steps, rollout_steps - 1)
    next_state_emb = jnp.take(next_emb, shifted_idx, axis=0)

    valid_shift = (time_idx + shift_steps) <= (rollout_steps - 1)
    next_emb_mask = jnp.broadcast_to(valid_shift[:, None], (rollout_steps, num_envs))

    step_index = diff_time_step.astype(jnp.int32)
    time_idx_2d = jnp.broadcast_to(time_idx[:, None], step_index.shape)

    episode_end = jnp.logical_or(done.astype(bool), truncated.astype(bool))
    episode_end_cumsum = jnp.cumsum(episode_end.astype(jnp.int32), axis=0)
    episode_end_cumsum = jnp.concatenate(
        [jnp.zeros((1, num_envs), dtype=jnp.int32), episode_end_cumsum], axis=0
    )

    if mask_next_state_on_episode_end:
        next_idx_2d = jnp.broadcast_to(
            jnp.minimum(time_idx + shift_steps, rollout_steps), step_index.shape
        )
        ends_before_start = jnp.take_along_axis(
            episode_end_cumsum, time_idx_2d, axis=0
        )
        ends_before_next = jnp.take_along_axis(
            episode_end_cumsum, next_idx_2d, axis=0
        )
        next_emb_mask = jnp.logical_and(
            next_emb_mask, (ends_before_next - ends_before_start) == 0
        )

    remaining_steps = jnp.maximum(
        0, jnp.asarray(shift_steps - 1, dtype=step_index.dtype) - step_index
    )
    reward_target_idx = jnp.minimum(
        time_idx_2d + remaining_steps, rollout_steps - 1
    )
    reward_target = jnp.take_along_axis(reward, reward_target_idx, axis=0)

    valid_reward_target = (time_idx_2d + remaining_steps) <= (rollout_steps - 1)
    ends_before_start = jnp.take_along_axis(episode_end_cumsum, time_idx_2d, axis=0)
    ends_before_target = jnp.take_along_axis(
        episode_end_cumsum, reward_target_idx, axis=0
    )
    reward_target_mask = jnp.logical_and(
        valid_reward_target, (ends_before_target - ends_before_start) == 0
    )
    reward_target = jnp.where(
        reward_target_mask, reward_target, jnp.zeros_like(reward_target)
    )

    return next_state_emb, next_emb_mask, reward_target, reward_target_mask
