from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp


def final_diffusion_step_mask(obs: dict[str, jax.Array], diff_steps: int) -> jax.Array:
    step_index = obs["diff_time_step"][..., 0]
    final_step = jnp.asarray(diff_steps - 1, dtype=step_index.dtype)
    return step_index == final_step


def maybe_env_time_value(
    obs: dict[str, jax.Array],
    env_value: Any,
    diff_steps: int,
    use_env_time_discounting: bool,
    *,
    legacy_value: Any | None = None,
    intermediate_value: Any = 1.0,
) -> jax.Array:
    if not use_env_time_discounting:
        return jnp.asarray(env_value if legacy_value is None else legacy_value)

    dtype = jnp.result_type(env_value, intermediate_value, jnp.float32)
    return jnp.where(
        final_diffusion_step_mask(obs, diff_steps),
        jnp.asarray(env_value, dtype=dtype),
        jnp.asarray(intermediate_value, dtype=dtype),
    )


def maybe_env_time_discount_lambda(
    obs: dict[str, jax.Array],
    gamma: float,
    lmbda: float,
    diff_steps: int,
    use_env_time_discounting: bool,
) -> tuple[jax.Array, jax.Array]:
    return (
        maybe_env_time_value(obs, gamma, diff_steps, use_env_time_discounting),
        maybe_env_time_value(obs, lmbda, diff_steps, use_env_time_discounting),
    )
