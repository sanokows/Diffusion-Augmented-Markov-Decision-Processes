from functools import partial

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from src.jaxrl.reppo_helpers.env_time_discounting import (
    maybe_env_time_discount_lambda,
    maybe_env_time_value,
)
from src.jaxrl.reppo_helpers.learning_DiffPPO import compute_gae_step
from src.jaxrl.reppo_helpers.learning_DiffReppo import compute_nstep_lambda_step


@struct.dataclass
class ReppoTransition:
    obs: dict
    soft_reward: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array


@struct.dataclass
class PPOTransition:
    obs: dict
    soft_reward: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array


def _obs_for_steps(steps):
    return {
        "diff_time_step": jnp.asarray(steps, dtype=jnp.float32).reshape(
            len(steps), 1, 1
        )
    }


def test_env_time_discount_uses_gamma_only_on_final_diffusion_step():
    obs = _obs_for_steps([0, 1, 2])

    disabled = maybe_env_time_value(obs, 0.9, 3, False)
    enabled = maybe_env_time_value(obs, 0.9, 3, True)
    discount, trace_decay = maybe_env_time_discount_lambda(obs, 0.9, 0.5, 3, True)

    np.testing.assert_allclose(np.asarray(disabled), np.asarray(0.9))
    np.testing.assert_allclose(np.asarray(enabled).squeeze(), [1.0, 1.0, 0.9])
    np.testing.assert_allclose(np.asarray(discount).squeeze(), [1.0, 1.0, 0.9])
    np.testing.assert_allclose(np.asarray(trace_decay).squeeze(), [1.0, 1.0, 0.5])


def test_reppo_lambda_target_preserves_legacy_and_supports_env_time_discounting():
    gamma = 0.9
    lmbda = 0.5
    batch = ReppoTransition(
        obs=_obs_for_steps([0, 1, 2]),
        soft_reward=jnp.asarray([[0.0], [0.0], [1.0]], dtype=jnp.float32),
        value=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32),
        done=jnp.zeros((3, 1), dtype=jnp.float32),
        truncated=jnp.zeros((3, 1), dtype=jnp.float32),
        importance_weight=jnp.zeros((3, 1), dtype=jnp.float32),
    )
    init = (
        batch.value[-1],
        jnp.ones_like(batch.truncated[0]),
        jnp.zeros_like(batch.importance_weight[0]),
    )

    _, legacy_targets = jax.lax.scan(
        partial(compute_nstep_lambda_step, gamma, lmbda),
        init,
        batch,
        reverse=True,
    )

    discount, trace_decay = maybe_env_time_discount_lambda(
        batch.obs, gamma, lmbda, 3, True
    )

    def env_time_step(carry, inputs):
        transition, gamma_t, lmbda_t = inputs
        return compute_nstep_lambda_step(gamma_t, lmbda_t, carry, transition)

    _, env_time_targets = jax.lax.scan(
        env_time_step,
        init,
        (batch, discount, trace_decay),
        reverse=True,
    )

    np.testing.assert_allclose(
        np.asarray(legacy_targets).squeeze(),
        [14.22, 21.6, 28.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(env_time_targets).squeeze(),
        [28.0, 28.0, 28.0],
        rtol=1e-6,
    )


def test_diffppo_gae_preserves_legacy_and_supports_env_time_discounting():
    gamma = 0.9
    lmbda = 0.5
    batch = PPOTransition(
        obs=_obs_for_steps([0, 1, 2]),
        soft_reward=jnp.asarray([[0.0], [0.0], [1.0]], dtype=jnp.float32),
        value=jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32),
        done=jnp.zeros((3, 1), dtype=jnp.float32),
        truncated=jnp.zeros((3, 1), dtype=jnp.float32),
    )
    init = (jnp.zeros((1,), dtype=jnp.float32), jnp.asarray([5.0], dtype=jnp.float32))

    _, legacy_advantages = jax.lax.scan(
        partial(compute_gae_step, gamma, lmbda),
        init,
        batch,
        reverse=True,
    )

    discount, trace_decay = maybe_env_time_discount_lambda(
        batch.obs, gamma, lmbda, 3, True
    )

    def env_time_step(carry, inputs):
        transition, gamma_t, lmbda_t = inputs
        return compute_gae_step(gamma_t, lmbda_t, carry, transition)

    _, env_time_advantages = jax.lax.scan(
        env_time_step,
        init,
        (batch, discount, trace_decay),
        reverse=True,
    )

    np.testing.assert_allclose(
        np.asarray(legacy_advantages).squeeze(),
        [6.18875, -4.025, -24.5],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(env_time_advantages + batch.value).squeeze(),
        [5.5, 5.5, 5.5],
        rtol=1e-6,
    )


def test_diffppo_env_time_gae_handles_truncated_transitions():
    gamma = 0.9
    lmbda = 0.5
    batch = PPOTransition(
        obs=_obs_for_steps([0, 1, 2]),
        soft_reward=jnp.zeros((3, 1), dtype=jnp.float32),
        value=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        done=jnp.zeros((3, 1), dtype=jnp.float32),
        truncated=jnp.asarray([[0.0], [1.0], [0.0]], dtype=jnp.float32),
    )
    init = (jnp.zeros((1,), dtype=jnp.float32), jnp.asarray([4.0], dtype=jnp.float32))
    discount, trace_decay = maybe_env_time_discount_lambda(
        batch.obs, gamma, lmbda, 3, True
    )

    def env_time_step(carry, inputs):
        transition, gamma_t, lmbda_t = inputs
        return compute_gae_step(gamma_t, lmbda_t, carry, transition)

    _, advantages = jax.lax.scan(
        env_time_step,
        init,
        (batch, discount, trace_decay),
        reverse=True,
    )

    np.testing.assert_allclose(
        np.asarray(advantages).squeeze(),
        [2.0, 1.0, 0.6],
        rtol=1e-6,
    )
