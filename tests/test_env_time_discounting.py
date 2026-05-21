from functools import partial
from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from src.jaxrl.reppo_helpers.env_time_discounting import (
    maybe_env_time_discount_lambda,
    maybe_env_time_value,
)
from src.jaxrl.reppo_helpers.learning_DA_MDP_PPO import compute_gae_step
from src.jaxrl.reppo_helpers.learning_DA_MDP_REPPO import (
    reward_normalization_scale,
    scale_temperature_for_reward_normalization,
    compute_nstep_lambda_step,
)
from src.jaxrl.normalization import NormalizationState
from src.jaxrl.DA_MDP_REPPO import (
    _init_reward_normalization_window_state,
    _sectioned_wandb_key,
    _update_reward_normalization_window_state,
    _validate_reward_normalization_window_rollouts,
)
from src.env_utils.jax_wrappers import DiffNormalizeVec


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


@struct.dataclass
class DummyDiffState:
    diff_time_step: jax.Array
    truncated: jax.Array
    info: dict


class DummyRewardNormCfg:
    def __init__(self, mode):
        self.reward_normalization_mode = mode
        self.reward_norm_epsilon = 0.0


class DummyRewardNormTrainState:
    def __init__(self, state):
        self.reward_normalization_state = state


class DummyDiffEnv:
    num_envs = 2
    num_diff_steps = 4

    def _obs(self, diff_time_step):
        zeros = jnp.zeros((self.num_envs, 1), dtype=jnp.float32)
        return {
            "orig_obs": zeros,
            "orig_actions": zeros,
            "normed_actions": zeros,
            "diff_time_step": diff_time_step,
        }

    def reset(self, key):
        del key
        diff_time_step = jnp.zeros((self.num_envs, 1), dtype=jnp.float32)
        truncated = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        state = DummyDiffState(
            diff_time_step=diff_time_step,
            truncated=truncated,
            info={"truncation": truncated},
        )
        obs = self._obs(diff_time_step)
        return obs, obs, state

    def step(self, key, state, action):
        del key, action
        next_diff_time_step = state.diff_time_step + 1.0
        real_env_step = next_diff_time_step.reshape(-1)[0] >= self.num_diff_steps
        next_diff_time_step = jnp.where(
            real_env_step, jnp.zeros_like(next_diff_time_step), next_diff_time_step
        )
        reward = jnp.where(
            real_env_step,
            jnp.ones((self.num_envs,), dtype=jnp.float32) * 2.0,
            jnp.zeros((self.num_envs,), dtype=jnp.float32),
        )
        truncated = jnp.zeros((self.num_envs,), dtype=jnp.float32)
        next_state = DummyDiffState(
            diff_time_step=next_diff_time_step,
            truncated=truncated,
            info={"truncation": truncated},
        )
        obs = self._obs(next_diff_time_step)
        done = jnp.zeros((self.num_envs,), dtype=jnp.bool_)
        return obs, obs, next_state, reward, done, {}


def _obs_for_steps(steps):
    return {
        "diff_time_step": jnp.asarray(steps, dtype=jnp.float32).reshape(
            len(steps), 1, 1
        )
    }


def test_dmerl_configs_use_reward_normalization_mode_not_removed_flag():
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "config" / "reppo_dmerl.yaml",
        repo_root / "READMEs" / "DMERL_Reppo_configs.md",
        *(
            repo_root / "READMEs" / "Sweeps" / "DMERL"
        ).rglob("*.sh"),
    ]
    for path in paths:
        text = path.read_text()
        assert "normalize_reward" not in text
        assert "reward_normalization_use_rollout_window" not in text


def test_reward_normalization_window_rollouts_validation():
    assert _validate_reward_normalization_window_rollouts(0) == 0
    assert _validate_reward_normalization_window_rollouts(4) == 4
    try:
        _validate_reward_normalization_window_rollouts(-1)
    except ValueError as exc:
        assert "reward_normalization_window_rollouts must be >= 0" in str(exc)
    else:
        raise AssertionError("negative window rollouts should raise ValueError")


def test_reward_normalization_scale_only_applies_to_soft_reward_and_q_target_modes():
    norm_state = NormalizationState(
        mean=jnp.asarray(3.0, dtype=jnp.float32),
        var=jnp.asarray(4.0, dtype=jnp.float32),
        count=8,
    )
    train_state = DummyRewardNormTrainState(norm_state)

    soft_cfg = DummyRewardNormCfg("soft_reward")
    q_target_cfg = DummyRewardNormCfg("q_target")
    none_cfg = DummyRewardNormCfg("none")

    np.testing.assert_allclose(
        np.asarray(reward_normalization_scale(soft_cfg, train_state)), 2.0
    )
    np.testing.assert_allclose(
        np.asarray(reward_normalization_scale(q_target_cfg, train_state)), 2.0
    )
    np.testing.assert_allclose(
        np.asarray(reward_normalization_scale(none_cfg, train_state)), 1.0
    )
    np.testing.assert_allclose(
        np.asarray(
            scale_temperature_for_reward_normalization(
                jnp.asarray(0.5, dtype=jnp.float32), soft_cfg, train_state
            )
        ),
        0.25,
    )


def test_reward_normalization_window_one_matches_current_rollout():
    values = jnp.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=jnp.float32)
    window_state = _init_reward_normalization_window_state(1)
    window_state, norm_state = _update_reward_normalization_window_state(
        window_state, values
    )

    np.testing.assert_allclose(np.asarray(norm_state.mean), 4.0)
    np.testing.assert_allclose(np.asarray(norm_state.var), 5.0)
    np.testing.assert_allclose(np.asarray(norm_state.count), 4)
    np.testing.assert_allclose(np.asarray(window_state.filled), 1)


def test_reward_normalization_window_aggregates_latest_rollouts():
    rollouts = [
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
        jnp.asarray([5.0, 7.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
    ]
    window_state = _init_reward_normalization_window_state(4)
    for values in rollouts:
        window_state, norm_state = _update_reward_normalization_window_state(
            window_state, values
        )

    concatenated = jnp.concatenate(rollouts)
    np.testing.assert_allclose(np.asarray(norm_state.mean), np.mean(concatenated))
    np.testing.assert_allclose(
        np.asarray(norm_state.var), np.var(concatenated), rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(norm_state.count), concatenated.shape[0])
    np.testing.assert_allclose(np.asarray(window_state.filled), 4)


def test_reward_normalization_window_drops_oldest_rollout():
    rollouts = [
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
        jnp.asarray([5.0, 7.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([10.0, 11.0], dtype=jnp.float32),
    ]
    window_state = _init_reward_normalization_window_state(4)
    for values in rollouts:
        window_state, norm_state = _update_reward_normalization_window_state(
            window_state, values
        )

    latest_four = jnp.concatenate(rollouts[1:])
    np.testing.assert_allclose(np.asarray(norm_state.mean), np.mean(latest_four))
    np.testing.assert_allclose(
        np.asarray(norm_state.var), np.var(latest_four), rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(norm_state.count), latest_four.shape[0])
    np.testing.assert_allclose(np.asarray(window_state.filled), 4)


def test_reward_normalization_metrics_use_dedicated_wandb_section():
    assert _sectioned_wandb_key("train/reward_norm/std") == "reward_norm/std"
    assert (
        _sectioned_wandb_key("train/reward_norm/window_rollouts")
        == "reward_norm/window_rollouts"
    )
    assert (
        _sectioned_wandb_key("train/reward_norm/window_filled")
        == "reward_norm/window_filled"
    )
    assert (
        _sectioned_wandb_key("train/reward/raw_mean")
        == "reward_norm/reward/raw_mean"
    )
    assert (
        _sectioned_wandb_key("train/soft_reward/normalized_mean")
        == "reward_norm/soft_reward/normalized_mean"
    )
    assert (
        _sectioned_wandb_key("train/q_target/raw_mean")
        == "reward_norm/q_target/raw_mean"
    )


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


def test_reward_normalization_keeps_intermediate_diffusion_rewards_zero():
    env = DiffNormalizeVec(
        DummyDiffEnv(),
        normalize_reward=True,
        num_diff_steps=DummyDiffEnv.num_diff_steps,
        update_stats=True,
    )
    action = jnp.zeros((DummyDiffEnv.num_envs, 1), dtype=jnp.float32)
    _, _, state = env.reset(None)

    rewards = []
    for _ in range(DummyDiffEnv.num_diff_steps + 1):
        _, _, state, reward, _, _ = env.step(None, state, action)
        rewards.append(np.asarray(reward))

    np.testing.assert_allclose(rewards[0], np.zeros((DummyDiffEnv.num_envs,)))
    np.testing.assert_allclose(rewards[1], np.zeros((DummyDiffEnv.num_envs,)))
    np.testing.assert_allclose(rewards[2], np.zeros((DummyDiffEnv.num_envs,)))
    assert np.any(np.abs(rewards[3]) > 0.0)
    np.testing.assert_allclose(rewards[4], np.zeros((DummyDiffEnv.num_envs,)))


def test_frozen_reward_normalization_keeps_intermediate_diffusion_rewards_zero():
    env = DiffNormalizeVec(
        DummyDiffEnv(),
        normalize_reward=True,
        num_diff_steps=DummyDiffEnv.num_diff_steps,
        update_stats=False,
    )
    action = jnp.zeros((DummyDiffEnv.num_envs, 1), dtype=jnp.float32)
    _, _, state = env.reset(None)
    state = state.replace(
        reward_mean=jnp.asarray(1.0, dtype=jnp.float32),
        reward_var=jnp.asarray(1.0, dtype=jnp.float32),
    )

    _, _, state, reward, _, _ = env.step(None, state, action)
    np.testing.assert_allclose(np.asarray(reward), np.zeros((DummyDiffEnv.num_envs,)))

    for _ in range(DummyDiffEnv.num_diff_steps - 2):
        _, _, state, _, _, _ = env.step(None, state, action)
    _, _, _, reward, _, _ = env.step(None, state, action)

    assert np.any(np.abs(np.asarray(reward)) > 0.0)
