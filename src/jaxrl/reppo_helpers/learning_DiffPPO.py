"""Helper functions that encapsulate the per-step math used by ReppoDMERLTrainer."""

from __future__ import annotations

from typing import Tuple

import jax
from flax import nnx
from jax import numpy as jnp
import optax

from src.jaxrl import utils


def compute_nstep_lambda_step(gamma, lmbda, carry, transition):
    gae, next_value, importance_weight = carry
    done = transition.done
    truncated = transition.truncated
    reward = transition.reward
    value = transition.value
    
    delta = reward + gamma * next_value * (1 - done) - value
    gae = delta + gamma * lmbda * (1 - done) * gae
    truncated_gae = reward + gamma * next_value - value
    gae = jnp.where(truncated, truncated_gae, gae)
    return (gae, value, transition.importance_weight), gae


def critic_loss_fn(params, train_state, minibatch, target_values, cfg):
    critic_model = nnx.merge(train_state.critic.graphdef, params)
    value = critic_model.critic_cat(minibatch.critic_obs, minibatch.action).squeeze()
    value_pred_clipped = minibatch.value + (value - minibatch.value).clip(-cfg.clip_ratio, cfg.clip_ratio)

    value_error = jnp.square(value - target_values)
    value_error_clipped = jnp.square(value_pred_clipped - target_values)
    value_loss = 0.5 * jnp.mean(
        (1.0 - minibatch.truncated)
        * jnp.maximum(value_error, value_error_clipped)
    )
    loss = value_loss
    critic_pnorm = utils.tree_norm(params)
    return loss, dict(
        value_loss=value_loss,
        loss=loss,
        q=value.mean(),
        reward_mean=minibatch.reward.mean(),
        target_values=target_values.mean(),
        critic_pnorm=critic_pnorm,
    )

def actor_loss_fn(params, updated_state, step_key, minibatch, advantages, action_size_target, cfg, actor_target_model):
    value_model = nnx.merge(
            updated_state.critic.graphdef,
            updated_state.critic.params,
        )
   
    actor_model = nnx.merge(updated_state.actor.graphdef, params)

    value = value_model(minibatch.critic_obs)

    pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(minibatch.obs, step_key)
    log_prob_ratio = gen_log_prob - dest_log_prob

    ratio = jnp.exp(gen_log_prob - minibatch.log_prob)

    temperature = actor_model.temperature()
    mean_log_prob_ratio = jnp.mean(log_prob_ratio)
    entropy_advantages = jax.lax.stop_gradient( -temperature * (log_prob_ratio - mean_log_prob_ratio) +  advantages)
    actor_loss1 = ratio * entropy_advantages
    actor_loss2 = (
        jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
        * entropy_advantages
    )
    actor_loss = -jnp.mean(
        (1.0 - minibatch.truncated)
        * jnp.minimum(actor_loss1, actor_loss2)
    )
    entropy_loss = jnp.mean(actor_model.entropy())

    loss = (
        actor_loss
    )

    return loss, dict(
        actor_loss=actor_loss,
        entropy_loss=entropy_loss,
        loss=loss,
        mean_value=value.mean(),
        mean_log_prob=gen_log_prob.mean(),
        mean_advantages=advantages.mean(),
        mean_action=minibatch.action.mean(),
        mean_reward=minibatch.reward.mean(),
    )

def train_step_env(Transition, cfg, env, actor_model, critic_model, carry, _):
    key, env_state, inner_state, obs, critic_obs = carry
    key, act_key, step_key = jax.random.split(key, 3)
    step_key = jax.random.split(step_key, cfg.num_envs)
    action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
        obs, act_key
    )
    action = jax.lax.stop_gradient(action)
    next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
        step_key, env_state, action
    )
    importance_weight = jnp.zeros((cfg.num_envs,))
    key, next_act_key = jax.random.split(key)
    next_action, next_gen_log_prob, next_dest_log_prob = (
        actor_model.vmap_sample_next_step(next_obs, next_act_key)
    )
    next_action = jax.lax.stop_gradient(next_action)
    value = critic_model.forward(next_critic_obs, next_action)
    log_ratio = jax.lax.stop_gradient(
        next_gen_log_prob - next_dest_log_prob
    )
    soft_reward = (
        reward
        - cfg.gamma * log_ratio.squeeze() * actor_model.temperature()
    )
    transition = Transition(
        obs=obs,
        critic_obs=critic_obs,
        action=action,
        next_emb=None,
        reward=reward,
        soft_reward=soft_reward,
        value=value,
        done=done,
        truncated=next_env_state.truncated,
        info=info,
        importance_weight=importance_weight,
    )
    return (
        key,
        next_env_state,
        inner_state,
        next_obs,
        next_critic_obs,
    ), transition
