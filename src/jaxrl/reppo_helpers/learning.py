"""Helper functions that encapsulate the per-step math used by ReppoDMERLTrainer."""

from __future__ import annotations

from typing import Tuple

import jax
from flax import nnx
from jax import numpy as jnp
import optax

from src.jaxrl import utils


def compute_nstep_lambda_step(
    gamma: float,
    lmbda: float,
    carry: Tuple[jax.Array, jax.Array, jax.Array],
    transition,
):
    """Single TD-lambda update for use inside a scan."""
    lambda_return, truncated, importance_weight = carry
    done = transition.done
    reward = transition.soft_reward
    value = transition.value

    lambda_sum = (
        jnp.exp(importance_weight) * lmbda * lambda_return
        + (1 - jnp.exp(importance_weight) * lmbda) * value
    )
    delta = gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
    lambda_return = reward + delta
    truncated = transition.truncated
    return (
        lambda_return,
        truncated,
        transition.importance_weight,
    ), lambda_return


def critic_loss_fn(params, train_state, minibatch, target_vals, cfg):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_pred = critic_model.critic_cat(minibatch.critic_obs, minibatch.action).squeeze()
        if cfg.hl_gauss:
            target_cat = jax.vmap(
                utils.hl_gauss, in_axes=(0, None, None, None)
            )(target_vals, cfg.num_bins, cfg.vmin, cfg.vmax)
            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        else:
            critic_update_loss = optax.squared_error(
                critic_pred.reshape(-1, 1),
                target_vals.reshape(-1, 1),
            )

        _, pred, pred_rew, value = critic_model.forward(minibatch.critic_obs, minibatch.action)
        aux_loss = optax.squared_error(pred, minibatch.next_emb)
        aux_rew_loss = optax.squared_error(
            pred_rew, minibatch.reward.reshape(-1, 1)
        )
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1))
            * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
        )
        critic_loss = optax.squared_error(value, target_vals)
        critic_loss = jnp.mean(critic_loss)
        loss = jnp.mean(
            (1.0 - minibatch.truncated)
            * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
        )
        critic_pnorm = utils.tree_norm(params)
        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=aux_rew_loss,
            q=value.mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_vals.mean(),
            critic_pnorm=critic_pnorm,
        )



def actor_loss_fn(params, updated_state, step_key, minibatch, target_vals, action_size_target, cfg, actor_target_model):
        critic_target_model = nnx.merge(
            updated_state.critic.graphdef,
            updated_state.critic.params,
        )
        actor_model = nnx.merge(updated_state.actor.graphdef, params)
        pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
            minibatch.obs, step_key
        )
        entropy_prior = actor_model.get_prior_entropy()
        log_prob_ratio = gen_log_prob - dest_log_prob
        value = critic_target_model.critic(minibatch.critic_obs, pred_action)
        entropy = -cfg.diffusion.diff_steps * jnp.mean(log_prob_ratio, axis=0)
        entropy = jax.lax.stop_gradient(entropy)
        # print the entropy in jax debug mode also print the target entropy and the temperature
        #jax.debug.print("Entropy: {ent}, target: {tar}, temp: {temp}", ent=entropy, tar=action_size_target, temp=actor_model.temperature())

        keys = jax.random.split(step_key, cfg.kl_action_rep)
        if cfg.reverse_kl:
            def compute_single(k):
                return actor_model.rkl_div_one_step(k, minibatch.obs, actor_target_model, stop_grad=False)
        else:
            def compute_single(k):
                return actor_model.fkl_div_one_step(k, minibatch.obs, actor_target_model, stop_grad=False)
        kl_log_ratios = jax.vmap(compute_single)(keys)
        kl_log_ratios = kl_log_ratios.mean(axis=0)
        kl = cfg.diffusion.diff_steps * kl_log_ratios.sum(-1)
        lagrangian = actor_model.lagrangian()

        if cfg.actor_kl_clip_mode == "full":
            actor_loss_val = (
                log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature())
                - value
                + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
            )
        elif cfg.actor_kl_clip_mode == "clipped":
            actor_loss_val = jnp.where(
                kl < cfg.kl_bound,
                log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature()) - value,
                kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
            )
        elif cfg.actor_kl_clip_mode == "value":
            actor_loss_val = (
                log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature())
                - value
            )
        else:
            raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

        target_entropy = action_size_target + entropy
        target_entropy_loss = (
            actor_model.temperature() * jax.lax.stop_gradient(target_entropy)
        ).mean()
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - cfg.kl_bound)
        lagrangian_loss = lagrangian_loss.mean()
        loss = jnp.mean(actor_loss_val)
        if cfg.update_entropy_lagrangian:
            loss += target_entropy_loss
        if cfg.update_kl_lagrangian:
            loss += lagrangian_loss

        actor_pnorm = utils.tree_norm(params)
        friction = actor_model.diffusion_model.friction.value
        friction_detached = jax.lax.stop_gradient(friction)
        return loss, dict(
            actor_loss=actor_loss_val,
            loss=loss,
            temp=actor_model.temperature(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean() * cfg.diffusion.diff_steps,
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            run_cost=0.0,
            sto_cost=0.0,
            terminal_cost=0.0,
            entropy=entropy,
            entropy_loss=target_entropy_loss,
            target_values=target_vals.mean(),
            actor_pnorm=actor_pnorm,
            friction=friction_detached.mean(),
            entropy_prior=entropy_prior,
        )