"""Helper functions that encapsulate the per-step math used by ReppoDMERLTrainer."""

from __future__ import annotations

from typing import Tuple

import jax
from flax import nnx
from jax import numpy as jnp
from jax import tree_util
import optax

from src.jaxrl import utils


def _resolve_temperature(actor_model, cfg, train_state=None) -> jax.Array:
    """Return either learned temperature or an exponential decay schedule."""
    if bool(getattr(cfg, "new_temp_mode", False)) and getattr(cfg, "train_mode", None) == "WPO":
        return jnp.asarray(getattr(cfg, "ent_start", 1.0), dtype=jnp.float32)
    if not bool(getattr(cfg, "use_temperature_decay", False)):
        return actor_model.temperature()
    start = getattr(cfg, "temperature_decay_start", None)
    end = getattr(cfg, "temperature_decay_end", None)
    if start is None:
        start = cfg.ent_start
    if end is None:
        end = start
    if start <= 0.0 or end <= 0.0:
        raise ValueError(
            "temperature_decay_start and temperature_decay_end must be > 0."
        )
    decay_steps = getattr(cfg, "temperature_decay_steps", None)
    if decay_steps is None:
        decay_steps = cfg.total_time_steps
    decay_steps = max(int(decay_steps), 1)
    time_steps = 0.0 if train_state is None else train_state.time_steps
    progress = jnp.clip(jnp.asarray(time_steps, dtype=jnp.float32) / decay_steps, 0.0, 1.0)
    start_val = jnp.asarray(start, dtype=jnp.float32)
    end_val = jnp.asarray(end, dtype=jnp.float32)
    log_ratio = jnp.log(end_val) - jnp.log(start_val)
    return start_val * jnp.exp(progress * log_ratio)


def _maybe_stop_grad_entropy(entropy: jax.Array, cfg) -> jax.Array:
    if bool(getattr(cfg, "stop_grad_entropy", True)):
        return jax.lax.stop_gradient(entropy)
    return entropy


def compute_action_q_grads(
    actor_model, critic_model, obs, critic_obs, temperature: jax.Array | None = None
):
    """Compute eval-style guidance gradient: q_grad + temperature * grad(log p)."""
    actions = obs["orig_actions"]

    def _q_grad(single_critic_obs, act):
        q_fn = lambda a: critic_model.critic(single_critic_obs, a).sum()
        return jax.grad(q_fn)(act)

    q_grad = jax.vmap(_q_grad)(critic_obs, actions)

    obs_for_logp = dict(obs)
    if bool(getattr(actor_model.diffusion_model, "langevin_param", False)):
        obs_for_logp["q_grad"] = jax.lax.stop_gradient(q_grad)

    def _forward_logp_grad(single_obs, single_action):
        def _logp(a):
            single_obs_batched = jax.tree.map(
                lambda x: jnp.expand_dims(x, axis=0), single_obs
            )
            a_batched = jnp.expand_dims(a, axis=0)
            _, dest_log_prob = actor_model.vmap_eval_log_prob(
                single_obs_batched, a_batched
            )
            return dest_log_prob.sum()

        return jax.grad(_logp)(single_action)

    grad_log_p = jax.vmap(_forward_logp_grad)(obs_for_logp, actions)
    if temperature is None:
        temperature = actor_model.temperature()
    guidance_temp = jax.lax.stop_gradient(jnp.asarray(temperature, dtype=grad_log_p.dtype))
    guidance_grad = q_grad + guidance_temp * grad_log_p
    return guidance_grad


def maybe_add_q_grad(
    obs,
    critic_obs,
    actor_model,
    critic_model,
    use_langevin: bool,
    temperature: jax.Array | None = None,
):
    """Attach guidance gradients to the observation dict when enabled."""
    if not use_langevin:
        return obs
    q_grad = compute_action_q_grads(
        actor_model, critic_model, obs, critic_obs, temperature=temperature
    )
    obs_with_grad = dict(obs)
    obs_with_grad["q_grad"] = jax.lax.stop_gradient(q_grad)
    return obs_with_grad


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


def _metric_scalar(x: jax.Array) -> jax.Array:
    return jnp.mean(jnp.asarray(x))


def _weighted_batch_mean(values: jax.Array, importance_ratio: jax.Array | None) -> jax.Array:
    """Importance-weighted mean over batch axis (axis 0)."""
    values = jnp.asarray(values)
    if values.ndim == 0:
        raise ValueError(
            "_weighted_batch_mean expects per-sample values with a batch axis, got scalar."
        )
    if importance_ratio is None:
        return jnp.mean(values)
    ratio = jnp.asarray(importance_ratio, dtype=values.dtype).reshape((-1,))
    if values.shape[0] != ratio.shape[0]:
        raise ValueError(
            "_weighted_batch_mean shape mismatch: values batch axis "
            f"{values.shape[0]} != importance_ratio length {ratio.shape[0]}."
        )
    ratio = ratio.reshape((ratio.shape[0],) + (1,) * (values.ndim - 1))
    return jnp.mean(ratio * values)


def _weighted_batch_mean_axis0(
    values: jax.Array, importance_ratio: jax.Array | None
) -> jax.Array:
    """Importance-weighted mean over batch axis while preserving remaining axes."""
    values = jnp.asarray(values)
    if values.ndim == 0:
        raise ValueError(
            "_weighted_batch_mean_axis0 expects per-sample values with a batch axis, got scalar."
        )
    if importance_ratio is None:
        return jnp.mean(values, axis=0)
    ratio = jnp.asarray(importance_ratio, dtype=values.dtype).reshape((-1,))
    if values.shape[0] != ratio.shape[0]:
        raise ValueError(
            "_weighted_batch_mean_axis0 shape mismatch: values batch axis "
            f"{values.shape[0]} != importance_ratio length {ratio.shape[0]}."
        )
    ratio = ratio.reshape((ratio.shape[0],) + (1,) * (values.ndim - 1))
    return jnp.mean(ratio * values, axis=0)


def critic_loss_fn(params, train_state, minibatch, target_vals, cfg, importance_ratio=None):
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

        _, pred, pred_rew, pred_next_diff_state, value = critic_model.forward(
            minibatch.critic_obs, minibatch.action
        )
        # `next_state_emb` is shifted by `diff_steps` in the trainer; tail elements are
        # invalid and must be masked out to avoid supervising with clamped indices.
        next_state_mask = minibatch.next_emb_mask.reshape(-1, 1).astype(pred.dtype)
        aux_loss = (
            (1.0 - minibatch.truncated.reshape(-1, 1))
            * next_state_mask
            * optax.squared_error(pred, minibatch.next_state_emb)
        )
        use_final_step_reward_target = bool(
            getattr(cfg, "use_final_step_reward_target", False)
        )
        if use_final_step_reward_target:
            reward_target = getattr(minibatch, "reward_target", minibatch.reward).reshape(
                -1, 1
            )
            reward_target_mask = getattr(
                minibatch, "reward_target_mask", jnp.ones_like(minibatch.reward)
            ).reshape(-1, 1).astype(pred_rew.dtype)
        else:
            reward_target = minibatch.reward.reshape(-1, 1)
            reward_target_mask = jnp.ones_like(reward_target, dtype=pred_rew.dtype)
        aux_rew_loss = (
            1.0 - minibatch.truncated.reshape(-1, 1)
        ) * reward_target_mask * optax.squared_error(pred_rew, reward_target)

        use_normed_actions = bool(getattr(cfg, "critic_use_normed_actions", True))
        if use_normed_actions:
            diff_steps = jnp.asarray(
                cfg.diffusion.diff_steps - 1,
                dtype=minibatch.obs["diff_time_step"].dtype,
            )
            is_last_step = (minibatch.obs["diff_time_step"][..., 0] == diff_steps).reshape(
                -1, 1
            )
            aux_weight = is_last_step.astype(aux_loss.dtype)
        else:
            step_index = minibatch.obs["diff_time_step"][..., 0]
            max_step = jnp.maximum(
                jnp.asarray(cfg.diffusion.diff_steps - 1, dtype=step_index.dtype), 1.0
            )
            min_weight = 1.0 / jnp.maximum(
                jnp.asarray(cfg.diffusion.diff_steps, dtype=step_index.dtype), 1.0
            )
            step_progress = jnp.clip(step_index / max_step, 0.0, 1.0)
            aux_weight = (
                min_weight + (1.0 - min_weight) * step_progress
            ).reshape(-1, 1).astype(aux_loss.dtype)
            #aux_weight = cfg.diffusion.diff_steps *aux_weight**2
            aux_weight = cfg.diffusion.diff_steps * (aux_weight == 1.0)

        masked_aux_terms = jnp.concatenate([aux_loss, aux_rew_loss], axis=-1)
        masked_aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1)) * aux_weight * masked_aux_terms,
            axis=-1,
        )
        if use_normed_actions:
            aux_next_diff_loss = (1.0 - minibatch.truncated.reshape(-1, 1)) * optax.squared_error(
                pred_next_diff_state, minibatch.next_emb
            )
            aux_next_diff_loss = jnp.mean(
                (1 - minibatch.done.reshape(-1, 1)) * aux_next_diff_loss,
                axis=-1,
            )
            alpha = cfg.aux_loss_alpha
            aux_loss = (
                alpha * jnp.sum(masked_aux_loss) / jnp.maximum(jnp.sum(aux_weight), 1.0)
                + (1 - alpha) * jnp.mean(aux_next_diff_loss)
            )
        else:
            alpha = 1.0
            aux_loss = jnp.mean(masked_aux_loss)
        critic_loss = _weighted_batch_mean(optax.squared_error(value, target_vals), importance_ratio)
        loss = _weighted_batch_mean(
            (1.0 - minibatch.truncated)
            * (critic_update_loss)
            + cfg.aux_loss_mult * aux_loss,
            importance_ratio,
        )
        metrics = dict(
            value_loss=critic_loss,
            critic_update_loss=_metric_scalar(_weighted_batch_mean(critic_update_loss, importance_ratio)),
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=_metric_scalar(
                _weighted_batch_mean(
                    aux_rew_loss * aux_weight.astype(aux_rew_loss.dtype), importance_ratio
                )
            ),
            q=_metric_scalar(_weighted_batch_mean(value, importance_ratio)),
            reward_mean=_metric_scalar(_weighted_batch_mean(minibatch.reward, importance_ratio)),
            target_values=_metric_scalar(_weighted_batch_mean(target_vals, importance_ratio)),
        )
        if bool(getattr(cfg, "log_pnorms", False)):
            metrics["critic_pnorm"] = utils.tree_norm(params)
        return loss, metrics



def actor_loss_fn(
    params,
    updated_state,
    critic_rollout_model,
    step_key,
    minibatch,
    target_vals,
    action_size_target,
    cfg,
    actor_target_model,
    importance_ratio=None,
):
        use_langevin = bool(cfg.diffusion.score_model.langevin_param)
        use_current_critic_for_actions = bool(
            getattr(cfg, "use_current_critic_for_actor_samples", False)
        )
        actor_model = nnx.merge(updated_state.actor.graphdef, params)
        temperature = _resolve_temperature(actor_model, cfg, updated_state)
        critic_current_model = nnx.merge(
            updated_state.critic.graphdef, updated_state.critic.params
        )
        critic_for_target = (
            critic_current_model if use_current_critic_for_actions else critic_rollout_model
        )
        obs_for_actions = maybe_add_q_grad(
            minibatch.obs,
            minibatch.critic_obs,
            actor_model,
            critic_current_model,
            use_langevin,
            temperature=temperature,
        )
        obs_for_target = maybe_add_q_grad(
            minibatch.obs,
            minibatch.critic_obs,
            actor_model,
            critic_for_target,
            use_langevin,
            temperature=temperature,
        )
        pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
            obs_for_actions, step_key
        )
        entropy_prior = actor_model.get_prior_entropy()
        log_prob_ratio = gen_log_prob - dest_log_prob
        value = critic_current_model.critic(minibatch.critic_obs, pred_action)
    
        #print the shape of log_prob_ratio
        #jax.debug.print("log_prob_ratio shape: {shape}", shape=log_prob_ratio.shape)
        entropy = -cfg.diffusion.diff_steps * _weighted_batch_mean_axis0(
            log_prob_ratio, importance_ratio
        )
        entropy = _maybe_stop_grad_entropy(entropy, cfg)
        # print the entropy in jax debug mode also print the target entropy and the temperature
        #jax.debug.print("Entropy: {ent}, target: {tar}, temp: {temp}", ent=entropy, tar=action_size_target, temp=actor_model.temperature())

        keys = jax.random.split(step_key, cfg.kl_action_rep)
        if cfg.reverse_kl:
            def compute_single(k):
                return actor_model.rkl_div_one_step(k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False)
        else:
            def compute_single(k):
                return actor_model.fkl_div_one_step(k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False)
                
        kl_log_ratios = jax.vmap(compute_single)(keys)
        kl_log_ratios = kl_log_ratios.mean(axis=0)
        kl = cfg.diffusion.diff_steps * kl_log_ratios.sum(-1)
        lagrangian = actor_model.lagrangian()

        clip_ratio = _weighted_batch_mean((kl >= cfg.kl_bound).astype(jnp.float32), importance_ratio)

        target_entropy = action_size_target + entropy
        kl_constraint = kl - cfg.kl_bound

        if cfg.actor_kl_clip_mode == "full":
            actor_loss_val = (
                log_prob_ratio * jax.lax.stop_gradient(temperature)
                - value
                + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
            )
        elif cfg.actor_kl_clip_mode == "clipped":
            actor_loss_val = jnp.where(
                kl < cfg.kl_bound,
                log_prob_ratio * jax.lax.stop_gradient(temperature) - value,
                kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
            )
        elif cfg.actor_kl_clip_mode == "value":
            actor_loss_val = (
                log_prob_ratio * jax.lax.stop_gradient(temperature)
                - value
            )
        else:
            raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

        target_entropy_loss = (
            temperature
            * jax.lax.stop_gradient(target_entropy) ### should ther ebe a stop grad for WPO?
        )
        # This is a global scalar (or shape-(1,)) term, not per-sample.
        target_entropy_loss = _metric_scalar(target_entropy_loss)
        lagrangian_loss = (
            -lagrangian
            * jax.lax.stop_gradient(kl_constraint)
        )
        lagrangian_loss = _weighted_batch_mean(lagrangian_loss, importance_ratio)

        loss = _weighted_batch_mean(actor_loss_val, importance_ratio)
        if cfg.update_entropy_lagrangian:
            loss += target_entropy_loss
        if cfg.update_kl_lagrangian:
            loss += lagrangian_loss

        friction = actor_model.diffusion_model.friction.value
        friction_detached = jax.lax.stop_gradient(friction)
        metrics = dict(
            actor_loss=_metric_scalar(_weighted_batch_mean(actor_loss_val, importance_ratio)),
            loss=loss,
            temp=_metric_scalar(temperature),
            abs_batch_action=_metric_scalar(
                _weighted_batch_mean(jnp.abs(minibatch.action), importance_ratio)
            ),
            abs_pred_action=_metric_scalar(
                _weighted_batch_mean(jnp.abs(pred_action), importance_ratio)
            ),
            reward_mean=_metric_scalar(
                _weighted_batch_mean(minibatch.reward, importance_ratio)
            )
            * cfg.diffusion.diff_steps,
            energy_mean=-_metric_scalar(
                _weighted_batch_mean(minibatch.reward, importance_ratio)
            )
            * cfg.diffusion.diff_steps
            + 1,
            kl=_metric_scalar(_weighted_batch_mean(kl, importance_ratio)),
            lagrangian=_metric_scalar(lagrangian),
            lagrangian_loss=lagrangian_loss,
            run_cost=0.0,
            sto_cost=0.0,
            terminal_cost=0.0,
            entropy=_metric_scalar(entropy),
            entropy_target=-action_size_target,
            entropy_loss=target_entropy_loss,
            kl_clip_ratio=clip_ratio,
            target_values=_metric_scalar(_weighted_batch_mean(target_vals, importance_ratio)),
            friction=friction_detached.mean(),
            entropy_prior=_metric_scalar(entropy_prior),
        )
        if bool(getattr(cfg, "log_pnorms", False)):
            metrics["actor_pnorm"] = utils.tree_norm(params)
        return loss, metrics


def actor_WPO_loss_fn(
    params,
    updated_state,
    critic_rollout_model,
    step_key,
    minibatch,
    target_vals,
    action_size_target,
    cfg,
    actor_target_model,
    importance_ratio=None,
):
        use_new_temp_mode = bool(getattr(cfg, "new_temp_mode", False))
        use_wpo_log_temp_update = (
            use_new_temp_mode and getattr(cfg, "train_mode", None) == "WPO"
        )
        use_langevin = bool(cfg.diffusion.score_model.langevin_param)
        use_current_critic_for_actions = bool(
            getattr(cfg, "use_current_critic_for_actor_samples", False)
        )
        critic_current_model = nnx.merge(
            updated_state.critic.graphdef, updated_state.critic.params
        )
        critic_for_actions = (
            critic_current_model if use_current_critic_for_actions else critic_rollout_model
        )
        actor_model_raw = nnx.merge(updated_state.actor.graphdef, params)
        temperature = (
            actor_model_raw.entropy_lagrangian()
            if use_wpo_log_temp_update
            else _resolve_temperature(actor_model_raw, cfg, updated_state)
        )
        obs_for_actions = maybe_add_q_grad(
            minibatch.obs,
            minibatch.critic_obs,
            actor_model_raw,
            critic_for_actions,
            use_langevin,
            temperature=temperature,
        )
        obs_for_target = maybe_add_q_grad(
            minibatch.obs,
            minibatch.critic_obs,
            actor_model_raw,
            critic_rollout_model,
            use_langevin,
            temperature=temperature,
        )

        stop_grad_params = jax.tree.map(jax.lax.stop_gradient, params)
        batch_size = minibatch.action.shape[0]
        fisher_keys = jax.random.split(step_key, batch_size)

        ### TODO shoudl actually be done seperatly for each state
        def _single_gen_log_prob(p, obs, key):
            actor_single = nnx.merge(updated_state.actor.graphdef, p)
            obs_batched = jax.tree_util.tree_map(lambda x: x[None], obs)
            #jax.debug.print("train_update_step_4_env shape: {shape}", shape=key.shape)
            _, gen_log_prob, _ = actor_single.vmap_sample_next_step(obs_batched, key)
            return gen_log_prob.squeeze()

        if cfg.remove_fisher_precond:
            fisher_actor_model = nnx.merge(updated_state.actor.graphdef, params)
            actor_model = fisher_actor_model
        else:
            per_sample_grads = jax.vmap(
                jax.grad(_single_gen_log_prob), in_axes=(None, 0, 0)
            )(params, obs_for_actions, fisher_keys)

            def _is_array_like(x):
                return hasattr(x, "shape") and hasattr(x, "dtype")

            def _skip_fisher(path):
                for key in path:
                    if isinstance(key, str) and ("temperature" in key or "lagrangian" in key):
                        return True
                return False

            def _precondition_delta(path, p, p0, g):
                delta = p - p0
                if _skip_fisher(path):
                    return delta
                if not _is_array_like(p):
                    return delta
                fisher_diag = jnp.mean(jnp.square(g), axis=0)
                inv_fisher = 1.0 / (fisher_diag + 1e-8)
                sg_inverse_fisher = jax.lax.stop_gradient(inv_fisher)
                return delta * sg_inverse_fisher

            precond_delta = tree_util.tree_map_with_path(
                _precondition_delta, params, stop_grad_params, per_sample_grads
            )

            def _apply_precond(p, p0, d):
                return p0 + d

            fisher_params = jax.tree.map(
                _apply_precond, params, stop_grad_params, precond_delta
            )
            fisher_actor_model = nnx.merge(updated_state.actor.graphdef, fisher_params)
            if cfg.kl_bound_fisher_precond:               
                actor_model = fisher_actor_model
            else:
                actor_model = nnx.merge(updated_state.actor.graphdef, params)
        temperature = (
            actor_model.entropy_lagrangian()
            if use_wpo_log_temp_update
            else _resolve_temperature(actor_model, cfg, updated_state)
        )

        #jax.debug.print("train_update_step_5_env shape: {shape}", shape=step_key.shape)
        pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
            obs_for_actions, step_key
        )
        entropy_prior = actor_model.get_prior_entropy()
        log_prob_ratio = gen_log_prob - dest_log_prob
        #print the shape of log_prob_ratio
        #jax.debug.print("log_prob_ratio shape: {shape}, {key_shape}", shape=log_prob_ratio.shape, key_shape=step_key.shape)

        stop_pred_action = jax.lax.stop_gradient(pred_action)

        def single_q(obs, act):
            batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
            q_val = critic_current_model.critic(batched_obs, act[None])
            # Expected shape is scalar-like (e.g. (1,) or (1, 1)); squeeze to scalar.
            return jnp.squeeze(q_val)

        def single_log_probs(obs, act):
            batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
            gen_lp, dest_lp = fisher_actor_model.vmap_eval_log_prob(batched_obs, act[None])
            return jnp.squeeze(gen_lp, axis=0), jnp.squeeze(dest_lp, axis=0)

        q_action_grad = jax.vmap(jax.grad(single_q, argnums=1))(
            minibatch.critic_obs, stop_pred_action
        )
        stop_q_action_grad = jax.lax.stop_gradient(q_action_grad)
        gen_log_prob_action_grad, dest_log_prob_action_grad = jax.vmap(
            jax.jacrev(single_log_probs, argnums=1)
        )(obs_for_actions, stop_pred_action)
        log_prob_action_grad = gen_log_prob_action_grad - dest_log_prob_action_grad

        use_W2_kl = cfg.use_W2_kl
        if use_W2_kl:
            raise ValueError("W2 KL is currently not supported for WPO loss.")
            def target_single_log_probs(obs, act):
                batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
                ### should new states be sampled here?
                gen_lp_old, _ = actor_target_model.vmap_eval_log_prob(batched_obs, act[None])
                # Expected shape is scalar-like (e.g. (1,)); squeeze to scalar.
                return jnp.squeeze(gen_lp_old)


            target_log_prob_grad = jax.vmap( 
                jax.grad(target_single_log_probs, argnums=1)
            )(obs_for_target, stop_pred_action)

            #jax.debug.print("target_log_prob_grad shape: {shape}", shape=target_log_prob_grad.shape)
            kl = cfg.diffusion.diff_steps * jnp.mean(jnp.sum(
                (target_log_prob_grad - gen_log_prob_action_grad) ** 2, axis=-1
            ), axis = 0)
                        # fKL constraint (matches actor_loss_fn).
            kl_keys = jax.random.split(step_key, cfg.kl_action_rep)
            if cfg.reverse_kl:
                def compute_single(k):
                    return actor_model.rkl_div_one_step(
                        k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False
                    )
            else:
                def compute_single(k):
                    return actor_model.fkl_div_one_step(
                        k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False
                    )
            kl_log_ratios = jax.vmap(compute_single)(kl_keys)
            kl_log_ratios = kl_log_ratios.mean(axis=0)
            kl_clip_value = cfg.diffusion.diff_steps * kl_log_ratios.sum(-1)
        else:
            # fKL constraint (matches actor_loss_fn).
            kl_keys = jax.random.split(step_key, cfg.kl_action_rep)
            if cfg.reverse_kl:
                def compute_single(k):
                    return actor_model.rkl_div_one_step(k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False)
            else:
                def compute_single(k):
                    return actor_model.fkl_div_one_step(k, obs_for_actions, obs_for_target, actor_target_model, stop_grad=False)
            kl_log_ratios = jax.vmap(compute_single)(kl_keys)
            kl_log_ratios = kl_log_ratios.mean(axis=0)
            kl = cfg.diffusion.diff_steps * kl_log_ratios.sum(-1)
            kl_clip_value = kl

        lagrangian = actor_model.lagrangian()

        actor_Q_loss = jnp.sum(
            jax.lax.stop_gradient(log_prob_action_grad * temperature - stop_q_action_grad)
            * (gen_log_prob_action_grad - dest_log_prob_action_grad),
            axis=-1,
        )

        safe_temperature = jnp.maximum(temperature, 1e-8)
        delta_t = (
            jax.lax.stop_gradient(log_prob_action_grad)
            - jax.lax.stop_gradient(stop_q_action_grad) / safe_temperature
        )
        delta_t_sq_mean = _weighted_batch_mean(
            (delta_t**2).sum(axis=-1), importance_ratio
        )
        actor_WPO_loss = temperature * delta_t_sq_mean

        clip_ratio = _weighted_batch_mean(
            (kl_clip_value >= cfg.kl_bound).astype(jnp.float32),
            importance_ratio,
        )
        if cfg.actor_kl_clip_mode == "full":
            actor_loss_val = (
                actor_Q_loss
                + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
            )
        elif cfg.actor_kl_clip_mode == "clipped":
            actor_loss_val = jnp.where(
                kl_clip_value < cfg.kl_bound,
                actor_Q_loss,
                kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
            )
        else:
            raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

        entropy_lagrangian = (
            actor_model.entropy_lagrangian() if use_new_temp_mode else temperature
        )

        # log_prob_ratio_maybe_stop_grad = jnp.where(
        #         kl_clip_value < cfg.kl_bound,
        #         log_prob_ratio,
        #         jax.lax.stop_gradient(log_prob_ratio))
        entropy = -cfg.diffusion.diff_steps * _weighted_batch_mean_axis0(
            log_prob_ratio,
            importance_ratio,
        )

        target_entropy = action_size_target + entropy
        kl_constraint = kl_clip_value - cfg.kl_bound

        wpo_temperature_objective = actor_WPO_loss 
        normal_ent_reg = actor_model.temperature()* jax.lax.stop_gradient(target_entropy) - jax.lax.stop_gradient(actor_model.temperature())* target_entropy

        lagrangian_loss = (
            -lagrangian * jax.lax.stop_gradient(kl_constraint)
        )
        lagrangian_loss = _weighted_batch_mean(lagrangian_loss, importance_ratio)
        entropy_penalty = jnp.array(0.0)
        kl_penalty = jnp.array(0.0)

        loss = _weighted_batch_mean(actor_loss_val, importance_ratio)
        if cfg.update_entropy_lagrangian:
            if use_wpo_log_temp_update:
                loss += - wpo_temperature_objective + normal_ent_reg
            else:
                loss += normal_ent_reg
        if cfg.update_kl_lagrangian:
            loss += lagrangian_loss

        friction = actor_model.diffusion_model.friction.value
        friction_detached = jax.lax.stop_gradient(friction)
        metrics = dict(
            actor_loss=_metric_scalar(_weighted_batch_mean(actor_loss_val, importance_ratio)),
            actor_WPO_loss=_metric_scalar(actor_WPO_loss),
            Kl_reg_loss=_metric_scalar(
                _weighted_batch_mean(
                    kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                    importance_ratio,
                )
            ),
            loss=loss,
            temp=_metric_scalar(temperature),
            abs_batch_action=_metric_scalar(
                _weighted_batch_mean(jnp.abs(minibatch.action), importance_ratio)
            ),
            abs_pred_action=_metric_scalar(
                _weighted_batch_mean(jnp.abs(pred_action), importance_ratio)
            ),
            reward_mean=_metric_scalar(
                _weighted_batch_mean(minibatch.reward, importance_ratio)
            )
            * cfg.diffusion.diff_steps,
            energy_mean=-_metric_scalar(
                _weighted_batch_mean(minibatch.reward, importance_ratio)
            )
            * cfg.diffusion.diff_steps
            + 1,
            kl=_metric_scalar(_weighted_batch_mean(kl_clip_value, importance_ratio)),
            kl_clip_value=_metric_scalar(_weighted_batch_mean(kl, importance_ratio)),
            lagrangian=_metric_scalar(lagrangian),
            lagrangian_loss=lagrangian_loss,
            run_cost=0.0,
            sto_cost=0.0,
            terminal_cost=0.0,
            entropy_target=-action_size_target,
            entropy=_metric_scalar(entropy),
            entropy_lagrangian=_metric_scalar(entropy_lagrangian),
            temp_entropy_lagrangian=_metric_scalar(entropy_lagrangian),
            entropy_loss=target_entropy,
            wpo_temperature_objective=_metric_scalar(wpo_temperature_objective),
            delta_t_sq=_metric_scalar(delta_t_sq_mean),
            entropy_penalty=entropy_penalty,
            kl_penalty=kl_penalty,
            kl_clip_ratio=clip_ratio,
            target_values=_metric_scalar(_weighted_batch_mean(target_vals, importance_ratio)),
            friction=friction_detached.mean(),
            entropy_prior=_metric_scalar(entropy_prior),
        )
        if bool(getattr(cfg, "log_pnorms", False)):
            metrics["actor_pnorm"] = utils.tree_norm(params)
        return loss, metrics

def train_step_env(Transition, cfg, env, actor_model, critic_model, carry, _):
    key, env_state, inner_state, obs, critic_obs = carry
    use_langevin = bool(cfg.diffusion.score_model.langevin_param)
    key, act_key, step_key = jax.random.split(key, 3)
    step_key = jax.random.split(step_key, cfg.num_envs)
    temperature = _resolve_temperature(actor_model, cfg, inner_state)
    obs_for_actor = maybe_add_q_grad(
        obs, critic_obs, actor_model, critic_model, use_langevin, temperature=temperature
    )

    jax.debug.print("train_update_step_1_env shape: {shape}", shape=act_key.shape)
    action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
        obs_for_actor, act_key
    )
    action = jax.lax.stop_gradient(action)
    next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
        step_key, env_state, action
    )
    importance_weight = jnp.zeros((cfg.num_envs,))
    key, next_act_key = jax.random.split(key)
    next_obs_for_actor = maybe_add_q_grad(
        next_obs,
        next_critic_obs,
        actor_model,
        critic_model,
        use_langevin,
        temperature=temperature,
    )

    next_action, next_gen_log_prob, next_dest_log_prob = (
        actor_model.vmap_sample_next_step(next_obs_for_actor, next_act_key)
    )
    next_action = jax.lax.stop_gradient(next_action)
    next_emb, _, _, _, value = critic_model.forward(next_critic_obs, next_action)
    log_ratio = jax.lax.stop_gradient(
        next_gen_log_prob - next_dest_log_prob
    )
    soft_reward = (
        reward
        - cfg.gamma * log_ratio.squeeze() * temperature
    )
    transition = Transition(
        obs=obs,
        critic_obs=critic_obs,
        action=action,
        next_emb=next_emb,
        next_state_emb=next_emb,
        next_emb_mask=jnp.ones_like(reward),
        reward=reward,
        reward_target=reward,
        reward_target_mask=jnp.ones_like(reward),
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
