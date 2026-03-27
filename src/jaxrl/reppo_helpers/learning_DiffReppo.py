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


def compute_action_q_grads(actor_model, critic_model, obs, critic_obs):
    """Compute time-interpolated gradient between prior and Q wrt actions."""
    actions = obs["orig_actions"]
    # Keep actions within arctanh domain to avoid inf/NaN in prior terms.
    clip_limit = jnp.arctanh(jnp.asarray(0.999, dtype=actions.dtype))
    actions = jnp.clip(actions, -clip_limit, clip_limit)
    steps = obs["diff_time_step"][..., 0]
    diff_steps = max(actor_model.diffusion_model.diff_steps - 1, 1)
    time_norm = jnp.clip(steps.astype(jnp.float32) / diff_steps, 0.0, 1.0)

    def _q_grad(single_critic_obs, act):
        q_fn = lambda a: critic_model.critic(single_critic_obs, a).sum()
        return jax.grad(q_fn)(act)

    q_grad = jax.vmap(_q_grad)(critic_obs, actions)
    def _prior_grad(act):
        return jax.grad(
            lambda a: actor_model.diffusion_model.prior_log_prob(a)
        )(act)

    prior_grad = jax.vmap(_prior_grad)(actions)
    if time_norm.ndim < q_grad.ndim:
        time_norm = jnp.reshape(
            time_norm,
            time_norm.shape + (1,) * (q_grad.ndim - time_norm.ndim),
        )
    #jax.debug.print("Time norm shape: {shape}, q_grad shape: {shape2}", shape=time_norm.shape, shape2 = q_grad.shape)
    #blended_grad = (1.0 - time_norm) * prior_grad - time_norm * q_grad
    blended_grad = - time_norm*q_grad
    return blended_grad


def maybe_add_q_grad(obs, critic_obs, actor_model, critic_model, use_langevin: bool):
    """Attach blended prior/Q gradients to the observation dict when enabled."""
    if not use_langevin:
        return obs
    q_grad = compute_action_q_grads(actor_model, critic_model, obs, critic_obs)
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

        _, pred, pred_rew, pred_next_diff_state, value = critic_model.forward(
            minibatch.critic_obs, minibatch.action
        )
        aux_loss = (1.0 - minibatch.truncated.reshape(-1, 1)) * optax.squared_error(pred, minibatch.next_state_emb)
        aux_next_diff_loss = (1.0 - minibatch.truncated.reshape(-1, 1)) * optax.squared_error(
            pred_next_diff_state, minibatch.next_emb
        )
        aux_rew_loss = (1.0 - minibatch.truncated.reshape(-1, 1)) * optax.squared_error(
            pred_rew, minibatch.reward.reshape(-1, 1)
        )

        diff_steps = jnp.asarray(
            cfg.diffusion.diff_steps - 1,
            dtype=minibatch.obs["diff_time_step"].dtype,
        )
        is_last_step = (minibatch.obs["diff_time_step"][..., 0] == diff_steps).reshape(-1, 1)
        aux_weight = is_last_step.astype(aux_loss.dtype)
        # jax.debug.print("aux_weight value: {value}, sum: {sum}", value=aux_weight, sum=jnp.sum(aux_weight))
        # jax.debug.print("diff_time_step: {value}", value=minibatch.obs["diff_time_step"][..., 0])
        
        masked_aux_terms = jnp.concatenate([aux_loss, aux_rew_loss], axis=-1)
        masked_aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1)) * aux_weight * masked_aux_terms,
            axis=-1,
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
        critic_loss = optax.squared_error(value, target_vals)
        critic_loss = jnp.mean(critic_loss)
        loss = jnp.mean(
            (1.0 - minibatch.truncated)
            * (critic_update_loss ) + cfg.aux_loss_mult * aux_loss
        )
        critic_pnorm = utils.tree_norm(params)
        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=aux_rew_loss * aux_weight.astype(aux_rew_loss.dtype),
            q=value.mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_vals.mean(),
            critic_pnorm=critic_pnorm,
        )



def actor_loss_fn(params, updated_state, critic_rollout_model, step_key, minibatch, target_vals, action_size_target, cfg, actor_target_model):
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
            minibatch.obs, minibatch.critic_obs, actor_model, critic_current_model, use_langevin
        )
        obs_for_target = maybe_add_q_grad(
            minibatch.obs, minibatch.critic_obs, actor_model, critic_for_target, use_langevin
        )
        pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
            obs_for_actions, step_key
        )
        entropy_prior = actor_model.get_prior_entropy()
        log_prob_ratio = gen_log_prob - dest_log_prob
        value = critic_current_model.critic(minibatch.critic_obs, pred_action)
    
        #print the shape of log_prob_ratio
        #jax.debug.print("log_prob_ratio shape: {shape}", shape=log_prob_ratio.shape)
        entropy = -cfg.diffusion.diff_steps * jnp.mean(log_prob_ratio, axis=0)
        entropy = jax.lax.stop_gradient(entropy)
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

        clip_ratio = jnp.mean((kl >= cfg.kl_bound).astype(jnp.float32))

        target_entropy = action_size_target + entropy
        kl_constraint = kl - cfg.kl_bound
        if cfg.use_augmented_lagrangian_dual:
            if cfg.actor_kl_clip_mode == "full":
                raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")
            elif cfg.actor_kl_clip_mode == "clipped":
                actor_loss_val = jnp.where(
                    kl < cfg.kl_bound,
                    log_prob_ratio * jax.lax.stop_gradient(temperature) - value + 0.5* cfg.augmented_lagrangian_entropy_coef* jnp.square(target_entropy),
                    kl * jax.lax.stop_gradient(lagrangian) * + 0.5 * cfg.augmented_lagrangian_kl_coef * jnp.square(kl_constraint),
                )
            elif cfg.actor_kl_clip_mode == "value":
                raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")
            else:
                raise ValueError(f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}")

            target_entropy_loss = (
                temperature * 0.5
                * cfg.augmented_lagrangian_entropy_coef
                * jax.lax.stop_gradient(target_entropy)
            ).mean()
            lagrangian_loss = (
                -lagrangian*0.5
                * cfg.augmented_lagrangian_kl_coef
                * jax.lax.stop_gradient(kl_constraint)
                + 0.5
                * cfg.augmented_lagrangian_kl_coef
                * jnp.square(kl_constraint)
            ).mean()
        else:
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
            ).mean()
            lagrangian_loss = (
                -lagrangian
                * jax.lax.stop_gradient(kl_constraint)
            ).mean()

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
            temp=temperature,
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean() * cfg.diffusion.diff_steps,
            energy_mean = -minibatch.reward.mean() * cfg.diffusion.diff_steps + 1,
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            run_cost=0.0,
            sto_cost=0.0,
            terminal_cost=0.0,
            entropy=entropy,
            entropy_target=-action_size_target,
            entropy_loss=target_entropy_loss,
            kl_clip_ratio=clip_ratio,
            target_values=target_vals.mean(),
            actor_pnorm=actor_pnorm,
            friction=friction_detached.mean(),
            entropy_prior=entropy_prior,
        )


def actor_WPO_loss_fn(params, updated_state, critic_rollout_model, step_key, minibatch, target_vals, action_size_target, cfg, actor_target_model):
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
        obs_for_actions = maybe_add_q_grad(
            minibatch.obs, minibatch.critic_obs, actor_model_raw, critic_for_actions, use_langevin
        )
        obs_for_target = maybe_add_q_grad(
            minibatch.obs, minibatch.critic_obs, actor_model_raw, critic_rollout_model, use_langevin
        )

        stop_grad_params = jax.tree.map(jax.lax.stop_gradient, params)
        batch_size = minibatch.action.shape[0]
        fisher_keys = jax.random.split(step_key, batch_size)

        ### TODO shoudl actually be done seperatly for each state
        def _single_gen_log_prob(p, obs, key):
            actor_single = nnx.merge(updated_state.actor.graphdef, p)
            obs_batched = jax.tree_util.tree_map(lambda x: x[None], obs)
            jax.debug.print("train_update_step_4_env shape: {shape}", shape=key.shape)
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
        temperature = _resolve_temperature(actor_model, cfg, updated_state)

        jax.debug.print("train_update_step_5_env shape: {shape}", shape=step_key.shape)
        pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
            obs_for_actions, step_key
        )
        entropy_prior = actor_model.get_prior_entropy()
        log_prob_ratio = gen_log_prob - dest_log_prob
        #print the shape of log_prob_ratio
        #jax.debug.print("log_prob_ratio shape: {shape}", shape=log_prob_ratio.shape)
        entropy = -cfg.diffusion.diff_steps * jnp.mean(log_prob_ratio, axis=0)
        entropy = jax.lax.stop_gradient(entropy)

        stop_pred_action = jax.lax.stop_gradient(pred_action)

        def single_q(obs, act):
            batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
            return jnp.squeeze(critic_current_model.critic(batched_obs, act[None]), axis=0)

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
            def target_single_log_probs(obs, act):
                batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
                ### should new states be sampled here?
                gen_lp_old, _ = actor_target_model.vmap_eval_log_prob(batched_obs, act[None])
                return jnp.squeeze(gen_lp_old, axis=0)


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

        actor_WPO_loss = temperature*jnp.mean((jax.lax.stop_gradient(log_prob_action_grad - stop_q_action_grad/temperature)**2).sum(axis=-1))

        clip_ratio = jnp.mean((kl_clip_value >= cfg.kl_bound).astype(jnp.float32))
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

        target_entropy = action_size_target + entropy
        kl_constraint = kl_clip_value - cfg.kl_bound

        target_entropy_loss = (
            temperature * jax.lax.stop_gradient(target_entropy)
        ).mean()
        lagrangian_loss = (
            -lagrangian * jax.lax.stop_gradient(kl_constraint)
        ).mean()
        entropy_penalty = jnp.array(0.0)
        kl_penalty = jnp.array(0.0)

        loss = jnp.mean(actor_loss_val)
        if cfg.update_entropy_lagrangian:
            loss += target_entropy_loss
        if cfg.update_kl_lagrangian:
            loss += lagrangian_loss

        actor_pnorm = utils.tree_norm(params)
        friction = actor_model.diffusion_model.friction.value
        friction_detached = jax.lax.stop_gradient(friction)
        metrics = dict(
            actor_loss=actor_loss_val,
            actor_WPO_loss=actor_WPO_loss,
            loss=loss,
            temp=temperature,
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean(),
            reward_mean=minibatch.reward.mean() * cfg.diffusion.diff_steps,
            energy_mean=-minibatch.reward.mean() * cfg.diffusion.diff_steps + 1,
            kl=kl_clip_value.mean(),
            kl_clip_value=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            run_cost=0.0,
            sto_cost=0.0,
            terminal_cost=0.0,
            entropy=entropy,
            entropy_loss=target_entropy_loss,
            entropy_penalty=entropy_penalty,
            kl_penalty=kl_penalty,
            kl_clip_ratio=clip_ratio,
            target_values=target_vals.mean(),
            actor_pnorm=actor_pnorm,
            friction=friction_detached.mean(),
            entropy_prior=entropy_prior,
        )
        return loss, metrics

def train_step_env(Transition, cfg, env, actor_model, critic_model, carry, _):
    key, env_state, inner_state, obs, critic_obs = carry
    use_langevin = bool(cfg.diffusion.score_model.langevin_param)
    key, act_key, step_key = jax.random.split(key, 3)
    step_key = jax.random.split(step_key, cfg.num_envs)
    obs_for_actor = maybe_add_q_grad(obs, critic_obs, actor_model, critic_model, use_langevin)

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
        next_obs, next_critic_obs, actor_model, critic_model, use_langevin
    )

    jax.debug.print("train_update_step_2_env shape: {shape}", shape=next_act_key.shape)
    next_action, next_gen_log_prob, next_dest_log_prob = (
        actor_model.vmap_sample_next_step(next_obs_for_actor, next_act_key)
    )
    next_action = jax.lax.stop_gradient(next_action)
    next_emb, _, _, _, value = critic_model.forward(next_critic_obs, next_action)
    log_ratio = jax.lax.stop_gradient(
        next_gen_log_prob - next_dest_log_prob
    )
    temperature = _resolve_temperature(actor_model, cfg, inner_state)
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
