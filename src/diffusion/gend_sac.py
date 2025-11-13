from functools import partial

from flax.training.train_state import TrainState

from common.type_aliases import RLTrainState
from diffusion.d_sac import DSAC
from typing import ClassVar, Dict, Type
from gymnasium import spaces

import jax
import numpy as np
import jax.numpy as jnp

from diffusion.policies import GenDPOL

from utils import load_state, save_model_state


class GenDSAC(DSAC):
    policy_aliases: ClassVar[Dict[str, Type[GenDPOL]]] = {  # type: ignore[assignment]
        "MlpPolicy": GenDPOL,
        # Minimal dict support using flatten()
        "MultiInputPolicy": GenDPOL,
    }

    policy: GenDPOL
    action_space: spaces.Box  # type: ignore[assignment]

    @staticmethod
    @partial(jax.jit, static_argnames=["crossq_style", "use_bnstats_from_live_net", "diff_init_std", "dim",
                                       "diff_n_steps"])
    def update_critic(
            crossq_style: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            actions: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            key,
            diff_init_std: float,
            dim: int,
            diff_n_steps: int,
    ):
        key, noise_key, dropout_key_target, dropout_key_current, redq_key = jax.random.split(key, 5)
        # sample action from the actor
        all_actions, log_probs, *_ = GenDPOL.sample_action(actor_state, actor_state.params, next_observations,
                                                           noise_key, n_steps=diff_n_steps, a_dim=dim)
        next_state_actions = jax.lax.stop_gradient(all_actions[0])
        next_log_prob = log_probs.sum(axis=1)  # change of variables is already included in the sample function

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        def mse_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params,
                        "batch_stats": qf_state.target_batch_stats if not use_bnstats_from_live_net else batch_stats
                    },
                    next_observations, next_state_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False
                )

                # shape is (n_critics, batch_size, 1)
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    observations, actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([observations, next_observations], axis=0),
                    jnp.concatenate([actions, next_state_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

            if next_q_values.shape[0] > 2:  # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2
                next_q_values = jax.random.choice(redq_key, next_q_values, (m_critics,), replace=False, axis=0)

            next_q_values = jnp.min(next_q_values, axis=0)
            # next_q_values = jnp.max(next_q_values, axis=0)
            # next_q_values = jnp.mean(next_q_values, axis=0)
            next_q_values = next_q_values #- ent_coef_value * next_log_prob.reshape(-1, 1) # TODO: Discuss this!
            target_q_values = rewards.reshape(-1, 1) + (
                    1 - dones.reshape(-1, 1)) * gamma * next_q_values  # shape is (batch_size, 1)

            loss = 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum()

            return loss, (state_updates, current_q_values, next_q_values)

        (qf_loss_value, (state_updates, current_q_values, next_q_values)), grads = \
            jax.value_and_grad(mse_loss, has_aux=True)(qf_state.params, qf_state.batch_stats, dropout_key_current)

        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            'critic_loss': qf_loss_value,
            'ent_coef': ent_coef_value,
            'current_q_values': current_q_values.mean(),
            'next_q_values': next_q_values.mean(),
        }
        return qf_state, metrics, key


    @staticmethod
    @partial(jax.jit, static_argnames=["diff_init_std", "dim", "diff_n_steps", "q_reduce_fn"])
    def update_actor(
            actor_state: TrainState,
            qf_state: RLTrainState,
            ent_coef_state: TrainState,
            observations: np.ndarray,
            key,
            diff_init_std: float,   # don't need this but included for compatibility
            dim: int,  # don't need this but included for compatibility
            diff_n_steps: int,
            q_reduce_fn=jnp.min,  # Changes for redq and droq
            # q_reduce_fn=jnp.mean,  # Changes for redq and droq
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(actor_params):
            out = GenDPOL.sample_action(actor_state, actor_params, observations, noise_key, diff_n_steps, dim)
            all_actions, log_probs, run_costs, sto_costs, init_log_prob_loss, betas = out
            # Squashing/change of variables already in the sample function
            actions = all_actions[0]
            latents = all_actions[1]
            log_probs_sum = log_probs.sum(axis=1)  # for stats
            qf_pi = qf_state.apply_fn(
                {
                    "params": qf_state.params,
                    "batch_stats": qf_state.batch_stats
                },
                observations,
                actions,
                rngs={"dropout": dropout_key}, train=False
            )

            min_qf_pi = q_reduce_fn(qf_pi, axis=0).squeeze()
            # min_qf_pi = (min_qf_pi - min_qf_pi.mean(axis=0, keepdims=True))/min_qf_pi.std(axis=0, keepdims=True)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = (- min_qf_pi + ent_coef_value*(run_costs.squeeze() + sto_costs + init_log_prob_loss)).mean()

            max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
            min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
            mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

            max_log_probs = jnp.max(log_probs, axis=0)
            min_log_probs = jnp.min(log_probs, axis=0)
            mean_log_probs = jnp.mean(log_probs, axis=0)
            latent_log_probs = {'max_ll': max_log_probs, 'min_ll': min_log_probs, 'mean_ll': mean_log_probs}
            latent_acts = {'max_la': max_actions, 'min_la': min_actions, 'mean_la': mean_actions}

            return actor_loss, (-log_probs_sum.mean(), run_costs.mean(), sto_costs.mean(), init_log_prob_loss.mean(),
                                latent_log_probs, latent_acts, betas)

        outs = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        (act_loss_value, (entropy, run_costs, sto_costs, init_log_prob_loss, latent_ll, latent_acts, beta)), grads = outs
        actor_state = actor_state.apply_gradients(grads=grads)
        # for stats
        metrics = {
            "entropy": entropy,
            "run_costs": run_costs,
            "sto_costs": sto_costs,
            "beta_scaler": beta.mean(axis=0)[0].squeeze(),
            "beta_scaler_min": beta[:, 0, 0].min(axis=0).squeeze(),
            "beta_scaler_max": beta[:, 0, 0].max(axis=0).squeeze(),
            "init_log_prob_loss": init_log_prob_loss,
        }
        return actor_state, qf_state, act_loss_value, key, [metrics, latent_ll, latent_acts]

    def _save_model(self):
        save_model_state(self.policy.actor_state, self.model_save_path, "actor_state", self.num_timesteps)
        save_model_state(self.policy.qf_state, self.model_save_path, "critic_state", self.num_timesteps)

    def load_model(self, path, n_steps):
        self.policy.actor_state = load_state(path, "actor_state", n_steps, train_state=self.policy.actor_state)
        self.policy.qf_state = load_state(path, "critic_state", n_steps, train_state=self.policy.qf_state)
