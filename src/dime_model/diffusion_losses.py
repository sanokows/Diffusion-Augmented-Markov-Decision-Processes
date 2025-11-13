import jax.numpy as jnp
import numpy as np
import jax

# from src.dime_model.dime_agent import DimeAgent
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from models.critic import multi_apply_qf


def reverse_KL(actor_state: TrainState,
               actor_params,
               qf_state: RLTrainState,
               ent_coef_state: TrainState,
               observations: np.ndarray,
               key,
               n_env_interacts: int,
               z_atoms: jnp.ndarray,
               sampler,
               q_reduce_fn,
               diffusion_model):
    key, dropout_key, noise_key = jax.random.split(key, 3)
    out = DimeAgent.sample_action(actor_state, actor_params, observations, noise_key, sampler)
    actions, run_costs, sto_costs, terminal_costs, latents, v_t = out
    qf_pi = qf_state.apply_fn(
        {
            "params": qf_state.params,
            "batch_stats": qf_state.batch_stats
        },
        observations,
        actions,
        rngs={"dropout": dropout_key}, train=False
    )

    qf_pi1 = jnp.sum(qf_pi[0] * z_atoms, axis=-1)
    qf_pi2 = jnp.sum(qf_pi[1] * z_atoms, axis=-1)
    min_qf_pi = q_reduce_fn(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
    ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
    actor_loss = (- min_qf_pi + ent_coef_value * (
            run_costs.squeeze() + sto_costs.squeeze() + terminal_costs.squeeze())).mean()

    max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
    min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
    mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

    latent_acts = {'max_la': max_actions, 'min_la': min_actions, 'mean_la': mean_actions}

    return actor_loss, (run_costs.mean(), sto_costs.mean(), terminal_costs.mean(), latent_acts)


def log_variance(actor_state: TrainState,
                 actor_params,
                 qf_state: RLTrainState,
                 ent_coef_state: TrainState,
                 observations: np.ndarray,
                 key,
                 n_env_interacts: int,
                 z_atoms: jnp.ndarray,
                 sampler,
                 q_reduce_fn,
                 diffusion_model):
    key, dropout_key, noise_key = jax.random.split(key, 3)
    observations_multi = np.repeat(observations[:, :, None], 10, axis=2)
    out = DimeAgent.multi_sample(actor_state, actor_params, observations_multi, noise_key, sampler, stop_grad=True)

    actions, run_costs, sto_costs, terminal_costs, latents, v_t = out
    qf_pi = multi_apply_qf(qf_state, observations_multi, actions, dropout_key, train=False)

    qf_pi1 = jnp.sum(qf_pi[0] * z_atoms[None, :, None], axis=1)
    qf_pi2 = jnp.sum(qf_pi[1] * z_atoms[None, :, None], axis=1)
    min_qf_pi = q_reduce_fn(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
    ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
    actor_loss = 0.5 * (- min_qf_pi + ent_coef_value * (run_costs + sto_costs + terminal_costs)).var(axis=1)
    actor_loss = actor_loss.mean()
    latent_acts = {'max_la': 0, 'min_la': 0, 'mean_la': 0}

    return actor_loss, (run_costs.mean(), sto_costs.mean(), terminal_costs.mean(), latent_acts)


def adjoint_matching(actor_state: TrainState,
                     actor_params,
                     qf_state: RLTrainState,
                     ent_coef_state: TrainState,
                     observations: np.ndarray,
                     key,
                     n_env_interacts: int,
                     z_atoms: jnp.ndarray,
                     sampler,
                     q_reduce_fn,
                     diffusion_model):
    key, dropout_key, noise_key = jax.random.split(key, 3)
    out = DimeAgent.sample_action(actor_state, actor_params, observations, noise_key, sampler, stop_grad=True)
    actions, run_costs, sto_costs, terminal_costs, latents, ctrls = out

    Q_fn_1 = lambda a, s: jnp.sum(qf_state.apply_fn(
        {
            "params": qf_state.params,
            "batch_stats": qf_state.batch_stats
        },
        s,
        a,
        rngs={"dropout": dropout_key}, train=False
    )[0] * z_atoms, axis=-1)

    Q_fn_2 = lambda a, s: jnp.sum(qf_state.apply_fn(
        {
            "params": qf_state.params,
            "batch_stats": qf_state.batch_stats
        },
        s,
        a,
        rngs={"dropout": dropout_key}, train=False
    )[1] * z_atoms, axis=-1)

    Q_fn = lambda a, s: q_reduce_fn(jnp.stack([Q_fn_1(a, s), Q_fn_2(a, s)], axis=0), axis=0).squeeze()

    nabla_Q = jax.vmap(jax.grad(Q_fn))(actions, observations)

    noise_bwd_int = jnp.exp(
        -jax.vmap(diffusion_model.backward_integrated_scheduler)(jnp.arange(ctrls.shape[1]))[:, None])
    scaled_adj_states = noise_bwd_int * jnp.repeat(jnp.expand_dims(nabla_Q, 1), noise_bwd_int.shape[0],
                                                   axis=1)

    ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params}, n_env_interacts)
    scale = 1 / ent_coef_value
    actor_loss = 0.5 * ((ctrls - scale * scaled_adj_states) ** 2).sum(-1).mean()

    max_actions = jnp.max(jnp.max(latents, axis=0), axis=1)
    min_actions = jnp.min(jnp.min(latents, axis=0), axis=1)
    mean_actions = jnp.mean(jnp.mean(latents, axis=0), axis=1)

    latent_acts = {'max_la': max_actions, 'min_la': min_actions, 'mean_la': mean_actions}

    return actor_loss, (run_costs.mean(), sto_costs.mean(), terminal_costs.mean(), latent_acts)
