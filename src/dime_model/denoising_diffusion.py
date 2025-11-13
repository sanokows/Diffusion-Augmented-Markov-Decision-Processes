from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp

from src.dime_model.utils import inverse_softplus
from src.dime_model.init_diffusion_model import init_model, init_dm


class DiffusionModel(NamedTuple):
    num_steps: int
    forward_model: Callable
    backward_model: Callable
    drift_fn: Callable
    diffusion_coef: Callable
    prior_sampler: Callable
    prior_log_prob: Callable
    backward_integrated_scheduler: Callable


def init_denoising_diffusion_model_state(key, cfg, dim, obs_dim):
    params = {'params': {'diffusion_coef': jnp.ones(dim) * inverse_softplus(cfg.diffusion_type.diff_coef)}}
    key, key_gen = jax.random.split(key)
    model_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=False)
    return model_state

def init_denoising_diffusion(cfg, dim):
    prior_log_prob, prior_sampler, diffusion_coef, scheduler = init_dm(cfg, dim)

    def forward_model(step, x, obs, model_state, params):
        return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, step)

    def backward_model(step, x, obs, model_state, params):
        return jnp.zeros_like(x)

    def drift_fn(step, x, params):
        return jax.grad(prior_log_prob)(x)

    return DiffusionModel(num_steps=cfg.diffusion_type.diff_steps,
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          diffusion_coef=diffusion_coef,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob,
                          backward_integrated_scheduler=scheduler[2])
