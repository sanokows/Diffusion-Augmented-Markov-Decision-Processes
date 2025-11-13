import jax
import jax.numpy as jnp

from src.diffusion.common.utils import init_dt
from src.diffusion.common.utils import inverse_softplus
from src.diffusion.common.diffusion_models import DiffusionModel
from src.diffusion.common.init_diffusion_model import init_underdamped, init_langevin, init_model


def init_dis(key, cfg, dim, obs_dim, target=None):

    params = {'params': {'betas': jnp.ones((cfg.alg.actor.diff_steps,)),
                         'prior_mean': jnp.zeros((dim,)),
                         'prior_std': jnp.ones((dim,)) * inverse_softplus(cfg.sampler.init_std),
                         'mass_std': jnp.ones(1) * inverse_softplus(1.),
                         'dt': init_dt(cfg),
                         'friction': jnp.ones(dim) * inverse_softplus(cfg.sampler.friction) if cfg.per_dim_friction else jnp.ones(1) * inverse_softplus(cfg.sampler.friction)}}

    prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn = init_underdamped(cfg, dim)
    if target is not None:
        langevin_fn = init_langevin(cfg, prior_log_prob, target.log_prob)

    def forward_model(step, x, obs, model_state, params, aux):
        langevin_vals = aux
        return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, step,
                                       jax.lax.stop_gradient(langevin_vals))

    def backward_model(step, x, obs, model_state, params, aux):
        return jnp.zeros_like(x)

    def drift_fn(step, x, params):
        if target is not None:
            if cfg.sampler.use_target_score:
                _, aux = langevin_fn(step, x, params)
            else:
                aux = None
        else:
            aux = None

        return jax.grad(prior_log_prob)(x, params), aux

    key, key_gen = jax.random.split(key)
    model_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=False)

    return DiffusionModel(num_steps=cfg.sampler.diff_steps,
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          delta_t_fn=delta_t_fn,
                          friction_fn=friction_fn,
                          mass_fn=mass_fn,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob,
                          ), model_state


def init_denoising_diffusion_model_state(key, cfg, dim, obs_dim):
    params = {
                'params': 
                {
                    'betas': jnp.ones((cfg.sampler.diff_steps,)),
                    'prior_mean': jnp.zeros((dim,)),
                    'prior_std': jnp.ones((dim,)) * inverse_softplus(cfg.sampler.init_std),
                    'mass_std': jnp.ones(1) * inverse_softplus(1.),
                    'dt': init_dt(cfg),
                    'friction': jnp.ones(dim) * inverse_softplus(cfg.sampler.friction) if cfg.per_dim_friction else jnp.ones(1) * inverse_softplus(cfg.sampler.friction) # diffusion_coef
                }
            }
    key, key_gen = jax.random.split(key)
    model_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=False)
    return model_state

def init_denoising_diffusion(cfg, dim, target=None):
    prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn = init_underdamped(cfg, dim)
    if target is not None:
        langevin_fn = init_langevin(cfg, prior_log_prob, target.log_prob)

    def forward_model(step, x, obs, model_state, params, aux):
        langevin_vals = aux
        return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, step,
                                       jax.lax.stop_gradient(langevin_vals))

    def backward_model(step, x, obs, model_state, params, aux):
        return jnp.zeros_like(x)

    def drift_fn(step, x, params):
        if target is not None:
            if cfg.sampler.use_target_score:
                _, aux = langevin_fn(step, x, params)
            else:
                aux = None
        else:
            aux = None

        return jax.grad(prior_log_prob)(x, params), aux

    return DiffusionModel(num_steps=cfg.sampler.diff_steps,
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          delta_t_fn=delta_t_fn,
                          friction_fn=friction_fn,
                          mass_fn=mass_fn,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob)