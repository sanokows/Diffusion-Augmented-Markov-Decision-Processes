import jax
import jax.numpy as jnp

from diffusion.common.utils import init_dt
from diffusion.common.utils import inverse_softplus
from diffusion.common.diffusion_models import DiffusionModel
from diffusion.common.init_diffusion_model import init_underdamped, init_langevin, init_model


def init_gbs(key, cfg, dim, obs_dim, target=None):

    params = {'params': {'betas': jnp.ones((cfg.alg.actor.diff_steps,)),
                         'prior_mean': jnp.zeros((dim,)),
                         'prior_std': jnp.ones((dim,)) * inverse_softplus(cfg.sampler.init_std),
                         'mass_std': jnp.ones(1) * inverse_softplus(1.),
                         'dt': init_dt(cfg),
                         'friction': jnp.ones(dim) * inverse_softplus(cfg.sampler.friction) if cfg.per_dim_friction else jnp.ones(1) * inverse_softplus(cfg.sampler.friction),
                         }}

    prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn = init_underdamped(cfg, dim)
    if target is not None:
        langevin_fn = init_langevin(cfg, prior_log_prob, target.log_prob)

    def forward_model(step, x, obs, model_state, params, aux):
        langevin_vals = aux
        return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, step,
                                       jax.lax.stop_gradient(langevin_vals))

    def backward_model(step, x, obs, model_state, params, aux):
        langevin_vals = aux
        model_params = jax.lax.stop_gradient(params['params']['bwd_params']) if cfg.use_path_gradient else params['params']['bwd_params']
        return model_state.apply_fn[1](model_params, x, obs, step, jax.lax.stop_gradient(langevin_vals))

    def drift_fn(step, x, params):
        if cfg.sampler.drift == 'anneal':
            return langevin_fn(step, x, params)
        elif cfg.sampler.drift == 'zero':
            return jnp.zeros_like(x), None
        elif cfg.sampler.drift == 'target':
            target_score = jax.grad(lambda x: jnp.squeeze(target.log_prob(x)))(x)
            return target_score, None
        else:
            raise ValueError(f'Unknown drift {cfg.sampler.drift}')

    key, key_gen = jax.random.split(key)
    model_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=True)

    return DiffusionModel(num_steps=cfg.alg.actor.diff_steps,
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          delta_t_fn=delta_t_fn,
                          friction_fn=friction_fn,
                          mass_fn=mass_fn,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob,
                          ), model_state
