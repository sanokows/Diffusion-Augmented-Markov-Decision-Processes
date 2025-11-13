import distrax
import jax.numpy as jnp
import jax
import optax
from flax.training import train_state
from jax._src.nn.functions import softplus

from src.dime_model.learning_rate_scheduler import get_learning_rate_scheduler
from src.dime_model.models.control_net import ControlNetwork


def init_dm(cfg, dim):
    diffusion_cfg = cfg.diffusion_type

    def prior_sampler(key, n_samples):
        samples = distrax.MultivariateNormalDiag(jnp.zeros((dim,)),
                                                 jnp.ones((dim,)) * diffusion_cfg.init_std).sample(seed=key,
                                                                                                   sample_shape=(n_samples,))
        return samples

    def prior_log_prob(x):
        log_probs = distrax.MultivariateNormalDiag(jnp.zeros(dim), 
                                                   jnp.ones(dim) * diffusion_cfg.init_std).log_prob(x)
        return log_probs

    scheduler = diffusion_cfg.diff_coef_schedule
    diff_coef_schedule = scheduler[0]

    def diffusion_coef(step, params):
        diff_coef = params['params']['diffusion_coef'] if diffusion_cfg.learn_diff_coef else jax.lax.stop_gradient(params['params']['diffusion_coef'])
        return diff_coef_schedule(step) * jax.nn.softplus(diff_coef)

    return prior_log_prob, prior_sampler, diffusion_coef, scheduler


def init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=True):
    # Define the model
    diffusion_cfg = cfg.diffusion_type
    in_dim = dim

    key, key_gen = jax.random.split(key)
    if learn_forward:
        fwd_model = ControlNetwork(dim=dim, **diffusion_cfg.control_net)
        fwd_params = fwd_model.init(key, 
                                    jnp.ones([cfg.batch_size, in_dim]),
                                    jnp.ones(([cfg.batch_size, obs_dim])),
                                    jnp.ones([cfg.batch_size, 1]))
        params['params']['fwd_params'] = fwd_params
        fwd_apply_fn = fwd_model.apply
    else:
        fwd_apply_fn = None

    key, key_gen = jax.random.split(key_gen)
    if learn_backward:
        bwd_model = ControlNetwork(dim=dim, **diffusion_cfg.control_net)
        bwd_params = bwd_model.init(key, 
                                    jnp.ones([cfg.batch_size, in_dim]),
                                    jnp.ones(([cfg.batch_size, obs_dim])),
                                    jnp.ones([cfg.batch_size, 1]))
        params['params']['bwd_params'] = bwd_params
        bwd_apply_fn = bwd_model.apply
    else:
        bwd_apply_fn = None

    if cfg.use_step_size_scheduler:
        model_opt = optax.adam(get_learning_rate_scheduler(cfg, cfg.optimizer.lr_actor), b1=cfg.optimizer.b1)

    else:
        model_opt = optax.adam(cfg.optimizer.lr_actor, b1=cfg.optimizer.b1)

    if cfg.optimizer.do_actor_grad_clip:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(cfg.optimizer.actor_grad_clip),
                                model_opt)
    else:
        optimizer = optax.chain(optax.zero_nans(), model_opt)

    model_state = train_state.TrainState.create(apply_fn=(fwd_apply_fn, bwd_apply_fn), params=params, tx=optimizer)

    fwd_model.apply = jax.jit(  # type: ignore[method-assign]
        fwd_model.apply
    )

    return model_state
