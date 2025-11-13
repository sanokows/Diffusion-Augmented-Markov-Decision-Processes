import jax
import distrax
import jax.numpy as jnp

from src.dime_model.utils import log_prob_kernel, sample_kernel, check_stop_grad


def single_sample(seed, model_state, params, obs, integrator, diffusion_model, stop_grad=False, ode=False, ode_coef=0.5):
    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(key, 1)
    key, key_gen = jax.random.split(key_gen)
    init_x = jnp.squeeze(init_x, 0)
    if stop_grad:
        init_x = jax.lax.stop_gradient(init_x)
    key, key_gen = jax.random.split(key_gen)
    aux = (init_x, jnp.zeros(1), key)
    integrate = integrator(model_state, params, obs, stop_grad, ode, ode_coef)
    aux, per_step_output = jax.lax.scan(integrate, aux, jnp.arange(0, diffusion_model.num_steps))
    final_x, log_ratio, _ = aux

    terminal_costs = diffusion_model.prior_log_prob(init_x)
    running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
    # running_cost = -log_ratio

    final_x = distrax.Tanh().forward(final_x)
    x_t, ctrls = per_step_output
    x_t = jnp.concatenate([jnp.expand_dims(init_x, 0), x_t])
    x_t = x_t.at[-1].set(final_x)
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape), x_t, ctrls


def sample(key, model_state, params, obs, integrator, diffusion_model, stop_grad=False, ode=False, ode_coef=0.5):
    keys = jax.random.split(key, num=obs.shape[0])
    in_tuple = (keys, model_state, params, obs, integrator, diffusion_model, stop_grad, ode, ode_coef)
    in_axes = (0, None, None, 0, None, None, None, None, None)
    rnd_result = jax.vmap(single_sample, in_axes=in_axes)(*in_tuple)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t, ctrls = rnd_result

    return x_0, running_costs, stochastic_costs, terminal_costs, x_t, ctrls

def get_integrator(cfg, diffusion_model):
    def integrator(model_state, params, obs, stop_grad=False, ode=False, ode_coef=0.5):
        dt = 1. / cfg.diffusion_type.diff_steps

        def integrate_EM(state, per_step_input):
            x, log_w, key_gen = state

            step = per_step_input
            # step = jnp.ones((x.shape[0], 1), dtype=jnp.float32) * per_step_input.astype(jnp.float32)

            # Compute SDE components
            # diff_coef = diffusion_model.diffusion_coef(step, params)
            # diff_coef_sq = diff_coef ** 2
            # brownian_scale = jnp.sqrt(2 * dt) * diff_coef
            diff_coef_sq = diffusion_model.diffusion_coef(step, params)
            brownian_scale = jnp.sqrt(2 * dt * diff_coef_sq)

            # Forward kernel: denoising process
            ctrl = -x + ode_coef * diffusion_model.forward_model(step, x, obs, model_state, params) if ode else -x + diffusion_model.forward_model(step, x, obs, model_state, params)
            fwd_mean = x + dt * diff_coef_sq * ctrl
            key, key_gen = jax.random.split(key_gen)
            x_new = fwd_mean if ode else sample_kernel(key, fwd_mean, brownian_scale)
            x_new = check_stop_grad(x_new, stop_grad)

            # Backward kernel: noising process
            bwd_mean = x_new + dt * diff_coef_sq * (-x_new)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(x_new, fwd_mean, brownian_scale)
            bwd_log_prob = log_prob_kernel(x, bwd_mean, brownian_scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, log_w, key_gen)
            per_step_output = (x_new, ctrl)
            return next_state, per_step_output

        if cfg.diffusion_type.integrator == 'EM':
            integrate = integrate_EM
        else:
            raise ValueError(f'No integrator named {cfg.diffusion_type.integrator}.')

        return integrate
    return integrator


def get_logratio(cfg, diffusion_model):
    def logratio(model_state, old_model_state, params, old_params, obs, stop_grad=False):
        dt = 1. / cfg.diffusion_type.diff_steps
        def per_sample_logratio(state, per_step_input):
            x, log_w, key_gen = state

            step = per_step_input
            # step = jnp.ones((x.shape[0], 1), dtype=jnp.float32) * per_step_input.astype(jnp.float32)

            # Compute SDE components
            diff_coef_sq = diffusion_model.diffusion_coef(step, params)
            brownian_scale = jnp.sqrt(2 * dt * diff_coef_sq)

            # Forward kernel
            ctrl = -2 * x + diffusion_model.forward_model(step, x, obs, model_state, params)
            old_ctrl = -2 * x + diffusion_model.forward_model(step, x, obs, old_model_state, old_params)

            fwd_mean = x + dt * diff_coef_sq * (x + ctrl)
            old_fwd_mean = x + dt * diff_coef_sq * (x + old_ctrl)

            key, key_gen = jax.random.split(key_gen)
            x_new = sample_kernel(key, old_fwd_mean, brownian_scale)
            x_new = check_stop_grad(x_new, stop_grad)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(x_new, fwd_mean, brownian_scale)
            old_fwd_log_prob = log_prob_kernel(x_new, old_fwd_mean, brownian_scale)

            # Update weight and return
            log_w += old_fwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, log_w, key_gen)
            per_step_output = (x_new, ctrl)
            return next_state, per_step_output
        return per_sample_logratio
    return logratio

def single_kl_div(seed, model_state, old_model_state, params, old_params, obs, logratio, diffusion_model, stop_grad=False):
    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(key, 1)
    key, key_gen = jax.random.split(key_gen)
    init_x = jnp.squeeze(init_x, 0)
    if stop_grad:
        init_x = jax.lax.stop_gradient(init_x)
    key, key_gen = jax.random.split(key_gen)
    aux = (init_x, jnp.zeros(1), key)
    integrate = logratio(model_state, old_model_state, params, old_params, obs, stop_grad)
    aux, per_step_output = jax.lax.scan(integrate, aux, jnp.arange(0, diffusion_model.num_steps))
    final_x, log_ratio, _ = aux
    final_x = distrax.Tanh().forward(final_x)
    return final_x, log_ratio

def kl_div(key, model_state, old_model_state, params, old_params, obs, logratio, diffusion_model, stop_grad=False):
    keys = jax.random.split(key, num=obs.shape[0])
    in_tuple = (keys, model_state, old_model_state, params, old_params, obs, logratio, diffusion_model, stop_grad)
    in_axes = (0, None, None, None, None, 0, None, None, None)
    x_0, log_ratios = jax.vmap(single_kl_div, in_axes=in_axes)(*in_tuple)

    return x_0, log_ratios