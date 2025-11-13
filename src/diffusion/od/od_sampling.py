import jax
import distrax
import jax.numpy as jnp


def single_sample(seed, model_state, params, obs, integrator, diffusion_model, stop_grad=False, ode=False, ode_coef=0.5):
    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(params, key, 1)
    key, key_gen = jax.random.split(key_gen)
    init_x = jnp.squeeze(init_x, 0)
    if stop_grad:
        init_x = jax.lax.stop_gradient(init_x)
    key, key_gen = jax.random.split(key_gen)
    aux = (init_x, jnp.zeros(1), key)
    integrate = integrator(model_state, params, obs, stop_grad, ode, ode_coef)
    aux, per_step_output = jax.lax.scan(integrate, aux, jnp.arange(0, diffusion_model.num_steps))
    final_x, log_ratio, _ = aux

    terminal_costs = diffusion_model.prior_log_prob(init_x, params)
    running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
    # running_cost = -log_ratio

    final_x = distrax.Tanh().forward(final_x)
    x_t = per_step_output
    x_t = jnp.concatenate([jnp.expand_dims(init_x, 0), x_t])
    x_t = x_t.at[-1].set(final_x)
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape), x_t, None


def sample(key, model_state, params, obs, integrator, diffusion_model, stop_grad=False, ode=False, ode_coef=0.5):
    keys = jax.random.split(key, num=obs.shape[0])
    in_tuple = (keys, model_state, params, obs, integrator, diffusion_model, stop_grad, ode, ode_coef)
    in_axes = (0, None, None, 0, None, None, None, None, None)
    rnd_result = jax.vmap(single_sample, in_axes=in_axes)(*in_tuple)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t, _ = rnd_result

    return x_0, running_costs, stochastic_costs, terminal_costs, x_t, None

def single_kl_div(seed, model_state, old_model_state, params, old_params, obs, logratio, diffusion_model, stop_grad=False):
    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(params, key, 1)
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


# This is going to be taken over by the actor update
# def neg_elbo(key, model_state, params, integrator, diffusion_model, batch_size):
#     rnd_result = rnd(key, model_state, params, integrator, diffusion_model, batch_size, stop_grad=False)
#     samples, running_costs, stochastic_costs, terminal_costs, x_t, _ = rnd_result
#     neg_elbo = running_costs + terminal_costs
#     return jnp.mean(neg_elbo)
#
#
# def log_var(key, model_state, params, integrator, diffusion_model, batch_size):
#     rnd_result = rnd(key, model_state, params, integrator, diffusion_model, batch_size, stop_grad=True)
#     samples, running_costs, stochastic_costs, terminal_costs, x_t, _ = rnd_result
#     rnds = running_costs + terminal_costs + stochastic_costs
#     return 0.5 * rnds.var()