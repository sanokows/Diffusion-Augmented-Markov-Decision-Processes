import math
from typing import Sequence, Union

from functools import partial

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from src.jaxrl import utils
from src.networks.diffusion.utils import inverse_softplus, sample_kernel, log_prob_kernel, check_stop_grad


def integrate_one_step(diffusion_model, curr_x , step, obs, key, stop_grad=False):
    step = step.astype(jnp.float32)
    x = curr_x
    key_gen = key

    # Compute SDE components
    mu, scale, eta = diffusion_model.compute_diffusion_stuff(step, x, obs, model=diffusion_model.forward_model)

    # Forward kernel
    drift = diffusion_model.drift_fn(step, x)
    # fwd_mean = x + eta * (drift + (ode_coef * diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + diffusion_model.forward_model(step, x, obs))
    fwd_mean = x + eta * mu

    key, key_gen = jax.random.split(key_gen)
    # x_new = fwd_mean if ode else sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
    x_new = sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
    
    obs_new = dict(obs)
    obs_new['diff_time_step'] = obs_new['diff_time_step'] + 1.0
    mu_bwd, scale_new, eta_new = diffusion_model.compute_diffusion_stuff(step+1, x, obs_new, model=diffusion_model.backward_model)

    bwd_mean = x_new + eta_new * mu_bwd

    # print scale new and scale
    #jax.debug.print("step: {s}, scale: {sc}, scale_new: {sn}", s=step, sc=scale, sn=scale_new)
    # Evaluate kernels
    fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
    bwd_log_prob = log_prob_kernel(x, bwd_mean, scale_new)

    # log_w = bwd_log_prob - fwd_log_prob
    # jax.debug.print("step: {s}, log_w before: {lw}, bwd_log_prob: {bp}, fwd_log_prob: {fp}", s=step, lw=log_w, bp=bwd_log_prob, fp=fwd_log_prob)

    key, key_gen = jax.random.split(key_gen)
    out_dict = {
        "gen_log_prob": fwd_log_prob, 
        "dest_log_prob": bwd_log_prob,
        "x_new": x_new,
    }
    return out_dict, key_gen

def ODE_integrate_one_step(diffusion_model, curr_x , step, obs, key, stop_grad=False):
    step = step.astype(jnp.float32)
    x = curr_x
    key_gen = key

    # Compute SDE components
    mu, scale, eta = diffusion_model.compute_diffusion_stuff(step, x, obs, model=diffusion_model.forward_model, ode_coeff= 0.5, train_mode = False)
    # Forward kernel
    x_new = x + eta * mu

    out_dict = {
        "x_new": x_new,
    }
    return out_dict, key_gen

def evaluate_one_step_log_prob(diffusion_model, curr_x , step, obs, actions, stop_grad=False):
    step = step.astype(jnp.float32)
    x = curr_x

    # Compute SDE components
    mu, scale, eta = diffusion_model.compute_diffusion_stuff(step, x, obs, model=diffusion_model.forward_model)

    # Forward kernel
    # fwd_mean = x + eta * (drift + (ode_coef * diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + diffusion_model.forward_model(step, x, obs))
    fwd_mean = x + eta * mu
    # x_new = fwd_mean if ode else sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
    x_new = actions

    # Backward kernel
    obs_new = dict(obs)
    obs_new['diff_time_step'] = obs_new['diff_time_step'] + 1.0
    mu_bwd, scale_new, eta_new = diffusion_model.compute_diffusion_stuff(step+1, x, obs_new, model=diffusion_model.backward_model)

    bwd_mean = x_new + eta_new * mu_bwd

    # Evaluate kernels
    fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
    bwd_log_prob = log_prob_kernel(x, bwd_mean, scale_new) 

    out_dict = {
        "gen_log_prob": fwd_log_prob, 
        "dest_log_prob": bwd_log_prob,
    }
    return out_dict

def logratio_one_step(diffusion_model, target_diffusion_model, curr_x , step, obs, key, stop_grad=True, kl_action_rep=1, target_obs=None):
    step = step.astype(jnp.float32)
    x = curr_x
    key_gen = key

    target_obs = obs if target_obs is None else target_obs

    # Compute SDE components
    mu, scale, eta = diffusion_model.compute_diffusion_stuff(step, x, obs, model=diffusion_model.forward_model)
    target_mu, target_scale, target_eta = target_diffusion_model.compute_diffusion_stuff(step, x, obs, model=target_diffusion_model.forward_model)

    fwd_mean = x + eta * mu
    old_fwd_mean = x + target_eta * target_mu
    key, key_gen = jax.random.split(key_gen)

    # x_new from old_fwd_mean
    x_new = sample_kernel(key, check_stop_grad(old_fwd_mean, stop_grad) if stop_grad else old_fwd_mean, target_scale)
    pi_old = distrax.Normal(loc=old_fwd_mean, scale=target_scale)
    x_new_logprob = pi_old.sample(seed=key, sample_shape=(kl_action_rep,))

    # Evaluate kernels
    fwd_log_prob = log_prob_kernel(x_new_logprob, fwd_mean, scale)
    old_fwd_log_prob = log_prob_kernel(x_new_logprob, old_fwd_mean, target_scale)

    # take mean over kl_action_rep
    gen_log_prob = jnp.mean(fwd_log_prob, axis=0)
    old_gen_log_prob = jnp.mean(old_fwd_log_prob, axis=0)

    # Update weight and return
    key, key_gen = jax.random.split(key_gen)
    out_dict = {
        "p_log_prob": old_gen_log_prob, ### samples are from p log prob
        "q_log_prob": gen_log_prob,
        "x_new": x_new,
    }
    return out_dict, key_gen


def sde_integrator(obs, diffusion_model, stop_grad=False, ode=False, ode_coef=1.0):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_w, key_gen = state

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step, obs)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        # fwd_mean = x + eta * (drift + (ode_coef * diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))

        key, key_gen = jax.random.split(key_gen)
        # x_new = fwd_mean if ode else sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
        x_new = sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        # Update weight and return
        log_w += bwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        return next_state, None

    return integrate_EM

def ode_integrator(obs, diffusion_model, stop_grad=False, ode=False, ode_coef=1.0):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_w, key_gen = state

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step, obs)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        # fwd_mean = x + eta * (drift + (ode_coef * diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = x + eta * (drift + ode_coef * diffusion_model.forward_model(step, x, obs))
        x_new = fwd_mean

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        # Update weight and return
        log_w += bwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        return next_state, None

    return integrate_EM

def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=True, kl_action_rep=1):
    def logratio_EM(state, step):
        x, log_w, key_gen = state

        step = step.astype(jnp.float32)

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step, obs)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = x + eta * (drift + target_diffusion_model.forward_model(step, x, obs))
        key, key_gen = jax.random.split(key_gen)

        # x_new from old_fwd_mean
        x_new = sample_kernel(key, check_stop_grad(old_fwd_mean, stop_grad) if stop_grad else old_fwd_mean, scale)
        pi_old = distrax.Normal(loc=old_fwd_mean, scale=scale)
        x_new_logprob = pi_old.sample(seed=key, sample_shape=(kl_action_rep,))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new_logprob, fwd_mean, scale)
        old_fwd_log_prob = log_prob_kernel(x_new_logprob, old_fwd_mean, scale)

        # take mean over kl_action_rep
        fwd_log_prob = jnp.mean(fwd_log_prob, axis=0)
        old_fwd_log_prob = jnp.mean(old_fwd_log_prob, axis=0)

        # Update weight and return
        log_w += old_fwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        return next_state, None
    return logratio_EM


def logratio_DIME(diffusion_model, target_diffusion_model, obs, stop_grad=True, kl_action_rep=1):
    def logratio_EM(state, step):
        x, log_w, key_gen = state

        step = step.astype(jnp.float32)

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step, obs)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = x + eta * (drift + target_diffusion_model.forward_model(step, x, obs))
        # stop_grad for old_diffusion
        old_fwd_mean = jax.lax.stop_gradient(old_fwd_mean)

        # x_new from old_fwd_mean
        key, key_gen = jax.random.split(key_gen)
        # x_new = sample_kernel(key, check_stop_grad(old_fwd_mean, stop_grad) if stop_grad else old_fwd_mean, scale)
        x_new = sample_kernel(key, old_fwd_mean, scale)
        x_new = jax.lax.stop_gradient(x_new)

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        old_fwd_log_prob = log_prob_kernel(x_new, old_fwd_mean, scale)

        # Update weight and return
        log_w += old_fwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        return next_state, None
    return logratio_EM


def scale_inverse_fisher_grad(
    mu: jax.Array,       # unused in Fisher (constant shift), kept for signature compatibility
    eta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Inverse-Fisher gradient scaling for:
        x ~ N(k + 0.5 * exp(log_std) * mu,  exp(2*log_std))

    Parameters are (mu, phi=log_std). Fisher does not depend on k.
    Forward values are unchanged; only gradients are preconditioned.
    """

    # Coeffs depend on mu (treat them as constants for preconditioning)
    mu_sg= jax.lax.stop_gradient(mu)
    eta_sg = jax.lax.stop_gradient(eta)
    eps = 1e-4
    # F^{-1} entries for (mu, phi)
    a = jax.lax.stop_gradient(2/(eta_sg+ eps) + 2 * mu_sg**2) # mu,mu
    b = jax.lax.stop_gradient(-2 *mu_sg*eta_sg)           # mu,phi = phi,mu
    d = jax.lax.stop_gradient(2 * eta_sg**2)                            # phi,phi


    dmu = mu - mu_sg
    deta = eta - eta_sg

    # Apply symmetric 2x2 inverse-Fisher transform
    mu_scaled = mu_sg + a * dmu + b * deta
    eta_scaled = eta_sg + b * dmu + d * deta
    return mu_scaled, eta_scaled

def scale_inverse_fisher_grad_for_backward(
    mu: jax.Array,       # unused in Fisher (constant shift), kept for signature compatibility
    eta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Inverse-Fisher gradient scaling for:
        x ~ N(k + 0.5 * exp(log_std) * mu,  exp(2*log_std))

    Parameters are (mu, phi=log_std). Fisher does not depend on k.
    Forward values are unchanged; only gradients are preconditioned.
    """

    # Coeffs depend on mu (treat them as constants for preconditioning)
    mu_sg= jax.lax.stop_gradient(mu)
    eta_sg = jax.lax.stop_gradient(eta)

    # F^{-1} entries for (mu, phi)
    a = jax.lax.stop_gradient(2*eta**2/(1+ mu**2*eta))


    dmu = mu - mu_sg
    deta = eta - eta_sg

    # Apply symmetric 2x2 inverse-Fisher transform
    mu_scaled = mu
    eta_scaled = eta_sg +  (deta)*a

    return mu_scaled, eta_scaled



def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
):
    "TODO: push to jax"
    return nnx.initializers.variance_scaling(
        0.3333,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def zeros_initializer(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs,
    in_features,
    out_features,
    use_norm=True,
    activation=nnx.swish,
    kernel_init=None,
    bias_init=None,
):
    linear_kwargs = {}
    if kernel_init is not None:
        linear_kwargs["kernel_init"] = kernel_init
    if bias_init is not None:
        linear_kwargs["bias_init"] = bias_init

    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
            **linear_kwargs,
        )
    ]
    if use_norm:
        # layers.append(nnx.RMSNorm(out_features, rngs=rngs))
        layers.append(nnx.LayerNorm(out_features, rngs=rngs))
    if activation is not None:
        layers.append(activation)
    return nnx.Sequential(*layers)


class Identity(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class FCNN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm: bool = True,
        use_output_norm: bool = False,
        layers: int = 2,
        input_activation: bool = False,
        input_skip: bool = False,
        hidden_skip: bool = False,
        output_skip: bool = False,
        output_kernel_init=None,
        output_bias_init=None,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = layers
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.input_skip = input_skip
        self.hidden_skip = hidden_skip
        self.output_skip = output_skip
        if layers == 1:
            hidden_dim = out_features
        self.input_layer = normed_activation_layer(
            rngs, 
            in_features,
            hidden_dim,
            use_norm=use_norm,
            activation=hidden_activation,
        )
        self.main_layers = [
            normed_activation_layer(
                rngs,
                hidden_dim,
                hidden_dim,
                use_norm=use_norm,
                activation=hidden_activation,
            )
            for _ in range(layers - 2)
        ]
        self.output_layer = normed_activation_layer(
            rngs,
            hidden_dim,
            out_features,
            use_norm=use_output_norm,
            activation=output_activation,
            kernel_init=output_kernel_init,
            bias_init=output_bias_init,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        def _potentially_skip(skip, x, layer):
            if skip:
                return x + layer(x)
            else:
                return layer(x)

        if self.input_activation:
            x = self.hidden_activation(x)
        if self.layers == 1:
            return _potentially_skip(self.input_skip, x, self.input_layer)
        x = _potentially_skip(self.input_skip, x, self.input_layer)
        for layer in self.main_layers:
            x = _potentially_skip(self.hidden_skip, x, layer)
        return _potentially_skip(self.output_skip, x, self.output_layer)


class CriticNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_encoder_norm: bool = False,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        use_skip=False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out
        self.feature_module = FCNN(
            in_features=obs_dim + action_dim + self.num_time_out,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            input_activation=not use_simplical_embedding,
            hidden_skip=use_skip,
            layers=head_layers,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            input_activation=not use_simplical_embedding,
            layers=pred_layers,
            rngs=rngs,
        )

        self.timestep_phase = nnx.Param(jnp.zeros((1, self.num_time_hid)))
        # Store timestep_coeff as a Variable (non-trainable parameter)
        self.timestep_coeff = nnx.Variable(
            jnp.linspace(start=0.1, stop=100, num=self.num_time_hid)[None]
        )

        # Time encoder network
        self.time_coder_state = nnx.Sequential(
            nnx.Linear(self.num_time_hid * 2, self.num_time_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(self.num_time_hid, self.num_time_out, rngs=rngs),
        )

    def from_dict_to_observation(self, obs_dict):
        orig_obs = obs_dict["orig_obs"]
        normed_prev_actions = obs_dict["normed_actions"]
        time = obs_dict["diff_time_step"]
        obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
        return obs, time

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def features(self, obs: jax.Array, action: jax.Array, time: jax.Array):
        time_emb = self.get_fourier_features(time)
        if len(action.shape) == 1:
            time_emb = time_emb[0]
        t_net = self.time_coder_state(time_emb)
        state = jnp.concatenate([obs, action, t_net], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    def critic(self, obs_dict: jax.Array, action: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, action, time)
        return self.critic_head(features)

    def critic_cat(self, obs_dict: jax.Array, action: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, action, time)
        return self.critic_head(features)

    def forward(self, obs_dict, action):
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, action, time)
        value = self.critic_head(features)
        pred = self.pred_module(features)
        pred_rew = pred[..., :1]
        pred_features = pred[..., 1:]
        return features, pred_features, pred_rew, value.squeeze(-1)


class CategoricalCriticNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out

        self.use_skip = use_skip

        if project_discrete_action:
            self.action_embedding = nnx.Embed(
                num_embeddings=action_dim,
                features=hidden_dim//2,
            )
            action_dim = hidden_dim // 2
        else:
            self.action_embedding = Identity()

        self.feature_module = FCNN(
            in_features=obs_dim + action_dim + self.num_time_out,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip= use_skip,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=self.num_bins,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=None,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            rngs=rngs,
        )

        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )
                # Initialize timestep parameters
        self.timestep_phase = nnx.Param(jnp.zeros((1, self.num_time_hid)))
        # Store timestep_coeff as a Variable (non-trainable parameter)
        self.timestep_coeff = nnx.Variable(
            jnp.linspace(start=0.1, stop=50, num=self.num_time_hid)[None]
        )

        # Time encoder network
        self.time_coder_state = nnx.Sequential(
            nnx.Linear(self.num_time_hid * 2, self.num_time_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(self.num_time_hid, self.num_time_out, rngs=rngs),
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def compute_action_embedding(self, action: jax.Array) -> jax.Array:
        tanh_action = action#jnp.tanh(action)
        return self.action_embedding(tanh_action)

    def features(self, obs: jax.Array, action: jax.Array, time: jax.Array):
        action_embedding = self.compute_action_embedding(action)
        time_emb = self.get_fourier_features(time)
        if len(action.shape) == 1:
            time_emb = time_emb[0]
        t_net = self.time_coder_state(time_emb)
        state = jnp.concatenate([obs, action_embedding, t_net], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features) + self.zero_dist.value * 40.0
        return cat
    
    def from_dict_to_observation(self, obs_dict):
        orig_obs = obs_dict["orig_obs"]
        normed_prev_actions = obs_dict["normed_actions"]
        time = obs_dict["diff_time_step"]
        obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
        return obs, time

    def critic_cat(self, obs_dict: jax.Array, action: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, action, time)
        return self.critic_head(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat(obs, action), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def forward(self, obs_dict, action):
        obs, time = self.from_dict_to_observation(obs_dict)
        #action = jnp.tanh(action) # tanh is pulled  into observation

        features = self.features(obs, action, time)
        value_cat = jax.nn.softmax(self.critic_head(features), axis=-1)

        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        preds = self.pred_module(features)
        pred_rew = preds[..., :1]
        pred_features = preds[..., 1:]
        if self.use_skip:
            pred_features = pred_features + features
        
        return features, pred_features, pred_rew, value


class CategoricalValueNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out

        self.use_skip = use_skip

        self.feature_module = FCNN(
            in_features=obs_dim + self.num_time_out,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=self.num_bins,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=None,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            rngs=rngs,
        )

        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )
        self.timestep_phase = nnx.Param(jnp.zeros((1, self.num_time_hid)))
        self.timestep_coeff = nnx.Variable(
            jnp.linspace(start=0.1, stop=50, num=self.num_time_hid)[None]
        )

        self.time_coder_state = nnx.Sequential(
            nnx.Linear(self.num_time_hid * 2, self.num_time_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(self.num_time_hid, self.num_time_out, rngs=rngs),
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def features(self, obs: jax.Array, time: jax.Array):
        time_emb = self.get_fourier_features(time)
        if len(obs.shape) == 1:
            time_emb = time_emb[0]
        t_net = self.time_coder_state(time_emb)
        state = jnp.concatenate([obs, t_net], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features) + self.zero_dist.value * 40.0
        return cat

    def from_dict_to_observation(self, obs_dict):
        orig_obs = obs_dict["orig_obs"]
        normed_prev_actions = obs_dict["normed_actions"]
        time = obs_dict["diff_time_step"]
        obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
        return obs, time

    def critic_cat(self, obs_dict: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, time)
        return self.critic_head(features)

    def critic(self, obs_dict: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat(obs_dict), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def forward(self, obs_dict):
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, time)
        value_cat = jax.nn.softmax(self.critic_head(features), axis=-1)

        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        preds = self.pred_module(features)
        pred_rew = preds[..., :1]
        pred_features = preds[..., 1:]
        if self.use_skip:
            pred_features = pred_features + features

        return features, pred_features, pred_rew, value


class ValueNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        layers: int = 2,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.value_module = FCNN(
            in_features=obs_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            use_norm=use_norm,
            layers=layers,
            hidden_skip=use_skip,
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.value_module(obs).squeeze(-1)

class DiffValueNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        project_discrete_action: bool = False,
        use_norm: bool = True,
        use_encoder_norm: bool = False,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        use_skip=False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out
        self.feature_module = FCNN(
            in_features=obs_dim  + self.num_time_out,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            input_activation=not use_simplical_embedding,
            hidden_skip=use_skip,
            layers=head_layers,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            input_activation=not use_simplical_embedding,
            layers=pred_layers,
            rngs=rngs,
        )

        self.timestep_phase = nnx.Param(jnp.zeros((1, self.num_time_hid)))
        # Store timestep_coeff as a Variable (non-trainable parameter)
        self.timestep_coeff = nnx.Variable(
            jnp.linspace(start=0.1, stop=100, num=self.num_time_hid)[None]
        )

        # Time encoder network
        self.time_coder_state = nnx.Sequential(
            nnx.Linear(self.num_time_hid * 2, self.num_time_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(self.num_time_hid, self.num_time_out, rngs=rngs),
        )

    def from_dict_to_observation(self, obs_dict):
        orig_obs = obs_dict["orig_obs"]
        normed_prev_actions = obs_dict["normed_actions"]
        time = obs_dict["diff_time_step"]
        obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
        return obs, time

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def features(self, obs: jax.Array, time: jax.Array):
        time_emb = self.get_fourier_features(time)
        if len(obs.shape) == 1:
            time_emb = time_emb[0]
        t_net = self.time_coder_state(time_emb)
        state = jnp.concatenate([obs, t_net], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    def critic(self, obs_dict: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, time)
        return self.critic_head(features)

    def critic_cat(self, obs_dict: jax.Array) -> jax.Array:
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, time)
        return self.critic_head(features)

    def forward(self, obs_dict):
        obs, time = self.from_dict_to_observation(obs_dict)
        features = self.features(obs, time)
        value = self.critic_head(features)
        pred = self.pred_module(features)
        pred_rew = pred[..., :1]
        pred_features = pred[..., 1:]
        return features, pred_features, pred_rew, value.squeeze(-1)


class GumbleSoftmaxDistribution(distrax.Distribution):
    def __init__(self, logits: jax.Array, temperature: jax.Array):
        self.logits = logits
        self.temperature = temperature

    def sample(self, seed=None):
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).log_prob(value)

    def sample_and_log_prob(self, *, seed, sample_shape=...):
        sample = self.sample(seed=seed)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class DiffusionModel(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nnx.Module = None,
        bwd_model: nnx.Module = None,
        diff_steps: int = 8,
        init_std: float = 2.5,
        friction: float = 1.0,
        per_dim_friction: bool = True,
        dt: float = 0.01,
        use_friction_mlp: bool = False,
        friction_mlp_hidden: int = 64,
        friction_mlp_layers: int = 2,
        friction_num_time_hid: int = 32,
        friction_num_time_out: int = 16,
        friction_mlp_use_obs: bool = True,
        learn_dt: bool = True,
        per_step_dt: bool = False,
        learn_prior: bool = False,
        learn_betas: bool = False,
        learn_friction: bool = True,
        learn_mass_matrix: bool = False,
        langevin_param: bool = False,
        dt_schedule: callable = None,
        train_mode: str = "reparam",
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.init_std = init_std
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        self.learn_prior = learn_prior
        self.learn_friction = learn_friction
        self.learn_mass_matrix = learn_mass_matrix
        self.learn_dt = learn_dt
        self.learn_betas = learn_betas
        self.per_step_dt = per_step_dt
        self.dt_schedule = dt_schedule
        self.langevin_param = langevin_param
        self.train_mode = train_mode
        self.use_friction_mlp = use_friction_mlp
        self.friction_num_time_hid = friction_num_time_hid
        self.friction_num_time_out = friction_num_time_out
        self.friction_mlp_use_obs = friction_mlp_use_obs
        # Actor observation often concatenates orig_obs and prev actions; use a derived
        # orig-obs dim for friction MLP input to avoid mismatched shapes at runtime.
        self.orig_obs_dim = (
            observation_dim - action_dim if observation_dim > action_dim else observation_dim
        )
        
        # Learnable parameters (converted from the params dict)
        self.betas = nnx.Param(jnp.ones((diff_steps,)))
        self.prior_mean = nnx.Param(jnp.zeros((action_dim,)))
        self.prior_std = nnx.Param(jnp.ones((action_dim,)) * inverse_softplus(init_std))
        self.mass_std = nnx.Param(jnp.ones(1) * inverse_softplus(1.0))

        # Initialize dt parameters
        if per_step_dt:
            self.dt = nnx.Param(inverse_softplus(jnp.ones(diff_steps) * dt * dt_schedule(jnp.arange(diff_steps))))
        else:
            self.dt = nnx.Param(jnp.ones(1) * inverse_softplus(dt))
        
        # Initialize friction parameters
        if per_dim_friction:
            self.friction = nnx.Param(jnp.ones(action_dim) * inverse_softplus(friction))
        else:
            self.friction = nnx.Param(jnp.ones(1) * inverse_softplus(friction))

        if self.use_friction_mlp:
            self.friction_timestep_phase = nnx.Param(jnp.zeros((1, self.friction_num_time_hid)))
            self.friction_timestep_coeff = nnx.Variable(
                jnp.linspace(start=0.1, stop=50, num=self.friction_num_time_hid)[None]
            )
            self.friction_time_coder_state = nnx.Sequential(
                nnx.Linear(self.friction_num_time_hid * 2, self.friction_num_time_hid, rngs=rngs),
                nnx.gelu,
                nnx.Linear(self.friction_num_time_hid, self.friction_num_time_out, rngs=rngs),
            )
            friction_out_dim = self.friction.value.shape[-1]
            friction_in_features = (
                self.friction_num_time_out if not self.friction_mlp_use_obs
                else self.orig_obs_dim + self.friction_num_time_out
            )
            self.friction_mlp = FCNN(
                in_features=friction_in_features,
                out_features=friction_out_dim,
                hidden_dim=friction_mlp_hidden,
                use_norm=True,
                use_output_norm=False,
                layers=friction_mlp_layers,
                output_kernel_init=zeros_initializer,
                output_bias_init=zeros_initializer,
                rngs=rngs,
            )

    def get_prior_entropy(self):
        dist = distrax.MultivariateNormalDiag(
            self.prior_mean.value, jax.nn.softplus(self.prior_std.value)
        )
        return dist.entropy()

    def prior_sampler(self, key, n_samples):
        """Sample from the prior distribution.
        
        Args:
            key: JAX random key
            n_samples: Number of samples to generate (batch size)
            
        Returns:
            Samples of shape (n_samples, action_dim)
        """
        # Ensure n_samples is a Python int for sample_shape
        if isinstance(n_samples, jax.Array):
            n_samples = int(n_samples)
        
        samples = distrax.MultivariateNormalDiag(
            self.prior_mean.value, jax.nn.softplus(self.prior_std.value)
        ).sample(seed=key, sample_shape=(n_samples,))
        
        return samples if self.learn_prior else jax.lax.stop_gradient(samples)

    def prior_log_prob(self, x):
        if self.learn_prior:
            log_probs = distrax.MultivariateNormalDiag(
                self.prior_mean.value, jax.nn.softplus(self.prior_std.value)
            ).log_prob(x)
        else:
            log_probs = distrax.MultivariateNormalDiag(
                jnp.zeros(self.action_dim), jnp.ones(self.action_dim) * self.init_std
            ).log_prob(x)
        return log_probs

    def delta_t_fn(self, step: jax.Array) -> jax.Array:
        """Time step function."""
        if self.per_step_dt:
            dt = self.dt.value[step.astype(int)] if self.learn_dt else jax.lax.stop_gradient(self.dt.value[step.astype(int)])
            return jax.nn.softplus(dt)
        else:
            dt = self.dt.value if self.learn_dt else jax.lax.stop_gradient(self.dt.value)
            return jax.nn.softplus(dt) * self.dt_schedule(step)

    def get_friction_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.friction_timestep_coeff.value * timesteps) + self.friction_timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.friction_timestep_coeff.value * timesteps) + self.friction_timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def friction_fn(self, step: jax.Array, obs_dict: dict[str, jax.Array] | None = None) -> jax.Array:
        """Friction coefficient function."""
        friction = jax.nn.softplus(self.friction.value)
        if self.use_friction_mlp:
            if obs_dict is None:
                raise ValueError("obs_dict must be provided when use_friction_mlp=True.")
            time = obs_dict["diff_time_step"]
            time_emb = self.get_friction_fourier_features(time)
            if self.friction_mlp_use_obs:
                orig_obs = obs_dict["orig_obs"]
                if len(orig_obs.shape) == 1:
                    time_emb = time_emb[0]
                t_net = self.friction_time_coder_state(time_emb)
                mlp_in = jnp.concatenate([orig_obs, t_net], axis=-1)
            else:
                if len(time_emb.shape) == 1:
                    time_emb = time_emb[0]
                mlp_in = self.friction_time_coder_state(time_emb)
            friction_out = self.friction_mlp(mlp_in)
            friction = jax.nn.softplus(self.friction.value + friction_out)

        return friction if self.learn_friction else jax.lax.stop_gradient(friction)
    
    def diffusion_coeff_fn(self, step: jax.Array, obs_dict: dict[str, jax.Array]) -> jax.Array:
        friction_value = self.friction_fn(step, obs_dict) 
        if(self.learn_friction==False):
            friction_value = jax.lax.stop_gradient(friction_value)
        dt = self.delta_t_fn(step)
        sigma_square = 1.0 / friction_value
        eta = dt * sigma_square
        log_scale = 0.5 * jnp.log(2.0 * eta)
        scale = jnp.sqrt(2*eta)
        # if self.train_mode == "WPO":
        #     log_scale_sg = jax.lax.stop_gradient(log_scale)
        #     log_scale = log_scale_sg + (log_scale - log_scale_sg) * 1/2
        #     scale = jnp.exp(log_scale)
        #     eta = jnp.exp(2*log_scale)/2
        return (scale, eta, log_scale) #if self.learn_friction else (jax.lax.stop_gradient(scale), jax.lax.stop_gradient(eta), jax.lax.stop_gradient(log_scale))
    
    def return_fisher_scaled_mean_and_scale(self, mu, eta, mode = "forward"):
        if(mode == "forward"):
            mu, eta = scale_inverse_fisher_grad(mu, eta)
        else:
            mu, eta = scale_inverse_fisher_grad_for_backward(mu, eta)
        scale = jnp.sqrt(2.0 * eta)
        return mu, scale, eta

    def compute_diffusion_stuff(self, step: jax.Array, x: jax.Array, obs_dict: dict[str, jax.Array], model = None, ode_coeff: float = 1.0, train_mode: bool = True) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute diffusion related quantities."""
        scale, eta, log_diffusion_coeff = self.diffusion_coeff_fn(step, obs_dict)
        drift = self.drift_fn(step, x)
        score = model(step, x, obs_dict)
        mu = drift + ode_coeff *score
        if (self.train_mode == "WPO" and train_mode == True):
            # if the model is the forward_model() t
            # if(model == self.forward_model):
            #     mu, scale, eta = self.return_fisher_scaled_mean_and_scale(mu, eta, mode="forward")
            # elif(model == self.backward_model):
            #     mu, scale, eta = self.return_fisher_scaled_mean_and_scale(mu, eta, mode="backward")
            # else:
            #     raise ValueError("model must be either forward_model or backward_model")
            pass
        else:
            pass
        return mu, scale, eta
    

    def mass_fn(self) -> jax.Array:
        """Mass function."""
        mass_std = jax.nn.softplus(self.mass_std.value)
        return mass_std if self.learn_mass_matrix else jax.lax.stop_gradient(mass_std)

    def get_prior_std(self) -> jax.Array:
        return jax.nn.softplus(self.prior_std.value)*jnp.ones(self.action_dim) if self.learn_prior else jnp.ones(self.action_dim) * self.init_std

    def drift_fn(self, step: jax.Array, x: jax.Array) -> jax.Array:
        """Drift function for diffusion (gradient of prior log prob)."""
        # return jax.grad(self.prior_log_prob)(x)
        # Fall back to analytical gradient: ∇_x log p(x) = -(x-μ)/σ²
        mean = self.prior_mean if self.learn_prior else jnp.zeros(self.action_dim)
        std = self.get_prior_std()
        grad = -(x - mean) / (std ** 2)
        return grad

    def forward_model(
        self, step: jax.Array, x: jax.Array, obs_dict: dict[str, jax.Array], aux: jax.Array = None
    ) -> jax.Array:
        """Forward model function."""
        if self.fwd_model is not None:
            orig_obs = obs_dict["orig_obs"]
            normed_prev_actions = obs_dict["normed_actions"]
            obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
            q_grad = obs_dict.get("q_grad") if self.langevin_param else None
            fwd_out = self.fwd_model(x, obs, step, q_grad=q_grad)
            # if self.train_mode == "WPO":
            #     friction = self.friction_fn(step, obs_dict)
            #     dt = self.delta_t_fn(step)
            #     sigma_square = 1.0 / friction
            #     eta = dt * sigma_square
            #     scale = jnp.sqrt(2.0 * eta)
            #     varsg = jax.lax.stop_gradient(scale ** 2)
            #     mean_sg = jax.lax.stop_gradient(fwd_out)
            #     fwd_out = mean_sg + (fwd_out - mean_sg) * varsg
            return fwd_out
        else:
            return jnp.zeros_like(x)

    def backward_model(
        self, step: jax.Array, x: jax.Array, obs_dict: dict[str, jax.Array], aux: jax.Array = None
    ) -> jax.Array:
        """Backward model function."""
        if self.bwd_model is not None:
            orig_obs = obs_dict["orig_obs"]
            normed_prev_actions = obs_dict["normed_actions"]
            obs = jnp.concatenate([orig_obs, normed_prev_actions], axis=-1)
            q_grad = obs_dict.get("q_grad") if self.langevin_param else None
            return self.bwd_model(x, obs, step, q_grad=q_grad)
        else:
            return jnp.zeros_like(x)

class DMERLActor(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nnx.Module,
        sde_integrator: callable,
        ode_integrator: callable,
        logratio: callable,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        action_clip_value: float = 1.0,
        use_temp_lagrangian_mlp: bool = False,
        temp_lagrangian_hidden: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.action_clip_value = action_clip_value
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.ode_integrator = ode_integrator
        self.logratio = logratio
        self.diff_steps = diffusion_model.diff_steps

        self.use_temp_lagrangian_mlp = use_temp_lagrangian_mlp
        if self.use_temp_lagrangian_mlp:
            if rngs is None:
                raise ValueError(
                    "rngs must be provided when use_temp_lagrangian_mlp=True."
                )
            seed_dim = 4
            self.temperature_seed = nnx.Param(jnp.zeros((1, seed_dim)))
            self.lagrangian_seed = nnx.Param(jnp.zeros((1, seed_dim)))
            self.temperature_bias = nnx.Param(jnp.ones(1) * math.log(ent_start))
            self.lagrangian_bias = nnx.Param(jnp.ones(1) * math.log(kl_start))
            self.temperature_mlp = FCNN(
                in_features=seed_dim,
                out_features=1,
                hidden_dim=temp_lagrangian_hidden,
                use_norm=False,
                output_activation=None,
                layers=2,
                output_kernel_init=zeros_initializer,
                output_bias_init=zeros_initializer,
                rngs=rngs,
            )
            self.lagrangian_mlp = FCNN(
                in_features=seed_dim,
                out_features=1,
                hidden_dim=temp_lagrangian_hidden,
                use_norm=False,
                output_activation=None,
                layers=2,
                output_kernel_init=zeros_initializer,
                output_bias_init=zeros_initializer,
                rngs=rngs,
            )
        else:
            self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
            self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

    def _sample_prior(self, key, n_samples = 1):
        key, key_gen = jax.random.split(key)
        init_x = self.diffusion_model.prior_sampler(key, n_samples)
        out_dict = {"init_x": init_x[0],
                    "log_prior": self.diffusion_model.prior_log_prob(init_x)
                    }
        return out_dict, key
    
    def vmap_sample_prior(self, keys, n_samples):
        in_axes = (0,) # keys
        keys = jax.random.split(keys, num=n_samples)
        out_dict, keys = jax.vmap(self._sample_prior, in_axes=in_axes)(keys)
        return out_dict
    
    def get_prior_entropy(self):
        return self.diffusion_model.get_prior_entropy()
    
    def _eval_log_prob(self, current_x, step, obs, actions):
        out_dict = evaluate_one_step_log_prob(self.diffusion_model, current_x, step, obs, actions, stop_grad=False)
        #gen_log_prob = out_dict["gen_log_prob"]
        #is_last_step = self.diff_steps - 1 == step
        #gen_log_prob_new = jnp.where(is_last_step, gen_log_prob - distrax.Tanh().forward_log_det_jacobian(actions).sum(), gen_log_prob)
        #gen_log_prob_new = gen_log_prob_new
        #out_dict["gen_log_prob"] = gen_log_prob_new
        return out_dict
    
    def vmap_eval_log_prob(self, obs, actions):
        in_axes = (0, 0, 0, 0) # keys, current_x, step, obs

        current_x = obs["orig_actions"]
        step = obs["diff_time_step"][...,0]
        out_dict = jax.vmap(self._eval_log_prob, in_axes=in_axes)( current_x, step, obs, actions)
        gen_log_prob = out_dict["gen_log_prob"]
        dest_log_prob = out_dict["dest_log_prob"]
        return gen_log_prob, dest_log_prob

    def _sample_next_step(self, key, current_x, step, obs):
        out_dict, key = integrate_one_step(self.diffusion_model, current_x, step, obs, key, stop_grad=False)
        x_new = out_dict["x_new"]
        gen_log_prob = out_dict["gen_log_prob"]
        is_last_step = self.diff_steps - 1 == step
        #gen_log_prob_new = jnp.where(is_last_step, gen_log_prob - distrax.Tanh().forward_log_det_jacobian(x_new).sum(), gen_log_prob)
        gen_log_prob_new = gen_log_prob
        # Clip logits so tanh(action) always respects action_clip_value on the final step.
        # clip_limit = jnp.arctanh(jnp.asarray(self.action_clip_value, dtype=x_new.dtype))
        # clipped_x_new = jnp.clip(x_new, -clip_limit, clip_limit)
        # out_dict["x_new"] = jnp.where(is_last_step, clipped_x_new, x_new)
        out_dict["x_new"] = x_new
        out_dict["gen_log_prob"] = gen_log_prob_new

        return out_dict, key
    
    def vmap_sample_next_step(self, obs, keys):
        in_axes = (0, 0, 0, 0) # keys, current_x, step, obs

        current_x = obs["orig_actions"]
        step = obs["diff_time_step"][...,0]
        keys = jax.random.split(keys, num=obs["orig_obs"].shape[0]) ### TODO pay attention are keys correctly split?
        out_dict, keys = jax.vmap(self._sample_next_step, in_axes=in_axes)(keys, current_x, step, obs)
        x_new = out_dict["x_new"]
        gen_log_prob = out_dict["gen_log_prob"]
        dest_log_prob = out_dict["dest_log_prob"]
        actions = x_new
        return actions, gen_log_prob, dest_log_prob

    def _ode_sample_next_step(self, key, current_x, step, obs):
        out_dict, key = ODE_integrate_one_step(self.diffusion_model, current_x, step, obs, key, stop_grad=False)
        return out_dict, key
    
    def vmap_ode_sample_next_step(self, obs, keys):
        in_axes = (None, 0, 0, 0) # keys, current_x, step, obs

        current_x = obs["orig_actions"]
        step = obs["diff_time_step"][...,0]
        out_dict, keys = jax.vmap(self._ode_sample_next_step, in_axes=in_axes)(keys, current_x, step, obs)
        actions = out_dict["x_new"]
        return actions, keys
    
    def sample_complete_loop(self, obs_dict , key):
        batch_size = obs_dict["orig_obs"].shape[0]
        key, key_gen = jax.random.split(key)
        out_dict = self.vmap_sample_prior(key, batch_size)
        init_x = out_dict["init_x"]
        key, key_gen = jax.random.split(key_gen)

        log_ratio = jnp.zeros((init_x.shape[0],), dtype=jnp.float32)
        raise ValueError("update reading of dict") 
        obs_dict["normed_actions"] = init_x
        obs_dict["orig_actions"] = init_x
        obs_dict["diff_time_step"] = jnp.zeros((init_x.shape[0],1), dtype=jnp.int32)

        for i in range(self.diffusion_model.diff_steps):
            step = jnp.array(i, dtype=jnp.int32)
            actions, gen_log_prob, dest_log_prob = self.vmap_sample_next_step(obs_dict, key_gen)

            log_ratio += gen_log_prob - dest_log_prob
            x_new = actions
            obs_dict["normed_actions"] = x_new
            obs_dict["orig_actions"] = x_new
            obs_dict["diff_time_step"] = obs_dict["diff_time_step"] + 1


        return log_ratio
    

    def _single_sde_sample(self, key, obs_dict, stop_grad, ode, ode_coef):
        """
        Private helper for SDE sampling.
        This is the inlined logic from the old global `single_sample`.
        """
        key, key_gen = jax.random.split(key)
        init_x = self.diffusion_model.prior_sampler(key, 1)
        key, key_gen = jax.random.split(key_gen)
        init_x = jnp.squeeze(init_x, 0)
        if stop_grad:
            init_x = jax.lax.stop_gradient(init_x)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, jnp.zeros(1), key)

        obs_dict["normed_actions"] = init_x
        obs_dict["orig_actions"] = init_x
        obs_dict["diff_time_step"] = jnp.zeros((init_x.shape[0],1), dtype=jnp.int32)
        raise ValueError("update reading of dict") 
        # --- Hard-coded to self.sde_integrator ---
        integrate = self.sde_integrator(obs_dict    , self.diffusion_model, stop_grad, ode, ode_coef)
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, _ = aux

        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        running_cost = -(log_ratio  + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
        stochastic_costs = jnp.zeros_like(running_cost)

        return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)

    def sample(
        self,
        key,
        obs_dict: jax.Array,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs_dict["orig_obs"].shape[0])
        # Define the function to vmap
        def _single_sample_for_vmap(key, obs):
            # This function closes over self, stop_grad, ode, ode_coef
            return self._single_sde_sample(key, obs, stop_grad, ode, ode_coef)
        
        in_axes = (0, 0) # keys, obs
        rnd_result = jax.vmap(_single_sample_for_vmap, in_axes=in_axes)(keys, obs_dict)
        
        x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        return (x_0, running_costs, stochastic_costs, terminal_costs)

    def _single_ode_sample(self, key, obs, stop_grad, ode, ode_coef):
        """
        Private helper for ODE sampling.
        This is the inlined logic from the old global `single_sample`.
        """
        key, key_gen = jax.random.split(key)
        init_x = self.diffusion_model.prior_sampler(key, 1)
        key, key_gen = jax.random.split(key_gen)
        init_x = jnp.squeeze(init_x, 0)
        if stop_grad:
            init_x = jax.lax.stop_gradient(init_x)
        key, key_gen = jax.random.split(key_gen)

        log_reverse = jnp.ones((self.diffusion_model.diff_steps,))
        log_forward = jnp.ones((self.diffusion_model.diff_steps,))
        aux = (init_x, jnp.zeros(1), log_reverse, log_forward, key)
        
        integrate = self.ode_integrator(obs, self.diffusion_model, stop_grad, ode, ode_coef)
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, log_reverse, log_forward, _ = aux

        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        running_cost = -(log_ratio  + distrax.Tanh().forward_log_det_jacobian(final_x).sum() )
        stochastic_costs = jnp.zeros_like(running_cost)

        log_prob_dict = {
            "log_reverse": log_reverse,
            "log_forward": log_forward,
        }

        return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape), log_prob_dict

    def det_action(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> jax.Array:
        """Sample actions from the ODE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])
        
        # Define the function to vmap
        def _single_sample_for_vmap(key, obs):
            # This function closes over self, stop_grad, ode, ode_coef
            return self._single_ode_sample(key, obs, stop_grad, ode, ode_coef)
        
        in_axes = (0, 0) # keys, obs
        rnd_result = jax.vmap(_single_sample_for_vmap, in_axes=in_axes)(keys, obs)

        x_0, running_costs, stochastic_costs, terminal_costs, log_probs_dict = rnd_result
        return (x_0, running_costs, stochastic_costs, terminal_costs, log_probs_dict)
    
    def _single_kl_internal(self, key, obs, target_diffusion_model, n_samples, stop_grad):
        key, key_gen = jax.random.split(key)
        init_x = self.diffusion_model.prior_sampler(key, 1)
        key, key_gen = jax.random.split(key_gen)
        init_x = jnp.squeeze(init_x, 0)
        if stop_grad:
            init_x = jax.lax.stop_gradient(init_x)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, jnp.zeros(1), key)
        
        integrate = self.logratio(
            self.diffusion_model, 
            target_diffusion_model.diffusion_model, 
            obs, 
            stop_grad=stop_grad, 
            kl_action_rep=n_samples
        )
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, _ = aux
        return log_ratio
    
    
    def fkl_div_one_step(self, key, obs_actions: jax.Array, obs_target: jax.Array, target_diffusion_model: nnx.Module, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the ONE STEP integrator.
        This method is designed to be vmapped externally (e.g., in actor_loss).
        """
        keys = jax.random.split(key, num=obs_actions["orig_obs"].shape[0])
        
        other_diff_model = self.diffusion_model 
        sample_diff_model = target_diffusion_model.diffusion_model # p model in D_kl(p||q)

        def _single_kl_for_vmap(key, obs_act, obs_tgt):
            # This function closes over self, target_diffusion_model, stop_grad
            current_x = obs_act["orig_actions"]
            step = obs_act["diff_time_step"]
            return logratio_one_step(other_diff_model, sample_diff_model, current_x , step, obs_act, key, stop_grad=stop_grad, target_obs=obs_tgt)
        

        in_axes = (0, 0, 0) # keys, obs_actions, obs_target
        out_dict, keys = jax.vmap(_single_kl_for_vmap, in_axes=in_axes)(keys, obs_actions, obs_target)
        p_log_probs = out_dict["p_log_prob"]
        q_log_probs = out_dict["q_log_prob"]
        log_ratios = p_log_probs - q_log_probs
        return log_ratios[..., None]
    
    def rkl_div_one_step(self, key, obs_actions: jax.Array, obs_target: jax.Array, target_diffusion_model: nnx.Module, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the ONE STEP integrator.
        This method is designed to be vmapped externally (e.g., in actor_loss).
        """
        keys = jax.random.split(key, num=obs_actions["orig_obs"].shape[0])
        sample_diff_model = self.diffusion_model # p model in D_kl(p||q)
        other_diff_model = target_diffusion_model.diffusion_model
        
        def _single_kl_for_vmap(key, obs_act, obs_tgt):
            # This function closes over self, target_diffusion_model, stop_grad
            current_x = obs_act["orig_actions"]
            step = obs_act["diff_time_step"]
            return logratio_one_step(other_diff_model, sample_diff_model, current_x , step, obs_tgt, key, stop_grad=stop_grad, target_obs=obs_act)
        

        in_axes = (0, 0, 0) # keys, obs_actions, obs_target
        out_dict, keys = jax.vmap(_single_kl_for_vmap, in_axes=in_axes)(keys, obs_target, obs_actions)
        p_log_probs = out_dict["p_log_prob"]
        q_log_probs = out_dict["q_log_prob"]
        log_ratios = p_log_probs - q_log_probs
        return log_ratios[..., None]

    def kl_div(self, key, obs: jax.Array, target_diffusion_model: nnx.Module, n_samples: int, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the EFFICIENT internal-sampling integrator.
        Averages n_samples *inside* the diffusion scan.
        """
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _single_kl_for_vmap(key, obs):
            # This function closes over self, target_diffusion_model, n_samples, stop_grad
            return self._single_kl_internal(key, obs, target_diffusion_model, n_samples, stop_grad)
        
        in_axes = (0, 0) # keys, obs
        log_ratios = jax.vmap(_single_kl_for_vmap, in_axes=in_axes)(keys, obs)
        return log_ratios

    def _single_kl_dime(self, key, obs, target_diffusion_model, stop_grad):
        key, key_gen = jax.random.split(key)
        init_x = self.diffusion_model.prior_sampler(key, 1)
        key, key_gen = jax.random.split(key_gen)
        init_x = jnp.squeeze(init_x, 0)
        if stop_grad:
            init_x = jax.lax.stop_gradient(init_x)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, jnp.zeros(1), key)
        
        integrate = self.logratio(
            self.diffusion_model, 
            target_diffusion_model.diffusion_model, 
            obs, 
            stop_grad=stop_grad
        )
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, _ = aux
        return log_ratio

    def kl_div_dime(self, key, obs: jax.Array, target_diffusion_model: nnx.Module, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the SINGLE PATH integrator.
        This method is designed to be vmapped externally (e.g., in actor_loss).
        """
        keys = jax.random.split(key, num=obs["orig_obs"].shape[0])
        
        def _single_kl_for_vmap(key, obs):
            # This function closes over self, target_diffusion_model, stop_grad
            return self._single_kl_dime(key, obs, target_diffusion_model, stop_grad)

        in_axes = (0, 0) # keys, obs
        log_ratios = jax.vmap(_single_kl_for_vmap, in_axes=in_axes)(keys, obs)
        return log_ratios

    def temperature(self) -> jax.Array:
        if self.use_temp_lagrangian_mlp:
            log_temp = self.temperature_mlp(self.temperature_seed.value).squeeze()
            log_temp = log_temp + self.temperature_bias.value
        else:
            log_temp = self.log_temperature.value
        return jnp.exp(log_temp)

    def lagrangian(self) -> jax.Array:
        if self.use_temp_lagrangian_mlp:
            log_lagrangian = self.lagrangian_mlp(self.lagrangian_seed.value).squeeze()
            log_lagrangian = log_lagrangian + self.lagrangian_bias.value
        else:
            log_lagrangian = self.log_lagrangian.value
        return jnp.exp(log_lagrangian)
