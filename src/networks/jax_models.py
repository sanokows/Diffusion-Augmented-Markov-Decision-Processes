import math
from typing import Sequence, Union

from functools import partial

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from src.jaxrl import utils
from src.networks.diffusion.utils import inverse_softplus, sample_kernel, log_prob_kernel, check_stop_grad


def sde_integrator(obs, diffusion_model, stop_grad=False, ode=False, ode_coef=1.0):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_w, key_gen = state

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        dt_next = diffusion_model.delta_t_fn(step + 1 )
        sigma_square_next = 1. / diffusion_model.friction_fn(step + 1)
        eta_next = dt_next * sigma_square_next
        scale_next = jnp.sqrt(2 * eta_next)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        # fwd_mean = x + eta * (drift + (ode_coef * diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))

        key, key_gen = jax.random.split(key_gen)
        # x_new = fwd_mean if ode else sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
        x_new = sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta_next * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale_next)
        # Update weight and return
        # print log_w before
        #jax.debug.print("step: {s}, log_w before: {lw}, bwd_log_prob: {bp}, fwd_log_prob: {fp}", s=step, lw=log_w, bp=bwd_log_prob, fp=fwd_log_prob)
        log_w += bwd_log_prob - fwd_log_prob
        #jax.debug.print("step: {s}, log_w after: {lw}", s=step, lw=log_w)

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
        sigma_square = 1. / diffusion_model.friction_fn(step)
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
        sigma_square = 1. / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        target_dt = target_diffusion_model.delta_t_fn(step)
        target_sigma_square = 1. / target_diffusion_model.friction_fn(step)
        target_eta = target_dt * target_sigma_square
        target_scale = jnp.sqrt(2 * target_eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        target_drift = target_diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = x + target_eta * (target_drift + target_diffusion_model.forward_model(step, x, obs))
        key, key_gen = jax.random.split(key_gen)

        # x_new from old_fwd_mean
        x_new = sample_kernel(key, check_stop_grad(old_fwd_mean, stop_grad) if stop_grad else old_fwd_mean, target_scale)
        pi_old = distrax.Normal(loc=old_fwd_mean, scale=target_scale)
        x_new_logprob = pi_old.sample(seed=key, sample_shape=(kl_action_rep,))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new_logprob, fwd_mean, scale)
        old_fwd_log_prob = log_prob_kernel(x_new_logprob, old_fwd_mean, target_scale)

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
        sigma_square = 1. / diffusion_model.friction_fn(step)
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


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs, in_features, out_features, use_norm=True, activation=nnx.swish
):
    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
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
        # self.norm = nnx.RMSNorm(in_features, rngs=rngs)
        self.norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.output_layer = normed_activation_layer(
            rngs,
            hidden_dim,
            out_features,
            use_norm=use_output_norm,
            activation=output_activation,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        def _potentially_skip(skip, x, layer):
            if skip:
                return x + layer(x)
            else:
                return layer(x)

        if self.input_activation:
            # x = self.norm(x)
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
        use_skip=False,
        *,
        rngs: nnx.Rngs,
    ):
        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
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

    def features(self, obs: jax.Array, action: jax.Array):
        state = jnp.concatenate([obs, action], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def forward(self, obs, action):
        features = self.features(obs, action)
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
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

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
            in_features=obs_dim + action_dim,
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

    def features(self, obs: jax.Array, action: jax.Array):
        action_embedding = self.action_embedding(action)
        state = jnp.concatenate([obs, action_embedding], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features) + self.zero_dist.value * 40.0
        return cat

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat(obs, action), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def forward(self, obs, action):
        features = self.features(obs, action)
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


class SACActorNetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        use_skip: bool = False,
        train_mode: str = "reparam",
        disable_wpo_fisher_preconditioning: bool = False,
        disable_temperature: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim * 2,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.disable_temperature = disable_temperature
        kl_start_value = math.log(kl_start)
        if self.disable_temperature:
            self.temperature_log_param = None
        else:
            start_value = math.log(ent_start)
            self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std
        if train_mode not in ("reparam", "WPO"):
            raise ValueError(f"Unknown train_mode: {train_mode}")
        self.train_mode = train_mode
        self.disable_wpo_fisher_preconditioning = disable_wpo_fisher_preconditioning

    def _compute_mean_std(
        self, obs: jax.Array, scale: float | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        loc = self.actor_module(obs)
        mean, log_std = jnp.split(loc, 2, axis=-1)

        if (
            self.train_mode == "WPO"
            and not self.disable_wpo_fisher_preconditioning
        ):
            log_std_sg = jax.lax.stop_gradient(log_std)
            log_std = log_std_sg + (log_std - log_std_sg) * 0.5
            std = (jnp.exp(log_std) + self.min_std) * scale
            var = std ** 2
            mean_sg = jax.lax.stop_gradient(mean)
            varsg = jax.lax.stop_gradient(var)
            mean = mean_sg + (mean - mean_sg) * (varsg)
        else:
            std = (jnp.exp(log_std) + self.min_std) * scale

        return mean, std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc, std = self._compute_mean_std(obs, scale)
        pi = distrax.Transformed(
            distrax.Normal(loc=loc, scale=std),
            distrax.Tanh()
        )
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc, _ = self._compute_mean_std(obs, 1.0)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        if self.disable_temperature:
            return jnp.zeros(1)
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    def __call__(self, obs: jax.Array) -> jax.Array:
        loc, std = self._compute_mean_std(obs, 1.0)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


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


class SACDiscreteActorNetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    def __call__(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, std = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


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
        learn_dt: bool = True,
        per_step_dt: bool = False,
        learn_prior: bool = False,
        learn_betas: bool = False,
        learn_friction: bool = True,
        learn_mass_matrix: bool = False,
        dt_schedule: callable = None,
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

    def friction_fn(self, step: jax.Array) -> jax.Array:
        """Friction coefficient function."""
        friction = jax.nn.softplus(self.friction.value)
        return friction if self.learn_friction else jax.lax.stop_gradient(friction)

    def mass_fn(self) -> jax.Array:
        """Mass function."""
        mass_std = jax.nn.softplus(self.mass_std.value)
        return mass_std if self.learn_mass_matrix else jax.lax.stop_gradient(mass_std)

    def drift_fn(self, step: jax.Array, x: jax.Array) -> jax.Array:
        """Drift function for diffusion (gradient of prior log prob)."""
        # return jax.grad(self.prior_log_prob)(x)
        # Fall back to analytical gradient: ∇_x log p(x) = -(x-μ)/σ²
        mean = self.prior_mean if self.learn_prior else jnp.zeros(self.action_dim)
        std = jax.nn.softplus(self.prior_std) if self.learn_prior else jnp.ones(self.action_dim) * self.init_std
        grad = -(x - mean) / (std ** 2)
        return grad

    def forward_model(
        self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)

    def backward_model(
        self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Backward model function."""
        if self.bwd_model is not None:
            return self.bwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)

class DIMEActor(nnx.Module):
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
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.ode_integrator = ode_integrator
        self.logratio = logratio

        # Parameters
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

    def _single_sde_sample(self, key, obs, stop_grad, ode, ode_coef):
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

        # --- Hard-coded to self.sde_integrator ---
        integrate = self.sde_integrator(obs, self.diffusion_model, stop_grad, ode, ode_coef)
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, _ = aux

        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
        unscaled_running_cost = -log_ratio
        stochastic_costs = jnp.zeros_like(running_cost)

        final_x = distrax.Tanh().forward(final_x)
        return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape), unscaled_running_cost

    def sample(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])

        # Define the function to vmap
        def _single_sample_for_vmap(key, obs):
            # This function closes over self, stop_grad, ode, ode_coef
            return self._single_sde_sample(key, obs, stop_grad, ode, ode_coef)
        
        in_axes = (0, 0) # keys, obs
        rnd_result = jax.vmap(_single_sample_for_vmap, in_axes=in_axes)(keys, obs)
        
        x_0, running_costs, stochastic_costs, terminal_costs,unscaled_running_costs = rnd_result
        return (x_0, running_costs, stochastic_costs, terminal_costs, unscaled_running_costs)

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
        aux = (init_x, jnp.zeros(1), key)
        
        integrate = self.ode_integrator(obs, self.diffusion_model, stop_grad, ode, ode_coef)
        
        aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
        final_x, log_ratio, _ = aux

        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
        stochastic_costs = jnp.zeros_like(running_cost)

        final_x = distrax.Tanh().forward(final_x)
        return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)

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

        x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        return (x_0, running_costs, stochastic_costs, terminal_costs)
    
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
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _single_kl_for_vmap(key, obs):
            # This function closes over self, target_diffusion_model, stop_grad
            return self._single_kl_dime(key, obs, target_diffusion_model, stop_grad)

        in_axes = (0, 0) # keys, obs
        log_ratios = jax.vmap(_single_kl_for_vmap, in_axes=in_axes)(keys, obs)
        return log_ratios

    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)
