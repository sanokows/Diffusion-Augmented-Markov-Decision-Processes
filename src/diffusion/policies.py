import jax.numpy as jnp
import jax
import distrax
import flax.linen as nn
import optax
import numpy as np
import tensorflow_probability
import time


from functools import partial

from flax.training.train_state import TrainState

from common.distributions import TanhTransformedDistribution
from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from diffusion.actors import PISGRADNet, StatDistrActor
from diffusion.noise_schedules import get_cosine_noise_schedule
from typing import Optional, Type, Callable, Any, Union, List, Dict
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from jax._src.nn.functions import softplus

from sac.policies import VectorCritic

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


def inverse_softplus(x):
    # Numerically stable implementation of inverse softplus
    # Threshold above which the approximation log(e^x - 1) ≈ x is used
    threshold = 20.0
    return jnp.where(x > threshold, x, jnp.log(jnp.expm1(x)))


class GenDPOL(BaseJaxPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Box,
                 lr_schedule: Schedule,
                 activation_fn: Type[nn.Module],
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 pis_weight_init: float = 1e-8,
                 pis_bias_init: float = 0.0,
                 diff_init_std: float = -1.0,
                 diff_steps: int = 64,
                 learn_beta: bool = False,
                 init_beta_scale: float = 1.0,    # only used if leran_beta is true
                 dropout_rate: float = 0.0,
                 layer_norm: bool = False,
                 batch_norm: bool = False,
                 batch_norm_momentum: float = 0.9,
                 batch_norm_mode: str = "bn",
                 use_sde: bool = False,
                 # Note: most gSDE parameters are not used
                 # this is to keep API consistent with SB3
                 log_std_init: float = -3,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 features_extractor_class=None,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Callable[...,
                 optax.GradientTransformation] = optax.adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 n_critics: int = 2,
                 share_features_extractor: bool = False,
                 squash_output=True,
                 ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_mode = batch_norm_mode
        self.activation_fn = activation_fn
        self.net_arch_pi = net_arch["pi"]
        self.net_arch_qf = net_arch["qf"]
        self.pis_weight_init = pis_weight_init
        self.pis_bias_init = pis_bias_init
        self.dim = int(np.prod(self.action_space.shape))
        self.diff_init_std = diff_init_std
        self.diff_steps = diff_steps
        self.n_critics = n_critics
        self.use_sde = use_sde
        self.net_arch_stat_distr = net_arch["stat_distr"]
        self.learn_beta = learn_beta
        self.init_beta_scale = init_beta_scale

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, score_key, stat_distr_key, qf_key, dropout_key, stat_distr_bn_key, bn_key = jax.random.split(key, 7)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]

        # TODO: No batch norm is supported for the actor so far
        self.score = PISGRADNet(dim=int(np.prod(self.action_space.shape)),
                                obs_dim=int(np.prod(obs.shape)),
                                net_arch=self.net_arch_pi,
                                weight_init=self.pis_weight_init,
                                bias_init=self.pis_bias_init,
                                )

        # Hack to make gSDE work without modifying internal SB3 code
        self.score.reset_noise = self.reset_noise

        actor_init_variables = self.score.init({"params": score_key}, action, obs, jnp.ones([batch_size, 1]),
                                               train=False)

        self.stat_distr = StatDistrActor(action_dim=int(np.prod(self.action_space.shape)),
                                         net_arch=self.net_arch_stat_distr)
        self.stat_distr.reset_noise = self.reset_noise
        stat_distr_init_variables = self.stat_distr.init({"params": stat_distr_key}, obs, train=True)

        if self.learn_beta:
            beta_scaler = jnp.ones(self.action_space.shape) * inverse_softplus(self.init_beta_scale)
            # beta_scaler = jnp.ones(1) * inverse_softplus(self.init_beta_scale)
        else:
            beta_scaler = jax.lax.stop_gradient(jnp.ones(self.action_space.shape))
            # beta_scaler = jax.lax.stop_gradient(jnp.ones(1))

        all_actor_params = {"params": {"score_params": actor_init_variables,
                                       "stat_distr_params": stat_distr_init_variables,
                                       "beta_scaler": beta_scaler}}

        # as we are not using batchrenorm here. Only using TrainState instead of ActorTrainState is ok
        actor_grad_clip = self.optimizer_kwargs.get("grad_clip", 0.0)
        self.actor_state = TrainState.create(
            apply_fn=(self.score.apply, self.stat_distr.apply),
            params=all_actor_params["params"],
            tx=optax.chain(optax.zero_nans(),
                           optax.clip(actor_grad_clip) if actor_grad_clip > 0.0 else optax.identity(),
                           self.optimizer_class(learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                                                **self.optimizer_kwargs)),
        )

        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_mode=self.batch_norm_mode,
            net_arch=self.net_arch_qf,
            activation_fn=self.activation_fn,
            n_critics=self.n_critics,
        )

        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.score.apply = jax.jit(  # type: ignore[method-assign]
            self.score.apply,
            static_argnames=("weight_init", "bias_init")  # if want to support batch norm, implementation goes here
        )

        self.stat_distr.apply = jax.jit(  # type: ignore[method-assign]
            self.stat_distr.apply
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )
        return key

    # zero mean unit var stat distr + original loss
    @staticmethod
    def _sample_action_single(key, observations, actor_state, actor_params, noise_schedule, a_dim, n_steps):
        dt = 1. / n_steps if n_steps != 0 else 0
        betas = noise_schedule

        stat_distr = distrax.MultivariateNormalDiag(jnp.zeros(a_dim), jnp.ones(a_dim) * 2.5)
        # stat_distr = distrax.MultivariateNormalDiag(jnp.zeros(a_dim), jnp.ones(a_dim) * 0.5)
        # stat_distr = actor_state.apply_fn[1](actor_params["stat_distr_params"], observations, train=False)

        def simulate_prior_to_target(state, per_step_input):
            x, key_gen = state
            step_float = per_step_input.astype(jnp.float32)
            step_int = per_step_input.astype(jnp.int32)

            # Compute SDE components
            beta_scaler = softplus(actor_params["beta_scaler"])
            beta_t = jnp.clip(betas(step_float) * beta_scaler, 0.0, 100.0)
            # beta_t = jax.lax.stop_gradient(jnp.clip(betas(step_float) * softplus(actor_params["beta_scaler"]), 0.0, 100.0))

            model_output = actor_state.apply_fn[0](actor_params["score_params"], x, observations,
                                                   step_float * jnp.ones(1))

            key, key_gen = jax.random.split(key_gen)
            noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

            # Euler-Maruyama integration of the SDE
            f = (x - stat_distr.loc) / (jnp.power(stat_distr.scale.diag, 2))
            mean = x - (f - 2 * model_output) * beta_t * dt  # TODO: with absorbing
            # mean = x + (f + 2 * model_output) * beta_t * dt
            std = jnp.sqrt(2 * beta_t * dt)
            x_new = mean + std * noise
            f_new = (x_new - stat_distr.loc) / (jnp.power(stat_distr.scale.diag, 2))

            fwd_kernel = distrax.MultivariateNormalDiag(x_new - beta_t*dt*f_new, jnp.ones(a_dim)*std)
            bwd_kernel = distrax.MultivariateNormalDiag(mean, jnp.ones(a_dim)*std)

            running_cost = bwd_kernel.log_prob(x_new) - fwd_kernel.log_prob(x)

            stochastic_cost = 0.0

            running_cost, x_new, log_prob = jax.lax.cond(
                step_int == n_steps - 1,
                lambda _: (
                    running_cost - tfp.bijectors.Tanh().forward_log_det_jacobian(x_new).sum(),
                    tfp.bijectors.Tanh().forward(x_new),
                    TanhTransformedDistribution(tfd.MultivariateNormalDiag(mean, jnp.ones(mean.shape) * std)).log_prob(
                        tfp.bijectors.Tanh().forward(x_new), ),
                ),
                lambda _: (
                    running_cost,
                    x_new,
                    distrax.MultivariateNormalDiag(mean, jnp.ones(mean.shape) * std).log_prob(x_new),
                ),
                operand=None
            )
            next_state = (x_new, key_gen)
            per_step_output = (running_cost, stochastic_cost, log_prob, x_new, mean, std, beta_scaler)
            return next_state, per_step_output

        key, key_gen = jax.random.split(key)
        init_x = stat_distr.sample(seed=key)

        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        final_x, _ = aux
        if n_steps != 0:
            aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(0, n_steps))
            running_cost, stochastic_cost, log_probs, x_t, means, stds, betas = per_step_output
            init_log_prob_loss = stat_distr.log_prob(init_x)
            final_x, _ = aux
        else:
            running_cost = jnp.array([0.0])
            stochastic_cost = jnp.array([0.0])
            log_probs = jnp.array([0.0])
            betas = jnp.array([0.0])
            init_log_prob_loss = log_probs
            stds = jnp.array([stat_distr.scale.diag])
            means = jnp.array([stat_distr.loc])
            x_t = jnp.array([init_x])
            # squash the latest action and apply change of variables
            final_x = tfp.bijectors.Tanh().forward(final_x)
            tmp_std = jnp.ones(means[-1].shape) * stds[-1] if stds[-1].shape[0] == 1 else stds[-1]
            trans_dist = TanhTransformedDistribution(tfd.MultivariateNormalDiag(loc=means[-1], scale_diag=tmp_std))
            log_probs = log_probs.at[-1].set(trans_dist.log_prob(final_x))
        return final_x, running_cost, stochastic_cost, log_probs, x_t, init_x, init_log_prob_loss.squeeze(), betas

    @staticmethod
    @partial(jax.jit, static_argnames=["n_steps", "a_dim", "return_logprob"])
    def sample_action(actor_state, params, observations, key, n_steps, a_dim, return_logprob=True):
        # noise_schedule = get_cosine_noise_schedule(n_steps, sigma_max=10.0, reverse=False)  # TODO: include into configs -> more general
        # noise_schedule = get_cosine_noise_schedule(n_steps, sigma_max=1.0, reverse=False)  # TODO: include into configs -> more general
        noise_schedule = get_cosine_noise_schedule(n_steps, sigma_max=1.0, sigma_min=0.001, reverse=False)  # TODO: include into configs -> more general
        batch_keys = jax.random.split(key, num=observations.shape[0])
        in_axes = (0, 0, None, None, None, None, None)
        out = (jax.vmap(GenDPOL._sample_action_single, in_axes=in_axes)(batch_keys, observations, actor_state, params,
                                                                        noise_schedule, a_dim, n_steps))
        x_0, running_costs, stochastic_costs, log_probs, x_t, init_x, init_log_prob_loss, betas = out
        return (x_0, x_t, init_x), log_probs, running_costs.sum(1), stochastic_costs.sum(1), init_log_prob_loss, betas

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, *_ = GenDPOL.sample_action(self.actor_state, self.actor_state.params, observation, self.noise_key,
                                            a_dim=self.dim,
                                            n_steps=self.diff_steps)
        return actions[0]

    def _predict2(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, *_ = GenDPOL.sample_action(self.actor_state, self.actor_state.params, observation, self.noise_key,
                                            a_dim=self.dim,
                                            n_steps=self.diff_steps)
        return actions

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)


    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a,
                rngs={"dropout": dropout_key},
                train=False
            )

        return jax.jit(Q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )
