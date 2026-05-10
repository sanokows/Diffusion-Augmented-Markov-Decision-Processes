import logging
import math
import time
import typing
from typing import Callable, Optional

import distrax
import hydra
import jax
import optax
import plotly.graph_objs as go
from hydra.core.hydra_config import HydraConfig
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.experimental import checkify
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
)
from src.jaxrl import utils
from src.jaxrl.normalization import NormalizationState, Normalizer

logging.basicConfig(level=logging.INFO)


def _rsl_gaussian_kl(
    old_mu: jax.Array,
    old_sigma: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    sigma = jnp.maximum(sigma, 1e-8)
    old_sigma = jnp.maximum(old_sigma, 1e-8)
    kl = jnp.sum(
        jnp.log(sigma / old_sigma + 1e-5)
        + (jnp.square(old_sigma) + jnp.square(old_mu - mu)) / (2.0 * jnp.square(sigma))
        - 0.5,
        axis=-1,
    )
    return kl


def _replace_learning_rate_in_opt_state(
    opt_state: typing.Any,
    learning_rate: jax.Array,
) -> typing.Any:
    if hasattr(opt_state, "hyperparams") and hasattr(opt_state, "_replace"):
        hyperparams = dict(opt_state.hyperparams)
        if "learning_rate" in hyperparams:
            hyperparams["learning_rate"] = jnp.asarray(learning_rate, dtype=jnp.float32)
            return opt_state._replace(hyperparams=hyperparams)
    if isinstance(opt_state, tuple) and hasattr(opt_state, "_fields"):
        updated = tuple(
            _replace_learning_rate_in_opt_state(state, learning_rate) for state in opt_state
        )
        try:
            return type(opt_state)(*updated)
        except TypeError:
            return opt_state._replace(**dict(zip(opt_state._fields, updated)))
    if isinstance(opt_state, tuple):
        return tuple(
            _replace_learning_rate_in_opt_state(state, learning_rate) for state in opt_state
        )
    return opt_state


def _update_adaptive_lr(
    cfg: "PPOConfig", current_lr: jax.Array, kl_mean: jax.Array
) -> jax.Array:
    if not cfg.adaptive_lr:
        return current_lr
    decreased_lr = jnp.maximum(
        jnp.asarray(cfg.adaptive_lr_min, dtype=jnp.float32),
        current_lr / jnp.asarray(cfg.adaptive_lr_factor, dtype=jnp.float32),
    )
    increased_lr = jnp.minimum(
        jnp.asarray(cfg.adaptive_lr_max, dtype=jnp.float32),
        current_lr * jnp.asarray(cfg.adaptive_lr_factor, dtype=jnp.float32),
    )
    current_lr = jnp.where(kl_mean > cfg.desired_kl * 2.0, decreased_lr, current_lr)
    current_lr = jnp.where(
        (kl_mean < cfg.desired_kl / 2.0) & (kl_mean > 0.0),
        increased_lr,
        current_lr,
    )
    return current_lr


def _reapply_cli_hyperparameter_overrides(cfg: DictConfig) -> DictConfig:
    try:
        task_overrides = HydraConfig.get().overrides.task
    except Exception:
        return cfg

    hyperparam_dotlist = []
    for override in task_overrides:
        cleaned = override.lstrip("+")
        if "=" not in cleaned:
            continue
        key, _ = cleaned.split("=", 1)
        if key.startswith("hyperparameters."):
            hyperparam_dotlist.append(cleaned)

    if hyperparam_dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(hyperparam_dotlist))
    return cfg


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
        state: Optional[PyTreeNode] = None,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        ...


class PPOConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    lmbda: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    total_time_steps: int
    num_steps: int
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_advantages: bool
    normalize_env: bool
    anneal_lr: bool
    normalize_rewards: bool = False
    reward_norm_epsilon: float = 1e-8
    reward_norm_clip: float = 10.0
    adaptive_lr: bool = False
    desired_kl: float = 0.01
    adaptive_lr_factor: float = 1.5
    adaptive_lr_min: float = 1e-5
    adaptive_lr_max: float = 1e-2
    num_eval: int = 25
    max_episode_steps: int = 1000


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    raw_reward: jax.Array
    log_prob: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    action_mean: jax.Array
    action_sigma: jax.Array
    info: dict[str, jax.Array]


class PPOTrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
    normalization_state: NormalizationState | None = None
    critic_normalization_state: NormalizationState | None = None
    reward_normalization_state: NormalizationState | None = None
    discounted_return: jax.Array | None = None
    learning_rate: jax.Array = struct.field(
        pytree_node=True,
        default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float32),
    )


class PPONetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        def linear_layer(in_features, out_features, scale=jnp.sqrt(2)):
            return nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                kernel_init=nnx.initializers.orthogonal(scale=scale),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        self.actor_module = nnx.Sequential(
            linear_layer(obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, action_dim, scale=0.01),
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))
        self.critic_module = nnx.Sequential(
            linear_layer(critic_obs_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, hidden_dim),
            nnx.tanh,
            linear_layer(hidden_dim, 1, scale=1.0),
        )

    def critic(self, obs: jax.Array) -> jax.Array:
        return self.critic_module(obs).squeeze()

    def actor(self, obs: jax.Array) -> distrax.Distribution:
        loc = self.actor_module(obs)
        pi = distrax.MultivariateNormalDiag(
            loc=loc, scale_diag=jnp.exp(self.log_std.value)
        )
        return pi


class ReppoPPOTrainer:
    """Trainer wrapper for PPO using the existing mjx implementation."""

    def __init__(
        self,
        cfg: PPOConfig,
        env: Environment,
        env_params: EnvParams | None = None,
        log_callback: Callable[[PPOTrainState, dict[str, jax.Array]], None] | None = None,
        num_seeds: int = 1,
    ) -> None:
        self.cfg = cfg
        self.env_params = env_params
        self.num_seeds = num_seeds
        self.log_callback = log_callback or (lambda *args: None)
        self.env = self._prepare_env(env)
        self.normalizer = Normalizer()
        self.reward_normalizer = Normalizer()
        self.eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        self.eval_fn = self._make_eval_fn(cfg.max_episode_steps)

    def _prepare_env(self, env: Environment) -> Environment:
        wrapped_env = ClipAction(env)
        wrapped_env = LogWrapper(wrapped_env, self.cfg.num_envs)
        return wrapped_env

    def _make_policy(self, train_state: PPOTrainState) -> Policy:
        normalizer = self.normalizer

        def policy(
            key: PRNGKey, obs: jax.Array, state: struct.PyTreeNode = None
        ) -> tuple[jax.Array, jax.Array]:
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)
            model = nnx.merge(train_state.graphdef, train_state.params)
            pi = model.actor(obs)
            value = model.critic(obs)
            action = pi.sample(seed=key)
            log_prob = pi.log_prob(action)
            return action, dict(log_prob=log_prob, value=value)

        return policy

    def _make_eval_fn(
        self, max_episode_steps: int
    ) -> Callable[[jax.random.PRNGKey, Policy], dict[str, float]]:
        env = self.env

        def evaluation_fn(key: jax.random.PRNGKey, policy: Policy):
            def step_env(carry, _):
                key, env_state, obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                action, _ = policy(act_key, obs)
                env_key = jax.random.split(env_key, env.num_envs)
                obs, _, env_state, reward, done, info = env.step(
                    env_key, env_state, action.clip(-1.0 + 1e-4, 1.0 - 1e-4)
                )
                return (key, env_state, obs), info

            key, init_key = jax.random.split(key)
            init_key = jax.random.split(init_key, env.num_envs)
            obs, _, env_state = env.reset(init_key)
            _, infos = jax.lax.scan(
                f=step_env,
                init=(key, env_state, obs),
                xs=None,
                length=max_episode_steps,
            )

            return {
                "episode_return": infos["returned_episode_returns"].mean(
                    where=infos["returned_episode"]
                ),
                "episode_return_std": infos["returned_episode_returns"].std(
                    where=infos["returned_episode"]
                ),
                "episode_length": infos["returned_episode_lengths"].mean(
                    where=infos["returned_episode"]
                ),
                "episode_length_std": infos["returned_episode_lengths"].std(
                    where=infos["returned_episode"]
                ),
                "num_episodes": infos["returned_episode"].sum(),
            }

        return evaluation_fn

    def _make_init_fn(self) -> Callable[[jax.random.PRNGKey], PPOTrainState]:
        cfg = self.cfg
        env = self.env
        env_params = self.env_params

        def init(key: jax.random.PRNGKey) -> PPOTrainState:
            num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
            eval_interval = self.eval_interval
            num_iterations = num_train_steps // eval_interval + int(
                num_train_steps % eval_interval != 0
            )
            key, model_key = jax.random.split(key)
            networks = PPONetworks(
                obs_dim=env.observation_space(env_params)[0].shape[0],
                critic_obs_dim=env.observation_space(env_params)[1].shape[0],
                action_dim=env.action_space(env_params).shape[0],
                rngs=nnx.Rngs(model_key),
            )

            if cfg.anneal_lr and not cfg.adaptive_lr:
                num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
                num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
                lr = optax.linear_schedule(cfg.lr, 1e-6, num_updates)
            else:
                lr = cfg.lr

            adam_tx = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
            if cfg.max_grad_norm is not None:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(cfg.max_grad_norm),
                    adam_tx,
                )
            else:
                optimizer = adam_tx

            key, env_key = jax.random.split(key)
            env_key = jax.random.split(env_key, cfg.num_envs)
            obs, critic_obs, env_state = env.reset(env_key)
            _env_state = env_state.unwrapped()
            key, randomize_steps_key = jax.random.split(key)
            _env_state.info["steps"] = jax.random.randint(
                randomize_steps_key,
                _env_state.info["steps"].shape,
                0,
                cfg.max_episode_steps,
            ).astype(jnp.float32)
            env_state.set_env_state(_env_state)

            if cfg.normalize_env:
                normalizer = Normalizer()
                norm_state = normalizer.init(obs)
                critic_normalizer = Normalizer()
                critic_norm_state = critic_normalizer.init(critic_obs)
            else:
                norm_state = None
                critic_norm_state = None
            if cfg.normalize_rewards:
                reward_norm_state = self.reward_normalizer.init(
                    jnp.zeros((cfg.num_envs,), dtype=jnp.float32)
                )
                discounted_return = jnp.zeros((cfg.num_envs,), dtype=jnp.float32)
            else:
                reward_norm_state = None
                discounted_return = jnp.zeros((cfg.num_envs,), dtype=jnp.float32)

            return PPOTrainState.create(
                iteration=0,
                time_steps=0,
                graphdef=nnx.graphdef(networks),
                params=nnx.state(networks),
                tx=optimizer,
                last_env_state=env_state,
                last_obs=obs,
                last_critic_obs=critic_obs,
                normalization_state=norm_state,
                critic_normalization_state=critic_norm_state,
                reward_normalization_state=reward_norm_state,
                discounted_return=discounted_return,
                learning_rate=jnp.asarray(cfg.lr, dtype=jnp.float32),
            )

        return init

    def _collect_rollout(
        self, key: PRNGKey, train_state: PPOTrainState
    ) -> tuple[Transition, PPOTrainState]:
        cfg = self.cfg
        env = self.env
        normalizer = self.normalizer
        model = nnx.merge(train_state.graphdef, train_state.params)

        def step_env(carry, _):
            key, env_state, train_state, obs, critic_obs = carry

            if cfg.normalize_env:
                norm_state = normalizer.update(train_state.normalization_state, obs)
                critic_norm_state = normalizer.update(
                    train_state.critic_normalization_state, critic_obs
                )
                obs_norm = normalizer.normalize(norm_state, obs)
                critic_obs_norm = normalizer.normalize(critic_norm_state, critic_obs)
                train_state = train_state.replace(
                    normalization_state=norm_state,
                    critic_normalization_state=critic_norm_state,
                )
            else:
                obs_norm = obs
                critic_obs_norm = critic_obs

            key, act_key, step_key = jax.random.split(key, 3)
            pi = model.actor(obs_norm)
            action = pi.sample(seed=act_key)
            step_key = jax.random.split(step_key, cfg.num_envs)
            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action.clip(-1.0 + 1e-4, 1.0 - 1e-4)
            )
            raw_reward = reward
            if cfg.normalize_rewards:
                episode_done = jnp.logical_or(done > 0, next_env_state.truncated > 0).astype(
                    jnp.float32
                )
                discounted_return = raw_reward + (
                    cfg.gamma * train_state.discounted_return * (1.0 - episode_done)
                )
                reward_norm_state = self.reward_normalizer.update(
                    train_state.reward_normalization_state, discounted_return
                )
                reward_scale = jnp.sqrt(reward_norm_state.var + cfg.reward_norm_epsilon)
                reward = jnp.clip(
                    raw_reward / reward_scale,
                    -cfg.reward_norm_clip,
                    cfg.reward_norm_clip,
                )
                train_state = train_state.replace(
                    reward_normalization_state=reward_norm_state,
                    discounted_return=discounted_return,
                )
            transition = Transition(
                obs=obs_norm,
                critic_obs=critic_obs_norm,
                action=action,
                reward=reward,
                raw_reward=raw_reward,
                log_prob=pi.log_prob(action),
                value=model.critic(critic_obs_norm),
                done=done,
                truncated=next_env_state.truncated,
                action_mean=pi.mean(),
                action_sigma=pi.stddev(),
                info=info,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def _learn_step(
        self, key: PRNGKey, train_state: PPOTrainState, batch: Transition
    ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
        cfg = self.cfg
        normalizer = self.normalizer
        model = nnx.merge(train_state.graphdef, train_state.params)

        if cfg.normalize_env:
            last_critic_obs = normalizer.normalize(
                train_state.critic_normalization_state, train_state.last_critic_obs
            )
        else:
            last_critic_obs = train_state.last_critic_obs
        last_value = model.critic(last_critic_obs)

        def compute_advantage(carry, transition):
            gae, next_value = carry
            done = transition.done
            truncated = transition.truncated
            reward = transition.reward
            value = transition.value
            delta = reward + cfg.gamma * next_value * (1 - done) - value
            gae = delta + cfg.gamma * cfg.lmbda * (1 - done) * gae
            truncated_gae = reward + cfg.gamma * next_value - value
            gae = jnp.where(truncated, truncated_gae, gae)
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            compute_advantage,
            (jnp.zeros_like(last_value), last_value),
            batch,
            reverse=True,
        )
        target_values = advantages + batch.value

        data = (batch, advantages, target_values)
        data = jax.tree.map(
            lambda x: x.reshape(
                (math.floor(cfg.num_steps * cfg.num_envs), *x.shape[2:])
            ),
            data,
        )

        def update(train_state, key):
            def minibatch_update(carry, indices):
                idx, train_state = carry
                minibatch, advantages, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )
                if cfg.normalize_advantages:
                    advantages = (advantages - jnp.mean(advantages)) / (
                        jnp.std(advantages) + 1e-8
                    )

                def loss_fn(params):
                    model = nnx.merge(train_state.graphdef, params)
                    pi = model.actor(minibatch.obs)
                    value = model.critic(minibatch.critic_obs)
                    log_prob = pi.log_prob(minibatch.action)
                    mu_batch = pi.mean()
                    sigma_batch = pi.stddev()
                    value_pred_clipped = minibatch.value + (
                        value - minibatch.value
                    ).clip(-cfg.clip_ratio, cfg.clip_ratio)
                    value_error = jnp.square(value - target_values)
                    value_error_clipped = jnp.square(value_pred_clipped - target_values)
                    value_loss = 0.5 * jnp.mean(
                        (1.0 - minibatch.truncated)
                        * jnp.maximum(value_error, value_error_clipped)
                    )

                    ratio = jnp.exp(log_prob - minibatch.log_prob)
                    checkify.check(
                        jnp.allclose(ratio, 1.0) | (idx != 1),
                        debug=True,
                        msg="Ratio not equal to 1 on first iteration: {r}",
                        r=ratio,
                    )

                    actor_loss1 = ratio * advantages
                    actor_loss2 = (
                        jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                        * advantages
                    )
                    actor_loss = -jnp.mean(
                        (1.0 - minibatch.truncated)
                        * jnp.minimum(actor_loss1, actor_loss2)
                    )
                    entropy_loss = jnp.mean(pi.entropy())
                    kl_mean = jnp.mean(
                        _rsl_gaussian_kl(
                            old_mu=minibatch.action_mean,
                            old_sigma=minibatch.action_sigma,
                            mu=mu_batch,
                            sigma=sigma_batch,
                        )
                    )

                    loss = (
                        actor_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_loss
                    )

                    return loss, dict(
                        actor_loss=actor_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        loss=loss,
                        mean_value=value.mean(),
                        mean_log_prob=log_prob.mean(),
                        mean_advantages=advantages.mean(),
                        mean_action=minibatch.action.mean(),
                        reward_mean=minibatch.raw_reward.mean(),
                        kl_mean=kl_mean,
                    )

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                output, grads = grad_fn(train_state.params)

                flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
                global_grad_norm = jnp.linalg.norm(flat_grads)

                metrics = output[1]
                metrics["advantages"] = advantages
                metrics["global_grad_norm"] = global_grad_norm
                new_lr = _update_adaptive_lr(
                    cfg, train_state.learning_rate, metrics["kl_mean"]
                )
                if cfg.adaptive_lr:
                    train_state = train_state.replace(
                        opt_state=_replace_learning_rate_in_opt_state(
                            train_state.opt_state, new_lr
                        )
                    )
                train_state = train_state.replace(learning_rate=new_lr)
                train_state = train_state.apply_gradients(grads)
                metrics["learning_rate"] = train_state.learning_rate
                return (idx + 1, train_state), metrics

            key, shuffle_key = jax.random.split(key)

            mini_batch_size = (
                math.floor(cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            )
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics = jax.lax.scan(
            f=update,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    def _train_eval_step(self, key, train_state):
        def train_step(
            state: PPOTrainState, key: PRNGKey
        ) -> tuple[PPOTrainState, dict[str, jax.Array]]:
            key, rollout_key, learn_key = jax.random.split(key, 3)
            transitions, state = self._collect_rollout(
                key=rollout_key, train_state=state
            )
            state, update_metrics = self._learn_step(
                key=learn_key, train_state=state, batch=transitions
            )
            metrics = dict(update_metrics)
            if state.reward_normalization_state is not None:
                metrics["reward_norm_count"] = jnp.asarray(
                    state.reward_normalization_state.count, dtype=jnp.float32
                )
                metrics["reward_norm_std"] = jnp.sqrt(
                    jnp.mean(state.reward_normalization_state.var)
                )
            state = state.replace(iteration=state.iteration + 1)
            return state, metrics

        eval_interval = self.eval_interval
        train_key, eval_key = jax.random.split(key)
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        policy = self._make_policy(train_state)
        eval_metrics = self.eval_fn(eval_key, policy)
        metrics = {
            "time_step": train_state.time_steps,
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

        return train_state, metrics

    def _loop_body(
        self, train_state: PPOTrainState, key: PRNGKey
    ) -> tuple[PPOTrainState, dict]:
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(self._train_eval_step)(
            jax.random.split(subkey, self.num_seeds), train_state
        )
        jax.debug.callback(self.log_callback, train_state, metrics)
        return train_state, metrics

    def _train_loop(self, key: PRNGKey) -> tuple[PPOTrainState, dict]:
        cfg = self.cfg
        eval_interval = self.eval_interval
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )

        key, init_key = jax.random.split(key)
        init_fn = self._make_init_fn()
        train_state = jax.vmap(init_fn)(jax.random.split(init_key, self.num_seeds))

        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(
            f=self._loop_body,
            init=train_state,
            xs=keys,
        )
        return state, metrics

    def build_train_fn(self) -> Callable[[PRNGKey, PPOConfig], tuple[PPOTrainState, dict]]:
        def train_fn(key: PRNGKey, cfg: PPOConfig):
            if cfg != self.cfg:
                logging.warning(
                    "Received cfg argument different from trainer configuration; using trainer cfg."
                )
            return self._train_loop(key)

        return train_fn


def plot_history(history: list[dict[str, jax.Array]]):
    steps = jnp.array([m["time_step"][0] for m in history])
    eval_return = jnp.array([m["eval/episode_return"].mean() for m in history])
    eval_return_std = jnp.array([m["eval/episode_return"].std() for m in history])
    fig = go.Figure(
        [
            go.Scatter(
                x=steps,
                y=eval_return,
                name="Mean Episode Return",
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return + eval_return_std,
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return - eval_return_std,
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(50, 127, 168, 0.3)",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        xaxis=dict(title=dict(text="Environment Steps")),
    )

    return fig


def run(cfg: DictConfig):
    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        advantages = metrics.pop("train/advantages", None)
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "sps": sps,
            "train/advantages": wandb.Histogram(advantages),
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        wandb.log(log_data, step=state.time_steps[0])

    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(cfg.env.name)
    elif cfg.env.type == "mjx":
        env_config = OmegaConf.select(cfg, "env.config")
        if env_config is not None:
            env_config = OmegaConf.to_container(env_config, resolve=True)
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            config=env_config,
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    trainer = ReppoPPOTrainer(
        cfg=PPOConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
    )
    train_fn = trainer.build_train_fn()

    key = jax.random.PRNGKey(cfg.seed)
    for i in range(cfg.trials):
        key, train_key = jax.random.split(key)
        wandb.init(
            mode=cfg.wandb.mode,
            project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}",
            entity=cfg.wandb.entity,
            tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
            config=OmegaConf.to_container(cfg),
            name=f"ppo-{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )
        start = time.perf_counter()
        _, metrics = jax.jit(train_fn, static_argnums=(1,))(train_key, trainer.cfg)
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        logging.info(f"Training took {duration:.2f} seconds.")
        wandb.finish()


def tune(cfg: DictConfig):
    def log_callback(state, metrics):
        episode_return = metrics["eval/episode_return"].mean()
        t = state.time_steps[0]
        wandb.log(
            {
                "episode_return": episode_return,
            },
            step=t,
        )

    env_config = OmegaConf.select(cfg, "env.config")
    if env_config is not None:
        env_config = OmegaConf.to_container(env_config, resolve=True)
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        config=env_config,
    )

    def train_agent():
        wandb.init(project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}")
        run_cfg = OmegaConf.to_container(cfg)
        for k, v in dict(wandb.config).items():
            run_cfg["hyperparameters"][k] = v
        ppo_cfg = PPOConfig(**run_cfg["hyperparameters"])
        trainer = ReppoPPOTrainer(
            cfg=ppo_cfg,
            env=env,
            log_callback=log_callback,
            num_seeds=cfg.num_seeds,
        )
        train_fn = trainer.build_train_fn()
        train_fn = jax.jit(train_fn, static_argnums=(1,))
        logging.info(f"Running experiment with params: \n {run_cfg}")
        key = jax.random.PRNGKey(cfg.seed)
        _, metrics = train_fn(key, trainer.cfg)
        jax.block_until_ready(metrics)

    sweep_id = wandb.sweep(
        sweep={
            "name": f"{cfg.name}-{cfg.env.name}",
            "method": "bayes",
            "metric": {"name": "episode_return", "goal": "maximize"},
            "parameters": {
                "lr": {
                    "values": [1e-4, 3e-4, 1e-3],
                },
                "normalize_env": {
                    "values": [True, False],
                },
            },
        },
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
    )
    wandb.agent(sweep_id, function=train_agent, count=cfg.tune.num_runs)


@hydra.main(version_base=None, config_path="../../config", config_name="ppo")
def main(cfg: DictConfig):
    ppo_overrides = OmegaConf.select(cfg, "PPO_overrides.hyperparameters", default={})
    if OmegaConf.is_config(ppo_overrides):
        ppo_overrides = OmegaConf.create(
            OmegaConf.to_container(ppo_overrides, resolve=False)
        )
    else:
        ppo_overrides = OmegaConf.create(ppo_overrides)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, ppo_overrides)
    cfg = _reapply_cli_hyperparameter_overrides(cfg)
    if cfg.tune:
        tune(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
