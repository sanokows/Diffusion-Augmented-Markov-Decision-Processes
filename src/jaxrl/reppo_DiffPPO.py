import logging
import math
import time
import typing
from typing import Callable, Optional
import functools

import distrax
import numpy as np
import hydra
import jax
import optax
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.experimental import checkify
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    TanhClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    MjxDiffEnvWrapper
)
from src.networks.diffusion.models import ControlNetwork
from src.networks.jax_models_DMERL import (
    DiffValueNetwork,
    DiffusionModel,
    DMERLActor,
    logratio_DIME as logratio,
    ode_integrator,
    sde_integrator,
)
from src.jaxrl import utils
from src.jaxrl.normalization import DictNormalizationState, DictNormalizer
from src.jaxrl.reppo_DMERL_new import randomize_env_steps


logging.basicConfig(level=logging.INFO)


def require(cfg, key):
    if cfg is None:
        raise KeyError(f"Missing required config key '{key}'")
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        raise KeyError(f"Missing required config key '{key}'")
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "__getitem__"):
        try:
            return cfg[key]
        except Exception as exc:
            raise KeyError(f"Missing required config key '{key}'") from exc
    raise KeyError(f"Missing required config key '{key}'")


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
    diffusion: DictConfig | dict | None = None
    num_eval: int = 25
    max_episode_steps: int = 1000
    env_action_clip_value: float = 1.0
    critic_hidden_dim: int = 512
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    action_clip_value: float = 1.0
    kl_start: float = 0.1
    kl_bound: float = 1.0
    kl_action_rep: int = 1
    reduce_kl: bool = True
    reverse_kl: bool = False
    ent_start: float = 0.1
    ent_target_mult: float = 0.5
    update_entropy_lagrangian: bool = False
    use_kl_regularization: bool = False
    actor_kl_clip_mode: str = "clipped"
    use_clipped_objective: bool = True
    temp_lagrangian_adam_gamma1: float = 0.9
    temp_lagrangian_adam_gamma2: float = 0.999
    weight_decay: float = 0.0
    use_temp_lagrangian_mlp: bool = False
    temp_lagrangian_hidden: int = 64


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    log_prob: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    info: dict[str, jax.Array]


class PPOTrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
    normalization_state: DictNormalizationState | None = None
    critic_normalization_state: DictNormalizationState | None = None


class PPONetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        cfg: PPOConfig | DictConfig | dict | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        diff_cfg = require(cfg, "diffusion")
        score_cfg = require(diff_cfg, "score_model")
        forward_model = None
        if require(diff_cfg, "learn_forward"):
            forward_model = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=require(score_cfg, "num_layers"),
                num_hid=require(score_cfg, "num_hid"),
                num_time_hid=require(score_cfg, "num_time_hid"),
                num_time_out=require(score_cfg, "num_time_out"),
                outer_clip=require(score_cfg, "outer_clip"),
                inner_clip=require(score_cfg, "inner_clip"),
                weight_init=require(score_cfg, "weight_init"),
                bias_init=require(score_cfg, "bias_init"),
                layer_norm=require(score_cfg, "layer_norm"),
                layer_norm_type=require(score_cfg, "layer_norm_type"),
                rngs=rngs,
            )

        backward_model = None
        if require(diff_cfg, "learn_backward"):
            backward_model = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=require(score_cfg, "num_layers"),
                num_hid=require(score_cfg, "num_hid"),
                num_time_hid=require(score_cfg, "num_time_hid"),
                num_time_out=require(score_cfg, "num_time_out"),
                outer_clip=require(score_cfg, "outer_clip"),
                inner_clip=require(score_cfg, "inner_clip"),
                weight_init=require(score_cfg, "weight_init"),
                bias_init=require(score_cfg, "bias_init"),
                layer_norm=require(score_cfg, "layer_norm"),
                layer_norm_type=require(score_cfg, "layer_norm_type"),
                rngs=rngs,
            )

        if require(diff_cfg, "use_step_size_scheduler"):
            dt_schedule_cfg = require(diff_cfg, "dt_schedule")
            dt_schedule = (
                hydra.utils.instantiate(dt_schedule_cfg)
                if dt_schedule_cfg is not None
                else lambda step: 1.0
            )
        else:
            dt_schedule = lambda step: 1.0

        diffusion_model = DiffusionModel(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            bwd_model=backward_model,
            diff_steps=require(diff_cfg, "diff_steps"),
            init_std=require(diff_cfg, "init_std"),
            friction=require(diff_cfg, "friction"),
            per_dim_friction=require(diff_cfg, "per_dim_friction"),
            use_friction_mlp=require(diff_cfg, "use_friction_mlp"),
            friction_mlp_hidden=require(diff_cfg, "friction_mlp_hidden"),
            friction_mlp_layers=require(diff_cfg, "friction_mlp_layers"),
            friction_num_time_hid=require(diff_cfg, "friction_num_time_hid"),
            friction_num_time_out=require(diff_cfg, "friction_num_time_out"),
            friction_mlp_use_obs=require(diff_cfg, "friction_mlp_use_obs"),
            dt=require(diff_cfg, "dt"),
            learn_dt=require(diff_cfg, "learn_dt"),
            per_step_dt=require(diff_cfg, "per_step_dt"),
            learn_prior=require(diff_cfg, "learn_prior"),
            learn_betas=require(diff_cfg, "learn_betas"),
            learn_friction=require(diff_cfg, "learn_friction"),
            learn_mass_matrix=require(diff_cfg, "learn_mass_matrix"),
            dt_schedule=dt_schedule,
            rngs=rngs,
        )

        critic_hidden_dim = require(cfg, "critic_hidden_dim")
        self.actor_module = DMERLActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            sde_integrator=sde_integrator,
            ode_integrator=ode_integrator,
            logratio=logratio,
            kl_start=require(cfg, "kl_start"),
            ent_start=require(cfg, "ent_start"),
            action_clip_value=require(cfg, "action_clip_value"),
            use_temp_lagrangian_mlp=require(cfg, "use_temp_lagrangian_mlp"),
            temp_lagrangian_hidden=require(cfg, "temp_lagrangian_hidden"),
            rngs=rngs,
        )
        self.critic_module = DiffValueNetwork(
            obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            num_time_hid=require(score_cfg, "num_time_hid"),
            num_time_out=require(score_cfg, "num_time_out"),
            use_norm=require(cfg, "use_critic_norm"),
            encoder_layers=require(cfg, "num_critic_encoder_layers"),
            head_layers=require(cfg, "num_critic_head_layers"),
            pred_layers=require(cfg, "num_critic_pred_layers"),
            use_simplical_embedding=require(cfg, "use_simplical_embedding"),
            use_skip=require(cfg, "use_critic_skip"),
            rngs=rngs,
        )

    def critic(self, obs: jax.Array) -> jax.Array:
        return self.critic_module.critic(obs).squeeze()

    def actor_log_prob_step(self, obs_dict, actions) -> distrax.Distribution:
        return self.actor_module.vmap_eval_log_prob(obs_dict, actions)
    
    def actor_sample_step(self, obs_dict, key) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.actor_module.vmap_sample_next_step(obs_dict, key)


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
        action_shape = jnp.prod(jnp.array(self.env.action_space(env_params).shape))
        self.action_size_target = action_shape * cfg.ent_target_mult
        diff_cfg = require(cfg, "diffusion")

        self.diffusion_steps = require(diff_cfg, "diff_steps")
        self.eval_env_steps = cfg.max_episode_steps * self.diffusion_steps
        self.num_collection_steps = cfg.num_steps * self.diffusion_steps
        self.num_minibatches = cfg.num_mini_batches * self.diffusion_steps
        self.normalizer = DictNormalizer()
        self.eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        self.eval_fn = self._make_eval_fn(cfg.max_episode_steps)

    def _prepare_env(self, env: Environment) -> Environment:
        wrapped_env = TanhClipAction(env)
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
            value = model.critic(obs)
            action, gen_log_prob, _ = model.actor_sample_step(obs, key)
            return action, dict(log_prob=gen_log_prob, value=value)

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
                    env_key, env_state, action
                )
                return (key, env_state, obs), info

            key, init_key = jax.random.split(key)
            init_key = jax.random.split(init_key, env.num_envs)
            obs, _, env_state = env.reset(init_key)
            _, infos = jax.lax.scan(
                f=step_env,
                init=(key, env_state, obs),
                xs=None,
                length=self.eval_env_steps,
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
            if hasattr(env, "get_obs_space_sizes"):
                obs_dim, critic_obs_dim = env.get_obs_space_sizes(env_params)
            else:
                obs_dim = env.observation_space(env_params)[0].shape[0]
                critic_obs_dim = env.observation_space(env_params)[1].shape[0]
            networks = PPONetworks(
                obs_dim=obs_dim,
                critic_obs_dim=critic_obs_dim,
                action_dim=env.action_space(env_params).shape[0],
                hidden_dim=require(cfg, "critic_hidden_dim"),
                cfg=cfg,
                rngs=nnx.Rngs(model_key),
            )

            if not cfg.anneal_lr:
                lr = cfg.lr
            else:
                num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
                num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
                lr = optax.linear_schedule(cfg.lr, 1e-6, num_updates)

            def _adam_with_decay(
                lr_val, weight_decay: float = 0.0, decay_mask=None, optim=optax.adam
            ):
                tx = optim(lr_val)
                if weight_decay is not None and weight_decay > 0.0:
                    tx = optax.chain(
                        optax.add_decayed_weights(weight_decay, mask=decay_mask), tx
                    )
                return tx

            def _label_weight_decay(params):
                flat = flatten_dict(params)
                labels = {}
                for k in flat.keys():
                    leaf_name = k[-1]
                    if "temperature" in leaf_name or "lagrangian" in leaf_name:
                        labels[k] = "temp_lagrangian"
                    elif "bias" in leaf_name or "norm" in leaf_name:
                        labels[k] = "no_decay"
                    else:
                        labels[k] = "default"
                return unflatten_dict(labels)

            param_tree = nnx.to_pure_dict(nnx.state(networks))
            decay_labels = _label_weight_decay(param_tree)
            diff_cfg = require(cfg, "diffusion")
            if isinstance(diff_cfg, dict):
                diff_steps = diff_cfg.get("diff_steps", None)
            else:
                diff_steps = getattr(diff_cfg, "diff_steps", None)
            if diff_steps is not None and diff_steps > 0:
                scale = (cfg.num_mini_batches * 4) / (128.0 *cfg.num_epochs)
                temp_lagrangian_adam_gamma1 = cfg.temp_lagrangian_adam_gamma1**scale
                temp_lagrangian_adam_gamma2 = cfg.temp_lagrangian_adam_gamma2**scale
            else:
                temp_lagrangian_adam_gamma1 = cfg.temp_lagrangian_adam_gamma1
                temp_lagrangian_adam_gamma2 = cfg.temp_lagrangian_adam_gamma2
            special_optimizer = functools.partial(
                optax.adam,
                b1=temp_lagrangian_adam_gamma1,
                b2=temp_lagrangian_adam_gamma2,
            )
            tx_cfg = {
                "default": _adam_with_decay(lr, weight_decay=cfg.weight_decay),
                "no_decay": _adam_with_decay(lr, weight_decay=0.0),
                "temp_lagrangian": _adam_with_decay(
                    lr, weight_decay=0.0, optim=special_optimizer
                ),
            }
            optimizer = optax.multi_transform(tx_cfg, decay_labels)
            if cfg.max_grad_norm is not None:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(cfg.max_grad_norm),
                    optimizer,
                )

            key, env_key = jax.random.split(key)
            env_key = jax.random.split(env_key, cfg.num_envs)
            obs, critic_obs, env_state = env.reset(env_key)

            key, env_state = randomize_env_steps(
                key, env_state, cfg.max_episode_steps
            )

            if cfg.normalize_env:
                normalizer = DictNormalizer()
                norm_state = normalizer.init(obs)
                critic_normalizer = DictNormalizer()
                critic_norm_state = critic_normalizer.init(critic_obs)
                obs = normalizer.normalize(norm_state, obs)
                critic_obs = critic_normalizer.normalize(critic_norm_state, critic_obs)
            else:
                norm_state = None
                critic_norm_state = None

            return PPOTrainState.create(
                iteration=0,
                time_steps=0,
                graphdef=nnx.graphdef(networks),
                params=param_tree,
                tx=optimizer,
                last_env_state=env_state,
                last_obs=obs,
                last_critic_obs=critic_obs,
                normalization_state=norm_state,
                critic_normalization_state=critic_norm_state,
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
                obs = normalizer.normalize(norm_state, obs)
                train_state = train_state.replace(normalization_state=norm_state)
                critic_obs = normalizer.normalize(
                    train_state.critic_normalization_state, critic_obs
                )

            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)
            action, gen_log_prob, dest_log_prob = model.actor_sample_step(
                obs, act_key
            )

            log_ratio = jax.lax.stop_gradient(
                gen_log_prob - dest_log_prob
            )

            action = jax.lax.stop_gradient(action)
            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            if cfg.update_entropy_lagrangian:
                temperature = model.actor_module.temperature()
                entropy_scale = temperature
                soft_reward = (
                    reward
                    - cfg.gamma * log_ratio.squeeze() * entropy_scale
                )
            else:
                entropy_scale = cfg.entropy_coef
                soft_reward = (
                    reward
                    - log_ratio.squeeze() * entropy_scale
                )
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                reward=reward,
                soft_reward=soft_reward,
                log_prob=gen_log_prob,
                value=model.critic(critic_obs),
                done=done,
                truncated=next_env_state.truncated,
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
            length=self.num_collection_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + (self.num_collection_steps * cfg.num_envs)//self.diffusion_steps,
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
            reward = transition.soft_reward
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
                (math.floor(self.num_collection_steps * cfg.num_envs), *x.shape[2:])
            ),
            data,
        )

        def update(train_state, key):
            def minibatch_update(carry, scan_inputs):
                idx, train_state = carry
                indices, step_key = scan_inputs
                minibatch, advantages, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def loss_fn(params):
                    model = nnx.merge(train_state.graphdef, params)

                    gen_log_prob, dest_log_prob = model.actor_log_prob_step(minibatch.obs, minibatch.action)
                    value = model.critic(minibatch.critic_obs)

                    log_ratio = gen_log_prob - dest_log_prob
                    value_pred_clipped = minibatch.value + (
                        value - minibatch.value
                    ).clip(-cfg.clip_ratio, cfg.clip_ratio)
                    value_error = jnp.square(value - target_values)
                    value_error_clipped = jnp.square(value_pred_clipped - target_values)
                    value_loss = 0.5 * jnp.mean(
                        (1.0 - minibatch.truncated)
                        * jnp.maximum(value_error, value_error_clipped)
                    )

                    ratio = jnp.exp(gen_log_prob - minibatch.log_prob)
                    lagrangian = model.actor_module.lagrangian()
                    checkify.check(
                        jnp.allclose(ratio, 1.0) | (idx != 1),
                        debug=True,
                        msg="Ratio not equal to 1 on first iteration: {r}",
                        r=ratio,
                    )

                    adv_base = advantages

                    if cfg.update_entropy_lagrangian:
                        entropy_scale = jax.lax.stop_gradient(model.actor_module.temperature())
                    else:
                        entropy_scale = self.cfg.entropy_coef

                    unnormed_advantages = (
                        adv_base
                        - minibatch.soft_reward
                        + minibatch.reward
                        - log_ratio * entropy_scale
                    )
                    if (cfg.normalize_advantages):
                        adv_base = (unnormed_advantages - jnp.mean(unnormed_advantages)) / (
                            jnp.std(unnormed_advantages) + 1e-8
                        )
                    else:
                        adv_base = unnormed_advantages
                    # print adv_base, unnormed_advantages statistics, reward and log ratio and ratio
                    # jax.debug.print("adv_base mean: {}, std: {}", adv_base.mean(), adv_base.std())
                    # jax.debug.print("unnormed_advantages mean: {}, std: {}", unnormed_advantages.mean(), unnormed_advantages.std())
                    # jax.debug.print("reward mean: {}, log_ratio mean: {}", minibatch.reward.mean(), log_ratio.mean())
                    # jax.debug.print("ratio mean: {}, std: {}", ratio.mean(), ratio.std())


                    adv_base = jax.lax.stop_gradient(adv_base)  ### when forward process is learned things have to be adapted

                    valid_mask = 1.0 - minibatch.truncated
                    lagrangian_loss = jnp.array(0.0)
                    if cfg.use_kl_regularization:
                        actor_target_model = model.actor_module
                        kl_keys = jax.random.split(step_key, cfg.kl_action_rep)
                        if cfg.reverse_kl:
                            def compute_single(k):
                                return model.actor_module.rkl_div_one_step(
                                    k, minibatch.obs, actor_target_model, stop_grad=False
                                )
                        else:
                            def compute_single(k):
                                return model.actor_module.fkl_div_one_step(
                                    k, minibatch.obs, actor_target_model, stop_grad=False
                                )
                        kl_log_ratios = jax.vmap(compute_single)(kl_keys)
                        kl_log_ratios = kl_log_ratios.mean(axis=0)
                        kl = self.diffusion_steps * kl_log_ratios.sum(-1)
                        adv_term = ratio * adv_base
                        kl_term = jax.lax.stop_gradient(lagrangian) * kl * (
                            cfg.reduce_kl
                        )
                        if cfg.actor_kl_clip_mode == "full":
                            combined = adv_term + kl_term
                        elif cfg.actor_kl_clip_mode == "clipped":
                            combined = jnp.where(
                                kl < cfg.kl_bound, adv_term, kl_term
                            )
                        elif cfg.actor_kl_clip_mode == "value":
                            combined = adv_term
                        else:
                            raise ValueError(
                                f"Unknown actor_kl_clip_mode: {cfg.actor_kl_clip_mode}"
                            )
                        actor_loss = -jnp.mean(valid_mask * combined)
                        lagrangian_loss = (
                            -lagrangian
                            * jax.lax.stop_gradient(kl - cfg.kl_bound)
                        ).mean()
                    else:
                        actor_loss1 = ratio * adv_base
                        actor_loss2 = (
                            jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                            * adv_base
                        )
                        actor_loss = -jnp.mean(
                            valid_mask * jnp.minimum(actor_loss1, actor_loss2)
                        )

                    entropy_loss = jnp.mean(gen_log_prob)

                    loss = (
                        actor_loss
                        + cfg.value_coef * value_loss
                    )
                    if cfg.update_entropy_lagrangian:
                        entropy = -self.diffusion_steps * jnp.mean(log_ratio, axis=0)
                        target_entropy = self.action_size_target + entropy
                        target_entropy_loss = (
                            model.actor_module.temperature()
                            * jax.lax.stop_gradient(target_entropy)
                        ).mean()
                        loss += target_entropy_loss
                    else:
                        entropy = 0.0
                        target_entropy = 0.0
                        target_entropy_loss = 0.0
                    if cfg.use_kl_regularization:
                        loss += lagrangian_loss
                    else:
                        kl = jnp.array(0.0)
                    #jax.debug.print("temperature  loss: {}", model.actor_module.temperature())
                    # print all losses here
                    #jax.debug.print("actor_loss: {}, value_loss: {}, entropy_loss: {}, entropy: {}, target_entropy: {}, target_entropy_loss: {}, kl: {}, lagrangian: {}, lagrangian_loss: {}, total_loss: {}", actor_loss, value_loss, entropy_loss, entropy, target_entropy, target_entropy_loss, kl, lagrangian, lagrangian_loss, loss)

                    return loss, dict(
                        actor_loss=actor_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        entropy=entropy,
                        target_entropy=target_entropy,
                        target_entropy_loss=target_entropy_loss,
                        temp=model.actor_module.temperature(),
                        kl=kl,
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        loss=loss,
                        mean_value=value.mean(),
                        mean_log_prob=gen_log_prob.mean(),
                        mean_advantages=adv_base.mean(),
                        mean_action=minibatch.action.mean(),
                        reward_mean=minibatch.reward.mean()*self.diffusion_steps,
                    )

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                output, grads = grad_fn(train_state.params)

                flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
                global_grad_norm = jnp.linalg.norm(flat_grads)

                metrics = output[1]
                metrics["advantages"] = advantages.mean()
                metrics["global_grad_norm"] = global_grad_norm
                train_state = train_state.apply_gradients(grads)
                return (idx + 1, train_state), metrics

            key, shuffle_key = jax.random.split(key)

            mini_batch_size = (self.num_collection_steps * cfg.num_envs) // self.num_minibatches
            indices = jax.random.permutation(shuffle_key, self.num_collection_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (self.num_minibatches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )
            minibatch_keys = jax.random.split(shuffle_key, self.num_minibatches)

            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, (minibatch_idxs, minibatch_keys)
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
        advantages_hist = None
        if advantages is not None:
            adv_np = np.asarray(jax.device_get(advantages))
            finite_mask = np.isfinite(adv_np)
            if finite_mask.any():
                advantages_hist = wandb.Histogram(adv_np[finite_mask])
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        if advantages_hist is not None:
            log_data["train/advantages"] = advantages_hist
        wandb.log(log_data, step=state.time_steps[0])

    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(cfg.env.name)
        raise ValueError("Brax not supported for DiffPPO.")
    elif cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(cfg.env.name, episode_length=cfg.env.max_episode_steps)
        diff_cfg = cfg.hyperparameters.diffusion
        env_action_clip_value = OmegaConf.select(
            cfg, "hyperparameters.env_action_clip_value", default=1.0
        )
        env = MjxDiffEnvWrapper(
            env,
            num_diff_steps=diff_cfg.diff_steps,
            diffusion_config=diff_cfg,
            low=-env_action_clip_value,
            high=env_action_clip_value,
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
            project=cfg.wandb.project,
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

    env = MjxGymnaxWrapper(cfg.env.name, episode_length=cfg.env.max_episode_steps)
    diff_cfg = cfg.hyperparameters.diffusion
    env_action_clip_value = OmegaConf.select(
        cfg, "hyperparameters.env_action_clip_value", default=1.0
    )
    env = MjxDiffEnvWrapper(
        env,
        num_diff_steps=diff_cfg.diff_steps,
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )

    def train_agent():
        wandb.init(project=cfg.wandb.project)
        run_cfg = OmegaConf.to_container(cfg)
        for k, v in dict(wandb.config).items():
            run_cfg["experiment"]["hyperparameters"][k] = v
        ppo_cfg = PPOConfig(**run_cfg["experiment"]["hyperparameters"])
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


@hydra.main(version_base=None, config_path="../../config", config_name="diff_ppo")
def main(cfg: DictConfig):
    if cfg.tune:
        tune(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
