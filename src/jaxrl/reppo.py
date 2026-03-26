import logging
import os
import pickle
import re
import time
import typing
from typing import Callable

import hydra
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import optuna
import plotly.graph_objs as go
import mujoco
from flax import nnx, struct
from flax.struct import PyTreeNode
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax import tree_util
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.env_utils.torso_com import (
    build_torso_com_traj_figure,
    get_torso_com_all,
    resolve_mj_model,
    save_torso_com_trajectory,
)
from src.jaxrl import utils
from src.networks.jax_models import (
    CategoricalCriticNetwork,
    CriticNetwork,
    SACActorNetworks,
)

if os.environ.get("JAX_DEBUG_NANS", "").lower() not in ("", "0", "false"):
    jax.config.update("jax_debug_nans", True)

logging.basicConfig(level=logging.INFO)


def _sanitize_filename_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned or "run"


def _saved_models_dir() -> str:
    # Hydra changes CWD into `outputs/...`; write checkpoints relative to repo root.
    try:
        repo_root = hydra.utils.get_original_cwd()
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    path = os.path.join(repo_root, "saved_models")
    os.makedirs(path, exist_ok=True)
    return path


def _to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


def _take_last_metrics(metrics: dict) -> dict:
    def _last(x):
        x = jnp.asarray(x)
        return x[-1] if x.ndim > 0 else x

    return jax.tree.map(_last, metrics)


def _assert_all_finite(name: str, array: jax.Array):
    """Host-side check that raises immediately when NaNs/Infs appear in JIT."""

    def _callback(x):
        if not np.isfinite(x).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    jax.debug.callback(_callback, array)


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    info: dict[str, jax.Array]


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    normalize_reward: bool = False
    action_clip_value: float = 1.0
    env_action_clip_value: float = 1.0
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    log_torso_com: bool = False
    log_torso_com_num_envs: int = 30
    log_torso_com_stride: int = 1
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    reverse_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    use_lax_scan: bool = True
    train_mode: str = "reparam"
    disable_wpo_fisher_preconditioning: bool = False
    disable_temperature: bool = False
    temperature_lr: float = 3e-4
    temperature_lr_mult: float = 1.0
    lagrangian_lr: float = 3e-4
    lagrangian_lr_mult: float = 1.0


class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array


def make_policy(
    train_state: SACTrainState, train_mode: str
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]:
    def policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        if train_mode == "WPO":
            action: jax.Array = actor_model.actor(obs).sample(seed=key)
        else:
            action: jax.Array = actor_model.det_action(obs)
        return action, {}

    return policy


def make_eval_fn(
    env: Environment,
    max_episode_steps: int,
    reward_scale: float = 1.0,
    torso_id: int | None = None,
    log_torso_com: bool = False,
    log_torso_com_num_envs: int = 30,
    log_torso_com_stride: int = 1,
) -> Callable[[jax.random.PRNGKey, Policy, PyTreeNode | None], dict[str, float]]:
    num_torso_envs = min(int(log_torso_com_num_envs), env.num_envs)
    torso_stride = max(1, int(log_torso_com_stride))

    def evaluation_fn(
        key: jax.random.PRNGKey, policy: Policy, norm_state: PyTreeNode | None
    ):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        # randomize initial steps
        key, env_key = jax.random.split(key)
        com_traj = None
        torso_env_indices = None
        if log_torso_com and (torso_id is not None and num_torso_envs > 0):
            key, sample_key = jax.random.split(key)
            torso_env_indices = jax.random.choice(
                sample_key, env.num_envs, (num_torso_envs,), replace=False
            )

            def step_env_with_com(carry, _):
                key, env_state, obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                action, _ = policy(act_key, obs)
                step_key = jax.random.split(env_key, env.num_envs)
                obs, _, env_state, reward, done, info = env.step(
                    step_key, env_state, action
                )
                com = get_torso_com_all(env_state, torso_id)
                sampled_com = com[torso_env_indices]
                return (key, env_state, obs), (info, sampled_com)

            _, (infos, com_traj) = jax.lax.scan(
                f=step_env_with_com,
                init=(key, env_state, obs),
                xs=None,
                length=max_episode_steps,
            )
            if torso_stride > 1:
                com_traj = com_traj[::torso_stride]
        else:
            _, infos = jax.lax.scan(
                f=step_env,
                init=(key, env_state, obs),
                xs=None,
                length=max_episode_steps,
            )

        metrics = {
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            )
            * reward_scale,
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
        if com_traj is not None:
            metrics["torso_com_traj"] = com_traj
            metrics["torso_com_env_indices"] = torso_env_indices
        return metrics

    return evaluation_fn


def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]

        actor_networks = SACActorNetworks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            ent_start=cfg.ent_start,
            kl_start=cfg.kl_start,
            use_norm=cfg.use_actor_norm,
            layers=cfg.num_actor_layers,
            use_skip=cfg.use_actor_skip,
            train_mode=cfg.train_mode,
            disable_wpo_fisher_preconditioning=cfg.disable_wpo_fisher_preconditioning,
            disable_temperature=cfg.disable_temperature,
            rngs=nnx.Rngs(model_key),
        )
        actor_target_networks = SACActorNetworks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.actor_hidden_dim,
            ent_start=cfg.ent_start,
            kl_start=cfg.kl_start,
            use_norm=cfg.use_actor_norm,
            layers=cfg.num_actor_layers,
            use_skip=cfg.use_actor_skip,
            train_mode=cfg.train_mode,
            disable_wpo_fisher_preconditioning=cfg.disable_wpo_fisher_preconditioning,
            disable_temperature=cfg.disable_temperature,
            rngs=nnx.Rngs(model_key),
        )

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        def _scale_lr(lr_val, mult: float):
            if callable(lr_val):
                return lambda step: lr_val(step) * mult
            return lr_val * mult

        def _resolve_special_lr(lr_val, special_lr, mult: float):
            if special_lr is not None:
                return special_lr
            return _scale_lr(lr_val, mult)

        def _label_actor_params(params):
            flat = flatten_dict(params)
            labels = {}
            for k in flat.keys():
                leaf_name = k[-1]
                if "temperature" in leaf_name:
                    labels[k] = "temperature"
                elif "lagrangian" in leaf_name:
                    labels[k] = "lagrangian"
                else:
                    labels[k] = "default"
            return unflatten_dict(labels)

        temperature_lr = _resolve_special_lr(
            lr, cfg.temperature_lr, cfg.temperature_lr_mult
        )
        lagrangian_lr = _resolve_special_lr(
            lr, cfg.lagrangian_lr, cfg.lagrangian_lr_mult
        )
        actor_param_tree = nnx.to_pure_dict(nnx.state(actor_networks))
        actor_labels = _label_actor_params(actor_param_tree)
        actor_optimizer = optax.multi_transform(
            {
                "default": optax.adam(lr),
                "temperature": optax.adam(temperature_lr),
                "lagrangian": optax.adam(lagrangian_lr),
            },
            actor_labels,
        )
        critic_optimizer = optax.adam(lr)
        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                actor_optimizer,
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                critic_optimizer,
            )

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=actor_param_tree,
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.to_pure_dict(nnx.state(actor_target_networks)),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.to_pure_dict(nnx.state(critic_networks)),
            tx=critic_optimizer,
        )

        actor_param_count = utils.count_params(actor_trainstate.params)
        critic_param_count = utils.count_params(critic_trainstate.params)
        
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")
        print(f"Total parameters: {actor_param_count + critic_param_count:,}")

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # randomize initial time step to prevent all envs stepping in tandem
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        return SACTrainState(
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
):
    env = LogWrapper(env, cfg.num_envs)
    env = ClipAction(env, low=-cfg.env_action_clip_value, high=cfg.env_action_clip_value)
    # env = VecEnv(env, cfg.num_envs)
    if cfg.normalize_env:
        env = NormalizeVec(env, normalize_reward=cfg.normalize_reward)
    torso_id = None
    if getattr(cfg, "log_torso_com", False):
        mj_model = resolve_mj_model(env)
        if mj_model is None:
            logging.warning("MJX model not found; skipping torso COM logging.")
        else:
            torso_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            if torso_id < 0:
                torso_id = None
                logging.warning("Torso body not found; skipping torso COM logging.")
    eval_fn = make_eval_fn(
        env,
        cfg.max_episode_steps,
        reward_scale=reward_scale,
        torso_id=torso_id,
        log_torso_com=getattr(cfg, "log_torso_com", False),
        log_torso_com_num_envs=getattr(cfg, "log_torso_com_num_envs", 30),
        log_torso_com_stride=getattr(cfg, "log_torso_com_stride", 1),
    )
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action
            og_pi = actor_model.actor(obs)
            pi = actor_model.actor(obs, scale=offset)
            action = pi.sample(seed=act_key)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            # compute importance weights
            action = jnp.clip(action, -cfg.action_clip_value, cfg.action_clip_value)
            raw_importance_weight = jnp.nan_to_num(
                og_pi.log_prob(action).sum(-1) - pi.log_prob(action).sum(-1),
                nan=jnp.log(cfg.lmbda_min),
            )
            importance_weight = jnp.clip(
                raw_importance_weight, min=jnp.log(cfg.lmbda_min), max=jnp.log(1.0)
            )

            # compute next state embedding and value
            next_action, next_log_prob = actor_model.actor(next_obs).sample_and_log_prob(
                seed=act_key ### why use the same action key here?
            ) ### shouldnt this action be used for the next env interaction?
            next_emb, _, _, value = critic_model.forward(
                next_critic_obs, next_action
            )
            soft_reward = (
                reward
                - cfg.gamma * next_log_prob.sum(-1).squeeze() * actor_model.temperature()
            )
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                info=info,
                importance_weight=importance_weight,
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

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, batch: Transition
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        # compute n-step lambda estimates

        def compute_nstep_lambda(carry, transition):
            lambda_return, truncated, importance_weight = carry
            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.soft_reward
            value = transition.value
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.importance_weight,
            ), lambda_return

        _, target_values = jax.lax.scan(
            compute_nstep_lambda,
            (
                batch.value[-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.importance_weight[0]),
            ),
            batch,
            reverse=True,
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    if cfg.hl_gauss:
                        target_cat = jax.vmap(
                            utils.hl_gauss, in_axes=(0, None, None, None)
                        )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                        critic_update_loss = optax.softmax_cross_entropy(
                            critic_pred, target_cat
                        )
                    else:
                        critic_update_loss = optax.squared_error(
                            critic_pred.reshape(-1,1),
                            target_values.reshape(-1,1),
                        )

                    # Aux loss
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred,  minibatch.next_emb)
                    aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * jnp.concatenate(
                            [aux_loss, aux_rew_loss], axis=-1
                        ), axis=-1)

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )
                    _assert_all_finite("critic_loss", loss)
                    _assert_all_finite(
                        "critic_update_loss", jnp.mean(critic_update_loss)
                    )
                    _assert_all_finite("critic_aux_loss", jnp.mean(aux_loss))

                    # log critic parameters norm
                    critic_pnorm = utils.tree_norm(params)

                    return loss, dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        rew_aux_loss= aux_rew_loss,
                        q=value.mean(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        critic_pnorm=critic_pnorm,
                    )

                def actor_loss(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    pi = actor_model.actor(minibatch.obs)
                    pred_action, log_prob = pi.sample_and_log_prob(seed=key)
                    value = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )
                    log_prob = log_prob.sum(-1)
                    entropy = -log_prob

                    # policy KL constraint
                    if cfg.reverse_kl:
                        pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                            sample_shape=(16,), seed=key
                        )
                        pi_action = jnp.clip(
                            pi_action,
                            -cfg.action_clip_value,
                            cfg.action_clip_value,
                        )

                        old_pi = actor_target_model.actor(minibatch.obs)

                        old_pi_act_log_prob = old_pi.log_prob(pi_action).sum(-1).mean(
                            0
                        )
                        pi_act_log_prob = pi_act_log_prob.sum(-1).mean(0)
                        kl = pi_act_log_prob - old_pi_act_log_prob
                    else:
                        old_pi_action, old_pi_act_log_prob = actor_target_model.actor(
                            minibatch.obs
                        ).sample_and_log_prob(sample_shape=(16,), seed=key)
                        old_pi_action = jnp.clip(
                            old_pi_action,
                            -cfg.action_clip_value,
                            cfg.action_clip_value,
                        )

                        old_pi_act_log_prob = old_pi_act_log_prob.sum(-1).mean(0)
                        pi_act_log_prob = pi.log_prob(old_pi_action).sum(-1).mean(0)

                        kl = old_pi_act_log_prob - pi_act_log_prob

                    temperature = actor_model.temperature()
                    lagrangian = actor_model.lagrangian()

                    if cfg.actor_kl_clip_mode == "full":
                        actor_loss = (
                            log_prob * jax.lax.stop_gradient(temperature)
                            - value
                            + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
                        )
                    elif cfg.actor_kl_clip_mode == "clipped":
                        actor_loss = jnp.where(
                            kl < cfg.kl_bound,
                            log_prob * jax.lax.stop_gradient(temperature)
                            - value,
                            kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                        )
                    elif cfg.actor_kl_clip_mode == "value":
                        actor_loss = (
                            log_prob * jax.lax.stop_gradient(temperature)
                            - value
                        )
                    else:
                        raise ValueError(
                            f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}"
                        )

                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                        temperature * jax.lax.stop_gradient(target_entropy)
                    )

                    # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl - cfg.kl_bound
                    )

                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += jnp.mean(target_entropy_loss)
                    if cfg.update_kl_lagrangian:
                        loss += jnp.mean(lagrangian_loss)
                    _assert_all_finite("actor_loss", loss)
                    _assert_all_finite("actor_entropy", jnp.mean(entropy))
                    _assert_all_finite("actor_kl", jnp.mean(kl))

                    # log actor parameters norm
                    actor_pnorm = utils.tree_norm(params)

                    return loss, dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        kl=kl.mean(),
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_values=target_values.mean(),
                        actor_pnorm=actor_pnorm,
                    )

                def actor_loss_WPO(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    pi = actor_model.actor(minibatch.obs)
                    pred_action, log_prob = pi.sample_and_log_prob(seed=key)
                    pred_action = jnp.clip(
                        pred_action, -cfg.action_clip_value, cfg.action_clip_value
                    )
                    value = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )

                    def single_q(obs, act):
                        return critic_target_model.critic(obs[None], act[None]).squeeze()

                    def single_log_prob(obs, act):
                        return actor_model.actor(obs[None]).log_prob(act[None]).sum()
                    
                    def single_target_log_prob(obs, act):
                        return actor_target_model.actor(obs[None]).log_prob(act[None]).sum()

                    stop_pred_action = jax.lax.stop_gradient(pred_action)
                    q_action_grad = jax.vmap(
                        jax.grad(single_q, argnums=1)
                    )(minibatch.critic_obs, stop_pred_action)
                    stop_q_action_grad = jax.lax.stop_gradient(q_action_grad)
                    log_prob_action_grad = jax.vmap(
                        jax.grad(single_log_prob, argnums=1)
                    )(minibatch.obs, stop_pred_action)


                    log_prob = log_prob.sum(-1)
                    entropy = -log_prob

                    # policy KL constraint
                    if cfg.reverse_kl:
                        pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                            sample_shape=(16,), seed=key
                        )
                        pi_action = jnp.clip(
                            pi_action,
                            -cfg.action_clip_value,
                            cfg.action_clip_value,
                        )

                        old_pi = actor_target_model.actor(minibatch.obs)

                        old_pi_act_log_prob = old_pi.log_prob(pi_action).sum(-1).mean(0)
                        pi_act_log_prob = pi_act_log_prob.sum(-1).mean(0)
                        kl = pi_act_log_prob - old_pi_act_log_prob
                    else:
                        old_pi_action, old_pi_act_log_prob = actor_target_model.actor(
                            minibatch.obs
                        ).sample_and_log_prob(sample_shape=(16,), seed=key)

                        old_pi_action = jnp.clip(old_pi_action, -cfg.action_clip_value, cfg.action_clip_value)
                        old_pi_action = jax.lax.stop_gradient(old_pi_action)

                        # target_log_prob_action_grad = jax.vmap(
                        #     jax.vmap(
                        #         jax.grad(single_target_log_prob, argnums=1),
                        #         in_axes=(0, 0),
                        #     ),
                        #     in_axes=(None, 0),
                        # )(minibatch.obs, old_pi_action)

                        # kl_log_prob_action_grad = jax.vmap(jax.vmap(
                        #         jax.grad(single_log_prob, argnums=1),
                        #         in_axes=(0, 0)),in_axes=(None, 0))(minibatch.obs, old_pi_action)
                        #print the shapes of log prob gradiens
                        # jax.debug.print("target_log_prob_action_grad shape={shape}", shape=target_log_prob_action_grad.shape)
                        # jax.debug.print("kl_log_prob_action_grad shape={shape}", shape=kl_log_prob_action_grad.shape)
                        #kl = jnp.mean(jnp.mean((target_log_prob_action_grad - kl_log_prob_action_grad)**2, axis = -1), axis = 0)

                        old_pi_act_log_prob = old_pi_act_log_prob.sum(-1).mean(0)
                        pi_act_log_prob = pi.log_prob(old_pi_action).sum(-1).mean(0)

                        Kl_value = old_pi_act_log_prob - pi_act_log_prob
                        kl = Kl_value

                    temperature = actor_model.temperature()
                    lagrangian = actor_model.lagrangian()

                    actor_Q_loss = jnp.sum(
                        (jax.lax.stop_gradient(log_prob_action_grad * temperature - stop_q_action_grad))
                        * log_prob_action_grad,
                        axis=-1,
                    )
                    # jax.debug.print(
                    #     "actor_Q_loss mean={mean} max={mx} min={mn} nan={nan}",
                    #     mean=jnp.nanmean(actor_Q_loss),
                    #     mx=jnp.nanmax(actor_Q_loss),
                    #     mn=jnp.nanmin(actor_Q_loss),
                    #     nan=jnp.any(jnp.isnan(actor_Q_loss)),
                    # )

                    if cfg.actor_kl_clip_mode == "full":
                        actor_loss = (
                            actor_Q_loss
                            + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
                        )
                    elif cfg.actor_kl_clip_mode == "clipped":
                        actor_loss = jnp.where(
                            Kl_value < cfg.kl_bound,
                            actor_Q_loss,
                            kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                        )
                    elif cfg.actor_kl_clip_mode == "value":
                        actor_loss = actor_Q_loss - value
                    else:
                        raise ValueError(
                            f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}"
                        )

                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                        temperature * jax.lax.stop_gradient(target_entropy)
                    )

                    # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        Kl_value - cfg.kl_bound
                    )

                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += jnp.mean(target_entropy_loss)
                    if cfg.update_kl_lagrangian:
                        loss += jnp.mean(lagrangian_loss)

                    # _assert_all_finite("q_action_grad mean", q_action_grad)
                    # _assert_all_finite("log_prob_action_grad mean", log_prob_action_grad)
                    # _assert_all_finite("actor_WPO_loss", loss)
                    # _assert_all_finite("actor_WPO_entropy", jnp.mean(entropy))
                    # _assert_all_finite("actor_WPO_kl", jnp.mean(kl))
                    # _assert_all_finite("actor_Q_loss", jnp.mean(actor_Q_loss))

                    # log actor parameters norm
                    actor_pnorm = utils.tree_norm(params)

                    return loss, dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        w2_kl=kl.mean(),
                        kl = Kl_value.mean(),
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_values=target_values.mean(),
                        actor_pnorm=actor_pnorm,
                        q_action_grad=q_action_grad,
                        policy_action_grad=log_prob_action_grad,
                    )

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(critic_grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic parameters norm
                critic_gnorm = utils.tree_norm(critic_grads)
                critic_metrics["critic_gnorm"] = critic_gnorm

                actor_loss_fn = (
                    actor_loss_WPO if cfg.train_mode == "WPO" else actor_loss
                )
                actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state.actor.params)
                actor_train_state = train_state.actor.apply_gradients(actor_grads)
                train_state = train_state.replace(
                    actor=actor_train_state,
                )
                actor_metrics = output[1]
                # log actor gradient norm
                actor_gnorm = utils.tree_norm(actor_grads)
                actor_metrics["actor_gnorm"] = actor_gnorm

                return (idx + 1, train_state), {
                    **critic_metrics,
                    **actor_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics = jax.lax.scan(
            f=update,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        target_values_mean = target_values.mean()
        target_values_min = target_values.min()
        target_values_max = target_values.max()
        update_metrics = {
            **update_metrics,
            "target_values_mean": target_values_mean,
            "target_values_min": target_values_min,
            "target_values_max": target_values_max,
        }

        return train_state, update_metrics

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, dict[str, jax.Array]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics, **update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            train_state, train_metrics = jax.lax.scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            policy = make_policy(train_state, getattr(cfg, "train_mode", "reparam"))
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            eval_metrics = eval_fn(eval_key, policy, norm_state)
            train_returns = {
                "train/episode_return": train_state.last_env_state.info[
                    "returned_episode_returns"
                ].mean(),
                "train/episode_length": train_state.last_env_state.info[
                    "returned_episode_lengths"
                ].mean(),
            }
            metrics = {
                "time_step": train_state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return train_state, metrics

        def loop_body(
            train_state: SACTrainState, key: PRNGKey
        ) -> tuple[SACTrainState, dict]:
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )
            jax.debug.callback(log_callback, train_state, metrics)
            return train_state, metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    return train_fn


# type object
def _get_optuna_type(trial: optuna.Trial, name, values: list):
    if all(isinstance(v, int) for v in values):
        return trial.suggest_int(name, low=min(values), high=max(values))
    elif all(isinstance(v, float) for v in values):
        return trial.suggest_float(name, low=min(values), high=max(values))
    elif all(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)
    elif all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, [True, False])
    else:
        raise ValueError("Values must be of the same type (int, float, or str).")


def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    """
    Run a single trial of the SAC training process with hyperparameter tuning.
    Args:
        cfg (DictConfig): Configuration for the SAC training.
        trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
    Returns:
        float: The mean episode return from the trial.
    """
    sweep_metrics = []

    if trial is not None:
        # Set hyperparameters from the trial
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
                # TODO: Why the fuck is this happening
                if isinstance(sampled_value, np.float64):
                    sampled_value = float(sampled_value)
                cfg.hyperparameters[name] = sampled_value
            else:
                raise ValueError(f"Hyperparameter {name} not found in config.")

    try:
        with open("completed_trials.txt", "r") as f:
            completed_trials = int(f.read())
    except FileNotFoundError:
        completed_trials = 0

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
        eval_length = metrics["eval/episode_length"].mean()
        torso_com_traj = metrics.pop("eval/torso_com_traj", None)
        torso_com_env_indices = metrics.pop("eval/torso_com_env_indices", None)
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f} sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            # performance metric: steps per second
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        fig = build_torso_com_traj_figure(
            torso_com_traj, torso_com_env_indices, title="Torso COM trajectory (XY)"
        )
        if fig is not None:
            log_data["figures/eval_torso_com_traj_xy"] = wandb.Image(fig)
            plt.close(fig)
        if torso_com_traj is not None:
            step_id = int(np.asarray(state.time_steps[0]))
            save_torso_com_trajectory(
                f"reppo_traj_step_{step_id}.pkl",
                torso_com_traj,
                torso_com_env_indices,
            )
        wandb.log(log_data, step=state.time_steps[0])

    # Set up the experiment
    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
    elif cfg.env.type == "mjx":
        env_config = OmegaConf.select(cfg, "env.config")
        if env_config is not None:
            env_config = OmegaConf.to_container(env_config, resolve=True)
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            config=env_config,
            asymmetric_observation=cfg.env.get("asymmetric_obs", False),
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    # build algo config with overrides

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        run_config = OmegaConf.to_container(cfg)
        run_config["method_name"] = "reppo"
        wandb.init(
            mode=cfg.wandb.mode,
            project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}",
            entity=cfg.wandb.entity,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                "hp_tune" if trial is not None else "val",
                *cfg.tags,
            ],
            config=run_config,
            name=(
                f"{cfg.name}-{cfg.env.name.lower()}"
                f"-{getattr(cfg.hyperparameters, 'train_mode', 'reparam')}"
            ),
            save_code=True,
        )

        logging.info(OmegaConf.to_yaml(cfg))

        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        state, metrics = jax.jit(train_fn, static_argnums=(1,))(
            key, ReppoConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Export final weights into repo_root/saved_models with a descriptive filename.
        try:
            final_metrics = _take_last_metrics(metrics)
            method_name = "reppo"
            env_name = str(cfg.env.name)
            train_mode = str(getattr(cfg.hyperparameters, "train_mode", "reparam"))
            timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            filename = "__".join(
                [
                    _sanitize_filename_component(method_name),
                    _sanitize_filename_component(env_name),
                    f"trainmode{_sanitize_filename_component(train_mode)}",
                    f"seed{int(cfg.seed)}",
                    f"trial{i}",
                    f"ts{timestamp}",
                ]
            ) + ".pkl"
            save_path = os.path.join(_saved_models_dir(), filename)
            checkpoint = {
                "method_name": method_name,
                "env_name": env_name,
                "train_mode": train_mode,
                "seed": int(cfg.seed),
                "trial": int(i),
                "saved_at": time.time(),
                "num_seeds": int(np.asarray(state.time_steps).shape[0])
                if np.asarray(state.time_steps).ndim > 0
                else 1,
                "actor_params": _to_numpy_tree(state.actor.params),
                "actor_target_params": _to_numpy_tree(state.actor_target.params),
                "critic_params": _to_numpy_tree(state.critic.params),
                "actor_step": _to_numpy_tree(state.actor.step),
                "critic_step": _to_numpy_tree(state.critic.step),
                "time_steps": _to_numpy_tree(state.time_steps),
                "iteration": _to_numpy_tree(state.iteration),
                "last_env_state": _to_numpy_tree(state.last_env_state)
                if bool(getattr(cfg.hyperparameters, "normalize_env", False))
                else None,
                "final_eval_metrics": _to_numpy_tree(
                    utils.filter_prefix("eval", final_metrics)
                ),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
            with open(save_path, "wb") as f:
                pickle.dump(checkpoint, f)
            logging.info("Saved final model checkpoint to %s", save_path)
        except Exception as e:
            logging.exception("Failed to export final model checkpoint: %s", e)

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="reppo")
def main(cfg: DictConfig):
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
