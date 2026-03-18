import logging
import os
import pickle
import re
import time
import typing
from functools import partial
from typing import Any, Callable

import hydra
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import optuna
import mujoco
from flax import nnx, struct
from flax.struct import PyTreeNode
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf
import copy

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    TanhClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    MjxDiffEnvWrapper,
    DiffNormalizeVec,
)
from src.env_utils.torso_com import (
    build_torso_com_traj_figure,
    get_torso_com_all,
    resolve_mj_model,
    save_torso_com_trajectory,
)
from src.jaxrl import utils
from src.jaxrl.reppo_helpers.learning_DiffReppo import (
    _resolve_temperature,
    actor_loss_fn,
    actor_WPO_loss_fn,
    compute_nstep_lambda_step,
    critic_loss_fn,
    maybe_add_q_grad,
)
from src.networks.diffusion.models import ControlNetwork
from src.networks.jax_models_DMERL import (
    CategoricalCriticNetwork,
    CriticNetwork,
    DiffusionModel,
    DMERLActor,
    sde_integrator,
    ode_integrator,
    logratio_DIME as logratio,
)

logging.basicConfig(level=logging.INFO)


def _sectioned_wandb_key(key: str) -> str:
    if key.startswith("/"):
        key = key.lstrip("/")
    if key.startswith("train/"):
        suffix = key.split("/", 1)[1]
        if suffix.startswith(("temp", "entropy")):
            return f"temperature/{suffix}"
        if suffix.startswith(("lagrangian", "kl")):
            return f"lagrangian/{suffix}"
        if suffix.startswith("target_value_"):
            return f"target_value/{suffix}"
        return f"train/{suffix}"
    if key.startswith("eval/"):
        suffix = key.split("/", 1)[1]
        return f"eval/{suffix}"
    if key.startswith("norm_init/"):
        suffix = key.split("/", 1)[1]
        return f"norm_init/{suffix}"
    if key.startswith("norm/"):
        suffix = key.split("/", 1)[1]
        return f"norm/{suffix}"
    if key.startswith("figures/"):
        suffix = key.split("/", 1)[1]
        return f"figures/{suffix}"
    if key.startswith("step_metrics/"):
        suffix = key.split("/", 1)[1]
        return f"step_metrics/{suffix}"
    return f"system/{key}"


def _sectioned_wandb_log(log_data: dict[str, Any]) -> dict[str, Any]:
    return {_sectioned_wandb_key(key): value for key, value in log_data.items()}


def _sanitize_dir_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "run"


def _to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        ...


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    next_state_emb: jax.Array
    next_emb_mask: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    info: dict[str, jax.Array]


def _timestep_coeff_norm(params):
    # Support both nnx modules and already-pure param dicts.
    pure_params = nnx.to_pure_dict(params) if not isinstance(params, dict) else params
    flat = flatten_dict(pure_params)
    for k, v in flat.items():
        if k[-1] == "timestep_coeff":
            coeff = v[0] if isinstance(v, tuple) else v
            return jnp.linalg.norm(coeff)
    return jnp.array(0.0)



class ReppoConfig(struct.PyTreeNode):
    lr: float
    lr_decay_factor: float
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
    kl_action_rep: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    weight_decay: float = 0.0
    num_collection_step_factor: float = 1.0
    temperature_lr: float | None = None
    lagrangian_lr: float | None = None
    temperature_lr_mult: float = 1.0
    lagrangian_lr_mult: float = 1.0
    temp_lagrangian_optim: str = "sgd"
    temp_lagrangian_adam_gamma1: float = 0.9
    temp_lagrangian_adam_gamma2: float = 0.999
    use_temp_lagrangian_ema_optim: bool = False
    use_temp_lagrangian_post_adam_ema: bool = False
    temp_lagrangian_ema_decay: float = 0.99
    use_temperature_decay: bool = False
    temperature_decay_start: float | None = None
    temperature_decay_end: float | None = None
    temperature_decay_steps: int | None = None
    action_clip_value: float = 1.0
    tanh_transform: bool = False
    use_temp_lagrangian_mlp: bool = False
    temp_lagrangian_hidden: int = 32
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
    kl_bound_fisher_precond: bool = False
    remove_fisher_precond: bool = False
    aux_loss_mult: float = 0.0
    aux_loss_alpha: float = 0.9
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_augmented_lagrangian_dual: bool = False
    augmented_lagrangian_entropy_coef: float = 1.0
    augmented_lagrangian_kl_coef: float = 1.0
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
    use_W2_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    train_mode: str = "reparam"
    use_lax_scan: bool = True
    use_friction_mlp: bool = False
    friction_mlp_hidden: int = 64
    friction_mlp_layers: int = 2
    friction_num_time_hid: int = 32
    friction_num_time_out: int = 16
    friction_mlp_use_obs: bool = True
    diffusion: Any = None
    ode_coefs: list | None = None
    project_unit_ball: bool = True
    project_only_if_exceeds: bool = True
    use_current_critic_for_actor_samples: bool = True
    normalize_reward: bool = False


class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array


def randomize_env_steps(
    key: jax.random.PRNGKey, env_state: EnvState, max_episode_steps: int
) -> tuple[jax.random.PRNGKey, EnvState]:
    """Fully unwrap env state, randomize step counter, and rewrap."""
    _env_state = env_state
    unwrap_idx = 0
    env_state_list = [_env_state]
    while hasattr(_env_state, "unwrapped"):
        print("unwrap[{}] type: {}", unwrap_idx, type(_env_state).__name__)
        _env_state = _env_state.unwrapped()
        env_state_list.append(_env_state)
        unwrap_idx += 1
    print("final unwrapped type: {}", type(_env_state).__name__)

    key, randomize_steps_key = jax.random.split(key)
    _env_state.info["steps"] = jax.random.randint(
        randomize_steps_key,
        _env_state.info["steps"].shape,
        0,
        max_episode_steps,
    ).astype(jnp.float32)

    rewrapped_state = _env_state
    for list_env_state in reversed(env_state_list[:-1]):
        print(
            "rewrapped {} to type: {}",
            type(list_env_state).__name__,
            type(rewrapped_state).__name__,
        )
        rewrapped_state = list_env_state.set_env_state(rewrapped_state)

    return key, rewrapped_state


class ReppoDMERLTrainer:
    """Readable class wrapper around the original DMERL training utilities."""

    def __init__(
        self,
        cfg: ReppoConfig,
        env: Environment,
        env_params: EnvParams | None = None,
        log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
        num_seeds: int = 1,
        reward_scale: float = 1.0,
    ) -> None:
        # print vmin and vmax
        print(f"Initial vmin: {cfg.vmin}, vmax: {cfg.vmax}")
        #raise NotImplementedError("DMERL reward scaling not implemented.")
        diff_steps = getattr(cfg.diffusion, "diff_steps", None)
        if diff_steps is not None and diff_steps > 0:
           ### adjust the gamma1 and gamma2 for temp lagrangian optimizers based in num_minibacthes
            temp_lagrangian_adam_gamma1 = cfg.temp_lagrangian_adam_gamma1 #** (128/(cfg.num_mini_batches*cfg.diffusion.diff_steps))
            temp_lagrangian_adam_gamma2 = cfg.temp_lagrangian_adam_gamma2 #** (128/(cfg.num_mini_batches*cfg.diffusion.diff_steps))
            # minimum value of gamma1 and gamma2 is 0.9 and 0.999 respectively
            temp_lagrangian_adam_gamma1 = max(temp_lagrangian_adam_gamma1, 0.9)
            temp_lagrangian_adam_gamma2 = max(temp_lagrangian_adam_gamma2, 0.999)

            print(128/(cfg.num_mini_batches*cfg.diffusion.diff_steps), cfg.num_mini_batches, cfg.diffusion.diff_steps)
            print(f"Adjusted temp_lagrangian_adam_gamma1: {temp_lagrangian_adam_gamma1}, temp_lagrangian_adam_gamma2: {temp_lagrangian_adam_gamma2}")


            temp_lr_multi = cfg.temperature_lr_mult
            lagrangian_lr_mult = cfg.lagrangian_lr_mult
            if cfg.temperature_lr is None:
                temp_lr_multi = temp_lr_multi
            if cfg.lagrangian_lr is None:
                lagrangian_lr_mult = lagrangian_lr_mult
            cfg = cfg.replace(
                temperature_lr_mult=temp_lr_multi,
                lagrangian_lr_mult=lagrangian_lr_mult,
                temp_lagrangian_adam_gamma1=temp_lagrangian_adam_gamma1,
                temp_lagrangian_adam_gamma2=temp_lagrangian_adam_gamma2,
            )


            # adjusted_total_time_steps = cfg.total_time_steps * diff_steps
            # cfg = cfg.replace(total_time_steps=adjusted_total_time_steps)

            pass
        self.cfg = cfg
        self.use_langevin_param = bool(cfg.diffusion.score_model.langevin_param)
        self.env_params = env_params
        self.log_callback = log_callback or (lambda *args: None)
        self.num_seeds = num_seeds
        self.reward_scale = reward_scale
        self.env = self._prepare_env(env)
        self.eval_env = copy.deepcopy(self.env)
        if cfg.log_torso_com:
            mj_model = resolve_mj_model(self.eval_env)
            if mj_model is None:
                self.torso_id = None
                logging.warning("MJX model not found; skipping torso COM logging.")
            else:
                torso_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
                )
                self.torso_id = torso_id if torso_id >= 0 else None
                if self.torso_id is None:
                    logging.warning("Torso body not found; skipping torso COM logging.")
        else:
            self.torso_id = None
        self.eval_env_steps = cfg.max_episode_steps*self.cfg.diffusion.diff_steps
        self.num_collection_steps = int(
            cfg.num_steps * self.cfg.diffusion.diff_steps * cfg.num_collection_step_factor
        )
        self.num_minibatches = cfg.num_mini_batches*self.cfg.diffusion.diff_steps
        action_shape = jnp.prod(jnp.array(self.env.action_space(env_params).shape))
        self.action_size_target = action_shape * cfg.ent_target_mult
        self.sde_eval_fn = self._make_sde_eval_fn()
        if(cfg.train_mode == "WPO"):
            self.eval_fn = self._make_sde_eval_fn(eval_policy=True)
            #self.eval_fn = self._make_ode_eval_fn()
        else:
            self.eval_fn = self._make_ode_eval_fn()

    def _init_step_sizes(self):
        self.mini_batch_size = (self.num_collection_steps * self.cfg.num_envs) // self.num_minibatches

        ### total time steps are in therms of env calls of the original env, same for num_steps
        self.num_train_steps = self.cfg.total_time_steps // int(self.cfg.num_steps * self.cfg.num_envs * self.cfg.num_collection_step_factor) 
        self.eval_interval = int(self.num_train_steps // self.cfg.num_eval)
        self.num_iterations = self.num_train_steps // self.eval_interval + int(
            self.num_train_steps % self.eval_interval != 0
        )
        self.minibatch_size_per_diff_step = self.mini_batch_size // self.cfg.diffusion.diff_steps

        log_data = {
            "step_metrics/mini_batch_size": self.mini_batch_size,
            "step_metrics/num_collection_steps": self.num_collection_steps,
            "step_metrics/num_train_steps": self.num_train_steps,
            "step_metrics/num_iterations": self.num_iterations,
            "step_metrics/mini_batch_size_per_diff_step": self.minibatch_size_per_diff_step,
            "step_metrics/eval_interval": self.eval_interval,
        }
        # print(log_data)
        # print(762/7)
        # print(self.cfg.num_eval)
        # raise NotImplementedError("DMERL reward scaling not implemented.")
        wandb.log(_sectioned_wandb_log(log_data), step=0)

    def _prepare_env(self, env: Environment) -> Environment:
        env = LogWrapper(env, self.cfg.num_envs)
        #env = TanhClipAction(env)
        if self.cfg.normalize_env:
            env = DiffNormalizeVec(
                env,
                normalize_reward=self.cfg.normalize_reward,
                num_diff_steps=self.cfg.diffusion.diff_steps,
            )
        return env

    def _make_sde_eval_fn(self, eval_policy = False) -> Callable[[jax.random.PRNGKey, SACTrainState, PyTreeNode | None], dict[str, float]]:
        env = self.eval_env
        max_episode_steps = self.eval_env_steps
        reward_scale = self.reward_scale
        torso_id = self.torso_id
        num_torso_envs = min(
            int(getattr(self.cfg, "log_torso_com_num_envs", 30)),
            env.num_envs,
        )
        torso_stride = max(1, int(getattr(self.cfg, "log_torso_com_stride", 1)))

        def sde_evaluation_fn(
            key: jax.random.PRNGKey,
            train_state: SACTrainState,
            norm_state: PyTreeNode | None,
        ):
            actor_model = nnx.merge(
                train_state.actor.graphdef, train_state.actor.params
            )
            critic_model = nnx.merge(
                train_state.critic.graphdef, train_state.critic.params
            )
            use_langevin = self.use_langevin_param

            def sde_policy(
                policy_key: PRNGKey, obs: jax.Array, critic_obs: jax.Array
            ) -> tuple[jax.Array, dict]:
                obs_for_actor = maybe_add_q_grad(
                    obs, critic_obs, actor_model, critic_model, use_langevin
                )
                action, *_ = actor_model.vmap_sample_next_step(
                    obs_for_actor, policy_key
                )
                return action, {}

            def step_env(carry, _):
                key, env_state, obs, critic_obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                action, _ = sde_policy(act_key, obs, critic_obs)
                step_key = jax.random.split(env_key, env.num_envs)
                obs, critic_obs, env_state, reward, done, info = env.step(
                    step_key, env_state, action
                )
                return (key, env_state, obs, critic_obs), info

            key, init_key = jax.random.split(key)
            init_key = jax.random.split(init_key, env.num_envs)
            obs, critic_obs, env_state = env.reset(init_key, norm_state)
            key, env_key = jax.random.split(key)
            com_traj = None
            torso_env_indices = None
            if eval_policy and (torso_id is not None and num_torso_envs > 0):
                key, sample_key = jax.random.split(key)
                torso_env_indices = jax.random.choice(
                    sample_key, env.num_envs, (num_torso_envs,), replace=False
                )

                def step_env_with_com(carry, _):
                    key, env_state, obs, critic_obs = carry
                    key, act_key, env_key = jax.random.split(key, 3)
                    action, _ = sde_policy(act_key, obs, critic_obs)
                    step_key = jax.random.split(env_key, env.num_envs)
                    obs, critic_obs, env_state, reward, done, info = env.step(
                        step_key, env_state, action
                    )
                    com = get_torso_com_all(env_state, torso_id)
                    sampled_com = com[torso_env_indices]
                    return (key, env_state, obs, critic_obs), (info, sampled_com)

                final_carry, (infos, com_traj) = jax.lax.scan(
                    f=step_env_with_com,
                    init=(key, env_state, obs, critic_obs),
                    xs=None,
                    length=max_episode_steps,
                )
                if torso_stride > 1:
                    com_traj = com_traj[::torso_stride]
            else:
                final_carry, infos = jax.lax.scan(
                    f=step_env,
                    init=(key, env_state, obs, critic_obs),
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

        return sde_evaluation_fn

    def _make_ode_eval_fn(self) -> Callable[[jax.random.PRNGKey, SACTrainState, float, PyTreeNode | None], dict[str, float]]:
        env = self.eval_env
        max_episode_steps = self.eval_env_steps
        reward_scale = self.reward_scale
        torso_id = self.torso_id
        num_torso_envs = min(
            int(getattr(self.cfg, "log_torso_com_num_envs", 30)),
            env.num_envs,
        )
        torso_stride = max(1, int(getattr(self.cfg, "log_torso_com_stride", 1)))

        def ode_evaluation_fn(
            key: jax.random.PRNGKey,
            train_state: SACTrainState,
            norm_state: PyTreeNode | None,
        ):
            actor_model = nnx.merge(
                train_state.actor.graphdef, train_state.actor.params
            )
            critic_model = nnx.merge(
                train_state.critic.graphdef, train_state.critic.params
            )
            use_langevin = self.use_langevin_param

            def ode_policy(
                policy_key: PRNGKey, obs: jax.Array, critic_obs: jax.Array
            ) -> tuple[jax.Array, dict]:
                obs_for_actor = maybe_add_q_grad(
                    obs, critic_obs, actor_model, critic_model, use_langevin
                )
                action, *_ = actor_model.vmap_ode_sample_next_step(
                    obs_for_actor, policy_key
                )
                return action, {}

            def step_env(carry, _):
                key, env_state, obs, critic_obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                action, _ = ode_policy(act_key, obs, critic_obs)
                step_key = jax.random.split(env_key, env.num_envs)
                obs, critic_obs, env_state, reward, done, info = env.step(
                    step_key, env_state, action
                )
                return (key, env_state, obs, critic_obs), info

            key, init_key = jax.random.split(key)
            init_key = jax.random.split(init_key, env.num_envs)
            obs, critic_obs, env_state = env.reset(init_key, norm_state)
            key, env_key = jax.random.split(key)
            com_traj = None
            torso_env_indices = None
            if torso_id is not None and num_torso_envs > 0:
                key, sample_key = jax.random.split(key)
                torso_env_indices = jax.random.choice(
                    sample_key, env.num_envs, (num_torso_envs,), replace=False
                )

                def step_env_with_com(carry, _):
                    key, env_state, obs, critic_obs = carry
                    key, act_key, env_key = jax.random.split(key, 3)
                    action, _ = ode_policy(act_key, obs, critic_obs)
                    step_key = jax.random.split(env_key, env.num_envs)
                    obs, critic_obs, env_state, reward, done, info = env.step(
                        step_key, env_state, action
                    )
                    com = get_torso_com_all(env_state, torso_id)
                    sampled_com = com[torso_env_indices]
                    return (key, env_state, obs, critic_obs), (info, sampled_com)

                final_carry, (infos, com_traj) = jax.lax.scan(
                    f=step_env_with_com,
                    init=(key, env_state, obs, critic_obs),
                    xs=None,
                    length=max_episode_steps,
                )
                if torso_stride > 1:
                    com_traj = com_traj[::torso_stride]
            else:
                final_carry, infos = jax.lax.scan(
                    f=step_env,
                    init=(key, env_state, obs, critic_obs),
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

        return ode_evaluation_fn

    def _make_init_fn(self) -> Callable[[jax.Array], SACTrainState]:
        cfg = self.cfg
        env = self.env
        env_params = self.env_params

        def init(key: jax.random.PRNGKey) -> SACTrainState:
            key, model_key = jax.random.split(key)
            model_key, actor_key, actor_target_key = jax.random.split(model_key, 3)
            obs_dim, critic_obs_dim = env.get_obs_space_sizes()
            action_dim = env.action_space(env_params).shape[0]
            dt_schedule = hydra.utils.call(cfg.diffusion.dt_schedule)
            langevin_param = bool(cfg.diffusion.score_model.langevin_param)

            forward_model = None
            if cfg.diffusion.learn_forward:
                forward_model = ControlNetwork(
                    action_dim=action_dim,
                    observation_dim=obs_dim,
                    num_layers=cfg.diffusion.score_model.num_layers,
                    num_hid=cfg.diffusion.score_model.num_hid,
                    num_time_hid=cfg.diffusion.score_model.num_time_hid,
                    num_time_out=cfg.diffusion.score_model.num_time_out,
                    outer_clip=cfg.diffusion.score_model.outer_clip,
                    inner_clip=cfg.diffusion.score_model.inner_clip,
                    weight_init=cfg.diffusion.score_model.weight_init,
                    bias_init=cfg.diffusion.score_model.bias_init,
                    layer_norm=cfg.diffusion.score_model.layer_norm,
                    layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                    use_langevin_param=langevin_param,
                    max_time=cfg.diffusion.diff_steps,
                    rngs=nnx.Rngs(model_key),
                )

            backward_model = None
            if cfg.diffusion.learn_backward:
                backward_model = ControlNetwork(
                    action_dim=action_dim,
                    observation_dim=obs_dim,
                    num_layers=cfg.diffusion.score_model.num_layers,
                    num_hid=cfg.diffusion.score_model.num_hid,
                    num_time_hid=cfg.diffusion.score_model.num_time_hid,
                    num_time_out=cfg.diffusion.score_model.num_time_out,
                    outer_clip=cfg.diffusion.score_model.outer_clip,
                    inner_clip=cfg.diffusion.score_model.inner_clip,
                    weight_init=cfg.diffusion.score_model.weight_init,
                    bias_init=cfg.diffusion.score_model.bias_init,
                    layer_norm=cfg.diffusion.score_model.layer_norm,
                    layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                    use_langevin_param=langevin_param,
                    max_time=cfg.diffusion.diff_steps,
                    rngs=nnx.Rngs(model_key),
                )

            diffusion_model = DiffusionModel(
                action_dim=action_dim,
                observation_dim=obs_dim,
                fwd_model=forward_model,
                bwd_model=backward_model,
                diff_steps=cfg.diffusion.diff_steps,
                init_std=cfg.diffusion.init_std,
                friction=cfg.diffusion.friction,
                per_dim_friction=cfg.diffusion.per_dim_friction,
                use_friction_mlp=cfg.diffusion.use_friction_mlp,
                friction_mlp_hidden=cfg.diffusion.friction_mlp_hidden,
                friction_mlp_layers=cfg.diffusion.friction_mlp_layers,
                friction_num_time_hid=cfg.diffusion.friction_num_time_hid,
                friction_num_time_out=cfg.diffusion.friction_num_time_out,
                friction_mlp_use_obs=cfg.diffusion.friction_mlp_use_obs,
                dt=cfg.diffusion.dt,
                learn_dt=cfg.diffusion.learn_dt,
                per_step_dt=cfg.diffusion.per_step_dt,
                learn_prior=cfg.diffusion.learn_prior,
                learn_betas=cfg.diffusion.learn_betas,
                learn_friction=cfg.diffusion.learn_friction,
                learn_mass_matrix=cfg.diffusion.learn_mass_matrix,
                langevin_param=langevin_param,
                train_mode=cfg.train_mode,
                dt_schedule=dt_schedule,
                rngs=nnx.Rngs(model_key),
            )

            actor_networks = DMERLActor(
                action_dim=action_dim,
                observation_dim=obs_dim,
                diffusion_model=diffusion_model,
                logratio=logratio,
                kl_start=cfg.kl_start,
                ent_start=cfg.ent_start,
                sde_integrator=sde_integrator,
                ode_integrator=ode_integrator,
                action_clip_value=cfg.action_clip_value,
                tanh_transform=cfg.tanh_transform,
                use_temp_lagrangian_mlp=cfg.use_temp_lagrangian_mlp,
                temp_lagrangian_hidden=cfg.temp_lagrangian_hidden,
                rngs=nnx.Rngs(actor_key),
            )
            actor_target_networks = DMERLActor(
                action_dim=action_dim,
                observation_dim=obs_dim,
                diffusion_model=diffusion_model,
                logratio=logratio,
                kl_start=cfg.kl_start,
                ent_start=cfg.ent_start,
                sde_integrator=sde_integrator,
                ode_integrator=ode_integrator,
                tanh_transform=cfg.tanh_transform,
                use_temp_lagrangian_mlp=cfg.use_temp_lagrangian_mlp,
                temp_lagrangian_hidden=cfg.temp_lagrangian_hidden,
                rngs=nnx.Rngs(actor_target_key),
            )

            if cfg.hl_gauss:
                critic_networks: nnx.Module = CategoricalCriticNetwork(
                    obs_dim=critic_obs_dim,
                    action_dim=action_dim,
                    hidden_dim=cfg.critic_hidden_dim,
                    num_bins=cfg.num_bins,
                    vmin=cfg.vmin,
                    vmax=cfg.vmax,
                    num_time_hid=cfg.diffusion.score_model.num_time_hid,
                    num_time_out=cfg.diffusion.score_model.num_time_out,
                    use_norm=cfg.use_critic_norm,
                    encoder_layers=cfg.num_critic_encoder_layers,
                    use_simplical_embedding=cfg.use_simplical_embedding,
                    head_layers=cfg.num_critic_head_layers,
                    pred_layers=cfg.num_critic_pred_layers,
                    use_skip=cfg.use_critic_skip,
                    rngs=nnx.Rngs(model_key),
                )
            else:
                critic_networks = CriticNetwork(
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
                raise NotImplementedError("DMERL LR annealing not implemented.")
                num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
                num_updates = num_iterations * cfg.num_epochs * self.num_minibatches
                min_lr = cfg.lr * cfg.lr_decay_factor
                lr = optax.linear_schedule(cfg.lr, min_lr, num_updates)

            def _scale_lr(lr_val, mult: float):
                if callable(lr_val):
                    return lambda step: lr_val(step) * mult
                return lr_val * mult

            def _adam_with_decay(lr_val, weight_decay: float = 0., decay_mask=None, optim = optax.adam):
                tx = optim(lr_val)
                if weight_decay is not None and weight_decay > 0.0:
                    tx = optax.chain(
                        optax.add_decayed_weights(weight_decay, mask=decay_mask), tx
                    )
                return tx

            def _ema_optimizer(lr_val, decay: float):
                return optax.chain(
                    optax.ema(decay=decay),
                    optax.scale(-lr_val),
                )

            def _select_special_optimizer(name: str):
                if cfg.use_temp_lagrangian_ema_optim and cfg.use_temp_lagrangian_post_adam_ema:
                    raise ValueError(
                        "use_temp_lagrangian_ema_optim and use_temp_lagrangian_post_adam_ema "
                        "cannot both be true."
                    )
                if cfg.use_temp_lagrangian_ema_optim:
                    return lambda lr_val: _ema_optimizer(
                        lr_val, decay=cfg.temp_lagrangian_ema_decay
                    )

                name = name.lower()
                if name == "adam":
                    base_adam = partial(
                        optax.adam,
                        b1=cfg.temp_lagrangian_adam_gamma1,
                        b2=cfg.temp_lagrangian_adam_gamma2,
                    )
                    if cfg.use_temp_lagrangian_post_adam_ema:
                        return lambda lr_val: optax.chain(
                            base_adam(lr_val),
                            optax.ema(decay=cfg.temp_lagrangian_ema_decay),
                        )
                    
                    return base_adam
                if name == "sgd":
                    if cfg.use_temp_lagrangian_post_adam_ema:
                        raise ValueError(
                            "use_temp_lagrangian_post_adam_ema requires temp_lagrangian_optim='adam'."
                        )
                    return optax.sgd
                raise ValueError(f"Unknown temp/lagrangian optimizer '{name}', expected 'adam' or 'sgd'.")

            def _layernorm_projection_prefixes(flat_params):
                prefixes = set()
                for k in flat_params.keys():
                    for idx, name in enumerate(k):
                        if name == "layers" and idx + 1 < len(k) and k[idx + 1] == 1:
                            prefixes.add(k[: idx + 1])
                return prefixes

            def _is_projection_candidate(key, norm_prefixes):
                for idx, name in enumerate(key):
                    if name == "layers" and idx + 1 < len(key):
                        if key[idx + 1] == 0 and key[: idx + 1] in norm_prefixes:
                            return True
                return False

            def _unit_ball_projection(only_if_exceeds: bool = True):
                def init_fn(params):
                    return optax.EmptyState()

                def update_fn(updates, state, params=None):
                    if params is None:
                        raise ValueError("Params must be provided for projection.")

                    def project(u, p):
                        new_p = p + u
                        norm = jnp.linalg.norm(new_p)
                        if only_if_exceeds:
                            new_p = jnp.where(norm > 1.0, new_p / (norm + 1e-8), new_p)
                        else:
                            new_p = new_p / (norm + 1e-8)
                        return new_p - p

                    projected_updates = jax.tree.map(project, updates, params)
                    return projected_updates, state

                return optax.GradientTransformation(init_fn, update_fn)

            def _label_critic_params(params):
                flat = flatten_dict(params)
                norm_prefixes = _layernorm_projection_prefixes(flat)
                labels = {}
                for k in flat.keys():
                    leaf_name = k[-1]
                    if cfg.project_unit_ball and _is_projection_candidate(k, norm_prefixes):
                        labels[k] = "projected"
                    elif leaf_name in ("timestep_phase", "timestep_coeff"):
                        labels[k] = "no_decay"
                    else:
                        labels[k] = "default"
                return unflatten_dict(labels)

            critic_param_tree = nnx.to_pure_dict(nnx.state(critic_networks))
            critic_labels = _label_critic_params(critic_param_tree)

            critic_tx_cfg = {
                "default": _adam_with_decay(lr, weight_decay=cfg.weight_decay),
                "projected": optax.chain(
                    _adam_with_decay(lr, weight_decay=0.0),
                    _unit_ball_projection(cfg.project_only_if_exceeds),
                ),
                "no_decay": _adam_with_decay(lr, weight_decay=0.0),
            }
            critic_optimizer = optax.multi_transform(critic_tx_cfg, critic_labels)
            if cfg.max_grad_norm is not None:
                critic_optimizer = optax.chain(
                    optax.clip_by_global_norm(cfg.max_grad_norm), critic_optimizer
                )

            def _resolve_special_lr(direct_lr, mult: float):
                if direct_lr is not None:
                    return direct_lr
                return _scale_lr(lr, mult)

            def _label_actor_params(params):
                flat = flatten_dict(params)  # tuple keys to avoid char-splitting
                norm_prefixes = _layernorm_projection_prefixes(flat)
                labels = {}
                for k in flat.keys():
                    leaf_name = k[-1]
                    if cfg.project_unit_ball and _is_projection_candidate(k, norm_prefixes):
                        labels[k] = "projected"
                    elif leaf_name in ("timestep_phase", "timestep_coeff"):
                        labels[k] = "no_decay"
                    elif "temperature" in leaf_name:
                        labels[k] = "temperature"
                    elif "lagrangian" in leaf_name:
                        labels[k] = "lagrangian"
                    else:
                        labels[k] = "default"
                return unflatten_dict(labels)

            actor_param_tree = nnx.to_pure_dict(nnx.state(actor_networks))
            actor_labels = _label_actor_params(actor_param_tree)
            temperature_lr = _resolve_special_lr(
                cfg.temperature_lr, cfg.temperature_lr_mult
            )
            lagrangian_lr = _resolve_special_lr(
                cfg.lagrangian_lr, cfg.lagrangian_lr_mult
            )

            special_optimizer = _select_special_optimizer(cfg.temp_lagrangian_optim)

            actor_tx_cfg = {
                "default": _adam_with_decay(lr, weight_decay=0.0),
                "projected": optax.chain(
                    _adam_with_decay(lr, weight_decay=0.0),
                    _unit_ball_projection(cfg.project_only_if_exceeds),
                ),
                "temperature": _adam_with_decay(temperature_lr, weight_decay=0.0, optim=special_optimizer),
                "lagrangian": _adam_with_decay(lagrangian_lr, weight_decay=0.0, optim=special_optimizer),
                "no_decay": _adam_with_decay(lr, weight_decay=0.0),
            }
            actor_optimizer = optax.multi_transform(actor_tx_cfg, actor_labels)
            if cfg.max_grad_norm is not None:
                actor_optimizer = optax.chain(
                    optax.clip_by_global_norm(cfg.max_grad_norm), actor_optimizer
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
                params=critic_param_tree,
                tx=critic_optimizer,
            )

            params_actor = jax.tree.map(lambda x: x[0], actor_trainstate.params)
            params_critic = jax.tree.map(lambda x: x[0], critic_trainstate.params)
            #print(params_actor)
            # jax.debug.print("Actor params: {params}", params=params_actor)
            # #print(params_critic)
            # jax.debug.print("Critic params: {params}", params=params_critic)
            # jax.debug.print("Actor norm: {params}", params=jax.tree.map(lambda x: jnp.linalg.norm(x), actor_trainstate.params))
            # #print(jax.tree.map(lambda x: jnp.linalg.norm(x), actor_trainstate.params))
            # #print(jax.tree.map(lambda x: jnp.linalg.norm(x), critic_trainstate.params))
            # jax.debug.print("Critic norm: {params}", params=jax.tree.map(lambda x: jnp.linalg.norm(x), critic_trainstate.params))
            # jax.debug.callback(lambda p: _host_print_layer_norms("actor", p), params_actor)
            # jax.debug.callback(lambda p: _host_print_layer_norms("critic", p), params_critic)
            #raise ValueError("Layer norms printed for debugging; remove after inspection.")

            key, env_key = jax.random.split(key)
            env_key = jax.random.split(env_key, cfg.num_envs)
            obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

            # Fully unwrap to the base env state (e.g., MjxGymnaxWrapper state) for step randomization.
            key, env_state = randomize_env_steps(
                key, env_state, cfg.max_episode_steps
            )

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

    def _collect_rollout(
        self, key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        cfg = self.cfg
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        diff_steps = self.cfg.diffusion.diff_steps

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

        use_langevin = self.use_langevin_param
        key, init_act_key = jax.random.split(key)
        obs_for_actor = maybe_add_q_grad(
            train_state.last_obs,
            train_state.last_critic_obs,
            actor_model,
            critic_model,
            use_langevin,
        )
        init_action, _, _ = actor_model.vmap_sample_next_step(
            obs_for_actor, init_act_key
        )
        init_action = jax.lax.stop_gradient(init_action)

        step_env = lambda carry, _: self.train_step_env(actor_model, critic_model, carry, _)
        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
                init_action,
            ),
            length=self.num_collection_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs, _ = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + (self.num_collection_steps * cfg.num_envs)//self.cfg.diffusion.diff_steps,
        )
        return transitions, train_state

    def train_step_env(self, actor_model, critic_model, carry, _):
        key, env_state, inner_state, obs, critic_obs, action = carry
        use_langevin = self.use_langevin_param
        key, next_act_key, step_key = jax.random.split(key, 3)
        step_key = jax.random.split(step_key, self.cfg.num_envs)
        action = jax.lax.stop_gradient(action)
        next_obs, next_critic_obs, next_env_state, reward, done, info = self.env.step(
            step_key, env_state, action
        )
        importance_weight = jnp.zeros((self.cfg.num_envs,))
        next_obs_for_actor = maybe_add_q_grad(
            next_obs, next_critic_obs, actor_model, critic_model, use_langevin
        )
        next_action, next_gen_log_prob, next_dest_log_prob = (
            actor_model.vmap_sample_next_step(next_obs_for_actor, next_act_key)
        )
        next_action = jax.lax.stop_gradient(next_action)
        next_emb, _, _, _, value = critic_model.forward(next_critic_obs, next_action)
        log_ratio = jax.lax.stop_gradient(
            next_gen_log_prob - next_dest_log_prob
        )
        temperature = _resolve_temperature(actor_model, self.cfg, inner_state)
        soft_reward = (
            reward
            - self.cfg.gamma * log_ratio.squeeze() * temperature
        )
        transition = Transition(
            obs=obs,
            critic_obs=critic_obs,
            action=action,
            next_emb=next_emb,
            next_state_emb=next_emb,
            next_emb_mask=jnp.ones_like(reward),
            reward=reward,
            soft_reward=soft_reward,
            value=value,
            done=done,
            truncated=next_env_state.env_state.truncated,
            info=info,
            importance_weight=importance_weight,
        )
        return (
            key,
            next_env_state,
            inner_state,
            next_obs,
            next_critic_obs,
            next_action,
        ), transition

    def _learn_step(
        self,
        key: PRNGKey,
        train_state: SACTrainState,
        batch: Transition,
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        cfg = self.cfg
        action_size_target = self.action_size_target

        # Build the TD-lambda scan body via the helper module so we avoid
        # defining tiny inner functions in the trainer.
        nstep_fn = partial(
            compute_nstep_lambda_step,
            cfg.gamma,
            cfg.lmbda,
        )
        _, target_values = jax.lax.scan(
            nstep_fn,
            (
                batch.value[-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.importance_weight[0]),
            ),
            batch,
            reverse=True,
        )
        # print min max and mean values of target_values for debugging
        # jax.debug.print("target_values stats - min: {min}, max: {max}, mean: {mean}",
        #                 min=jnp.min(target_values),
        #                 max=jnp.max(target_values),
        #                 mean=jnp.mean(target_values))
        target_vals_flat = target_values.reshape(-1)
        target_vals_finite = jnp.nan_to_num(
            target_vals_flat,
            nan=0.0,
            posinf=cfg.vmax,
            neginf=cfg.vmin,
        )
        target_val_mean = jnp.mean(target_vals_finite)
        target_val_min = jnp.min(target_vals_finite)
        target_val_max = jnp.max(target_vals_finite)
        target_val_hist_counts, target_val_hist_edges = jnp.histogram(
            target_vals_finite,
            bins=cfg.num_bins,
        )
        shift_steps = self.cfg.diffusion.diff_steps
        time_idx = jnp.arange(self.num_collection_steps)
        shifted_idx = jnp.minimum(time_idx + shift_steps, self.num_collection_steps - 1)
        next_state_emb = jnp.take(batch.next_emb, shifted_idx, axis=0)
        valid_shift = (time_idx + shift_steps) <= (self.num_collection_steps - 1)
        next_emb_mask = jnp.broadcast_to(
            valid_shift[:, None], (self.num_collection_steps, cfg.num_envs)
        )
        batch = batch.replace(next_state_emb=next_state_emb, next_emb_mask=next_emb_mask)
        # Flatten rollout data to (num_steps * num_envs, ...) for easier indexing.
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((self.num_collection_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        key, train_key = jax.random.split(key)
        # Each epoch draws a fresh permutation and feeds it through the helper
        # minibatch update; using partial keeps the training loop concise.
        epoch_fn = partial(
            self._run_epoch_update,
            data=data,
            action_size_target=action_size_target,
            actor_target_model=actor_target_model,
        )
        train_state, update_metrics = jax.lax.scan(
            f=epoch_fn,
            init=train_state,
            xs=jax.random.split(train_key, cfg.num_epochs)
        )
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        update_metrics["target_value_nonfinite"] = jnp.sum(
            ~jnp.isfinite(target_vals_flat)
        )
        update_metrics["target_value_hist_counts"] = target_val_hist_counts
        update_metrics["target_value_hist_edges"] = target_val_hist_edges
        update_metrics["target_value_mean"] = target_val_mean
        update_metrics["target_value_min"] = target_val_min
        update_metrics["target_value_max"] = target_val_max
        temperature = update_metrics["temp"]
        lagrangian = update_metrics["lagrangian"]
        # jax.debug.print("Temperature: {temperature}, Lagrangian: {lagrangian}",
        #                 temperature=temperature, lagrangian=lagrangian)
        return train_state, update_metrics

    def _run_epoch_update(
        self,
        train_state: SACTrainState,
        epoch_key: PRNGKey,
        *,
        data,
        action_size_target: float,
        actor_target_model,
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        """Shuffle data once and run minibatch SGD updates for a single epoch."""
        cfg = self.cfg
        mini_batch_size = (self.num_collection_steps * cfg.num_envs) // self.num_minibatches
        indices = jax.random.permutation(epoch_key, self.num_collection_steps * cfg.num_envs)
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape((self.num_minibatches, mini_batch_size, *x.shape[1:])),
            indices,
        )
        minibatch_keys = jax.random.split(epoch_key, self.num_minibatches)
        scan_inputs = (minibatch_idxs, minibatch_keys)
        minibatch_fn = partial(
            self.minibatch_update_step,
            cfg,
            action_size_target,
            actor_target_model,
            data,
        )
        train_state, metrics = jax.lax.scan(
            minibatch_fn,
            train_state,
            scan_inputs,
        )
        metrics = jax.tree.map(lambda x: x.mean(0), metrics)
        return train_state, metrics

    def minibatch_update_step(self, 
        cfg,
        action_size_target: float,
        actor_target_model,
        data,
        train_state,
        inputs,
    ):
        """Run one SGD step over a minibatch and return updated train state and metrics."""
        indices, step_key = inputs
        minibatch, target_vals = jax.tree.map(
            lambda x: jnp.take(x, indices, axis=0), data
        )

        critic_loss_fn_ = lambda p: critic_loss_fn(p, train_state, minibatch, target_vals, cfg)
        
        critic_grad_fn = jax.value_and_grad(critic_loss_fn_, has_aux=True)
        critic_output, critic_grads = critic_grad_fn(train_state.critic.params)
        critic_train_state = train_state.critic.apply_gradients(critic_grads)
        updated_state = train_state.replace(critic=critic_train_state)
        critic_metrics = critic_output[1]
        critic_metrics["critic_gnorm"] = utils.tree_norm(critic_grads)
        critic_metrics["timestep_coeff_norm"] = _timestep_coeff_norm(
            critic_train_state.params
        )

        critic_rollout_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        selected_actor_loss = (
            actor_WPO_loss_fn
            if cfg.train_mode == "WPO"
            else actor_loss_fn
        )
        if selected_actor_loss is actor_WPO_loss_fn:
            actor_loss_fn_ = lambda p: selected_actor_loss(
                p,
                updated_state,
                critic_rollout_model,
                step_key,
                minibatch,
                target_vals,
                action_size_target,
                cfg,
                actor_target_model,
            )
        else:
            actor_loss_fn_ = lambda p: selected_actor_loss(
                p,
                updated_state,
                critic_rollout_model,
                step_key,
                minibatch,
                target_vals,
                action_size_target,
                cfg,
                actor_target_model,
            )

        actor_grad_fn = jax.value_and_grad(actor_loss_fn_, has_aux=True)
        actor_output, actor_grads = actor_grad_fn(updated_state.actor.params)
        actor_metrics = actor_output[1]
        actor_train_state = updated_state.actor.apply_gradients(actor_grads)
        updated_state = updated_state.replace(actor=actor_train_state)
        actor_metrics["actor_gnorm"] = utils.tree_norm(actor_grads)
        actor_metrics["actor_timestep_coeff_norm"] = _timestep_coeff_norm(
            actor_train_state.params
        )

        metrics = {**critic_metrics, **actor_metrics}
        return updated_state, metrics

    def _train_eval_step(
        self, key: PRNGKey, train_state: SACTrainState
    ) -> tuple[SACTrainState, dict]:
        cfg = self.cfg

        def train_step(
            state: SACTrainState, subkey: PRNGKey
        ) -> tuple[SACTrainState, dict[str, jax.Array]]:
            key, rollout_key, learn_key = jax.random.split(subkey, 3)
            transitions, state = self._collect_rollout(key=rollout_key, train_state=state)
            state, update_metrics = self._learn_step(
                key=learn_key, train_state=state, batch=transitions
            )
            metrics = {**update_metrics}
            state = state.replace(iteration=state.iteration + 1)
            return state, metrics

        train_key, eval_key = jax.random.split(key)
        eval_interval = self.eval_interval
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        norm_state = train_state.last_env_state if cfg.normalize_env else None
        eval_key, init_seed_key = jax.random.split(eval_key)

        eval_metrics = self.eval_fn(init_seed_key, train_state, norm_state)

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

    def _loop_body(
        self, train_state: SACTrainState, key: PRNGKey
    ) -> tuple[SACTrainState, dict]:
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(self._train_eval_step)(
            jax.random.split(subkey, self.num_seeds), train_state
        )
        jax.debug.callback(self.log_callback, train_state, metrics)
        return train_state, metrics

    def _train_loop(self, key: PRNGKey) -> tuple[SACTrainState, dict]:
        # num iteratns defines the number of training steps for each logging interval
        num_iterations = self.num_iterations
        key, init_key = jax.random.split(key)
        init_fn = self._make_init_fn()
        train_state = jax.vmap(init_fn)(jax.random.split(init_key, self.num_seeds))

        actor_init_norm = utils.tree_norm(train_state.actor.params)
        critic_init_norm = utils.tree_norm(train_state.critic.params)
        # count parameters per-network (take first seed to avoid double-counting vmapped params)
        actor_param_count = utils.count_params(jax.tree.map(lambda x: x[0], train_state.actor.params))
        critic_param_count = utils.count_params(jax.tree.map(lambda x: x[0], train_state.critic.params))

        def _log_init_norms(actor_norm, critic_norm, actor_count, critic_count):
            log_data = {
                "norm_init/actor": float(np.asarray(actor_norm).mean()),
                "norm_init/critic": float(np.asarray(critic_norm).mean()),
                "norm_init/actor_params": int(actor_count),
                "norm_init/critic_params": int(critic_count),
            }
            wandb.log(_sectioned_wandb_log(log_data), step=0)

        jax.debug.callback(
            _log_init_norms,
            actor_init_norm,
            critic_init_norm,
            actor_param_count,
            critic_param_count,
        )
        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(
            f=self._loop_body,
            init=train_state,
            xs=keys,
        )
        return state, metrics

    def build_train_fn(self) -> Callable[[PRNGKey, ReppoConfig], tuple[SACTrainState, dict]]:
        def train_fn(key: PRNGKey, cfg: ReppoConfig):
            if cfg != self.cfg:
                logging.warning(
                    "Received cfg argument different from trainer configuration; using trainer cfg."
                )
            return self._train_loop(key)

        return train_fn


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
    sweep_metrics = []

    if trial is not None:
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
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
    save_dir = None

    def _move_metrics_to_norm(log_data: dict[str, Any], keys: list[str]) -> None:
        """Move selected metrics into the norm/ namespace to avoid duplicate logging."""
        for key in keys:
            if key in log_data:
                value = log_data.pop(key)
                suffix = key.split("/", 1)[1] if "/" in key else key
                log_data[f"norm/{suffix}"] = value

    def log_callback(state, metrics):
        nonlocal save_dir
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        if save_dir is None:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            save_dir = os.path.join(repo_root, "saved_models")
            os.makedirs(save_dir, exist_ok=True)

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()

        lr_cfg = cfg.hyperparameters
        actor_step = int(np.asarray(state.actor.step).max())
        if lr_cfg.anneal_lr:
            num_iterations = (
                lr_cfg.total_time_steps // lr_cfg.num_steps // lr_cfg.num_envs
            )
            num_updates = max(
                1, num_iterations * lr_cfg.num_epochs * lr_cfg.num_mini_batches*lr_cfg.diffusion.diff_steps
            )
            progress = min(actor_step / num_updates, 1.0)
            min_lr = lr_cfg.lr * lr_cfg.lr_decay_factor
            current_lr = float((1.0 - progress) * lr_cfg.lr + progress * min_lr)
        else:
            current_lr = float(lr_cfg.lr)

        log_msg = (
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, "
            f"episode_length={eval_length:.3f}, lr={current_lr:.6f}"
        )
        for key_name in metrics.keys():
            if "episode_return_ode_" in key_name:
                ode_coef_str = key_name.split("_ode_")[-1]
                ode_coef = float(ode_coef_str) / 100.0
                ode_return = metrics[key_name].mean()
                log_msg += f", ode_{ode_coef}_return={ode_return:.3f}"
        log_msg += f" sps={sps:.2f}"
        logging.info(log_msg)

        train_metrics = utils.filter_prefix("train", metrics)
        target_hist_counts = train_metrics.pop("train/target_value_hist_counts", None)
        target_hist_edges = train_metrics.pop("train/target_value_hist_edges", None)
        torso_com_traj = metrics.pop("eval/torso_com_traj", None)
        torso_com_env_indices = metrics.pop("eval/torso_com_env_indices", None)

        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            "train/lr": current_lr,
            "sps": sps,
            **jax.tree.map(jnp.mean, train_metrics),
        }
        for key_name, value in metrics.items():
            if key_name.startswith("eval/"):
                log_data[key_name] = value.mean() if hasattr(value, "mean") else value
        if target_hist_counts is not None and target_hist_edges is not None:
            # Convert JAX arrays to NumPy before plotting to ensure wandb.Image
            # receives a fully rendered Matplotlib figure.
            counts = np.asarray((target_hist_counts))
            edges = np.asarray((target_hist_edges))
            if counts.ndim > 1:
                counts = counts.sum(axis=0)
            if edges.ndim > 1:
                edges = edges[0]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                edgecolor="black",
            )
            ax.set_title("Target Value Histogram")
            ax.set_xlabel("Target value")
            ax.set_ylabel("Count")
            fig.tight_layout()
            log_data["figures/target_value_histogram"] = wandb.Image(fig)
            # save the figure to a file for debugging. path should be same folder as the script
            path = os.path.join(os.getcwd(), "target_value_histogram.png")
            fig.savefig(path)
            plt.close(fig)
        fig = build_torso_com_traj_figure(
            torso_com_traj, torso_com_env_indices, title="Torso COM trajectory (XY)"
        )
        if fig is not None:
            log_data["figures/eval_torso_com_traj_xy"] = wandb.Image(fig)
            plt.close(fig)
        if torso_com_traj is not None:
            step_id = int(np.asarray(state.time_steps[0]))
            save_torso_com_trajectory(
                f"DMERL_traj_step_{step_id}.pkl",
                torso_com_traj,
                torso_com_env_indices,
            )

        actor_params = jax.tree.map(lambda x: x[0], state.actor.params)
        actor_model = nnx.merge(state.actor.graphdef, actor_params)
        diff_model = actor_model.diffusion_model

        obs0 = jax.tree.map(lambda x: x[0], state.last_obs)
        obs0 = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.shape and x.shape[0] > 0 else x,
            obs0,
        )

        if isinstance(obs0, dict):
            obs_dict = {k: obs0[k] for k in obs0}
            if "orig_obs" not in obs_dict:
                raise KeyError("obs_dict missing 'orig_obs' for diffusion logging")
            if "normed_actions" not in obs_dict:
                obs_dict["normed_actions"] = jnp.zeros(
                    (diff_model.action_dim,), dtype=jnp.float32
                )
        else:
            obs_dict = {
                "orig_obs": obs0,
                "normed_actions": jnp.zeros(
                    (diff_model.action_dim,), dtype=jnp.float32
                ),
            }

        obs_dict["orig_obs"] = jnp.asarray(obs_dict["orig_obs"])
        obs_dict["normed_actions"] = jnp.asarray(obs_dict["normed_actions"])

        def _scale_for_step(step):
            obs_step = dict(obs_dict)
            obs_step["diff_time_step"] = jnp.array([[step]], dtype=jnp.float32)
            scale, _, _ = diff_model.diffusion_coeff_fn(
                jnp.array(step, dtype=jnp.int32), obs_step
            )
            return scale

        scales = jnp.stack(
            [_scale_for_step(step) for step in range(diff_model.diff_steps)]
        )
        scales_np = np.asarray(scales)
        if scales_np.ndim > 1:
            scale_mean = scales_np.mean(axis=-1)
        else:
            scale_mean = scales_np

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(diff_model.diff_steps), scale_mean, marker="o", markersize=2)
        ax.set_title("Diffusion coefficient scale vs diffusion step")
        ax.set_xlabel("Diffusion step")
        ax.set_ylabel("Scale (mean over action dims)")
        fig.tight_layout()
        log_data["figures/diffusion_coeff_scale"] = wandb.Image(fig)
        plt.close(fig)

        # compute the effective learning rate of the actor and the critic
        actor_gnorm = log_data.get("train/actor_gnorm", 0.0)
        actor_pnorm = log_data.get("train/actor_pnorm", 0.0)
        critic_gnorm = log_data.get("train/critic_gnorm", 0.0)
        critic_pnorm = log_data.get("train/critic_pnorm", 0.0)

        actor_effective_lr = (
            current_lr * (actor_gnorm / (actor_pnorm + 1e-10))
            if actor_pnorm > 0
            else 0.0
        )
        critic_effective_lr = (
            current_lr * (critic_gnorm / (critic_pnorm + 1e-10))
            if critic_pnorm > 0
            else 0.0
        )
        log_data["norm/actor_effective_lr"] = actor_effective_lr
        log_data["norm/critic_effective_lr"] = critic_effective_lr
        _move_metrics_to_norm(
            log_data,
            [
                "train/actor_pnorm",
                "train/critic_pnorm",
                "train/actor_gnorm",
                "train/critic_gnorm",
                "train/timestep_coeff_norm",
                "train/actor_timestep_coeff_norm",
            ],
        )

        wandb.log(_sectioned_wandb_log(log_data), step=int(state.time_steps[0]))

        step = int(np.asarray(state.time_steps[0]))
        checkpoint = {
            "actor_params": _to_numpy_tree(state.actor.params),
            "actor_target_params": _to_numpy_tree(state.actor_target.params),
            "critic_params": _to_numpy_tree(state.critic.params),
            "actor_step": _to_numpy_tree(state.actor.step),
            "critic_step": _to_numpy_tree(state.critic.step),
            "time_steps": _to_numpy_tree(state.time_steps),
            "iteration": _to_numpy_tree(state.iteration),
            "num_seeds": int(np.asarray(state.time_steps).shape[0])
            if np.asarray(state.time_steps).ndim > 0
            else 1,
            "last_env_state": _to_numpy_tree(state.last_env_state)
            if cfg.hyperparameters.normalize_env
            else None,
            "eval_metrics": _to_numpy_tree(utils.filter_prefix("eval", metrics)),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "saved_at": time.time(),
        }
        save_path = os.path.join(save_dir, "checkpoint.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(checkpoint, f)

    if cfg.env.type == "brax":
        raise ValueError("Wrappers are not implemented yet")
    elif cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
        diff_cfg = cfg.hyperparameters.diffusion
        env_action_clip_value = cfg.hyperparameters.env_action_clip_value
        env = MjxDiffEnvWrapper(
            env,
            num_diff_steps=diff_cfg.diff_steps,
            diffusion_config=diff_cfg,
            low=-env_action_clip_value,
            high=env_action_clip_value,
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    trainer = ReppoDMERLTrainer(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    train_fn = trainer.build_train_fn()

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i
        run_config = OmegaConf.to_container(cfg)
        run_config["method_name"] = "reppo_DMERL_new"
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
            name=f"{cfg.name}-{cfg.env.name.lower()}-{getattr(cfg.hyperparameters, 'train_mode', 'reparam')}",
            save_code=True,
        )

        trainer._init_step_sizes()
        logging.info(OmegaConf.to_yaml(cfg))
        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        _, metrics = jax.jit(train_fn, static_argnums=(1,))(key, trainer.cfg)
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        wandb.finish()
        sweep_metrics.append(metrics["eval/episode_return"])
        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="reppo_dmerl")
def main(cfg: DictConfig):
    cfg.hyperparameters = OmegaConf.merge(
        cfg.hyperparameters, cfg.experiment_overrides.hyperparameters
    )
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
