import logging
import math
import os
import pickle
import re
import time
import typing
from typing import Any, Callable, Optional
import functools

import distrax
import numpy as np
import hydra
import jax
import optax
import plotly.graph_objs as go
from hydra.core.hydra_config import HydraConfig
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
    CategoricalValueNetwork,
    DiffValueNetwork,
    DiffusionModel,
    DMERLActor,
    logratio_DIME as logratio,
    ode_integrator,
    sde_integrator,
)
from src.jaxrl import utils
from src.jaxrl.normalization import (
    DictNormalizationState,
    DictNormalizer,
    NormalizationState,
    Normalizer,
)
from src.jaxrl.reppo_helpers.rollout_aux_targets import build_rollout_aux_targets
from src.jaxrl.reppo_helpers.diffusion_index_sampling import (
    prepare_diffusion_importance_sampling,
    sample_minibatch_indices,
)
from src.jaxrl.reppo_helpers.env_time_discounting import (
    maybe_env_time_discount_lambda,
    maybe_env_time_value,
)
from src.jaxrl.reppo_helpers.learning_DA_MDP_PPO import compute_gae_step
from src.jaxrl.reppo_DMERL_old import randomize_env_steps


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


def _resolve_train_mode(cfg: DictConfig) -> str:
    mode = OmegaConf.select(cfg, "hyperparameters.train_mode")
    if mode is not None and str(mode).strip():
        return str(mode)
    mode = OmegaConf.select(cfg, "hyperparameters.diffusion.train_mode")
    if mode is not None and str(mode).strip():
        return str(mode)
    return "reparam"


def _sectioned_wandb_key(key: str) -> str:
    if key.startswith("/"):
        key = key.lstrip("/")
    if key == "sps":
        return "sps"
    if key.startswith("learning_rate/"):
        suffix = key.split("/", 1)[1]
        return f"learning_rate/{suffix}"
    if key.startswith("train/"):
        suffix = key.split("/", 1)[1]
        if suffix.startswith(("temp", "entropy", "target_entropy")):
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
    if key.startswith("normalization/"):
        suffix = key.split("/", 1)[1]
        return f"normalization/{suffix}"
    if key.startswith("figures/"):
        suffix = key.split("/", 1)[1]
        return f"figures/{suffix}"
    if key.startswith("step_metrics/"):
        suffix = key.split("/", 1)[1]
        return f"step_metrics/{suffix}"
    return f"system/{key}"


def _sectioned_wandb_log(log_data: dict[str, Any]) -> dict[str, Any]:
    return {_sectioned_wandb_key(key): value for key, value in log_data.items()}


def _dict_norm_stats(prefix: str, state: DictNormalizationState | None) -> dict[str, jax.Array]:
    stats: dict[str, jax.Array] = {}
    if state is None:
        return stats

    if state.obs_state is not None:
        stats[f"{prefix}_obs_norm_count"] = jnp.asarray(
            state.obs_state.count, dtype=jnp.float32
        )
        stats[f"{prefix}_obs_norm_std"] = jnp.sqrt(jnp.mean(state.obs_state.var))
        stats[f"{prefix}_obs_norm_mean_abs"] = jnp.mean(jnp.abs(state.obs_state.mean))

    if state.actions_state is not None:
        stats[f"{prefix}_action_norm_count"] = jnp.asarray(
            state.actions_state.count, dtype=jnp.float32
        )
        stats[f"{prefix}_action_norm_std"] = jnp.sqrt(
            jnp.mean(state.actions_state.var)
        )
        stats[f"{prefix}_action_norm_mean_abs"] = jnp.mean(
            jnp.abs(state.actions_state.mean)
        )
    return stats


def _weighted_batch_mean(values: jax.Array, importance_ratio: jax.Array | None) -> jax.Array:
    """Importance-weighted mean over batch axis (axis 0)."""
    values = jnp.asarray(values)
    if values.ndim == 0:
        raise ValueError(
            "_weighted_batch_mean expects per-sample values with a batch axis, got scalar."
        )
    if importance_ratio is None:
        return jnp.mean(values)
    ratio = jnp.asarray(importance_ratio, dtype=values.dtype).reshape((-1,))
    if values.shape[0] != ratio.shape[0]:
        raise ValueError(
            "_weighted_batch_mean shape mismatch: values batch axis "
            f"{values.shape[0]} != importance_ratio length {ratio.shape[0]}."
        )
    ratio = ratio.reshape((ratio.shape[0],) + (1,) * (values.ndim - 1))
    return jnp.mean(ratio * values)


def _weighted_batch_mean_axis0(
    values: jax.Array, importance_ratio: jax.Array | None
) -> jax.Array:
    """Importance-weighted mean over batch axis while preserving remaining axes."""
    values = jnp.asarray(values)
    if values.ndim == 0:
        raise ValueError(
            "_weighted_batch_mean_axis0 expects per-sample values with a batch axis, got scalar."
        )
    if importance_ratio is None:
        return jnp.mean(values, axis=0)
    ratio = jnp.asarray(importance_ratio, dtype=values.dtype).reshape((-1,))
    if values.shape[0] != ratio.shape[0]:
        raise ValueError(
            "_weighted_batch_mean_axis0 shape mismatch: values batch axis "
            f"{values.shape[0]} != importance_ratio length {ratio.shape[0]}."
        )
    ratio = ratio.reshape((ratio.shape[0],) + (1,) * (values.ndim - 1))
    return jnp.mean(ratio * values, axis=0)


def _rsl_gaussian_kl(
    old_mu: jax.Array,
    old_sigma: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    sigma = jnp.maximum(sigma, 1e-8)
    old_sigma = jnp.maximum(old_sigma, 1e-8)
    return jnp.sum(
        jnp.log(sigma / old_sigma + 1e-5)
        + (jnp.square(old_sigma) + jnp.square(old_mu - mu)) / (2.0 * jnp.square(sigma))
        - 0.5,
        axis=-1,
    )


def _replace_learning_rate_in_opt_state(
    opt_state: Any,
    learning_rate: jax.Array,
    *,
    transform_names: set[str] | None = None,
) -> Any:
    if isinstance(opt_state, tuple) and hasattr(opt_state, "_fields"):
        updated = tuple(
            _replace_learning_rate_in_opt_state(
                state, learning_rate, transform_names=transform_names
            )
            for state in opt_state
        )
        try:
            return type(opt_state)(*updated)
        except TypeError:
            return opt_state._replace(**dict(zip(opt_state._fields, updated)))

    if isinstance(opt_state, tuple):
        return tuple(
            _replace_learning_rate_in_opt_state(
                state, learning_rate, transform_names=transform_names
            )
            for state in opt_state
        )

    if hasattr(opt_state, "inner_states") and hasattr(opt_state, "_replace"):
        inner_states = dict(opt_state.inner_states)
        updated = {}
        for name, state in inner_states.items():
            if transform_names is None or name in transform_names:
                updated[name] = _replace_learning_rate_in_opt_state(
                    state, learning_rate, transform_names=None
                )
            else:
                updated[name] = state
        return opt_state._replace(inner_states=updated)

    if hasattr(opt_state, "inner_state") and hasattr(opt_state, "_replace"):
        return opt_state._replace(
            inner_state=_replace_learning_rate_in_opt_state(
                opt_state.inner_state,
                learning_rate,
                transform_names=transform_names,
            )
        )

    if hasattr(opt_state, "hyperparams") and hasattr(opt_state, "_replace"):
        hyperparams = dict(opt_state.hyperparams)
        if "learning_rate" in hyperparams:
            hyperparams["learning_rate"] = jnp.asarray(learning_rate, dtype=jnp.float32)
            return opt_state._replace(hyperparams=hyperparams)

    return opt_state


def _update_adaptive_lr(cfg: "PPOConfig", current_lr: jax.Array, kl_mean: jax.Array) -> jax.Array:
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


def _compute_forward_kernel_moments(actor_module: DMERLActor, obs: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    diffusion_model = actor_module.diffusion_model
    current_x = obs["orig_actions"]
    step = obs["diff_time_step"][..., 0]

    def _single_step_moments(step_i, x_i, obs_i):
        mu_i, sigma_i, eta_i = diffusion_model.compute_diffusion_stuff(
            step_i, x_i, obs_i, model=diffusion_model.forward_model
        )
        return x_i + eta_i * mu_i, jnp.maximum(sigma_i, 1e-8)

    return jax.vmap(_single_step_moments)(step, current_x, obs)


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


def _extract_hyperparameter_overrides(cfg: DictConfig, path: str) -> DictConfig:
    """Accept both group styles:
    1) nested: <group>.hyperparameters.*
    2) flat:   <group>.*
    """

    def _to_cfg(value: Any) -> DictConfig:
        if value is None:
            return OmegaConf.create({})
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=False)
        if isinstance(value, dict):
            return OmegaConf.create(value)
        return OmegaConf.create({})

    def _collect_nested_hyperparameters(value: Any) -> list[dict]:
        if isinstance(value, dict):
            collected: list[dict] = []
            for key, sub_value in value.items():
                if key == "hyperparameters" and isinstance(sub_value, dict):
                    collected.append(sub_value)
                    continue
                collected.extend(_collect_nested_hyperparameters(sub_value))
            return collected
        if isinstance(value, list):
            collected: list[dict] = []
            for item in value:
                collected.extend(_collect_nested_hyperparameters(item))
            return collected
        return []

    raw_cfg = _to_cfg(OmegaConf.select(cfg, path, default={}))
    raw_obj = OmegaConf.to_container(raw_cfg, resolve=False)
    if not isinstance(raw_obj, dict):
        return OmegaConf.create({})

    nested_candidates = _collect_nested_hyperparameters(raw_obj)
    if nested_candidates:
        merged = OmegaConf.create({})
        for candidate in nested_candidates:
            merged = OmegaConf.merge(merged, OmegaConf.create(candidate))
        return merged

    flat_overrides = {key: value for key, value in raw_obj.items() if key != "defaults"}
    return OmegaConf.create(flat_overrides)


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
    normalize_rewards: bool = False
    reward_norm_epsilon: float = 1e-8
    reward_norm_clip: float = 10.0
    normalize_reward: bool = False
    normalize_soft_reward: bool = False
    adaptive_lr: bool = False
    desired_kl: float = 0.01
    adaptive_lr_factor: float = 1.5
    adaptive_lr_min: float = 1e-5
    adaptive_lr_max: float = 1e-2
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
    critic_use_normed_actions: bool = False
    use_categorical_value: bool = False
    vmin: float = -10.0
    vmax: float = 10.0
    num_bins: int = 51
    hl_gauss: bool = False
    aux_loss_mult: float = 0.0
    aux_loss_alpha: float = 0.9
    use_final_step_reward_target: bool = False
    use_env_time_discounting: bool = False
    action_clip_value: float = 1.0
    tanh_transform: bool = False
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
    temp_lagrangian_optim: str = "adam"
    temp_lagrangian_adam_gamma1: float = 0.9
    temp_lagrangian_adam_gamma2: float = 0.999
    use_temp_lagrangian_ema_optim: bool = False
    use_temp_lagrangian_post_adam_ema: bool = False
    temp_lagrangian_ema_decay: float = 0.99
    temperature_lr: float | None = None
    weight_decay: float = 0.0
    num_collection_step_factor: float = 1.0
    use_temp_lagrangian_mlp: bool = False
    temp_lagrangian_hidden: int = 64
    importance_sample_diffusion_steps: bool = False
    diffusion_step_sampling_mode: str = "power"
    diffusion_step_sampling_exponent: float = 1.0
    diffusion_step_song_reverse_time: bool = True
    diffusion_step_sampling_min_prob: float = 0.0
    diffusion_step_importance_clip: float | None = None


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    next_emb: jax.Array
    next_state_emb: jax.Array
    next_emb_mask: jax.Array
    reward: jax.Array
    reward_target: jax.Array
    reward_target_mask: jax.Array
    soft_reward: jax.Array
    soft_reward_raw: jax.Array
    discounted_soft_return: jax.Array
    raw_reward: jax.Array
    log_prob: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    fwd_mean: jax.Array
    fwd_scale: jax.Array
    info: dict[str, jax.Array]


class PPOTrainState(nnx.TrainState):
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
    normalization_state: DictNormalizationState | None = None
    critic_normalization_state: DictNormalizationState | None = None
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
                max_time=require(diff_cfg, "diff_steps"),
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
                max_time=require(diff_cfg, "diff_steps"),
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
            train_mode=diff_cfg.get("train_mode", "reparam"),
            dt_schedule=dt_schedule,
            rngs=rngs,
        )

        critic_hidden_dim = require(cfg, "critic_hidden_dim")
        if isinstance(cfg, dict):
            critic_use_normed_actions = bool(cfg.get("critic_use_normed_actions", False))
        else:
            critic_use_normed_actions = bool(
                getattr(cfg, "critic_use_normed_actions", False)
            )
        if not critic_use_normed_actions:
            raise ValueError(
                "DA_MDP_PPO requires `critic_use_normed_actions=True`. "
                "Set `hyperparameters.critic_use_normed_actions: true` in config."
            )
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
            tanh_transform=require(cfg, "tanh_transform"),
            use_temp_lagrangian_mlp=require(cfg, "use_temp_lagrangian_mlp"),
            temp_lagrangian_hidden=require(cfg, "temp_lagrangian_hidden"),
            rngs=rngs,
        )
        if require(cfg, "use_categorical_value"):
            self.critic_module = CategoricalValueNetwork(
                obs_dim=critic_obs_dim,
                normed_action_dim=action_dim,
                hidden_dim=critic_hidden_dim,
                num_bins=require(cfg, "num_bins"),
                vmin=require(cfg, "vmin"),
                vmax=require(cfg, "vmax"),
                num_time_hid=require(score_cfg, "num_time_hid"),
                num_time_out=require(score_cfg, "num_time_out"),
                use_norm=require(cfg, "use_critic_norm"),
                encoder_layers=require(cfg, "num_critic_encoder_layers"),
                head_layers=require(cfg, "num_critic_head_layers"),
                pred_layers=require(cfg, "num_critic_pred_layers"),
                use_simplical_embedding=require(cfg, "use_simplical_embedding"),
                use_skip=require(cfg, "use_critic_skip"),
                use_value_head=not cfg.hl_gauss,
                use_normed_actions=critic_use_normed_actions,
                rngs=rngs,
            )
        else:
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

    def actor_ode_sample_step(self, obs_dict, key) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.actor_module.vmap_ode_sample_next_step(obs_dict, key)


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
        self.num_collection_steps = int(
            cfg.num_steps * self.diffusion_steps * cfg.num_collection_step_factor
        )
        self.num_minibatches = cfg.num_mini_batches * self.diffusion_steps
        self.use_diffusion_importance_sampling = bool(
            getattr(cfg, "importance_sample_diffusion_steps", False)
        )
        self.normalizer = DictNormalizer()
        self.reward_normalizer = Normalizer()
        self.use_reward_normalization = bool(
            getattr(cfg, "normalize_rewards", False)
            or getattr(cfg, "normalize_reward", False)
            or getattr(cfg, "normalize_soft_reward", False)
        )
        self.num_train_steps = cfg.total_time_steps // int(cfg.num_steps * cfg.num_envs * cfg.num_collection_step_factor) 
        self.eval_interval = int(self.num_train_steps // cfg.num_eval)

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
    
    def _make_eval_policy(self, train_state: PPOTrainState) -> Policy:
        normalizer = self.normalizer

        def policy(
            key: PRNGKey, obs: jax.Array, state: struct.PyTreeNode = None
        ) -> tuple[jax.Array, jax.Array]:
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)
            model = nnx.merge(train_state.graphdef, train_state.params)
            action, gen_log_prob, _ = model.actor_sample_step(obs, key)

            return action, dict(log_prob=None, value=None)

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
            num_train_steps = self.num_train_steps
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

            if cfg.anneal_lr and not cfg.adaptive_lr:
                num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
                num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
                lr = optax.linear_schedule(cfg.lr, 1e-6, num_updates)
            else:
                lr = cfg.lr

            def _adam_with_decay(
                lr_val,
                weight_decay: float = 0.0,
                decay_mask=None,
                optim=optax.adam,
                inject_hparams: bool = False,
            ):
                tx = (
                    optax.inject_hyperparams(optim)(learning_rate=lr_val)
                    if inject_hparams
                    else optim(lr_val)
                )
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

            def _select_special_optimizer(name: str, b1: float, b2: float):
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
                    base_adam = functools.partial(
                        optax.adam,
                        b1=b1,
                        b2=b2,
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
                raise ValueError(
                    f"Unknown temp/lagrangian optimizer '{name}', expected 'adam' or 'sgd'."
                )

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
                scale = (128 * 4) / (cfg.diffusion.diff_steps*cfg.num_mini_batches*cfg.num_epochs)
                temp_lagrangian_adam_gamma1 = cfg.temp_lagrangian_adam_gamma1##**scale
                temp_lagrangian_adam_gamma2 = cfg.temp_lagrangian_adam_gamma2#**scale
            else:
                temp_lagrangian_adam_gamma1 = cfg.temp_lagrangian_adam_gamma1
                temp_lagrangian_adam_gamma2 = cfg.temp_lagrangian_adam_gamma2
            temp_lagrangian_adam_gamma1 = max(temp_lagrangian_adam_gamma1, 0.9)
            temp_lagrangian_adam_gamma2 = max(temp_lagrangian_adam_gamma2, 0.999)
            special_optimizer = _select_special_optimizer(
                cfg.temp_lagrangian_optim,
                temp_lagrangian_adam_gamma1,
                temp_lagrangian_adam_gamma2,
            )
            tx_cfg = {
                "default": _adam_with_decay(
                    lr, weight_decay=cfg.weight_decay, inject_hparams=True
                ),
                "no_decay": _adam_with_decay(
                    lr, weight_decay=0.0, inject_hparams=True
                ),
                "temp_lagrangian": _adam_with_decay(
                    cfg.temperature_lr if cfg.temperature_lr is not None else lr,
                    weight_decay=0.0,
                    optim=special_optimizer,
                    inject_hparams=False,
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
            else:
                norm_state = None
                critic_norm_state = None
            if self.use_reward_normalization:
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
                params=param_tree,
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
        diff_steps = self.diffusion_steps

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

            old_fwd_mean, old_fwd_scale = _compute_forward_kernel_moments(
                model.actor_module, obs_norm
            )

            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)
            action, gen_log_prob, dest_log_prob = model.actor_sample_step(
                obs_norm, act_key
            )

            log_ratio = jax.lax.stop_gradient(
                gen_log_prob - dest_log_prob
            )

            action = jax.lax.stop_gradient(action)
            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            raw_reward = reward
            env_time_discount = maybe_env_time_value(
                obs,
                cfg.gamma,
                diff_steps,
                cfg.use_env_time_discounting,
            )
            if cfg.update_entropy_lagrangian:
                temperature = model.actor_module.temperature()
                entropy_scale = temperature
                log_ratio_discount = env_time_discount
            else:
                entropy_scale = cfg.entropy_coef
                log_ratio_discount = maybe_env_time_value(
                    obs,
                    cfg.gamma,
                    diff_steps,
                    cfg.use_env_time_discounting,
                    legacy_value=1.0,
                )
            soft_reward_raw = (
                raw_reward - log_ratio_discount * log_ratio.squeeze() * entropy_scale
            )

            if self.use_reward_normalization:
                episode_done = jnp.logical_or(
                    done > 0, next_env_state.env_state.truncated > 0
                ).astype(jnp.float32)
                discounted_soft_return = soft_reward_raw + (
                    env_time_discount
                    * train_state.discounted_return
                    * (1.0 - episode_done)
                )
                train_state = train_state.replace(
                    discounted_return=discounted_soft_return,
                )
            else:
                discounted_soft_return = soft_reward_raw
            reward = raw_reward
            soft_reward = soft_reward_raw

            if cfg.normalize_env:
                next_critic_obs_for_aux = normalizer.normalize(
                    train_state.critic_normalization_state, next_critic_obs
                )
            else:
                next_critic_obs_for_aux = next_critic_obs
            next_features = (
                jax.lax.stop_gradient(
                    model.critic_module.forward(next_critic_obs_for_aux)[0]
                )
                if cfg.use_categorical_value
                else jnp.zeros((cfg.num_envs, cfg.critic_hidden_dim))
            )
            transition = Transition(
                obs=obs_norm,
                critic_obs=critic_obs_norm,
                action=action,
                next_emb=next_features,
                next_state_emb=next_features,
                next_emb_mask=jnp.ones_like(reward),
                reward=reward,
                reward_target=reward,
                reward_target_mask=jnp.ones_like(reward),
                soft_reward=soft_reward,
                soft_reward_raw=soft_reward_raw,
                discounted_soft_return=discounted_soft_return,
                raw_reward=raw_reward,
                log_prob=gen_log_prob,
                value=model.critic(critic_obs_norm),
                done=done,
                truncated=next_env_state.env_state.truncated,
                fwd_mean=old_fwd_mean,
                fwd_scale=old_fwd_scale,
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
        diff_steps = self.diffusion_steps

        if cfg.normalize_env:
            last_critic_obs = normalizer.normalize(
                train_state.critic_normalization_state, train_state.last_critic_obs
            )
        else:
            last_critic_obs = train_state.last_critic_obs
        last_value = model.critic(last_critic_obs)

        if self.use_reward_normalization:
            rollout_discounted_soft_returns = batch.discounted_soft_return.reshape(-1)
            rollout_reward_scale = jnp.sqrt(
                jnp.var(rollout_discounted_soft_returns)
                + jnp.asarray(
                    cfg.reward_norm_epsilon,
                    dtype=rollout_discounted_soft_returns.dtype,
                )
            )
            reward_norm_state = self.reward_normalizer.update(
                train_state.reward_normalization_state,
                rollout_discounted_soft_returns,
            )
            train_state = train_state.replace(reward_normalization_state=reward_norm_state)
            reward_norm = batch.raw_reward / rollout_reward_scale
            soft_reward_norm = batch.soft_reward_raw / rollout_reward_scale
            batch = batch.replace(
                reward=reward_norm,
                reward_target=reward_norm,
                reward_target_mask=jnp.ones_like(reward_norm),
                soft_reward=soft_reward_norm,
            )
        else:
            rollout_reward_scale = jnp.asarray(1.0, dtype=batch.raw_reward.dtype)

        if cfg.use_env_time_discounting:
            discount, trace_decay = maybe_env_time_discount_lambda(
                batch.obs,
                cfg.gamma,
                cfg.lmbda,
                diff_steps,
                True,
            )

            def compute_advantage(carry, inputs):
                transition, gamma_t, lmbda_t = inputs
                return compute_gae_step(gamma_t, lmbda_t, carry, transition)

            advantage_scan_inputs = (batch, discount, trace_decay)
        else:
            def compute_advantage(carry, transition):
                return compute_gae_step(cfg.gamma, cfg.lmbda, carry, transition)

            advantage_scan_inputs = batch

        _, value_advantages = jax.lax.scan(
            compute_advantage,
            (jnp.zeros_like(last_value), last_value),
            advantage_scan_inputs,
            reverse=True,
        )
        target_values = value_advantages + batch.value
        actor_advantages = value_advantages * rollout_reward_scale
        target_vals_flat = target_values.reshape(-1)
        target_vals_finite = jnp.nan_to_num(
            target_vals_flat,
            nan=0.0,
            posinf=cfg.vmax,
            neginf=cfg.vmin,
        )
        target_value_mean = jnp.mean(target_vals_finite)
        target_value_min = jnp.min(target_vals_finite)
        target_value_max = jnp.max(target_vals_finite)
        # Build rollout-aligned aux targets (shifted embedding + final-step reward labels).
        next_state_emb, next_emb_mask, reward_target, reward_target_mask = (
            build_rollout_aux_targets(
                batch.next_emb,
                batch.reward,
                batch.done,
                batch.truncated,
                batch.obs["diff_time_step"][..., 0],
                self.cfg.diffusion.diff_steps,
                mask_next_state_on_episode_end=True,
            )
        )
        batch = batch.replace(
            next_state_emb=next_state_emb,
            next_emb_mask=next_emb_mask,
            reward_target=reward_target,
            reward_target_mask=reward_target_mask,
        )

        data = (batch, actor_advantages, value_advantages, target_values)
        data = jax.tree.map(
            lambda x: x.reshape(
                (math.floor(self.num_collection_steps * cfg.num_envs), *x.shape[2:])
            ),
            data,
        )
        total_size = math.floor(self.num_collection_steps * cfg.num_envs)
        if self.use_diffusion_importance_sampling:
            step_indices = data[0].obs["diff_time_step"][..., 0]
            sampling_mode = str(
                getattr(self.cfg, "diffusion_step_sampling_mode", "power")
            ).lower()
            beta_per_sample = None
            if sampling_mode == "song":
                diff_model = model.actor_module.diffusion_model
                step_indices_for_coeff = step_indices.reshape((-1, 1))
                _, eta, _ = diff_model.diffusion_coeff_fn(
                    step_indices_for_coeff, data[0].obs
                )
                beta_per_sample = eta
                if beta_per_sample.ndim > 1:
                    beta_per_sample = jnp.mean(beta_per_sample, axis=-1)
                beta_per_sample = beta_per_sample.reshape(-1).astype(jnp.float32)
            per_sample_probs, per_sample_importance_ratio, step_probs = (
                prepare_diffusion_importance_sampling(
                    step_indices,
                    self.cfg.diffusion.diff_steps,
                    sampling_mode=sampling_mode,
                    exponent=self.cfg.diffusion_step_sampling_exponent,
                    beta_per_sample=beta_per_sample,
                    song_reverse_time=bool(
                        getattr(self.cfg, "diffusion_step_song_reverse_time", True)
                    ),
                    min_step_prob=self.cfg.diffusion_step_sampling_min_prob,
                    importance_clip=self.cfg.diffusion_step_importance_clip,
                )
            )
        else:
            per_sample_probs = jnp.full(
                (total_size,),
                1.0 / max(total_size, 1),
                dtype=jnp.float32,
            )
            per_sample_importance_ratio = jnp.ones((total_size,), dtype=jnp.float32)
            step_probs = jnp.full(
                (self.cfg.diffusion.diff_steps,),
                1.0 / max(self.cfg.diffusion.diff_steps, 1),
                dtype=jnp.float32,
            )

        def update(train_state, key):
            def minibatch_update(carry, scan_inputs):
                idx, train_state = carry
                indices, step_key, importance_ratio = scan_inputs
                minibatch, actor_advantages, value_advantages, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def loss_fn(params):
                    model = nnx.merge(train_state.graphdef, params)

                    gen_log_prob, dest_log_prob = model.actor_log_prob_step(minibatch.obs, minibatch.action)
                    log_ratio = gen_log_prob - dest_log_prob
                    fwd_mean, fwd_scale = _compute_forward_kernel_moments(
                        model.actor_module, minibatch.obs
                    )
                    kl_mean = jnp.mean(
                        _rsl_gaussian_kl(
                            old_mu=minibatch.fwd_mean,
                            old_sigma=minibatch.fwd_scale,
                            mu=fwd_mean,
                            sigma=fwd_scale,
                        )
                    )
                    if cfg.use_categorical_value:
                        if cfg.hl_gauss:
                            critic_pred = model.critic_module.critic_cat(
                            minibatch.critic_obs
                            ).squeeze()
                            target_cat = jax.vmap(
                                utils.hl_gauss, in_axes=(0, None, None, None)
                            )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                            critic_update_loss = optax.softmax_cross_entropy(
                                critic_pred, target_cat
                            )
                            _, pred, pred_rew, pred_next_diff_state, value = (
                                model.critic_module.forward(minibatch.critic_obs)
                            )
                        else:
                            _, pred, pred_rew, pred_next_diff_state, value = (
                                model.critic_module.forward_value(minibatch.critic_obs)
                            )
                            critic_update_loss = optax.squared_error(
                                value.reshape(-1, 1),
                                target_values.reshape(-1, 1),
                            )
                        next_state_mask = minibatch.next_emb_mask.reshape(-1, 1).astype(
                            pred.dtype
                        )
                        aux_loss = (
                            (1.0 - minibatch.truncated.reshape(-1, 1))
                            * next_state_mask
                            * optax.squared_error(pred, minibatch.next_state_emb)
                        )
                        use_final_step_reward_target = bool(
                            getattr(cfg, "use_final_step_reward_target", False)
                        )
                        if use_final_step_reward_target:
                            reward_target = getattr(
                                minibatch, "reward_target", minibatch.reward
                            ).reshape(-1, 1)
                            reward_target_mask = getattr(
                                minibatch,
                                "reward_target_mask",
                                jnp.ones_like(minibatch.reward),
                            ).reshape(-1, 1).astype(pred_rew.dtype)
                        else:
                            reward_target = minibatch.reward.reshape(-1, 1)
                            reward_target_mask = jnp.ones_like(
                                reward_target, dtype=pred_rew.dtype
                            )
                        aux_rew_loss = (
                            1.0 - minibatch.truncated.reshape(-1, 1)
                        ) * reward_target_mask * optax.squared_error(
                            pred_rew, reward_target
                        )

                        use_normed_actions = bool(
                            getattr(cfg, "critic_use_normed_actions", True)
                        )
                        if use_normed_actions:
                            diff_steps = jnp.asarray(
                                cfg.diffusion.diff_steps - 1,
                                dtype=minibatch.obs["diff_time_step"].dtype,
                            )
                            is_last_step = (
                                minibatch.obs["diff_time_step"][..., 0] == diff_steps
                            ).reshape(-1, 1)
                            aux_weight = is_last_step.astype(aux_loss.dtype)
                        else:
                            step_index = minibatch.obs["diff_time_step"][..., 0]
                            max_step = jnp.maximum(
                                jnp.asarray(
                                    cfg.diffusion.diff_steps - 1,
                                    dtype=step_index.dtype,
                                ),
                                1.0,
                            )
                            min_weight = 1.0 / jnp.maximum(
                                jnp.asarray(
                                    cfg.diffusion.diff_steps, dtype=step_index.dtype
                                ),
                                1.0,
                            )
                            step_progress = jnp.clip(step_index / max_step, 0.0, 1.0)
                            aux_weight = (
                                min_weight + (1.0 - min_weight) * step_progress
                            ).reshape(-1, 1).astype(aux_loss.dtype)
                            # aux_weight = cfg.diffusion.diff_steps * aux_weight**2
                            aux_weight = (
                                cfg.diffusion.diff_steps
                                * (aux_weight == 1.0)
                            )

                        masked_aux_terms = jnp.concatenate(
                            [aux_loss, aux_rew_loss], axis=-1
                        )
                        masked_aux_loss = jnp.mean(
                            (1 - minibatch.done.reshape(-1, 1))
                            * aux_weight
                            * masked_aux_terms,
                            axis=-1,
                        )
                        if use_normed_actions:
                            aux_next_diff_loss = (
                                1.0 - minibatch.truncated.reshape(-1, 1)
                            ) * optax.squared_error(
                                pred_next_diff_state, minibatch.next_emb
                            )
                            aux_next_diff_loss = jnp.mean(
                                (1 - minibatch.done.reshape(-1, 1))
                                * aux_next_diff_loss,
                                axis=-1,
                            )
                            alpha = cfg.aux_loss_alpha
                            aux_loss = (
                                alpha
                                * jnp.sum(masked_aux_loss)
                                / jnp.maximum(jnp.sum(aux_weight), 1.0)
                                + (1 - alpha) * jnp.mean(aux_next_diff_loss)
                            )
                        else:
                            alpha = 1.0
                            aux_loss = jnp.mean(masked_aux_loss) 

                        value_loss = _weighted_batch_mean(
                            (1.0 - minibatch.truncated) * (critic_update_loss)
                            + cfg.aux_loss_mult * aux_loss
                            ,
                            importance_ratio,
                        )
                    else:
                        value = model.critic(minibatch.critic_obs)
                        value_pred_clipped = minibatch.value + (
                            value - minibatch.value
                        ).clip(-cfg.clip_ratio, cfg.clip_ratio)
                        value_error = jnp.square(value - target_values)
                        value_error_clipped = jnp.square(value_pred_clipped - target_values)
                        value_loss = 0.5 * _weighted_batch_mean(
                            (1.0 - minibatch.truncated)
                            * jnp.maximum(value_error, value_error_clipped),
                            importance_ratio,
                        )
                        critic_loss = value_loss

                    ratio = jnp.exp(gen_log_prob - minibatch.log_prob)
                    lagrangian = model.actor_module.lagrangian()
                    checkify.check(
                        jnp.allclose(ratio, 1.0) | (idx != 1),
                        debug=True,
                        msg="Ratio not equal to 1 on first iteration: {r}",
                        r=ratio,
                    )

                    adv_base = actor_advantages

                    if cfg.update_entropy_lagrangian:
                        entropy_scale = jax.lax.stop_gradient(model.actor_module.temperature())
                    else:
                        entropy_scale = self.cfg.entropy_coef

                    unnormed_advantages = (
                        adv_base
                        - minibatch.soft_reward_raw
                        + minibatch.raw_reward
                        - log_ratio * entropy_scale
                    )
                    if (cfg.normalize_advantages):
                        mean = jax.lax.stop_gradient(jnp.mean(unnormed_advantages))
                        sdt = jax.lax.stop_gradient(jnp.std(unnormed_advantages)) + 1e-8
                        adv_base = (unnormed_advantages - mean) / (
                            sdt
                        )
                    else:
                        mean = jax.lax.stop_gradient(jnp.mean(unnormed_advantages))
                        sdt = 1
                        adv_base = (unnormed_advantages - mean) / (
                            sdt
                        )

                    adv_base = jax.lax.stop_gradient(adv_base)  ### when forward process is learned things have to be adapted

                    valid_mask = 1.0 - minibatch.truncated
                    clipped = jnp.logical_or(
                        ratio > 1 + cfg.clip_ratio, ratio < 1 - cfg.clip_ratio
                    )
                    clip_fraction = _weighted_batch_mean(
                        valid_mask * clipped.astype(jnp.float32), importance_ratio
                    ) / (
                        _weighted_batch_mean(valid_mask, importance_ratio) + 1e-8
                    )
                    lagrangian_loss = jnp.array(0.0)
                    
                    actor_loss1 = ratio * adv_base
                    actor_loss2 = (
                        jnp.clip(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                        * adv_base
                    )
                    actor_loss = _weighted_batch_mean(
                        -(valid_mask * jnp.minimum(actor_loss1, actor_loss2)),
                        importance_ratio,
                    )
                    do_update = ( (actor_loss1 < actor_loss2))| ((ratio >= 1 - cfg.clip_ratio) & (ratio <= 1 + cfg.clip_ratio))
                    do_update = valid_mask.astype(bool) * do_update

                    # grad_not_tracked = (actor_loss1 > actor_loss2) * ((ratio < 1 - cfg.clip_ratio) | (ratio > 1 + cfg.clip_ratio)) 
                    # grad_tracked = (1.0 - grad_not_tracked)* valid_mask.astype(bool)

                    # ### check element wise if do_update == grad_tracked
                    # jax.debug.print("do_update: {}, grad_tracked: {}", do_update, grad_tracked)
                    # ### chekck if all elements are the same
                    # jax.debug.print("All equal: {}", jnp.all(do_update == grad_tracked))

                    scaled_dest_log_prob = dest_log_prob/sdt
                    stop_grad_ratio = jax.lax.stop_gradient(ratio)
                    masked_scaled_dest_log_prob = jnp.where(do_update, scaled_dest_log_prob, jax.lax.stop_gradient(scaled_dest_log_prob))
                    dest_loss = _weighted_batch_mean(
                        -stop_grad_ratio * masked_scaled_dest_log_prob * entropy_scale,
                        importance_ratio,
                    )
                    actor_loss += dest_loss


                    loss = (
                        actor_loss
                        + cfg.value_coef * value_loss
                    )
                    if cfg.update_entropy_lagrangian:
                        entropy = -self.diffusion_steps * _weighted_batch_mean_axis0(
                            log_ratio, importance_ratio
                        )
                        target_entropy = self.action_size_target + entropy
                        target_entropy_loss = (
                            model.actor_module.temperature()
                            * jax.lax.stop_gradient(target_entropy) - jax.lax.stop_gradient(model.actor_module.temperature())
                            * target_entropy
                        )
                        target_entropy_loss = jnp.mean(target_entropy_loss)
                        loss += target_entropy_loss
                    else:
                        entropy = -self.diffusion_steps * _weighted_batch_mean_axis0(
                            log_ratio, importance_ratio
                        )
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
                        entropy_loss=target_entropy_loss,
                        entropy=entropy,
                        target_entropy=self.action_size_target,
                        temp=entropy_scale,
                        kl=kl,
                        kl_mean=kl_mean,
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        loss=loss,
                        mean_value=_weighted_batch_mean(value, importance_ratio),
                        mean_log_prob=_weighted_batch_mean(gen_log_prob, importance_ratio),
                        mean_advantages=_weighted_batch_mean(adv_base, importance_ratio),
                        mean_action=_weighted_batch_mean(minibatch.action, importance_ratio),
                        abs_batch_action=_weighted_batch_mean(
                            jnp.abs(minibatch.action), importance_ratio
                        ),
                        abs_pred_action=_weighted_batch_mean(
                            jnp.abs(minibatch.action), importance_ratio
                        ),
                        reward_mean=_weighted_batch_mean(
                            minibatch.raw_reward, importance_ratio
                        )
                        * self.diffusion_steps,
                        target_value_mean=target_value_mean,
                        target_value_min=target_value_min,
                        target_value_max=target_value_max,
                        clip_ratio=clip_fraction,
                        critic_update_loss=(
                            _weighted_batch_mean(critic_update_loss, importance_ratio)
                            if cfg.use_categorical_value
                            else jnp.array(0.0)
                        ),
                        aux_loss=(
                            jnp.mean(aux_loss)
                            if cfg.use_categorical_value
                            else jnp.array(0.0)
                        ),
                        rew_aux_loss=(
                            _weighted_batch_mean(
                                aux_rew_loss
                                * aux_weight.astype(aux_rew_loss.dtype),
                                importance_ratio,
                            )
                            if cfg.use_categorical_value
                            else jnp.array(0.0)
                        ),
                    )

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                output, grads = grad_fn(train_state.params)

                flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
                global_grad_norm = jnp.linalg.norm(flat_grads)

                metrics = output[1]
                safe_rollout_reward_scale = jnp.maximum(
                    rollout_reward_scale, jnp.asarray(1e-8, dtype=rollout_reward_scale.dtype)
                )
                metrics["advantages"] = _weighted_batch_mean(
                    value_advantages, importance_ratio
                )
                metrics["actor_advantages"] = _weighted_batch_mean(
                    actor_advantages, importance_ratio
                )
                metrics["rollout_reward_scale"] = rollout_reward_scale
                metrics["advantage_scale_error"] = _weighted_batch_mean(
                    jnp.abs(
                        actor_advantages / safe_rollout_reward_scale
                        - value_advantages
                    ),
                    importance_ratio,
                )
                metrics["global_grad_norm"] = global_grad_norm
                metrics["importance_ratio_mean"] = jnp.mean(importance_ratio)
                metrics["importance_ratio_max"] = jnp.max(importance_ratio)
                metrics["importance_ratio_min"] = jnp.min(importance_ratio)
                new_lr = _update_adaptive_lr(
                    cfg, train_state.learning_rate, metrics["kl_mean"]
                )
                if cfg.adaptive_lr:
                    train_state = train_state.replace(
                        opt_state=_replace_learning_rate_in_opt_state(
                            train_state.opt_state,
                            new_lr,
                            transform_names={"default", "no_decay"},
                        )
                    )
                train_state = train_state.replace(learning_rate=new_lr)
                train_state = train_state.apply_gradients(grads)
                metrics["learning_rate"] = train_state.learning_rate
                return (idx + 1, train_state), metrics

            key, shuffle_key = jax.random.split(key)
            mini_batch_size = total_size // self.num_minibatches
            minibatch_idxs, minibatch_importance_ratio, minibatch_keys = (
                sample_minibatch_indices(
                    shuffle_key,
                    total_size=total_size,
                    num_minibatches=self.num_minibatches,
                    mini_batch_size=mini_batch_size,
                    use_importance_sampling=self.use_diffusion_importance_sampling,
                    per_sample_probs=per_sample_probs,
                    per_sample_importance_ratio=per_sample_importance_ratio,
                )
            )

            train_state, metrics = jax.lax.scan(
                minibatch_update,
                train_state,
                (minibatch_idxs, minibatch_keys, minibatch_importance_ratio),
            )
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            metrics["step_sampling_probs"] = step_probs
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
            metrics.update(
                _dict_norm_stats("actor", state.normalization_state)
            )
            metrics.update(
                _dict_norm_stats("value", state.critic_normalization_state)
            )
            if state.reward_normalization_state is not None:
                metrics["reward_norm_count"] = jnp.asarray(
                    state.reward_normalization_state.count, dtype=jnp.float32
                )
                metrics["reward_norm_std"] = jnp.sqrt(
                    jnp.mean(state.reward_normalization_state.var)
                )
                metrics["reward_norm_mean_abs"] = jnp.mean(
                    jnp.abs(state.reward_normalization_state.mean)
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
        policy = self._make_eval_policy(train_state)
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
        num_train_steps = self.num_train_steps
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
        for metric_key in (
            "train/step_sampling_probs",
            "train/importance_ratio_mean",
            "train/importance_ratio_max",
            "train/importance_ratio_min",
        ):
            metrics.pop(metric_key, None)
        kl_mean = metrics.pop("train/kl_mean", None)
        learning_rate = metrics.pop("train/learning_rate", None)
        normalization_metrics = {}
        for metric_key in list(metrics.keys()):
            if (
                metric_key.startswith("train/reward_norm_")
                or metric_key.startswith("train/actor_obs_norm_")
                or metric_key.startswith("train/actor_action_norm_")
                or metric_key.startswith("train/value_obs_norm_")
                or metric_key.startswith("train/value_action_norm_")
            ):
                normalization_metrics[
                    f"normalization/{metric_key.split('train/', 1)[1]}"
                ] = jnp.mean(metrics.pop(metric_key))
        advantages_hist = None
        if advantages is not None:
            adv_np = np.asarray(jax.device_get(advantages))
            finite_mask = np.isfinite(adv_np)
            if finite_mask.any():
                finite_adv = adv_np[finite_mask]
                if np.ptp(finite_adv) > 0:
                    advantages_hist = wandb.Histogram(finite_adv)
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
            **normalization_metrics,
        }
        if kl_mean is not None:
            log_data["learning_rate/kl_mean"] = jnp.mean(kl_mean)
        if learning_rate is not None:
            log_data["learning_rate/lr"] = jnp.mean(learning_rate)
        if advantages_hist is not None:
            log_data["train/advantages"] = advantages_hist
        wandb.log(_sectioned_wandb_log(log_data), step=state.time_steps[0])

    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(cfg.env.name)
        raise ValueError("Brax not supported for DiffPPO.")
    elif cfg.env.type == "mjx":
        env_config = OmegaConf.select(cfg, "env.config")
        if env_config is not None:
            env_config = OmegaConf.to_container(env_config, resolve=True)
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            config=env_config,
        )
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
        run_config = OmegaConf.to_container(cfg)
        run_config["method_name"] = "DA_MDP_PPO"
        wandb.init(
            mode=cfg.wandb.mode,
            project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}",
            entity=cfg.wandb.entity,
            tags=[cfg.name, cfg.env.name, cfg.env.type, *cfg.tags],
            config=run_config,
            name=f"ppo-{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )
        start = time.perf_counter()
        state, metrics = jax.jit(train_fn, static_argnums=(1,))(train_key, trainer.cfg)
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Export final weights into repo_root/saved_models with a descriptive filename.
        try:
            final_metrics = _take_last_metrics(metrics)
            method_name = "DA_MDP_PPO"
            env_name = str(cfg.env.name)
            train_mode = _resolve_train_mode(cfg)
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
                "params": _to_numpy_tree(state.params),
                "time_steps": _to_numpy_tree(state.time_steps),
                "iteration": _to_numpy_tree(state.iteration),
                "normalization_state": _to_numpy_tree(state.normalization_state)
                if bool(getattr(cfg.hyperparameters, "normalize_env", False))
                else None,
                "critic_normalization_state": _to_numpy_tree(state.critic_normalization_state)
                if bool(getattr(cfg.hyperparameters, "normalize_env", False))
                else None,
                "reward_normalization_state": _to_numpy_tree(state.reward_normalization_state)
                if bool(getattr(cfg.hyperparameters, "normalize_rewards", False))
                or bool(getattr(cfg.hyperparameters, "normalize_reward", False))
                or bool(getattr(cfg.hyperparameters, "normalize_soft_reward", False))
                else None,
                "last_env_state": _to_numpy_tree(state.last_env_state),
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
        wandb.init(project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}")
        run_cfg = OmegaConf.to_container(cfg)
        for k, v in dict(wandb.config).items():
            run_cfg["hyperparameters"][k] = v
        wandb.config.update({"method_name": "DA_MDP_PPO"}, allow_val_change=True)
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
        project=f"{cfg.wandb.project}{getattr(cfg.wandb, 'project_suffix', '')}",
        entity=cfg.wandb.entity,
    )
    wandb.agent(sweep_id, function=train_agent, count=cfg.tune.num_runs)


@hydra.main(version_base=None, config_path="../../config/DA_MDP_PPO", config_name="default")
def main(cfg: DictConfig):
    diffppo_overrides = _extract_hyperparameter_overrides(cfg, "overrides")
    diffppo_features = _extract_hyperparameter_overrides(cfg, "features")
    experiment_overrides = _extract_hyperparameter_overrides(cfg, "experiment_overrides")
    cfg.hyperparameters = OmegaConf.merge(
        cfg.hyperparameters, diffppo_overrides, diffppo_features, experiment_overrides
    )
    cfg = _reapply_cli_hyperparameter_overrides(cfg)
    legacy_norm_rewards = bool(getattr(cfg.hyperparameters, "normalize_reward", False)) or bool(
        getattr(cfg.hyperparameters, "normalize_soft_reward", False)
    )
    cfg.hyperparameters.normalize_rewards = bool(
        getattr(cfg.hyperparameters, "normalize_rewards", False) or legacy_norm_rewards
    )
    if cfg.tune:
        tune(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
