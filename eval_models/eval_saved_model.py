import argparse
import logging
import os
import pickle
import sys
import importlib
import ast
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    DiffNormalizeVec,
    LogWrapper,
    MjxDiffEnvWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.networks.jax_models import SACActorNetworks

logging.basicConfig(level=logging.INFO)


def _artifacts_dir() -> str:
    path = os.path.join(_REPO_ROOT, "artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_render_out(
    *,
    method_name: str,
    checkpoint_path: str,
    out_path: str | None,
    diffusion_sampler: str | None = None,
) -> str:
    # Always write renders into repo_root/artifacts (ignore any provided directory).
    artifacts_dir = _artifacts_dir()
    method_lower = str(method_name).lower()
    if "dime" in method_lower:
        method_tag = "dime"
    elif "dmerl" in method_lower:
        method_tag = "DMERL"
    else:
        method_tag = "reppo"
    sampler_lower = str(diffusion_sampler or "").lower()
    sampler_tag = sampler_lower if sampler_lower in ("sde", "ode") else None
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if out_path is None:
        # Most checkpoints already encode the method in their filename. Only prefix when missing.
        ckpt_lower = ckpt_base.lower()
        if ("dmerl" in ckpt_lower) or ("dime" in ckpt_lower) or ckpt_lower.startswith(("reppo__", "dime__")):
            stub = ckpt_base
        else:
            stub = f"{method_tag}__{ckpt_base}"
        if sampler_tag is None:
            filename = f"{stub}__turning_double_well_traj.gif"
        else:
            filename = f"{stub}__{sampler_tag}__turning_double_well_traj.gif"
        return os.path.join(artifacts_dir, filename)
    base = os.path.basename(out_path)
    if not base:
        base = f"{method_tag}__{ckpt_base}__turning_double_well_traj.gif"
    if not base.lower().endswith(".gif"):
        base = base + ".gif"
    # Ensure the method is encoded in the name even when the user passes a custom basename.
    if not base.lower().startswith(method_tag.lower() + "__"):
        base = f"{method_tag}__{base}"
    if sampler_tag is not None:
        prefix = method_tag.lower() + "__"
        rest = base[len(method_tag) + 2 :] if base.lower().startswith(prefix) else base
        if not rest.lower().startswith(sampler_tag + "__"):
            base = f"{method_tag}__{sampler_tag}__{rest}"
    return os.path.join(artifacts_dir, base)


def _resolve_tdw_analysis_out(*, checkpoint_path: str, out_path: str | None) -> str:
    artifacts_dir = _artifacts_dir()
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if out_path is None:
        filename = f"{ckpt_base}__turning_double_well_action_analysis.png"
        return os.path.join(artifacts_dir, filename)
    base = os.path.basename(out_path)
    if not base:
        base = f"{ckpt_base}__turning_double_well_action_analysis.png"
    if not base.lower().endswith(".png"):
        base = base + ".png"
    return os.path.join(artifacts_dir, base)


def _tdw_build_env_for_analysis(cfg):
    # Use the checkpoint config, but force the discrete-initial-heading assumption.
    env_config = OmegaConf.select(cfg, "env.config")
    env_kwargs: dict[str, Any] = {}
    if env_config is not None:
        env_kwargs = OmegaConf.to_container(env_config, resolve=True) or {}
    env_kwargs = dict(env_kwargs)
    env_kwargs["randomize_initial_heading"] = True
    env_kwargs["snap_action_to_optimal"] = True
    env_kwargs.setdefault("horizon", int(OmegaConf.select(cfg, "env.max_episode_steps") or 200))
    from src.env_utils.turning_double_well_env import TurningDoubleWellEnv

    return TurningDoubleWellEnv(**env_kwargs)


def _tdw_discrete_starting_angles(env) -> jax.Array:
    # Angles are returned wrapped to [-pi, pi] for stable downstream computations.
    return env._wrap_angle(env._snapped_initial_heading_support_radians)


def _tdw_obs_from_angle(env, angle_radians: jax.Array) -> jax.Array:
    pos = jnp.zeros((2,), dtype=jnp.float32)
    direction = env._angle_to_direction(angle_radians)
    return env._observation(pos, direction)


def _normalize_obs(raw_obs: jax.Array, norm_state, *, critic: bool = False) -> jax.Array:
    if norm_state is None:
        return raw_obs
    mean = getattr(norm_state, "critic_mean", None) if critic else getattr(norm_state, "mean", None)
    var = getattr(norm_state, "critic_var", None) if critic else getattr(norm_state, "var", None)
    if mean is None or var is None:
        return raw_obs
    return (raw_obs - mean) / jnp.sqrt(var + 1e-2)


def _tdw_action_analysis_sac(
    *,
    cfg,
    checkpoint_path: str,
    actor_model,
    critic_model,
    norm_state,
    out_path: str | None,
    num_samples: int,
    num_grid: int,
    num_bins: int,
    seed: int,
    max_states: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))

    fig, axes = plt.subplots(
        nrows=len(angles_np),
        ncols=2,
        figsize=(12, max(2.2, 2.2 * len(angles_np))),
        squeeze=False,
    )

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for row_idx, (theta, key) in enumerate(zip(angles_np, keys)):
        theta = jnp.asarray(theta, dtype=jnp.float32)
        raw_obs = _tdw_obs_from_angle(env, theta)
        obs = _normalize_obs(raw_obs, norm_state, critic=False)
        critic_obs = _normalize_obs(raw_obs, norm_state, critic=True)

        # Sample actions from the actor distribution for this state.
        pi = actor_model.actor(obs)
        sampled = pi.sample(seed=key, sample_shape=(int(num_samples),))
        sampled_np = np.asarray(jax.device_get(sampled)).reshape(-1)

        # Reward curve over actions for this starting angle.
        next_angle = env._wrap_angle(theta + grid_actions * env.max_turn_radians)
        reward_curve = env.reward_from_angle(next_angle)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        # Q curve over actions for this starting angle (if critic is available).
        q_curve_np = None
        if critic_model is not None:
            critic_obs_grid = jnp.broadcast_to(critic_obs, (grid_actions.shape[0],) + critic_obs.shape)
            action_grid = grid_actions.reshape(-1, 1)
            q_vals = critic_model.critic(critic_obs_grid, action_grid).squeeze(-1)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        ax_hist = axes[row_idx, 0]
        ax_q = axes[row_idx, 1]

        ax_hist.hist(sampled_np, bins=int(num_bins), range=(-1.0, 1.0), density=True, alpha=0.6)
        ax_hist.set_xlim(-1.0, 1.0)
        ax_hist.set_ylabel("density")
        theta_deg = float((np.rad2deg(float(theta)) + 360.0) % 360.0)
        ax_hist.set_title(f"θ0={theta_deg:.1f}°: action samples + reward")
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.0)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward")

        if q_curve_np is not None:
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.0)
            ax_q.set_title("Q(s,a) over action grid")
            ax_q.set_xlim(-1.0, 1.0)
            ax_q.set_xlabel("action")
            ax_q.set_ylabel("Q")
        else:
            ax_q.axis("off")
            ax_q.text(0.5, 0.5, "No critic in checkpoint", ha="center", va="center")

    fig.tight_layout()
    out_path = _resolve_tdw_analysis_out(checkpoint_path=checkpoint_path, out_path=out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved TurningDoubleWell action analysis to %s", out_path)


def _tdw_action_analysis_dmerl(
    *,
    cfg,
    checkpoint_path: str,
    actor_model,
    critic_model,
    norm_state,
    out_path: str | None,
    num_samples: int,
    num_grid: int,
    num_bins: int,
    seed: int,
    max_states: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    diff_steps = int(getattr(actor_model, "diff_steps", 1))
    if diff_steps <= 0:
        diff_steps = 1
    last_step = diff_steps - 1

    # Normalization stats (DiffNormalizeVecObsEnvState when normalize_env=true).
    obs_mean = getattr(norm_state, "mean", None)
    obs_var = getattr(norm_state, "var", None)
    act_mean = getattr(norm_state, "action_mean", None)
    act_var = getattr(norm_state, "action_var", None)

    def _norm_obs(x):
        if obs_mean is None or obs_var is None:
            return x
        return (x - obs_mean) / jnp.sqrt(obs_var + 1e-2)

    def _norm_act(x):
        if act_mean is None or act_var is None:
            return x
        return (x - act_mean) / jnp.sqrt(act_var + 1e-2)

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))

    fig, axes = plt.subplots(
        nrows=len(angles_np),
        ncols=2,
        figsize=(12, max(2.2, 2.2 * len(angles_np))),
        squeeze=False,
    )

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for row_idx, (theta, key) in enumerate(zip(angles_np, keys)):
        theta = jnp.asarray(theta, dtype=jnp.float32)
        raw_obs = _tdw_obs_from_angle(env, theta)
        obs_norm = _norm_obs(raw_obs)

        # Histogram: sample the *final* env action by running the diffusion chain.
        key, prior_key = jax.random.split(key)
        current_x = actor_model.diffusion_model.prior_sampler(prior_key, int(num_samples))
        current_x = jnp.asarray(current_x, dtype=jnp.float32)

        chain_key = key
        for step_idx in range(diff_steps):
            chain_key, step_key = jax.random.split(chain_key)
            obs_dict = {
                "orig_obs": jnp.broadcast_to(obs_norm, (current_x.shape[0],) + obs_norm.shape),
                "orig_actions": current_x,
                "normed_actions": _norm_act(current_x),
                "diff_time_step": jnp.full((current_x.shape[0], 1), step_idx, dtype=jnp.int32),
            }
            current_x, *_ = actor_model.vmap_sample_next_step(obs_dict, step_key)

        # Action applied to TurningDoubleWellEnv is tanh(raw) (then clipped by env wrapper in training).
        sampled_actions = jnp.tanh(current_x)
        sampled_np = np.asarray(jax.device_get(sampled_actions)).reshape(-1)

        # Reward curve over env actions for this starting angle.
        next_angle = env._wrap_angle(theta + grid_actions * env.max_turn_radians)
        reward_curve = env.reward_from_angle(next_angle)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        # Q curve: evaluate critic at last diffusion step with a neutral "previous action" (zeros).
        q_curve_np = None
        if critic_model is not None:
            prev_x = jnp.zeros((1,), dtype=jnp.float32)
            obs_dict_last = {
                "orig_obs": obs_norm[None, :],
                "orig_actions": prev_x[None, :],
                "normed_actions": _norm_act(prev_x)[None, :],
                "diff_time_step": jnp.full((1, 1), last_step, dtype=jnp.int32),
            }
            critic_in = jnp.concatenate([obs_dict_last["orig_obs"], obs_dict_last["normed_actions"]], axis=-1)
            critic_in = jnp.broadcast_to(critic_in, (grid_actions.shape[0],) + critic_in.shape[1:])
            raw_action = jnp.arctanh(jnp.clip(grid_actions, -0.999, 0.999)).reshape(-1, 1)
            q_vals = critic_model.critic(critic_in, raw_action).squeeze(-1)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        ax_hist = axes[row_idx, 0]
        ax_q = axes[row_idx, 1]

        ax_hist.hist(sampled_np, bins=int(num_bins), range=(-1.0, 1.0), density=True, alpha=0.6)
        ax_hist.set_xlim(-1.0, 1.0)
        ax_hist.set_ylabel("density")
        theta_deg = float((np.rad2deg(float(theta)) + 360.0) % 360.0)
        ax_hist.set_title(f"θ0={theta_deg:.1f}°: DMERL final-step actions + reward")
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.0)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward")

        if q_curve_np is not None:
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.0)
            ax_q.set_title(f"Q(s,a) at diffusion step {last_step}")
            ax_q.set_xlim(-1.0, 1.0)
            ax_q.set_xlabel("env action (tanh(raw))")
            ax_q.set_ylabel("Q")
        else:
            ax_q.axis("off")
            ax_q.text(0.5, 0.5, "No critic available", ha="center", va="center")

    fig.tight_layout()
    out_path = _resolve_tdw_analysis_out(checkpoint_path=checkpoint_path, out_path=out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved TurningDoubleWell action analysis to %s", out_path)


def _tdw_action_analysis_dime(
    *,
    cfg,
    checkpoint_path: str,
    actor_model,
    critic_model,
    norm_state,
    out_path: str | None,
    num_samples: int,
    num_grid: int,
    num_bins: int,
    seed: int,
    max_states: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))

    fig, axes = plt.subplots(
        nrows=len(angles_np),
        ncols=2,
        figsize=(12, max(2.2, 2.2 * len(angles_np))),
        squeeze=False,
    )

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for row_idx, (theta, key) in enumerate(zip(angles_np, keys)):
        theta = jnp.asarray(theta, dtype=jnp.float32)
        raw_obs = _tdw_obs_from_angle(env, theta)
        obs = _normalize_obs(raw_obs, norm_state, critic=False)
        critic_obs = _normalize_obs(raw_obs, norm_state, critic=True)

        # Sample many actions by vmapping single-sample calls.
        sample_keys = jax.random.split(key, int(num_samples))

        def _one(k):
            action, *_ = actor_model.sample(k, obs[None, :])
            return action[0]

        sampled = jax.vmap(_one)(sample_keys)
        sampled_np = np.asarray(jax.device_get(sampled)).reshape(-1)

        next_angle = env._wrap_angle(theta + grid_actions * env.max_turn_radians)
        reward_curve = env.reward_from_angle(next_angle)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        q_curve_np = None
        if critic_model is not None:
            critic_obs_grid = jnp.broadcast_to(critic_obs, (grid_actions.shape[0],) + critic_obs.shape)
            action_grid = grid_actions.reshape(-1, 1)
            q_vals = critic_model.critic(critic_obs_grid, action_grid).squeeze(-1)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        ax_hist = axes[row_idx, 0]
        ax_q = axes[row_idx, 1]

        ax_hist.hist(sampled_np, bins=int(num_bins), range=(-1.0, 1.0), density=True, alpha=0.6)
        ax_hist.set_xlim(-1.0, 1.0)
        ax_hist.set_ylabel("density")
        theta_deg = float((np.rad2deg(float(theta)) + 360.0) % 360.0)
        ax_hist.set_title(f"θ0={theta_deg:.1f}°: DIME action samples + reward")
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.0)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward")

        if q_curve_np is not None:
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.0)
            ax_q.set_title("Q(s,a) over action grid")
            ax_q.set_xlim(-1.0, 1.0)
            ax_q.set_xlabel("action")
            ax_q.set_ylabel("Q")
        else:
            ax_q.axis("off")
            ax_q.text(0.5, 0.5, "No critic in checkpoint", ha="center", va="center")

    fig.tight_layout()
    out_path = _resolve_tdw_analysis_out(checkpoint_path=checkpoint_path, out_path=out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved TurningDoubleWell action analysis to %s", out_path)


def _load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _select_seed(tree, seed_idx: int, num_seeds: int):
    def _maybe_index(x):
        x = np.asarray(x)
        # Training uses `jax.vmap(..., num_seeds)` even when `num_seeds == 1`,
        # so parameters/env-state often carry a leading axis of size 1.
        if num_seeds > 0 and x.ndim > 0 and x.shape[0] == num_seeds:
            return x[seed_idx]
        return x

    return jax.tree.map(_maybe_index, tree)


def _override_cfg(cfg, horizon: int | None, num_envs: int | None):
    if horizon is not None:
        OmegaConf.update(cfg, "env.max_episode_steps", int(horizon), merge=False)
        if OmegaConf.select(cfg, "hyperparameters.max_episode_steps") is not None:
            OmegaConf.update(
                cfg, "hyperparameters.max_episode_steps", int(horizon), merge=False
            )
    if num_envs is not None and OmegaConf.select(cfg, "hyperparameters.num_envs") is not None:
        OmegaConf.update(cfg, "hyperparameters.num_envs", int(num_envs), merge=False)
    return cfg


def _parse_kv_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Invalid override '{item}'. Expected key=value.")
    key, raw = item.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise ValueError(f"Invalid override '{item}'. Empty key.")

    lowered = raw.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    if lowered in ("none", "null"):
        return key, None

    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        pass
    try:
        return key, ast.literal_eval(raw)
    except Exception:
        return key, raw


def _apply_env_config_overrides(cfg, overrides: list[str]):
    if not overrides:
        return cfg
    if OmegaConf.select(cfg, "env.config") is None:
        OmegaConf.update(cfg, "env.config", {}, merge=False)
    for item in overrides:
        key, value = _parse_kv_override(item)
        OmegaConf.update(cfg, f"env.config.{key}", value, merge=False)
    return cfg


def _build_base_env(cfg, *, horizon: int):
    if cfg.env.type == "brax":
        return BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=horizon,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.get("terminate", True),
        )
    if cfg.env.type != "mjx":
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    env_config = OmegaConf.select(cfg, "env.config")
    if env_config is not None:
        env_config = OmegaConf.to_container(env_config, resolve=True)
    return MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=horizon,
        reward_scale=cfg.env.reward_scaling,
        push_distractions=cfg.env.get("push_distractions", False),
        config=env_config,
        asymmetric_observation=cfg.env.get("asymmetric_observation", False),
    )


def _unwrap_env(env):
    cur = env
    while hasattr(cur, "env"):
        cur = getattr(cur, "env")
    return cur


def _render_turning_double_well_reppo(
    *,
    method_name: str,
    diffusion_sampler: str = "auto",
    cfg,
    actor_graphdef=None,
    actor_params: Any | None = None,
    train_state=None,
    checkpoint_path: str,
    norm_state: Any,
    horizon: int,
    num_envs: int,
    out_path: str | None,
    overlay: bool,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> None:
    is_dmerl = str(method_name) == "reppo_DMERL_new"
    is_dime = "dime" in str(method_name).lower()
    sampler = str(diffusion_sampler or "auto").lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")

    cfg_render = cfg
    if is_dmerl:
        # Build a cfg variant with num_envs reduced for render (so we don't step 1000+ envs).
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["hyperparameters"]["num_envs"] = int(num_envs)
        cfg_render = OmegaConf.create(cfg_dict)

    normalize_env = bool(OmegaConf.select(cfg_render, "hyperparameters.normalize_env") or False)

    # Build the same wrapper stack as training for consistent observations.
    if is_dmerl:
        base_env = _build_base_env(cfg_render, horizon=horizon)
        diff_cfg = cfg_render.hyperparameters.diffusion
        env_action_clip_value = float(cfg_render.hyperparameters.env_action_clip_value)
        env = MjxDiffEnvWrapper(
            base_env,
            num_diff_steps=int(diff_cfg.diff_steps),
            diffusion_config=diff_cfg,
            low=-env_action_clip_value,
            high=env_action_clip_value,
        )
        env = LogWrapper(env, int(num_envs))
        if normalize_env:
            env = DiffNormalizeVec(
                env,
                normalize_reward=bool(cfg_render.hyperparameters.normalize_reward),
                num_diff_steps=int(diff_cfg.diff_steps),
            )
    else:
        env = _build_base_env(cfg_render, horizon=horizon)
        if normalize_env:
            env = LogWrapper(env, int(num_envs))
        env = ClipAction(
            env,
            low=-float(cfg_render.hyperparameters.env_action_clip_value),
            high=float(cfg_render.hyperparameters.env_action_clip_value),
        )
        if normalize_env:
            env = NormalizeVec(
                env, normalize_reward=bool(cfg_render.hyperparameters.normalize_reward)
            )

    base_env = _unwrap_env(env)

    if not hasattr(base_env, "render_trajectory"):
        logging.warning("Underlying env does not expose render_trajectory; skipping render.")
        return

    out_path = _resolve_render_out(
        method_name=method_name,
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        diffusion_sampler=diffusion_sampler,
    )

    if is_dmerl and train_state is None:
        raise ValueError("DMERL rendering requires train_state.")
    if train_state is not None:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
    else:
        if actor_graphdef is None or actor_params is None:
            raise ValueError("render requires either train_state or (actor_graphdef, actor_params).")
        actor_model = nnx.merge(actor_graphdef, actor_params)
        critic_model = None

    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, int(num_envs))
    if normalize_env and norm_state is not None:
        obs, critic_obs, env_state = env.reset(init_keys, norm_state)
    else:
        obs, critic_obs, env_state = env.reset(init_keys)

    try:
        from src.env_utils.turning_double_well_env import TurningDoubleWellState
    except Exception:
        TurningDoubleWellState = None

    def _state_to_numpy(state):
        if TurningDoubleWellState is None:
            return jax.tree.map(lambda x: np.asarray(x), state)
        return TurningDoubleWellState(
            pos=np.asarray(state.pos),
            direction=np.asarray(state.direction),
            angle=np.asarray(state.angle),
            t=np.asarray(state.t),
            rng=np.asarray(state.rng),
            obs=np.asarray(state.obs),
            reward=np.asarray(state.reward),
            done=np.asarray(state.done),
            info=jax.tree.map(lambda x: np.asarray(x), state.info),
        )

    def _unwrap_env_state(state):
        cur = state
        while hasattr(cur, "env_state"):
            cur = getattr(cur, "env_state")
        return cur

    states = [_state_to_numpy(_unwrap_env_state(env_state))]
    rewards = []

    if is_dmerl:
        from src.jaxrl.reppo_helpers.learning_DiffReppo import maybe_add_q_grad

        diff_cfg = cfg_render.hyperparameters.diffusion
        diff_steps = int(diff_cfg.diff_steps)
        total_steps = int(horizon) * diff_steps
        use_langevin = bool(
            OmegaConf.select(cfg_render, "hyperparameters.diffusion.score_model.langevin_param")
            or False
        )
        train_mode = str(OmegaConf.select(cfg_render, "hyperparameters.train_mode") or "")
        sampler_mode = sampler
        if sampler_mode == "auto":
            sampler_mode = "sde" if train_mode == "WPO" else "ode"

        for step_idx in range(total_steps):
            key, act_key, env_key = jax.random.split(key, 3)
            obs_for_actor = maybe_add_q_grad(
                obs, critic_obs, actor_model, critic_model, use_langevin
            )
            if sampler_mode == "sde":
                action, *_ = actor_model.vmap_sample_next_step(obs_for_actor, act_key)
            elif sampler_mode == "ode":
                action, _ = actor_model.vmap_ode_sample_next_step(obs_for_actor, act_key)
            else:
                raise ValueError(f"Unknown diffusion sampler: {sampler_mode}")
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, critic_obs, env_state, reward, done, info = env.step(
                step_keys, env_state, action
            )
            # Only append when the underlying env advanced (once per diffusion loop).
            if (step_idx + 1) % diff_steps == 0:
                rewards.append(np.asarray(reward))
                states.append(_state_to_numpy(_unwrap_env_state(env_state)))
    else:
        for _ in range(int(horizon)):
            key, act_key, step_key = jax.random.split(key, 3)
            step_keys = jax.random.split(step_key, int(num_envs))
            if is_dime:
                dime_sampler = sampler
                if dime_sampler == "auto":
                    dime_sampler = "sde"
                if dime_sampler == "sde":
                    action, *_ = actor_model.sample(act_key, obs)
                else:
                    action, *_ = actor_model.det_action(act_key, obs, ode=True, ode_coef=1.0)
            else:
                action = actor_model.det_action(obs)
            obs, critic_obs, env_state, reward, done, info = env.step(
                step_keys, env_state, action
            )
            rewards.append(np.asarray(reward))
            states.append(_state_to_numpy(_unwrap_env_state(env_state)))

    rewards_np = np.stack(rewards, axis=0) if rewards else None
    frames = base_env.render_trajectory(
        states,
        rewards=rewards_np,
        width=int(width),
        height=int(height),
        overlay=bool(overlay),
        max_envs=int(num_envs),
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, frames, fps=int(fps))
        png_path = os.path.splitext(out_path)[0] + ".png"
        imageio.imwrite(png_path, frames[-1])
        logging.info("Saved TurningDoubleWellEnv render to %s (and %s)", out_path, png_path)
    except Exception as e:
        npz_path = os.path.splitext(out_path)[0] + ".npz"
        np.savez(npz_path, frames=np.asarray(frames))
        logging.warning("Failed to write GIF/PNG (%s). Saved raw frames to %s", e, npz_path)


def _eval_reppo(checkpoint: dict[str, Any], cfg, args) -> dict[str, float]:
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    num_envs = int(hp.num_envs)
    reward_scaling = float(cfg.env.reward_scaling)

    env = _build_base_env(cfg, horizon=horizon)
    # Mirror the training wrapper stack in `src/jaxrl/reppo.py`:
    # LogWrapper -> ClipAction -> (optional) NormalizeVec
    if bool(hp.normalize_env):
        env = LogWrapper(env, num_envs)
    env = ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        env = NormalizeVec(env, normalize_reward=bool(hp.normalize_reward))

    obs_dim = int(env.observation_space(None)[0].shape[0])
    action_dim = int(env.action_space(None).shape[0])

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = checkpoint["actor_params"]
    norm_state = checkpoint.get("last_env_state", None)
    actor_params = _select_seed(actor_params, seed_idx=seed_idx, num_seeds=num_seeds)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, seed_idx=seed_idx, num_seeds=num_seeds)

    actor_key = jax.random.PRNGKey(0)
    actor_template = SACActorNetworks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(hp.actor_hidden_dim),
        ent_start=float(hp.ent_start),
        kl_start=float(hp.kl_start),
        use_norm=bool(hp.use_actor_norm),
        layers=int(hp.num_actor_layers),
        use_skip=bool(hp.use_actor_skip),
        train_mode=str(getattr(hp, "train_mode", "reparam")),
        disable_wpo_fisher_preconditioning=bool(getattr(hp, "disable_wpo_fisher_preconditioning", False)),
        disable_temperature=bool(getattr(hp, "disable_temperature", False)),
        rngs=nnx.Rngs(actor_key),
    )
    actor_graphdef = nnx.graphdef(actor_template)
    actor_params = _to_jax_tree(actor_params)
    if norm_state is not None:
        norm_state = _to_jax_tree(norm_state)
    if not bool(hp.normalize_env):
        norm_state = None

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        actor_model = nnx.merge(actor_graphdef, actor_params)
        critic_model = None
        critic_params = checkpoint.get("critic_params", None)
        if critic_params is not None:
            from src.networks.jax_models import CriticNetwork, CategoricalCriticNetwork

            critic_obs_dim = int(env.observation_space(None)[1].shape[0])
            if bool(getattr(hp, "hl_gauss", False)):
                critic_template = CategoricalCriticNetwork(
                    obs_dim=critic_obs_dim,
                    action_dim=action_dim,
                    hidden_dim=int(getattr(hp, "critic_hidden_dim", 512)),
                    num_bins=int(getattr(hp, "num_bins", 51)),
                    vmin=float(getattr(hp, "vmin", -10.0)),
                    vmax=float(getattr(hp, "vmax", 10.0)),
                    use_norm=bool(getattr(hp, "use_critic_norm", True)),
                    use_simplical_embedding=bool(getattr(hp, "use_simplical_embedding", False)),
                    encoder_layers=int(getattr(hp, "num_critic_encoder_layers", 1)),
                    head_layers=int(getattr(hp, "num_critic_head_layers", 1)),
                    pred_layers=int(getattr(hp, "num_critic_pred_layers", 1)),
                    use_skip=bool(getattr(hp, "use_critic_skip", False)),
                    rngs=nnx.Rngs(jax.random.PRNGKey(0)),
                )
            else:
                critic_template = CriticNetwork(
                    obs_dim=critic_obs_dim,
                    action_dim=action_dim,
                    hidden_dim=int(getattr(hp, "critic_hidden_dim", 512)),
                    use_norm=bool(getattr(hp, "use_critic_norm", True)),
                    use_simplical_embedding=bool(getattr(hp, "use_simplical_embedding", False)),
                    encoder_layers=int(getattr(hp, "num_critic_encoder_layers", 1)),
                    head_layers=int(getattr(hp, "num_critic_head_layers", 1)),
                    pred_layers=int(getattr(hp, "num_critic_pred_layers", 1)),
                    use_skip=bool(getattr(hp, "use_critic_skip", False)),
                    rngs=nnx.Rngs(jax.random.PRNGKey(0)),
                )
            critic_graphdef = nnx.graphdef(critic_template)
            critic_params = _to_jax_tree(_select_seed(critic_params, seed_idx=seed_idx, num_seeds=num_seeds))
            critic_model = nnx.merge(critic_graphdef, critic_params)

        _tdw_action_analysis_sac(
            cfg=cfg,
            checkpoint_path=str(args.checkpoint),
            actor_model=actor_model,
            critic_model=critic_model,
            norm_state=norm_state,
            out_path=getattr(args, "tdw_action_analysis_out", None),
            num_samples=int(getattr(args, "tdw_action_analysis_samples", 4096)),
            num_grid=int(getattr(args, "tdw_action_analysis_grid", 401)),
            num_bins=int(getattr(args, "tdw_action_analysis_bins", 60)),
            seed=int(getattr(args, "tdw_action_analysis_seed", 0)),
            max_states=getattr(args, "tdw_action_analysis_max_states", None),
        )

    if bool(getattr(args, "render", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        render_num_envs = max(1, int(getattr(args, "render_num_envs", 10)))
        _render_turning_double_well_reppo(
            method_name=str(checkpoint.get("method_name", "reppo")),
            diffusion_sampler=getattr(args, "diffusion_sampler", "auto"),
            cfg=cfg,
            actor_graphdef=actor_graphdef,
            actor_params=actor_params,
            checkpoint_path=str(args.checkpoint),
            norm_state=norm_state,
            horizon=horizon,
            num_envs=render_num_envs,
            out_path=getattr(args, "render_out", None),
            overlay=not bool(getattr(args, "render_grid", False)),
            width=int(getattr(args, "render_width", 960)),
            height=int(getattr(args, "render_height", 720)),
            fps=int(getattr(args, "render_fps", 20)),
            seed=int(getattr(args, "render_seed", 0)),
        )

    @jax.jit
    def rollout(key: jax.Array, actor_params_jax, norm_state_jax):
        actor_model = nnx.merge(actor_graphdef, actor_params_jax)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, num_envs)
        if norm_state_jax is None:
            obs, _, env_state = env.reset(init_keys)
        else:
            obs, _, env_state = env.reset(init_keys, norm_state_jax)

        def step_fn(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            if str(getattr(hp, "train_mode", "reparam")) == "WPO":
                act_keys = jax.random.split(act_key, num_envs)
                action = jax.vmap(lambda k, o: actor_model.actor(o).sample(seed=k))(
                    act_keys, obs
                )
            else:
                action = actor_model.det_action(obs)
            step_keys = jax.random.split(env_key, num_envs)
            next_obs, _, next_state, reward, done, info = env.step(step_keys, env_state, action)
            return (key, next_state, next_obs), reward

        (_, _, _), rewards = jax.lax.scan(step_fn, (key, env_state, obs), xs=None, length=horizon)
        returns = rewards.sum(axis=0) * (1.0 / reward_scaling)
        return returns.mean(), returns.std()

    eval_key = jax.random.PRNGKey(123)
    mean_return, std_return = rollout(eval_key, actor_params, norm_state)
    return {
        "episode_return_mean": float(np.asarray(mean_return)),
        "episode_return_std": float(np.asarray(std_return)),
        "horizon": float(horizon),
        "num_envs": float(num_envs),
    }


def _eval_reppo_dmerl_new(checkpoint: dict[str, Any], cfg, args) -> dict[str, float]:
    # Lazy import: DMERL pulls in extra modules.
    from src.jaxrl.reppo_DMERL_new import ReppoDMERLTrainer, ReppoConfig

    horizon = int(cfg.env.max_episode_steps)
    base_env = _build_base_env(cfg, horizon=horizon)
    diff_cfg = cfg.hyperparameters.diffusion
    env_action_clip_value = float(cfg.hyperparameters.env_action_clip_value)
    env = MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )

    trainer = ReppoDMERLTrainer(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        num_seeds=1,
        reward_scale=1.0 / float(cfg.env.reward_scaling),
    )
    init_fn = trainer._make_init_fn()
    key = jax.random.PRNGKey(0)
    train_state = init_fn(key)

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = _select_seed(checkpoint["actor_params"], seed_idx, num_seeds)
    critic_params = _select_seed(checkpoint["critic_params"], seed_idx, num_seeds)
    actor_target_params = _select_seed(checkpoint.get("actor_target_params", checkpoint["actor_params"]), seed_idx, num_seeds)
    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, seed_idx, num_seeds)

    actor_params = _to_jax_tree(actor_params)
    critic_params = _to_jax_tree(critic_params)
    actor_target_params = _to_jax_tree(actor_target_params)
    if norm_state is not None:
        norm_state = _to_jax_tree(norm_state)

    train_state = train_state.replace(
        actor=train_state.actor.replace(params=actor_params),
        critic=train_state.critic.replace(params=critic_params),
        actor_target=train_state.actor_target.replace(params=actor_target_params),
    )
    if not bool(cfg.hyperparameters.normalize_env):
        norm_state = None

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        _tdw_action_analysis_dmerl(
            cfg=cfg,
            checkpoint_path=str(args.checkpoint),
            actor_model=actor_model,
            critic_model=critic_model,
            norm_state=norm_state,
            out_path=getattr(args, "tdw_action_analysis_out", None),
            num_samples=int(getattr(args, "tdw_action_analysis_samples", 4096)),
            num_grid=int(getattr(args, "tdw_action_analysis_grid", 401)),
            num_bins=int(getattr(args, "tdw_action_analysis_bins", 60)),
            seed=int(getattr(args, "tdw_action_analysis_seed", 0)),
            max_states=getattr(args, "tdw_action_analysis_max_states", None),
        )

    if bool(getattr(args, "render", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        render_num_envs = max(1, int(getattr(args, "render_num_envs", 10)))
        _render_turning_double_well_reppo(
            method_name=str(checkpoint.get("method_name", "reppo_DMERL_new")),
            diffusion_sampler=getattr(args, "diffusion_sampler", "auto"),
            cfg=cfg,
            train_state=train_state,
            checkpoint_path=str(args.checkpoint),
            norm_state=norm_state,
            horizon=horizon,
            num_envs=render_num_envs,
            out_path=getattr(args, "render_out", None),
            overlay=not bool(getattr(args, "render_grid", False)),
            width=int(getattr(args, "render_width", 960)),
            height=int(getattr(args, "render_height", 720)),
            fps=int(getattr(args, "render_fps", 20)),
            seed=int(getattr(args, "render_seed", 0)),
        )

    eval_key = jax.random.PRNGKey(123)
    sampler = str(getattr(args, "diffusion_sampler", "auto")).lower()
    if sampler == "auto":
        eval_fn = trainer.eval_fn
    elif sampler == "sde":
        train_mode = str(getattr(cfg.hyperparameters, "train_mode", ""))
        eval_fn = trainer._make_sde_eval_fn(eval_policy=(train_mode == "WPO"))
    elif sampler == "ode":
        eval_fn = trainer._make_ode_eval_fn()
    else:
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")

    metrics = eval_fn(eval_key, train_state, norm_state)
    metrics = jax.tree.map(lambda x: float(np.asarray(x)), metrics)
    return metrics




def _eval_reppo_dime(checkpoint: dict[str, Any], cfg, args) -> dict[str, float]:
    # DIME policy weights saved from `src/jaxrl/reppo_dime.py`.
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    num_envs = int(hp.num_envs)
    reward_scaling = float(cfg.env.reward_scaling)

    env = _build_base_env(cfg, horizon=horizon)
    # Mirror the wrapper stack in `src/jaxrl/reppo_dime.py`:
    # LogWrapper -> ClipAction -> (optional) NormalizeVec
    env = LogWrapper(env, num_envs)
    env = ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        if bool(getattr(hp, "normalize_reward", False)):
            logging.warning(
                "reppo_dime training currently uses NormalizeVec(normalize_reward=False); "
                "ignoring normalize_reward=true for evaluation."
            )
        env = NormalizeVec(env)

    obs_dim = int(env.observation_space(None)[0].shape[0])
    action_dim = int(env.action_space(None).shape[0])

    def _call_target(spec: Any):
        if spec is None:
            raise ValueError("Missing diffusion.dt_schedule; cannot reconstruct DIME actor.")
        if callable(spec):
            return spec
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        if not isinstance(spec, dict) or "_target_" not in spec:
            raise TypeError(f"Unsupported dt_schedule spec: {type(spec)} ({spec})")
        target = str(spec["_target_"])
        module_path, attr = target.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_path), attr)
        kwargs = {k: v for k, v in spec.items() if k != "_target_"}
        return fn(**kwargs)

    def _build_actor_graphdef():
        from src.networks.diffusion.models import ControlNetwork
        from src.networks.jax_models import (
            DIMEActor,
            DiffusionModel,
            logratio_DIME,
            ode_integrator,
            sde_integrator,
        )

        diff_cfg = hp.diffusion
        dt_schedule = _call_target(getattr(diff_cfg, "dt_schedule", None))

        key = jax.random.PRNGKey(0)
        _, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        score_cfg = diff_cfg.score_model
        forward_model = None
        if bool(getattr(diff_cfg, "learn_forward", False)):
            forward_model = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=int(score_cfg.num_layers),
                num_hid=int(score_cfg.num_hid),
                num_time_hid=int(score_cfg.num_time_hid),
                num_time_out=int(score_cfg.num_time_out),
                outer_clip=float(score_cfg.outer_clip),
                inner_clip=float(score_cfg.inner_clip),
                weight_init=float(score_cfg.weight_init),
                bias_init=float(score_cfg.bias_init),
                layer_norm=bool(score_cfg.layer_norm),
                layer_norm_type=str(score_cfg.layer_norm_type),
                max_time=float(diff_cfg.diff_steps),
                rngs=rngs,
            )

        backward_model = None
        if bool(getattr(diff_cfg, "learn_backward", False)):
            backward_model = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=int(score_cfg.num_layers),
                num_hid=int(score_cfg.num_hid),
                num_time_hid=int(score_cfg.num_time_hid),
                num_time_out=int(score_cfg.num_time_out),
                outer_clip=float(score_cfg.outer_clip),
                inner_clip=float(score_cfg.inner_clip),
                weight_init=float(score_cfg.weight_init),
                bias_init=float(score_cfg.bias_init),
                layer_norm=bool(score_cfg.layer_norm),
                layer_norm_type=str(score_cfg.layer_norm_type),
                max_time=float(diff_cfg.diff_steps),
                rngs=rngs,
            )

        diffusion_model = DiffusionModel(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            bwd_model=backward_model,
            diff_steps=int(diff_cfg.diff_steps),
            init_std=float(diff_cfg.init_std),
            friction=float(diff_cfg.friction),
            per_dim_friction=bool(diff_cfg.per_dim_friction),
            dt=float(diff_cfg.dt),
            learn_dt=bool(diff_cfg.learn_dt),
            per_step_dt=bool(diff_cfg.per_step_dt),
            learn_prior=bool(diff_cfg.learn_prior),
            learn_betas=bool(diff_cfg.learn_betas),
            learn_friction=bool(diff_cfg.learn_friction),
            learn_mass_matrix=bool(diff_cfg.learn_mass_matrix),
            dt_schedule=dt_schedule,
            rngs=rngs,
        )

        actor_networks = DIMEActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            logratio=logratio_DIME,
            kl_start=float(hp.kl_start),
            ent_start=float(hp.ent_start),
            sde_integrator=sde_integrator,
            ode_integrator=ode_integrator,
        )
        return nnx.graphdef(actor_networks)

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = _select_seed(
        checkpoint["actor_params"], seed_idx=seed_idx, num_seeds=num_seeds
    )
    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, seed_idx=seed_idx, num_seeds=num_seeds)

    actor_graphdef = _build_actor_graphdef()
    actor_params = _to_jax_tree(actor_params)
    if norm_state is not None:
        norm_state = _to_jax_tree(norm_state)
    if not bool(hp.normalize_env):
        norm_state = None

    if bool(getattr(args, "render", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        render_num_envs = max(1, int(getattr(args, "render_num_envs", 10)))
        _render_turning_double_well_reppo(
            method_name=str(checkpoint.get("method_name", "reppo_dime")),
            diffusion_sampler=getattr(args, "diffusion_sampler", "auto"),
            cfg=cfg,
            actor_graphdef=actor_graphdef,
            actor_params=actor_params,
            checkpoint_path=str(args.checkpoint),
            norm_state=norm_state,
            horizon=horizon,
            num_envs=render_num_envs,
            out_path=getattr(args, "render_out", None),
            overlay=not bool(getattr(args, "render_grid", False)),
            width=int(getattr(args, "render_width", 960)),
            height=int(getattr(args, "render_height", 720)),
            fps=int(getattr(args, "render_fps", 20)),
            seed=int(getattr(args, "render_seed", 0)),
        )

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        actor_model = nnx.merge(actor_graphdef, actor_params)
        critic_model = None
        critic_params = checkpoint.get("critic_params", None)
        if critic_params is not None:
            from src.networks.jax_models import CriticNetwork, CategoricalCriticNetwork

            critic_obs_dim = int(env.observation_space(None)[1].shape[0])
            if bool(getattr(hp, "hl_gauss", False)):
                critic_template = CategoricalCriticNetwork(
                    obs_dim=critic_obs_dim,
                    action_dim=action_dim,
                    hidden_dim=int(getattr(hp, "critic_hidden_dim", 512)),
                    num_bins=int(getattr(hp, "num_bins", 51)),
                    vmin=float(getattr(hp, "vmin", -10.0)),
                    vmax=float(getattr(hp, "vmax", 10.0)),
                    use_norm=bool(getattr(hp, "use_critic_norm", True)),
                    use_simplical_embedding=bool(getattr(hp, "use_simplical_embedding", False)),
                    encoder_layers=int(getattr(hp, "num_critic_encoder_layers", 1)),
                    head_layers=int(getattr(hp, "num_critic_head_layers", 1)),
                    pred_layers=int(getattr(hp, "num_critic_pred_layers", 1)),
                    use_skip=bool(getattr(hp, "use_critic_skip", False)),
                    rngs=nnx.Rngs(jax.random.PRNGKey(0)),
                )
            else:
                critic_template = CriticNetwork(
                    obs_dim=critic_obs_dim,
                    action_dim=action_dim,
                    hidden_dim=int(getattr(hp, "critic_hidden_dim", 512)),
                    use_norm=bool(getattr(hp, "use_critic_norm", True)),
                    use_simplical_embedding=bool(getattr(hp, "use_simplical_embedding", False)),
                    encoder_layers=int(getattr(hp, "num_critic_encoder_layers", 1)),
                    head_layers=int(getattr(hp, "num_critic_head_layers", 1)),
                    pred_layers=int(getattr(hp, "num_critic_pred_layers", 1)),
                    use_skip=bool(getattr(hp, "use_critic_skip", False)),
                    rngs=nnx.Rngs(jax.random.PRNGKey(0)),
                )
            critic_graphdef = nnx.graphdef(critic_template)
            critic_params = _to_jax_tree(_select_seed(critic_params, seed_idx=seed_idx, num_seeds=num_seeds))
            critic_model = nnx.merge(critic_graphdef, critic_params)

        _tdw_action_analysis_dime(
            cfg=cfg,
            checkpoint_path=str(args.checkpoint),
            actor_model=actor_model,
            critic_model=critic_model,
            norm_state=norm_state,
            out_path=getattr(args, "tdw_action_analysis_out", None),
            num_samples=int(getattr(args, "tdw_action_analysis_samples", 4096)),
            num_grid=int(getattr(args, "tdw_action_analysis_grid", 401)),
            num_bins=int(getattr(args, "tdw_action_analysis_bins", 60)),
            seed=int(getattr(args, "tdw_action_analysis_seed", 0)),
            max_states=getattr(args, "tdw_action_analysis_max_states", None),
        )

    reward_scale = 1.0 / reward_scaling

    sampler = str(getattr(args, "diffusion_sampler", "auto")).lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    use_ode = sampler == "ode"

    @jax.jit
    def rollout(key: jax.Array, actor_params_jax, norm_state_jax):
        actor_model = nnx.merge(actor_graphdef, actor_params_jax)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, num_envs)
        if norm_state_jax is None:
            obs, _, env_state = env.reset(init_keys)
        else:
            obs, _, env_state = env.reset(init_keys, norm_state_jax)

        def step_fn(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            if use_ode:
                action, *_ = actor_model.det_action(act_key, obs, ode=True, ode_coef=1.0)
            else:
                action, *_ = actor_model.sample(act_key, obs)
            step_keys = jax.random.split(env_key, num_envs)
            next_obs, _, next_state, reward, done, info = env.step(
                step_keys, env_state, action
            )
            return (key, next_state, next_obs), (reward, info)

        (_, _, _), (rewards, infos) = jax.lax.scan(
            step_fn, (key, env_state, obs), xs=None, length=horizon
        )

        if isinstance(infos, dict) and "returned_episode" in infos:
            returned = infos["returned_episode"]
            num_episodes = returned.sum()
            sum_returns = infos["returned_episode_returns"].sum(where=returned) * reward_scale
            scaled_returns = infos["returned_episode_returns"] * reward_scale
            sum_sq_returns = jnp.square(scaled_returns).sum(where=returned)
            mean_return = jnp.where(num_episodes > 0, sum_returns / num_episodes, 0.0)
            var_return = jnp.where(
                num_episodes > 0,
                sum_sq_returns / num_episodes - mean_return**2,
                0.0,
            )
            std_return = jnp.sqrt(jnp.maximum(var_return, 0.0))
            return mean_return, std_return, num_episodes

        returns = rewards.sum(axis=0) * reward_scale
        return returns.mean(), returns.std(), jnp.array(0, dtype=jnp.int32)

    eval_key = jax.random.PRNGKey(123)
    mean_return, std_return, num_episodes = rollout(eval_key, actor_params, norm_state)
    return {
        "episode_return_mean": float(np.asarray(mean_return)),
        "episode_return_std": float(np.asarray(std_return)),
        "num_episodes": float(np.asarray(num_episodes)),
        "horizon": float(horizon),
        "num_envs": float(num_envs),
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model checkpoint (reppo, reppo_DMERL_new, or reppo_dime)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pkl checkpoint under saved_models/.")
    parser.add_argument("--seed-idx", type=int, default=0, help="Which trained seed index (when checkpoint contains multiple seeds).")
    parser.add_argument("--horizon", type=int, default=None, help="Override evaluation horizon / max_episode_steps.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel envs (only safe when normalize_env is false).")
    parser.add_argument(
        "--env-config-override",
        action="append",
        default=[],
        help=(
            "Override entries in cfg.env.config via key=value (repeatable). "
            "For TurningDoubleWellEnv, these map to TurningDoubleWellEnv(**kwargs), e.g. "
            "--env-config-override randomize_initial_heading=true "
            "--env-config-override snap_action_to_optimal=false."
        ),
    )
    parser.add_argument(
        "--diffusion-sampler",
        choices=["auto", "sde", "ode"],
        default="auto",
        help=(
            "For diffusion-based methods (reppo_DMERL_new, reppo_dime), choose whether actions are sampled via the SDE or ODE path during evaluation (and TurningDoubleWell rendering). 'auto' keeps the method default."
        ),
    )

    # TurningDoubleWellEnv rendering (reppo + reppo_DMERL_new + reppo_dime).
    parser.add_argument(
        "--render",
        action="store_true",
        help="If env.name=TurningDoubleWellEnv, render a batch trajectory GIF (and PNG) into artifacts/ (optionally set --render-out basename).",
    )
    parser.add_argument("--render-num-envs", type=int, default=10, help="Number of agents to visualize in parallel (TurningDoubleWellEnv only).")
    parser.add_argument(
        "--render-out",
        type=str,
        default=None,
        help="Output GIF name (basename only). The file is always written to artifacts/. If you pass a custom name, it will be prefixed with `reppo__`, `DMERL__`, or `dime__` if missing.",
    )
    # Overlay is now the default; keep --render-overlay as a silent compatibility flag.
    parser.add_argument("--render-overlay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--render-grid",
        action="store_true",
        help="Render each agent in its own subplot (default overlays all agents into one plot).",
    )
    parser.add_argument("--render-width", type=int, default=960, help="Render width in pixels (TurningDoubleWellEnv only).")
    parser.add_argument("--render-height", type=int, default=720, help="Render height in pixels (TurningDoubleWellEnv only).")
    parser.add_argument("--render-fps", type=int, default=20, help="FPS for the rendered GIF (TurningDoubleWellEnv only).")
    parser.add_argument("--render-seed", type=int, default=0, help="PRNG seed for the rendered rollout (TurningDoubleWellEnv only).")

    # TurningDoubleWellEnv action/Q analysis (histograms + reward/Q overlays).
    parser.add_argument(
        "--tdw-action-analysis",
        action="store_true",
        help=(
            "If env.name=TurningDoubleWellEnv, enumerate all discrete starting headings "
            "(multiples of well_angle_deg) and, for each heading, sample many actions and plot: "
            "(1) action histogram overlaid with the one-step reward curve and "
            "(2) Q(s,a) over the action grid when the critic is available."
        ),
    )
    parser.add_argument("--tdw-action-analysis-samples", type=int, default=4096, help="Number of action samples per starting heading.")
    parser.add_argument("--tdw-action-analysis-grid", type=int, default=401, help="Number of action grid points for reward/Q curves.")
    parser.add_argument("--tdw-action-analysis-bins", type=int, default=60, help="Histogram bins.")
    parser.add_argument("--tdw-action-analysis-seed", type=int, default=0, help="PRNG seed used for action sampling.")
    parser.add_argument("--tdw-action-analysis-out", type=str, default=None, help="PNG basename (written into artifacts/).")
    parser.add_argument("--tdw-action-analysis-max-states", type=int, default=None, help="Optional cap on number of starting headings to plot (useful when well_angle_deg is small).")

    args = parser.parse_args()

    ckpt_path = os.path.expanduser(args.checkpoint)
    checkpoint = _load_checkpoint(ckpt_path)
    checkpoint["seed_idx"] = int(args.seed_idx)

    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing cfg; cannot reconstruct env/model.")
    cfg = OmegaConf.create(cfg_dict)

    if args.num_envs is not None and bool(OmegaConf.select(cfg, "hyperparameters.normalize_env") or False):
        logging.warning("--num-envs override requested but normalize_env=true; ignoring override for safety.")
        num_envs_override = None
    else:
        num_envs_override = args.num_envs

    cfg = _override_cfg(cfg, horizon=args.horizon, num_envs=num_envs_override)
    cfg = _apply_env_config_overrides(cfg, list(getattr(args, "env_config_override", []) or []))
    method_name = checkpoint.get("method_name", None)
    if method_name is None:
        # Heuristic fallback for older checkpoints that don't store method_name.
        if OmegaConf.select(cfg, "hyperparameters.diffusion") is not None:
            if OmegaConf.select(cfg, "hyperparameters.temperature_lagragian_lr") is not None:
                method_name = "reppo_dime"
            else:
                method_name = "reppo_DMERL_new"
        else:
            method_name = "reppo"
    method_name = str(method_name)

    if method_name == "reppo_DMERL_new":
        metrics = _eval_reppo_dmerl_new(checkpoint, cfg, args)
    elif "dime" in method_name.lower():
        metrics = _eval_reppo_dime(checkpoint, cfg, args)
    else:
        metrics = _eval_reppo(checkpoint, cfg, args)

    logging.info("Evaluation metrics:")
    for k, v in metrics.items():
        logging.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
