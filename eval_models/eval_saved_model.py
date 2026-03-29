import argparse
import json
import logging
import os
import pickle
import sys
import importlib
import ast
import math
import inspect
from datetime import datetime, timezone
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
    TanhClipAction,
)
from src.jaxrl.normalization import DictNormalizer
from src.networks.jax_models import SACActorNetworks

logging.basicConfig(level=logging.INFO)


def _progress(message: str) -> None:
    print(f"[eval_saved_model] {message}", flush=True)


def _progress_bar(prefix: str, current: int, total: int) -> None:
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "." * (width - filled)
    pct = 100.0 * current / total
    line = (
        f"\r[eval_saved_model] {prefix}: [{bar}] "
        f"{current}/{total} ({pct:5.1f}%)"
    )
    if current >= total:
        print(line, flush=True)
    else:
        print(line, end="", flush=True)


def _artifacts_dir() -> str:
    path = os.path.join(_REPO_ROOT, "artifacts")
    os.makedirs(path, exist_ok=True)
    return path


def _trajectory_data_dir() -> str:
    path = os.path.join(_REPO_ROOT, "eval_models", "saved_trajectories")
    os.makedirs(path, exist_ok=True)
    return path


def _cfg_with_num_envs(cfg, num_envs: int):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["hyperparameters"]["num_envs"] = int(num_envs)
    return OmegaConf.create(cfg_dict)


def _obs_batch_to_numpy(obs_batch: Any) -> np.ndarray:
    if isinstance(obs_batch, dict):
        if "orig_obs" in obs_batch:
            obs_arr = np.asarray(jax.device_get(obs_batch["orig_obs"]), dtype=np.float32)
            if obs_arr.ndim == 1:
                return obs_arr[None, :]
            return obs_arr.reshape(obs_arr.shape[0], -1)
        parts: list[np.ndarray] = []
        for key in sorted(obs_batch.keys()):
            arr = np.asarray(jax.device_get(obs_batch[key]), dtype=np.float32)
            if arr.ndim == 0:
                arr = arr[None, None]
            elif arr.ndim == 1:
                arr = arr[:, None]
            else:
                arr = arr.reshape(arr.shape[0], -1)
            parts.append(arr)
        if not parts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(parts, axis=-1)
    arr = np.asarray(jax.device_get(obs_batch), dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def _diff_raw_obs_from_state(env_state: Any) -> np.ndarray | None:
    cur = env_state
    for _ in range(12):
        obs = getattr(cur, "obs", None)
        if obs is not None:
            arr = np.asarray(jax.device_get(obs), dtype=np.float32)
            if arr.ndim == 0:
                return arr.reshape(1, 1)
            if arr.ndim == 1:
                return arr[None, :]
            return arr.reshape(arr.shape[0], -1)
        if hasattr(cur, "env_state"):
            cur = getattr(cur, "env_state")
            continue
        break
    return None


def _digamma_integer(n: int) -> float:
    if n <= 0:
        raise ValueError("digamma_integer requires n > 0.")
    if n == 1:
        return -float(np.euler_gamma)
    return float(np.sum(1.0 / np.arange(1, n, dtype=np.float64)) - np.euler_gamma)


def _kth_neighbor_distances(
    samples: np.ndarray,
    k: int,
    batch_size: int = 256,
    *,
    show_progress: bool = False,
    progress_prefix: str = "kNN distance batches",
) -> np.ndarray:
    n = int(samples.shape[0])
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 1 <= k < num_samples, got k={k}, N={n}.")
    x = np.asarray(samples, dtype=np.float64)
    out = np.empty((n,), dtype=np.float64)
    starts = list(range(0, n, int(batch_size)))
    total_batches = len(starts)
    if show_progress and total_batches > 0:
        _progress_bar(progress_prefix, 0, total_batches)
    for batch_idx, start in enumerate(starts, start=1):
        end = min(start + int(batch_size), n)
        xb = x[start:end]  # [B, D]
        diff = xb[:, None, :] - x[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        row_idx = np.arange(end - start)
        col_idx = np.arange(start, end)
        dist_sq[row_idx, col_idx] = np.inf
        kth_sq = np.partition(dist_sq, kth=k - 1, axis=1)[:, k - 1]
        out[start:end] = np.sqrt(np.maximum(kth_sq, 1e-32))
        if show_progress:
            _progress_bar(progress_prefix, batch_idx, total_batches)
    return out


def _estimate_knn_entropy(
    samples: np.ndarray,
    *,
    k: int,
    max_samples: int | None,
    seed: int,
    batch_size: int,
    show_progress: bool = False,
    progress_prefix: str = "kNN distance batches",
) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D [N, D], got shape={x.shape}.")
    n_all = int(x.shape[0])
    if n_all < 2:
        raise ValueError("Need at least 2 samples for kNN entropy.")
    if max_samples is not None and n_all > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_all, size=int(max_samples), replace=False)
        x = x[idx]
    n = int(x.shape[0])
    if k >= n:
        raise ValueError(
            f"kNN entropy needs k < N after sampling, got k={k}, N={n}."
        )
    d = int(x.shape[1])
    eps = _kth_neighbor_distances(
        x,
        int(k),
        batch_size=int(batch_size),
        show_progress=bool(show_progress),
        progress_prefix=progress_prefix,
    )
    log_cd = (d / 2.0) * math.log(math.pi) - math.lgamma(1.0 + d / 2.0)
    entropy_nats = (
        _digamma_integer(n)
        - _digamma_integer(int(k))
        + log_cd
        + d * float(np.mean(np.log(eps)))
    )
    return {
        "knn_entropy_nats": float(entropy_nats),
        "knn_entropy_bits": float(entropy_nats / math.log(2.0)),
        "knn_k": float(k),
        "knn_num_samples": float(n),
        "knn_dim": float(d),
    }


def _trajectory_entropy_samples(
    trajectories: dict[str, np.ndarray], mode: str
) -> np.ndarray:
    states = np.asarray(trajectories["state_trajectories"], dtype=np.float32)
    if mode == "trajectories":
        # [R, T+1, X, D] -> [R, X, T+1, D] -> [R*X, (T+1)*D]
        states = np.transpose(states, (0, 2, 1, 3))
        return states.reshape(states.shape[0] * states.shape[1], -1)
    if mode == "states":
        # Each state point is one sample.
        return states.reshape(-1, states.shape[-1])
    raise ValueError(f"Unknown trajectory entropy mode: {mode}")


def _compute_trajectory_knn_entropy(
    trajectories: dict[str, np.ndarray], args
) -> dict[str, float]:
    mode = str(getattr(args, "traj_knn_mode", "both"))
    valid_modes = ("both", "trajectories", "states")
    if mode not in valid_modes:
        raise ValueError(f"Unknown --traj-knn-mode: {mode}")

    k_requested = int(getattr(args, "traj_knn_k", 5))
    max_samples = getattr(args, "traj_knn_max_samples", None)
    base_kwargs = dict(
        max_samples=max_samples,
        seed=int(getattr(args, "traj_seed", 0)),
        batch_size=int(getattr(args, "traj_knn_batch_size", 256)),
    )
    out: dict[str, float] = {}
    requested = ("trajectories", "states") if mode == "both" else (mode,)
    for submode in requested:
        samples = _trajectory_entropy_samples(trajectories, mode=submode)
        n_eff = int(samples.shape[0])
        if max_samples is not None:
            n_eff = min(n_eff, int(max_samples))
        k_eff = min(k_requested, n_eff - 1)
        if k_eff < 1:
            raise ValueError(
                f"Not enough samples for kNN entropy in mode '{submode}'. "
                f"Need at least 2 points, got {n_eff}."
            )
        _progress(
            f"Estimating kNN entropy for mode='{submode}' with "
            f"N={n_eff}, D={samples.shape[1]}, k={k_eff}, batch_size={base_kwargs['batch_size']}"
        )
        stats = _estimate_knn_entropy(
            samples,
            k=k_eff,
            show_progress=True,
            progress_prefix=f"kNN {submode}",
            **base_kwargs,
        )
        prefix = f"knn_{submode}"
        out[f"{prefix}_entropy_nats"] = stats["knn_entropy_nats"]
        out[f"{prefix}_entropy_bits"] = stats["knn_entropy_bits"]
        out[f"{prefix}_k"] = stats["knn_k"]
        out[f"{prefix}_num_samples"] = stats["knn_num_samples"]
        out[f"{prefix}_dim"] = stats["knn_dim"]
    out["knn_mode"] = mode
    return out


def _collect_output_dir(
    *,
    checkpoint_path: str,
    method_name: str,
    num_envs: int,
    repeats: int,
    user_out: str | None,
) -> str:
    root = _trajectory_data_dir()
    if user_out:
        stem = os.path.splitext(os.path.basename(str(user_out)))[0]
    else:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = (
            f"{ckpt_stem}__{method_name}"
            f"__x{int(num_envs)}__y{int(repeats)}__{ts}"
        )
    out_dir = os.path.join(root, stem)
    if not os.path.exists(out_dir):
        return out_dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(root, f"{stem}__{ts}")


def _save_trajectory_bundle(
    *,
    out_dir: str,
    trajectories: dict[str, np.ndarray],
    checkpoint_path: str,
    cfg,
    method_name: str,
    train_mode: str | None,
    horizon: int,
    num_envs: int,
    repeats: int,
    diffusion_sampler: str,
    knn_entropy: dict[str, Any] | None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "trajectories.npz")
    np.savez_compressed(
        data_path,
        state_trajectories=trajectories["state_trajectories"],
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "method_name": str(method_name),
        "train_mode": str(train_mode or ""),
        "diffusion_sampler": str(diffusion_sampler),
        "env_type": str(OmegaConf.select(cfg, "env.type")),
        "env_name": str(OmegaConf.select(cfg, "env.name")),
        "horizon": int(horizon),
        "num_envs": int(num_envs),
        "repeats": int(repeats),
        "num_trajectories": int(num_envs) * int(repeats),
        "state_dim": int(trajectories["state_trajectories"].shape[-1]),
        "trajectory_steps": int(trajectories["state_trajectories"].shape[1]),
        "data_file": os.path.abspath(data_path),
    }
    if knn_entropy is not None:
        metadata["knn_entropy"] = knn_entropy

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {
        "trajectory_output_dir": os.path.abspath(out_dir),
        "trajectory_data_path": os.path.abspath(data_path),
        "trajectory_metadata_path": os.path.abspath(metadata_path),
    }


def _method_display_name(
    method_name: str,
    train_mode: str | None = None,
    entropy_coef: float | None = None,
) -> str:
    method_lower = str(method_name).lower()
    mode_upper = str(train_mode or "").upper()
    if "diffppo" in method_lower:
        if entropy_coef is not None and abs(float(entropy_coef)) <= 1e-12:
            return "DPPO (DME-PPO temp = 0)"
        return "DME-PPO"
    if "dime" in method_lower:
        return "REPPO-DIME"
    if "dmerl" in method_lower:
        if mode_upper == "WPO":
            return "DME-WPO"
        return "DME-REPPO"
    return "REPPO"


def _resolve_train_mode(checkpoint: dict[str, Any], cfg) -> str:
    ckpt_mode = checkpoint.get("train_mode", None)
    if ckpt_mode is not None and str(ckpt_mode).strip():
        return str(ckpt_mode)
    cfg_mode = OmegaConf.select(cfg, "hyperparameters.train_mode")
    if cfg_mode is not None and str(cfg_mode).strip():
        return str(cfg_mode)
    diff_mode = OmegaConf.select(cfg, "hyperparameters.diffusion.train_mode")
    if diff_mode is not None and str(diff_mode).strip():
        return str(diff_mode)
    return ""


def _entropy_coef_token(entropy_coef: float | None) -> str | None:
    if entropy_coef is None:
        return None
    value = float(entropy_coef)
    if abs(value) <= 1e-12:
        return "entcoef0"
    text = f"{value:.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return f"entcoef{text}"


def _tau_label(entropy_coef: float | None) -> str:
    if entropy_coef is None:
        return r"$\mathcal{T}$"
    return rf"$\mathcal{{T}} = {float(entropy_coef):.6g}$"


def _diffppo_plot_title_name(entropy_coef: float | None) -> str:
    if entropy_coef is not None and abs(float(entropy_coef)) <= 1e-12:
        return r"DPPO (DME-PPO $\mathcal{T} = 0$)"
    return f"DME-PPO ({_tau_label(entropy_coef)})"


def _resolve_render_out(
    *,
    method_name: str,
    train_mode: str | None,
    entropy_coef: float | None = None,
    checkpoint_path: str,
    out_path: str | None,
    diffusion_sampler: str | None = None,
) -> str:
    # Always write renders into repo_root/artifacts (ignore any provided directory).
    artifacts_dir = _artifacts_dir()
    method_tag = _method_display_name(
        method_name, train_mode=train_mode, entropy_coef=entropy_coef
    )
    if "diffppo" in str(method_name).lower():
        ent_token = _entropy_coef_token(entropy_coef)
        if ent_token is not None:
            method_tag = f"{method_tag}__{ent_token}"
    sampler_lower = str(diffusion_sampler or "").lower()
    sampler_tag = sampler_lower if sampler_lower in ("sde", "ode") else None
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if out_path is None:
        ckpt_lower = ckpt_base.lower()
        if ckpt_lower.startswith(method_tag.lower() + "__"):
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


def _resolve_tdw_analysis_out_pair(*, checkpoint_path: str, out_path: str | None) -> tuple[str, str]:
    hist_out = _resolve_tdw_analysis_out(checkpoint_path=checkpoint_path, out_path=out_path)
    stem, ext = os.path.splitext(hist_out)
    q_out = f"{stem}__q{ext}"
    return hist_out, q_out


def _sanitize_env_config(env_name: str, env_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if env_config is None:
        return None
    sanitized = dict(env_config)
    if str(env_name) == "TurningDoubleWellEnv":
        # Keep only kwargs supported by the local env constructor.
        try:
            from src.env_utils.turning_double_well_env import TurningDoubleWellEnv

            signature = inspect.signature(TurningDoubleWellEnv.__init__)
            params = signature.parameters
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if not has_kwargs:
                accepted = {
                    name for name, p in params.items() if name != "self"
                }
                dropped = sorted([k for k in sanitized if k not in accepted])
                if dropped:
                    logging.warning(
                        "Dropping unsupported TurningDoubleWellEnv config keys: %s",
                        ", ".join(dropped),
                    )
                    sanitized = {k: v for k, v in sanitized.items() if k in accepted}
        except Exception:
            # Best effort fallback for known stale keys from older/newer checkpoints.
            sanitized.pop("use_multi_minima_wells", None)
            sanitized.pop("num_minima", None)
    return sanitized


def _tdw_build_env_for_analysis(cfg):
    # Use the checkpoint config, but force the discrete-initial-heading assumption.
    env_config = OmegaConf.select(cfg, "env.config")
    env_kwargs: dict[str, Any] = {}
    if env_config is not None:
        env_kwargs = OmegaConf.to_container(env_config, resolve=True) or {}
    env_kwargs = _sanitize_env_config("TurningDoubleWellEnv", env_kwargs) or {}
    env_kwargs["randomize_initial_heading"] = True
    env_kwargs["snap_action_to_optimal"] = True
    env_kwargs.setdefault("horizon", int(OmegaConf.select(cfg, "env.max_episode_steps") or 200))
    from src.env_utils.turning_double_well_env import TurningDoubleWellEnv

    return TurningDoubleWellEnv(**env_kwargs)


def _tdw_discrete_starting_angles(env) -> jax.Array:
    # Enumerate full-circle discrete headings in multiples of well_angle_deg.
    # Example: well_angle_deg=45 -> 8 initial headings.
    well_angle_deg = float(getattr(env, "well_angle_deg", 0.0) or 0.0)
    if well_angle_deg > 0.0:
        angles_deg = jnp.arange(-180.0, 180.0, well_angle_deg, dtype=jnp.float32)
        angles_rad = jnp.deg2rad(angles_deg)
        return env._wrap_angle(angles_rad)
    # Fallback to env-provided support if well_angle_deg is unavailable.
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


def _tdw_subplot_grid(n_plots: int, max_cols: int = 4) -> tuple[int, int]:
    n_plots = max(1, int(n_plots))
    ncols = min(int(max_cols), max(1, int(math.ceil(math.sqrt(n_plots)))))
    nrows = int(math.ceil(n_plots / ncols))
    return nrows, ncols


def _tdw_state_title(theta_radians: jax.Array) -> str:
    theta_deg = float((np.rad2deg(float(theta_radians)) + 360.0) % 360.0)
    return rf"$\theta_0 = {theta_deg:.1f}^\circ$"


def _tdw_apply_axis_style(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)


def _tdw_action_analysis_sac(
    *,
    cfg,
    checkpoint_path: str,
    actor_model,
    critic_model,
    norm_state,
    method_name: str = "reppo",
    train_mode: str | None = None,
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))
    method_display = _method_display_name(method_name, train_mode=train_mode)

    nrows, ncols = _tdw_subplot_grid(len(angles_np), max_cols=4)
    fig_hist, axes_hist = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig_hist.suptitle(
        f"{method_display} | Action Samples and Reward Curve",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    hist_axes = axes_hist.reshape(-1)
    fig_q = None
    q_axes = None
    if critic_model is not None:
        fig_q, axes_q = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.0 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        fig_q.suptitle(
            f"{method_display} | Q(s,a) Over Action Grid",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )
        q_axes = axes_q.reshape(-1)

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for plot_idx, (theta, key) in enumerate(zip(angles_np, keys)):
        theta = jnp.asarray(theta, dtype=jnp.float32)
        raw_obs = _tdw_obs_from_angle(env, theta)
        obs = _normalize_obs(raw_obs, norm_state, critic=False)
        critic_obs = _normalize_obs(raw_obs, norm_state, critic=True)

        # Sample actions from the actor distribution for this state.
        pi = actor_model.actor(obs)
        sampled = pi.sample(seed=key, sample_shape=(int(num_samples),))
        sampled_np = np.asarray(jax.device_get(sampled)).reshape(-1)

        # Reward curve over actions (depends only on delta turn, not absolute orientation).
        delta = grid_actions * env.max_turn_radians
        reward_curve = env.reward_from_angle(delta)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        # Q curve over actions for this starting angle (if critic is available).
        q_curve_np = None
        if fig_q is not None:
            critic_obs_grid = jnp.broadcast_to(critic_obs, (grid_actions.shape[0],) + critic_obs.shape)
            action_grid = grid_actions.reshape(-1, 1)
            q_vals = critic_model.critic(critic_obs_grid, action_grid).squeeze(-1)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        state_title = _tdw_state_title(theta)
        ax_hist = hist_axes[plot_idx]
        ax_hist.hist(
            sampled_np,
            bins=int(num_bins),
            range=(-1.0, 1.0),
            density=True,
            alpha=0.45,
            color="tab:blue",
        )
        _tdw_apply_axis_style(ax_hist, xlabel="action", ylabel="density", title=state_title)
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.6)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward", fontsize=12, color="tab:red")
        ax_hist2.tick_params(axis="y", labelsize=10, colors="tab:red")
        ax_hist2.spines["right"].set_color("tab:red")

        if fig_q is not None and q_curve_np is not None:
            ax_q = q_axes[plot_idx]
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.4)
            _tdw_apply_axis_style(ax_q, xlabel="action", ylabel="Q", title=state_title)

    for ax in hist_axes[len(angles_np) :]:
        ax.axis("off")
    fig_hist.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.45, label="Policy action samples (histogram)"),
            Line2D([0], [0], color="tab:red", linewidth=2.6, label="Reward curve"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig_hist.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    hist_out_path, q_out_path = _resolve_tdw_analysis_out_pair(
        checkpoint_path=checkpoint_path, out_path=out_path
    )
    os.makedirs(os.path.dirname(hist_out_path) or ".", exist_ok=True)
    fig_hist.savefig(hist_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    logging.info("Saved TurningDoubleWell action histogram analysis to %s", hist_out_path)

    if fig_q is not None:
        for ax in q_axes[len(angles_np) :]:
            ax.axis("off")
        fig_q.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig_q.savefig(q_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig_q)
        logging.info("Saved TurningDoubleWell Q-function analysis to %s", q_out_path)


def _tdw_action_analysis_dmerl(
    *,
    cfg,
    checkpoint_path: str,
    actor_model,
    critic_model,
    norm_state,
    method_name: str = "reppo_DMERL_new",
    train_mode: str | None = None,
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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
    method_display = _method_display_name(method_name, train_mode=train_mode)

    nrows, ncols = _tdw_subplot_grid(len(angles_np), max_cols=4)
    fig_hist, axes_hist = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig_hist.suptitle(
        f"{method_display} | Action Samples and Reward Curve",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    hist_axes = axes_hist.reshape(-1)
    fig_q = None
    q_axes = None
    if critic_model is not None:
        fig_q, axes_q = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.0 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        fig_q.suptitle(
            f"{method_display} | Q(s,a) Over Action Grid",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )
        q_axes = axes_q.reshape(-1)

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for plot_idx, (theta, key) in enumerate(zip(angles_np, keys)):
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

        # Reward curve over env actions (depends only on delta turn, not absolute orientation).
        delta = grid_actions * env.max_turn_radians
        reward_curve = env.reward_from_angle(delta)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        # Q curve: evaluate critic at last diffusion step with a neutral "previous action" (zeros).
        q_curve_np = None
        if fig_q is not None:
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

        state_title = _tdw_state_title(theta)
        ax_hist = hist_axes[plot_idx]
        ax_hist.hist(
            sampled_np,
            bins=int(num_bins),
            range=(-1.0, 1.0),
            density=True,
            alpha=0.45,
            color="tab:blue",
        )
        _tdw_apply_axis_style(ax_hist, xlabel="action", ylabel="density", title=state_title)
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.6)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward", fontsize=12, color="tab:red")
        ax_hist2.tick_params(axis="y", labelsize=10, colors="tab:red")
        ax_hist2.spines["right"].set_color("tab:red")

        if fig_q is not None and q_curve_np is not None:
            ax_q = q_axes[plot_idx]
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.4)
            _tdw_apply_axis_style(ax_q, xlabel="env action (tanh(raw))", ylabel="Q", title=state_title)

    for ax in hist_axes[len(angles_np) :]:
        ax.axis("off")
    fig_hist.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.45, label="Policy action samples (histogram)"),
            Line2D([0], [0], color="tab:red", linewidth=2.6, label="Reward curve"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig_hist.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    hist_out_path, q_out_path = _resolve_tdw_analysis_out_pair(
        checkpoint_path=checkpoint_path, out_path=out_path
    )
    os.makedirs(os.path.dirname(hist_out_path) or ".", exist_ok=True)
    fig_hist.savefig(hist_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    logging.info("Saved TurningDoubleWell action histogram analysis to %s", hist_out_path)

    if fig_q is not None:
        for ax in q_axes[len(angles_np) :]:
            ax.axis("off")
        fig_q.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig_q.savefig(q_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig_q)
        logging.info("Saved TurningDoubleWell Q-function analysis to %s", q_out_path)


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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))
    method_display = _method_display_name("reppo_dime")

    nrows, ncols = _tdw_subplot_grid(len(angles_np), max_cols=4)
    fig_hist, axes_hist = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig_hist.suptitle(
        f"{method_display} | Action Samples and Reward Curve",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    hist_axes = axes_hist.reshape(-1)
    fig_q = None
    q_axes = None
    if critic_model is not None:
        fig_q, axes_q = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.0 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        fig_q.suptitle(
            f"{method_display} | Q(s,a) Over Action Grid",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )
        q_axes = axes_q.reshape(-1)

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for plot_idx, (theta, key) in enumerate(zip(angles_np, keys)):
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

        delta = grid_actions * env.max_turn_radians
        reward_curve = env.reward_from_angle(delta)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        q_curve_np = None
        if fig_q is not None:
            critic_obs_grid = jnp.broadcast_to(critic_obs, (grid_actions.shape[0],) + critic_obs.shape)
            action_grid = grid_actions.reshape(-1, 1)
            q_vals = critic_model.critic(critic_obs_grid, action_grid).squeeze(-1)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        state_title = _tdw_state_title(theta)
        ax_hist = hist_axes[plot_idx]
        ax_hist.hist(
            sampled_np,
            bins=int(num_bins),
            range=(-1.0, 1.0),
            density=True,
            alpha=0.45,
            color="tab:blue",
        )
        _tdw_apply_axis_style(ax_hist, xlabel="action", ylabel="density", title=state_title)
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.6)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward", fontsize=12, color="tab:red")
        ax_hist2.tick_params(axis="y", labelsize=10, colors="tab:red")
        ax_hist2.spines["right"].set_color("tab:red")

        if fig_q is not None and q_curve_np is not None:
            ax_q = q_axes[plot_idx]
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.4)
            _tdw_apply_axis_style(ax_q, xlabel="action", ylabel="Q", title=state_title)

    for ax in hist_axes[len(angles_np) :]:
        ax.axis("off")
    fig_hist.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.45, label="Policy action samples (histogram)"),
            Line2D([0], [0], color="tab:red", linewidth=2.6, label="Reward curve"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig_hist.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    hist_out_path, q_out_path = _resolve_tdw_analysis_out_pair(
        checkpoint_path=checkpoint_path, out_path=out_path
    )
    os.makedirs(os.path.dirname(hist_out_path) or ".", exist_ok=True)
    fig_hist.savefig(hist_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    logging.info("Saved TurningDoubleWell action histogram analysis to %s", hist_out_path)

    if fig_q is not None:
        for ax in q_axes[len(angles_np) :]:
            ax.axis("off")
        fig_q.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig_q.savefig(q_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig_q)
        logging.info("Saved TurningDoubleWell Q-function analysis to %s", q_out_path)


def _tdw_action_analysis_diffppo(
    *,
    cfg,
    checkpoint_path: str,
    model,
    norm_state,
    method_name: str = "reppo_DiffPPO",
    train_mode: str | None = None,
    out_path: str | None,
    num_samples: int,
    num_grid: int,
    num_bins: int,
    seed: int,
    max_states: int | None,
    include_q: bool,
    diffusion_sampler: str = "auto",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    env = _tdw_build_env_for_analysis(cfg)
    angles = _tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    diff_steps = int(getattr(model.actor_module, "diff_steps", 1))
    if diff_steps <= 0:
        diff_steps = 1
    last_step = diff_steps - 1
    normalizer = DictNormalizer()
    sampler_mode = str(diffusion_sampler or "auto").lower()
    if sampler_mode not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown diffusion sampler: {sampler_mode}")
    if sampler_mode == "auto":
        sampler_mode = "sde" if str(train_mode or "").upper() == "WPO" else "ode"

    def _normalize_obs_dict(obs_dict):
        if norm_state is None:
            return obs_dict
        return normalizer.normalize(norm_state, obs_dict)

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions))
    ent_coef = float(OmegaConf.select(cfg, "hyperparameters.entropy_coef") or 0.0)
    method_display = _diffppo_plot_title_name(ent_coef)

    nrows, ncols = _tdw_subplot_grid(len(angles_np), max_cols=4)
    fig_hist, axes_hist = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig_hist.suptitle(
        f"{method_display} | Action Samples and Reward Curve",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    hist_axes = axes_hist.reshape(-1)
    fig_q = None
    q_axes = None
    if include_q:
        fig_q, axes_q = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.0 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        fig_q.suptitle(
            f"{method_display} | Q(s,a) Over Action Grid",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )
        q_axes = axes_q.reshape(-1)

    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    for plot_idx, (theta, key) in enumerate(zip(angles_np, keys)):
        theta = jnp.asarray(theta, dtype=jnp.float32)
        raw_obs = _tdw_obs_from_angle(env, theta)

        key, prior_key = jax.random.split(key)
        current_x = model.actor_module.diffusion_model.prior_sampler(
            prior_key, int(num_samples)
        )
        current_x = jnp.asarray(current_x, dtype=jnp.float32)

        chain_key = key
        for step_idx in range(diff_steps):
            chain_key, step_key = jax.random.split(chain_key)
            obs_dict = {
                "orig_obs": jnp.broadcast_to(
                    raw_obs, (current_x.shape[0],) + raw_obs.shape
                ),
                "orig_actions": current_x,
                "normed_actions": current_x,
                "diff_time_step": jnp.full(
                    (current_x.shape[0], 1), step_idx, dtype=jnp.int32
                ),
            }
            obs_dict = _normalize_obs_dict(obs_dict)
            if sampler_mode == "sde":
                current_x, *_ = model.actor_sample_step(obs_dict, step_key)
            else:
                current_x, _ = model.actor_ode_sample_step(obs_dict, step_key)

        sampled_actions = jnp.tanh(current_x)
        sampled_np = np.asarray(jax.device_get(sampled_actions)).reshape(-1)

        delta = grid_actions * env.max_turn_radians
        reward_curve = env.reward_from_angle(delta)
        reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

        q_curve_np = None
        if fig_q is not None:
            raw_action = jnp.arctanh(jnp.clip(grid_actions, -0.999, 0.999)).reshape(-1, 1)
            obs_dict_last = {
                "orig_obs": jnp.broadcast_to(
                    raw_obs, (raw_action.shape[0],) + raw_obs.shape
                ),
                "orig_actions": raw_action,
                "normed_actions": raw_action,
                "diff_time_step": jnp.full(
                    (raw_action.shape[0], 1), last_step, dtype=jnp.int32
                ),
            }
            obs_dict_last = _normalize_obs_dict(obs_dict_last)
            q_vals = model.critic(obs_dict_last)
            q_curve_np = np.asarray(jax.device_get(q_vals)).reshape(-1)

        state_title = _tdw_state_title(theta)
        ax_hist = hist_axes[plot_idx]
        ax_hist.hist(
            sampled_np,
            bins=int(num_bins),
            range=(-1.0, 1.0),
            density=True,
            alpha=0.45,
            color="tab:blue",
        )
        _tdw_apply_axis_style(ax_hist, xlabel="action", ylabel="density", title=state_title)
        ax_hist2 = ax_hist.twinx()
        ax_hist2.plot(grid_actions_np, reward_curve_np, color="tab:red", linewidth=2.6)
        ax_hist2.set_ylim(-0.05, 1.05)
        ax_hist2.set_ylabel("reward", fontsize=12, color="tab:red")
        ax_hist2.tick_params(axis="y", labelsize=10, colors="tab:red")
        ax_hist2.spines["right"].set_color("tab:red")

        if fig_q is not None and q_curve_np is not None:
            ax_q = q_axes[plot_idx]
            ax_q.plot(grid_actions_np, q_curve_np, color="tab:blue", linewidth=2.4)
            _tdw_apply_axis_style(
                ax_q, xlabel="env action (tanh(raw))", ylabel="Q", title=state_title
            )

    for ax in hist_axes[len(angles_np) :]:
        ax.axis("off")
    fig_hist.legend(
        handles=[
            Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.45, label="Policy action samples (histogram)"),
            Line2D([0], [0], color="tab:red", linewidth=2.6, label="Reward curve"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig_hist.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    hist_out_path, q_out_path = _resolve_tdw_analysis_out_pair(
        checkpoint_path=checkpoint_path, out_path=out_path
    )
    os.makedirs(os.path.dirname(hist_out_path) or ".", exist_ok=True)
    fig_hist.savefig(hist_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    logging.info("Saved TurningDoubleWell action histogram analysis to %s", hist_out_path)

    if fig_q is not None:
        for ax in q_axes[len(angles_np) :]:
            ax.axis("off")
        fig_q.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig_q.savefig(q_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig_q)
        logging.info("Saved TurningDoubleWell Q-function analysis to %s", q_out_path)


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
    env_config = _sanitize_env_config(str(cfg.env.name), env_config)
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
    train_mode: str | None = None,
    entropy_coef: float | None = None,
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
    method_display = _method_display_name(
        method_name, train_mode=train_mode, entropy_coef=entropy_coef
    )
    method_lower = str(method_name).lower()
    is_diffppo = method_lower == "reppo_diffppo"
    if is_diffppo:
        method_display = _diffppo_plot_title_name(entropy_coef)
    is_dmerl = ("dmerl" in method_lower and "dime" not in method_lower) or is_diffppo
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
    if is_diffppo:
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
        env = TanhClipAction(env)
        env = LogWrapper(env, int(num_envs))
    elif is_dmerl:
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
        train_mode=train_mode,
        entropy_coef=entropy_coef,
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        diffusion_sampler=diffusion_sampler,
    )

    if is_dmerl and train_state is None:
        raise ValueError("DMERL rendering requires train_state.")
    if train_state is not None:
        if is_diffppo:
            actor_model = nnx.merge(train_state.graphdef, train_state.params)
            critic_model = None
        else:
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
    if is_diffppo:
        obs, critic_obs, env_state = env.reset(init_keys)
    elif normalize_env and norm_state is not None:
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

    if is_diffppo:
        diff_cfg = cfg_render.hyperparameters.diffusion
        diff_steps = int(diff_cfg.diff_steps)
        total_steps = int(horizon) * diff_steps
        mode_upper = str(train_mode or "").upper()
        sampler_mode = sampler
        if sampler_mode == "auto":
            sampler_mode = "sde" if mode_upper == "WPO" else "ode"

        normalizer = DictNormalizer()

        def _norm_obs(obs_dict):
            if (not normalize_env) or norm_state is None:
                return obs_dict
            return normalizer.normalize(norm_state, obs_dict)

        for step_idx in range(total_steps):
            key, act_key, env_key = jax.random.split(key, 3)
            obs_for_actor = _norm_obs(obs)
            if sampler_mode == "sde":
                action, *_ = actor_model.actor_sample_step(obs_for_actor, act_key)
            elif sampler_mode == "ode":
                action, _ = actor_model.actor_ode_sample_step(obs_for_actor, act_key)
            else:
                raise ValueError(f"Unknown diffusion sampler: {sampler_mode}")
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, critic_obs, env_state, _reward, done, info = env.step(
                step_keys, env_state, action
            )
            if (step_idx + 1) % diff_steps == 0:
                states.append(_state_to_numpy(_unwrap_env_state(env_state)))
    elif is_dmerl:
        from src.jaxrl.reppo_helpers.learning_DiffReppo import maybe_add_q_grad

        diff_cfg = cfg_render.hyperparameters.diffusion
        diff_steps = int(diff_cfg.diff_steps)
        total_steps = int(horizon) * diff_steps
        use_langevin = bool(
            OmegaConf.select(cfg_render, "hyperparameters.diffusion.score_model.langevin_param")
            or False
        )
        mode_upper = str(train_mode or OmegaConf.select(cfg_render, "hyperparameters.train_mode") or "").upper()
        sampler_mode = sampler
        if sampler_mode == "auto":
            sampler_mode = "sde" if mode_upper == "WPO" else "ode"

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
            obs, critic_obs, env_state, _reward, done, info = env.step(
                step_keys, env_state, action
            )
            # Only append when the underlying env advanced (once per diffusion loop).
            if (step_idx + 1) % diff_steps == 0:
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
            obs, critic_obs, env_state, _reward, done, info = env.step(
                step_keys, env_state, action
            )
            states.append(_state_to_numpy(_unwrap_env_state(env_state)))

    frames = base_env.render_trajectory(
        states,
        rewards=None,
        width=int(width),
        height=int(height),
        overlay=bool(overlay),
        max_envs=int(num_envs),
        figure_title=method_display,
        title_fontsize=18,
        show_reward=False,
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


def _collect_reppo_trajectories(
    *,
    cfg,
    actor_graphdef,
    actor_params,
    norm_state,
    horizon: int,
    num_envs: int,
    repeats: int,
    seed: int,
    train_mode: str,
) -> dict[str, np.ndarray]:
    _progress(
        f"[reppo] Collecting state trajectories: repeats={int(repeats)}, "
        f"num_envs={int(num_envs)}, horizon={int(horizon)}"
    )
    hp = cfg.hyperparameters
    env = _build_base_env(cfg, horizon=horizon)
    if bool(hp.normalize_env):
        env = LogWrapper(env, int(num_envs))
    env = ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        env = NormalizeVec(env, normalize_reward=bool(hp.normalize_reward))

    actor_model = nnx.merge(actor_graphdef, actor_params)
    obs_runs: list[np.ndarray] = []

    for rep in range(int(repeats)):
        _progress(f"[reppo] Trajectory repeat {rep + 1}/{int(repeats)}")
        key = jax.random.PRNGKey(int(seed) + rep)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, int(num_envs))
        if norm_state is None:
            obs, _critic_obs, env_state = env.reset(init_keys)
        else:
            obs, _critic_obs, env_state = env.reset(init_keys, norm_state)
        rep_obs = [_obs_batch_to_numpy(obs)]

        for _ in range(int(horizon)):
            key, act_key, env_key = jax.random.split(key, 3)
            if str(train_mode).upper() == "WPO":
                act_keys = jax.random.split(act_key, int(num_envs))
                action = jax.vmap(lambda k, o: actor_model.actor(o).sample(seed=k))(
                    act_keys, obs
                )
            else:
                action = actor_model.det_action(obs)
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, _critic_obs, env_state, _reward, _done, _info = env.step(
                step_keys, env_state, action
            )
            rep_obs.append(_obs_batch_to_numpy(obs))

        obs_runs.append(np.stack(rep_obs, axis=0))

    return {
        "state_trajectories": np.stack(obs_runs, axis=0),
    }


def _collect_dime_trajectories(
    *,
    cfg,
    actor_graphdef,
    actor_params,
    norm_state,
    horizon: int,
    num_envs: int,
    repeats: int,
    seed: int,
    diffusion_sampler: str,
) -> dict[str, np.ndarray]:
    _progress(
        f"[reppo_dime] Collecting state trajectories: repeats={int(repeats)}, "
        f"num_envs={int(num_envs)}, horizon={int(horizon)}"
    )
    hp = cfg.hyperparameters
    env = _build_base_env(cfg, horizon=horizon)
    env = LogWrapper(env, int(num_envs))
    env = ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        env = NormalizeVec(env)
    actor_model = nnx.merge(actor_graphdef, actor_params)

    sampler = str(diffusion_sampler or "auto").lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    if sampler == "auto":
        sampler = "sde"

    obs_runs: list[np.ndarray] = []

    for rep in range(int(repeats)):
        _progress(f"[reppo_dime] Trajectory repeat {rep + 1}/{int(repeats)}")
        key = jax.random.PRNGKey(int(seed) + rep)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, int(num_envs))
        if norm_state is None:
            obs, _critic_obs, env_state = env.reset(init_keys)
        else:
            obs, _critic_obs, env_state = env.reset(init_keys, norm_state)

        rep_obs = [_obs_batch_to_numpy(obs)]

        for _ in range(int(horizon)):
            key, act_key, env_key = jax.random.split(key, 3)
            if sampler == "ode":
                action, *_ = actor_model.det_action(
                    act_key, obs, ode=True, ode_coef=1.0
                )
            else:
                action, *_ = actor_model.sample(act_key, obs)
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, _critic_obs, env_state, _reward, _done, _info = env.step(
                step_keys, env_state, action
            )
            rep_obs.append(_obs_batch_to_numpy(obs))

        obs_runs.append(np.stack(rep_obs, axis=0))

    return {
        "state_trajectories": np.stack(obs_runs, axis=0),
    }


def _collect_dmerl_trajectories(
    *,
    cfg,
    train_state,
    norm_state,
    horizon: int,
    num_envs: int,
    repeats: int,
    seed: int,
    diffusion_sampler: str,
    train_mode: str,
) -> dict[str, np.ndarray]:
    from src.jaxrl.reppo_helpers.learning_DiffReppo import maybe_add_q_grad

    _progress(
        f"[reppo_DMERL_new] Collecting state trajectories: repeats={int(repeats)}, "
        f"num_envs={int(num_envs)}, horizon={int(horizon)}"
    )
    hp = cfg.hyperparameters
    diff_cfg = hp.diffusion
    env_action_clip_value = float(hp.env_action_clip_value)
    base_env = _build_base_env(cfg, horizon=horizon)
    env = MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )
    env = LogWrapper(env, int(num_envs))
    if bool(hp.normalize_env):
        env = DiffNormalizeVec(
            env,
            normalize_reward=bool(hp.normalize_reward),
            num_diff_steps=int(diff_cfg.diff_steps),
        )
    actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
    critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
    use_langevin = bool(
        OmegaConf.select(cfg, "hyperparameters.diffusion.score_model.langevin_param")
        or False
    )
    sampler = str(diffusion_sampler or "auto").lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    if sampler == "auto":
        sampler = "sde" if str(train_mode or "").upper() == "WPO" else "ode"

    diff_steps = int(diff_cfg.diff_steps)
    total_steps = int(horizon) * diff_steps
    obs_runs: list[np.ndarray] = []

    for rep in range(int(repeats)):
        _progress(f"[reppo_DMERL_new] Trajectory repeat {rep + 1}/{int(repeats)}")
        key = jax.random.PRNGKey(int(seed) + rep)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, int(num_envs))
        if norm_state is not None:
            obs, critic_obs, env_state = env.reset(init_keys, norm_state)
        else:
            obs, critic_obs, env_state = env.reset(init_keys)

        init_raw = _diff_raw_obs_from_state(env_state)
        rep_obs = [init_raw if init_raw is not None else _obs_batch_to_numpy(obs)]

        for step_idx in range(total_steps):
            key, act_key, env_key = jax.random.split(key, 3)
            obs_for_actor = maybe_add_q_grad(
                obs, critic_obs, actor_model, critic_model, use_langevin
            )
            if sampler == "sde":
                action, *_ = actor_model.vmap_sample_next_step(obs_for_actor, act_key)
            else:
                action, _ = actor_model.vmap_ode_sample_next_step(obs_for_actor, act_key)
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, critic_obs, env_state, _reward, _done, _info = env.step(
                step_keys, env_state, action
            )
            if (step_idx + 1) % diff_steps == 0:
                step_raw = _diff_raw_obs_from_state(env_state)
                rep_obs.append(
                    step_raw if step_raw is not None else _obs_batch_to_numpy(obs)
                )

        obs_runs.append(np.stack(rep_obs, axis=0))

    return {
        "state_trajectories": np.stack(obs_runs, axis=0),
    }


def _collect_diffppo_trajectories(
    *,
    cfg,
    train_state,
    horizon: int,
    num_envs: int,
    repeats: int,
    seed: int,
    diffusion_sampler: str,
) -> dict[str, np.ndarray]:
    _progress(
        f"[reppo_DiffPPO] Collecting state trajectories: repeats={int(repeats)}, "
        f"num_envs={int(num_envs)}, horizon={int(horizon)}"
    )
    hp = cfg.hyperparameters
    diff_cfg = hp.diffusion
    env_action_clip_value = float(hp.env_action_clip_value)
    base_env = _build_base_env(cfg, horizon=horizon)
    env = MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )
    env = TanhClipAction(env)
    env = LogWrapper(env, int(num_envs))
    model = nnx.merge(train_state.graphdef, train_state.params)
    normalizer = DictNormalizer()

    sampler = str(diffusion_sampler or "auto").lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    if sampler == "auto":
        sampler = "sde"

    diff_steps = int(diff_cfg.diff_steps)
    total_steps = int(horizon) * diff_steps
    norm_state = train_state.normalization_state
    obs_runs: list[np.ndarray] = []

    for rep in range(int(repeats)):
        _progress(f"[reppo_DiffPPO] Trajectory repeat {rep + 1}/{int(repeats)}")
        key = jax.random.PRNGKey(int(seed) + rep)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, int(num_envs))
        obs, critic_obs, env_state = env.reset(init_keys)

        init_raw = _diff_raw_obs_from_state(env_state)
        rep_obs = [init_raw if init_raw is not None else _obs_batch_to_numpy(obs)]

        for step_idx in range(total_steps):
            key, act_key, env_key = jax.random.split(key, 3)
            obs_for_actor = (
                normalizer.normalize(norm_state, obs) if norm_state is not None else obs
            )
            if sampler == "ode":
                action, _ = model.actor_ode_sample_step(obs_for_actor, act_key)
            else:
                action, *_ = model.actor_sample_step(obs_for_actor, act_key)
            step_keys = jax.random.split(env_key, int(num_envs))
            obs, critic_obs, env_state, _reward, _done, _info = env.step(
                step_keys, env_state, action
            )
            if (step_idx + 1) % diff_steps == 0:
                step_raw = _diff_raw_obs_from_state(env_state)
                rep_obs.append(
                    step_raw if step_raw is not None else _obs_batch_to_numpy(obs)
                )

        obs_runs.append(np.stack(rep_obs, axis=0))

    return {
        "state_trajectories": np.stack(obs_runs, axis=0),
    }


def _persist_trajectory_metrics(
    *,
    trajectories: dict[str, np.ndarray],
    args,
    cfg,
    checkpoint_path: str,
    method_name: str,
    train_mode: str | None,
    horizon: int,
    num_envs: int,
    repeats: int,
    diffusion_sampler: str,
) -> dict[str, Any]:
    knn_entropy = None
    if not bool(getattr(args, "traj_no_knn_entropy", False)):
        _progress("Computing kNN entropy on saved state trajectories")
        knn_entropy = _compute_trajectory_knn_entropy(trajectories, args)
        if "knn_trajectories_entropy_nats" in knn_entropy:
            logging.info(
                "kNN entropy (trajectory-points): %.6f nats (k=%d, samples=%d, dim=%d)",
                knn_entropy["knn_trajectories_entropy_nats"],
                int(knn_entropy["knn_trajectories_k"]),
                int(knn_entropy["knn_trajectories_num_samples"]),
                int(knn_entropy["knn_trajectories_dim"]),
            )
        if "knn_states_entropy_nats" in knn_entropy:
            logging.info(
                "kNN entropy (state-points): %.6f nats (k=%d, samples=%d, dim=%d)",
                knn_entropy["knn_states_entropy_nats"],
                int(knn_entropy["knn_states_k"]),
                int(knn_entropy["knn_states_num_samples"]),
                int(knn_entropy["knn_states_dim"]),
            )

    out_dir = _collect_output_dir(
        checkpoint_path=checkpoint_path,
        method_name=method_name,
        num_envs=num_envs,
        repeats=repeats,
        user_out=getattr(args, "traj_out", None),
    )
    _progress(f"Saving trajectory bundle to {out_dir}")
    out_paths = _save_trajectory_bundle(
        out_dir=out_dir,
        trajectories=trajectories,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        method_name=method_name,
        train_mode=train_mode,
        horizon=horizon,
        num_envs=num_envs,
        repeats=repeats,
        diffusion_sampler=diffusion_sampler,
        knn_entropy=knn_entropy,
    )
    logging.info("Saved trajectories to %s", out_paths["trajectory_data_path"])
    logging.info("Saved trajectory metadata to %s", out_paths["trajectory_metadata_path"])

    result: dict[str, Any] = dict(out_paths)
    if knn_entropy is not None:
        result.update(knn_entropy)
    return result


def _eval_reppo(checkpoint: dict[str, Any], cfg, args) -> dict[str, Any]:
    _progress("[reppo] Preparing evaluation")
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
    method_name = str(checkpoint.get("method_name", "reppo"))
    train_mode = _resolve_train_mode(checkpoint, cfg)

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        actor_model = nnx.merge(actor_graphdef, actor_params)
        critic_model = None
        if bool(getattr(args, "tdw_action_analysis_q", False)):
            critic_params = checkpoint.get("critic_params", None)
        else:
            critic_params = None
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
            method_name=method_name,
            train_mode=train_mode,
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
            method_name=method_name,
            train_mode=train_mode,
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

    _progress("[reppo] Running evaluation rollout")
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
    _progress("[reppo] Evaluation rollout complete")
    metrics: dict[str, Any] = {
        "episode_return_mean": float(np.asarray(mean_return)),
        "episode_return_std": float(np.asarray(std_return)),
        "horizon": float(horizon),
        "num_envs": float(num_envs),
    }
    if bool(getattr(args, "collect_trajectories", False)):
        traj_num_envs = (
            int(getattr(args, "traj_num_envs"))
            if getattr(args, "traj_num_envs", None) is not None
            else int(hp.num_envs)
        )
        traj_repeats = max(1, int(getattr(args, "traj_repeats", 1)))
        cfg_collect = (
            _cfg_with_num_envs(cfg, traj_num_envs)
            if traj_num_envs != int(hp.num_envs)
            else cfg
        )
        _progress(
            f"[reppo] Starting trajectory collection (X={traj_num_envs}, Y={traj_repeats})"
        )
        traj = _collect_reppo_trajectories(
            cfg=cfg_collect,
            actor_graphdef=actor_graphdef,
            actor_params=actor_params,
            norm_state=norm_state,
            horizon=horizon,
            num_envs=traj_num_envs,
            repeats=traj_repeats,
            seed=int(getattr(args, "traj_seed", 0)),
            train_mode=str(train_mode or getattr(hp, "train_mode", "reparam")),
        )
        metrics.update(
            _persist_trajectory_metrics(
                trajectories=traj,
                args=args,
                cfg=cfg_collect,
                checkpoint_path=str(args.checkpoint),
                method_name=method_name,
                train_mode=train_mode,
                horizon=horizon,
                num_envs=traj_num_envs,
                repeats=traj_repeats,
                diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
            )
        )
    return metrics


def _eval_reppo_dmerl_new(checkpoint: dict[str, Any], cfg, args) -> dict[str, Any]:
    # Lazy import: DMERL pulls in extra modules.
    from src.jaxrl.reppo_DMERL_new import ReppoDMERLTrainer, ReppoConfig

    _progress("[reppo_DMERL_new] Preparing evaluation")
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
    method_name = str(checkpoint.get("method_name", "reppo_DMERL_new"))
    train_mode = _resolve_train_mode(checkpoint, cfg)

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = None
        if bool(getattr(args, "tdw_action_analysis_q", False)):
            critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        _tdw_action_analysis_dmerl(
            cfg=cfg,
            checkpoint_path=str(args.checkpoint),
            actor_model=actor_model,
            critic_model=critic_model,
            norm_state=norm_state,
            method_name=method_name,
            train_mode=train_mode,
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
            method_name=method_name,
            train_mode=train_mode,
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

    _progress("[reppo_DMERL_new] Running evaluation metrics")
    eval_key = jax.random.PRNGKey(123)
    sampler = str(getattr(args, "diffusion_sampler", "auto")).lower()
    if sampler == "auto":
        eval_fn = trainer.eval_fn
    elif sampler == "sde":
        eval_fn = trainer._make_sde_eval_fn(eval_policy=(str(train_mode).upper() == "WPO"))
    elif sampler == "ode":
        eval_fn = trainer._make_ode_eval_fn()
    else:
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")

    metrics = eval_fn(eval_key, train_state, norm_state)
    metrics = jax.tree.map(lambda x: float(np.asarray(x)), metrics)
    _progress("[reppo_DMERL_new] Evaluation metrics complete")
    if bool(getattr(args, "collect_trajectories", False)):
        traj_num_envs = (
            int(getattr(args, "traj_num_envs"))
            if getattr(args, "traj_num_envs", None) is not None
            else int(cfg.hyperparameters.num_envs)
        )
        traj_repeats = max(1, int(getattr(args, "traj_repeats", 1)))
        cfg_collect = (
            _cfg_with_num_envs(cfg, traj_num_envs)
            if traj_num_envs != int(cfg.hyperparameters.num_envs)
            else cfg
        )
        _progress(
            f"[reppo_DMERL_new] Starting trajectory collection (X={traj_num_envs}, Y={traj_repeats})"
        )
        traj = _collect_dmerl_trajectories(
            cfg=cfg_collect,
            train_state=train_state,
            norm_state=norm_state,
            horizon=horizon,
            num_envs=traj_num_envs,
            repeats=traj_repeats,
            seed=int(getattr(args, "traj_seed", 0)),
            diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
            train_mode=train_mode,
        )
        metrics.update(
            _persist_trajectory_metrics(
                trajectories=traj,
                args=args,
                cfg=cfg_collect,
                checkpoint_path=str(args.checkpoint),
                method_name=method_name,
                train_mode=train_mode,
                horizon=horizon,
                num_envs=traj_num_envs,
                repeats=traj_repeats,
                diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
            )
        )
    return metrics


def _eval_reppo_diffppo(checkpoint: dict[str, Any], cfg, args) -> dict[str, Any]:
    # Lazy import: DiffPPO pulls in extra modules.
    from src.jaxrl.reppo_DiffPPO import PPOConfig, ReppoPPOTrainer

    _progress("[reppo_DiffPPO] Preparing evaluation")
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    base_env = _build_base_env(cfg, horizon=horizon)
    diff_cfg = hp.diffusion
    env_action_clip_value = float(hp.env_action_clip_value)
    env = MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )

    trainer = ReppoPPOTrainer(
        cfg=PPOConfig(**cfg.hyperparameters),
        env=env,
        num_seeds=1,
    )
    init_fn = trainer._make_init_fn()
    key = jax.random.PRNGKey(0)
    train_state = init_fn(key)

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    params = checkpoint.get("params", None)
    if params is None:
        raise ValueError(
            "DiffPPO checkpoint is missing `params`. Re-train with the updated "
            "`src/jaxrl/reppo_DiffPPO.py` checkpoint exporter."
        )
    params = _select_seed(params, seed_idx, num_seeds)

    norm_state = checkpoint.get("normalization_state", None)
    critic_norm_state = checkpoint.get("critic_normalization_state", None)
    reward_norm_state = checkpoint.get("reward_normalization_state", None)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, seed_idx, num_seeds)
    if critic_norm_state is not None:
        critic_norm_state = _select_seed(critic_norm_state, seed_idx, num_seeds)
    if reward_norm_state is not None:
        reward_norm_state = _select_seed(reward_norm_state, seed_idx, num_seeds)

    params = _to_jax_tree(params)
    if norm_state is not None:
        norm_state = _to_jax_tree(norm_state)
    if critic_norm_state is not None:
        critic_norm_state = _to_jax_tree(critic_norm_state)
    if reward_norm_state is not None:
        reward_norm_state = _to_jax_tree(reward_norm_state)

    train_state = train_state.replace(
        params=params,
        normalization_state=norm_state if bool(getattr(hp, "normalize_env", False)) else None,
        critic_normalization_state=(
            critic_norm_state if bool(getattr(hp, "normalize_env", False)) else None
        ),
        reward_normalization_state=(
            reward_norm_state
            if bool(getattr(hp, "normalize_reward", False))
            or bool(getattr(hp, "normalize_soft_reward", False))
            else None
        ),
    )

    method_name = str(checkpoint.get("method_name", "reppo_DiffPPO"))
    train_mode = _resolve_train_mode(checkpoint, cfg)

    if bool(getattr(args, "tdw_action_analysis", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        model = nnx.merge(train_state.graphdef, train_state.params)
        _tdw_action_analysis_diffppo(
            cfg=cfg,
            checkpoint_path=str(args.checkpoint),
            model=model,
            norm_state=train_state.normalization_state,
            method_name=method_name,
            train_mode=train_mode,
            out_path=getattr(args, "tdw_action_analysis_out", None),
            num_samples=int(getattr(args, "tdw_action_analysis_samples", 4096)),
            num_grid=int(getattr(args, "tdw_action_analysis_grid", 401)),
            num_bins=int(getattr(args, "tdw_action_analysis_bins", 60)),
            seed=int(getattr(args, "tdw_action_analysis_seed", 0)),
            max_states=getattr(args, "tdw_action_analysis_max_states", None),
            include_q=bool(getattr(args, "tdw_action_analysis_q", False)),
            diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
        )

    if bool(getattr(args, "render", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        render_num_envs = max(1, int(getattr(args, "render_num_envs", 10)))
        _render_turning_double_well_reppo(
            method_name=method_name,
            train_mode=train_mode,
            entropy_coef=float(getattr(hp, "entropy_coef", 0.0)),
            diffusion_sampler=getattr(args, "diffusion_sampler", "auto"),
            cfg=cfg,
            train_state=train_state,
            checkpoint_path=str(args.checkpoint),
            norm_state=train_state.normalization_state,
            horizon=horizon,
            num_envs=render_num_envs,
            out_path=getattr(args, "render_out", None),
            overlay=not bool(getattr(args, "render_grid", False)),
            width=int(getattr(args, "render_width", 960)),
            height=int(getattr(args, "render_height", 720)),
            fps=int(getattr(args, "render_fps", 20)),
            seed=int(getattr(args, "render_seed", 0)),
        )

    _progress("[reppo_DiffPPO] Running evaluation metrics")
    sampler = str(getattr(args, "diffusion_sampler", "auto")).lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    if sampler in ("auto", "sde"):
        policy = trainer._make_eval_policy(train_state)
    else:
        normalizer = trainer.normalizer

        def policy(key, obs, state=None):
            del state
            if train_state.normalization_state is not None:
                obs = normalizer.normalize(train_state.normalization_state, obs)
            model = nnx.merge(train_state.graphdef, train_state.params)
            action, _ = model.actor_ode_sample_step(obs, key)
            return action, dict(log_prob=None, value=None)

    eval_key = jax.random.PRNGKey(123)
    metrics = trainer.eval_fn(eval_key, policy)
    metrics = jax.tree.map(lambda x: float(np.asarray(x)), metrics)
    _progress("[reppo_DiffPPO] Evaluation metrics complete")
    if bool(getattr(args, "collect_trajectories", False)):
        traj_num_envs = (
            int(getattr(args, "traj_num_envs"))
            if getattr(args, "traj_num_envs", None) is not None
            else int(cfg.hyperparameters.num_envs)
        )
        traj_repeats = max(1, int(getattr(args, "traj_repeats", 1)))
        cfg_collect = (
            _cfg_with_num_envs(cfg, traj_num_envs)
            if traj_num_envs != int(cfg.hyperparameters.num_envs)
            else cfg
        )
        _progress(
            f"[reppo_DiffPPO] Starting trajectory collection (X={traj_num_envs}, Y={traj_repeats})"
        )
        traj = _collect_diffppo_trajectories(
            cfg=cfg_collect,
            train_state=train_state,
            horizon=horizon,
            num_envs=traj_num_envs,
            repeats=traj_repeats,
            seed=int(getattr(args, "traj_seed", 0)),
            diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
        )
        metrics.update(
            _persist_trajectory_metrics(
                trajectories=traj,
                args=args,
                cfg=cfg_collect,
                checkpoint_path=str(args.checkpoint),
                method_name=method_name,
                train_mode=train_mode,
                horizon=horizon,
                num_envs=traj_num_envs,
                repeats=traj_repeats,
                diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
            )
        )
    return metrics




def _eval_reppo_dime(checkpoint: dict[str, Any], cfg, args) -> dict[str, Any]:
    # DIME policy weights saved from `src/jaxrl/reppo_dime.py`.
    _progress("[reppo_dime] Preparing evaluation")
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
    method_name = str(checkpoint.get("method_name", "reppo_dime"))
    train_mode = _resolve_train_mode(checkpoint, cfg)

    if bool(getattr(args, "render", False)) and str(cfg.env.name) == "TurningDoubleWellEnv":
        render_num_envs = max(1, int(getattr(args, "render_num_envs", 10)))
        _render_turning_double_well_reppo(
            method_name=method_name,
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
        if bool(getattr(args, "tdw_action_analysis_q", False)):
            critic_params = checkpoint.get("critic_params", None)
        else:
            critic_params = None
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

    _progress("[reppo_dime] Running evaluation rollout")
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
    _progress("[reppo_dime] Evaluation rollout complete")
    metrics: dict[str, Any] = {
        "episode_return_mean": float(np.asarray(mean_return)),
        "episode_return_std": float(np.asarray(std_return)),
        "num_episodes": float(np.asarray(num_episodes)),
        "horizon": float(horizon),
        "num_envs": float(num_envs),
    }
    if bool(getattr(args, "collect_trajectories", False)):
        traj_num_envs = (
            int(getattr(args, "traj_num_envs"))
            if getattr(args, "traj_num_envs", None) is not None
            else int(cfg.hyperparameters.num_envs)
        )
        traj_repeats = max(1, int(getattr(args, "traj_repeats", 1)))
        cfg_collect = (
            _cfg_with_num_envs(cfg, traj_num_envs)
            if traj_num_envs != int(cfg.hyperparameters.num_envs)
            else cfg
        )
        _progress(
            f"[reppo_dime] Starting trajectory collection (X={traj_num_envs}, Y={traj_repeats})"
        )
        traj = _collect_dime_trajectories(
            cfg=cfg_collect,
            actor_graphdef=actor_graphdef,
            actor_params=actor_params,
            norm_state=norm_state,
            horizon=horizon,
            num_envs=traj_num_envs,
            repeats=traj_repeats,
            seed=int(getattr(args, "traj_seed", 0)),
            diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
        )
        metrics.update(
            _persist_trajectory_metrics(
                trajectories=traj,
                args=args,
                cfg=cfg_collect,
                checkpoint_path=str(args.checkpoint),
                method_name=method_name,
                train_mode=train_mode,
                horizon=horizon,
                num_envs=traj_num_envs,
                repeats=traj_repeats,
                diffusion_sampler=str(getattr(args, "diffusion_sampler", "auto")),
            )
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model checkpoint (reppo, reppo_DMERL_new, reppo_DiffPPO, or reppo_dime)."
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
            "For diffusion-based methods (reppo_DMERL_new, reppo_DiffPPO, reppo_dime), choose whether actions are sampled via the SDE or ODE path during evaluation (and TurningDoubleWell rendering). 'auto' keeps the method default."
        ),
    )
    parser.add_argument(
        "--collect-trajectories",
        action="store_true",
        help=(
            "Collect and save full rollout trajectories during evaluation into "
            "eval_models/saved_trajectories/."
        ),
    )
    parser.add_argument(
        "--traj-num-envs",
        type=int,
        default=None,
        help=(
            "Number of parallel envs (X) used for trajectory collection. "
            "Defaults to checkpoint cfg hyperparameters.num_envs."
        ),
    )
    parser.add_argument(
        "--traj-repeats",
        type=int,
        default=1,
        help=(
            "Number of repeated trajectory batches (Y). Total trajectories = X * Y."
        ),
    )
    parser.add_argument(
        "--traj-seed",
        type=int,
        default=0,
        help="Base seed for trajectory collection repeats.",
    )
    parser.add_argument(
        "--traj-out",
        type=str,
        default=None,
        help=(
            "Optional output folder name (basename) under eval_models/saved_trajectories/."
        ),
    )
    parser.add_argument(
        "--traj-no-knn-entropy",
        action="store_true",
        help="Disable kNN entropy estimation over the saved trajectory data.",
    )
    parser.add_argument(
        "--traj-knn-mode",
        choices=["both", "trajectories", "states"],
        default="both",
        help=(
            "Compute kNN entropy over full trajectories, pooled states, or both."
        ),
    )
    parser.add_argument(
        "--traj-knn-k",
        type=int,
        default=5,
        help="k in the kNN entropy estimator.",
    )
    parser.add_argument(
        "--traj-knn-max-samples",
        type=int,
        default=0,
        help=(
            "Optional cap on sample count used for kNN entropy. "
            "Set <=0 to use all samples."
        ),
    )
    parser.add_argument(
        "--traj-knn-batch-size",
        type=int,
        default=256,
        help="Batch size used by pairwise-distance chunks in kNN entropy estimation.",
    )

    # TurningDoubleWellEnv rendering (reppo + reppo_DMERL_new + reppo_DiffPPO + reppo_dime).
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
        help="Output GIF name (basename only). The file is always written to artifacts/. If you pass a custom name, it will be prefixed with `REPPO__`, `DME-REPPO__`/`DME-WPO__`, or `REPPO-DIME__` if missing.",
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
            "(2) optional Q(s,a) over the action grid as a separate plot "
            "(enable with --tdw-action-analysis-q)."
        ),
    )
    parser.add_argument("--tdw-action-analysis-samples", type=int, default=4096, help="Number of action samples per starting heading.")
    parser.add_argument("--tdw-action-analysis-grid", type=int, default=401, help="Number of action grid points for reward/Q curves.")
    parser.add_argument("--tdw-action-analysis-bins", type=int, default=60, help="Histogram bins.")
    parser.add_argument("--tdw-action-analysis-seed", type=int, default=0, help="PRNG seed used for action sampling.")
    parser.add_argument(
        "--tdw-action-analysis-out",
        type=str,
        default=None,
        help=(
            "Histogram PNG basename (written into artifacts/). "
            "When --tdw-action-analysis-q is enabled, Q-curve output uses the same basename with __q suffix."
        ),
    )
    parser.add_argument("--tdw-action-analysis-max-states", type=int, default=None, help="Optional cap on number of starting headings to plot (useful when well_angle_deg is small).")
    parser.add_argument(
        "--tdw-action-analysis-q",
        action="store_true",
        help="Also evaluate and plot Q(s,a) over the action grid (disabled by default).",
    )

    args = parser.parse_args()
    if args.traj_repeats < 1:
        raise ValueError("--traj-repeats must be >= 1.")
    if args.traj_num_envs is not None and args.traj_num_envs < 1:
        raise ValueError("--traj-num-envs must be >= 1 when provided.")
    if args.traj_knn_k < 1:
        raise ValueError("--traj-knn-k must be >= 1.")
    if args.traj_knn_batch_size < 1:
        raise ValueError("--traj-knn-batch-size must be >= 1.")
    if args.traj_knn_max_samples is not None and args.traj_knn_max_samples <= 0:
        args.traj_knn_max_samples = None

    ckpt_path = os.path.expanduser(args.checkpoint)
    _progress(f"Loading checkpoint from {ckpt_path}")
    checkpoint = _load_checkpoint(ckpt_path)
    _progress("Checkpoint loaded")
    checkpoint["seed_idx"] = int(args.seed_idx)

    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing cfg; cannot reconstruct env/model.")
    cfg = OmegaConf.create(cfg_dict)
    _progress("Checkpoint config reconstructed")

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
            if str(OmegaConf.select(cfg, "name") or "").lower() == "diff_ppo":
                method_name = "reppo_DiffPPO"
            elif OmegaConf.select(cfg, "hyperparameters.temperature_lagragian_lr") is not None:
                method_name = "reppo_dime"
            else:
                method_name = "reppo_DMERL_new"
        else:
            method_name = "reppo"
    method_name = str(method_name)
    method_name_lower = method_name.lower()
    _progress(f"Detected method: {method_name}")
    _progress(
        f"Effective env={cfg.env.name} type={cfg.env.type} horizon={cfg.env.max_episode_steps}"
    )

    if method_name_lower == "reppo_dmerl_new":
        _progress("Dispatching to reppo_DMERL_new evaluator")
        metrics = _eval_reppo_dmerl_new(checkpoint, cfg, args)
    elif method_name_lower == "reppo_diffppo":
        _progress("Dispatching to reppo_DiffPPO evaluator")
        metrics = _eval_reppo_diffppo(checkpoint, cfg, args)
    elif "dime" in method_name_lower:
        _progress("Dispatching to reppo_dime evaluator")
        metrics = _eval_reppo_dime(checkpoint, cfg, args)
    else:
        _progress("Dispatching to reppo evaluator")
        metrics = _eval_reppo(checkpoint, cfg, args)

    _progress("Evaluation complete, printing metrics")
    logging.info("Evaluation metrics:")
    for k, v in metrics.items():
        logging.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
