import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import distrax
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
from src.networks.diffusion.models import ControlNetwork
from src.networks.jax_models import (
    DIMEActor,
    DiffusionModel as DIMEDiffusionModel,
    SACActorNetworks,
    logratio_DIME,
    ode_integrator as ode_integrator_DIME,
    sde_integrator as sde_integrator_DIME,
)
from src.networks.jax_models_DMERL import (
    DMERLActor,
    DiffusionModel as DMERLDiffusionModel,
    logratio as logratio_DMERL,
    ode_integrator as ode_integrator_DMERL,
    sde_integrator as sde_integrator_DMERL,
)

logging.basicConfig(level=logging.INFO)


def _artifacts_dir() -> str:
    path = os.path.join(_REPO_ROOT, "artifacts", "partition_sum")
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned or "run"


def _default_out_paths(checkpoint_path: str) -> tuple[str, str, str]:
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{_sanitize_name(ckpt_stem)}__partition_sum__{ts}"
    out_dir = _artifacts_dir()
    return (
        os.path.join(out_dir, f"{stem}.json"),
        os.path.join(out_dir, f"{stem}.npz"),
        os.path.join(out_dir, f"{stem}.png"),
    )


def _load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "--:--"
    seconds_int = int(round(seconds))
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _print_progress_line(desc: str, done: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    done_clamped = max(0, min(int(done), int(total)))
    frac = float(done_clamped) / float(total)
    bar_width = 30
    filled = int(round(frac * bar_width))
    filled = max(0, min(filled, bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = max(0.0, time.perf_counter() - start_time)
    rate = float(done_clamped) / elapsed if elapsed > 0.0 else 0.0
    remaining = max(int(total) - done_clamped, 0)
    eta = float(remaining) / rate if rate > 0.0 else float("inf")
    line = (
        f"\r{desc} [{bar}] {done_clamped}/{total} "
        f"({100.0 * frac:5.1f}%) elapsed {_format_duration(elapsed)} "
        f"eta {_format_duration(eta)}"
    )
    print(line, end="", file=sys.stderr, flush=True)
    if done_clamped >= int(total):
        print(file=sys.stderr, flush=True)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _select_seed(tree, seed_idx: int, num_seeds: int):
    def _maybe_index(x):
        x = np.asarray(x)
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


def _resolve_train_mode(checkpoint: dict[str, Any], cfg) -> str:
    ckpt_mode = checkpoint.get("train_mode", None)
    if ckpt_mode is not None and str(ckpt_mode).strip():
        return str(ckpt_mode)
    cfg_mode = OmegaConf.select(cfg, "hyperparameters.train_mode")
    if cfg_mode is not None and str(cfg_mode).strip():
        return str(cfg_mode)
    return "reparam"


def _build_dime_actor_template(hp, obs_dim: int, action_dim: int, *, actor_key: jax.Array):
    diffusion_cfg = getattr(hp, "diffusion", None)
    if diffusion_cfg is None:
        raise ValueError("DIME checkpoint is missing hyperparameters.diffusion config.")

    try:
        import hydra
    except Exception as exc:
        raise ImportError(
            "hydra is required to instantiate DIME diffusion dt_schedule."
        ) from exc

    dt_schedule_cfg = getattr(diffusion_cfg, "dt_schedule", None)
    if dt_schedule_cfg is None:
        raise ValueError("DIME diffusion config is missing dt_schedule.")
    dt_schedule = hydra.utils.call(dt_schedule_cfg)

    score_cfg = diffusion_cfg.score_model
    model_rngs = nnx.Rngs(actor_key)

    if bool(getattr(diffusion_cfg, "learn_forward", False)):
        forward_model: nnx.Module | None = ControlNetwork(
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
            max_time=float(diffusion_cfg.diff_steps),
            rngs=model_rngs,
        )
    else:
        forward_model = None

    if bool(getattr(diffusion_cfg, "learn_backward", False)):
        backward_model: nnx.Module | None = ControlNetwork(
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
            max_time=float(diffusion_cfg.diff_steps),
            rngs=model_rngs,
        )
    else:
        backward_model = None

    diffusion_model = DIMEDiffusionModel(
        action_dim=action_dim,
        observation_dim=obs_dim,
        fwd_model=forward_model,
        bwd_model=backward_model,
        diff_steps=int(diffusion_cfg.diff_steps),
        init_std=float(diffusion_cfg.init_std),
        friction=float(diffusion_cfg.friction),
        per_dim_friction=bool(diffusion_cfg.per_dim_friction),
        dt=float(diffusion_cfg.dt),
        learn_dt=bool(diffusion_cfg.learn_dt),
        per_step_dt=bool(diffusion_cfg.per_step_dt),
        learn_prior=bool(diffusion_cfg.learn_prior),
        learn_betas=bool(diffusion_cfg.learn_betas),
        learn_friction=bool(diffusion_cfg.learn_friction),
        learn_mass_matrix=bool(diffusion_cfg.learn_mass_matrix),
        dt_schedule=dt_schedule,
        rngs=model_rngs,
    )

    return DIMEActor(
        action_dim=action_dim,
        observation_dim=obs_dim,
        diffusion_model=diffusion_model,
        sde_integrator=sde_integrator_DIME,
        ode_integrator=ode_integrator_DIME,
        logratio=logratio_DIME,
        kl_start=float(getattr(hp, "kl_start", 0.1)),
        ent_start=float(getattr(hp, "ent_start", 1.0)),
    )


def _build_dmerl_actor_template(
    hp,
    obs_dim: int,
    action_dim: int,
    *,
    train_mode: str,
    actor_key: jax.Array,
):
    diffusion_cfg = getattr(hp, "diffusion", None)
    if diffusion_cfg is None:
        raise ValueError("DMERL checkpoint is missing hyperparameters.diffusion config.")

    try:
        import hydra
    except Exception as exc:
        raise ImportError(
            "hydra is required to instantiate DMERL diffusion dt_schedule."
        ) from exc

    dt_schedule_cfg = getattr(diffusion_cfg, "dt_schedule", None)
    if dt_schedule_cfg is None:
        raise ValueError("DMERL diffusion config is missing dt_schedule.")
    dt_schedule = hydra.utils.call(dt_schedule_cfg)

    score_cfg = diffusion_cfg.score_model
    model_rngs = nnx.Rngs(actor_key)
    use_langevin_param = bool(getattr(score_cfg, "langevin_param", False))

    forward_model = None
    if bool(getattr(diffusion_cfg, "learn_forward", False)):
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
            use_langevin_param=use_langevin_param,
            max_time=float(diffusion_cfg.diff_steps),
            rngs=model_rngs,
        )

    backward_model = None
    if bool(getattr(diffusion_cfg, "learn_backward", False)):
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
            use_langevin_param=use_langevin_param,
            max_time=float(diffusion_cfg.diff_steps),
            rngs=model_rngs,
        )

    diffusion_model = DMERLDiffusionModel(
        action_dim=action_dim,
        observation_dim=obs_dim,
        fwd_model=forward_model,
        bwd_model=backward_model,
        diff_steps=int(diffusion_cfg.diff_steps),
        init_std=float(diffusion_cfg.init_std),
        friction=float(diffusion_cfg.friction),
        per_dim_friction=bool(diffusion_cfg.per_dim_friction),
        use_friction_mlp=bool(getattr(diffusion_cfg, "use_friction_mlp", False)),
        friction_mlp_hidden=int(getattr(diffusion_cfg, "friction_mlp_hidden", 64)),
        friction_mlp_layers=int(getattr(diffusion_cfg, "friction_mlp_layers", 2)),
        friction_num_time_hid=int(
            getattr(diffusion_cfg, "friction_num_time_hid", 32)
        ),
        friction_num_time_out=int(
            getattr(diffusion_cfg, "friction_num_time_out", 16)
        ),
        friction_mlp_use_obs=bool(
            getattr(diffusion_cfg, "friction_mlp_use_obs", True)
        ),
        dt=float(diffusion_cfg.dt),
        learn_dt=bool(diffusion_cfg.learn_dt),
        per_step_dt=bool(diffusion_cfg.per_step_dt),
        learn_prior=bool(diffusion_cfg.learn_prior),
        learn_betas=bool(diffusion_cfg.learn_betas),
        learn_friction=bool(diffusion_cfg.learn_friction),
        learn_mass_matrix=bool(diffusion_cfg.learn_mass_matrix),
        langevin_param=use_langevin_param,
        dt_schedule=dt_schedule,
        train_mode=str(train_mode),
        rngs=model_rngs,
    )

    return DMERLActor(
        action_dim=action_dim,
        observation_dim=obs_dim,
        diffusion_model=diffusion_model,
        sde_integrator=sde_integrator_DMERL,
        ode_integrator=ode_integrator_DMERL,
        logratio=logratio_DMERL,
        kl_start=float(getattr(hp, "kl_start", 0.1)),
        ent_start=float(getattr(hp, "ent_start", 0.1)),
        entropy_lagrangian_start=float(
            getattr(
                hp,
                "entropy_lagrangian_start",
                getattr(hp, "ent_start", 0.1),
            )
        ),
        action_clip_value=float(getattr(hp, "action_clip_value", 1.0)),
        tanh_transform=bool(getattr(hp, "tanh_transform", False)),
        use_temp_lagrangian_mlp=bool(getattr(hp, "use_temp_lagrangian_mlp", False)),
        temp_lagrangian_hidden=int(getattr(hp, "temp_lagrangian_hidden", 32)),
        rngs=nnx.Rngs(actor_key),
    )


def _upgrade_actor_params_for_backward_compat(actor_params: Any, hp) -> Any:
    if not isinstance(actor_params, dict):
        return actor_params

    upgraded = dict(actor_params)
    if "entropy_lagrangian_log_param" not in upgraded:
        entropy_start = getattr(hp, "entropy_lagrangian_start", None)
        if entropy_start is None:
            entropy_start = getattr(hp, "ent_start", 1.0)
        entropy_start = float(entropy_start)
        if not np.isfinite(entropy_start) or entropy_start <= 0.0:
            entropy_start = max(float(getattr(hp, "ent_start", 1.0)), 1e-6)
        upgraded["entropy_lagrangian_log_param"] = np.asarray(
            [np.log(entropy_start)], dtype=np.float32
        )
        logging.info(
            "Actor checkpoint missing entropy_lagrangian_log_param; "
            "added compatibility parameter initialized from entropy start."
        )
    return upgraded


def _upgrade_dmerl_actor_params_for_backward_compat(actor_params: Any, hp) -> Any:
    if not isinstance(actor_params, dict):
        return actor_params

    if bool(getattr(hp, "use_temp_lagrangian_mlp", False)):
        return actor_params

    upgraded = dict(actor_params)

    def _log_init(value: float, fallback: float) -> np.ndarray:
        v = float(value)
        if not np.isfinite(v) or v <= 0.0:
            v = float(fallback)
        v = max(v, 1e-6)
        return np.asarray([np.log(v)], dtype=np.float32)

    if "log_temperature" not in upgraded:
        ent_start = float(getattr(hp, "ent_start", 0.1))
        upgraded["log_temperature"] = _log_init(ent_start, 0.1)
        logging.info(
            "DMERL actor checkpoint missing log_temperature; initialized from ent_start."
        )

    if "log_lagrangian" not in upgraded:
        kl_start = float(getattr(hp, "kl_start", 0.1))
        upgraded["log_lagrangian"] = _log_init(kl_start, 0.1)
        logging.info(
            "DMERL actor checkpoint missing log_lagrangian; initialized from kl_start."
        )

    if "log_entropy_lagrangian" not in upgraded:
        ent_lagrangian_start = float(
            getattr(hp, "entropy_lagrangian_start", getattr(hp, "ent_start", 0.1))
        )
        upgraded["log_entropy_lagrangian"] = _log_init(
            ent_lagrangian_start, float(getattr(hp, "ent_start", 0.1))
        )
        logging.info(
            "DMERL actor checkpoint missing log_entropy_lagrangian; "
            "initialized from entropy_lagrangian_start."
        )

    return upgraded


def _logsumexp_np(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return -np.inf
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def _resolve_sample_counts(
    total_samples: int,
    num_envs: int,
    rollout_repeats: int,
    explicit_sample_counts: list[int] | None,
) -> np.ndarray:
    if total_samples < 1:
        raise ValueError("total_samples must be >= 1.")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1.")
    if rollout_repeats < 1:
        raise ValueError("rollout_repeats must be >= 1.")

    if explicit_sample_counts is not None:
        if len(explicit_sample_counts) == 0:
            raise ValueError(
                "--sample-counts was provided but no values were given."
            )
        counts = sorted(set(int(v) for v in explicit_sample_counts))
    else:
        # Default curve: one point per rollout block so the x-axis advances in
        # chunks of num_envs and has exactly rollout_repeats points.
        counts = [int(num_envs) * i for i in range(1, int(rollout_repeats) + 1)]

    counts = [c for c in counts if 1 <= int(c) <= int(total_samples)]
    if not counts:
        raise ValueError(
            "No valid sample counts after filtering; values must be in [1, total_samples]."
        )
    if counts[-1] != int(total_samples):
        counts.append(int(total_samples))
    return np.asarray(counts, dtype=np.int32)


def _compute_logz_curve(
    log_w_flat: np.ndarray,
    sample_counts: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    log_w_values = np.asarray(log_w_flat, dtype=np.float64).reshape(-1)
    total_samples = int(log_w_values.shape[0])
    log_z_sum_curve: list[float] = []
    log_z_mean_curve: list[float] = []
    sample_index_offsets: list[int] = [0]
    sample_indices_chunks: list[np.ndarray] = []
    for n in sample_counts.tolist():
        n_int = int(n)
        if n_int > total_samples:
            raise ValueError(
                f"Sample count {n_int} exceeds available samples {total_samples}."
            )
        if n_int == total_samples:
            sample_idx = np.arange(total_samples, dtype=np.int32)
        else:
            sample_idx = np.asarray(
                rng.choice(total_samples, size=n_int, replace=False),
                dtype=np.int32,
            )
        sample_indices_chunks.append(sample_idx)
        sample_index_offsets.append(sample_index_offsets[-1] + n_int)

        log_w_subsample = log_w_values[sample_idx]
        log_z_sum_n = _logsumexp_np(log_w_subsample)
        log_z_mean_n = float(log_z_sum_n - np.log(float(n_int)))
        log_z_sum_curve.append(log_z_sum_n)
        log_z_mean_curve.append(log_z_mean_n)
    sample_indices_flat = (
        np.concatenate(sample_indices_chunks, axis=0)
        if sample_indices_chunks
        else np.asarray([], dtype=np.int32)
    )
    return {
        "log_z_sum_curve": np.asarray(log_z_sum_curve, dtype=np.float64),
        "log_z_mean_curve": np.asarray(log_z_mean_curve, dtype=np.float64),
        "sample_index_offsets": np.asarray(sample_index_offsets, dtype=np.int32),
        "sample_indices_flat": np.asarray(sample_indices_flat, dtype=np.int32),
    }


def _save_logz_plot(
    sample_counts: np.ndarray,
    log_z_mean_curve: np.ndarray,
    out_path: str,
    *,
    method_name: str,
    env_name: str,
    temperature: float,
    xscale: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        sample_counts,
        log_z_mean_curve,
        linewidth=2.0,
        label=r"$\log Z_{\mathrm{mean}}=\log\frac{1}{N}\sum_i w_i$",
    )
    xscale_mode = str(xscale).lower()
    if xscale_mode not in ("auto", "log", "linear"):
        raise ValueError(f"Unknown plot xscale mode: {xscale}")
    if xscale_mode == "log" or (
        xscale_mode == "auto" and int(sample_counts[-1]) > 50
    ):
        ax.set_xscale("log")
    ax.set_xlabel("Number of sampled trajectories")
    ax.set_ylabel(r"$\log Z_{\mathrm{mean}}$ estimate")
    ax.set_title(
        f"log Z mean estimate vs samples | {method_name} | {env_name} | T={temperature:.6g}"
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _compute_partition_sum(checkpoint: dict[str, Any], cfg, args) -> dict[str, Any]:
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    num_envs = int(hp.num_envs)
    method_name = str(checkpoint.get("method_name", "reppo")).lower()
    train_mode = _resolve_train_mode(checkpoint, cfg)

    if method_name == "reppo_dmerl_new":
        if cfg.env.type != "mjx":
            raise ValueError(
                "reppo_DMERL_new checkpoints are only supported for MJX envs."
            )
        base_env = _build_base_env(cfg, horizon=horizon)
        diff_cfg = hp.diffusion
        env = MjxDiffEnvWrapper(
            base_env,
            num_diff_steps=int(diff_cfg.diff_steps),
            diffusion_config=diff_cfg,
            low=-float(hp.env_action_clip_value),
            high=float(hp.env_action_clip_value),
        )
        env = LogWrapper(env, num_envs)
        if bool(hp.normalize_env):
            env = DiffNormalizeVec(
                env,
                normalize_reward=bool(hp.normalize_reward),
                num_diff_steps=int(diff_cfg.diff_steps),
                update_stats=False,
            )
        obs_dim, _critic_obs_dim = env.get_obs_space_sizes()
        obs_dim = int(obs_dim)
        action_dim = int(env.action_space(None).shape[0])
    else:
        env = _build_base_env(cfg, horizon=horizon)
        env = LogWrapper(env, num_envs)
        env = ClipAction(
            env,
            low=-float(hp.env_action_clip_value),
            high=float(hp.env_action_clip_value),
        )
        if bool(hp.normalize_env):
            env = NormalizeVec(
                env,
                normalize_reward=bool(hp.normalize_reward),
                update_stats=False,
            )
        obs_dim = int(env.observation_space(None)[0].shape[0])
        action_dim = int(env.action_space(None).shape[0])

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(args.seed_idx)
    actor_params = _select_seed(checkpoint["actor_params"], seed_idx=seed_idx, num_seeds=num_seeds)
    if method_name == "reppo":
        actor_params = _upgrade_actor_params_for_backward_compat(actor_params, hp)
    elif method_name == "reppo_dmerl_new":
        actor_params = _upgrade_dmerl_actor_params_for_backward_compat(actor_params, hp)
    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, seed_idx=seed_idx, num_seeds=num_seeds)

    actor_params = _to_jax_tree(actor_params)
    if norm_state is not None:
        norm_state = _to_jax_tree(norm_state)
    if not bool(hp.normalize_env):
        norm_state = None

    actor_key = jax.random.PRNGKey(0)
    if method_name == "reppo":
        actor_template = SACActorNetworks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=int(hp.actor_hidden_dim),
            ent_start=float(hp.ent_start),
            kl_start=float(hp.kl_start),
            use_norm=bool(hp.use_actor_norm),
            layers=int(hp.num_actor_layers),
            min_std=float(getattr(hp, "actor_min_std", 0.05)),
            use_skip=bool(hp.use_actor_skip),
            train_mode=str(train_mode),
            disable_wpo_fisher_preconditioning=bool(
                getattr(hp, "disable_wpo_fisher_preconditioning", False)
            ),
            disable_temperature=bool(getattr(hp, "disable_temperature", False)),
            rngs=nnx.Rngs(actor_key),
        )
    elif method_name == "reppo_dime":
        actor_template = _build_dime_actor_template(
            hp, obs_dim=obs_dim, action_dim=action_dim, actor_key=actor_key
        )
    elif method_name == "reppo_dmerl_new":
        actor_template = _build_dmerl_actor_template(
            hp,
            obs_dim=obs_dim,
            action_dim=action_dim,
            train_mode=train_mode,
            actor_key=actor_key,
        )
    else:
        raise ValueError(
            "compute_partition_sum.py supports method_name in "
            "{'reppo', 'reppo_dime', 'reppo_dmerl_new'}; "
            f"got '{method_name}'."
        )
    actor_graphdef = nnx.graphdef(actor_template)

    clip_value = float(
        min(
            float(getattr(hp, "action_clip_value", 1.0)),
            float(getattr(hp, "env_action_clip_value", 1.0)),
        )
    )
    temp_value = (
        float(args.temperature)
        if args.temperature is not None
        else float(getattr(hp, "ent_start", 1.0))
    )
    if temp_value <= 0.0 or not np.isfinite(temp_value):
        raise ValueError(
            f"Temperature must be finite and > 0, got {temp_value}."
        )
    inv_temp = jnp.asarray(1.0 / temp_value, dtype=jnp.float32)
    rollout_repeats = int(args.rollout_repeats)
    if rollout_repeats < 1:
        raise ValueError("--rollout-repeats must be >= 1.")

    if method_name == "reppo":
        @jax.jit
        def rollout(key: jax.Array, actor_params_jax, norm_state_jax):
            actor_model = nnx.merge(actor_graphdef, actor_params_jax)
            key, init_key = jax.random.split(key)
            init_keys = jax.random.split(init_key, num_envs)
            if norm_state_jax is None:
                obs, _critic_obs, env_state = env.reset(init_keys)
            else:
                obs, _critic_obs, env_state = env.reset(init_keys, norm_state_jax)

            def step_fn(carry, _):
                key, env_state, obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                pi = actor_model.actor(obs)
                action = pi.sample(seed=act_key)
                action = jnp.clip(action, -clip_value, clip_value)
                log_q = pi.log_prob(action).sum(axis=-1)
                zero_term = jnp.zeros_like(log_q)
                step_keys = jax.random.split(env_key, num_envs)
                next_obs, _next_critic_obs, next_state, reward, done, info = env.step(
                    step_keys, env_state, action
                )
                return (key, next_state, next_obs), (reward, log_q, zero_term, zero_term)

            (_, _, _), (rewards, log_q_steps, log_p_minus_log_q_steps, log_prior_steps) = jax.lax.scan(
                step_fn, (key, env_state, obs), xs=None, length=horizon
            )

            returns = rewards.sum(axis=0)
            log_q_traj = log_q_steps.sum(axis=0)
            log_p_traj = inv_temp * returns
            log_w_traj = log_p_traj - log_q_traj
            return (
                rewards,
                log_q_steps,
                log_p_minus_log_q_steps,
                log_prior_steps,
                returns,
                log_q_traj,
                log_p_traj,
                log_w_traj,
            )
    elif method_name == "reppo_dime":
        @jax.jit
        def rollout(key: jax.Array, actor_params_jax, norm_state_jax):
            actor_model = nnx.merge(actor_graphdef, actor_params_jax)
            key, init_key = jax.random.split(key)
            init_keys = jax.random.split(init_key, num_envs)
            if norm_state_jax is None:
                obs, _critic_obs, env_state = env.reset(init_keys)
            else:
                obs, _critic_obs, env_state = env.reset(init_keys, norm_state_jax)

            def step_fn(carry, _):
                key, env_state, obs = carry
                key, act_key, env_key = jax.random.split(key, 3)
                action, _run_cost, _sto_cost, terminal_cost, unscaled_run_cost = actor_model.sample(
                    act_key, obs, stop_grad=True
                )
                action = jnp.clip(action, -clip_value, clip_value)
                log_p_minus_log_q = jnp.asarray(unscaled_run_cost).squeeze(-1)
                log_prior = jnp.asarray(terminal_cost).squeeze(-1)
                # Reuse generic "log_q" bookkeeping so log_w = (1/T)R - log_q_traj
                # matches DIME formula: (1/T)R + sum(log_p-log_q) - sum(log_prior).
                pseudo_log_q = -log_p_minus_log_q + log_prior

                step_keys = jax.random.split(env_key, num_envs)
                next_obs, _next_critic_obs, next_state, reward, done, info = env.step(
                    step_keys, env_state, action
                )
                return (
                    key,
                    next_state,
                    next_obs,
                ), (
                    reward,
                    pseudo_log_q,
                    log_p_minus_log_q,
                    log_prior,
                )

            (_, _, _), (
                rewards,
                log_q_steps,
                log_p_minus_log_q_steps,
                log_prior_steps,
            ) = jax.lax.scan(
                step_fn, (key, env_state, obs), xs=None, length=horizon
            )

            returns = rewards.sum(axis=0)
            log_q_traj = log_q_steps.sum(axis=0)
            log_p_traj = inv_temp * returns
            log_w_traj = log_p_traj - log_q_traj
            return (
                rewards,
                log_q_steps,
                log_p_minus_log_q_steps,
                log_prior_steps,
                returns,
                log_q_traj,
                log_p_traj,
                log_w_traj,
            )
    else:
        use_langevin_param = bool(
            getattr(hp.diffusion.score_model, "langevin_param", False)
        )
        diff_steps = int(hp.diffusion.diff_steps)
        actor_has_tanh_transform = bool(getattr(hp, "tanh_transform", False))

        @jax.jit
        def rollout(key: jax.Array, actor_params_jax, norm_state_jax):
            actor_model = nnx.merge(actor_graphdef, actor_params_jax)
            key, init_key = jax.random.split(key)
            init_keys = jax.random.split(init_key, num_envs)
            if norm_state_jax is None:
                obs, _critic_obs, env_state = env.reset(init_keys)
            else:
                obs, _critic_obs, env_state = env.reset(init_keys, norm_state_jax)

            def step_fn(carry, _):
                key, env_state, obs = carry
                key, act_key, env_key = jax.random.split(key, 3)

                if use_langevin_param:
                    obs_for_actor = dict(obs)
                    obs_for_actor["q_grad"] = jnp.zeros_like(obs["orig_actions"])
                else:
                    obs_for_actor = obs

                action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(
                    obs_for_actor, act_key
                )
                step_idx = jnp.asarray(obs["diff_time_step"]).reshape(-1)
                is_last_step = step_idx >= float(diff_steps - 1)
                if actor_has_tanh_transform:
                    tanh_log_det_correction = jnp.zeros_like(step_idx)
                else:
                    tanh_log_det_correction = jnp.where(
                        is_last_step,
                        distrax.Tanh().forward_log_det_jacobian(action).sum(axis=-1),
                        jnp.zeros_like(step_idx),
                    )
                log_p_minus_log_q = (
                    jnp.asarray(gen_log_prob - dest_log_prob).reshape(-1)
                    - tanh_log_det_correction
                )

                # Prior term appears once per diffusion chain (when diff_time_step == 0).
                is_chain_start = step_idx <= 0.5
                prior_log_prob = jnp.asarray(
                    actor_model.diffusion_model.prior_log_prob(obs["orig_actions"])
                ).reshape(-1)
                log_prior = jnp.where(
                    is_chain_start,
                    prior_log_prob,
                    jnp.zeros_like(prior_log_prob),
                )

                # Keep unified bookkeeping so log_w = (1/T)R - log_q_traj.
                pseudo_log_q = -log_p_minus_log_q + log_prior

                step_keys = jax.random.split(env_key, num_envs)
                next_obs, _next_critic_obs, next_state, reward, done, info = env.step(
                    step_keys, env_state, action
                )
                return (
                    key,
                    next_state,
                    next_obs,
                ), (
                    reward,
                    pseudo_log_q,
                    log_p_minus_log_q,
                    log_prior,
                )

            (_, _, _), (
                rewards,
                log_q_steps,
                log_p_minus_log_q_steps,
                log_prior_steps,
            ) = jax.lax.scan(
                step_fn, (key, env_state, obs), xs=None, length=horizon
            )

            returns = rewards.sum(axis=0)
            log_q_traj = log_q_steps.sum(axis=0)
            log_p_traj = inv_temp * returns
            log_w_traj = log_p_traj - log_q_traj
            return (
                rewards,
                log_q_steps,
                log_p_minus_log_q_steps,
                log_prior_steps,
                returns,
                log_q_traj,
                log_p_traj,
                log_w_traj,
            )

    base_key = jax.random.PRNGKey(int(args.seed))
    rollout_keys = jax.random.split(base_key, rollout_repeats)

    rewards_repeats: list[np.ndarray] = []
    log_q_steps_repeats: list[np.ndarray] = []
    log_p_minus_log_q_steps_repeats: list[np.ndarray] = []
    log_prior_steps_repeats: list[np.ndarray] = []
    returns_repeats: list[np.ndarray] = []
    log_q_traj_repeats: list[np.ndarray] = []
    log_p_traj_repeats: list[np.ndarray] = []
    log_w_traj_repeats: list[np.ndarray] = []

    show_progress = not bool(getattr(args, "no_progress", False))
    progress_start = time.perf_counter()

    for rep_idx, rep_key in enumerate(rollout_keys, start=1):
        (
            rewards,
            log_q_steps,
            log_p_minus_log_q_steps,
            log_prior_steps,
            returns,
            log_q_traj,
            log_p_traj,
            log_w_traj,
        ) = rollout(rep_key, actor_params, norm_state)
        rewards_repeats.append(np.asarray(jax.device_get(rewards), dtype=np.float64))
        log_q_steps_repeats.append(
            np.asarray(jax.device_get(log_q_steps), dtype=np.float64)
        )
        log_p_minus_log_q_steps_repeats.append(
            np.asarray(jax.device_get(log_p_minus_log_q_steps), dtype=np.float64)
        )
        log_prior_steps_repeats.append(
            np.asarray(jax.device_get(log_prior_steps), dtype=np.float64)
        )
        returns_repeats.append(np.asarray(jax.device_get(returns), dtype=np.float64))
        log_q_traj_repeats.append(
            np.asarray(jax.device_get(log_q_traj), dtype=np.float64)
        )
        log_p_traj_repeats.append(
            np.asarray(jax.device_get(log_p_traj), dtype=np.float64)
        )
        log_w_traj_repeats.append(
            np.asarray(jax.device_get(log_w_traj), dtype=np.float64)
        )
        if show_progress:
            _print_progress_line("Rollout repeats", rep_idx, rollout_repeats, progress_start)

    rewards_np = np.stack(rewards_repeats, axis=0)  # [K, H, X]
    log_q_steps_np = np.stack(log_q_steps_repeats, axis=0)  # [K, H, X]
    log_p_minus_log_q_steps_np = np.stack(log_p_minus_log_q_steps_repeats, axis=0)  # [K, H, X]
    log_prior_steps_np = np.stack(log_prior_steps_repeats, axis=0)  # [K, H, X]
    returns_np = np.stack(returns_repeats, axis=0)  # [K, X]
    log_q_traj_np = np.stack(log_q_traj_repeats, axis=0)  # [K, X]
    log_p_traj_np = np.stack(log_p_traj_repeats, axis=0)  # [K, X]
    log_w_traj_np = np.stack(log_w_traj_repeats, axis=0)  # [K, X]
    log_p_minus_log_q_traj_np = log_p_minus_log_q_steps_np.sum(axis=1)  # [K, X]
    log_prior_traj_np = log_prior_steps_np.sum(axis=1)  # [K, X]

    returns_flat = returns_np.reshape(-1)
    log_q_traj_flat = log_q_traj_np.reshape(-1)
    log_p_traj_flat = log_p_traj_np.reshape(-1)
    log_w_traj_flat = log_w_traj_np.reshape(-1)
    log_p_minus_log_q_traj_flat = log_p_minus_log_q_traj_np.reshape(-1)
    log_prior_traj_flat = log_prior_traj_np.reshape(-1)

    total_samples = int(log_w_traj_flat.shape[0])
    curve_seed = (
        int(args.curve_seed) if args.curve_seed is not None else int(args.seed) + 1337
    )
    sample_counts = _resolve_sample_counts(
        total_samples=total_samples,
        num_envs=num_envs,
        rollout_repeats=rollout_repeats,
        explicit_sample_counts=args.sample_counts,
    )
    rng = np.random.default_rng(curve_seed)
    curve = _compute_logz_curve(log_w_traj_flat, sample_counts, rng)
    log_z_sum_curve = curve["log_z_sum_curve"]
    log_z_mean_curve = curve["log_z_mean_curve"]
    sample_index_offsets = curve["sample_index_offsets"]
    sample_indices_flat = curve["sample_indices_flat"]
    log_z_sum = float(log_z_sum_curve[-1])
    log_z_mean = float(log_z_mean_curve[-1])
    normalized_w = np.exp(log_w_traj_flat - log_z_sum)
    ess = float(1.0 / np.sum(np.square(normalized_w)))

    summary = {
        "checkpoint": os.path.abspath(str(args.checkpoint)),
        "method_name": str(checkpoint.get("method_name", "reppo")),
        "env_name": str(cfg.env.name),
        "env_type": str(cfg.env.type),
        "train_mode": str(train_mode),
        "horizon": int(horizon),
        "num_envs": int(num_envs),
        "rollout_repeats": int(rollout_repeats),
        "total_trajectories": int(total_samples),
        "seed_idx": int(seed_idx),
        "seed": int(args.seed),
        "temperature": float(temp_value),
        "inv_temperature": float(1.0 / temp_value),
        "curve_seed": int(curve_seed),
        "sample_shuffle_seed": int(curve_seed),
        "curve_sampling_mode": "independent_without_replacement",
        "weight_formula": (
            "(1/T)*R - log_q_traj"
            if method_name == "reppo"
            else (
                "(1/T)*R + sum(log_p_minus_log_q) - sum(log_prior)"
                if method_name == "reppo_dime"
                else (
                    "(1/T)*R + sum(gen_log_prob - dest_log_prob - "
                    "final_step_tanh_log_det) - sum(log_prior)"
                )
            )
        ),
        "sample_counts": sample_counts.tolist(),
        "log_Z_sum_curve": log_z_sum_curve.tolist(),
        "log_Z_mean_curve": log_z_mean_curve.tolist(),
        "log_Z_sum": float(log_z_sum),
        "log_Z_mean": float(log_z_mean),
        "Z_sum": float(np.exp(log_z_sum)),
        "Z_mean": float(np.exp(log_z_mean)),
        "ess": float(ess),
        "returns_mean": float(np.mean(returns_flat)),
        "returns_std": float(np.std(returns_flat)),
        "log_q_traj_mean": float(np.mean(log_q_traj_flat)),
        "log_q_traj_std": float(np.std(log_q_traj_flat)),
        "log_w_traj_mean": float(np.mean(log_w_traj_flat)),
        "log_w_traj_std": float(np.std(log_w_traj_flat)),
        "log_p_minus_log_q_traj_mean": float(np.mean(log_p_minus_log_q_traj_flat)),
        "log_p_minus_log_q_traj_std": float(np.std(log_p_minus_log_q_traj_flat)),
        "log_prior_traj_mean": float(np.mean(log_prior_traj_flat)),
        "log_prior_traj_std": float(np.std(log_prior_traj_flat)),
        "repeat_log_Z_sum": [
            _logsumexp_np(log_w_traj_np[i].reshape(-1))
            for i in range(log_w_traj_np.shape[0])
        ],
    }

    return {
        "summary": summary,
        "rewards": rewards_np,
        "log_q_steps": log_q_steps_np,
        "log_p_minus_log_q_steps": log_p_minus_log_q_steps_np,
        "log_prior_steps": log_prior_steps_np,
        "returns": returns_np,
        "log_q_traj": log_q_traj_np,
        "log_p_traj": log_p_traj_np,
        "log_w_traj": log_w_traj_np,
        "log_p_minus_log_q_traj": log_p_minus_log_q_traj_np,
        "log_prior_traj": log_prior_traj_np,
        "returns_flat": returns_flat,
        "log_q_traj_flat": log_q_traj_flat,
        "log_p_traj_flat": log_p_traj_flat,
        "log_w_traj_flat": log_w_traj_flat,
        "log_p_minus_log_q_traj_flat": log_p_minus_log_q_traj_flat,
        "log_prior_traj_flat": log_prior_traj_flat,
        "sample_counts": sample_counts,
        "log_Z_sum_curve": log_z_sum_curve,
        "log_Z_mean_curve": log_z_mean_curve,
        "curve_sample_index_offsets": sample_index_offsets,
        "curve_sample_indices_flat": sample_indices_flat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute partition-sum estimate for reppo/reppo_dime/reppo_DMERL_new "
            "checkpoints using trajectory importance weights."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to a reppo, reppo_dime, or reppo_DMERL_new checkpoint "
            "under saved_models/."
        ),
    )
    parser.add_argument(
        "--seed-idx",
        type=int,
        default=0,
        help="Which trained seed index to evaluate when checkpoint stores multiple seeds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="PRNG seed for rollout sampling.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override rollout horizon (defaults to cfg.env.max_episode_steps).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel envs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature T for p ~ exp((1/T) * R). Defaults to cfg.hyperparameters.ent_start.",
    )
    parser.add_argument(
        "--rollout-repeats",
        type=int,
        default=1,
        help=(
            "Repeat the rollout K times. Total sampled trajectories = "
            "num_envs * rollout_repeats."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar for rollout repeats.",
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit sample counts N used to compute log Z curves. "
            "If omitted, defaults to one point per rollout block: "
            "num_envs, 2*num_envs, ..., rollout_repeats*num_envs."
        ),
    )
    parser.add_argument(
        "--curve-seed",
        type=int,
        default=None,
        help=(
            "Seed for independent trajectory subsampling at each log-Z curve point. "
            "Default: seed + 1337."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional output JSON path for summary metrics.",
    )
    parser.add_argument(
        "--out-npz",
        type=str,
        default=None,
        help="Optional output NPZ path for per-step rewards/log-probs and per-trajectory statistics.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Optional output PNG path for log Z vs sample-count curve.",
    )
    parser.add_argument(
        "--plot-xscale",
        choices=("auto", "log", "linear"),
        default="auto",
        help=(
            "X-axis scaling for the log Z plot. "
            "'auto' uses log scale when max sample count > 50."
        ),
    )
    args = parser.parse_args()
    if int(args.rollout_repeats) < 1:
        raise ValueError("--rollout-repeats must be >= 1.")

    ckpt_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    checkpoint = _load_checkpoint(ckpt_path)
    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing cfg; cannot reconstruct env/model.")
    cfg = OmegaConf.create(cfg_dict)

    method_name = str(checkpoint.get("method_name", "reppo")).lower()
    if method_name not in {"reppo", "reppo_dime", "reppo_dmerl_new"}:
        raise ValueError(
            "compute_partition_sum.py currently supports method_name in "
            f"{{'reppo', 'reppo_dime', 'reppo_dmerl_new'}}, got '{method_name}'."
        )

    cfg = _override_cfg(cfg, horizon=args.horizon, num_envs=args.num_envs)

    default_json, default_npz, default_plot = _default_out_paths(ckpt_path)
    out_json = (
        os.path.abspath(os.path.expanduser(args.out_json))
        if args.out_json is not None
        else default_json
    )
    out_npz = (
        os.path.abspath(os.path.expanduser(args.out_npz))
        if args.out_npz is not None
        else default_npz
    )
    out_plot = (
        os.path.abspath(os.path.expanduser(args.out_plot))
        if args.out_plot is not None
        else default_plot
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_plot) or ".", exist_ok=True)

    result = _compute_partition_sum(checkpoint, cfg, args)
    summary = result["summary"]
    summary["plot_path"] = out_plot

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    np.savez_compressed(
        out_npz,
        rewards=result["rewards"],
        log_q_steps=result["log_q_steps"],
        log_p_minus_log_q_steps=result["log_p_minus_log_q_steps"],
        log_prior_steps=result["log_prior_steps"],
        returns=result["returns"],
        log_q_traj=result["log_q_traj"],
        log_p_traj=result["log_p_traj"],
        log_w_traj=result["log_w_traj"],
        log_p_minus_log_q_traj=result["log_p_minus_log_q_traj"],
        log_prior_traj=result["log_prior_traj"],
        returns_flat=result["returns_flat"],
        log_q_traj_flat=result["log_q_traj_flat"],
        log_p_traj_flat=result["log_p_traj_flat"],
        log_w_traj_flat=result["log_w_traj_flat"],
        log_p_minus_log_q_traj_flat=result["log_p_minus_log_q_traj_flat"],
        log_prior_traj_flat=result["log_prior_traj_flat"],
        sample_counts=result["sample_counts"],
        log_Z_sum_curve=result["log_Z_sum_curve"],
        log_Z_mean_curve=result["log_Z_mean_curve"],
        curve_sample_index_offsets=result["curve_sample_index_offsets"],
        curve_sample_indices_flat=result["curve_sample_indices_flat"],
    )
    _save_logz_plot(
        result["sample_counts"],
        result["log_Z_mean_curve"],
        out_plot,
        method_name=str(summary["method_name"]),
        env_name=str(summary["env_name"]),
        temperature=float(summary["temperature"]),
        xscale=str(args.plot_xscale),
    )

    logging.info("Computed partition sum for checkpoint: %s", ckpt_path)
    logging.info("Temperature: %.8f", summary["temperature"])
    logging.info("log_Z_sum: %.8f", summary["log_Z_sum"])
    logging.info("log_Z_mean: %.8f", summary["log_Z_mean"])
    logging.info(
        "ESS: %.4f / %d samples",
        summary["ess"],
        summary["total_trajectories"],
    )
    logging.info("Saved summary JSON: %s", out_json)
    logging.info("Saved trajectory NPZ: %s", out_npz)
    logging.info("Saved log-Z curve plot: %s", out_plot)


if __name__ == "__main__":
    main()
