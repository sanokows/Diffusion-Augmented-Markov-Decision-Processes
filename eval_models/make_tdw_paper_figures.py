#!/usr/bin/env python3
"""Build compact TurningDoubleWellEnv paper figures across methods.

Generates two composite figures with fixed method ordering:
1) Action histograms (1 row x methods): all state histograms overlaid per method,
   with a shared state-color legend above.
2) Trajectory behavior (1 row x methods): static trajectory snapshots per method.

The script reconstructs each policy from checkpoint (reusing eval_saved_model internals),
renders trajectory snapshots, samples state-conditional actions, and composes both rows.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import eval_models.eval_saved_model as esm


@dataclass(frozen=True)
class MethodSpec:
    slot: str
    label: str
    checkpoint: str
    sampler: str


@dataclass
class PreparedMethod:
    spec: MethodSpec
    checkpoint: dict[str, Any]
    cfg: Any
    method_name: str
    train_mode: str
    norm_state: Any
    actor_graphdef: Any | None = None
    actor_params: Any | None = None
    train_state: Any | None = None
    actor_model: Any | None = None
    entropy_coef: float | None = None


def _progress(msg: str) -> None:
    print(f"[make_tdw_paper_figures] {msg}", flush=True)


def _instantiate_hparam_config(config_cls, hp_cfg, *, label: str):
    if OmegaConf.is_config(hp_cfg):
        hp_dict = {k: hp_cfg[k] for k in hp_cfg.keys()}
    else:
        hp_dict = dict(hp_cfg)

    accepted = set(getattr(config_cls, "__dataclass_fields__", {}).keys())
    if not accepted:
        return config_cls(**hp_dict)

    filtered = {k: v for k, v in hp_dict.items() if k in accepted}
    dropped = sorted(k for k in hp_dict.keys() if k not in accepted)
    if dropped:
        logging.warning(
            "Dropping unsupported %s hyperparameter keys: %s",
            label,
            ", ".join(dropped),
        )
    return config_cls(**filtered)


def _detect_method_name(checkpoint: dict[str, Any], cfg) -> str:
    method_name = checkpoint.get("method_name", None)
    if method_name is None:
        if OmegaConf.select(cfg, "hyperparameters.diffusion") is not None:
            if str(OmegaConf.select(cfg, "name") or "").lower() == "diff_ppo":
                method_name = "reppo_DiffPPO"
            elif OmegaConf.select(cfg, "hyperparameters.temperature_lagragian_lr") is not None:
                method_name = "reppo_dime"
            else:
                method_name = "reppo_DMERL_new"
        else:
            method_name = "reppo"
    return str(method_name)


def _load_checkpoint_and_cfg(
    checkpoint_path: str,
    *,
    seed_idx: int,
    horizon: int | None,
    env_config_overrides: list[str],
) -> tuple[dict[str, Any], Any, str, str]:
    ckpt_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = esm._load_checkpoint(ckpt_path)
    checkpoint["seed_idx"] = int(seed_idx)

    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError(f"Checkpoint is missing cfg: {ckpt_path}")
    cfg = OmegaConf.create(cfg_dict)
    cfg = esm._override_cfg(cfg, horizon=horizon, num_envs=None)
    cfg = esm._apply_env_config_overrides(cfg, list(env_config_overrides or []))

    method_name = _detect_method_name(checkpoint, cfg)
    train_mode = esm._resolve_train_mode(checkpoint, cfg)
    return checkpoint, cfg, method_name, str(train_mode or "")


def _require_turning_double_well(cfg, *, checkpoint: str) -> None:
    env_name = str(OmegaConf.select(cfg, "env.name") or "")
    if env_name != "TurningDoubleWellEnv":
        raise ValueError(
            f"Expected TurningDoubleWellEnv, got env.name={env_name!r} for checkpoint {checkpoint}"
        )


def _parse_seed_trial_from_filename(path: str) -> tuple[str | None, str | None]:
    name = os.path.basename(path)
    seed_match = re.search(r"__seed([^_]+)__", name)
    trial_match = re.search(r"__trial([^_]+)__", name)
    seed = seed_match.group(1) if seed_match else None
    trial = trial_match.group(1) if trial_match else None
    return seed, trial


def _autodetect_diffppo_zero_checkpoint(
    diffppo_checkpoint: str,
    *,
    seed_idx: int,
    horizon: int | None,
    env_config_overrides: list[str],
) -> tuple[str, float | None]:
    _base_ckpt, base_cfg, _base_method, _base_mode = _load_checkpoint_and_cfg(
        diffppo_checkpoint,
        seed_idx=seed_idx,
        horizon=horizon,
        env_config_overrides=env_config_overrides,
    )
    base_env = str(OmegaConf.select(base_cfg, "env.name") or "")
    base_abs = os.path.abspath(os.path.expanduser(diffppo_checkpoint))
    base_seed, base_trial = _parse_seed_trial_from_filename(base_abs)

    search_dir = os.path.dirname(base_abs) or "."
    candidate_paths = sorted(
        glob.glob(os.path.join(search_dir, f"reppo_DiffPPO__{base_env}__*.pkl"))
    )
    if not candidate_paths:
        candidate_paths = sorted(glob.glob(os.path.join(search_dir, "*.pkl")))

    best_path = base_abs
    best_score = float("inf")
    best_entropy: float | None = None

    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        try:
            ckpt = esm._load_checkpoint(abs_path)
            cfg_dict = ckpt.get("cfg")
            if cfg_dict is None:
                continue
            cfg = OmegaConf.create(cfg_dict)
            method_name = _detect_method_name(ckpt, cfg).lower()
            if method_name != "reppo_diffppo":
                continue
            env_name = str(OmegaConf.select(cfg, "env.name") or "")
            if env_name != base_env:
                continue

            cand_seed, cand_trial = _parse_seed_trial_from_filename(abs_path)
            if base_seed is not None and cand_seed is not None and cand_seed != base_seed:
                continue
            if base_trial is not None and cand_trial is not None and cand_trial != base_trial:
                continue

            ent = OmegaConf.select(cfg, "hyperparameters.entropy_coef")
            if ent is None:
                continue
            ent_f = float(ent)
            score = abs(ent_f)

            if score < best_score:
                best_score = score
                best_path = abs_path
                best_entropy = ent_f
                continue

            if score == best_score:
                # Tie-breaker: prefer newer timestamp in filename when available.
                # Fallback to lexicographic (usually encodes timestamp suffix).
                if os.path.basename(abs_path) > os.path.basename(best_path):
                    best_path = abs_path
                    best_entropy = ent_f
        except Exception:
            continue

    # If we could not score candidates, fall back to the user-provided DiffPPO checkpoint.
    if best_score == float("inf"):
        ent_base = OmegaConf.select(base_cfg, "hyperparameters.entropy_coef")
        try:
            best_entropy = float(ent_base) if ent_base is not None else None
        except Exception:
            best_entropy = None
        return base_abs, best_entropy

    return best_path, best_entropy


def _resolve_diffppo_pair(
    diffppo_checkpoints: list[str],
    *,
    checkpoint_diffppo_zero: str | None,
    seed_idx: int,
    horizon: int | None,
    env_config_overrides: list[str],
) -> tuple[str, str, float | None, str]:
    if not diffppo_checkpoints:
        raise ValueError("At least one --checkpoint-diffppo must be provided.")

    candidates = [os.path.abspath(os.path.expanduser(p)) for p in diffppo_checkpoints]
    candidates = list(dict.fromkeys(candidates))

    if checkpoint_diffppo_zero:
        zero_ckpt = os.path.abspath(os.path.expanduser(checkpoint_diffppo_zero))
        regular_ckpt = candidates[0]
        if regular_ckpt == zero_ckpt:
            for cand in candidates:
                if cand != zero_ckpt:
                    regular_ckpt = cand
                    break
        return regular_ckpt, zero_ckpt, None, "explicit"

    if len(candidates) == 1:
        regular_ckpt = candidates[0]
        zero_ckpt, zero_entropy = _autodetect_diffppo_zero_checkpoint(
            regular_ckpt,
            seed_idx=seed_idx,
            horizon=horizon,
            env_config_overrides=env_config_overrides,
        )
        return regular_ckpt, zero_ckpt, zero_entropy, "auto_scan"

    scored: list[tuple[str, float | None, float]] = []
    for ckpt_path in candidates:
        _ckpt, cfg, method_name, _train_mode = _load_checkpoint_and_cfg(
            ckpt_path,
            seed_idx=seed_idx,
            horizon=horizon,
            env_config_overrides=env_config_overrides,
        )
        if method_name.lower() != "reppo_diffppo":
            continue
        ent = OmegaConf.select(cfg, "hyperparameters.entropy_coef")
        ent_f = float(ent) if ent is not None else None
        score = abs(ent_f) if ent_f is not None else float("inf")
        scored.append((ckpt_path, ent_f, score))

    if len(scored) < 2:
        regular_ckpt = candidates[0]
        zero_ckpt, zero_entropy = _autodetect_diffppo_zero_checkpoint(
            regular_ckpt,
            seed_idx=seed_idx,
            horizon=horizon,
            env_config_overrides=env_config_overrides,
        )
        return regular_ckpt, zero_ckpt, zero_entropy, "auto_scan_fallback"

    zero_idx = min(range(len(scored)), key=lambda i: scored[i][2])
    zero_ckpt, zero_entropy, _ = scored[zero_idx]

    regular_ckpt = None
    for cand in candidates:
        if os.path.abspath(cand) != os.path.abspath(zero_ckpt):
            regular_ckpt = cand
            break
    if regular_ckpt is None:
        regular_ckpt = candidates[0]

    return os.path.abspath(regular_ckpt), os.path.abspath(zero_ckpt), zero_entropy, "auto_from_candidates"


def _prepare_reppo(
    spec: MethodSpec,
    checkpoint: dict[str, Any],
    cfg,
    method_name: str,
    train_mode: str,
) -> PreparedMethod:
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    num_envs = int(hp.num_envs)

    env = esm._build_base_env(cfg, horizon=horizon)
    if bool(hp.normalize_env):
        env = esm.LogWrapper(env, num_envs)
    env = esm.ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        env = esm.NormalizeVec(
            env,
            normalize_reward=bool(hp.normalize_reward),
            update_stats=False,
        )

    obs_dim = int(env.observation_space(None)[0].shape[0])
    action_dim = int(env.action_space(None).shape[0])

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = esm._select_seed(checkpoint["actor_params"], seed_idx, num_seeds)

    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = esm._select_seed(norm_state, seed_idx, num_seeds)

    actor_template = esm.SACActorNetworks(
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
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    actor_graphdef = nnx.graphdef(actor_template)
    actor_params = esm._to_jax_tree(actor_params)
    actor_params = esm._ensure_reppo_actor_params_compat(
        actor_params, ent_start=float(hp.ent_start)
    )
    if norm_state is not None:
        norm_state = esm._to_jax_tree(norm_state)
    if not bool(hp.normalize_env):
        norm_state = None

    actor_model = nnx.merge(actor_graphdef, actor_params)
    return PreparedMethod(
        spec=spec,
        checkpoint=checkpoint,
        cfg=cfg,
        method_name=method_name,
        train_mode=train_mode,
        norm_state=norm_state,
        actor_graphdef=actor_graphdef,
        actor_params=actor_params,
        actor_model=actor_model,
    )


def _prepare_dmerl(
    spec: MethodSpec,
    checkpoint: dict[str, Any],
    cfg,
    method_name: str,
    train_mode: str,
) -> PreparedMethod:
    from src.jaxrl.reppo_DMERL_new import ReppoConfig, ReppoDMERLTrainer

    horizon = int(cfg.env.max_episode_steps)
    base_env = esm._build_base_env(cfg, horizon=horizon)
    diff_cfg = cfg.hyperparameters.diffusion
    env_action_clip_value = float(cfg.hyperparameters.env_action_clip_value)
    env = esm.MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )

    trainer = ReppoDMERLTrainer(
        cfg=_instantiate_hparam_config(
            ReppoConfig, cfg.hyperparameters, label="ReppoDMERL"
        ),
        env=env,
        num_seeds=1,
        reward_scale=1.0 / float(cfg.env.reward_scaling),
    )
    train_state = trainer._make_init_fn()(jax.random.PRNGKey(0))

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = esm._select_seed(checkpoint["actor_params"], seed_idx, num_seeds)
    critic_params = esm._select_seed(checkpoint["critic_params"], seed_idx, num_seeds)
    actor_target_params = esm._select_seed(
        checkpoint.get("actor_target_params", checkpoint["actor_params"]),
        seed_idx,
        num_seeds,
    )
    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = esm._select_seed(norm_state, seed_idx, num_seeds)

    actor_params = esm._to_jax_tree(actor_params)
    actor_params = esm._ensure_dmerl_actor_params_compat(
        actor_params, ent_start=float(getattr(cfg.hyperparameters, "ent_start", 0.1))
    )
    critic_params = esm._to_jax_tree(critic_params)
    actor_target_params = esm._to_jax_tree(actor_target_params)
    if norm_state is not None:
        norm_state = esm._to_jax_tree(norm_state)

    train_state = train_state.replace(
        actor=train_state.actor.replace(params=actor_params),
        critic=train_state.critic.replace(params=critic_params),
        actor_target=train_state.actor_target.replace(params=actor_target_params),
    )
    if not bool(cfg.hyperparameters.normalize_env):
        norm_state = None

    actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
    return PreparedMethod(
        spec=spec,
        checkpoint=checkpoint,
        cfg=cfg,
        method_name=method_name,
        train_mode=train_mode,
        norm_state=norm_state,
        train_state=train_state,
        actor_model=actor_model,
    )


def _prepare_diffppo(
    spec: MethodSpec,
    checkpoint: dict[str, Any],
    cfg,
    method_name: str,
    train_mode: str,
) -> PreparedMethod:
    from src.jaxrl.reppo_DiffPPO import PPOConfig, ReppoPPOTrainer

    if not bool(OmegaConf.select(cfg, "hyperparameters.critic_use_normed_actions")):
        logging.warning(
            "DiffPPO checkpoint has hyperparameters.critic_use_normed_actions=%s; "
            "overriding to true for compatibility with current evaluator.",
            OmegaConf.select(cfg, "hyperparameters.critic_use_normed_actions"),
        )
        OmegaConf.update(
            cfg, "hyperparameters.critic_use_normed_actions", True, merge=False
        )

    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)
    base_env = esm._build_base_env(cfg, horizon=horizon)
    diff_cfg = hp.diffusion
    env_action_clip_value = float(hp.env_action_clip_value)
    env = esm.MjxDiffEnvWrapper(
        base_env,
        num_diff_steps=int(diff_cfg.diff_steps),
        diffusion_config=diff_cfg,
        low=-env_action_clip_value,
        high=env_action_clip_value,
    )

    trainer = ReppoPPOTrainer(
        cfg=_instantiate_hparam_config(
            PPOConfig, cfg.hyperparameters, label="ReppoDiffPPO"
        ),
        env=env,
        num_seeds=1,
    )
    train_state = trainer._make_init_fn()(jax.random.PRNGKey(0))

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))

    params = checkpoint.get("params", None)
    if params is None:
        raise ValueError(
            "DiffPPO checkpoint is missing `params`. "
            "Re-train with the updated checkpoint exporter."
        )
    params = esm._select_seed(params, seed_idx, num_seeds)

    norm_state = checkpoint.get("normalization_state", None)
    critic_norm_state = checkpoint.get("critic_normalization_state", None)
    reward_norm_state = checkpoint.get("reward_normalization_state", None)
    if norm_state is not None:
        norm_state = esm._select_seed(norm_state, seed_idx, num_seeds)
    if critic_norm_state is not None:
        critic_norm_state = esm._select_seed(critic_norm_state, seed_idx, num_seeds)
    if reward_norm_state is not None:
        reward_norm_state = esm._select_seed(reward_norm_state, seed_idx, num_seeds)

    params = esm._to_jax_tree(params)
    params = esm._ensure_diffppo_params_compat(
        params, ent_start=float(getattr(hp, "ent_start", 0.1))
    )
    if norm_state is not None:
        norm_state = esm._to_jax_tree(norm_state)
    if critic_norm_state is not None:
        critic_norm_state = esm._to_jax_tree(critic_norm_state)
    if reward_norm_state is not None:
        reward_norm_state = esm._to_jax_tree(reward_norm_state)

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

    model = nnx.merge(train_state.graphdef, train_state.params)
    return PreparedMethod(
        spec=spec,
        checkpoint=checkpoint,
        cfg=cfg,
        method_name=method_name,
        train_mode=train_mode,
        norm_state=train_state.normalization_state,
        train_state=train_state,
        actor_model=model,
        entropy_coef=float(getattr(hp, "entropy_coef", 0.0)),
    )


def _prepare_dime(
    spec: MethodSpec,
    checkpoint: dict[str, Any],
    cfg,
    method_name: str,
    train_mode: str,
) -> PreparedMethod:
    hp = cfg.hyperparameters
    horizon = int(cfg.env.max_episode_steps)

    env = esm._build_base_env(cfg, horizon=horizon)
    env = esm.LogWrapper(env, int(hp.num_envs))
    env = esm.ClipAction(
        env,
        low=-float(hp.env_action_clip_value),
        high=float(hp.env_action_clip_value),
    )
    if bool(hp.normalize_env):
        env = esm.NormalizeVec(env, update_stats=False)

    obs_dim = int(env.observation_space(None)[0].shape[0])
    action_dim = int(env.action_space(None).shape[0])

    def _call_target(spec_any: Any):
        if spec_any is None:
            raise ValueError("Missing diffusion.dt_schedule; cannot reconstruct DIME actor.")
        if callable(spec_any):
            return spec_any
        if OmegaConf.is_config(spec_any):
            spec_any = OmegaConf.to_container(spec_any, resolve=True)
        if not isinstance(spec_any, dict) or "_target_" not in spec_any:
            raise TypeError(f"Unsupported dt_schedule spec: {type(spec_any)} ({spec_any})")
        target = str(spec_any["_target_"])
        module_path, attr = target.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_path), attr)
        kwargs = {k: v for k, v in spec_any.items() if k != "_target_"}
        return fn(**kwargs)

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

    rngs = nnx.Rngs(jax.random.PRNGKey(0))
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
    actor_graphdef = nnx.graphdef(actor_networks)

    num_seeds = int(checkpoint.get("num_seeds", 1))
    seed_idx = int(checkpoint.get("seed_idx", 0))
    actor_params = esm._select_seed(checkpoint["actor_params"], seed_idx=seed_idx, num_seeds=num_seeds)

    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = esm._select_seed(norm_state, seed_idx=seed_idx, num_seeds=num_seeds)

    actor_params = esm._to_jax_tree(actor_params)
    if norm_state is not None:
        norm_state = esm._to_jax_tree(norm_state)
    if not bool(hp.normalize_env):
        norm_state = None

    actor_model = nnx.merge(actor_graphdef, actor_params)
    return PreparedMethod(
        spec=spec,
        checkpoint=checkpoint,
        cfg=cfg,
        method_name=method_name,
        train_mode=train_mode,
        norm_state=norm_state,
        actor_graphdef=actor_graphdef,
        actor_params=actor_params,
        actor_model=actor_model,
    )


def _prepare_method(spec: MethodSpec, args) -> PreparedMethod:
    checkpoint, cfg, method_name, train_mode = _load_checkpoint_and_cfg(
        spec.checkpoint,
        seed_idx=int(args.seed_idx),
        horizon=args.horizon,
        env_config_overrides=list(args.env_config_override or []),
    )
    _require_turning_double_well(cfg, checkpoint=spec.checkpoint)

    method_name_lower = method_name.lower()
    if method_name_lower == "reppo_dmerl_new":
        return _prepare_dmerl(spec, checkpoint, cfg, method_name, train_mode)
    if method_name_lower == "reppo_diffppo":
        return _prepare_diffppo(spec, checkpoint, cfg, method_name, train_mode)
    if "dime" in method_name_lower:
        return _prepare_dime(spec, checkpoint, cfg, method_name, train_mode)
    return _prepare_reppo(spec, checkpoint, cfg, method_name, train_mode)


def _resolve_sampler(method: PreparedMethod) -> str:
    sampler = str(method.spec.sampler or "auto").lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown sampler for {method.spec.slot}: {sampler}")

    method_name_lower = method.method_name.lower()
    env_name = OmegaConf.select(method.cfg, "env.name")

    if method_name_lower == "reppo_dmerl_new":
        if sampler == "auto":
            return esm._resolve_dmerl_sampler_auto(
                train_mode=method.train_mode,
                env_name=env_name,
            )
        return sampler

    if method_name_lower == "reppo_diffppo":
        if sampler == "auto":
            return "sde" if str(method.train_mode or "").upper() == "WPO" else "ode"
        return sampler

    if "dime" in method_name_lower:
        return "sde" if sampler == "auto" else sampler

    return "sde"


def _sample_actions_for_method(
    method: PreparedMethod,
    *,
    num_samples: int,
    num_grid: int,
    max_states: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = esm._tdw_build_env_for_analysis(method.cfg)
    angles = esm._tdw_discrete_starting_angles(env)
    if max_states is not None:
        angles = angles[: int(max_states)]
    angles_np = np.asarray(jax.device_get(angles))

    grid_actions = jnp.linspace(-1.0, 1.0, int(num_grid), dtype=jnp.float32)
    grid_actions_np = np.asarray(jax.device_get(grid_actions)).reshape(-1)
    reward_curve = env.reward_from_angle(grid_actions * env.max_turn_radians)
    reward_curve_np = np.asarray(jax.device_get(reward_curve)).reshape(-1)

    method_name_lower = method.method_name.lower()
    base_key = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(base_key, len(angles_np))

    sampled_per_state: list[np.ndarray] = []

    if method_name_lower == "reppo_dmerl_new":
        sampler_mode = _resolve_sampler(method)
        diff_steps = int(getattr(method.actor_model, "diff_steps", 1))
        if diff_steps <= 0:
            diff_steps = 1

        obs_mean = getattr(method.norm_state, "mean", None)
        obs_var = getattr(method.norm_state, "var", None)
        act_mean = getattr(method.norm_state, "action_mean", None)
        act_var = getattr(method.norm_state, "action_var", None)

        def _norm_obs(x):
            if obs_mean is None or obs_var is None:
                return x
            return (x - obs_mean) / jnp.sqrt(obs_var + 1e-2)

        def _norm_act(x):
            if act_mean is None or act_var is None:
                return x
            return (x - act_mean) / jnp.sqrt(act_var + 1e-2)

        for theta, key in zip(angles_np, keys):
            theta_j = jnp.asarray(theta, dtype=jnp.float32)
            raw_obs = esm._tdw_obs_from_angle(env, theta_j)
            obs_norm = _norm_obs(raw_obs)

            key, prior_key = jax.random.split(key)
            current_x = method.actor_model.diffusion_model.prior_sampler(prior_key, int(num_samples))
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
                if sampler_mode == "sde":
                    current_x, *_ = method.actor_model.vmap_sample_next_step(obs_dict, step_key)
                else:
                    current_x, _ = method.actor_model.vmap_ode_sample_next_step(obs_dict, step_key)

            sampled_actions = jnp.tanh(current_x)
            sampled_per_state.append(np.asarray(jax.device_get(sampled_actions)).reshape(-1))

    elif method_name_lower == "reppo_diffppo":
        sampler_mode = _resolve_sampler(method)
        model = method.actor_model
        diff_steps = int(getattr(model.actor_module, "diff_steps", 1))
        if diff_steps <= 0:
            diff_steps = 1
        normalizer = esm.DictNormalizer()

        def _normalize_obs_dict(obs_dict):
            if method.norm_state is None:
                return obs_dict
            return normalizer.normalize(method.norm_state, obs_dict)

        for theta, key in zip(angles_np, keys):
            theta_j = jnp.asarray(theta, dtype=jnp.float32)
            raw_obs = esm._tdw_obs_from_angle(env, theta_j)

            key, prior_key = jax.random.split(key)
            current_x = model.actor_module.diffusion_model.prior_sampler(prior_key, int(num_samples))
            current_x = jnp.asarray(current_x, dtype=jnp.float32)

            chain_key = key
            for step_idx in range(diff_steps):
                chain_key, step_key = jax.random.split(chain_key)
                obs_dict = {
                    "orig_obs": jnp.broadcast_to(raw_obs, (current_x.shape[0],) + raw_obs.shape),
                    "orig_actions": current_x,
                    "normed_actions": current_x,
                    "diff_time_step": jnp.full((current_x.shape[0], 1), step_idx, dtype=jnp.int32),
                }
                obs_dict = _normalize_obs_dict(obs_dict)
                if sampler_mode == "sde":
                    current_x, *_ = model.actor_sample_step(obs_dict, step_key)
                else:
                    current_x, _ = model.actor_ode_sample_step(obs_dict, step_key)

            sampled_actions = jnp.tanh(current_x)
            sampled_per_state.append(np.asarray(jax.device_get(sampled_actions)).reshape(-1))

    elif "dime" in method_name_lower:
        for theta, key in zip(angles_np, keys):
            theta_j = jnp.asarray(theta, dtype=jnp.float32)
            raw_obs = esm._tdw_obs_from_angle(env, theta_j)
            obs = esm._normalize_obs(raw_obs, method.norm_state, critic=False)

            sample_keys = jax.random.split(key, int(num_samples))

            def _one(k):
                action, *_ = method.actor_model.sample(k, obs[None, :])
                return action[0]

            sampled = jax.vmap(_one)(sample_keys)
            sampled_per_state.append(np.asarray(jax.device_get(sampled)).reshape(-1))

    else:
        for theta, key in zip(angles_np, keys):
            theta_j = jnp.asarray(theta, dtype=jnp.float32)
            raw_obs = esm._tdw_obs_from_angle(env, theta_j)
            obs = esm._normalize_obs(raw_obs, method.norm_state, critic=False)

            pi = method.actor_model.actor(obs)
            sampled = pi.sample(seed=key, sample_shape=(int(num_samples),))
            sampled_per_state.append(np.asarray(jax.device_get(sampled)).reshape(-1))

    sampled_np = np.stack(sampled_per_state, axis=0)
    return angles_np, sampled_np, grid_actions_np, reward_curve_np


def _render_snapshot(method: PreparedMethod, args) -> dict[str, str]:
    render_basename = f"{args.output_stem}__{method.spec.slot}__turning_double_well_traj.gif"

    outputs = esm._render_turning_double_well_reppo(
        method_name=method.method_name,
        train_mode=method.train_mode,
        entropy_coef=method.entropy_coef,
        diffusion_sampler=method.spec.sampler,
        cfg=method.cfg,
        actor_graphdef=method.actor_graphdef,
        actor_params=method.actor_params,
        train_state=method.train_state,
        checkpoint_path=os.path.abspath(os.path.expanduser(method.spec.checkpoint)),
        norm_state=method.norm_state,
        horizon=int(method.cfg.env.max_episode_steps),
        num_envs=int(args.render_num_envs),
        out_path=render_basename,
        overlay=True,
        width=int(args.render_width),
        height=int(args.render_height),
        fps=int(args.render_fps),
        seed=int(args.render_seed),
        render_format="gif",
    )
    return outputs


def _state_angle_label(theta_radians: float) -> str:
    theta_deg = (float(np.rad2deg(float(theta_radians))) + 360.0) % 360.0
    return rf"$\theta_0={theta_deg:.1f}^\circ$"


def _compose_hist_figure(
    methods: list[PreparedMethod],
    action_data: dict[str, dict[str, Any]],
    *,
    out_path: str,
    bins: int,
    state_cmap: str,
    dpi: int,
) -> None:
    n_methods = len(methods)
    if n_methods == 0:
        raise ValueError("No methods provided for histogram figure.")

    ref_angles = action_data[methods[0].spec.slot]["angles"]
    n_states = int(ref_angles.shape[0])

    cmap = plt.get_cmap(state_cmap)
    colors = [cmap(i / max(n_states, 1)) for i in range(n_states)]

    fig, axes = plt.subplots(1, n_methods, figsize=(3.9 * n_methods, 4.6), squeeze=False)
    axes_1d = axes.reshape(-1)

    for idx, method in enumerate(methods):
        slot = method.spec.slot
        ax = axes_1d[idx]

        sampled = action_data[slot]["sampled"]
        grid_actions = action_data[slot]["grid_actions"]
        reward_curve = action_data[slot]["reward_curve"]

        for state_idx in range(sampled.shape[0]):
            ax.hist(
                sampled[state_idx],
                bins=int(bins),
                range=(-1.0, 1.0),
                density=True,
                histtype="step",
                linewidth=1.3,
                color=colors[state_idx],
                alpha=0.95,
            )

        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel("action", fontsize=11)
        if idx == 0:
            ax.set_ylabel("density", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.28)
        ax.set_title(method.spec.label, fontsize=12, pad=7)

        ax2 = ax.twinx()
        ax2.plot(grid_actions, reward_curve, color="black", linewidth=2.0)
        ax2.set_ylim(-0.05, 1.05)
        if idx == n_methods - 1:
            ax2.set_ylabel("reward", fontsize=11, color="black")
        else:
            ax2.set_yticklabels([])
        ax2.tick_params(axis="y", labelsize=8, colors="black")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=colors[i], linewidth=2.0, label=_state_angle_label(ref_angles[i]))
        for i in range(n_states)
    ]
    legend_handles.append(Line2D([0], [0], color="black", linewidth=2.3, label="reward"))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=min(max(3, n_states // 2 + 1), len(legend_handles)),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _compose_trajectory_figure(
    methods: list[PreparedMethod],
    trajectory_pngs: dict[str, str],
    *,
    out_path: str,
    dpi: int,
) -> None:
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4.3 * n_methods, 3.9), squeeze=False)
    axes_1d = axes.reshape(-1)

    for idx, method in enumerate(methods):
        ax = axes_1d[idx]
        png_path = trajectory_pngs[method.spec.slot]
        img = plt.imread(png_path)
        ax.imshow(img)
        ax.set_title(method.spec.label, fontsize=12, pad=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create compact TurningDoubleWell paper figures: "
            "action-histogram row and trajectory row across methods."
        )
    )

    p.add_argument("--checkpoint-reppo", required=True)
    p.add_argument(
        "--checkpoint-diffppo",
        action="append",
        required=True,
        help=(
            "DiffPPO checkpoint (repeatable). "
            "If passed multiple times and --checkpoint-diffppo-zero is omitted, "
            "the script auto-selects the zero-temp run among these by entropy_coef closest to 0."
        ),
    )
    p.add_argument(
        "--checkpoint-diffppo-zero",
        default=None,
        help=(
            "Optional explicit DiffPPO zero-temperature checkpoint. "
            "If omitted, it is auto-detected from DiffPPO checkpoints "
            "(same env and seed/trial when available) by selecting entropy_coef closest to 0."
        ),
    )
    p.add_argument("--checkpoint-dmerl", required=True)
    p.add_argument("--checkpoint-dmerl-wpo", required=True)
    p.add_argument("--checkpoint-dime", required=True)

    p.add_argument("--sampler-reppo", default="auto", choices=["auto", "sde", "ode"])
    p.add_argument("--sampler-diffppo", default="sde", choices=["auto", "sde", "ode"])
    p.add_argument("--sampler-diffppo-zero", default="ode", choices=["auto", "sde", "ode"])
    p.add_argument("--sampler-dmerl", default="auto", choices=["auto", "sde", "ode"])
    p.add_argument("--sampler-dmerl-wpo", default="auto", choices=["auto", "sde", "ode"])
    p.add_argument("--sampler-dime", default="auto", choices=["auto", "sde", "ode"])

    p.add_argument("--seed-idx", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--env-config-override", action="append", default=[])

    p.add_argument("--analysis-samples", type=int, default=5000)
    p.add_argument("--analysis-grid", type=int, default=401)
    p.add_argument("--analysis-bins", type=int, default=60)
    p.add_argument("--analysis-max-states", type=int, default=None)
    p.add_argument("--analysis-seed", type=int, default=0)

    p.add_argument("--render-num-envs", type=int, default=20)
    p.add_argument("--render-width", type=int, default=1200)
    p.add_argument("--render-height", type=int, default=800)
    p.add_argument("--render-fps", type=int, default=20)
    p.add_argument("--render-seed", type=int, default=0)

    p.add_argument("--output-dir", default=os.path.join("artifacts", "tdw_paper"))
    p.add_argument("--output-stem", default="tdw_paper")
    p.add_argument("--state-cmap", default="hsv")
    p.add_argument("--dpi", type=int, default=220)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    diffppo_ckpt, diffppo_zero_ckpt, diffppo_zero_entropy, diffppo_zero_mode = _resolve_diffppo_pair(
        list(args.checkpoint_diffppo or []),
        checkpoint_diffppo_zero=args.checkpoint_diffppo_zero,
        seed_idx=int(args.seed_idx),
        horizon=args.horizon,
        env_config_overrides=list(args.env_config_override or []),
    )
    _progress(
        f"Using DiffPPO checkpoint: {diffppo_ckpt}"
    )
    _progress(
        "Using DiffPPO zero-temp checkpoint: "
        f"{diffppo_zero_ckpt} "
        f"(mode={diffppo_zero_mode}, entropy_coef={diffppo_zero_entropy if diffppo_zero_entropy is not None else 'unknown'})"
    )

    specs = [
        MethodSpec("reppo", "REPPO", args.checkpoint_reppo, args.sampler_reppo),
        MethodSpec("diffppo", "DiffPPO", diffppo_ckpt, args.sampler_diffppo),
        MethodSpec("diffppo_zero", "DPPO (DME-PPO zero temp)", diffppo_zero_ckpt, args.sampler_diffppo_zero),
        MethodSpec("dmerl", "DMERL", args.checkpoint_dmerl, args.sampler_dmerl),
        MethodSpec("dmerl_wpo", "DMERL-WPO", args.checkpoint_dmerl_wpo, args.sampler_dmerl_wpo),
        MethodSpec("dime", "DIME", args.checkpoint_dime, args.sampler_dime),
    ]

    _progress("Preparing models from checkpoints")
    prepared_methods: list[PreparedMethod] = []
    for spec in specs:
        _progress(f"Loading {spec.label}: {spec.checkpoint}")
        prepared = _prepare_method(spec, args)
        prepared_methods.append(prepared)

    _progress("Sampling action distributions per state")
    action_data: dict[str, dict[str, Any]] = {}
    reference_angles: np.ndarray | None = None

    for method in prepared_methods:
        angles, sampled, grid_actions, reward_curve = _sample_actions_for_method(
            method,
            num_samples=int(args.analysis_samples),
            num_grid=int(args.analysis_grid),
            max_states=args.analysis_max_states,
            seed=int(args.analysis_seed),
        )

        if reference_angles is None:
            reference_angles = angles
        else:
            if len(reference_angles) != len(angles) or not np.allclose(reference_angles, angles, atol=1e-6, rtol=0.0):
                raise ValueError(
                    "State angle sets differ between methods. "
                    "Use consistent env overrides / checkpoints for comparable panels."
                )

        action_data[method.spec.slot] = {
            "angles": angles,
            "sampled": sampled,
            "grid_actions": grid_actions,
            "reward_curve": reward_curve,
            "sampler_resolved": _resolve_sampler(method),
        }

    _progress("Rendering static trajectory snapshots")
    trajectory_pngs: dict[str, str] = {}
    trajectory_outputs: dict[str, dict[str, str]] = {}
    for method in prepared_methods:
        outputs = _render_snapshot(method, args)
        last_frame = outputs.get("last_frame_png", "")
        if not last_frame or not os.path.isfile(last_frame):
            raise RuntimeError(
                f"Failed to produce last-frame trajectory PNG for {method.spec.label} "
                f"({method.spec.checkpoint})."
            )
        trajectory_pngs[method.spec.slot] = last_frame
        trajectory_outputs[method.spec.slot] = outputs

    hist_png = os.path.abspath(os.path.join(args.output_dir, f"{args.output_stem}__action_hist_row.png"))
    traj_png = os.path.abspath(os.path.join(args.output_dir, f"{args.output_stem}__trajectory_row.png"))

    _progress(f"Composing action figure: {hist_png}")
    _compose_hist_figure(
        prepared_methods,
        action_data,
        out_path=hist_png,
        bins=int(args.analysis_bins),
        state_cmap=str(args.state_cmap),
        dpi=int(args.dpi),
    )

    _progress(f"Composing trajectory figure: {traj_png}")
    _compose_trajectory_figure(
        prepared_methods,
        trajectory_pngs,
        out_path=traj_png,
        dpi=int(args.dpi),
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": os.path.abspath(args.output_dir),
        "output_stem": str(args.output_stem),
        "action_hist_figure": hist_png,
        "trajectory_figure": traj_png,
        "methods": [
            {
                "slot": m.spec.slot,
                "label": m.spec.label,
                "checkpoint": os.path.abspath(os.path.expanduser(m.spec.checkpoint)),
                "method_name": m.method_name,
                "train_mode": m.train_mode,
                "sampler_arg": m.spec.sampler,
                "sampler_resolved": action_data[m.spec.slot]["sampler_resolved"],
                "trajectory_outputs": trajectory_outputs[m.spec.slot],
            }
            for m in prepared_methods
        ],
        "diffppo_zero_checkpoint_mode": diffppo_zero_mode,
        "diffppo_zero_checkpoint_entropy_coef": diffppo_zero_entropy,
        "analysis": {
            "samples_per_state": int(args.analysis_samples),
            "bins": int(args.analysis_bins),
            "grid_points": int(args.analysis_grid),
            "max_states": None if args.analysis_max_states is None else int(args.analysis_max_states),
            "seed": int(args.analysis_seed),
        },
        "render": {
            "num_envs": int(args.render_num_envs),
            "width": int(args.render_width),
            "height": int(args.render_height),
            "fps": int(args.render_fps),
            "seed": int(args.render_seed),
        },
        "env_overrides": list(args.env_config_override or []),
    }
    manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{args.output_stem}__manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    _progress("Done")
    _progress(f"Action figure: {hist_png}")
    _progress(f"Trajectory figure: {traj_png}")
    _progress(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
