import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from src.env_utils.jax_wrappers import MjxDiffEnvWrapper, MjxGymnaxWrapper
from src.jaxrl.DA_MDP_REPPO import ReppoConfig, ReppoDMERLTrainer

logging.basicConfig(level=logging.INFO)


def _load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _select_seed(tree, seed_idx: int, num_seeds: int):
    def _maybe_index(x):
        x = np.asarray(x)
        if num_seeds > 0 and x.ndim > 0 and x.shape[0] == num_seeds:
            return x[seed_idx]
        return x

    return jax.tree.map(_maybe_index, tree)


def _build_env(cfg):
    if cfg.env.type == "brax":
        raise ValueError("Brax environment type is not supported in this evaluator.")
    if cfg.env.type != "mjx":
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    env_config = OmegaConf.select(cfg, "env.config")
    if env_config is not None:
        env_config = OmegaConf.to_container(env_config, resolve=True)
    env = MjxGymnaxWrapper(
        cfg.env.name,
        episode_length=cfg.env.max_episode_steps,
        reward_scale=cfg.env.reward_scaling,
        push_distractions=cfg.env.get("push_distractions", False),
        config=env_config,
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
    return env


def _resolve_sampler(train_mode: str, sampler_arg: str) -> str:
    sampler = str(sampler_arg).lower()
    if sampler not in ("auto", "sde", "ode"):
        raise ValueError(f"Unknown --diffusion-sampler: {sampler}")
    if sampler == "auto":
        return "sde" if str(train_mode).upper() == "WPO" else "ode"
    return sampler


def _resolve_samplers(train_mode: str, sampler_arg: str) -> list[str]:
    sampler = str(sampler_arg).lower()
    if sampler == "both":
        return ["ode", "sde"]
    return [_resolve_sampler(train_mode=train_mode, sampler_arg=sampler)]


def _resolve_alphas(args) -> list[float]:
    values: list[float] = []
    if args.alphas is not None and len(args.alphas) > 0:
        values.extend([float(v) for v in args.alphas])

    has_range = (
        args.alpha_start is not None
        or args.alpha_end is not None
        or args.alpha_steps is not None
    )
    if has_range:
        if (
            args.alpha_start is None
            or args.alpha_end is None
            or args.alpha_steps is None
        ):
            raise ValueError(
                "When using alpha range, pass all of --alpha-start, --alpha-end, and --alpha-steps."
            )
        alpha_steps = int(args.alpha_steps)
        if alpha_steps < 2:
            raise ValueError("--alpha-steps must be >= 2.")
        values.extend(
            np.linspace(float(args.alpha_start), float(args.alpha_end), alpha_steps).tolist()
        )

    if not values:
        values = [0.0]

    deduped: list[float] = []
    for v in values:
        if not any(abs(v - u) <= 1e-12 for u in deduped):
            deduped.append(float(v))
    return deduped


def _resolve_guidance_dts(args) -> list[float]:
    if args.guidance_dts is None:
        return [1.0]
    if len(args.guidance_dts) == 0:
        raise ValueError("--guidance-dts was provided but no values were given.")
    deduped: list[float] = []
    for v in args.guidance_dts:
        fv = float(v)
        if not any(abs(fv - u) <= 1e-12 for u in deduped):
            deduped.append(fv)
    return deduped


def _masked_mean_std(values: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    mask_f = mask.astype(jnp.float32)
    count = jnp.sum(mask_f)
    mean = jnp.where(count > 0.0, jnp.sum(values * mask_f) / count, jnp.nan)
    centered = values - mean
    var = jnp.where(count > 0.0, jnp.sum((centered * centered) * mask_f) / count, jnp.nan)
    std = jnp.sqrt(jnp.maximum(var, 0.0))
    return mean, std, count


def _std_to_sem(std: jax.Array, count: jax.Array) -> jax.Array:
    return jnp.where(count > 0.0, std / jnp.sqrt(count), jnp.nan)


def _clip_by_l2_norm(x: jax.Array, max_norm: float, eps: float = 1e-8) -> jax.Array:
    max_norm_arr = jnp.asarray(max_norm, dtype=x.dtype)
    if x.ndim == 1:
        norm = jnp.linalg.norm(x)
        scale = jnp.minimum(1.0, max_norm_arr / (norm + eps))
        return x * scale
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_norm_arr / (norm + eps))
    return x * scale


def _normalize_obs_dict_with_stats(
    obs_dict: dict[str, jax.Array],
    *,
    obs_mean: jax.Array,
    obs_var: jax.Array,
    action_mean: jax.Array,
    action_var: jax.Array,
) -> dict[str, jax.Array]:
    out = dict(obs_dict)
    out["orig_obs"] = (obs_dict["orig_obs"] - obs_mean) / jnp.sqrt(obs_var + 1e-2)
    if "normed_actions" in obs_dict:
        out["normed_actions"] = (obs_dict["normed_actions"] - action_mean) / jnp.sqrt(
            action_var + 1e-2
        )
    return out


def _extract_fixed_norm_stats(norm_state) -> dict[str, jax.Array] | None:
    if norm_state is None:
        return None
    required = (
        "mean",
        "var",
        "action_mean",
        "action_var",
        "critic_mean",
        "critic_var",
        "critic_action_mean",
        "critic_action_var",
    )
    missing = [k for k in required if not hasattr(norm_state, k)]
    if missing:
        raise ValueError(
            "Loaded normalization state is missing fields for fixed normalization: "
            + ", ".join(missing)
        )
    return {k: getattr(norm_state, k) for k in required}


def _extract_raw_obs_from_env_state(env_state):
    cur = env_state
    for _ in range(5):
        obs = getattr(cur, "obs", None)
        critic_obs = getattr(cur, "critic_obs", None)
        if obs is not None and critic_obs is not None:
            return obs, critic_obs
        if hasattr(cur, "env_state"):
            cur = getattr(cur, "env_state")
            continue
        break
    return None, None


def _rebuild_raw_obs_dicts(
    obs_dict: dict[str, jax.Array],
    critic_obs_dict: dict[str, jax.Array],
    env_state,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    raw_obs, raw_critic_obs = _extract_raw_obs_from_env_state(env_state)
    if raw_obs is None:
        raw_obs = obs_dict["orig_obs"]
    if raw_critic_obs is None:
        raw_critic_obs = critic_obs_dict["orig_obs"]
    raw_actor = dict(obs_dict)
    raw_actor["orig_obs"] = raw_obs
    raw_actor["normed_actions"] = obs_dict["orig_actions"]
    raw_critic = dict(critic_obs_dict)
    raw_critic["orig_obs"] = raw_critic_obs
    raw_critic["normed_actions"] = critic_obs_dict["orig_actions"]
    return raw_actor, raw_critic


def _control_score_without_q_grad(control_model, x: jax.Array, obs_vec: jax.Array, step: jax.Array) -> jax.Array:
    time_emb = control_model.get_fourier_features(step)
    if x.ndim == 1:
        time_emb = time_emb[0]
    t_net = control_model.time_coder_state(time_emb)
    net_in = jnp.concatenate((x, obs_vec, t_net), axis=-1)
    out = control_model.state_time_net(net_in)
    return jnp.clip(out, -control_model.outer_clip, control_model.outer_clip)


def _make_alpha_eval_fn(
    trainer: ReppoDMERLTrainer,
    sampler: str,
    guidance_dt: float,
    q_temperature: float,
    q_grad_clip: float | None,
    q_grad_clip_mode: str,
    q_guidance_last_percent: float,
    fixed_norm_stats: dict[str, jax.Array] | None,
    normalizer_mode: str,
):
    env = trainer.eval_env
    max_episode_steps = trainer.eval_env_steps
    reward_scale = trainer.reward_scale
    ode_coeff = 0.5 if sampler == "ode" else 1.0

    def eval_with_alpha(
        key: jax.random.PRNGKey,
        train_state,
        norm_state,
        alpha: float,
    ):
        alpha_scalar = float(alpha)
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        diffusion_model = actor_model.diffusion_model
        control_model = diffusion_model.fwd_model
        alpha = jnp.asarray(alpha_scalar, dtype=jnp.float32)
        guidance_last_fraction = jnp.asarray(
            np.clip(float(q_guidance_last_percent), 0.0, 100.0) / 100.0,
            dtype=jnp.float32,
        )
        guidance_start = 1.0 - guidance_last_fraction

        if control_model is None:
            raise ValueError("Actor diffusion model has no forward score network.")

        def _policy_step(policy_key, obs, critic_obs, env_state):
            if normalizer_mode in ("fixed", "off"):
                raw_obs, raw_critic_obs = _rebuild_raw_obs_dicts(obs, critic_obs, env_state)
            else:
                raw_obs, raw_critic_obs = obs, critic_obs

            if normalizer_mode == "fixed" and fixed_norm_stats is not None:
                obs_for_actor = _normalize_obs_dict_with_stats(
                    raw_obs,
                    obs_mean=fixed_norm_stats["mean"],
                    obs_var=fixed_norm_stats["var"],
                    action_mean=fixed_norm_stats["action_mean"],
                    action_var=fixed_norm_stats["action_var"],
                )
                critic_obs_for_actor = _normalize_obs_dict_with_stats(
                    raw_critic_obs,
                    obs_mean=fixed_norm_stats["critic_mean"],
                    obs_var=fixed_norm_stats["critic_var"],
                    action_mean=fixed_norm_stats["critic_action_mean"],
                    action_var=fixed_norm_stats["critic_action_var"],
                )
            else:
                obs_for_actor = raw_obs
                critic_obs_for_actor = raw_critic_obs

            batch_size = obs["orig_obs"].shape[0]
            if policy_key.ndim == 1:
                policy_keys = jax.random.split(policy_key, batch_size)
            else:
                policy_keys = policy_key

            step = obs_for_actor["diff_time_step"][..., 0].astype(jnp.float32)
            action = obs_for_actor["orig_actions"]
            obs_vec = jnp.concatenate(
                [obs_for_actor["orig_obs"], obs_for_actor["normed_actions"]], axis=-1
            )

            base_score = jax.vmap(_control_score_without_q_grad, in_axes=(None, 0, 0, 0))(
                control_model, action, obs_vec, step
            )

            # Apply Q guidance only in the last X% of diffusion steps.
            diff_steps = max(int(diffusion_model.diff_steps), 1)
            step_norm = (
                step / float(max(diff_steps - 1, 1))
                if diff_steps > 1
                else jnp.ones_like(step)
            )
            q_gate = (step_norm >= guidance_start).astype(base_score.dtype)
            if q_gate.ndim < base_score.ndim:
                q_gate = q_gate.reshape(q_gate.shape + (1,) * (base_score.ndim - q_gate.ndim))
            effective_alpha = alpha * q_gate

            if alpha_scalar <= 0.0:
                guided_score = base_score
            else:
                def _q_grad(single_critic_obs, single_action):
                    q_fn = lambda a: critic_model.critic(single_critic_obs, a).sum()
                    return jax.grad(q_fn)(single_action)

                q_grad = jax.vmap(_q_grad)(critic_obs_for_actor, action)
                inv_temp = jnp.asarray(1.0 / q_temperature, dtype=q_grad.dtype)
                q_grad = inv_temp * q_grad

                def _forward_logp_grad(single_obs, single_action):
                    def _logp(a):
                        single_obs_batched = jax.tree.map(
                            lambda x: jnp.expand_dims(x, axis=0), single_obs
                        )
                        a_batched = jnp.expand_dims(a, axis=0)
                        _, dest_log_prob = actor_model.vmap_eval_log_prob(
                            single_obs_batched, a_batched
                        )
                        return dest_log_prob.sum()

                    return jax.grad(_logp)(single_action)

                grad_log_p = jax.vmap(_forward_logp_grad)(obs_for_actor, action)
                guidance_grad = q_grad + grad_log_p
                if q_grad_clip is not None:
                    if q_grad_clip_mode == "l2":
                        guidance_grad = _clip_by_l2_norm(
                            guidance_grad, max_norm=float(q_grad_clip)
                        )
                    elif q_grad_clip_mode == "elementwise":
                        clip_v = jnp.asarray(q_grad_clip, dtype=guidance_grad.dtype)
                        guidance_grad = jnp.clip(guidance_grad, -clip_v, clip_v)
                    else:
                        raise ValueError(f"Unknown q_grad_clip_mode: {q_grad_clip_mode}")

                guidance_dt_arr = jnp.asarray(guidance_dt, dtype=guidance_grad.dtype)
                guidance_grad = guidance_dt_arr * guidance_grad

                # Mask out inactive timesteps to avoid 0 * NaN leakage in the blend.
                guidance_grad = jnp.where(
                    q_gate > 0,
                    guidance_grad,
                    jnp.zeros_like(guidance_grad),
                )
                guided_score = (1.0 - effective_alpha) * base_score + effective_alpha * guidance_grad
            drift = jax.vmap(diffusion_model.drift_fn)(step, action)

            def _coeff(single_step, single_obs):
                scale, eta, _ = diffusion_model.diffusion_coeff_fn(single_step, single_obs)
                return scale, eta

            scale, eta = jax.vmap(_coeff)(step, obs_for_actor)
            mu = drift + ode_coeff * guided_score
            mean = action + eta * mu

            if sampler == "ode":
                return mean

            noise = jax.vmap(
                lambda k, x: jax.random.normal(k, shape=x.shape, dtype=x.dtype)
            )(policy_keys, action)
            return mean + scale * noise

        def _step_env(carry, _):
            step_key, env_state, obs, critic_obs = carry
            step_key, act_key, env_key = jax.random.split(step_key, 3)
            env_action = _policy_step(act_key, obs, critic_obs, env_state)
            keys = jax.random.split(env_key, env.num_envs)
            obs, critic_obs, env_state, _reward, _done, info = env.step(
                keys, env_state, env_action
            )
            return (step_key, env_state, obs, critic_obs), info

        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, env.num_envs)
        if norm_state is None:
            obs, critic_obs, env_state = env.reset(init_keys)
        else:
            obs, critic_obs, env_state = env.reset(init_keys, norm_state)

        _, infos = jax.lax.scan(
            _step_env,
            init=(key, env_state, obs, critic_obs),
            xs=None,
            length=max_episode_steps,
        )

        returned = infos["returned_episode"]
        ret_mean, ret_std, num_eps = _masked_mean_std(
            infos["returned_episode_returns"], returned
        )
        len_mean, len_std, _ = _masked_mean_std(
            infos["returned_episode_lengths"], returned
        )
        ret_sem = _std_to_sem(ret_std, num_eps)
        len_sem = _std_to_sem(len_std, num_eps)
        return {
            "episode_return": ret_mean * reward_scale,
            "episode_return_std": ret_std,
            "episode_return_sem": ret_sem,
            "episode_length": len_mean,
            "episode_length_std": len_std,
            "episode_length_sem": len_sem,
            "num_episodes": num_eps,
        }

    return eval_with_alpha


def _default_plot_out(checkpoint_path: str) -> str:
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(_REPO_ROOT, "artifacts", "guidance")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ckpt_base}__alpha_sweep_episode_return.png")


def _default_json_out(checkpoint_path: str) -> str:
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(_REPO_ROOT, "artifacts", "guidance")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ckpt_base}__alpha_sweep_results.json")


def _save_episode_return_plot(results_by_sampler: dict[str, list[dict[str, float]]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    plotted = 0
    sampler_linestyle = {"ode": "-", "sde": "--"}
    sampler_marker = {"ode": "o", "sde": "s"}
    cmap = plt.get_cmap("viridis")
    for sampler in ("ode", "sde"):
        rows = results_by_sampler.get(sampler, [])
        if not rows:
            continue
        dts = sorted({float(r.get("guidance_dt", 1.0)) for r in rows})
        n_dt = len(dts)
        for dt_idx, guidance_dt in enumerate(dts):
            rows_dt = [
                r
                for r in rows
                if abs(float(r.get("guidance_dt", 1.0)) - guidance_dt) <= 1e-12
            ]
            rows_sorted = sorted(rows_dt, key=lambda r: r["alpha"])
            alpha_vals = np.asarray([r["alpha"] for r in rows_sorted], dtype=np.float64)
            returns = np.asarray([r["episode_return"] for r in rows_sorted], dtype=np.float64)
            ret_sem = np.asarray([r["episode_return_sem"] for r in rows_sorted], dtype=np.float64)
            cmap_pos = 0.5 if n_dt == 1 else float(dt_idx) / float(max(n_dt - 1, 1))
            color = cmap(cmap_pos)
            ax.plot(
                alpha_vals,
                returns,
                marker=sampler_marker.get(sampler, "o"),
                markersize=6.5,
                linewidth=2.4,
                linestyle=sampler_linestyle.get(sampler, "-"),
                color=color,
                label=f"{sampler.upper()} dt={guidance_dt:.3g}",
            )
            if np.all(np.isfinite(ret_sem)):
                ax.fill_between(
                    alpha_vals,
                    returns - ret_sem,
                    returns + ret_sem,
                    alpha=0.12,
                    color=color,
                    linewidth=0.0,
                )
            plotted += 1

    if plotted == 0:
        raise ValueError("No sampler results available to plot.")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Episode Return")
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.2, length=6, direction="out")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DA_MDP_REPPO checkpoints with score/Q-gradient interpolation. "
            "Guided score: (1-alpha)*score + alpha*dt*(grad_log_p + (1/T)*grad_a Q)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint.pkl saved under saved_models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for evaluation rollouts.",
    )
    parser.add_argument(
        "--seed-idx",
        type=int,
        default=0,
        help="Which trained seed index to evaluate when checkpoint contains multiple seeds.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of parallel envs used during evaluation.",
    )
    parser.add_argument(
        "--normalizer-stats-mode",
        choices=("fixed", "online", "off"),
        default="fixed",
        help=(
            "How to handle observation normalization stats during evaluation. "
            "'fixed': use loaded checkpoint stats and keep them fixed. "
            "'online': update stats online during rollout (training-like wrapper behavior). "
            "'off': disable normalization."
        ),
    )
    parser.add_argument(
        "--diffusion-sampler",
        choices=("both", "auto", "sde", "ode"),
        default="both",
        help=(
            "'both' (default) evaluates both ODE and SDE. "
            "'auto' matches training mode (WPO->sde, else ode)."
        ),
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=None,
        help="Explicit list of alpha values to evaluate (e.g. --alphas 0.0 0.25 0.5 0.75 1.0).",
    )
    parser.add_argument("--alpha-start", type=float, default=None, help="Start of alpha range.")
    parser.add_argument("--alpha-end", type=float, default=None, help="End of alpha range.")
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=None,
        help="Number of points in alpha range (inclusive linspace).",
    )
    parser.add_argument(
        "--guidance-dts",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional list of dt values that scale guidance before blending. "
            "For each dt, a full alpha sweep is evaluated and plotted as a separate line."
        ),
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature T used in guidance as (1/T) * grad_a Q. "
            "Default is 1.0."
        ),
    )
    parser.add_argument(
        "--q-grad-clip",
        type=float,
        default=None,
        help=(
            "Optional clip threshold for (grad_log_p + grad_a Q). "
            "If omitted, no clipping is applied."
        ),
    )
    parser.add_argument(
        "--q-grad-clip-mode",
        choices=("l2", "elementwise"),
        default="l2",
        help=(
            "Clipping mode for --q-grad-clip. "
            "'l2' clips by vector norm, 'elementwise' clips each component."
        ),
    )
    parser.add_argument(
        "--q-guidance-last-percent",
        type=float,
        default=100.0,
        help=(
            "Apply Q guidance only in the last X percent of diffusion steps "
            "(0 disables Q guidance, 100 applies it at all steps)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Path to store sweep results as JSON. "
            "Default: artifacts/guidance/<checkpoint>__alpha_sweep_results.json"
        ),
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional path for the final episode_return-vs-alpha plot (PNG).",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing cfg; cannot reconstruct model.")
    cfg = OmegaConf.create(cfg_dict)
    if args.num_envs is not None:
        if int(args.num_envs) <= 0:
            raise ValueError("--num-envs must be > 0.")
        OmegaConf.update(cfg, "hyperparameters.num_envs", int(args.num_envs), force_add=True)

    num_seeds = int(checkpoint.get("num_seeds", 1))
    if not (0 <= int(args.seed_idx) < num_seeds):
        raise ValueError(
            f"--seed-idx must be in [0, {num_seeds - 1}], got {args.seed_idx}."
        )

    train_mode = str(
        checkpoint.get(
            "train_mode",
            OmegaConf.select(cfg, "hyperparameters.train_mode") or "reparam",
        )
    )
    samplers = _resolve_samplers(train_mode=train_mode, sampler_arg=args.diffusion_sampler)
    alphas = _resolve_alphas(args)
    guidance_dts = _resolve_guidance_dts(args)
    normalizer_mode = str(args.normalizer_stats_mode).lower()
    if normalizer_mode not in ("fixed", "online", "off"):
        raise ValueError(f"Unknown --normalizer-stats-mode: {normalizer_mode}")

    env = _build_env(cfg)
    trainer = ReppoDMERLTrainer(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        num_seeds=1,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    init_fn = trainer._make_init_fn()
    init_key = jax.random.PRNGKey(args.seed)
    train_state = init_fn(init_key)

    actor_params = _select_seed(checkpoint["actor_params"], int(args.seed_idx), num_seeds)
    critic_params = _select_seed(checkpoint["critic_params"], int(args.seed_idx), num_seeds)
    actor_target_params = _select_seed(
        checkpoint.get("actor_target_params", checkpoint["actor_params"]),
        int(args.seed_idx),
        num_seeds,
    )
    norm_state = checkpoint.get("last_env_state", None)
    if norm_state is not None:
        norm_state = _select_seed(norm_state, int(args.seed_idx), num_seeds)

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
    q_temperature = float(args.guidance_temperature)
    if not np.isfinite(q_temperature) or q_temperature <= 0.0:
        raise ValueError("--guidance-temperature must be finite and > 0.")
    env_norm_state = norm_state if trainer.cfg.normalize_env else None
    fixed_norm_stats = None
    if normalizer_mode == "fixed":
        fixed_norm_stats = _extract_fixed_norm_stats(norm_state)
    if normalizer_mode in ("fixed", "online") and trainer.cfg.normalize_env and env_norm_state is None:
        logging.warning(
            "Requested normalizer mode '%s' but checkpoint has no normalization state; evaluation will proceed without loaded stats.",
            normalizer_mode,
        )

    q_grad_clip = args.q_grad_clip
    q_grad_clip_mode = str(args.q_grad_clip_mode)
    q_guidance_last_percent = float(args.q_guidance_last_percent)
    if q_guidance_last_percent < 0.0 or q_guidance_last_percent > 100.0:
        raise ValueError("--q-guidance-last-percent must be in [0, 100].")

    eval_key = jax.random.PRNGKey(args.seed + 1)
    sampler_keys = jax.random.split(eval_key, len(samplers))

    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info(
        "Train mode: %s | samplers: %s | seed_idx: %d",
        train_mode,
        ", ".join(samplers),
        int(args.seed_idx),
    )
    logging.info("Alpha sweep: %s", ", ".join(f"{a:.6g}" for a in alphas))
    logging.info("Guidance dt values: %s", ", ".join(f"{v:.6g}" for v in guidance_dts))
    logging.info(
        "Using guidance temperature T=%.6g (1/T=%.6g) for Q-gradient scaling",
        q_temperature,
        1.0 / q_temperature,
    )
    logging.info("Eval num_envs: %d", int(cfg.hyperparameters.num_envs))
    logging.info("Normalizer stats mode: %s", normalizer_mode)
    if q_grad_clip is not None:
        logging.info(
            "Using q-grad clip: %.6g (mode=%s)",
            float(q_grad_clip),
            q_grad_clip_mode,
        )
    logging.info(
        "Q guidance active in last %.3g%% of diffusion steps",
        q_guidance_last_percent,
    )

    results_by_sampler: dict[str, list[dict[str, float]]] = {}
    best_by_sampler: dict[str, dict[str, float]] = {}
    best_by_sampler_dt: dict[str, dict[str, dict[str, float]]] = {}

    for sampler_key, sampler in zip(sampler_keys, samplers):
        rows: list[dict[str, float]] = []
        best_for_dt: dict[str, dict[str, float]] = {}
        logging.info("Running sampler=%s", sampler)
        dt_keys = jax.random.split(sampler_key, len(guidance_dts))
        for dt_key, guidance_dt in zip(dt_keys, guidance_dts):
            eval_fn = _make_alpha_eval_fn(
                trainer=trainer,
                sampler=sampler,
                guidance_dt=float(guidance_dt),
                q_temperature=float(q_temperature),
                q_grad_clip=q_grad_clip,
                q_grad_clip_mode=q_grad_clip_mode,
                q_guidance_last_percent=q_guidance_last_percent,
                fixed_norm_stats=fixed_norm_stats,
                normalizer_mode=normalizer_mode,
            )
            alpha_keys = jax.random.split(dt_key, len(alphas))
            rows_dt: list[dict[str, float]] = []
            for key, alpha in zip(alpha_keys, alphas):
                metrics = eval_fn(key, train_state, env_norm_state, float(alpha))
                metrics_np = jax.tree.map(lambda x: float(np.asarray(x)), metrics)
                row = {
                    "alpha": float(alpha),
                    "guidance_dt": float(guidance_dt),
                    "guidance_temperature": float(q_temperature),
                    "sampler": sampler,
                    **metrics_np,
                }
                rows.append(row)
                rows_dt.append(row)
                logging.info(
                    "sampler=%s | dt=%.6g | alpha=%.6g | return=%.6f | return_sem=%.6f | length=%.6f | episodes=%d",
                    sampler,
                    float(guidance_dt),
                    row["alpha"],
                    row["episode_return"],
                    row["episode_return_sem"],
                    row["episode_length"],
                    int(round(row["num_episodes"])),
                )
            best_dt = max(rows_dt, key=lambda r: r["episode_return"])
            best_for_dt[f"{float(guidance_dt):.12g}"] = best_dt
            logging.info(
                "Best alpha for sampler=%s dt=%.6g: %.6g (return=%.6f)",
                sampler,
                float(guidance_dt),
                best_dt["alpha"],
                best_dt["episode_return"],
            )
        results_by_sampler[sampler] = rows
        best_by_sampler_dt[sampler] = best_for_dt
        best = max(rows, key=lambda r: r["episode_return"])
        best_by_sampler[sampler] = best
        logging.info(
            "Best setting for sampler=%s: dt=%.6g alpha=%.6g (return=%.6f)",
            sampler,
            best["guidance_dt"],
            best["alpha"],
            best["episode_return"],
        )

    all_rows = [row for rows in results_by_sampler.values() for row in rows]
    best_overall = max(all_rows, key=lambda r: r["episode_return"])
    logging.info(
        "Best overall: sampler=%s dt=%.6g alpha=%.6g (return=%.6f)",
        best_overall["sampler"],
        best_overall["guidance_dt"],
        best_overall["alpha"],
        best_overall["episode_return"],
    )

    plot_out = args.plot_out
    if plot_out is None:
        plot_out = _default_plot_out(checkpoint_path)
    else:
        plot_out = os.path.abspath(os.path.expanduser(plot_out))
    _save_episode_return_plot(results_by_sampler, plot_out)
    logging.info("Saved alpha sweep plot to %s", plot_out)

    payload = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "train_mode": train_mode,
        "samplers": samplers,
        "guidance_dts": [float(v) for v in guidance_dts],
        "guidance_temperature": float(q_temperature),
        "guidance_inv_temperature": float(1.0 / q_temperature),
        "seed": int(args.seed),
        "seed_idx": int(args.seed_idx),
        "normalizer_stats_mode": normalizer_mode,
        "q_grad_clip": None if q_grad_clip is None else float(q_grad_clip),
        "q_grad_clip_mode": q_grad_clip_mode,
        "q_guidance_last_percent": float(q_guidance_last_percent),
        "plot_path": os.path.abspath(plot_out),
        "results_by_sampler": results_by_sampler,
        "best_by_sampler": best_by_sampler,
        "best_by_sampler_dt": best_by_sampler_dt,
        "best_overall": best_overall,
    }
    json_out = args.output_json
    if json_out is None:
        json_out = _default_json_out(checkpoint_path)
    else:
        json_out = os.path.abspath(os.path.expanduser(json_out))
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logging.info("Saved sweep results to %s", json_out)


if __name__ == "__main__":
    main()
