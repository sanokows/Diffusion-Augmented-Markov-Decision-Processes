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


def _make_guidance_ratio_eval_fn(
    trainer: ReppoDMERLTrainer,
    sampler: str,
    q_grad_clip: float | None,
    q_grad_clip_mode: str,
    fixed_norm_stats: dict[str, jax.Array] | None,
    normalizer_mode: str,
):
    env = trainer.eval_env
    max_episode_steps = trainer.eval_env_steps
    reward_scale = trainer.reward_scale
    ode_coeff = 0.5 if sampler == "ode" else 1.0

    def eval_ratio(
        key: jax.random.PRNGKey,
        train_state,
        norm_state,
    ):
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        diffusion_model = actor_model.diffusion_model
        control_model = diffusion_model.fwd_model

        if control_model is None:
            raise ValueError("Actor diffusion model has no forward score network.")

        diff_steps = max(int(diffusion_model.diff_steps), 1)

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

            step = obs_for_actor["diff_time_step"][..., 0].astype(jnp.float32)
            step_idx = jnp.clip(step.astype(jnp.int32), 0, diff_steps - 1)
            action = obs_for_actor["orig_actions"]
            obs_vec = jnp.concatenate(
                [obs_for_actor["orig_obs"], obs_for_actor["normed_actions"]], axis=-1
            )

            base_score = jax.vmap(_control_score_without_q_grad, in_axes=(None, 0, 0, 0))(
                control_model, action, obs_vec, step
            )

            def _q_grad(single_critic_obs, single_action):
                q_fn = lambda a: critic_model.critic(single_critic_obs, a).sum()
                return jax.grad(q_fn)(single_action)

            q_grad = jax.vmap(_q_grad)(critic_obs_for_actor, action)

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
                    guidance_grad = _clip_by_l2_norm(guidance_grad, max_norm=float(q_grad_clip))
                elif q_grad_clip_mode == "elementwise":
                    clip_v = jnp.asarray(q_grad_clip, dtype=guidance_grad.dtype)
                    guidance_grad = jnp.clip(guidance_grad, -clip_v, clip_v)
                else:
                    raise ValueError(f"Unknown q_grad_clip_mode: {q_grad_clip_mode}")

            drift = jax.vmap(diffusion_model.drift_fn)(step, action)

            def _coeff(single_step, single_obs):
                scale, eta, _ = diffusion_model.diffusion_coeff_fn(single_step, single_obs)
                return scale, eta

            scale, eta = jax.vmap(_coeff)(step, obs_for_actor)
            mu = drift + ode_coeff * base_score
            mean = action + eta * mu

            score_norm = jnp.linalg.norm(base_score, axis=-1)
            guidance_norm = jnp.linalg.norm(guidance_grad, axis=-1)
            valid = jnp.isfinite(score_norm) & jnp.isfinite(guidance_norm) & (guidance_norm > 1e-12)
            ratio = jnp.where(valid, score_norm / (guidance_norm + 1e-12), 0.0)

            if sampler == "ode":
                next_action = mean
            else:
                batch_size = obs["orig_obs"].shape[0]
                if policy_key.ndim == 1:
                    policy_keys = jax.random.split(policy_key, batch_size)
                else:
                    policy_keys = policy_key
                noise = jax.vmap(
                    lambda k, x: jax.random.normal(k, shape=x.shape, dtype=x.dtype)
                )(policy_keys, action)
                next_action = mean + scale * noise

            return next_action, step_idx, ratio, score_norm, guidance_norm, valid

        def _step_env(carry, _):
            (
                step_key,
                env_state,
                obs,
                critic_obs,
                ratio_sum,
                ratio_sq_sum,
                score_norm_sum,
                guidance_norm_sum,
                sample_count,
            ) = carry
            step_key, act_key, env_key = jax.random.split(step_key, 3)
            env_action, step_idx, ratio, score_norm, guidance_norm, valid = _policy_step(
                act_key, obs, critic_obs, env_state
            )
            valid_f = valid.astype(jnp.float32)
            ratio_w = ratio * valid_f
            ratio_sq_w = ratio_w * ratio_w
            score_norm_w = score_norm * valid_f
            guidance_norm_w = guidance_norm * valid_f

            ratio_sum = ratio_sum + jnp.bincount(step_idx, weights=ratio_w, length=diff_steps)
            ratio_sq_sum = ratio_sq_sum + jnp.bincount(step_idx, weights=ratio_sq_w, length=diff_steps)
            score_norm_sum = score_norm_sum + jnp.bincount(step_idx, weights=score_norm_w, length=diff_steps)
            guidance_norm_sum = guidance_norm_sum + jnp.bincount(
                step_idx, weights=guidance_norm_w, length=diff_steps
            )
            sample_count = sample_count + jnp.bincount(
                step_idx, weights=valid_f, length=diff_steps
            )

            keys = jax.random.split(env_key, env.num_envs)
            obs, critic_obs, env_state, _reward, _done, info = env.step(
                keys, env_state, env_action
            )
            return (
                step_key,
                env_state,
                obs,
                critic_obs,
                ratio_sum,
                ratio_sq_sum,
                score_norm_sum,
                guidance_norm_sum,
                sample_count,
            ), info

        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, env.num_envs)
        if norm_state is None:
            obs, critic_obs, env_state = env.reset(init_keys)
        else:
            obs, critic_obs, env_state = env.reset(init_keys, norm_state)

        init_zeros = jnp.zeros((diff_steps,), dtype=jnp.float32)
        (
            _,
            _,
            _,
            _,
            ratio_sum,
            ratio_sq_sum,
            score_norm_sum,
            guidance_norm_sum,
            sample_count,
        ), infos = jax.lax.scan(
            _step_env,
            init=(
                key,
                env_state,
                obs,
                critic_obs,
                init_zeros,
                init_zeros,
                init_zeros,
                init_zeros,
                init_zeros,
            ),
            xs=None,
            length=max_episode_steps,
        )

        returned = infos["returned_episode"]
        ret_mean, _, num_eps = _masked_mean_std(infos["returned_episode_returns"], returned)

        mean = jnp.where(sample_count > 0.0, ratio_sum / sample_count, jnp.nan)
        second = jnp.where(sample_count > 0.0, ratio_sq_sum / sample_count, jnp.nan)
        var = jnp.maximum(second - mean * mean, 0.0)
        std = jnp.sqrt(var)
        sem = _std_to_sem(std, sample_count)

        score_norm_mean = jnp.where(sample_count > 0.0, score_norm_sum / sample_count, jnp.nan)
        guidance_norm_mean = jnp.where(
            sample_count > 0.0, guidance_norm_sum / sample_count, jnp.nan
        )

        return {
            "ratio_mean": mean,
            "ratio_sem": sem,
            "ratio_std": std,
            "score_norm_mean": score_norm_mean,
            "guidance_norm_mean": guidance_norm_mean,
            "num_samples": sample_count,
            "episode_return": ret_mean * reward_scale,
            "num_episodes": num_eps,
        }

    return eval_ratio


def _default_plot_out(checkpoint_path: str) -> str:
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(_REPO_ROOT, "artifacts", "guidance")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ckpt_base}__guidance_ratio_by_step.png")


def _default_json_out(checkpoint_path: str) -> str:
    ckpt_base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(_REPO_ROOT, "artifacts", "guidance")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ckpt_base}__guidance_ratio_by_step.json")


def _save_ratio_plot(results_by_sampler: dict[str, list[dict[str, float]]], out_path: str) -> None:
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
    colors = {"ode": "#1f77b4", "sde": "#d62728"}
    plotted = 0

    for sampler in ("ode", "sde"):
        rows = results_by_sampler.get(sampler, [])
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: int(r["diff_step"]))
        steps = np.asarray([r["diff_step"] for r in rows_sorted], dtype=np.int32)
        ratio = np.asarray([r["ratio_mean"] for r in rows_sorted], dtype=np.float64)
        sem = np.asarray([r["ratio_sem"] for r in rows_sorted], dtype=np.float64)
        color = colors.get(sampler, None)

        finite_mask = np.isfinite(ratio)
        if not np.any(finite_mask):
            continue
        ax.plot(
            steps,
            ratio,
            marker="o",
            markersize=5.5,
            linewidth=2.6,
            color=color,
            label=sampler.upper(),
        )
        finite_sem = finite_mask & np.isfinite(sem)
        if np.any(finite_sem):
            ax.fill_between(
                steps,
                ratio - sem,
                ratio + sem,
                where=finite_sem,
                alpha=0.18,
                color=color,
                linewidth=0.0,
            )
        plotted += 1

    if plotted == 0:
        raise ValueError("No finite ratio values available to plot.")

    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel(r"$\|\mathrm{score}\|_2 / \|\mathrm{guidance}\|_2$")
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
            "Evaluate DA_MDP_REPPO checkpoints and plot per-diffusion-step "
            "ratio between score norm and guidance norm."
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
        "--q-grad-clip",
        type=float,
        default=None,
        help=(
            "Optional clip threshold for guidance gradient (grad_log_p + grad_a Q). "
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
        "--output-json",
        type=str,
        default=None,
        help=(
            "Path to store per-step ratio results as JSON. "
            "Default: artifacts/guidance/<checkpoint>__guidance_ratio_by_step.json"
        ),
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help=(
            "Path to write the per-step ratio plot. "
            "Default: artifacts/guidance/<checkpoint>__guidance_ratio_by_step.png"
        ),
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

    eval_key = jax.random.PRNGKey(args.seed + 1)
    sampler_keys = jax.random.split(eval_key, len(samplers))

    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info(
        "Train mode: %s | samplers: %s | seed_idx: %d",
        train_mode,
        ", ".join(samplers),
        int(args.seed_idx),
    )
    logging.info("Eval num_envs: %d", int(cfg.hyperparameters.num_envs))
    logging.info("Normalizer stats mode: %s", normalizer_mode)
    if q_grad_clip is not None:
        logging.info(
            "Using q-grad clip: %.6g (mode=%s)",
            float(q_grad_clip),
            q_grad_clip_mode,
        )

    results_by_sampler: dict[str, list[dict[str, float]]] = {}
    sampler_summary: dict[str, dict[str, float]] = {}

    for sampler_key, sampler in zip(sampler_keys, samplers):
        eval_fn = _make_guidance_ratio_eval_fn(
            trainer=trainer,
            sampler=sampler,
            q_grad_clip=q_grad_clip,
            q_grad_clip_mode=q_grad_clip_mode,
            fixed_norm_stats=fixed_norm_stats,
            normalizer_mode=normalizer_mode,
        )
        metrics = eval_fn(sampler_key, train_state, env_norm_state)
        metrics_np = jax.tree.map(lambda x: np.asarray(x), metrics)

        ratio_mean = np.asarray(metrics_np["ratio_mean"], dtype=np.float64)
        ratio_sem = np.asarray(metrics_np["ratio_sem"], dtype=np.float64)
        ratio_std = np.asarray(metrics_np["ratio_std"], dtype=np.float64)
        score_norm_mean = np.asarray(metrics_np["score_norm_mean"], dtype=np.float64)
        guidance_norm_mean = np.asarray(metrics_np["guidance_norm_mean"], dtype=np.float64)
        num_samples = np.asarray(metrics_np["num_samples"], dtype=np.float64)

        rows: list[dict[str, float]] = []
        for diff_step in range(ratio_mean.shape[0]):
            rows.append(
                {
                    "diff_step": int(diff_step),
                    "sampler": sampler,
                    "ratio_mean": float(ratio_mean[diff_step]),
                    "ratio_sem": float(ratio_sem[diff_step]),
                    "ratio_std": float(ratio_std[diff_step]),
                    "score_norm_mean": float(score_norm_mean[diff_step]),
                    "guidance_norm_mean": float(guidance_norm_mean[diff_step]),
                    "num_samples": float(num_samples[diff_step]),
                }
            )
        results_by_sampler[sampler] = rows

        valid = np.isfinite(ratio_mean) & (num_samples > 0.0)
        avg_ratio = float(np.nan) if not np.any(valid) else float(np.sum(ratio_mean[valid] * num_samples[valid]) / np.sum(num_samples[valid]))
        sampler_summary[sampler] = {
            "weighted_ratio_mean": avg_ratio,
            "total_samples": float(np.sum(num_samples)),
            "episode_return": float(np.asarray(metrics_np["episode_return"])),
            "num_episodes": float(np.asarray(metrics_np["num_episodes"])),
        }
        logging.info(
            "sampler=%s | weighted ratio mean=%.6f | total samples=%.0f | episode_return=%.6f | episodes=%d",
            sampler,
            avg_ratio,
            sampler_summary[sampler]["total_samples"],
            sampler_summary[sampler]["episode_return"],
            int(round(sampler_summary[sampler]["num_episodes"])),
        )

    plot_out = args.plot_out
    if plot_out is None:
        plot_out = _default_plot_out(checkpoint_path)
    else:
        plot_out = os.path.abspath(os.path.expanduser(plot_out))
    _save_ratio_plot(results_by_sampler, plot_out)
    logging.info("Saved per-step guidance ratio plot to %s", plot_out)

    payload = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "train_mode": train_mode,
        "samplers": samplers,
        "seed": int(args.seed),
        "seed_idx": int(args.seed_idx),
        "normalizer_stats_mode": normalizer_mode,
        "q_grad_clip": None if q_grad_clip is None else float(q_grad_clip),
        "q_grad_clip_mode": q_grad_clip_mode,
        "plot_path": os.path.abspath(plot_out),
        "results_by_sampler": results_by_sampler,
        "sampler_summary": sampler_summary,
    }

    json_out = args.output_json
    if json_out is None:
        json_out = _default_json_out(checkpoint_path)
    else:
        json_out = os.path.abspath(os.path.expanduser(json_out))
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logging.info("Saved per-step guidance ratio results to %s", json_out)


if __name__ == "__main__":
    main()
