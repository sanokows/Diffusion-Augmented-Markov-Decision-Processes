import argparse
import logging
import os
import pickle
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from src.env_utils.jax_wrappers import MjxGymnaxWrapper, MjxDiffEnvWrapper
from src.jaxrl.reppo_DMERL_new import ReppoDMERLTrainer, ReppoConfig

logging.basicConfig(level=logging.INFO)


def _load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _select_seed(tree, seed_idx: int = 0):
    return jax.tree.map(lambda x: x[seed_idx], tree)


def _build_env(cfg):
    if cfg.env.type == "brax":
        raise ValueError("Brax environment type is not supported in this evaluator.")
    if cfg.env.type != "mjx":
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

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
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved reppo_DMERL_new model checkpoint."
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
    args = parser.parse_args()

    checkpoint_path = os.path.expanduser(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    cfg_dict = checkpoint.get("cfg")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing cfg; cannot reconstruct model.")

    cfg = OmegaConf.create(cfg_dict)
    env = _build_env(cfg)

    trainer = ReppoDMERLTrainer(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        num_seeds=1,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    init_fn = trainer._make_init_fn()
    key = jax.random.PRNGKey(args.seed)
    train_state = init_fn(key)

    num_seeds = checkpoint.get("num_seeds", 1)
    actor_params = checkpoint["actor_params"]
    critic_params = checkpoint["critic_params"]
    actor_target_params = checkpoint.get("actor_target_params", actor_params)
    norm_state = checkpoint.get("last_env_state", None)

    if num_seeds > 1:
        actor_params = _select_seed(actor_params)
        critic_params = _select_seed(critic_params)
        actor_target_params = _select_seed(actor_target_params)
        if norm_state is not None:
            norm_state = _select_seed(norm_state)

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

    if not trainer.cfg.normalize_env:
        norm_state = None

    eval_key = jax.random.PRNGKey(args.seed + 1)
    eval_metrics = trainer.eval_fn(eval_key, train_state, norm_state)
    eval_metrics = jax.tree.map(lambda x: np.asarray(x), eval_metrics)

    logging.info("Evaluation metrics:")
    for key, value in eval_metrics.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            logging.info("%s: %.6f", key, float(value))
        else:
            logging.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
