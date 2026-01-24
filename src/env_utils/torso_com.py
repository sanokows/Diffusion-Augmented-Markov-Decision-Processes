"""Helpers for logging torso center of mass in MJX environments."""

from __future__ import annotations

import mujoco


def unwrap_to_mjx_state(state_like):
    """Peel nested wrapper states until reaching mjx_env.State with .data."""
    current = state_like
    while hasattr(current, "env_state"):
        current = current.env_state
    return current


def get_torso_com_all(state_like, torso_id: int):
    """Return torso COMs (world frame) for all envs in a possibly batched state."""
    base_state = unwrap_to_mjx_state(state_like)
    com = base_state.data.subtree_com
    if com.ndim == 2:
        com = com[None, ...]
    return com[:, torso_id]


def resolve_mj_model(env_like):
    """Walk wrapper chain until an mj_model attribute is found."""
    current = env_like
    while True:
        if hasattr(current, "mj_model"):
            return current.mj_model
        if hasattr(current, "env"):
            current = current.env
        else:
            return None


def resolve_torso_id(env_like, torso_name: str = "torso"):
    """Resolve the torso body id from a wrapped MJX env, or return None."""
    mj_model = resolve_mj_model(env_like)
    if mj_model is None:
        return None
    torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, torso_name)
    if torso_id < 0:
        return None
    return torso_id
