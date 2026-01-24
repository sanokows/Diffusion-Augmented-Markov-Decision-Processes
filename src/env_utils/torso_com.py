"""Helpers for logging torso center of mass in MJX environments."""

from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import mujoco
import numpy as np


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


def build_torso_com_traj_figure(
    torso_com_traj,
    torso_com_env_indices=None,
    title: str = "Torso COM trajectory (XY)",
    max_legend_envs: int = 6,
):
    """Return a matplotlib figure for a torso COM XY trajectory, or None."""
    if torso_com_traj is None:
        return None
    com_traj_np = np.asarray(torso_com_traj)
    if com_traj_np.ndim == 4:
        com_traj_np = com_traj_np[0]
    if com_traj_np.ndim != 3 or com_traj_np.shape[-1] < 2:
        return None
    xy = com_traj_np
    idx_np = None
    if torso_com_env_indices is not None:
        idx_np = np.asarray(torso_com_env_indices)
        if idx_np.ndim > 1:
            idx_np = idx_np[0]
    if idx_np is None or idx_np.shape[0] != xy.shape[1]:
        idx_np = np.arange(xy.shape[1])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    cmap = plt.get_cmap("tab10")
    for i in range(xy.shape[1]):
        color = cmap(i % 10)
        line_x = xy[:-1, i, 0]
        line_y = xy[:-1, i, 1]
        # Draw trajectory line; no explicit connection between last and first.
        ax.plot(line_x, line_y, alpha=0.85, linewidth=1.5, color=color)
        ax.scatter(
            line_x[0],
            line_y[0],
            marker="o",
            s=30,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
            label="start" if i == 0 else None,
        )
        ax.scatter(
            line_x[-1],
            line_y[-1],
            marker="X",
            s=36,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
            label="end" if i == 0 else None,
        )

    legend_handles = [
        plt.Line2D([], [], linestyle="None", marker="o", color="black", label="start"),
        plt.Line2D([], [], linestyle="None", marker="X", color="black", label="end"),
    ]
    if xy.shape[1] <= max_legend_envs:
        env_labels = [f"env {int(idx)}" for idx in idx_np]
        legend_handles.extend(
            [
                plt.Line2D([], [], color=cmap(i % 10), lw=1.5, label=env_labels[i])
                for i in range(len(env_labels))
            ]
        )
    ax.legend(
        handles=legend_handles,
        loc="best",
        fontsize=9,
        frameon=True,
    )
    #fig.tight_layout()
    return fig


def save_torso_com_trajectory(
    filename: str,
    torso_com_traj,
    torso_com_env_indices=None,
    output_dir: str | None = None,
):
    """Persist torso COM trajectory data to a pickle file."""
    if torso_com_traj is None:
        return None
    if output_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(repo_root, "results", "trajectories")
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "torso_com_traj": np.asarray(torso_com_traj),
        "torso_com_env_indices": None
        if torso_com_env_indices is None
        else np.asarray(torso_com_env_indices),
    }
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path
