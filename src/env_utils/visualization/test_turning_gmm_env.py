from __future__ import annotations

import argparse
import inspect
from math import ceil, sqrt
import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# File location: <repo>/src/env_utils/visualization/test_turning_gmm_env.py
# so repo root is three levels above this file.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from src.env_utils.turning_GMM_env import TurningGMMEnv


# ---------------------------
# Unit tests
# ---------------------------


def test_action_part_mapping_over_orientation_edges():
    env = TurningGMMEnv(
        randomize_initial_heading=False,
        num_action_state=4,
        gmm_seed=0,
    )

    degrees = jnp.asarray([0.0, 89.999, 90.0, 179.999, 180.0, 269.999, 270.0, 359.999])
    indices = env._action_part_index_from_angle(jnp.deg2rad(degrees))
    expected = jnp.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)

    assert bool(jnp.all(indices == expected))


def test_step_reward_uses_current_heading_action_part():
    env = TurningGMMEnv(
        horizon=5,
        randomize_initial_heading=False,
        num_action_state=2,
        num_gmm_components=1,
        gmm_std=0.2,
        gmm_seed=0,
    )
    # Deterministic part-specific means in A-space to make the test robust.
    env._gmm_means_a = jnp.asarray([[-0.2], [0.7]], dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    angle = jnp.deg2rad(jnp.asarray(30.0, dtype=jnp.float32))
    state = state.replace(
        angle=angle,
        direction=env._angle_to_direction(angle),
    )

    action_value = jnp.asarray(0.2, dtype=jnp.float32)
    next_state = env.step(state, jnp.asarray([action_value], dtype=jnp.float32))

    expected_reward = env.reward_from_action_and_part(
        action_value,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(jnp.isclose(next_state.reward, expected_reward, atol=1e-6))


def test_visualization_includes_action_part_degree_range(tmp_path):
    env = TurningGMMEnv(num_action_state=8, gmm_seed=0)
    output_path = tmp_path / "action_part_landscape.png"

    fig = env.visualize_action_part_reward_landscape(
        2,
        num_points=101,
        output_path=str(output_path),
    )
    assert output_path.exists()
    assert fig._suptitle is not None
    assert "heading in" in fig._suptitle.get_text().lower()
    plt.close(fig)


def test_point_symmetric_mode_odd_components_and_state_pairing():
    env = TurningGMMEnv(
        num_action_state=8,
        num_gmm_components=5,
        gmm_mean_a_margin_d=5.0,
        gmm_std=0.02,
        gmm_seed=7,
        point_symmetric_mode=True,
    )

    means = np.asarray(env._gmm_means_a, dtype=np.float32)
    half_parts = env.num_action_state // 2

    # Opposite heading bins (180 deg apart) share the same landscape.
    assert np.allclose(means[:half_parts], means[half_parts:], atol=1e-7)

    # Each part has mirrored means around zero with one center mean at exactly zero.
    for part_idx in range(env.num_action_state):
        m = np.sort(means[part_idx])
        assert np.isclose(m[env.num_gmm_components // 2], 0.0, atol=1e-7)
        left = m[: env.num_gmm_components // 2]
        right = m[env.num_gmm_components // 2 + 1 :]
        assert np.allclose(left, -right[::-1], atol=1e-6)

    actions = jnp.linspace(-1.0, 1.0, 257)
    for part_idx in range(half_parts):
        r0 = np.asarray(
            env.reward_from_action_and_part(actions, jnp.asarray(part_idx, dtype=jnp.int32))
        )
        r1 = np.asarray(
            env.reward_from_action_and_part(
                actions, jnp.asarray(part_idx + half_parts, dtype=jnp.int32)
            )
        )
        assert np.allclose(r0, r1, atol=1e-6)


def test_point_symmetric_mode_even_components_are_mirrored():
    env = TurningGMMEnv(
        num_action_state=6,
        num_gmm_components=4,
        gmm_mean_a_margin_d=5.0,
        gmm_std=0.02,
        gmm_seed=11,
        point_symmetric_mode=True,
    )

    means = np.asarray(env._gmm_means_a, dtype=np.float32)
    for part_idx in range(env.num_action_state):
        m = np.sort(means[part_idx])
        left = m[: env.num_gmm_components // 2]
        right = m[env.num_gmm_components // 2 :]
        assert np.allclose(left, -right[::-1], atol=1e-6)
        assert np.all(right >= env.gmm_mean_a_margin - 1e-6)
        assert np.all(right <= env.gmm_mean_a_max + 1e-6)


def test_point_symmetric_mode_requires_even_num_action_state():
    raised = False
    try:
        TurningGMMEnv(
            num_action_state=7,
            num_gmm_components=5,
            point_symmetric_mode=True,
            gmm_seed=0,
        )
    except ValueError:
        raised = True
    assert raised


# ---------------------------
# Visualization helpers
# ---------------------------


def plot_reward_landscape_for_all_action_parts(
    env: TurningGMMEnv,
    *,
    output_path: pathlib.Path,
    num_points: int = 1001,
) -> pathlib.Path:
    """Save one figure with subplots for all action-part reward landscapes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
    actions_jnp = jnp.asarray(actions, dtype=jnp.float32)
    rel_turn_deg = np.asarray(env._action_to_turn_degrees(actions_jnp))

    num_parts = env.num_action_state
    ncols = int(ceil(sqrt(num_parts)))
    nrows = int(ceil(num_parts / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for part_idx, ax in enumerate(axes_flat):
        if part_idx >= num_parts:
            ax.axis("off")
            continue

        rewards = np.asarray(
            env.reward_from_action_and_part(
                actions_jnp,
                jnp.asarray(part_idx, dtype=jnp.int32),
            )
        )
        start_deg, end_deg = env.action_part_degree_range(part_idx)

        ax.plot(rel_turn_deg, rewards, color="tab:blue", linewidth=2.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"part={part_idx} | heading in [{start_deg:.1f}, {end_deg:.1f}) deg")
        ax.set_xlabel("Relative orientation (turn) [deg]")
        ax.set_ylabel("Reward = log p_GMM(a)")

    fig.suptitle(
        "TurningGMMEnv reward landscape by action part",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_reward_landscape_x_space_for_all_action_parts(
    env: TurningGMMEnv,
    *,
    output_path: pathlib.Path,
    num_points: int = 1001,
) -> pathlib.Path:
    """Save one figure with subplots for all action-part reward landscapes in A-space."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    means_a_all = np.asarray(env._gmm_means_a, dtype=np.float32)
    a_pad = max(0.05, 6.0 * env.gmm_std)
    a_min = float(max(-1.0, np.min(means_a_all) - a_pad))
    a_max = float(min(1.0, np.max(means_a_all) + a_pad))
    a_values = np.linspace(a_min, a_max, num_points, dtype=np.float32)
    a_values_jnp = jnp.asarray(a_values, dtype=jnp.float32)

    num_parts = env.num_action_state
    ncols = int(ceil(sqrt(num_parts)))
    nrows = int(ceil(num_parts / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for part_idx, ax in enumerate(axes_flat):
        if part_idx >= num_parts:
            ax.axis("off")
            continue

        density = np.asarray(
            env.gmm_density_from_action_and_part(
                a_values_jnp,
                jnp.asarray(part_idx, dtype=jnp.int32),
            )
        )
        rewards = np.log(np.maximum(density, env.min_density))
        start_deg, end_deg = env.action_part_degree_range(part_idx)

        ax.plot(a_values, rewards, color="tab:orange", linewidth=2.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"part={part_idx} | heading in [{start_deg:.1f}, {end_deg:.1f}) deg")
        ax.set_xlabel("action a")
        ax.set_ylabel("Reward = log p_GMM(a)")

    fig.suptitle(
        "TurningGMMEnv reward landscape in A-space by action part",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_density_a_space_for_all_action_parts(
    env: TurningGMMEnv,
    *,
    output_path: pathlib.Path,
    num_points: int = 2001,
) -> pathlib.Path:
    """Save one figure with subplots for all action-part densities p(a) in A-space.

    Also reports per-part normalization constant estimate:
    Z = integral_{-1}^{1} p(a) da
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    a_values = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
    a_values_jnp = jnp.asarray(a_values, dtype=jnp.float32)

    num_parts = env.num_action_state
    ncols = int(ceil(sqrt(num_parts)))
    nrows = int(ceil(num_parts / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for part_idx, ax in enumerate(axes_flat):
        if part_idx >= num_parts:
            ax.axis("off")
            continue

        density = np.asarray(
            env.gmm_density_from_action_and_part(
                a_values_jnp,
                jnp.asarray(part_idx, dtype=jnp.int32),
            )
        )
        z = float(np.trapezoid(density, a_values))
        start_deg, end_deg = env.action_part_degree_range(part_idx)

        ax.plot(a_values, density, color="tab:purple", linewidth=2.0)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"part={part_idx} | heading [{start_deg:.1f}, {end_deg:.1f}) deg | Z={z:.4f}"
        )
        ax.set_xlabel("action a")
        ax.set_ylabel("Density p_GMM(a)")

    fig.suptitle(
        "TurningGMMEnv density p(a) in A-space by action part",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def rollout_parallel_component_mean_policy(
    env: TurningGMMEnv,
    *,
    seed: int,
    batch_size: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out parallel agents by sampling one GMM component and using its mean.

    For each environment at each timestep:
    - infer current action part from heading
    - sample one component index uniformly from that part's equal-weight mixture
    - use the sampled component mean as action
    """

    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(key, batch_size)
    state = env.reset(reset_keys)

    states = [state]
    rewards = []
    sampled_actions = []
    sampled_parts = []

    rng = np.random.default_rng(seed)
    means_by_part = np.asarray(env._gmm_means_a, dtype=np.float32)

    for _ in range(env.horizon):
        part_idx = np.asarray(env._action_part_index_from_angle(state.angle), dtype=np.int32)

        sampled_component_idx = rng.integers(
            low=0,
            high=env.num_gmm_components,
            size=batch_size,
        ).astype(np.int32)
        action_values = means_by_part[part_idx, sampled_component_idx].astype(np.float32)
        action_values = np.clip(action_values, -1.0, 1.0)

        action = jnp.asarray(action_values[:, None], dtype=jnp.float32)
        state = env.step(state, action)

        states.append(state)
        rewards.append(np.asarray(state.reward, dtype=np.float32))
        sampled_actions.append(action_values)
        sampled_parts.append(part_idx)

        if bool(np.asarray(jnp.all(state.done))):
            break

    return (
        states,
        np.asarray(rewards, dtype=np.float32),
        np.asarray(sampled_actions, dtype=np.float32),
        np.asarray(sampled_parts, dtype=np.int32),
    )


def plot_parallel_rollout_evolution(
    env: TurningGMMEnv,
    *,
    states: list,
    rewards: np.ndarray,
    sampled_parts: np.ndarray,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Save a summary figure showing how batched agents evolve over time."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions = env.rollout_positions(states)  # [T+1, B, 2]
    t_steps = rewards.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax_traj = axes[0]
    for env_idx in range(positions.shape[1]):
        path = positions[:, env_idx]
        ax_traj.plot(path[:, 0], path[:, 1], linewidth=1.4, alpha=0.85)
        ax_traj.scatter(path[0, 0], path[0, 1], s=15, color="black", alpha=0.75)
        ax_traj.scatter(path[-1, 0], path[-1, 1], s=18, marker="x", color="tab:red")
    ax_traj.set_title("Parallel agent trajectories")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.25)

    ax_rew = axes[1]
    mean_reward = np.mean(rewards, axis=1)
    std_reward = np.std(rewards, axis=1)
    ts = np.arange(t_steps)
    ax_rew.plot(ts, mean_reward, color="tab:green", linewidth=2.0)
    ax_rew.fill_between(ts, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
    ax_rew.set_title("Step reward over time")
    ax_rew.set_xlabel("t")
    ax_rew.set_ylabel("reward")
    ax_rew.grid(True, alpha=0.25)

    ax_occ = axes[2]
    occupancy = np.zeros((env.num_action_state, t_steps), dtype=np.float32)
    for t in range(t_steps):
        counts = np.bincount(sampled_parts[t], minlength=env.num_action_state)
        occupancy[:, t] = counts.astype(np.float32) / float(sampled_parts.shape[1])
    im = ax_occ.imshow(
        occupancy,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
    )
    ax_occ.set_title("Action-part occupancy over time")
    ax_occ.set_xlabel("t")
    ax_occ.set_ylabel("action part")
    fig.colorbar(im, ax=ax_occ, fraction=0.046, pad=0.04, label="fraction of agents")

    fig.suptitle("TurningGMMEnv parallel rollout under component-mean policy", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _save_rollout_gif_if_possible(
    env: TurningGMMEnv,
    *,
    states: list,
    rewards: np.ndarray,
    output_path: pathlib.Path,
    fps: int = 10,
) -> pathlib.Path | None:
    """Try to save a GIF; return None if imageio is unavailable."""

    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    positions = env.rollout_positions(states)
    titles = [f"agent {idx}" for idx in range(positions.shape[1])]
    frames = env.render_trajectory(
        positions,
        rewards=rewards,
        overlay=True,
        width=960,
        height=720,
        titles=titles,
        max_envs=positions.shape[1],
        figure_title="TurningGMMEnv component-mean rollout (shared panel)",
        title_fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def generate_gmm_visualization_artifacts(
    *,
    output_dir: pathlib.Path = pathlib.Path("artifacts/turning_gmm"),
    num_action_state: int = 8,
    num_gmm_components: int = 5,
    gmm_mean_a_margin_d: float = 5.0,
    gmm_std: float = 0.05,
    gmm_seed: int | None = 0,
    point_symmetric_mode: bool = True,
    horizon: int = 120,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, pathlib.Path | None]:
    """Create all requested GMM env visual artifacts.

    Defaults are deterministic (`gmm_seed=0`, `point_symmetric_mode=True`).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        horizon=horizon,
        randomize_initial_heading=True,
        num_action_state=num_action_state,
        num_gmm_components=num_gmm_components,
        gmm_mean_a_margin_d=gmm_mean_a_margin_d,
        gmm_std=gmm_std,
        gmm_seed=gmm_seed,
        point_symmetric_mode=point_symmetric_mode,
    )
    signature = inspect.signature(TurningGMMEnv.__init__)
    resolved_hparams: dict[str, object] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in env_kwargs:
            resolved_hparams[name] = env_kwargs[name]
        else:
            resolved_hparams[name] = parameter.default

    print("Creating TurningGMMEnv with hyperparameters:")
    for name, value in resolved_hparams.items():
        print(f"  - {name}: {value}")

    env = TurningGMMEnv(**env_kwargs)

    reward_landscape_a_path = plot_reward_landscape_for_all_action_parts(
        env,
        output_path=output_dir / "reward_landscape_all_action_parts_A_space.png",
    )
    reward_landscape_action_path = plot_reward_landscape_x_space_for_all_action_parts(
        env,
        output_path=output_dir / "reward_landscape_all_action_parts_A_space_detail.png",
    )
    density_a_path = plot_density_a_space_for_all_action_parts(
        env,
        output_path=output_dir / "density_all_action_parts_A_space.png",
    )

    states, rewards, _, sampled_parts = rollout_parallel_component_mean_policy(
        env,
        seed=seed,
        batch_size=batch_size,
    )

    rollout_summary_path = plot_parallel_rollout_evolution(
        env,
        states=states,
        rewards=rewards,
        sampled_parts=sampled_parts,
        output_path=output_dir / "parallel_rollout_component_mean_summary.png",
    )

    rollout_gif_path = _save_rollout_gif_if_possible(
        env,
        states=states,
        rewards=rewards,
        output_path=output_dir / "parallel_rollout_component_mean.gif",
        fps=10,
    )

    return {
        "reward_landscape_a_space": reward_landscape_a_path,
        "reward_landscape_a_space_detail": reward_landscape_action_path,
        "density_a_space": density_a_path,
        "rollout_summary": rollout_summary_path,
        "rollout_gif": rollout_gif_path,
    }


# ---------------------------
# CLI entry point
# ---------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TurningGMMEnv artifact visualizations.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("artifacts/turning_gmm"))
    parser.add_argument("--num-action-state", type=int, default=8)
    parser.add_argument("--num-gmm-components", type=int, default=5)
    parser.add_argument(
        "--gmm-mean-a-margin-d",
        type=float,
        default=5.0,
        help="Margin factor d; effective A-space margin is d * gmm_std.",
    )
    parser.add_argument("--gmm-std", type=float, default=0.05)
    parser.add_argument(
        "--gmm-seed",
        type=int,
        default=0,
        help="Seed for deterministic GMM mean sampling (default: 0).",
    )
    parser.add_argument(
        "--point-symmetric-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable special symmetry mode: per-part GMM means are symmetric around a=0, "
            "and opposite heading parts (180 deg apart) share the same landscape."
        ),
    )
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    outputs = generate_gmm_visualization_artifacts(
        output_dir=args.output_dir,
        num_action_state=args.num_action_state,
        num_gmm_components=args.num_gmm_components,
        gmm_mean_a_margin_d=args.gmm_mean_a_margin_d,
        gmm_std=args.gmm_std,
        gmm_seed=args.gmm_seed,
        point_symmetric_mode=args.point_symmetric_mode,
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Saved TurningGMMEnv artifacts:")
    for key, value in outputs.items():
        if value is None:
            print(f"- {key}: skipped (imageio not available)")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    _main()
