"""Visualize TurningDoubleWell per-state reward landscapes and optimal rollouts.

This script writes:
- reward landscape plot for each possible heading state (subplot titles show state angle)
- trajectory summary plot for a random-optimal rollout policy
- optional GIF of the rollout
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils.turning_double_well_env import TurningDoubleWellEnv


@dataclass(frozen=True)
class RandomOptimalPolicy:
    name: str
    action_values: np.ndarray
    turn_degrees: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/turning_double_well_optimal"))
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num-trajectories", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-turn-deg", type=float, default=90.0)
    parser.add_argument("--well-angle-deg", type=float, default=45.0)
    parser.add_argument("--well-height", type=float, default=0.3)
    parser.add_argument("--transition-noise-deg", type=float, default=0.0)

    parser.add_argument("--landscape-points", type=int, default=4001)
    parser.add_argument("--global-max-atol", type=float, default=1e-4)
    parser.add_argument("--subplot-cols", type=int, default=None)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rollout GIF.",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> TurningDoubleWellEnv:
    return TurningDoubleWellEnv(
        horizon=args.horizon,
        max_turn_deg=args.max_turn_deg,
        randomize_initial_heading=True,
        well_angle_deg=args.well_angle_deg,
        well_height=args.well_height,
        transition_noise_deg=args.transition_noise_deg,
        snap_action_to_optimal=True,
    )


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray:
    angle_deg = np.asarray(angle_deg, dtype=np.float32)
    return ((angle_deg + 180.0) % 360.0) - 180.0


def possible_state_angles_deg(env: TurningDoubleWellEnv) -> np.ndarray:
    support = getattr(env, "_snapped_initial_heading_support_radians", None)
    if support is None:
        heading = np.rad2deg(np.asarray(getattr(env, "initial_heading_radians"), dtype=np.float32))
        support_deg = np.asarray([heading], dtype=np.float32)
    else:
        support_deg = np.rad2deg(np.asarray(support, dtype=np.float32))
    wrapped = wrap_degrees(support_deg)
    return np.unique(np.round(wrapped, decimals=6)).astype(np.float32)


def global_reward_maxima_turns_deg(
    env: TurningDoubleWellEnv,
    *,
    num_points: int,
    atol: float,
) -> np.ndarray:
    degrees, _, reward = env.reward_landscape(num_points=num_points)
    max_reward = float(np.max(reward))
    candidate_idx = np.flatnonzero(reward >= max_reward - float(atol))
    if candidate_idx.size == 0:
        raise RuntimeError("No global maxima were found in reward landscape.")

    split_points = np.where(np.diff(candidate_idx) > 1)[0] + 1
    clusters = np.split(candidate_idx, split_points)

    maxima = []
    for cluster in clusters:
        best = cluster[np.argmax(reward[cluster])]
        maxima.append(float(degrees[best]))

    maxima = np.asarray(maxima, dtype=np.float32)
    maxima = np.unique(np.round(maxima, decimals=6)).astype(np.float32)
    return maxima


def build_random_optimal_policy(
    env: TurningDoubleWellEnv,
    *,
    maxima_turn_deg: np.ndarray,
) -> RandomOptimalPolicy:
    if maxima_turn_deg.size == 0:
        raise RuntimeError("No maxima turn angles provided for policy construction.")

    maxima_actions = np.asarray(maxima_turn_deg / float(env.max_turn_deg), dtype=np.float32)
    maxima_actions = np.clip(maxima_actions, -1.0, 1.0)
    return RandomOptimalPolicy(
        name="random-global-maxima",
        action_values=maxima_actions.astype(np.float32),
        turn_degrees=maxima_turn_deg.astype(np.float32),
    )


def rollout_random_optimal_policy(
    env: TurningDoubleWellEnv,
    *,
    policy: RandomOptimalPolicy,
    seed: int,
    num_trajectories: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    if num_trajectories <= 0:
        raise ValueError("num-trajectories must be positive.")
    if policy.action_values.size == 0:
        raise RuntimeError("Policy has no optimal actions to sample from.")

    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(key, num_trajectories)
    state = env.reset(reset_keys)

    states = [state]
    rewards = []
    sampled_actions = []
    rng = np.random.default_rng(seed)

    for _ in range(env.horizon):
        # For each state/agent at each step, choose uniformly from the global-max actions.
        selected_idx = rng.integers(0, policy.action_values.size, size=(num_trajectories,))
        action_values = policy.action_values[selected_idx]

        action = jnp.asarray(action_values[:, None], dtype=jnp.float32)
        state = env.step(state, action)

        states.append(state)
        rewards.append(np.asarray(state.reward, dtype=np.float32))
        sampled_actions.append(action_values.astype(np.float32))

        if bool(np.asarray(jnp.all(state.done))):
            break

    return states, np.asarray(rewards, dtype=np.float32), np.asarray(sampled_actions, dtype=np.float32)


def plot_reward_landscape_by_state(
    env: TurningDoubleWellEnv,
    *,
    state_angles_deg: np.ndarray,
    maxima_turn_deg: np.ndarray,
    output_path: Path,
    num_points: int,
    subplot_cols: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if state_angles_deg.size == 0:
        raise RuntimeError("No state angles available to plot.")

    turn_deg = np.linspace(-float(env.max_turn_deg), float(env.max_turn_deg), num_points, dtype=np.float32)
    turn_rad = np.deg2rad(turn_deg).astype(np.float32)
    reward = np.asarray(env.reward_from_angle(jnp.asarray(turn_rad, dtype=jnp.float32)), dtype=np.float32)

    n_plots = int(state_angles_deg.size)
    cols = int(subplot_cols) if subplot_cols is not None else int(np.ceil(np.sqrt(n_plots)))
    cols = max(1, cols)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(4.2 * cols, 3.0 * rows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for idx, state_angle in enumerate(state_angles_deg):
        ax = axes_flat[idx]
        ax.plot(turn_deg, reward, color="tab:blue", linewidth=1.8)
        for maxima in maxima_turn_deg:
            ax.axvline(float(maxima), color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlim(-float(env.max_turn_deg), float(env.max_turn_deg))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"state angle = {float(state_angle):.1f} deg", fontsize=10)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Relative turn angle [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reward")

    maxima_text = ", ".join(f"{float(v):.2f}" for v in maxima_turn_deg)
    fig.suptitle(
        f"TurningDoubleWell: reward landscape for each possible heading state\n"
        f"Global reward maxima in relative turn space [deg]: [{maxima_text}]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_summary(
    env: TurningDoubleWellEnv,
    *,
    states: list,
    rewards: np.ndarray,
    sampled_actions: np.ndarray,
    policy: RandomOptimalPolicy,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = env.rollout_positions(states)
    returns = np.sum(rewards, axis=0) if rewards.size else np.zeros((positions.shape[1],), dtype=np.float32)
    sampled_turn_deg = sampled_actions * float(env.max_turn_deg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax_traj = axes[0]
    for agent_idx in range(positions.shape[1]):
        path = positions[:, agent_idx]
        ax_traj.plot(path[:, 0], path[:, 1], linewidth=1.8, alpha=0.8)
        ax_traj.scatter(path[0, 0], path[0, 1], s=20, marker="o", color="black", alpha=0.8)
        ax_traj.scatter(path[-1, 0], path[-1, 1], s=24, marker="x", color="tab:red")
    ax_traj.set_title("Random-optimal trajectories")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.25)

    ax_actions = axes[1]
    ax_actions.hist(sampled_turn_deg.reshape(-1), bins=30, color="tab:purple", alpha=0.85)
    for chosen_turn in policy.turn_degrees:
        ax_actions.axvline(float(chosen_turn), color="tab:green", linestyle="--", linewidth=1.0)
    ax_actions.set_title("Sampled optimal turn angles")
    ax_actions.set_xlabel("Turn angle [deg]")
    ax_actions.set_ylabel("Count")
    ax_actions.grid(True, alpha=0.25)

    ax_returns = axes[2]
    ax_returns.hist(returns, bins=min(20, max(5, returns.size)), color="tab:orange", alpha=0.9)
    ax_returns.set_title("Episode return distribution")
    ax_returns.set_xlabel("Return")
    ax_returns.set_ylabel("Count")
    ax_returns.grid(True, alpha=0.25)

    mean_return = float(np.mean(returns)) if returns.size else 0.0
    std_return = float(np.std(returns)) if returns.size else 0.0
    fig.suptitle(f"Policy={policy.name} | mean return = {mean_return:.3f} +- {std_return:.3f}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rollout_gif(
    env: TurningDoubleWellEnv,
    *,
    states: list,
    rewards: np.ndarray,
    output_path: Path,
    fps: int,
    policy_name: str,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to export GIFs.") from exc

    positions = env.rollout_positions(states)
    titles = [f"agent {idx}" for idx in range(positions.shape[1])]
    frames = env.render_trajectory(
        positions,
        rewards=rewards,
        overlay=False,
        width=240,
        height=220,
        titles=titles,
        max_envs=positions.shape[1],
        figure_title=f"TurningDoubleWell | {policy_name}",
    )
    imageio.mimsave(output_path, frames, fps=fps)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    state_angles_deg = possible_state_angles_deg(env)
    maxima_turn_deg = global_reward_maxima_turns_deg(
        env,
        num_points=args.landscape_points,
        atol=args.global_max_atol,
    )
    policy = build_random_optimal_policy(env, maxima_turn_deg=maxima_turn_deg)

    states, rewards, sampled_actions = rollout_random_optimal_policy(
        env,
        policy=policy,
        seed=args.seed,
        num_trajectories=args.num_trajectories,
    )

    landscape_path = args.output_dir / "reward_landscape_by_state_angle.png"
    summary_path = args.output_dir / "optimal_policy_rollout_summary.png"

    plot_reward_landscape_by_state(
        env,
        state_angles_deg=state_angles_deg,
        maxima_turn_deg=maxima_turn_deg,
        output_path=landscape_path,
        num_points=args.landscape_points,
        subplot_cols=args.subplot_cols,
    )
    plot_rollout_summary(
        env,
        states=states,
        rewards=rewards,
        sampled_actions=sampled_actions,
        policy=policy,
        output_path=summary_path,
    )

    print(f"Saved per-state reward landscapes: {landscape_path}")
    print(f"Saved rollout summary: {summary_path}")

    if args.save_gif:
        gif_path = args.output_dir / "optimal_policy_rollout.gif"
        try:
            save_rollout_gif(
                env,
                states=states,
                rewards=rewards,
                output_path=gif_path,
                fps=args.gif_fps,
                policy_name=policy.name,
            )
            print(f"Saved rollout GIF: {gif_path}")
        except RuntimeError as exc:
            print(f"Skipped GIF export: {exc}")

    maxima_text = ", ".join(f"{float(v):.3f}" for v in maxima_turn_deg)
    policy_turns = ", ".join(f"{float(v):.3f}" for v in policy.turn_degrees)
    print(f"Global reward maxima turn angles [deg]: {maxima_text}")
    print(f"Policy ({policy.name}) samples from turn angles [deg]: {policy_turns}")


if __name__ == "__main__":
    main()
